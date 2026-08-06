#![allow(unexpected_cfgs)]

use super::{
    MappedSafetensors, MappedTensorMetadata, MetalMoeManifest, UniqueJson, ValidatedMappedFp8,
    accelerate_sgemm_right_transposed, decode_f8_e4m3fn, read_f32_file, sha256_hex, sha256_reader,
    validate_fp8_views, validate_mapped_fp8, write_create_new,
};
use crate::text_endpoint::{
    ComponentSafetyMonitor, ProcessActivityDelta, SafetySnapshot, component_pytorch_noaux_route,
    process_activity,
};
use foreign_types::ForeignTypeRef;
use metal::{CompileOptions, Device, MTLCommandBufferStatus, MTLResourceOptions, MTLSize};
use objc::{msg_send, sel, sel_impl};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File};
use std::path::Path;
use std::time::Instant;

const GATE: &str = "model.layers.43.mlp.experts.32.gate_proj.weight";
const UP: &str = "model.layers.43.mlp.experts.32.up_proj.weight";
const DOWN: &str = "model.layers.43.mlp.experts.32.down_proj.weight";
const KERNEL: &str = "block_fp8_gemv_parallel_lut_blocked";
const LANES: u64 = 64;
const WARMUPS: usize = 5;
const MEASUREMENTS: usize = 30;

#[repr(C)]
#[derive(Clone, Copy)]
struct ProjectionShape {
    rows: u32,
    columns: u32,
    block_rows: u32,
    block_columns: u32,
}

pub(crate) struct BoundedMetalExpertRuntime {
    device: metal::Device,
    queue: metal::CommandQueue,
    pipeline: metal::ComputePipelineState,
    lut: Vec<f32>,
    pub(crate) compile_ms: f64,
    pub(crate) kernel_sha256: String,
    pub(crate) device_name: String,
}

#[derive(Clone, Copy)]
pub(crate) struct NoCopyProjectionBacking {
    pub(crate) weight_region_bytes: usize,
    pub(crate) scale_region_bytes: usize,
    pub(crate) page_bytes: usize,
}

#[derive(Clone, Copy)]
enum SourceBufferMode {
    Copied,
    NoCopy(NoCopyProjectionBacking),
}

pub(crate) struct RoutedNoCopyExpert<'a> {
    pub(crate) expert: u32,
    pub(crate) gate: &'a ValidatedMappedFp8<'a>,
    pub(crate) up: &'a ValidatedMappedFp8<'a>,
    pub(crate) down: &'a ValidatedMappedFp8<'a>,
    pub(crate) backing: [NoCopyProjectionBacking; 3],
}

#[derive(Debug, Serialize)]
pub struct TransactionPhaseTomography {
    pub phase: &'static str,
    pub projection_dispatches: usize,
    pub source_buffer_bind_ms: f64,
    pub small_buffer_install_ms: f64,
    pub command_encode_ms: f64,
    pub commit_call_ms: f64,
    pub synchronous_wait_ms: f64,
    pub gpu_interval_ms: f64,
    pub readback_ms: f64,
    pub explicit_release_ms: f64,
    pub wall_ms: f64,
    pub activity: ProcessActivityDelta,
}

#[derive(Debug, Serialize)]
pub struct RoutedTransactionTomography {
    pub layer: usize,
    pub dynamic_input_ms: f64,
    pub gate_up_cpu_stage_ms: f64,
    pub dynamic_hidden_ms: f64,
    pub down_cpu_stage_ms: f64,
    pub phases: Vec<TransactionPhaseTomography>,
    pub command_buffers: usize,
    pub commits: usize,
    pub waits: usize,
    pub projection_dispatches: usize,
    pub wall_ms: f64,
    pub activity: ProcessActivityDelta,
}

pub(crate) struct RoutedTransactionOutput {
    pub(crate) experts: Vec<(u32, BoundedMetalExpertOutput)>,
    pub(crate) tomography: RoutedTransactionTomography,
}

fn validate_no_copy_region(
    address: usize,
    data_bytes: usize,
    region_bytes: usize,
    page_bytes: usize,
) -> bool {
    page_bytes != 0
        && page_bytes.is_power_of_two()
        && address.is_multiple_of(page_bytes)
        && region_bytes.is_multiple_of(page_bytes)
        && region_bytes >= data_bytes
}

fn validate_phase_completion(
    phase: &str,
    status: MTLCommandBufferStatus,
    gpu_interval_ms: Option<f64>,
    synchronous_wait_ms: f64,
) -> Result<f64, String> {
    if status != MTLCommandBufferStatus::Completed {
        return Err(format!("{phase}: Metal command failed: {status:?}"));
    }
    let gpu_interval_ms =
        gpu_interval_ms.ok_or_else(|| format!("{phase}: GPU timestamps unavailable"))?;
    if gpu_interval_ms > synchronous_wait_ms + 1.0 {
        return Err(format!("{phase}: GPU interval exceeds wait"));
    }
    Ok(gpu_interval_ms)
}

pub(crate) struct BoundedMetalExpertOutput {
    pub(crate) gate: Vec<f32>,
    pub(crate) up: Vec<f32>,
    pub(crate) swiglu: Vec<f32>,
    pub(crate) down: Vec<f32>,
    pub(crate) gate_pre_round: Vec<f32>,
    pub(crate) up_pre_round: Vec<f32>,
    pub(crate) down_pre_round: Vec<f32>,
    pub(crate) sparse_repair_counts: [usize; 3],
    pub(crate) installed_source_bytes: u64,
    pub(crate) sparse_decoded_weight_bytes: u64,
    pub(crate) tomography: Option<ExpertTomography>,
}

#[derive(Debug, Serialize)]
pub struct ProjectionTomography {
    pub projection: &'static str,
    pub rows: usize,
    pub columns: usize,
    pub source_weight_bytes: u64,
    pub source_scale_bytes: u64,
    pub source_buffer_install_ms: f64,
    pub small_buffer_install_ms: f64,
    pub command_encode_ms: f64,
    pub commit_call_ms: f64,
    pub synchronous_wait_ms: f64,
    pub gpu_interval_ms: Option<f64>,
    pub readback_ms: f64,
    pub explicit_release_ms: f64,
    pub wall_ms: f64,
    pub source_install_activity: ProcessActivityDelta,
    pub total_activity: ProcessActivityDelta,
}

#[derive(Debug, Serialize)]
pub struct ExpertTomography {
    pub layer: usize,
    pub expert: u32,
    pub source_shards: Vec<String>,
    pub tensor_lookup_validation_ms: f64,
    pub dynamic_input_ms: f64,
    pub gate_up_bf16_round_ms: f64,
    pub gate_up_sparse_repair_ms: f64,
    pub swiglu_ms: f64,
    pub dynamic_hidden_ms: f64,
    pub down_bf16_round_ms: f64,
    pub down_sparse_repair_ms: f64,
    pub weighted_scatter_ms: f64,
    pub matrix_transient_release_ms: f64,
    pub projections: Vec<ProjectionTomography>,
    pub wall_ms: f64,
    pub activity: ProcessActivityDelta,
}

impl BoundedMetalExpertRuntime {
    pub(crate) fn compile(kernel_path: &Path) -> Result<Self, String> {
        let kernel_bytes =
            fs::read(kernel_path).map_err(|error| format!("{}: {error}", kernel_path.display()))?;
        let kernel_sha256 = sha256_hex(&kernel_bytes);
        let kernel_source = String::from_utf8(kernel_bytes)
            .map_err(|_| "Metal kernel source is not UTF-8".to_owned())?;
        if !kernel_source.contains(&format!("kernel void {KERNEL}")) {
            return Err(format!("kernel source lacks {KERNEL}"));
        }
        let device = Device::system_default().ok_or("no Metal device is available")?;
        let device_name = device.name().to_owned();
        let options = CompileOptions::new();
        options.set_fast_math_enabled(false);
        let compile_started = Instant::now();
        let library = device
            .new_library_with_source(&kernel_source, &options)
            .map_err(|error| format!("Metal compilation failed: {error}"))?;
        let function = library
            .get_function(KERNEL, None)
            .map_err(|error| format!("Metal kernel lookup failed: {error}"))?;
        let pipeline = device
            .new_compute_pipeline_state_with_function(&function)
            .map_err(|error| format!("Metal pipeline creation failed: {error}"))?;
        let compile_ms = compile_started.elapsed().as_secs_f64() * 1000.0;
        if pipeline.max_total_threads_per_threadgroup() < LANES {
            return Err("Metal pipeline cannot dispatch 64 lanes".to_owned());
        }
        let queue = device.new_command_queue();
        let lut = (0_u16..=255)
            .map(|value| decode_f8_e4m3fn(value as u8))
            .collect();
        Ok(Self {
            device,
            queue,
            pipeline,
            lut,
            compile_ms,
            kernel_sha256,
            device_name,
        })
    }

    pub(crate) fn execute(
        &self,
        projections: [&ValidatedMappedFp8<'_>; 3],
        input: &[f32],
    ) -> Result<BoundedMetalExpertOutput, String> {
        self.execute_internal(projections, input, None)
    }

    pub(crate) fn execute_profiled(
        &self,
        layer: usize,
        expert: u32,
        projections: [&ValidatedMappedFp8<'_>; 3],
        input: &[f32],
    ) -> Result<BoundedMetalExpertOutput, String> {
        self.execute_internal(projections, input, Some((layer, expert)))
    }

    pub(crate) fn execute_profiled_no_copy(
        &self,
        layer: usize,
        expert: u32,
        projections: [&ValidatedMappedFp8<'_>; 3],
        backing: [NoCopyProjectionBacking; 3],
        input: &[f32],
    ) -> Result<BoundedMetalExpertOutput, String> {
        self.execute_internal_with_modes(
            projections,
            input,
            Some((layer, expert)),
            backing.map(SourceBufferMode::NoCopy),
        )
    }

    pub(crate) fn execute_two_barrier_routed_transaction(
        &self,
        layer: usize,
        experts: &[RoutedNoCopyExpert<'_>],
        input: &[f32],
    ) -> Result<RoutedTransactionOutput, String> {
        if experts.len() != 8
            || experts
                .iter()
                .map(|expert| expert.expert)
                .collect::<BTreeSet<_>>()
                .len()
                != experts.len()
        {
            return Err("routed transaction requires eight distinct experts".to_owned());
        }
        let wall_started = Instant::now();
        let activity_before = process_activity()?;
        let dynamic_input_started = Instant::now();
        let staged_input = dynamic_fp8_dequantized(input)?;
        let dynamic_input_ms = dynamic_input_started.elapsed().as_secs_f64() * 1000.0;
        let mut gate_up_tensors = Vec::with_capacity(16);
        let mut gate_up_inputs = Vec::with_capacity(16);
        for expert in experts {
            gate_up_tensors.push((expert.gate, expert.backing[0]));
            gate_up_inputs.push(staged_input.as_slice());
            gate_up_tensors.push((expert.up, expert.backing[1]));
            gate_up_inputs.push(staged_input.as_slice());
        }
        let (gate_up_pre_round, gate_up_phase) =
            self.execute_no_copy_phase("gate_up", &gate_up_tensors, &gate_up_inputs, true)?;
        let gate_up_cpu_started = Instant::now();
        let mut staged = Vec::with_capacity(experts.len());
        let mut dynamic_hidden_ms = 0.0;
        for (index, expert) in experts.iter().enumerate() {
            let gate_pre_round = gate_up_pre_round[index * 2].clone();
            let up_pre_round = gate_up_pre_round[index * 2 + 1].clone();
            let mut gate = gate_pre_round.clone();
            let mut up = up_pre_round.clone();
            round_bf16_values(&mut gate);
            round_bf16_values(&mut up);
            let gate_repairs =
                repair_uncertain_rows(expert.gate, &staged_input, &gate_pre_round, &mut gate)?;
            let up_repairs =
                repair_uncertain_rows(expert.up, &staged_input, &up_pre_round, &mut up)?;
            let hidden = gate
                .iter()
                .zip(&up)
                .map(|(&gate, &up)| staged_swiglu(gate, up))
                .collect::<Vec<_>>();
            let dynamic_hidden_started = Instant::now();
            let staged_hidden = dynamic_fp8_dequantized(&hidden)?;
            dynamic_hidden_ms += dynamic_hidden_started.elapsed().as_secs_f64() * 1000.0;
            staged.push((
                gate,
                up,
                hidden,
                staged_hidden,
                gate_pre_round,
                up_pre_round,
                gate_repairs,
                up_repairs,
            ));
        }
        let gate_up_cpu_stage_ms = gate_up_cpu_started.elapsed().as_secs_f64() * 1000.0;
        let down_tensors = experts
            .iter()
            .map(|expert| (expert.down, expert.backing[2]))
            .collect::<Vec<_>>();
        let down_inputs = staged
            .iter()
            .map(|values| values.3.as_slice())
            .collect::<Vec<_>>();
        let (down_pre_round, down_phase) =
            self.execute_no_copy_phase("down", &down_tensors, &down_inputs, false)?;
        let down_cpu_started = Instant::now();
        let mut outputs = Vec::with_capacity(experts.len());
        for ((expert, values), down_pre_round) in experts.iter().zip(staged).zip(down_pre_round) {
            let (
                gate,
                up,
                hidden,
                staged_hidden,
                gate_pre_round,
                up_pre_round,
                gate_repairs,
                up_repairs,
            ) = values;
            let mut down = down_pre_round.clone();
            round_bf16_values(&mut down);
            let down_repairs =
                repair_uncertain_rows(expert.down, &staged_hidden, &down_pre_round, &mut down)?;
            if down.iter().any(|value| !value.is_finite()) {
                return Err("routed transaction produced non-finite output".to_owned());
            }
            let projections = [expert.gate, expert.up, expert.down];
            let installed_source_bytes = projections.iter().try_fold(0_u64, |total, tensor| {
                total
                    .checked_add(tensor.weight.metadata.data_bytes)
                    .and_then(|value| value.checked_add(tensor.scale.metadata.data_bytes))
                    .ok_or("transaction installed-byte ledger overflow")
            })?;
            let sparse_decoded_weight_bytes = (gate_repairs * expert.gate.columns
                + up_repairs * expert.up.columns
                + down_repairs * expert.down.columns)
                as u64;
            outputs.push((
                expert.expert,
                BoundedMetalExpertOutput {
                    gate,
                    up,
                    swiglu: hidden,
                    down,
                    gate_pre_round,
                    up_pre_round,
                    down_pre_round,
                    sparse_repair_counts: [gate_repairs, up_repairs, down_repairs],
                    installed_source_bytes,
                    sparse_decoded_weight_bytes,
                    tomography: None,
                },
            ));
        }
        let down_cpu_stage_ms = down_cpu_started.elapsed().as_secs_f64() * 1000.0;
        let wall_ms = wall_started.elapsed().as_secs_f64() * 1000.0;
        Ok(RoutedTransactionOutput {
            experts: outputs,
            tomography: RoutedTransactionTomography {
                layer,
                dynamic_input_ms,
                gate_up_cpu_stage_ms,
                dynamic_hidden_ms,
                down_cpu_stage_ms,
                phases: vec![gate_up_phase, down_phase],
                command_buffers: 2,
                commits: 2,
                waits: 2,
                projection_dispatches: 24,
                wall_ms,
                activity: process_activity()?.checked_delta(activity_before)?,
            },
        })
    }

    fn execute_no_copy_phase(
        &self,
        phase: &'static str,
        tensors: &[(&ValidatedMappedFp8<'_>, NoCopyProjectionBacking)],
        activations: &[&[f32]],
        shared_activation: bool,
    ) -> Result<(Vec<Vec<f32>>, TransactionPhaseTomography), String> {
        if tensors.is_empty() || tensors.len() != activations.len() {
            return Err(format!("{phase}: projection/input count mismatch"));
        }
        let shape = ProjectionShape {
            rows: tensors[0].0.rows as u32,
            columns: tensors[0].0.columns as u32,
            block_rows: 128,
            block_columns: 128,
        };
        if tensors
            .iter()
            .zip(activations)
            .any(|((tensor, backing), activation)| {
                tensor.rows != shape.rows as usize
                    || tensor.columns != shape.columns as usize
                    || activation.len() != tensor.columns
                    || activation.iter().any(|value| !value.is_finite())
                    || !validate_no_copy_region(
                        tensor.weight.bytes.as_ptr() as usize,
                        tensor.weight.bytes.len(),
                        backing.weight_region_bytes,
                        backing.page_bytes,
                    )
                    || !validate_no_copy_region(
                        tensor.scale.bytes.as_ptr() as usize,
                        tensor.scale.bytes.len(),
                        backing.scale_region_bytes,
                        backing.page_bytes,
                    )
            })
        {
            return Err(format!(
                "{phase}: invalid projection shape, input, or no-copy backing"
            ));
        }
        if shared_activation
            && activations.iter().any(|activation| {
                activation.as_ptr() != activations[0].as_ptr()
                    || activation.len() != activations[0].len()
            })
        {
            return Err(format!("{phase}: shared activation identity mismatch"));
        }
        let wall_started = Instant::now();
        let activity_before = process_activity()?;
        let shared = MTLResourceOptions::StorageModeShared;
        let source_started = Instant::now();
        let sources = tensors
            .iter()
            .map(|(tensor, backing)| {
                (
                    self.device.new_buffer_with_bytes_no_copy(
                        tensor.weight.bytes.as_ptr().cast(),
                        backing.weight_region_bytes as u64,
                        shared,
                        None,
                    ),
                    self.device.new_buffer_with_bytes_no_copy(
                        tensor.scale.bytes.as_ptr().cast(),
                        backing.scale_region_bytes as u64,
                        shared,
                        None,
                    ),
                )
            })
            .collect::<Vec<_>>();
        let source_buffer_bind_ms = source_started.elapsed().as_secs_f64() * 1000.0;
        let small_started = Instant::now();
        let input_buffers = if shared_activation {
            vec![self.device.new_buffer_with_data(
                activations[0].as_ptr().cast(),
                std::mem::size_of_val(activations[0]) as u64,
                shared,
            )]
        } else {
            activations
                .iter()
                .map(|activation| {
                    self.device.new_buffer_with_data(
                        activation.as_ptr().cast(),
                        std::mem::size_of_val(*activation) as u64,
                        shared,
                    )
                })
                .collect()
        };
        let outputs = tensors
            .iter()
            .map(|(tensor, _)| self.device.new_buffer((tensor.rows * 4) as u64, shared))
            .collect::<Vec<_>>();
        let shape_buffer = self.device.new_buffer_with_data(
            (&shape as *const ProjectionShape).cast(),
            std::mem::size_of::<ProjectionShape>() as u64,
            shared,
        );
        let lut_buffer = self.device.new_buffer_with_data(
            self.lut.as_ptr().cast(),
            std::mem::size_of_val(self.lut.as_slice()) as u64,
            shared,
        );
        let small_buffer_install_ms = small_started.elapsed().as_secs_f64() * 1000.0;
        let encode_started = Instant::now();
        let command = self.queue.new_command_buffer();
        let encoder = command.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.pipeline);
        for index in 0..tensors.len() {
            encoder.set_buffer(0, Some(&sources[index].0), 0);
            encoder.set_buffer(1, Some(&sources[index].1), 0);
            encoder.set_buffer(
                2,
                Some(&input_buffers[if shared_activation { 0 } else { index }]),
                0,
            );
            encoder.set_buffer(3, Some(&outputs[index]), 0);
            encoder.set_buffer(4, Some(&shape_buffer), 0);
            encoder.set_buffer(5, Some(&lut_buffer), 0);
            encoder.set_threadgroup_memory_length(0, LANES * 4);
            encoder.dispatch_thread_groups(
                MTLSize {
                    width: tensors[index].0.rows as u64,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: LANES,
                    height: 1,
                    depth: 1,
                },
            );
        }
        encoder.end_encoding();
        let command_encode_ms = encode_started.elapsed().as_secs_f64() * 1000.0;
        let commit_started = Instant::now();
        command.commit();
        let commit_call_ms = commit_started.elapsed().as_secs_f64() * 1000.0;
        let wait_started = Instant::now();
        command.wait_until_completed();
        let synchronous_wait_ms = wait_started.elapsed().as_secs_f64() * 1000.0;
        let gpu_interval_ms = validate_phase_completion(
            phase,
            command.status(),
            completed_gpu_interval_ms(command),
            synchronous_wait_ms,
        )?;
        let readback_started = Instant::now();
        let values = outputs
            .iter()
            .zip(tensors)
            .map(|(buffer, (tensor, _))| {
                // SAFETY: the command has completed and each shared output buffer
                // contains exactly `rows` initialized F32 values.
                unsafe {
                    std::slice::from_raw_parts(buffer.contents().cast::<f32>(), tensor.rows)
                        .to_vec()
                }
            })
            .collect::<Vec<_>>();
        let readback_ms = readback_started.elapsed().as_secs_f64() * 1000.0;
        let release_started = Instant::now();
        drop(sources);
        drop(input_buffers);
        drop(outputs);
        drop(shape_buffer);
        drop(lut_buffer);
        let explicit_release_ms = release_started.elapsed().as_secs_f64() * 1000.0;
        let tomography = TransactionPhaseTomography {
            phase,
            projection_dispatches: tensors.len(),
            source_buffer_bind_ms,
            small_buffer_install_ms,
            command_encode_ms,
            commit_call_ms,
            synchronous_wait_ms,
            gpu_interval_ms,
            readback_ms,
            explicit_release_ms,
            wall_ms: wall_started.elapsed().as_secs_f64() * 1000.0,
            activity: process_activity()?.checked_delta(activity_before)?,
        };
        Ok((values, tomography))
    }

    pub(crate) fn probe_no_copy_mapping(
        &self,
        region: &[u8],
        page_bytes: usize,
    ) -> Result<f64, String> {
        const PROBE_BYTES: usize = 4096;
        if page_bytes == 0
            || !page_bytes.is_power_of_two()
            || !(region.as_ptr() as usize).is_multiple_of(page_bytes)
            || !region.len().is_multiple_of(page_bytes)
            || region.len() < PROBE_BYTES
        {
            return Err("no-copy probe region is not page aligned or is too short".to_owned());
        }
        let source = r#"
            #include <metal_stdlib>
            using namespace metal;
            kernel void pw_nocopy_read_probe(
                device const uchar *input [[buffer(0)]],
                device uchar *output [[buffer(1)]],
                uint index [[thread_position_in_grid]]) {
                output[index] = input[index];
            }
        "#;
        let options = CompileOptions::new();
        options.set_fast_math_enabled(false);
        let library = self
            .device
            .new_library_with_source(source, &options)
            .map_err(|error| format!("no-copy probe compilation failed: {error}"))?;
        let function = library
            .get_function("pw_nocopy_read_probe", None)
            .map_err(|error| format!("no-copy probe function lookup failed: {error}"))?;
        let pipeline = self
            .device
            .new_compute_pipeline_state_with_function(&function)
            .map_err(|error| format!("no-copy probe pipeline failed: {error}"))?;
        let shared = MTLResourceOptions::StorageModeShared;
        let started = Instant::now();
        let input = self.device.new_buffer_with_bytes_no_copy(
            region.as_ptr().cast(),
            region.len() as u64,
            shared,
            None,
        );
        let output = self.device.new_buffer(PROBE_BYTES as u64, shared);
        let command = self.queue.new_command_buffer();
        let encoder = command.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(&input), 0);
        encoder.set_buffer(1, Some(&output), 0);
        encoder.dispatch_threads(
            MTLSize {
                width: PROBE_BYTES as u64,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: 64,
                height: 1,
                depth: 1,
            },
        );
        encoder.end_encoding();
        command.commit();
        command.wait_until_completed();
        if command.status() != MTLCommandBufferStatus::Completed {
            return Err(format!(
                "no-copy probe command failed: {:?}",
                command.status()
            ));
        }
        // SAFETY: completion precedes the exactly PROBE_BYTES-long shared read.
        let actual =
            unsafe { std::slice::from_raw_parts(output.contents().cast::<u8>(), PROBE_BYTES) };
        if actual != &region[..PROBE_BYTES] {
            return Err("no-copy probe returned non-identical bytes".to_owned());
        }
        drop(input);
        drop(output);
        Ok(started.elapsed().as_secs_f64() * 1000.0)
    }

    fn execute_internal(
        &self,
        projections: [&ValidatedMappedFp8<'_>; 3],
        input: &[f32],
        identity: Option<(usize, u32)>,
    ) -> Result<BoundedMetalExpertOutput, String> {
        self.execute_internal_with_modes(
            projections,
            input,
            identity,
            [SourceBufferMode::Copied; 3],
        )
    }

    fn execute_internal_with_modes(
        &self,
        projections: [&ValidatedMappedFp8<'_>; 3],
        input: &[f32],
        identity: Option<(usize, u32)>,
        source_modes: [SourceBufferMode; 3],
    ) -> Result<BoundedMetalExpertOutput, String> {
        let execution = execute_staged_expert(
            &self.device,
            &self.queue,
            &self.pipeline,
            &self.lut,
            projections,
            input,
            identity,
            source_modes,
        )?;
        let installed_source_bytes = projections.iter().try_fold(0_u64, |total, tensor| {
            total
                .checked_add(tensor.weight.metadata.data_bytes)
                .and_then(|value| value.checked_add(tensor.scale.metadata.data_bytes))
                .ok_or("Metal installed-byte ledger overflow")
        })?;
        let sparse_decoded_weight_bytes = (execution.repairs[0] * projections[0].columns
            + execution.repairs[1] * projections[1].columns
            + execution.repairs[2] * projections[2].columns)
            as u64;
        Ok(BoundedMetalExpertOutput {
            gate: execution.gate,
            up: execution.up,
            swiglu: execution.hidden,
            down: execution.down,
            gate_pre_round: execution.gate_pre_round,
            up_pre_round: execution.up_pre_round,
            down_pre_round: execution.down_pre_round,
            sparse_repair_counts: execution.repairs,
            installed_source_bytes,
            sparse_decoded_weight_bytes,
            tomography: execution.tomography,
        })
    }
}

#[derive(Debug, Serialize)]
pub struct StagedMetalExpertReport {
    pub schema_version: u32,
    pub semantic: &'static str,
    pub commit: String,
    pub gate_up_source_file: String,
    pub gate_up_source_sha256: String,
    pub down_source_file: String,
    pub down_source_sha256: String,
    pub gate: MappedTensorMetadata,
    pub up: MappedTensorMetadata,
    pub down: MappedTensorMetadata,
    pub kernel_sha256: String,
    pub kernel_function: &'static str,
    pub device: String,
    pub input_sha256: String,
    pub reference_sha256: String,
    pub output_sha256: String,
    pub output_first8: Vec<f32>,
    pub relative_l2: f64,
    pub maximum_absolute_error: f32,
    pub bf16_equal_values: usize,
    pub bf16_total_values: usize,
    pub bf16_equality_fraction: f64,
    pub compile_ms: f64,
    pub cold_wall_ms: f64,
    pub warmups: usize,
    pub measurements: usize,
    pub wall_ms: Vec<f64>,
    pub wall_p10_ms: f64,
    pub wall_median_ms: f64,
    pub wall_p90_ms: f64,
    pub timing_gate_passed: bool,
    pub speedup_vs_pw0096_mean_expert: f64,
    pub speedup_gate_passed: bool,
    pub logical_source_bytes_per_execution: u64,
    pub maximum_resident_tensor_buffer_bytes: u64,
    pub sparse_repair_counts: [usize; 3],
    pub batch_size: u32,
    pub concurrency: u32,
    pub accepted_tokens: u32,
    #[serde(rename = "A")]
    pub accepted_per_verification: u32,
    #[serde(rename = "U")]
    pub unique_expert_sets: u32,
    pub cache_state: &'static str,
    pub timed_scope: &'static str,
    pub safety_snapshots: Vec<SafetySnapshot>,
    pub performance_claim: Option<String>,
    pub implementation: &'static str,
}

#[derive(Debug, Serialize)]
pub struct BoundedRoutedRowReport {
    pub schema_version: u32,
    pub semantic: &'static str,
    pub commit: String,
    pub manifest_sha256: String,
    pub input_sha256: String,
    pub router_sha256: String,
    pub reference_manifest_sha256: String,
    pub reference_sha256: String,
    pub output_sha256: String,
    pub selected_experts: Vec<u32>,
    pub route_weights: Vec<f32>,
    pub maximum_route_weight_absolute_error: f32,
    pub minimum_topk_boundary_margin: f32,
    pub relative_l2: f64,
    pub maximum_absolute_error: f32,
    pub bf16_equality_fraction: f64,
    pub device: String,
    pub compile_ms: f64,
    pub cold_wall_ms: f64,
    pub warmups: usize,
    pub measurements: usize,
    pub wall_ms: Vec<f64>,
    pub wall_p10_ms: f64,
    pub wall_median_ms: f64,
    pub wall_p90_ms: f64,
    pub speedup_vs_pw0096_routed_layer: f64,
    pub timing_gates_passed: bool,
    pub logical_source_bytes_per_execution: u64,
    pub maximum_resident_tensor_buffer_bytes: u64,
    pub sparse_repair_counts: [usize; 3],
    pub sparse_decoded_weight_bytes: u64,
    pub batch_size: u32,
    pub concurrency: u32,
    pub accepted_tokens: u32,
    #[serde(rename = "A")]
    pub accepted_per_verification: u32,
    #[serde(rename = "U")]
    pub unique_experts: u32,
    pub safety_snapshots: Vec<SafetySnapshot>,
    pub cache_state: &'static str,
    pub performance_claim: Option<String>,
}

#[derive(Deserialize)]
struct RoutedRowOracleManifest {
    schema_version: u32,
    semantic: String,
    revision: String,
    checkpoint_verification_sha256: String,
    source_manifest_sha256: String,
    input_sha256: String,
    selected_experts: Vec<u32>,
    route_weights: Vec<f32>,
    expert_outputs: BTreeMap<String, BTreeMap<String, RoutedRowOracleCapture>>,
    output_sha256: String,
}

#[derive(Deserialize)]
struct RoutedRowOracleCapture {
    file: String,
    sha256: String,
    shape: Vec<usize>,
    dtype: String,
}

struct RoutedRowExecution {
    output: Vec<f32>,
    selected: Vec<u32>,
    weights: Vec<f32>,
    minimum_boundary_margin: f32,
    expert_outputs: Vec<(u32, StagedExpertExecution)>,
    wall_ms: f64,
}

fn round_bf16(value: f32) -> f32 {
    let bits = value.to_bits();
    if bits & 0x7f80_0000 == 0x7f80_0000 {
        if bits & 0x007f_ffff == 0 {
            return value;
        }
        return f32::from_bits(u32::from(((bits >> 16) as u16) | 0x0040) << 16);
    }
    let bias = 0x7fff + ((bits >> 16) & 1);
    f32::from_bits(bits.wrapping_add(bias) & 0xffff_0000)
}

fn round_bf16_values(values: &mut [f32]) {
    values
        .iter_mut()
        .for_each(|value| *value = round_bf16(*value));
}

fn staged_swiglu(gate: f32, up: f32) -> f32 {
    let silu = round_bf16(gate / (1.0 + (-gate).exp()));
    round_bf16(silu * up)
}

fn encode_f8_e4m3fn(value: f32) -> Result<u8, String> {
    if !value.is_finite() || value.abs() > 448.0 {
        return Err("E4M3FN encoder requires a finite value in [-448,448]".to_owned());
    }
    if value == 0.0 {
        return Ok(if value.is_sign_negative() { 0x80 } else { 0 });
    }
    let magnitude = value.abs();
    let mut best = 0_u8;
    let mut distance = f32::INFINITY;
    for candidate in 0_u8..=0x7e {
        let candidate_distance = (magnitude - decode_f8_e4m3fn(candidate)).abs();
        if candidate_distance < distance
            || (candidate_distance == distance && candidate & 1 == 0 && best & 1 != 0)
        {
            best = candidate;
            distance = candidate_distance;
        }
    }
    Ok(best | if value.is_sign_negative() { 0x80 } else { 0 })
}

fn dynamic_fp8_dequantized(input: &[f32]) -> Result<Vec<f32>, String> {
    if input.is_empty() || !input.len().is_multiple_of(128) || input.iter().any(|x| !x.is_finite())
    {
        return Err("dynamic FP8 input shape or value mismatch".to_owned());
    }
    let mut output = Vec::with_capacity(input.len());
    for group in input.chunks_exact(128) {
        let absmax = group
            .iter()
            .map(|x| x.abs())
            .fold(0.0_f32, f32::max)
            .max(1.0e-10);
        let scale = absmax / 448.0;
        for &value in group {
            let bits = encode_f8_e4m3fn((value / scale).clamp(-448.0, 448.0))?;
            output.push(decode_f8_e4m3fn(bits) * scale);
        }
    }
    Ok(output)
}

fn bf16_midpoint_distance_ulps(value: f32) -> u32 {
    (value.to_bits() & 0xffff).abs_diff(0x8000)
}

fn repair_uncertain_rows(
    tensor: &ValidatedMappedFp8<'_>,
    input: &[f32],
    pre_round: &[f32],
    rounded: &mut [f32],
) -> Result<usize, String> {
    const MAX_MIDPOINT_DISTANCE_ULPS: u32 = 4;
    let rows = pre_round
        .iter()
        .enumerate()
        .filter_map(|(row, value)| {
            (bf16_midpoint_distance_ulps(*value) <= MAX_MIDPOINT_DISTANCE_ULPS).then_some(row)
        })
        .collect::<Vec<_>>();
    if rows.is_empty() {
        return Ok(0);
    }
    let mut decoded = Vec::with_capacity(rows.len() * tensor.columns);
    for &row in &rows {
        let scale_row = row / 128 * tensor.scale_columns;
        for column in 0..tensor.columns {
            decoded.push(
                decode_f8_e4m3fn(tensor.weight.bytes[row * tensor.columns + column])
                    * tensor.scales[scale_row + column / 128],
            );
        }
    }
    let mut corrected =
        accelerate_sgemm_right_transposed(input, &decoded, 1, rows.len(), tensor.columns)?;
    round_bf16_values(&mut corrected);
    for (&row, value) in rows.iter().zip(corrected) {
        rounded[row] = value;
    }
    Ok(rows.len())
}

fn completed_gpu_interval_ms(command: &metal::CommandBufferRef) -> Option<f64> {
    // SAFETY: the command buffer has completed and these MTLCommandBuffer
    // selectors return process-independent monotonic GPU timestamps in seconds.
    let start: f64 = unsafe { msg_send![command.as_ptr(), GPUStartTime] };
    // SAFETY: same completed command buffer and API contract as above.
    let end: f64 = unsafe { msg_send![command.as_ptr(), GPUEndTime] };
    (start.is_finite() && end.is_finite() && start > 0.0 && end >= start)
        .then_some((end - start) * 1000.0)
}

#[allow(clippy::too_many_arguments)]
fn metal_project(
    device: &metal::DeviceRef,
    queue: &metal::CommandQueueRef,
    pipeline: &metal::ComputePipelineStateRef,
    lut: &[f32],
    tensor: &ValidatedMappedFp8<'_>,
    activation: &[f32],
    projection: &'static str,
    tomography_enabled: bool,
    source_mode: SourceBufferMode,
) -> Result<(Vec<f32>, Option<ProjectionTomography>), String> {
    if activation.len() != tensor.columns || activation.iter().any(|x| !x.is_finite()) {
        return Err("Metal projection input mismatch".to_owned());
    }
    let shape = ProjectionShape {
        rows: tensor.rows as u32,
        columns: tensor.columns as u32,
        block_rows: 128,
        block_columns: 128,
    };
    let wall_started = Instant::now();
    let activity_started = tomography_enabled.then(process_activity).transpose()?;
    let shared = MTLResourceOptions::StorageModeShared;
    let source_install_started = Instant::now();
    let source_activity_started = tomography_enabled.then(process_activity).transpose()?;
    let (weight, scale) = match source_mode {
        SourceBufferMode::Copied => (
            device.new_buffer_with_data(
                tensor.weight.bytes.as_ptr().cast(),
                tensor.weight.bytes.len() as u64,
                shared,
            ),
            device.new_buffer_with_data(
                tensor.scale.bytes.as_ptr().cast(),
                tensor.scale.bytes.len() as u64,
                shared,
            ),
        ),
        SourceBufferMode::NoCopy(backing) => {
            if backing.page_bytes == 0
                || !backing.page_bytes.is_power_of_two()
                || !(tensor.weight.bytes.as_ptr() as usize).is_multiple_of(backing.page_bytes)
                || !(tensor.scale.bytes.as_ptr() as usize).is_multiple_of(backing.page_bytes)
                || !backing
                    .weight_region_bytes
                    .is_multiple_of(backing.page_bytes)
                || !backing
                    .scale_region_bytes
                    .is_multiple_of(backing.page_bytes)
                || backing.weight_region_bytes < tensor.weight.bytes.len()
                || backing.scale_region_bytes < tensor.scale.bytes.len()
            {
                return Err(format!(
                    "{projection}: no-copy source mapping is not page aligned or is undersized"
                ));
            }
            (
                device.new_buffer_with_bytes_no_copy(
                    tensor.weight.bytes.as_ptr().cast(),
                    backing.weight_region_bytes as u64,
                    shared,
                    None,
                ),
                device.new_buffer_with_bytes_no_copy(
                    tensor.scale.bytes.as_ptr().cast(),
                    backing.scale_region_bytes as u64,
                    shared,
                    None,
                ),
            )
        }
    };
    let source_buffer_install_ms = source_install_started.elapsed().as_secs_f64() * 1000.0;
    let source_install_activity = if let Some(before) = source_activity_started {
        process_activity()?.checked_delta(before)?
    } else {
        ProcessActivityDelta::default()
    };
    let small_install_started = Instant::now();
    let input_buffer = device.new_buffer_with_data(
        activation.as_ptr().cast(),
        std::mem::size_of_val(activation) as u64,
        shared,
    );
    let output_buffer = device.new_buffer((tensor.rows * 4) as u64, shared);
    let shape_buffer = device.new_buffer_with_data(
        (&shape as *const ProjectionShape).cast(),
        std::mem::size_of::<ProjectionShape>() as u64,
        shared,
    );
    let lut_buffer = device.new_buffer_with_data(
        lut.as_ptr().cast(),
        std::mem::size_of_val(lut) as u64,
        shared,
    );
    let small_buffer_install_ms = small_install_started.elapsed().as_secs_f64() * 1000.0;
    let encode_started = Instant::now();
    let command = queue.new_command_buffer();
    let encoder = command.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(&weight), 0);
    encoder.set_buffer(1, Some(&scale), 0);
    encoder.set_buffer(2, Some(&input_buffer), 0);
    encoder.set_buffer(3, Some(&output_buffer), 0);
    encoder.set_buffer(4, Some(&shape_buffer), 0);
    encoder.set_buffer(5, Some(&lut_buffer), 0);
    encoder.set_threadgroup_memory_length(0, LANES * 4);
    encoder.dispatch_thread_groups(
        MTLSize {
            width: tensor.rows as u64,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: LANES,
            height: 1,
            depth: 1,
        },
    );
    encoder.end_encoding();
    let command_encode_ms = encode_started.elapsed().as_secs_f64() * 1000.0;
    let commit_started = Instant::now();
    command.commit();
    let commit_call_ms = commit_started.elapsed().as_secs_f64() * 1000.0;
    let wait_started = Instant::now();
    command.wait_until_completed();
    let synchronous_wait_ms = wait_started.elapsed().as_secs_f64() * 1000.0;
    if command.status() != MTLCommandBufferStatus::Completed {
        return Err(format!("Metal projection failed: {:?}", command.status()));
    }
    let gpu_interval_ms = tomography_enabled
        .then(|| completed_gpu_interval_ms(command))
        .flatten();
    // SAFETY: completion precedes reading the exactly rows-long shared F32 buffer.
    let readback_started = Instant::now();
    let output = unsafe {
        std::slice::from_raw_parts(output_buffer.contents().cast::<f32>(), tensor.rows).to_vec()
    };
    let readback_ms = readback_started.elapsed().as_secs_f64() * 1000.0;
    let release_started = Instant::now();
    drop(weight);
    drop(scale);
    drop(input_buffer);
    drop(output_buffer);
    drop(shape_buffer);
    drop(lut_buffer);
    let explicit_release_ms = release_started.elapsed().as_secs_f64() * 1000.0;
    let wall_ms = wall_started.elapsed().as_secs_f64() * 1000.0;
    let tomography = if let Some(before) = activity_started {
        Some(ProjectionTomography {
            projection,
            rows: tensor.rows,
            columns: tensor.columns,
            source_weight_bytes: tensor.weight.metadata.data_bytes,
            source_scale_bytes: tensor.scale.metadata.data_bytes,
            source_buffer_install_ms,
            small_buffer_install_ms,
            command_encode_ms,
            commit_call_ms,
            synchronous_wait_ms,
            gpu_interval_ms,
            readback_ms,
            explicit_release_ms,
            wall_ms,
            source_install_activity,
            total_activity: process_activity()?.checked_delta(before)?,
        })
    } else {
        None
    };
    Ok((output, tomography))
}

#[allow(clippy::too_many_arguments)]
fn execute_staged_expert(
    device: &metal::DeviceRef,
    queue: &metal::CommandQueueRef,
    pipeline: &metal::ComputePipelineStateRef,
    lut: &[f32],
    projections: [&ValidatedMappedFp8<'_>; 3],
    input: &[f32],
    identity: Option<(usize, u32)>,
    source_modes: [SourceBufferMode; 3],
) -> Result<StagedExpertExecution, String> {
    let wall_started = Instant::now();
    let activity_started = identity.map(|_| process_activity()).transpose()?;
    let [gate, up, down] = projections;
    let dynamic_input_started = Instant::now();
    let staged_input = dynamic_fp8_dequantized(input)?;
    let dynamic_input_ms = dynamic_input_started.elapsed().as_secs_f64() * 1000.0;
    let profiling = identity.is_some();
    let (gate_pre_round, gate_tomography) = metal_project(
        device,
        queue,
        pipeline,
        lut,
        gate,
        &staged_input,
        "gate",
        profiling,
        source_modes[0],
    )?;
    let (up_pre_round, up_tomography) = metal_project(
        device,
        queue,
        pipeline,
        lut,
        up,
        &staged_input,
        "up",
        profiling,
        source_modes[1],
    )?;
    let mut gate_output = gate_pre_round.clone();
    let mut up_output = up_pre_round.clone();
    let gate_up_round_started = Instant::now();
    round_bf16_values(&mut gate_output);
    round_bf16_values(&mut up_output);
    let gate_up_bf16_round_ms = gate_up_round_started.elapsed().as_secs_f64() * 1000.0;
    let gate_up_repair_started = Instant::now();
    let gate_repairs =
        repair_uncertain_rows(gate, &staged_input, &gate_pre_round, &mut gate_output)?;
    let up_repairs = repair_uncertain_rows(up, &staged_input, &up_pre_round, &mut up_output)?;
    let gate_up_sparse_repair_ms = gate_up_repair_started.elapsed().as_secs_f64() * 1000.0;
    let swiglu_started = Instant::now();
    let hidden = gate_output
        .iter()
        .zip(&up_output)
        .map(|(&g, &u)| staged_swiglu(g, u))
        .collect::<Vec<_>>();
    let swiglu_ms = swiglu_started.elapsed().as_secs_f64() * 1000.0;
    let dynamic_hidden_started = Instant::now();
    let staged_hidden = dynamic_fp8_dequantized(&hidden)?;
    let dynamic_hidden_ms = dynamic_hidden_started.elapsed().as_secs_f64() * 1000.0;
    let (down_pre_round, down_tomography) = metal_project(
        device,
        queue,
        pipeline,
        lut,
        down,
        &staged_hidden,
        "down",
        profiling,
        source_modes[2],
    )?;
    let mut output = down_pre_round.clone();
    let down_round_started = Instant::now();
    round_bf16_values(&mut output);
    let down_bf16_round_ms = down_round_started.elapsed().as_secs_f64() * 1000.0;
    let down_repair_started = Instant::now();
    let down_repairs = repair_uncertain_rows(down, &staged_hidden, &down_pre_round, &mut output)?;
    let down_sparse_repair_ms = down_repair_started.elapsed().as_secs_f64() * 1000.0;
    if output.iter().any(|x| !x.is_finite()) {
        return Err("staged Metal expert produced non-finite output".to_owned());
    }
    let tomography = if let (Some((layer, expert)), Some(before)) = (identity, activity_started) {
        Some(ExpertTomography {
            layer,
            expert,
            source_shards: Vec::new(),
            tensor_lookup_validation_ms: 0.0,
            dynamic_input_ms,
            gate_up_bf16_round_ms,
            gate_up_sparse_repair_ms,
            swiglu_ms,
            dynamic_hidden_ms,
            down_bf16_round_ms,
            down_sparse_repair_ms,
            weighted_scatter_ms: 0.0,
            matrix_transient_release_ms: 0.0,
            projections: vec![
                gate_tomography.ok_or("missing gate tomography")?,
                up_tomography.ok_or("missing up tomography")?,
                down_tomography.ok_or("missing down tomography")?,
            ],
            wall_ms: wall_started.elapsed().as_secs_f64() * 1000.0,
            activity: process_activity()?.checked_delta(before)?,
        })
    } else {
        None
    };
    Ok(StagedExpertExecution {
        gate: gate_output,
        up: up_output,
        hidden,
        down: output,
        gate_pre_round,
        up_pre_round,
        down_pre_round,
        repairs: [gate_repairs, up_repairs, down_repairs],
        tomography,
    })
}

struct StagedExpertExecution {
    gate: Vec<f32>,
    up: Vec<f32>,
    hidden: Vec<f32>,
    down: Vec<f32>,
    gate_pre_round: Vec<f32>,
    up_pre_round: Vec<f32>,
    down_pre_round: Vec<f32>,
    repairs: [usize; 3],
    tomography: Option<ExpertTomography>,
}

pub fn run_staged_metal_fp8_expert(
    gate_up_source: &Path,
    down_source: &Path,
    kernel_path: &Path,
    input_path: &Path,
    reference_path: &Path,
    output_path: &Path,
    commit: &str,
) -> Result<StagedMetalExpertReport, String> {
    if commit.len() != 40
        || !commit
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err("implementation commit must be a lowercase 40-hex Git object".to_owned());
    }
    if output_path.exists() {
        return Err(format!("refusing to overwrite {}", output_path.display()));
    }
    let mut safety = ComponentSafetyMonitor::start_normative()?;
    let gate_up_source_sha256 = sha256_reader(
        &mut File::open(gate_up_source)
            .map_err(|e| format!("{}: {e}", gate_up_source.display()))?,
    )?;
    let down_source_sha256 = sha256_reader(
        &mut File::open(down_source).map_err(|e| format!("{}: {e}", down_source.display()))?,
    )?;
    let gate_up_mapping = MappedSafetensors::open(gate_up_source)?;
    let down_mapping = MappedSafetensors::open(down_source)?;
    let (input_bytes, mut input) = read_f32_file(input_path, Some(4096))?;
    round_bf16_values(&mut input);
    let gate = validate_mapped_fp8(&gate_up_mapping, GATE, &format!("{GATE}_scale_inv"), &input)?;
    let up = validate_mapped_fp8(&gate_up_mapping, UP, &format!("{UP}_scale_inv"), &input)?;
    let down_shape = vec![0.0_f32; 2048];
    let down = validate_mapped_fp8(
        &down_mapping,
        DOWN,
        &format!("{DOWN}_scale_inv"),
        &down_shape,
    )?;
    if (gate.rows, gate.columns) != (2048, 4096)
        || (up.rows, up.columns) != (2048, 4096)
        || (down.rows, down.columns) != (4096, 2048)
    {
        return Err("layer-43/expert-32 projection shape mismatch".to_owned());
    }
    let (reference_bytes, reference) = read_f32_file(reference_path, Some(4096))?;
    let kernel_source =
        fs::read_to_string(kernel_path).map_err(|e| format!("{}: {e}", kernel_path.display()))?;
    if !kernel_source.contains(&format!("kernel void {KERNEL}")) {
        return Err(format!("kernel source lacks {KERNEL}"));
    }
    let device = Device::system_default().ok_or("no Metal device is available")?;
    let options = CompileOptions::new();
    options.set_fast_math_enabled(false);
    let compile_start = Instant::now();
    let library = device
        .new_library_with_source(&kernel_source, &options)
        .map_err(|e| format!("Metal compilation failed: {e}"))?;
    let function = library
        .get_function(KERNEL, None)
        .map_err(|e| format!("Metal kernel lookup failed: {e}"))?;
    let pipeline = device
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|e| format!("Metal pipeline creation failed: {e}"))?;
    let compile_ms = compile_start.elapsed().as_secs_f64() * 1000.0;
    if pipeline.max_total_threads_per_threadgroup() < LANES {
        return Err("Metal pipeline cannot dispatch 64 lanes".to_owned());
    }
    let queue = device.new_command_queue();
    let lut = (0_u16..=255)
        .map(|x| decode_f8_e4m3fn(x as u8))
        .collect::<Vec<_>>();
    safety.checkpoint("after_compile")?;

    let execute = || -> Result<(Vec<f32>, [usize; 3], f64), String> {
        let start = Instant::now();
        let execution = execute_staged_expert(
            &device,
            &queue,
            &pipeline,
            &lut,
            [&gate, &up, &down],
            &input,
            None,
            [SourceBufferMode::Copied; 3],
        )?;
        Ok((
            execution.down,
            execution.repairs,
            start.elapsed().as_secs_f64() * 1000.0,
        ))
    };

    let (_, _, cold_wall_ms) = execute()?;
    for _ in 0..WARMUPS {
        execute()?;
    }
    safety.checkpoint("after_warmups")?;
    let mut wall_ms = Vec::with_capacity(MEASUREMENTS);
    let mut output = Vec::new();
    let mut sparse_repair_counts = [0_usize; 3];
    for _ in 0..MEASUREMENTS {
        let (candidate, repairs, elapsed) = execute()?;
        output = candidate;
        sparse_repair_counts = repairs;
        wall_ms.push(elapsed);
    }
    safety.checkpoint("after_timed_series")?;
    let mut squared_error = 0.0_f64;
    let mut squared_reference = 0.0_f64;
    let mut maximum_absolute_error = 0.0_f32;
    let mut bf16_equal_values = 0;
    for (&candidate, &expected) in output.iter().zip(&reference) {
        let difference = candidate - expected;
        squared_error += f64::from(difference).powi(2);
        squared_reference += f64::from(expected).powi(2);
        maximum_absolute_error = maximum_absolute_error.max(difference.abs());
        bf16_equal_values += usize::from(candidate.to_bits() == expected.to_bits());
    }
    if squared_reference == 0.0 {
        return Err("expert reference has zero L2 norm".to_owned());
    }
    let relative_l2 = (squared_error / squared_reference).sqrt();
    let bf16_equality_fraction = bf16_equal_values as f64 / output.len() as f64;
    if relative_l2 > 5.0e-4 || maximum_absolute_error > 2.0e-2 || bf16_equality_fraction < 0.99 {
        return Err(format!(
            "staged expert parity failed: rel L2 {relative_l2}, max abs {maximum_absolute_error}, BF16 equality {bf16_equality_fraction}"
        ));
    }
    let output_bytes = output
        .iter()
        .flat_map(|x| x.to_le_bytes())
        .collect::<Vec<_>>();
    write_create_new(output_path, &output_bytes)?;
    let mut ordered = wall_ms.clone();
    ordered.sort_by(f64::total_cmp);
    let percentile =
        |fraction: f64| ordered[((ordered.len() - 1) as f64 * fraction).round() as usize];
    let wall_p10_ms = percentile(0.10);
    let wall_median_ms = percentile(0.50);
    let wall_p90_ms = percentile(0.90);
    let speedup = 397.5 / wall_median_ms;
    let safety_snapshots = safety.released()?;
    let logical_source_bytes_per_execution = gate.weight.metadata.data_bytes
        + gate.scale.metadata.data_bytes
        + up.weight.metadata.data_bytes
        + up.scale.metadata.data_bytes
        + down.weight.metadata.data_bytes
        + down.scale.metadata.data_bytes
        + input_bytes.len() as u64
        + output_bytes.len() as u64;
    let maximum_resident_tensor_buffer_bytes = (gate.weight.metadata.data_bytes
        + gate.scale.metadata.data_bytes)
        .max(up.weight.metadata.data_bytes + up.scale.metadata.data_bytes)
        .max(down.weight.metadata.data_bytes + down.scale.metadata.data_bytes);
    Ok(StagedMetalExpertReport {
        schema_version: 1,
        semantic: "mimo_layer43_expert32_dynamic_fp8_bf16_staged_source_fp8_metal",
        commit: commit.to_owned(),
        gate_up_source_file: gate_up_source.display().to_string(),
        gate_up_source_sha256,
        down_source_file: down_source.display().to_string(),
        down_source_sha256,
        gate: gate.weight.metadata.clone(),
        up: up.weight.metadata.clone(),
        down: down.weight.metadata.clone(),
        kernel_sha256: sha256_hex(kernel_source.as_bytes()),
        kernel_function: KERNEL,
        device: device.name().to_owned(),
        input_sha256: sha256_hex(&input_bytes),
        reference_sha256: sha256_hex(&reference_bytes),
        output_sha256: sha256_hex(&output_bytes),
        output_first8: output.iter().copied().take(8).collect(),
        relative_l2,
        maximum_absolute_error,
        bf16_equal_values,
        bf16_total_values: output.len(),
        bf16_equality_fraction,
        compile_ms,
        cold_wall_ms,
        warmups: WARMUPS,
        measurements: MEASUREMENTS,
        wall_ms,
        wall_p10_ms,
        wall_median_ms,
        wall_p90_ms,
        timing_gate_passed: wall_median_ms <= 100.0,
        speedup_vs_pw0096_mean_expert: speedup,
        speedup_gate_passed: speedup >= 10.0,
        logical_source_bytes_per_execution,
        maximum_resident_tensor_buffer_bytes,
        sparse_repair_counts,
        batch_size: 1,
        concurrency: 1,
        accepted_tokens: 0,
        accepted_per_verification: 0,
        unique_expert_sets: 1,
        cache_state: "warm OS source cache after full-file identity hashing; pipeline and LUT resident; every timed execution installs and releases real expert tensor buffers",
        timed_scope: "dynamic FP8 activation staging, gate/up/down tensor installation and Metal dispatch, BF16 boundaries, CPU SwiGLU, waits, readback",
        safety_snapshots,
        performance_claim: None,
        implementation: "rust_owned_metal_source_fp8_dynamic_activation_bf16_staged_one_row_expert",
    })
}

#[allow(clippy::too_many_arguments)]
pub fn run_bounded_metal_routed_row(
    manifest_path: &Path,
    artifact_root: &Path,
    router_path: &Path,
    kernel_path: &Path,
    input_path: &Path,
    reference_path: &Path,
    output_path: &Path,
    commit: &str,
) -> Result<BoundedRoutedRowReport, String> {
    const REVISION: &str = "63651580ca774f8504f676040460aed3e1244ac1";
    const MANIFEST_SHA256: &str =
        "a2b30be7ab767c754fd4680887420246dcd28d314bff242b144f664a8ff12470";
    const INPUT_SHA256: &str = "ac6776035eee0537ab0d7d7975d4ad92e08bf67930b58d47a4d9f2e051113150";
    const ROUTER_SHA256: &str = "12c1579d28b78dd69ec9342eb9d1f378efc5aa3c2f2a28b5ec73578e6a8bbcdd";
    const ROUTER_WEIGHT: &str = "model.layers.43.mlp.gate.weight";
    const ROUTER_BIAS: &str = "model.layers.43.mlp.gate.e_score_correction_bias";
    if output_path.exists() {
        return Err(format!("refusing to overwrite {}", output_path.display()));
    }
    if commit.len() != 40
        || !commit
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err("implementation commit must be a lowercase 40-hex Git object".to_owned());
    }
    let mut safety = ComponentSafetyMonitor::start_normative()?;
    let manifest_bytes =
        fs::read(manifest_path).map_err(|error| format!("{}: {error}", manifest_path.display()))?;
    if sha256_hex(&manifest_bytes) != MANIFEST_SHA256 {
        return Err("PW-0037 manifest SHA-256 mismatch".to_owned());
    }
    let unique: UniqueJson =
        serde_json::from_slice(&manifest_bytes).map_err(|error| format!("manifest: {error}"))?;
    let manifest: MetalMoeManifest =
        serde_json::from_value(unique.0).map_err(|error| format!("manifest: {error}"))?;
    if manifest.schema_version != 1
        || manifest.semantic != "mimo_layer43_fixture_scheduled_source_fp8_moe_block"
        || manifest.revision != REVISION
        || manifest.layer != 43
        || manifest.batch_size != 8
        || manifest.top_k != 8
        || manifest.selected_experts_by_position.len() != 8
        || manifest.route_weights_by_position.len() != 8
    {
        return Err("PW-0037 manifest identity mismatch".to_owned());
    }
    let frozen_selected = manifest.selected_experts_by_position[0].clone();
    let frozen_weights = manifest.route_weights_by_position[0].clone();
    if frozen_selected.len() != 8
        || frozen_weights.len() != 8
        || frozen_selected
            .iter()
            .copied()
            .collect::<BTreeSet<_>>()
            .len()
            != 8
        || frozen_weights.iter().any(|value| !value.is_finite())
    {
        return Err("PW-0037 row-zero route mismatch".to_owned());
    }
    let (input_bytes, all_input) = read_f32_file(input_path, Some(8 * 4096))?;
    if sha256_hex(&input_bytes) != INPUT_SHA256 {
        return Err("PW-0037 input SHA-256 mismatch".to_owned());
    }
    let mut input = all_input[..4096].to_vec();
    round_bf16_values(&mut input);
    let (reference_bytes, reference) = read_f32_file(reference_path, Some(4096))?;
    let reference_manifest_path = reference_path.with_file_name("manifest.json");
    let reference_manifest_bytes = fs::read(&reference_manifest_path)
        .map_err(|error| format!("{}: {error}", reference_manifest_path.display()))?;
    let reference_manifest: RoutedRowOracleManifest =
        serde_json::from_slice(&reference_manifest_bytes)
            .map_err(|error| format!("routed-row oracle manifest: {error}"))?;
    if reference_manifest.schema_version != 1
        || reference_manifest.semantic
            != "mimo_layer43_verified_checkpoint_bf16_staged_routed_row_oracle"
        || reference_manifest.revision != REVISION
        || reference_manifest.checkpoint_verification_sha256
            != "9ddc8a99755f04ae2ea3c2484f6dd022d3f3a681b5a72c915ee4de833dbb0d03"
        || reference_manifest.source_manifest_sha256 != MANIFEST_SHA256
        || reference_manifest.input_sha256 != INPUT_SHA256
        || reference_manifest.selected_experts != frozen_selected
        || reference_manifest.route_weights.len() != 8
        || reference_manifest.expert_outputs.len() != 8
        || reference_manifest
            .route_weights
            .iter()
            .any(|value| !value.is_finite())
        || reference_manifest.output_sha256 != sha256_hex(&reference_bytes)
    {
        return Err("routed-row oracle manifest identity mismatch".to_owned());
    }
    let oracle_weights = reference_manifest.route_weights.clone();

    let mut router_file =
        File::open(router_path).map_err(|error| format!("{}: {error}", router_path.display()))?;
    if sha256_reader(&mut router_file)? != ROUTER_SHA256 {
        return Err("layer-43 router SHA-256 mismatch".to_owned());
    }
    let router = MappedSafetensors::open(router_path)?;
    let router_weight = router.tensor(ROUTER_WEIGHT)?;
    let router_bias = router.tensor(ROUTER_BIAS)?;
    if router_weight.metadata.dtype != "F32"
        || router_weight.metadata.shape != [256, 4096]
        || router_bias.metadata.dtype != "F32"
        || router_bias.metadata.shape != [256]
    {
        return Err("layer-43 router tensor mismatch".to_owned());
    }
    let decode_f32 = |bytes: &[u8]| {
        bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("four-byte F32")))
            .collect::<Vec<_>>()
    };
    let router_weights = decode_f32(router_weight.bytes);
    let correction = decode_f32(router_bias.bytes);
    if router_weights
        .iter()
        .chain(&correction)
        .any(|x| !x.is_finite())
    {
        return Err("layer-43 router contains non-finite values".to_owned());
    }

    let mut mappings = BTreeMap::new();
    for (name, expected) in &manifest.artifact_sha256 {
        if Path::new(name).file_name().and_then(|value| value.to_str()) != Some(name) {
            return Err(format!("unsafe artifact name: {name}"));
        }
        let path = artifact_root.join(name);
        let mut file = File::open(&path).map_err(|error| format!("{}: {error}", path.display()))?;
        if sha256_reader(&mut file)? != *expected {
            return Err(format!("artifact SHA-256 mismatch: {name}"));
        }
        mappings.insert(name.clone(), MappedSafetensors::open(&path)?);
    }
    let mut projections = BTreeMap::new();
    let down_authority = vec![0.0_f32; 2048];
    for &expert_id in &frozen_selected {
        let expert = manifest
            .experts
            .iter()
            .find(|candidate| candidate.expert == expert_id)
            .ok_or_else(|| format!("selected expert {expert_id} absent from manifest"))?;
        let prefix = format!("model.layers.43.mlp.experts.{expert_id}");
        if expert.prefix != prefix {
            return Err("expert prefix mismatch".to_owned());
        }
        let matrix =
            |projection: &str, shape_input: &[f32]| -> Result<ValidatedMappedFp8<'_>, String> {
                let weight_key = format!("{projection}_weight");
                let scale_key = format!("{projection}_scale");
                let weight_file = expert
                    .tensor_files
                    .get(&weight_key)
                    .ok_or("missing weight artifact")?;
                let scale_file = expert
                    .tensor_files
                    .get(&scale_key)
                    .ok_or("missing scale artifact")?;
                let weight_mapping = mappings
                    .get(weight_file)
                    .ok_or("unknown expert weight artifact")?;
                let scale_mapping = mappings
                    .get(scale_file)
                    .ok_or("unknown expert scale artifact")?;
                let name = format!("{prefix}.{projection}_proj.weight");
                validate_fp8_views(
                    weight_mapping.tensor(&name)?,
                    scale_mapping.tensor(&format!("{name}_scale_inv"))?,
                    shape_input,
                )
            };
        let gate = matrix("gate", &input)?;
        let up = matrix("up", &input)?;
        let down = matrix("down", &down_authority)?;
        if (gate.rows, gate.columns) != (2048, 4096)
            || (up.rows, up.columns) != (2048, 4096)
            || (down.rows, down.columns) != (4096, 2048)
        {
            return Err("expert projection shape mismatch".to_owned());
        }
        projections.insert(expert_id, (gate, up, down));
    }

    let kernel_source = fs::read_to_string(kernel_path)
        .map_err(|error| format!("{}: {error}", kernel_path.display()))?;
    if !kernel_source.contains(&format!("kernel void {KERNEL}")) {
        return Err(format!("kernel source lacks {KERNEL}"));
    }
    let device = Device::system_default().ok_or("no Metal device is available")?;
    let options = CompileOptions::new();
    options.set_fast_math_enabled(false);
    let compile_start = Instant::now();
    let library = device
        .new_library_with_source(&kernel_source, &options)
        .map_err(|error| format!("Metal compilation failed: {error}"))?;
    let function = library
        .get_function(KERNEL, None)
        .map_err(|error| format!("Metal kernel lookup failed: {error}"))?;
    let pipeline = device
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|error| format!("Metal pipeline creation failed: {error}"))?;
    let compile_ms = compile_start.elapsed().as_secs_f64() * 1000.0;
    let queue = device.new_command_queue();
    let lut = (0_u16..=255)
        .map(|value| decode_f8_e4m3fn(value as u8))
        .collect::<Vec<_>>();
    safety.checkpoint("after_compile")?;

    let execute = || -> Result<RoutedRowExecution, String> {
        let start = Instant::now();
        let logits = accelerate_sgemm_right_transposed(&input, &router_weights, 1, 256, 4096)?;
        let (selected, weights, minimum_boundary_margin) =
            component_pytorch_noaux_route(&logits, &correction)?;
        let mut output = vec![0.0_f32; 4096];
        let mut expert_outputs = Vec::with_capacity(8);
        for (&expert_id, &route_weight) in selected.iter().zip(&weights) {
            let (gate, up, down) = projections
                .get(&expert_id)
                .ok_or_else(|| format!("runtime selected unauthoritative expert {expert_id}"))?;
            let expert_execution = execute_staged_expert(
                &device,
                &queue,
                &pipeline,
                &lut,
                [gate, up, down],
                &input,
                None,
                [SourceBufferMode::Copied; 3],
            )?;
            for (destination, value) in output.iter_mut().zip(&expert_execution.down) {
                *destination += *value * route_weight;
            }
            expert_outputs.push((expert_id, expert_execution));
        }
        round_bf16_values(&mut output);
        Ok(RoutedRowExecution {
            output,
            selected,
            weights,
            minimum_boundary_margin,
            expert_outputs,
            wall_ms: start.elapsed().as_secs_f64() * 1000.0,
        })
    };
    let cold_wall_ms = execute()?.wall_ms;
    for _ in 0..WARMUPS {
        execute()?;
    }
    safety.checkpoint("after_warmups")?;
    let mut wall_ms = Vec::with_capacity(MEASUREMENTS);
    let mut final_execution = None;
    for _ in 0..MEASUREMENTS {
        let execution = execute()?;
        wall_ms.push(execution.wall_ms);
        final_execution = Some(execution);
    }
    safety.checkpoint("after_timed_series")?;
    let final_execution = final_execution.ok_or("missing routed-row execution")?;
    let output = final_execution.output;
    let selected = final_execution.selected;
    let weights = final_execution.weights;
    let minimum_topk_boundary_margin = final_execution.minimum_boundary_margin;
    let expert_outputs = final_execution.expert_outputs;
    let sparse_repair_counts =
        expert_outputs
            .iter()
            .fold([0_usize; 3], |mut total, (_, expert)| {
                for (destination, count) in total.iter_mut().zip(expert.repairs) {
                    *destination += count;
                }
                total
            });
    if selected != frozen_selected {
        return Err(format!("native route order mismatch: {selected:?}"));
    }
    let maximum_route_weight_absolute_error = weights
        .iter()
        .zip(&oracle_weights)
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0_f32, f32::max);
    if maximum_route_weight_absolute_error > 3.0e-8 {
        return Err(format!(
            "native route-weight error {maximum_route_weight_absolute_error}"
        ));
    }
    let mut expert_diagnostics = Vec::new();
    for (expert, actual) in &expert_outputs {
        let captures = reference_manifest
            .expert_outputs
            .get(&expert.to_string())
            .ok_or("missing expert oracle captures")?;
        for (stage, values) in [
            ("gate", actual.gate.as_slice()),
            ("up", actual.up.as_slice()),
            ("swiglu", actual.hidden.as_slice()),
            ("down", actual.down.as_slice()),
        ] {
            let capture = captures.get(stage).ok_or("missing expert oracle stage")?;
            if capture.shape != [values.len()]
                || capture.dtype != "BF16_widened_F32"
                || Path::new(&capture.file)
                    .file_name()
                    .and_then(|name| name.to_str())
                    != Some(capture.file.as_str())
            {
                return Err("expert oracle capture metadata mismatch".to_owned());
            }
            let (bytes, expected) = read_f32_file(
                &reference_path.with_file_name(&capture.file),
                Some(values.len()),
            )?;
            if sha256_hex(&bytes) != capture.sha256 {
                return Err("expert oracle capture SHA-256 mismatch".to_owned());
            }
            let equal = values
                .iter()
                .zip(&expected)
                .filter(|(left, right)| left.to_bits() == right.to_bits())
                .count();
            let maximum = values
                .iter()
                .zip(&expected)
                .map(|(left, right)| (left - right).abs())
                .fold(0.0_f32, f32::max);
            let pre_round = match stage {
                "gate" => Some(actual.gate_pre_round.as_slice()),
                "up" => Some(actual.up_pre_round.as_slice()),
                "down" => Some(actual.down_pre_round.as_slice()),
                _ => None,
            };
            let repair_count = match stage {
                "gate" => actual.repairs[0],
                "up" => actual.repairs[1],
                "down" => actual.repairs[2],
                _ => 0,
            };
            let mismatches = values
                .iter()
                .zip(&expected)
                .enumerate()
                .filter(|(_, (left, right))| left.to_bits() != right.to_bits())
                .take(8)
                .map(|(index, (candidate, oracle))| {
                    let pre = pre_round.map_or(*candidate, |values| values[index]);
                    let low = pre.to_bits() & 0xffff;
                    let midpoint_distance = low.abs_diff(0x8000);
                    format!(
                        "{index}:actual={:#010x},expected={:#010x},pre={:#010x},midpoint_distance={midpoint_distance}",
                        candidate.to_bits(),
                        oracle.to_bits(),
                        pre.to_bits()
                    )
                })
                .collect::<Vec<_>>();
            expert_diagnostics.push(format!(
                "expert={expert},stage={stage},equal={equal}/{},max={maximum},repairs={repair_count},mismatches={mismatches:?}",
                values.len()
            ));
        }
    }
    let mut squared_error = 0.0_f64;
    let mut squared_reference = 0.0_f64;
    let mut maximum_absolute_error = 0.0_f32;
    let mut equal = 0_usize;
    for (&actual, &expected) in output.iter().zip(&reference) {
        let difference = actual - expected;
        squared_error += f64::from(difference).powi(2);
        squared_reference += f64::from(expected).powi(2);
        maximum_absolute_error = maximum_absolute_error.max(difference.abs());
        equal += usize::from(actual.to_bits() == expected.to_bits());
    }
    if squared_reference == 0.0 {
        return Err("routed-row reference has zero L2 norm".to_owned());
    }
    let relative_l2 = (squared_error / squared_reference).sqrt();
    let bf16_equality_fraction = equal as f64 / output.len() as f64;
    if relative_l2 > 5.0e-4 || maximum_absolute_error > 2.0e-2 || bf16_equality_fraction < 0.99 {
        let mut diagnostic_ordered = wall_ms.clone();
        diagnostic_ordered.sort_by(f64::total_cmp);
        let diagnostic_median =
            diagnostic_ordered[((diagnostic_ordered.len() - 1) as f64 * 0.5).round() as usize];
        let failure_safety = safety.released()?;
        let minimum_free = failure_safety
            .iter()
            .map(|snapshot| snapshot.system_memory_free_percent)
            .min()
            .ok_or("missing failure safety snapshot")?;
        let maximum_peak = failure_safety
            .iter()
            .map(|snapshot| snapshot.process_peak_resident_bytes)
            .max()
            .ok_or("missing failure safety snapshot")?;
        let maximum_swap_growth = failure_safety
            .iter()
            .map(|snapshot| snapshot.swap_growth_bytes)
            .max()
            .ok_or("missing failure safety snapshot")?;
        let maximum_new_throttled = failure_safety
            .iter()
            .map(|snapshot| snapshot.new_throttled_pages)
            .max()
            .ok_or("missing failure safety snapshot")?;
        let post_release = failure_safety
            .last()
            .ok_or("missing post-release safety snapshot")?
            .process_physical_footprint_bytes;
        return Err(format!(
            "routed-row parity failed: rel L2 {relative_l2}, max abs {maximum_absolute_error}, BF16 equality {bf16_equality_fraction}, route error {maximum_route_weight_absolute_error}, median ms {diagnostic_median}, safety min-free {minimum_free}%, peak {maximum_peak}, post-release {post_release}, swap-growth {maximum_swap_growth}, new-throttled {maximum_new_throttled}, experts {expert_diagnostics:?}"
        ));
    }
    let output_bytes = output
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect::<Vec<_>>();
    write_create_new(output_path, &output_bytes)?;
    let mut ordered = wall_ms.clone();
    ordered.sort_by(f64::total_cmp);
    let percentile =
        |fraction: f64| ordered[((ordered.len() - 1) as f64 * fraction).round() as usize];
    let wall_p10_ms = percentile(0.10);
    let wall_median_ms = percentile(0.50);
    let wall_p90_ms = percentile(0.90);
    let safety_snapshots = safety.released()?;
    let expert_bytes = projections
        .values()
        .map(|(gate, up, down)| {
            gate.weight.metadata.data_bytes
                + gate.scale.metadata.data_bytes
                + up.weight.metadata.data_bytes
                + up.scale.metadata.data_bytes
                + down.weight.metadata.data_bytes
                + down.scale.metadata.data_bytes
        })
        .sum::<u64>();
    let logical_source_bytes_per_execution = expert_bytes
        + router_weight.metadata.data_bytes
        + router_bias.metadata.data_bytes
        + 4096 * 4
        + 4096 * 4;
    Ok(BoundedRoutedRowReport {
        schema_version: 1,
        semantic: "mimo_layer43_bounded_dynamic_bf16_staged_metal_routed_row",
        commit: commit.to_owned(),
        manifest_sha256: MANIFEST_SHA256.to_owned(),
        input_sha256: INPUT_SHA256.to_owned(),
        router_sha256: ROUTER_SHA256.to_owned(),
        reference_manifest_sha256: sha256_hex(&reference_manifest_bytes),
        reference_sha256: sha256_hex(&reference_bytes),
        output_sha256: sha256_hex(&output_bytes),
        selected_experts: selected,
        route_weights: weights,
        maximum_route_weight_absolute_error,
        minimum_topk_boundary_margin,
        relative_l2,
        maximum_absolute_error,
        bf16_equality_fraction,
        device: device.name().to_owned(),
        compile_ms,
        cold_wall_ms,
        warmups: WARMUPS,
        measurements: MEASUREMENTS,
        wall_ms,
        wall_p10_ms,
        wall_median_ms,
        wall_p90_ms,
        speedup_vs_pw0096_routed_layer: 3180.0 / wall_median_ms,
        timing_gates_passed: wall_median_ms <= 100.0 && 3180.0 / wall_median_ms >= 10.0,
        logical_source_bytes_per_execution,
        maximum_resident_tensor_buffer_bytes: 8_390_656,
        sparse_repair_counts,
        sparse_decoded_weight_bytes: (sparse_repair_counts[0] * 4096
            + sparse_repair_counts[1] * 4096
            + sparse_repair_counts[2] * 2048) as u64
            * 4,
        batch_size: 1,
        concurrency: 1,
        accepted_tokens: 0,
        accepted_per_verification: 0,
        unique_experts: 8,
        safety_snapshots,
        cache_state: "warm OS cache after artifact identity hashing; pipeline/LUT/router resident; one expert tensor set installed and released at a time",
        performance_claim: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bf16_rounding_is_ties_to_even() {
        assert_eq!(
            round_bf16(f32::from_bits(0x3f80_8000)).to_bits(),
            0x3f80_0000
        );
        assert_eq!(
            round_bf16(f32::from_bits(0x3f81_8000)).to_bits(),
            0x3f82_0000
        );
    }

    #[test]
    fn dynamic_fp8_rejects_bad_inputs() {
        assert!(dynamic_fp8_dequantized(&[]).is_err());
        assert!(dynamic_fp8_dequantized(&[0.0; 127]).is_err());
        let mut values = vec![0.0; 128];
        values[9] = f32::NAN;
        assert!(dynamic_fp8_dequantized(&values).is_err());
    }

    #[test]
    fn swiglu_rounds_silu_before_the_product() {
        let gate = 1.03125_f32;
        let up = 0.71484375_f32;
        let silu = round_bf16(gate / (1.0 + (-gate).exp()));
        assert_eq!(staged_swiglu(gate, up), round_bf16(silu * up));
    }

    #[test]
    fn bf16_midpoint_distance_is_value_derived() {
        assert_eq!(bf16_midpoint_distance_ulps(f32::from_bits(0x3f80_8000)), 0);
        assert_eq!(bf16_midpoint_distance_ulps(f32::from_bits(0xbf81_8001)), 1);
        assert_eq!(
            bf16_midpoint_distance_ulps(f32::from_bits(0x3f80_a000)),
            8192
        );
    }

    #[test]
    fn dynamic_fp8_is_deterministic_and_finite() {
        let input = (0..256)
            .map(|x| (x as f32 - 127.0) / 31.0)
            .collect::<Vec<_>>();
        let first = dynamic_fp8_dequantized(&input).expect("valid input");
        let second = dynamic_fp8_dequantized(&input).expect("valid input");
        assert_eq!(first, second);
        assert!(first.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn tomography_serialization_retains_projection_identity_and_null_gpu_time() {
        let activity = ProcessActivityDelta::default();
        let projection = ProjectionTomography {
            projection: "gate",
            rows: 2048,
            columns: 4096,
            source_weight_bytes: 8_388_608,
            source_scale_bytes: 2048,
            source_buffer_install_ms: 1.0,
            small_buffer_install_ms: 2.0,
            command_encode_ms: 3.0,
            commit_call_ms: 4.0,
            synchronous_wait_ms: 5.0,
            gpu_interval_ms: None,
            readback_ms: 6.0,
            explicit_release_ms: 7.0,
            wall_ms: 28.0,
            source_install_activity: activity,
            total_activity: activity,
        };
        let value = serde_json::to_value(projection).expect("serialize tomography");
        assert_eq!(value["projection"], "gate");
        assert_eq!(value["rows"], 2048);
        assert!(value["gpu_interval_ms"].is_null());
    }

    #[test]
    fn transaction_regions_fail_closed_on_every_alignment_and_length_boundary() {
        assert!(validate_no_copy_region(0x4000, 16_000, 16_384, 16_384));
        assert!(!validate_no_copy_region(0x4001, 16_000, 16_384, 16_384));
        assert!(!validate_no_copy_region(0x4000, 16_000, 16_383, 16_384));
        assert!(!validate_no_copy_region(0x4000, 16_385, 16_384, 16_384));
        assert!(!validate_no_copy_region(0x4000, 1, 16_384, 0));
        assert!(!validate_no_copy_region(0x4000, 1, 16_384, 12_288));
    }

    #[test]
    fn transaction_phase_completion_fails_closed_and_bounds_gpu_time() {
        assert_eq!(
            validate_phase_completion("gate_up", MTLCommandBufferStatus::Completed, Some(4.0), 4.0,),
            Ok(4.0)
        );
        assert!(
            validate_phase_completion("gate_up", MTLCommandBufferStatus::Error, Some(1.0), 1.0)
                .is_err()
        );
        assert!(
            validate_phase_completion("gate_up", MTLCommandBufferStatus::Completed, None, 1.0)
                .is_err()
        );
        assert!(
            validate_phase_completion(
                "gate_up",
                MTLCommandBufferStatus::Completed,
                Some(2.01),
                1.0,
            )
            .is_err()
        );
    }
}
