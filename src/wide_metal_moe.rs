use super::{MappedNoCopyRegion, decode_f8_e4m3fn};
use metal::{CompileOptions, Device, MTLCommandBufferStatus, MTLResourceOptions, MTLSize, NSRange};
use std::fs;
use std::path::Path;
use std::time::Instant;

const BATCH: usize = 8;
const HIDDEN: usize = 4096;
const INTERMEDIATE: usize = 2048;
const PROJECTION_LANES: u64 = 64;

fn active_rows(values: &[f32], columns: usize) -> Result<usize, String> {
    if columns == 0 || values.is_empty() || !values.len().is_multiple_of(columns) {
        return Err("wide row shape mismatch".to_owned());
    }
    let rows = values.len() / columns;
    if !(1..=BATCH).contains(&rows) {
        return Err("wide row count must be between one and eight".to_owned());
    }
    Ok(rows)
}

fn padded_rows(values: &[f32], rows: usize, columns: usize) -> Vec<f32> {
    let mut padded = vec![0.0_f32; BATCH * columns];
    padded[..rows * columns].copy_from_slice(values);
    padded
}

pub(crate) struct WideProjectionBinding<'a> {
    pub(crate) weight: MappedNoCopyRegion<'a>,
    pub(crate) scale: MappedNoCopyRegion<'a>,
    pub(crate) copy_weight: bool,
    pub(crate) rows: usize,
    pub(crate) columns: usize,
}

pub(crate) struct WideExpertBinding<'a> {
    pub(crate) expert: u32,
    pub(crate) positions: Vec<u32>,
    pub(crate) route_weights: Vec<f32>,
    pub(crate) gate: WideProjectionBinding<'a>,
    pub(crate) up: WideProjectionBinding<'a>,
    pub(crate) down: WideProjectionBinding<'a>,
}

pub(crate) struct WideMoeExecution {
    pub(crate) output: Vec<f32>,
    pub(crate) wall_ms: f64,
    pub(crate) unique_experts: usize,
    pub(crate) expert_rows: usize,
    pub(crate) mapped_source_bytes: u64,
}

pub(crate) struct WideLinearExecution {
    pub(crate) output: Vec<f32>,
    pub(crate) wall_ms: f64,
    pub(crate) mapped_source_bytes: u64,
}

pub(crate) struct WideMetalMoeRuntime {
    device: metal::Device,
    queue: metal::CommandQueue,
    projection_pipelines: Vec<metal::ComputePipelineState>,
    dynamic_pipeline: metal::ComputePipelineState,
    swiglu_pipeline: metal::ComputePipelineState,
    round_pipeline: metal::ComputePipelineState,
    scatter_pipeline: metal::ComputePipelineState,
    full_qkv_pipeline: metal::ComputePipelineState,
    bf16_pipeline: metal::ComputePipelineState,
    lut_buffer: metal::Buffer,
    pub(crate) compile_ms: f64,
    pub(crate) device_name: String,
}

#[repr(C)]
struct GemvShape {
    rows: u32,
    columns: u32,
    block_rows: u32,
    block_columns: u32,
}

#[repr(C)]
struct ScatterShape {
    count: u32,
    width: u32,
}

struct ProjectionBuffers {
    weight: metal::Buffer,
    scale: metal::Buffer,
    weight_offset: u64,
    scale_offset: u64,
}

struct ExpertBuffers {
    count: usize,
    input: metal::Buffer,
    gate: ProjectionBuffers,
    up: ProjectionBuffers,
    down: ProjectionBuffers,
    route_weights: metal::Buffer,
    positions: metal::Buffer,
    scatter_shape: metal::Buffer,
    hidden_count: metal::Buffer,
    output_count: metal::Buffer,
}

impl WideMetalMoeRuntime {
    pub(crate) fn compile(kernel_path: &Path) -> Result<Self, String> {
        let source = fs::read_to_string(kernel_path)
            .map_err(|error| format!("{}: {error}", kernel_path.display()))?;
        let device = Device::system_default().ok_or("no Metal device is available")?;
        if device.max_threads_per_threadgroup().width < 128 {
            return Err("Metal device cannot dispatch the wide MoE kernels".to_owned());
        }
        let options = CompileOptions::new();
        options.set_fast_math_enabled(false);
        let started = Instant::now();
        let library = device
            .new_library_with_source(&source, &options)
            .map_err(|error| format!("wide MoE Metal compilation failed: {error}"))?;
        let pipeline = |name: &str| -> Result<metal::ComputePipelineState, String> {
            let function = library
                .get_function(name, None)
                .map_err(|error| format!("wide MoE kernel {name}: {error}"))?;
            device
                .new_compute_pipeline_state_with_function(&function)
                .map_err(|error| format!("wide MoE pipeline {name}: {error}"))
        };
        let projection_pipelines = (1..=BATCH)
            .map(|width| pipeline(&format!("block_fp8_gemm{width}_shared_weight_lut_blocked")))
            .collect::<Result<Vec<_>, _>>()?;
        let dynamic_pipeline = pipeline("dynamic_fp8_dequantized_group128")?;
        let swiglu_pipeline = pipeline("bf16_staged_swiglu")?;
        let round_pipeline = pipeline("bf16_round_in_place")?;
        let scatter_pipeline = pipeline("route_weighted_scatter_add_f32")?;
        let full_qkv_pipeline = pipeline("full_qkv_fp8_gemm8_shared_weight_lut_blocked")?;
        let bf16_pipeline = pipeline("bf16_gemm8_shared_weight")?;
        let decode_lut = (0_u16..=255)
            .map(|bits| decode_f8_e4m3fn(bits as u8))
            .collect::<Vec<_>>();
        let shared = MTLResourceOptions::StorageModeShared;
        let lut_buffer = device.new_buffer_with_data(
            decode_lut.as_ptr().cast(),
            std::mem::size_of_val(decode_lut.as_slice()) as u64,
            shared,
        );
        let compile_ms = started.elapsed().as_secs_f64() * 1000.0;
        let queue = device.new_command_queue();
        let device_name = device.name().to_owned();
        Ok(Self {
            device,
            queue,
            projection_pipelines,
            dynamic_pipeline,
            swiglu_pipeline,
            round_pipeline,
            scatter_pipeline,
            full_qkv_pipeline,
            bf16_pipeline,
            lut_buffer,
            compile_ms,
            device_name,
        })
    }

    pub(crate) fn execute_fp8_linear(
        &self,
        input: &[f32],
        binding: &WideProjectionBinding<'_>,
        full_qkv_layout: bool,
    ) -> Result<WideLinearExecution, String> {
        let active_rows = active_rows(input, binding.columns)?;
        if input.iter().any(|value| !value.is_finite())
            || binding.weight.tensor_bytes != binding.rows * binding.columns
            || (!full_qkv_layout
                && binding.scale.tensor_bytes != binding.rows / 128 * (binding.columns / 128) * 4)
            || (full_qkv_layout
                && (binding.rows, binding.columns, binding.scale.tensor_bytes)
                    != (13_568, HIDDEN, 108 * 32 * 4))
        {
            return Err("wide FP8 linear layout mismatch".to_owned());
        }
        let padded_input = padded_rows(input, active_rows, binding.columns);
        let started = Instant::now();
        let shared = MTLResourceOptions::StorageModeShared;
        let no_copy = |region: &MappedNoCopyRegion<'_>| {
            self.device.new_buffer_with_bytes_no_copy(
                region.bytes.as_ptr().cast(),
                region.bytes.len() as u64,
                shared,
                None,
            )
        };
        let weight = if binding.copy_weight {
            let bytes = &binding.weight.bytes[binding.weight.tensor_offset
                ..binding.weight.tensor_offset + binding.weight.tensor_bytes];
            self.device
                .new_buffer_with_data(bytes.as_ptr().cast(), bytes.len() as u64, shared)
        } else {
            no_copy(&binding.weight)
        };
        let scale = no_copy(&binding.scale);
        let input_buffer = self.device.new_buffer_with_data(
            padded_input.as_ptr().cast(),
            std::mem::size_of_val(padded_input.as_slice()) as u64,
            shared,
        );
        let staged = self
            .device
            .new_buffer((padded_input.len() * 4) as u64, shared);
        let output_count = BATCH * binding.rows;
        let output = self.device.new_buffer((output_count * 4) as u64, shared);
        let error = 0_u32;
        let error_buffer = self.device.new_buffer_with_data(
            (&error as *const u32).cast(),
            std::mem::size_of::<u32>() as u64,
            shared,
        );
        let shape = GemvShape {
            rows: binding.rows as u32,
            columns: binding.columns as u32,
            block_rows: 128,
            block_columns: 128,
        };
        let shape_buffer = self.device.new_buffer_with_data(
            (&shape as *const GemvShape).cast(),
            std::mem::size_of::<GemvShape>() as u64,
            shared,
        );
        let output_count_u32 = output_count as u32;
        let output_count_buffer = self.device.new_buffer_with_data(
            (&output_count_u32 as *const u32).cast(),
            std::mem::size_of::<u32>() as u64,
            shared,
        );
        let command = self.queue.new_command_buffer();
        let encoder = command.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.dynamic_pipeline);
        encoder.set_buffer(0, Some(&input_buffer), 0);
        encoder.set_buffer(1, Some(&staged), 0);
        encoder.set_buffer(2, Some(&self.lut_buffer), 0);
        encoder.set_buffer(3, Some(&error_buffer), 0);
        encoder.set_threadgroup_memory_length(0, 128 * 4);
        encoder.dispatch_thread_groups(
            MTLSize {
                width: (padded_input.len() / 128) as u64,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: 128,
                height: 1,
                depth: 1,
            },
        );
        encoder.set_compute_pipeline_state(if full_qkv_layout {
            &self.full_qkv_pipeline
        } else {
            &self.projection_pipelines[BATCH - 1]
        });
        encoder.set_buffer(
            0,
            Some(&weight),
            if binding.copy_weight {
                0
            } else {
                binding.weight.tensor_offset as u64
            },
        );
        encoder.set_buffer(1, Some(&scale), binding.scale.tensor_offset as u64);
        encoder.set_buffer(2, Some(&staged), 0);
        encoder.set_buffer(3, Some(&output), 0);
        encoder.set_buffer(4, Some(&shape_buffer), 0);
        encoder.set_buffer(5, Some(&self.lut_buffer), 0);
        encoder.set_threadgroup_memory_length(0, PROJECTION_LANES * BATCH as u64 * 4);
        encoder.dispatch_thread_groups(
            MTLSize {
                width: binding.rows as u64,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: PROJECTION_LANES,
                height: 1,
                depth: 1,
            },
        );
        encoder.set_compute_pipeline_state(&self.round_pipeline);
        encoder.set_buffer(0, Some(&output), 0);
        encoder.set_buffer(1, Some(&output_count_buffer), 0);
        encoder.set_buffer(2, Some(&error_buffer), 0);
        encoder.dispatch_threads(
            MTLSize {
                width: output_count as u64,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: 256,
                height: 1,
                depth: 1,
            },
        );
        encoder.end_encoding();
        command.commit();
        command.wait_until_completed();
        if command.status() != MTLCommandBufferStatus::Completed
            || unsafe { *(error_buffer.contents().cast::<u32>()) } != 0
        {
            return Err("wide FP8 linear Metal transaction failed".to_owned());
        }
        let mut values = unsafe {
            std::slice::from_raw_parts(output.contents().cast::<f32>(), output_count).to_vec()
        };
        values.truncate(active_rows * binding.rows);
        Ok(WideLinearExecution {
            output: values,
            wall_ms: started.elapsed().as_secs_f64() * 1000.0,
            mapped_source_bytes: (binding.weight.bytes.len() + binding.scale.bytes.len()) as u64,
        })
    }

    pub(crate) fn execute_bf16_linear(
        &self,
        input: &[f32],
        weight_region: &MappedNoCopyRegion<'_>,
        rows: usize,
        columns: usize,
    ) -> Result<WideLinearExecution, String> {
        let active_rows = active_rows(input, columns)?;
        if input.iter().any(|value| !value.is_finite())
            || weight_region.tensor_bytes != rows * columns * 2
        {
            return Err("wide BF16 linear layout mismatch".to_owned());
        }
        let padded_input = padded_rows(input, active_rows, columns);
        let started = Instant::now();
        let shared = MTLResourceOptions::StorageModeShared;
        let weight = self.device.new_buffer_with_bytes_no_copy(
            weight_region.bytes.as_ptr().cast(),
            weight_region.bytes.len() as u64,
            shared,
            None,
        );
        let input_buffer = self.device.new_buffer_with_data(
            padded_input.as_ptr().cast(),
            std::mem::size_of_val(padded_input.as_slice()) as u64,
            shared,
        );
        let output_count = BATCH * rows;
        let output = self.device.new_buffer((output_count * 4) as u64, shared);
        let shape = GemvShape {
            rows: rows as u32,
            columns: columns as u32,
            block_rows: 1,
            block_columns: 1,
        };
        let shape_buffer = self.device.new_buffer_with_data(
            (&shape as *const GemvShape).cast(),
            std::mem::size_of::<GemvShape>() as u64,
            shared,
        );
        let error = 0_u32;
        let error_buffer = self.device.new_buffer_with_data(
            (&error as *const u32).cast(),
            std::mem::size_of::<u32>() as u64,
            shared,
        );
        let output_count_u32 = output_count as u32;
        let output_count_buffer = self.device.new_buffer_with_data(
            (&output_count_u32 as *const u32).cast(),
            std::mem::size_of::<u32>() as u64,
            shared,
        );
        let command = self.queue.new_command_buffer();
        let encoder = command.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.bf16_pipeline);
        encoder.set_buffer(0, Some(&weight), weight_region.tensor_offset as u64);
        encoder.set_buffer(1, Some(&input_buffer), 0);
        encoder.set_buffer(2, Some(&output), 0);
        encoder.set_buffer(3, Some(&shape_buffer), 0);
        encoder.set_threadgroup_memory_length(0, PROJECTION_LANES * BATCH as u64 * 4);
        encoder.dispatch_thread_groups(
            MTLSize {
                width: rows as u64,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: PROJECTION_LANES,
                height: 1,
                depth: 1,
            },
        );
        encoder.set_compute_pipeline_state(&self.round_pipeline);
        encoder.set_buffer(0, Some(&output), 0);
        encoder.set_buffer(1, Some(&output_count_buffer), 0);
        encoder.set_buffer(2, Some(&error_buffer), 0);
        encoder.dispatch_threads(
            MTLSize {
                width: output_count as u64,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: 256,
                height: 1,
                depth: 1,
            },
        );
        encoder.end_encoding();
        command.commit();
        command.wait_until_completed();
        if command.status() != MTLCommandBufferStatus::Completed
            || unsafe { *(error_buffer.contents().cast::<u32>()) } != 0
        {
            return Err("wide BF16 linear Metal transaction failed".to_owned());
        }
        let mut values = unsafe {
            std::slice::from_raw_parts(output.contents().cast::<f32>(), output_count).to_vec()
        };
        values.truncate(active_rows * rows);
        Ok(WideLinearExecution {
            output: values,
            wall_ms: started.elapsed().as_secs_f64() * 1000.0,
            mapped_source_bytes: weight_region.bytes.len() as u64,
        })
    }

    pub(crate) fn execute(
        &self,
        input: &[f32],
        experts: &[WideExpertBinding<'_>],
    ) -> Result<WideMoeExecution, String> {
        let active_rows = active_rows(input, HIDDEN)?;
        if input.iter().any(|value| !value.is_finite())
            || experts.is_empty()
            || experts
                .iter()
                .map(|expert| expert.positions.len())
                .sum::<usize>()
                != active_rows * 8
        {
            return Err("wide MoE input or schedule mismatch".to_owned());
        }
        let padded_input = padded_rows(input, active_rows, HIDDEN);
        let started = Instant::now();
        let shared = MTLResourceOptions::StorageModeShared;
        let projection =
            |binding: &WideProjectionBinding<'_>| -> Result<ProjectionBuffers, String> {
                if binding.weight.tensor_bytes != binding.rows * binding.columns
                    || binding.scale.tensor_bytes
                        != binding.rows / 128 * (binding.columns / 128) * 4
                    || binding.rows % 128 != 0
                    || binding.columns % 128 != 0
                {
                    return Err("wide MoE projection layout mismatch".to_owned());
                }
                let weight = if binding.copy_weight {
                    let bytes = &binding.weight.bytes[binding.weight.tensor_offset
                        ..binding.weight.tensor_offset + binding.weight.tensor_bytes];
                    self.device.new_buffer_with_data(
                        bytes.as_ptr().cast(),
                        bytes.len() as u64,
                        shared,
                    )
                } else {
                    self.device.new_buffer_with_bytes_no_copy(
                        binding.weight.bytes.as_ptr().cast(),
                        binding.weight.bytes.len() as u64,
                        shared,
                        None,
                    )
                };
                Ok(ProjectionBuffers {
                    weight,
                    scale: self.device.new_buffer_with_bytes_no_copy(
                        binding.scale.bytes.as_ptr().cast(),
                        binding.scale.bytes.len() as u64,
                        shared,
                        None,
                    ),
                    weight_offset: if binding.copy_weight {
                        0
                    } else {
                        binding.weight.tensor_offset as u64
                    },
                    scale_offset: binding.scale.tensor_offset as u64,
                })
            };
        let mut mapped_source_bytes = 0_u64;
        let mut buffers = Vec::with_capacity(experts.len());
        for expert in experts {
            let count = expert.positions.len();
            if count == 0
                || count > BATCH
                || expert.route_weights.len() != count
                || expert
                    .positions
                    .iter()
                    .any(|position| *position as usize >= active_rows)
                || expert
                    .route_weights
                    .iter()
                    .any(|weight| !weight.is_finite())
                || (expert.gate.rows, expert.gate.columns) != (INTERMEDIATE, HIDDEN)
                || (expert.up.rows, expert.up.columns) != (INTERMEDIATE, HIDDEN)
                || (expert.down.rows, expert.down.columns) != (HIDDEN, INTERMEDIATE)
            {
                return Err(format!(
                    "wide MoE expert {} identity mismatch",
                    expert.expert
                ));
            }
            for region in [
                &expert.gate.weight,
                &expert.gate.scale,
                &expert.up.weight,
                &expert.up.scale,
                &expert.down.weight,
                &expert.down.scale,
            ] {
                mapped_source_bytes = mapped_source_bytes
                    .checked_add(region.bytes.len() as u64)
                    .ok_or("wide MoE mapped-byte overflow")?;
            }
            let mut gathered = vec![0.0_f32; BATCH * HIDDEN];
            for (local, &position) in expert.positions.iter().enumerate() {
                gathered[local * HIDDEN..(local + 1) * HIDDEN].copy_from_slice(
                    &padded_input[position as usize * HIDDEN..(position as usize + 1) * HIDDEN],
                );
            }
            let mut weights = [0.0_f32; BATCH];
            weights[..count].copy_from_slice(&expert.route_weights);
            let mut positions = [0_u32; BATCH];
            positions[..count].copy_from_slice(&expert.positions);
            let scatter_shape = ScatterShape {
                count: count as u32,
                width: HIDDEN as u32,
            };
            let hidden_count = (count * INTERMEDIATE) as u32;
            let output_count = (count * HIDDEN) as u32;
            buffers.push(ExpertBuffers {
                count,
                input: self.device.new_buffer_with_data(
                    gathered.as_ptr().cast(),
                    std::mem::size_of_val(gathered.as_slice()) as u64,
                    shared,
                ),
                gate: projection(&expert.gate)?,
                up: projection(&expert.up)?,
                down: projection(&expert.down)?,
                route_weights: self.device.new_buffer_with_data(
                    weights.as_ptr().cast(),
                    std::mem::size_of_val(&weights) as u64,
                    shared,
                ),
                positions: self.device.new_buffer_with_data(
                    positions.as_ptr().cast(),
                    std::mem::size_of_val(&positions) as u64,
                    shared,
                ),
                scatter_shape: self.device.new_buffer_with_data(
                    (&scatter_shape as *const ScatterShape).cast(),
                    std::mem::size_of::<ScatterShape>() as u64,
                    shared,
                ),
                hidden_count: self.device.new_buffer_with_data(
                    (&hidden_count as *const u32).cast(),
                    std::mem::size_of::<u32>() as u64,
                    shared,
                ),
                output_count: self.device.new_buffer_with_data(
                    (&output_count as *const u32).cast(),
                    std::mem::size_of::<u32>() as u64,
                    shared,
                ),
            });
        }

        let gate_shape = GemvShape {
            rows: INTERMEDIATE as u32,
            columns: HIDDEN as u32,
            block_rows: 128,
            block_columns: 128,
        };
        let down_shape = GemvShape {
            rows: HIDDEN as u32,
            columns: INTERMEDIATE as u32,
            block_rows: 128,
            block_columns: 128,
        };
        let gate_shape_buffer = self.device.new_buffer_with_data(
            (&gate_shape as *const GemvShape).cast(),
            std::mem::size_of::<GemvShape>() as u64,
            shared,
        );
        let down_shape_buffer = self.device.new_buffer_with_data(
            (&down_shape as *const GemvShape).cast(),
            std::mem::size_of::<GemvShape>() as u64,
            shared,
        );
        let staged_input = self.device.new_buffer((BATCH * HIDDEN * 4) as u64, shared);
        let gate_output = self
            .device
            .new_buffer((BATCH * INTERMEDIATE * 4) as u64, shared);
        let up_output = self
            .device
            .new_buffer((BATCH * INTERMEDIATE * 4) as u64, shared);
        let hidden_output = self
            .device
            .new_buffer((BATCH * INTERMEDIATE * 4) as u64, shared);
        let staged_hidden = self
            .device
            .new_buffer((BATCH * INTERMEDIATE * 4) as u64, shared);
        let expert_output = self.device.new_buffer((BATCH * HIDDEN * 4) as u64, shared);
        let block_output = self.device.new_buffer((BATCH * HIDDEN * 4) as u64, shared);
        let zero = 0_u32;
        let error_buffer = self.device.new_buffer_with_data(
            (&zero as *const u32).cast(),
            std::mem::size_of::<u32>() as u64,
            shared,
        );
        let block_count = (active_rows * HIDDEN) as u32;
        let block_count_buffer = self.device.new_buffer_with_data(
            (&block_count as *const u32).cast(),
            std::mem::size_of::<u32>() as u64,
            shared,
        );

        let command = self.queue.new_command_buffer();
        let blit = command.new_blit_command_encoder();
        blit.fill_buffer(
            &block_output,
            NSRange::new(0, (BATCH * HIDDEN * 4) as u64),
            0,
        );
        blit.end_encoding();
        let encoder = command.new_compute_command_encoder();
        for expert in &buffers {
            let pipeline = &self.projection_pipelines[expert.count - 1];
            encoder.set_compute_pipeline_state(&self.dynamic_pipeline);
            encoder.set_buffer(0, Some(&expert.input), 0);
            encoder.set_buffer(1, Some(&staged_input), 0);
            encoder.set_buffer(2, Some(&self.lut_buffer), 0);
            encoder.set_buffer(3, Some(&error_buffer), 0);
            encoder.set_threadgroup_memory_length(0, 128 * 4);
            encoder.dispatch_thread_groups(
                MTLSize {
                    width: (expert.count * HIDDEN / 128) as u64,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: 128,
                    height: 1,
                    depth: 1,
                },
            );

            let project = |encoder: &metal::ComputeCommandEncoderRef,
                           source: &ProjectionBuffers,
                           input: &metal::BufferRef,
                           output: &metal::BufferRef,
                           shape: &metal::BufferRef,
                           rows: usize,
                           columns: usize| {
                encoder.set_compute_pipeline_state(pipeline);
                encoder.set_buffer(0, Some(&source.weight), source.weight_offset);
                encoder.set_buffer(1, Some(&source.scale), source.scale_offset);
                encoder.set_buffer(2, Some(input), 0);
                encoder.set_buffer(3, Some(output), 0);
                encoder.set_buffer(4, Some(shape), 0);
                encoder.set_buffer(5, Some(&self.lut_buffer), 0);
                encoder
                    .set_threadgroup_memory_length(0, PROJECTION_LANES * expert.count as u64 * 4);
                encoder.dispatch_thread_groups(
                    MTLSize {
                        width: rows as u64,
                        height: 1,
                        depth: 1,
                    },
                    MTLSize {
                        width: PROJECTION_LANES,
                        height: 1,
                        depth: 1,
                    },
                );
                let _ = columns;
            };
            project(
                encoder,
                &expert.gate,
                &staged_input,
                &gate_output,
                &gate_shape_buffer,
                INTERMEDIATE,
                HIDDEN,
            );
            project(
                encoder,
                &expert.up,
                &staged_input,
                &up_output,
                &gate_shape_buffer,
                INTERMEDIATE,
                HIDDEN,
            );
            encoder.set_compute_pipeline_state(&self.swiglu_pipeline);
            encoder.set_buffer(0, Some(&gate_output), 0);
            encoder.set_buffer(1, Some(&up_output), 0);
            encoder.set_buffer(2, Some(&hidden_output), 0);
            encoder.set_buffer(3, Some(&expert.hidden_count), 0);
            encoder.set_buffer(4, Some(&error_buffer), 0);
            encoder.dispatch_threads(
                MTLSize {
                    width: (expert.count * INTERMEDIATE) as u64,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: 256,
                    height: 1,
                    depth: 1,
                },
            );
            encoder.set_compute_pipeline_state(&self.dynamic_pipeline);
            encoder.set_buffer(0, Some(&hidden_output), 0);
            encoder.set_buffer(1, Some(&staged_hidden), 0);
            encoder.set_buffer(2, Some(&self.lut_buffer), 0);
            encoder.set_buffer(3, Some(&error_buffer), 0);
            encoder.set_threadgroup_memory_length(0, 128 * 4);
            encoder.dispatch_thread_groups(
                MTLSize {
                    width: (expert.count * INTERMEDIATE / 128) as u64,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: 128,
                    height: 1,
                    depth: 1,
                },
            );
            project(
                encoder,
                &expert.down,
                &staged_hidden,
                &expert_output,
                &down_shape_buffer,
                HIDDEN,
                INTERMEDIATE,
            );
            encoder.set_compute_pipeline_state(&self.round_pipeline);
            encoder.set_buffer(0, Some(&expert_output), 0);
            encoder.set_buffer(1, Some(&expert.output_count), 0);
            encoder.set_buffer(2, Some(&error_buffer), 0);
            encoder.dispatch_threads(
                MTLSize {
                    width: (expert.count * HIDDEN) as u64,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: 256,
                    height: 1,
                    depth: 1,
                },
            );
            encoder.set_compute_pipeline_state(&self.scatter_pipeline);
            encoder.set_buffer(0, Some(&expert_output), 0);
            encoder.set_buffer(1, Some(&expert.route_weights), 0);
            encoder.set_buffer(2, Some(&expert.positions), 0);
            encoder.set_buffer(3, Some(&block_output), 0);
            encoder.set_buffer(4, Some(&expert.scatter_shape), 0);
            encoder.dispatch_threads(
                MTLSize {
                    width: (expert.count * HIDDEN) as u64,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: 256,
                    height: 1,
                    depth: 1,
                },
            );
        }
        encoder.set_compute_pipeline_state(&self.round_pipeline);
        encoder.set_buffer(0, Some(&block_output), 0);
        encoder.set_buffer(1, Some(&block_count_buffer), 0);
        encoder.set_buffer(2, Some(&error_buffer), 0);
        encoder.dispatch_threads(
            MTLSize {
                width: block_count as u64,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: 256,
                height: 1,
                depth: 1,
            },
        );
        encoder.end_encoding();
        command.commit();
        command.wait_until_completed();
        if command.status() != MTLCommandBufferStatus::Completed {
            return Err(format!("wide MoE command failed: {:?}", command.status()));
        }
        let flags = unsafe { *error_buffer.contents().cast::<u32>() };
        if flags != 0 {
            return Err(format!("wide MoE semantic kernels set error flags {flags}"));
        }
        let output = unsafe {
            std::slice::from_raw_parts(block_output.contents().cast::<f32>(), BATCH * HIDDEN)
                .get(..active_rows * HIDDEN)
                .expect("active wide output is bounded")
                .to_vec()
        };
        if output.iter().any(|value| !value.is_finite()) {
            return Err("wide MoE produced non-finite output".to_owned());
        }
        Ok(WideMoeExecution {
            output,
            wall_ms: started.elapsed().as_secs_f64() * 1000.0,
            unique_experts: buffers.len(),
            expert_rows: buffers.iter().map(|expert| expert.count).sum(),
            mapped_source_bytes,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn partial_wide_rows_are_zero_padded_without_changing_real_rows() {
        let values = (0..3 * 4).map(|value| value as f32).collect::<Vec<_>>();
        assert_eq!(active_rows(&values, 4), Ok(3));
        let padded = padded_rows(&values, 3, 4);
        assert_eq!(&padded[..values.len()], values.as_slice());
        assert_eq!(padded.len(), BATCH * 4);
        assert!(padded[values.len()..].iter().all(|value| *value == 0.0));
        assert!(active_rows(&[], 4).is_err());
        assert!(active_rows(&[0.0; 5], 4).is_err());
        assert!(active_rows(&[0.0; 9], 1).is_err());
    }
}
