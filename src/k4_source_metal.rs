use crate::k4_source_bundle::{BundlePayload, K4_EXACTNESS_CLASS, K4SourceLayerBundle};
use crate::text_endpoint::{ComponentSafetyMonitor, SafetySnapshot};
use crate::{decode_f8_e4m3fn, sha256_reader, write_create_new};
use metal::{CompileOptions, Device, MTLCommandBufferStatus, MTLResourceOptions, MTLSize};
use objc::{msg_send, sel, sel_impl};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File};
use std::path::Path;
use std::time::Instant;

const HIDDEN: usize = 4096;
const INTERMEDIATE: usize = 2048;
const K4_CAPACITY: usize = 5;
const TARGET_HOST_MEMORY_BYTES: u64 = 16 * 1024 * 1024 * 1024;
const TARGET_HOST_MODEL: &str = "Macmini9,1";
const TARGET_METAL_DEVICE: &str = "Apple M1";
const SOURCE_EXPERIMENT: &str = "PW-0478";
const RAW_ROUTED_TWO_TPS_BUDGET_MS: f64 = 500.0;
const RAW_ROUTED_THREE_TPS_BUDGET_MS: f64 = 1000.0 / 3.0;
const PW0478_BUNDLE_SHA256: &str =
    "1851c1fe713abce8e6583908937ca831ed591c4abc23517511ff7624f3f9294c";
const PW0478_BUNDLE_MANIFEST_SHA256: &str =
    "86e486d4cc3fcad237b504ffbd6d276ed7c53688f44709f7bc5c5a334479555b";
const PW0478_FIXTURE_SHA256: &str =
    "05439a232c2002530002d95ac29831b38a5c74b1049903406747620b3ce4f64e";
const PW0478_KERNEL_SHA256: [(&str, &str); 4] = [
    (
        "qtip_k4_bundle_batched.metal",
        "50c835699e7f80403d8127bdbe19e572acbf89774144f3bc079cd3a9c68b58c8",
    ),
    (
        "qtip_trellis_gemv_parallel.metal",
        "4da03f279aa6e0ac26723d7f3660caffee94e74a3b12f9886eaa10512ec260ce",
    ),
    (
        "block_fp8_gemv.metal",
        "9bc149eee32ebf28af35929d5fa160edfe9e1767cdcde59a54ec61b7016882ee",
    ),
    (
        "mixed_route_reduce.metal",
        "d20446229683edb5855e6e2b9cf1aadc0183f5d10b976fe52f165cb03384ac84",
    ),
];

fn host_total_memory_bytes() -> Result<u64, String> {
    let mut bytes = 0_u64;
    let mut bytes_len = std::mem::size_of_val(&bytes);
    // SAFETY: `hw.memsize` is NUL-terminated, the output points to a live u64,
    // and `bytes_len` describes that exact destination allocation.
    let status = unsafe {
        libc::sysctlbyname(
            b"hw.memsize\0".as_ptr().cast(),
            (&mut bytes as *mut u64).cast(),
            &mut bytes_len,
            std::ptr::null_mut(),
            0,
        )
    };
    if status != 0 {
        return Err(format!(
            "failed to read hw.memsize: {}",
            std::io::Error::last_os_error()
        ));
    }
    if bytes_len != std::mem::size_of_val(&bytes) || bytes == 0 {
        return Err(format!(
            "hw.memsize returned invalid u64 payload: bytes={bytes}, length={bytes_len}"
        ));
    }
    Ok(bytes)
}

fn host_model() -> Result<String, String> {
    let name = b"hw.model\0";
    let mut bytes_len = 0_usize;
    // SAFETY: the first call requests only the required output length for the
    // NUL-terminated immutable sysctl name.
    let status = unsafe {
        libc::sysctlbyname(
            name.as_ptr().cast(),
            std::ptr::null_mut(),
            &mut bytes_len,
            std::ptr::null_mut(),
            0,
        )
    };
    if status != 0 || bytes_len < 2 {
        return Err(format!(
            "failed to size hw.model: {}",
            std::io::Error::last_os_error()
        ));
    }
    let mut bytes = vec![0_u8; bytes_len];
    // SAFETY: `bytes` is live and sized from the successful query above, and
    // `bytes_len` describes its writable allocation.
    let status = unsafe {
        libc::sysctlbyname(
            name.as_ptr().cast(),
            bytes.as_mut_ptr().cast(),
            &mut bytes_len,
            std::ptr::null_mut(),
            0,
        )
    };
    if status != 0 || bytes_len == 0 || bytes_len > bytes.len() {
        return Err(format!(
            "failed to read hw.model: {}",
            std::io::Error::last_os_error()
        ));
    }
    bytes.truncate(bytes_len);
    if bytes.last() == Some(&0) {
        bytes.pop();
    }
    String::from_utf8(bytes).map_err(|error| format!("hw.model is not UTF-8: {error}"))
}

fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn file_sha256(path: &Path) -> Result<String, String> {
    sha256_reader(&mut File::open(path).map_err(|error| format!("{}: {error}", path.display()))?)
}

fn raw_routed_budget_pass(p90_ms: f64, budget_ms: f64) -> bool {
    p90_ms.is_finite() && budget_ms.is_finite() && p90_ms < budget_ms
}

fn f32_slices_bit_equal(left: &[f32], right: &[f32]) -> bool {
    left.len() == right.len()
        && left
            .iter()
            .zip(right)
            .all(|(left, right)| left.to_bits() == right.to_bits())
}

#[repr(C)]
#[derive(Clone, Copy)]
struct ProjectionShape {
    rows: u32,
    columns: u32,
    tile_columns: u32,
    rank: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct ProjectionOffsets {
    packed: [u32; K4_CAPACITY],
    left_sign: [u32; K4_CAPACITY],
    right_sign: [u32; K4_CAPACITY],
    global_scale: [u32; K4_CAPACITY],
    row_scale: [u32; K4_CAPACITY],
    correction_left: [u32; K4_CAPACITY],
    correction_right: [u32; K4_CAPACITY],
}

#[repr(C)]
#[derive(Clone, Copy)]
struct GemvShape {
    rows: u32,
    columns: u32,
    block_rows: u32,
    block_columns: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct BundleGemvOffsets {
    weights: [u32; K4_CAPACITY],
    scales: [u32; K4_CAPACITY],
}

struct ProjectionWork {
    experts: usize,
    rows: usize,
    columns: usize,
    shape: metal::Buffer,
    offsets: metal::Buffer,
    transformed_input: metal::Buffer,
    raw: metal::Buffer,
    transformed_output: metal::Buffer,
    rank_output: metal::Buffer,
    final_output: metal::Buffer,
    count: metal::Buffer,
    row_count: metal::Buffer,
}

struct SourceWork {
    expert: u32,
    gate_weight: BundlePayload,
    gate_scales: BundlePayload,
    up_weight: BundlePayload,
    up_scales: BundlePayload,
    down_weight: BundlePayload,
    down_scales: BundlePayload,
}

struct SourcePanelWork {
    experts: Vec<u32>,
    gate_offsets: metal::Buffer,
    up_offsets: metal::Buffer,
    down_offsets: metal::Buffer,
    gate_shape: metal::Buffer,
    down_shape: metal::Buffer,
    gate: metal::Buffer,
    up: metal::Buffer,
    hidden: metal::Buffer,
    dynamic_hidden: metal::Buffer,
    output: metal::Buffer,
    hidden_count: metal::Buffer,
    expert_count: metal::Buffer,
}

struct Pipelines {
    dynamic: metal::ComputePipelineState,
    signed_shared: metal::ComputePipelineState,
    signed_inputs: metal::ComputePipelineState,
    projection: metal::ComputePipelineState,
    output_fwht: metal::ComputePipelineState,
    low_rank_shared: metal::ComputePipelineState,
    low_rank_inputs: metal::ComputePipelineState,
    finish: metal::ComputePipelineState,
    swiglu: metal::ComputePipelineState,
    source_gate_up: metal::ComputePipelineState,
    source_projection_batched: metal::ComputePipelineState,
    reduce_source_panel: metal::ComputePipelineState,
}

pub(crate) struct K4SourceMetalRuntime {
    device: metal::Device,
    queue: metal::CommandQueue,
    pipelines: Pipelines,
    decode_lut: metal::Buffer,
    boundaries: metal::Buffer,
    pub(crate) device_name: String,
    pub(crate) compile_ms: f64,
    pub(crate) kernel_sha256: BTreeMap<String, String>,
}

fn pipeline(
    device: &metal::DeviceRef,
    library: &metal::LibraryRef,
    name: &str,
) -> Result<metal::ComputePipelineState, String> {
    let function = library
        .get_function(name, None)
        .map_err(|error| format!("K4/source function {name}: {error}"))?;
    device
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|error| format!("K4/source pipeline {name}: {error}"))
}

fn compile_library(
    device: &metal::DeviceRef,
    path: &Path,
    options: &metal::CompileOptionsRef,
) -> Result<(metal::Library, String), String> {
    let source =
        fs::read_to_string(path).map_err(|error| format!("{}: {error}", path.display()))?;
    let source_sha256 = sha256_hex(source.as_bytes());
    let library = device
        .new_library_with_source(&source, options)
        .map_err(|error| format!("{} [sha256={source_sha256}]: {error}", path.display()))?;
    Ok((library, source_sha256))
}

fn buffer_value<T>(device: &metal::DeviceRef, value: &T) -> metal::Buffer {
    device.new_buffer_with_data(
        (value as *const T).cast(),
        std::mem::size_of::<T>() as u64,
        MTLResourceOptions::StorageModeShared,
    )
}

fn buffer_slice<T>(device: &metal::DeviceRef, values: &[T]) -> metal::Buffer {
    device.new_buffer_with_data(
        values.as_ptr().cast(),
        std::mem::size_of_val(values) as u64,
        MTLResourceOptions::StorageModeShared,
    )
}

fn u32_offset(payload: &BundlePayload) -> Result<u32, String> {
    u32::try_from(payload.offset).map_err(|_| "K4/source bundle offset exceeds U32".to_owned())
}

impl ProjectionWork {
    fn new(
        device: &metal::DeviceRef,
        bundle: &K4SourceLayerBundle,
        name: &str,
    ) -> Result<Self, String> {
        let mut offsets = ProjectionOffsets::default();
        let mut dimensions = None;
        let experts = bundle.k4_experts.len();
        if experts == 0 || experts > K4_CAPACITY {
            return Err("K4 bundle expert count drift".to_owned());
        }
        for (slot, expert) in bundle.k4_experts.iter().enumerate() {
            let projection = bundle
                .records
                .get(expert)
                .and_then(|record| record.projections.get(name))
                .ok_or_else(|| format!("K4 bundle lacks {name} expert {expert}"))?;
            let payload = |role: &str| {
                projection
                    .payloads
                    .get(role)
                    .ok_or_else(|| format!("K4 projection lacks {role}"))
            };
            offsets.packed[slot] = u32_offset(payload("packed")?)?;
            offsets.left_sign[slot] = u32_offset(payload("left_sign")?)?;
            offsets.right_sign[slot] = u32_offset(payload("right_sign")?)?;
            offsets.global_scale[slot] = u32_offset(payload("global_scale")?)?;
            offsets.row_scale[slot] = u32_offset(payload("row_scale")?)?;
            offsets.correction_left[slot] = u32_offset(payload("correction_left")?)?;
            offsets.correction_right[slot] = u32_offset(payload("correction_right")?)?;
            let current = (projection.rows, projection.columns, projection.rank);
            if dimensions
                .replace(current)
                .is_some_and(|old| old != current)
            {
                return Err(format!("K4 {name} panel dimension drift"));
            }
        }
        let (rows, columns, rank) = dimensions.ok_or("empty K4 projection panel")?;
        let shape = ProjectionShape {
            rows: rows as u32,
            columns: columns as u32,
            tile_columns: (columns / 16) as u32,
            rank: rank as u32,
        };
        let count = columns as u32;
        let row_count = rows as u32;
        let shared = MTLResourceOptions::StorageModeShared;
        Ok(Self {
            experts,
            rows,
            columns,
            shape: buffer_value(device, &shape),
            offsets: buffer_value(device, &offsets),
            transformed_input: device.new_buffer((experts * columns * 4) as u64, shared),
            raw: device.new_buffer((experts * rows * 4) as u64, shared),
            transformed_output: device.new_buffer((experts * rows * 4) as u64, shared),
            rank_output: device.new_buffer((experts * rank * 4) as u64, shared),
            final_output: device.new_buffer((experts * rows * 4) as u64, shared),
            count: buffer_value(device, &count),
            row_count: buffer_value(device, &row_count),
        })
    }
}

impl SourceWork {
    fn new(bundle: &K4SourceLayerBundle, expert: u32) -> Result<Self, String> {
        let record = bundle
            .records
            .get(&expert)
            .ok_or_else(|| format!("source bundle lacks expert {expert}"))?;
        let payload = |role: &str| {
            record
                .payloads
                .get(role)
                .cloned()
                .ok_or_else(|| format!("source expert {expert} lacks {role}"))
        };
        Ok(Self {
            expert,
            gate_weight: payload("gate_weight")?,
            gate_scales: payload("gate_scales")?,
            up_weight: payload("up_weight")?,
            up_scales: payload("up_scales")?,
            down_weight: payload("down_weight")?,
            down_scales: payload("down_scales")?,
        })
    }
}

impl SourcePanelWork {
    fn new(device: &metal::DeviceRef, bundle: &K4SourceLayerBundle) -> Result<Self, String> {
        if !matches!(bundle.source_experts.len(), 3 | 4 | 5) {
            return Err("source bundle expert count drift".to_owned());
        }
        let sources = bundle
            .source_experts
            .iter()
            .copied()
            .map(|expert| SourceWork::new(bundle, expert))
            .collect::<Result<Vec<_>, _>>()?;
        let offsets = |weight: fn(&SourceWork) -> &BundlePayload,
                       scales: fn(&SourceWork) -> &BundlePayload| {
            let mut result = BundleGemvOffsets::default();
            for (slot, source) in sources.iter().enumerate() {
                result.weights[slot] = u32_offset(weight(source))?;
                result.scales[slot] = u32_offset(scales(source))?;
            }
            Ok::<_, String>(result)
        };
        let gate_offsets = offsets(|source| &source.gate_weight, |source| &source.gate_scales)?;
        let up_offsets = offsets(|source| &source.up_weight, |source| &source.up_scales)?;
        let down_offsets = offsets(|source| &source.down_weight, |source| &source.down_scales)?;
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
        let expert_count = sources.len() as u32;
        let hidden_count = expert_count * INTERMEDIATE as u32;
        let shared = MTLResourceOptions::StorageModeShared;
        Ok(Self {
            experts: sources.iter().map(|source| source.expert).collect(),
            gate_offsets: buffer_value(device, &gate_offsets),
            up_offsets: buffer_value(device, &up_offsets),
            down_offsets: buffer_value(device, &down_offsets),
            gate_shape: buffer_value(device, &gate_shape),
            down_shape: buffer_value(device, &down_shape),
            gate: device.new_buffer((sources.len() * INTERMEDIATE * 4) as u64, shared),
            up: device.new_buffer((sources.len() * INTERMEDIATE * 4) as u64, shared),
            hidden: device.new_buffer((sources.len() * INTERMEDIATE * 4) as u64, shared),
            dynamic_hidden: device.new_buffer((sources.len() * INTERMEDIATE * 4) as u64, shared),
            output: device.new_buffer((sources.len() * HIDDEN * 4) as u64, shared),
            hidden_count: buffer_value(device, &hidden_count),
            expert_count: buffer_value(device, &expert_count),
        })
    }
}

pub(crate) struct K4SourceExecution {
    pub(crate) output: Vec<f32>,
    pub(crate) wall_ms: f64,
    pub(crate) gpu_ms: f64,
}

impl K4SourceMetalRuntime {
    pub(crate) fn compile(kernel_root: &Path) -> Result<Self, String> {
        let device = Device::system_default().ok_or("no Metal device is available")?;
        if device.max_threads_per_threadgroup().width < 1024 {
            return Err("K4 bundle FWHT requires 1024 Metal threads".to_owned());
        }
        let options = CompileOptions::new();
        options.set_fast_math_enabled(false);
        let started = Instant::now();
        let (bundle_library, bundle_kernel_sha256) = compile_library(
            &device,
            &kernel_root.join("qtip_k4_bundle_batched.metal"),
            &options,
        )?;
        let (common_library, common_kernel_sha256) = compile_library(
            &device,
            &kernel_root.join("qtip_trellis_gemv_parallel.metal"),
            &options,
        )?;
        let (block_library, block_kernel_sha256) =
            compile_library(&device, &kernel_root.join("block_fp8_gemv.metal"), &options)?;
        let (reduce_library, reduce_kernel_sha256) = compile_library(
            &device,
            &kernel_root.join("mixed_route_reduce.metal"),
            &options,
        )?;
        let kernel_sha256 = [
            ("qtip_k4_bundle_batched.metal", bundle_kernel_sha256),
            ("qtip_trellis_gemv_parallel.metal", common_kernel_sha256),
            ("block_fp8_gemv.metal", block_kernel_sha256),
            ("mixed_route_reduce.metal", reduce_kernel_sha256),
        ]
        .into_iter()
        .map(|(name, sha256)| (name.to_owned(), sha256))
        .collect();
        let pipelines = Pipelines {
            dynamic: pipeline(
                &device,
                &common_library,
                "dynamic_fp8_dequantized_group128_binary",
            )?,
            signed_shared: pipeline(
                &device,
                &bundle_library,
                "qtip_k4_bundle_fwht_signed_shared",
            )?,
            signed_inputs: pipeline(
                &device,
                &bundle_library,
                "qtip_k4_bundle_fwht_signed_inputs",
            )?,
            projection: pipeline(&device, &bundle_library, "qtip_k4_bundle_projection")?,
            output_fwht: pipeline(&device, &common_library, "qtip_fwht_fused_batched")?,
            low_rank_shared: pipeline(&device, &bundle_library, "qtip_k4_bundle_low_rank_shared")?,
            low_rank_inputs: pipeline(&device, &bundle_library, "qtip_k4_bundle_low_rank_inputs")?,
            finish: pipeline(&device, &bundle_library, "qtip_k4_bundle_finish")?,
            swiglu: pipeline(&device, &block_library, "bf16_staged_swiglu")?,
            source_gate_up: pipeline(
                &device,
                &block_library,
                "block_fp8_bundle_gate_up_parallel_lut_blocked_shared_input",
            )?,
            source_projection_batched: pipeline(
                &device,
                &block_library,
                "block_fp8_bundle_gemv_parallel_lut_blocked_batched_input",
            )?,
            reduce_source_panel: pipeline(
                &device,
                &reduce_library,
                "mixed_route_weighted_reduce_dynamic_source_panel_bf16",
            )?,
        };
        let decode = (0_u16..=255)
            .map(|bits| decode_f8_e4m3fn(bits as u8))
            .collect::<Vec<_>>();
        let boundaries = (0..126)
            .map(|index| (decode[index] + decode[index + 1]) * 0.5)
            .collect::<Vec<_>>();
        let decode_lut = buffer_slice(&device, &decode);
        let boundaries = buffer_slice(&device, &boundaries);
        let queue = device.new_command_queue();
        let device_name = device.name().to_owned();
        Ok(Self {
            device,
            queue,
            pipelines,
            decode_lut,
            boundaries,
            device_name,
            compile_ms: started.elapsed().as_secs_f64() * 1000.0,
            kernel_sha256,
        })
    }

    pub(crate) fn execute(
        &self,
        bundle: &K4SourceLayerBundle,
        input: &[f32],
        selected: &[u32],
        weights: &[f32],
    ) -> Result<K4SourceExecution, String> {
        self.execute_repeated(bundle, input, selected, weights, 1)
    }

    pub(crate) fn execute_repeated(
        &self,
        bundle: &K4SourceLayerBundle,
        input: &[f32],
        selected: &[u32],
        weights: &[f32],
        repeats: usize,
    ) -> Result<K4SourceExecution, String> {
        if input.len() != HIDDEN
            || selected.len() != 8
            || weights.len() != 8
            || repeats == 0
            || input.iter().chain(weights).any(|value| !value.is_finite())
            || selected.iter().copied().collect::<BTreeSet<_>>()
                != bundle
                    .k4_experts
                    .iter()
                    .copied()
                    .chain(bundle.source_experts.iter().copied())
                    .collect::<BTreeSet<_>>()
        {
            return Err("K4/source execution identity or input mismatch".to_owned());
        }
        let shared = MTLResourceOptions::StorageModeShared;
        let bundle_buffer = self.device.new_buffer_with_bytes_no_copy(
            bundle.mapping.as_ptr().cast_mut().cast(),
            bundle.mapping.len() as u64,
            shared,
            None,
        );
        if bundle_buffer.contents() != bundle.mapping.as_ptr().cast_mut().cast() {
            return Err("K4/source Metal bundle lost no-copy identity".to_owned());
        }
        let input_buffer = buffer_slice(&self.device, input);
        let dynamic_input = self.device.new_buffer((HIDDEN * 4) as u64, shared);
        let gate = ProjectionWork::new(&self.device, bundle, "gate")?;
        let up = ProjectionWork::new(&self.device, bundle, "up")?;
        let down = ProjectionWork::new(&self.device, bundle, "down")?;
        let source_panel = SourcePanelWork::new(&self.device, bundle)?;
        let k4_count = bundle.k4_experts.len();
        let hidden = self
            .device
            .new_buffer((k4_count * INTERMEDIATE * 4) as u64, shared);
        let dynamic_hidden = self
            .device
            .new_buffer((k4_count * INTERMEDIATE * 4) as u64, shared);
        let hidden_count = (k4_count * INTERMEDIATE) as u32;
        let hidden_count_buffer = buffer_value(&self.device, &hidden_count);
        let selected_buffer = buffer_slice(&self.device, selected);
        let weight_buffer = buffer_slice(&self.device, weights);
        let k4_id_buffer = buffer_slice(&self.device, &bundle.k4_experts);
        let source_id_buffer = buffer_slice(&self.device, &bundle.source_experts);
        let k4_count_buffer = buffer_value(&self.device, &(k4_count as u32));
        let output = self.device.new_buffer((HIDDEN * 4) as u64, shared);
        let error = 0_u32;
        let error_buffer = buffer_value(&self.device, &error);

        let started = Instant::now();
        let command = self.queue.new_command_buffer();
        let encoder = command.new_compute_command_encoder();
        for _ in 0..repeats {
            encoder.set_compute_pipeline_state(&self.pipelines.dynamic);
            encoder.set_buffer(0, Some(&input_buffer), 0);
            encoder.set_buffer(1, Some(&dynamic_input), 0);
            encoder.set_buffer(2, Some(&self.decode_lut), 0);
            encoder.set_buffer(3, Some(&error_buffer), 0);
            encoder.set_buffer(4, Some(&self.boundaries), 0);
            encoder.set_threadgroup_memory_length(0, 128 * 4);
            encoder.dispatch_thread_groups(
                MTLSize {
                    width: 32,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: 128,
                    height: 1,
                    depth: 1,
                },
            );

            let encode_projection = |work: &ProjectionWork,
                                     projection_input: &metal::BufferRef,
                                     inputs_are_batched: bool| {
                encoder.set_compute_pipeline_state(if inputs_are_batched {
                    &self.pipelines.signed_inputs
                } else {
                    &self.pipelines.signed_shared
                });
                encoder.set_buffer(0, Some(&bundle_buffer), 0);
                encoder.set_buffer(1, Some(projection_input), 0);
                encoder.set_buffer(2, Some(&work.transformed_input), 0);
                encoder.set_buffer(3, Some(&work.count), 0);
                encoder.set_buffer(4, Some(&work.offsets), 0);
                encoder.set_threadgroup_memory_length(0, (work.columns * 4) as u64);
                encoder.dispatch_thread_groups(
                    MTLSize {
                        width: work.experts as u64,
                        height: 1,
                        depth: 1,
                    },
                    MTLSize {
                        width: 1024,
                        height: 1,
                        depth: 1,
                    },
                );
                encoder.set_compute_pipeline_state(&self.pipelines.projection);
                encoder.set_buffer(0, Some(&bundle_buffer), 0);
                encoder.set_buffer(1, Some(&bundle_buffer), bundle.tlut.offset);
                encoder.set_buffer(2, Some(&work.transformed_input), 0);
                encoder.set_buffer(3, Some(&work.raw), 0);
                encoder.set_buffer(4, Some(&work.shape), 0);
                encoder.set_buffer(5, Some(&work.offsets), 0);
                encoder.set_threadgroup_memory_length(0, 64 * 4);
                encoder.dispatch_thread_groups(
                    MTLSize {
                        width: work.rows as u64,
                        height: work.experts as u64,
                        depth: 1,
                    },
                    MTLSize {
                        width: 64,
                        height: 1,
                        depth: 1,
                    },
                );
                encoder.set_compute_pipeline_state(&self.pipelines.output_fwht);
                encoder.set_buffer(0, Some(&work.raw), 0);
                encoder.set_buffer(1, Some(&work.transformed_output), 0);
                // qtip_fwht_fused_batched expects the per-expert row count.
                encoder.set_buffer(2, Some(&work.row_count), 0);
                encoder.set_threadgroup_memory_length(0, (work.rows * 4) as u64);
                encoder.dispatch_thread_groups(
                    MTLSize {
                        width: work.experts as u64,
                        height: 1,
                        depth: 1,
                    },
                    MTLSize {
                        width: 1024,
                        height: 1,
                        depth: 1,
                    },
                );
                encoder.set_compute_pipeline_state(if inputs_are_batched {
                    &self.pipelines.low_rank_inputs
                } else {
                    &self.pipelines.low_rank_shared
                });
                encoder.set_buffer(0, Some(&bundle_buffer), 0);
                encoder.set_buffer(1, Some(projection_input), 0);
                encoder.set_buffer(2, Some(&work.rank_output), 0);
                encoder.set_buffer(3, Some(&work.shape), 0);
                encoder.set_buffer(4, Some(&work.offsets), 0);
                encoder.dispatch_threads(
                    MTLSize {
                        width: 1,
                        height: work.experts as u64,
                        depth: 1,
                    },
                    MTLSize {
                        width: 1,
                        height: 1,
                        depth: 1,
                    },
                );
                encoder.set_compute_pipeline_state(&self.pipelines.finish);
                encoder.set_buffer(0, Some(&bundle_buffer), 0);
                encoder.set_buffer(1, Some(&work.transformed_output), 0);
                encoder.set_buffer(2, Some(&work.rank_output), 0);
                encoder.set_buffer(3, Some(&work.final_output), 0);
                encoder.set_buffer(4, Some(&work.shape), 0);
                encoder.set_buffer(5, Some(&work.offsets), 0);
                encoder.dispatch_threads(
                    MTLSize {
                        width: work.rows as u64,
                        height: work.experts as u64,
                        depth: 1,
                    },
                    MTLSize {
                        width: 256,
                        height: 1,
                        depth: 1,
                    },
                );
                // Gate/up are rounded by BF16-staged SwiGLU and down is rounded by
                // route reduction. Those consumer-side operations exactly repeat
                // this idempotent pass, so keep only the authoritative consumer.
            };
            encode_projection(&gate, &dynamic_input, false);
            encode_projection(&up, &dynamic_input, false);
            encoder.set_compute_pipeline_state(&self.pipelines.swiglu);
            encoder.set_buffer(0, Some(&gate.final_output), 0);
            encoder.set_buffer(1, Some(&up.final_output), 0);
            encoder.set_buffer(2, Some(&hidden), 0);
            encoder.set_buffer(3, Some(&hidden_count_buffer), 0);
            encoder.set_buffer(4, Some(&error_buffer), 0);
            encoder.dispatch_threads(
                MTLSize {
                    width: hidden_count as u64,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: 256,
                    height: 1,
                    depth: 1,
                },
            );
            encoder.set_compute_pipeline_state(&self.pipelines.dynamic);
            encoder.set_buffer(0, Some(&hidden), 0);
            encoder.set_buffer(1, Some(&dynamic_hidden), 0);
            encoder.set_buffer(2, Some(&self.decode_lut), 0);
            encoder.set_buffer(3, Some(&error_buffer), 0);
            encoder.set_buffer(4, Some(&self.boundaries), 0);
            encoder.set_threadgroup_memory_length(0, 128 * 4);
            encoder.dispatch_thread_groups(
                MTLSize {
                    width: (hidden_count / 128) as u64,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: 128,
                    height: 1,
                    depth: 1,
                },
            );
            encode_projection(&down, &dynamic_hidden, true);

            let encode_source_projection =
                |offsets: &metal::BufferRef,
                 source_input: &metal::BufferRef,
                 source_output: &metal::BufferRef,
                 shape: &metal::BufferRef,
                 rows: usize| {
                    encoder.set_compute_pipeline_state(&self.pipelines.source_projection_batched);
                    encoder.set_buffer(0, Some(&bundle_buffer), 0);
                    encoder.set_buffer(1, Some(source_input), 0);
                    encoder.set_buffer(2, Some(source_output), 0);
                    encoder.set_buffer(3, Some(shape), 0);
                    encoder.set_buffer(4, Some(&self.decode_lut), 0);
                    encoder.set_buffer(5, Some(offsets), 0);
                    encoder.set_buffer(6, Some(&source_panel.expert_count), 0);
                    encoder.set_threadgroup_memory_length(0, 32 * 4);
                    encoder.dispatch_thread_groups(
                        MTLSize {
                            width: (rows * source_panel.experts.len()) as u64,
                            height: 1,
                            depth: 1,
                        },
                        MTLSize {
                            width: 32,
                            height: 1,
                            depth: 1,
                        },
                    );
                };
            encoder.set_compute_pipeline_state(&self.pipelines.source_gate_up);
            encoder.set_buffer(0, Some(&bundle_buffer), 0);
            encoder.set_buffer(1, Some(&dynamic_input), 0);
            encoder.set_buffer(2, Some(&source_panel.gate), 0);
            encoder.set_buffer(3, Some(&source_panel.up), 0);
            encoder.set_buffer(4, Some(&source_panel.gate_shape), 0);
            encoder.set_buffer(5, Some(&self.decode_lut), 0);
            encoder.set_buffer(6, Some(&source_panel.gate_offsets), 0);
            encoder.set_buffer(7, Some(&source_panel.up_offsets), 0);
            encoder.set_buffer(8, Some(&source_panel.expert_count), 0);
            encoder.dispatch_thread_groups(
                MTLSize {
                    width: (INTERMEDIATE * source_panel.experts.len()) as u64,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: 32,
                    height: 1,
                    depth: 1,
                },
            );
            // bf16_staged_swiglu performs the same idempotent BF16 rounding on
            // gate and up before consuming them, so separate panel-wide passes
            // would only add command-dispatch cost.
            encoder.set_compute_pipeline_state(&self.pipelines.swiglu);
            encoder.set_buffer(0, Some(&source_panel.gate), 0);
            encoder.set_buffer(1, Some(&source_panel.up), 0);
            encoder.set_buffer(2, Some(&source_panel.hidden), 0);
            encoder.set_buffer(3, Some(&source_panel.hidden_count), 0);
            encoder.set_buffer(4, Some(&error_buffer), 0);
            encoder.dispatch_threads(
                MTLSize {
                    width: (source_panel.experts.len() * INTERMEDIATE) as u64,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: 256,
                    height: 1,
                    depth: 1,
                },
            );
            encoder.set_compute_pipeline_state(&self.pipelines.dynamic);
            encoder.set_buffer(0, Some(&source_panel.hidden), 0);
            encoder.set_buffer(1, Some(&source_panel.dynamic_hidden), 0);
            encoder.set_buffer(2, Some(&self.decode_lut), 0);
            encoder.set_buffer(3, Some(&error_buffer), 0);
            encoder.set_buffer(4, Some(&self.boundaries), 0);
            encoder.set_threadgroup_memory_length(0, 128 * 4);
            encoder.dispatch_thread_groups(
                MTLSize {
                    width: (source_panel.experts.len() * INTERMEDIATE / 128) as u64,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: 128,
                    height: 1,
                    depth: 1,
                },
            );
            encode_source_projection(
                &source_panel.down_offsets,
                &source_panel.dynamic_hidden,
                &source_panel.output,
                &source_panel.down_shape,
                HIDDEN,
            );
            // The route reduction rounds every selected source output before
            // accumulation; an earlier in-place round is therefore redundant.
            if source_panel.experts != bundle.source_experts {
                return Err("source execution order drift".to_owned());
            }
            encoder.set_buffer(0, Some(&down.final_output), 0);
            encoder.set_compute_pipeline_state(&self.pipelines.reduce_source_panel);
            encoder.set_buffer(1, Some(&source_panel.output), 0);
            encoder.set_buffer(2, Some(&selected_buffer), 0);
            encoder.set_buffer(3, Some(&weight_buffer), 0);
            encoder.set_buffer(4, Some(&k4_id_buffer), 0);
            encoder.set_buffer(5, Some(&source_id_buffer), 0);
            encoder.set_buffer(6, Some(&k4_count_buffer), 0);
            encoder.set_buffer(7, Some(&source_panel.expert_count), 0);
            encoder.set_buffer(8, Some(&output), 0);
            encoder.set_buffer(9, Some(&error_buffer), 0);
            encoder.dispatch_threads(
                MTLSize {
                    width: HIDDEN as u64,
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
        encoder.end_encoding();
        command.commit();
        command.wait_until_completed();
        let wall_ms = started.elapsed().as_secs_f64() * 1000.0;
        let gpu_start_seconds: f64 = unsafe { msg_send![command, GPUStartTime] };
        let gpu_end_seconds: f64 = unsafe { msg_send![command, GPUEndTime] };
        let gpu_ms = (gpu_end_seconds - gpu_start_seconds) * 1000.0;
        if !gpu_ms.is_finite() || gpu_ms <= 0.0 {
            return Err(format!("invalid K4/source GPU duration {gpu_ms}"));
        }
        let flags = unsafe { *error_buffer.contents().cast::<u32>() };
        if command.status() != MTLCommandBufferStatus::Completed || flags != 0 {
            return Err(format!(
                "K4/source Metal transaction failed: status={:?}, flags={flags}",
                command.status()
            ));
        }
        let values =
            unsafe { std::slice::from_raw_parts(output.contents().cast::<f32>(), HIDDEN).to_vec() };
        if values.iter().any(|value| !value.is_finite()) {
            return Err("K4/source output is nonfinite".to_owned());
        }
        Ok(K4SourceExecution {
            output: values,
            wall_ms,
            gpu_ms,
        })
    }
}

#[derive(Deserialize)]
struct LayerFixture {
    semantic: String,
    layer: usize,
    input_f32: Vec<f32>,
    native_router_experts: Vec<u32>,
    native_router_weights: Vec<f32>,
    candidate_routed_f32: Vec<f32>,
    #[serde(default)]
    decode_candidate_routed_f32: Option<Vec<f32>>,
}

#[derive(Debug, Serialize)]
pub struct TimingDistribution {
    pub minimum_ms: f64,
    pub median_ms: f64,
    pub p90_ms: f64,
    pub maximum_ms: f64,
}

fn timing_distribution(values: &[f64]) -> TimingDistribution {
    let mut ordered = values.to_vec();
    ordered.sort_by(f64::total_cmp);
    let percentile = |fraction: f64| {
        let index = ((ordered.len() as f64 * fraction).ceil() as usize)
            .saturating_sub(1)
            .min(ordered.len() - 1);
        ordered[index]
    };
    TimingDistribution {
        minimum_ms: ordered[0],
        median_ms: percentile(0.5),
        p90_ms: percentile(0.9),
        maximum_ms: *ordered.last().expect("nonempty timing distribution"),
    }
}

#[derive(Debug, Serialize)]
pub struct K4SourceMetalLayerReport {
    pub schema_version: u32,
    pub semantic: &'static str,
    pub commit: String,
    pub bundle_sha256: String,
    pub bundle_bytes: u64,
    pub layer: usize,
    pub device: String,
    pub compile_ms: f64,
    pub first_command_wall_ms: f64,
    pub first_gpu_ms: f64,
    pub warmups: usize,
    pub samples: usize,
    pub command_wall: TimingDistribution,
    pub gpu_time: TimingDistribution,
    pub complete_call_wall: TimingDistribution,
    pub relative_l2: f64,
    pub maximum_absolute_error: f32,
    pub unequal_count: usize,
    pub expert_ids: Vec<u32>,
    pub route_candidate_relative_l2: f64,
    pub exactness_class: &'static str,
    pub expert_identity_preserved: bool,
    pub source_function_preserved: bool,
    pub identity_substitution_allowed: bool,
    pub candidate_route_gate_pass: bool,
    pub safety_snapshots: Vec<SafetySnapshot>,
    pub status: &'static str,
}

pub fn run_k4_source_metal_layer_fixture(
    bundle_path: &Path,
    manifest_path: &Path,
    fixture_path: &Path,
    kernel_root: &Path,
    output_path: &Path,
    commit: &str,
) -> Result<K4SourceMetalLayerReport, String> {
    if output_path.exists() {
        return Err(format!("refusing to overwrite {}", output_path.display()));
    }
    if commit.len() != 40
        || !commit
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
    {
        return Err("implementation commit must be lowercase 40-hex".to_owned());
    }
    let mut safety = ComponentSafetyMonitor::start_normative()?;
    let fixture_bytes =
        fs::read(fixture_path).map_err(|error| format!("{}: {error}", fixture_path.display()))?;
    let fixture: LayerFixture = serde_json::from_slice(&fixture_bytes)
        .map_err(|error| format!("{}: {error}", fixture_path.display()))?;
    let bundle = K4SourceLayerBundle::open(bundle_path, manifest_path)?;
    if !matches!(
        fixture.semantic.as_str(),
        "pw0361_layer28_native_weight_three_source_fixture"
            | "pw0416_layer28_qualified_native_weight_k4_source_fixture"
            | "pw0424_layer28_three_k4_five_source_native_fixture"
            | "pw0316_layer4_four_k4_four_source_fixture"
            | "pw0317_layer4_three_k4_five_source_fixture"
    ) || fixture.layer != bundle.layer
        || fixture.input_f32.len() != HIDDEN
        || fixture.candidate_routed_f32.len() != HIDDEN
        || fixture
            .decode_candidate_routed_f32
            .as_ref()
            .is_some_and(|values| values.len() != HIDDEN)
    {
        return Err("K4/source layer fixture identity mismatch".to_owned());
    }
    let runtime = K4SourceMetalRuntime::compile(kernel_root)?;
    safety.checkpoint("k4_source_bundle_and_runtime_ready")?;
    let execution = runtime.execute(
        &bundle,
        &fixture.input_f32,
        &fixture.native_router_experts,
        &fixture.native_router_weights,
    )?;
    let mut error_squared = 0.0_f64;
    let mut reference_squared = 0.0_f64;
    let mut maximum_absolute_error = 0.0_f32;
    let mut unequal_count = 0;
    let mut mismatch_examples = Vec::new();
    for (index, (&actual, &expected)) in execution
        .output
        .iter()
        .zip(&fixture.candidate_routed_f32)
        .enumerate()
    {
        let error = actual - expected;
        error_squared += f64::from(error) * f64::from(error);
        reference_squared += f64::from(expected) * f64::from(expected);
        maximum_absolute_error = maximum_absolute_error.max(error.abs());
        unequal_count += usize::from(actual.to_bits() != expected.to_bits());
        if actual.to_bits() != expected.to_bits() && mismatch_examples.len() < 8 {
            mismatch_examples.push((
                index,
                actual.to_bits(),
                expected.to_bits(),
                actual,
                expected,
            ));
        }
    }
    let relative_l2 = error_squared.sqrt() / reference_squared.sqrt().max(1.0e-30);
    if unequal_count != 0 || relative_l2 != 0.0 || maximum_absolute_error != 0.0 {
        let decode_candidate_bitexact = fixture
            .decode_candidate_routed_f32
            .as_ref()
            .is_some_and(|expected| f32_slices_bit_equal(&execution.output, expected));
        let bundle_sha256 = bundle.bundle_sha256.clone();
        let bundle_bytes = bundle.bundle_bytes;
        let layer = bundle.layer;
        let fixture_semantic = fixture.semantic.clone();
        drop(execution);
        drop(runtime);
        drop(bundle);
        drop(fixture);
        drop(fixture_bytes);
        let (safety_snapshots, release_result) = safety.released_preserving();
        let buffer_release_error = release_result.err();
        let failure = serde_json::json!({
            "schema_version": 1,
            "semantic": "prismwing_k4_source_bundle_native_metal_layer_rejection",
            "status": "rejected_batch_fixture_bit_mismatch",
            "commit": commit,
            "bundle_sha256": bundle_sha256,
            "bundle_bytes": bundle_bytes,
            "layer": layer,
            "fixture_semantic": fixture_semantic,
            "relative_l2": relative_l2,
            "maximum_absolute_error": maximum_absolute_error,
            "unequal_count": unequal_count,
            "mismatch_examples": mismatch_examples,
            "decode_candidate_bitexact": decode_candidate_bitexact,
            "buffer_release_error": buffer_release_error,
            "safety_snapshots": safety_snapshots,
            "batch_size": 1,
            "concurrency": 1,
            "accepted_tokens": 0,
            "performance_claim": null,
            "decision": "reject_batch_derived_layer_fixture",
        });
        write_create_new(
            output_path,
            &serde_json::to_vec_pretty(&failure).map_err(|error| error.to_string())?,
        )?;
        return Err(match failure["buffer_release_error"].as_str() {
            Some(release_error) => format!(
                "K4/source Metal bit mismatch: unequal={unequal_count}, relative={relative_l2}, max={maximum_absolute_error}; buffer release also failed: {release_error}"
            ),
            None => format!(
                "K4/source Metal bit mismatch: unequal={unequal_count}, relative={relative_l2}, max={maximum_absolute_error}, examples={mismatch_examples:?}"
            ),
        });
    }
    safety.checkpoint("k4_source_exact_parity")?;
    const WARMUPS: usize = 20;
    const SAMPLES: usize = 100;
    for _ in 0..WARMUPS {
        let warm = runtime.execute(
            &bundle,
            &fixture.input_f32,
            &fixture.native_router_experts,
            &fixture.native_router_weights,
        )?;
        if warm.output != execution.output {
            return Err("K4/source warm execution is not deterministic".to_owned());
        }
    }
    safety.checkpoint("k4_source_warmups_complete")?;
    let mut command_wall = Vec::with_capacity(SAMPLES);
    let mut gpu_time = Vec::with_capacity(SAMPLES);
    let mut complete_call_wall = Vec::with_capacity(SAMPLES);
    for _ in 0..SAMPLES {
        let call_started = Instant::now();
        let sample = runtime.execute(
            &bundle,
            &fixture.input_f32,
            &fixture.native_router_experts,
            &fixture.native_router_weights,
        )?;
        complete_call_wall.push(call_started.elapsed().as_secs_f64() * 1000.0);
        command_wall.push(sample.wall_ms);
        gpu_time.push(sample.gpu_ms);
        if sample.output != execution.output {
            return Err("K4/source sampled execution is not deterministic".to_owned());
        }
    }
    safety.checkpoint("k4_source_timed_series_complete")?;
    let command_wall = timing_distribution(&command_wall);
    let gpu_time = timing_distribution(&gpu_time);
    let complete_call_wall = timing_distribution(&complete_call_wall);
    let bundle_sha256 = bundle.bundle_sha256.clone();
    let bundle_bytes = bundle.bundle_bytes;
    let layer = bundle.layer;
    let route_candidate_relative_l2 = bundle.route_candidate_relative_l2;
    let candidate_route_gate_pass = bundle.candidate_route_gate_pass;
    let device = runtime.device_name.clone();
    let compile_ms = runtime.compile_ms;
    let expert_ids = fixture.native_router_experts.clone();
    let first_command_wall_ms = execution.wall_ms;
    let first_gpu_ms = execution.gpu_ms;
    drop(execution);
    drop(runtime);
    drop(bundle);
    drop(fixture);
    drop(fixture_bytes);
    let safety_snapshots = safety.released()?;
    let report = K4SourceMetalLayerReport {
        schema_version: 2,
        semantic: "prismwing_k4_source_bundle_native_metal_layer",
        commit: commit.to_owned(),
        bundle_sha256,
        bundle_bytes,
        layer,
        device,
        compile_ms,
        first_command_wall_ms,
        first_gpu_ms,
        warmups: WARMUPS,
        samples: SAMPLES,
        command_wall,
        gpu_time,
        complete_call_wall,
        relative_l2,
        maximum_absolute_error,
        unequal_count,
        expert_ids,
        route_candidate_relative_l2,
        exactness_class: K4_EXACTNESS_CLASS,
        expert_identity_preserved: true,
        source_function_preserved: false,
        identity_substitution_allowed: false,
        candidate_route_gate_pass,
        safety_snapshots,
        status: if candidate_route_gate_pass {
            "modified_k4_source_layer_candidate_fixture_bitexact"
        } else {
            "modified_k4_source_layer_fixture_bitexact_without_route_gate"
        },
    };
    write_create_new(
        output_path,
        &serde_json::to_vec_pretty(&report).map_err(|error| error.to_string())?,
    )?;
    Ok(report)
}

#[derive(Debug, Serialize)]
pub struct K4SourceMetalRepeatedReport {
    pub schema_version: u32,
    pub semantic: &'static str,
    pub status: &'static str,
    pub decision: &'static str,
    pub implementation_commit: String,
    pub source_experiment: &'static str,
    pub authority_sha256: BTreeMap<String, String>,
    pub authority_observation_errors: BTreeMap<String, String>,
    pub bundle_sha256: String,
    pub bundle_bytes: u64,
    pub device: String,
    pub host_model: String,
    pub host_total_memory_bytes: u64,
    pub target_host_model: &'static str,
    pub target_metal_device: &'static str,
    pub target_host_memory_bytes: u64,
    pub target_host_model_match: bool,
    pub target_host_memory_match: bool,
    pub target_host_device_match: bool,
    pub target_host_match: bool,
    pub target_hypothesis_evaluated: bool,
    pub modified_endpoint_overlay_slice_authorized: bool,
    pub compile_ms: f64,
    pub repeated_layers: usize,
    pub warmups: usize,
    pub samples: usize,
    pub deterministic_output_comparisons: usize,
    pub transaction_wall: TimingDistribution,
    pub gpu_time: TimingDistribution,
    pub complete_call_wall: TimingDistribution,
    pub raw_routed_complete_call_p90_ms: f64,
    pub raw_routed_two_tps_budget_ms: f64,
    pub observed_host_raw_routed_two_tps_budget_pass: bool,
    pub target_raw_routed_two_tps_necessary_condition: Option<bool>,
    pub raw_routed_three_tps_budget_ms: f64,
    pub observed_host_raw_routed_three_tps_budget_pass: bool,
    pub target_raw_routed_three_tps_diagnostic: Option<bool>,
    pub fixed_nonexpert_ms: f64,
    pub fixed_nonexpert_ledger_device: &'static str,
    pub fixed_nonexpert_ledger_context_only: bool,
    pub legacy_m4_context_component_wall_p90_ms: Option<f64>,
    pub legacy_m4_context_component_wall_tps: Option<f64>,
    pub legacy_m4_context_component_gpu_p90_ms: Option<f64>,
    pub legacy_m4_context_component_gpu_tps: Option<f64>,
    pub relative_l2: f64,
    pub maximum_absolute_error: f32,
    pub unequal_count: usize,
    pub exactness_class: &'static str,
    pub expert_identity_preserved: bool,
    pub source_function_preserved: bool,
    pub identity_substitution_allowed: bool,
    pub candidate_route_gate_pass: bool,
    pub safety_snapshots: Vec<SafetySnapshot>,
    pub performance_claim: Option<String>,
}

#[derive(Debug, Serialize)]
struct K4SourceMetalRepeatedFailureReport {
    schema_version: u32,
    semantic: &'static str,
    status: &'static str,
    decision: &'static str,
    implementation_commit: String,
    source_experiment: &'static str,
    error: String,
    buffer_release_error: Option<String>,
    authority_sha256: BTreeMap<String, String>,
    authority_observation_errors: BTreeMap<String, String>,
    device: Option<String>,
    host_model: String,
    host_total_memory_bytes: u64,
    target_host_model: &'static str,
    target_metal_device: &'static str,
    target_host_memory_bytes: u64,
    target_host_model_match: bool,
    target_host_device_match: bool,
    target_host_memory_match: bool,
    target_host_match: bool,
    safety_snapshots: Vec<SafetySnapshot>,
    performance_claim: Option<String>,
}

struct K4SourceRepeatedMeasurements {
    bundle_sha256: String,
    bundle_bytes: u64,
    device: String,
    compile_ms: f64,
    transaction_wall: TimingDistribution,
    gpu_time: TimingDistribution,
    complete_call_wall: TimingDistribution,
    relative_l2: f64,
    maximum_absolute_error: f32,
    unequal_count: usize,
    candidate_route_gate_pass: bool,
}

#[allow(clippy::too_many_arguments)]
fn write_repeated_failure(
    output_path: &Path,
    implementation_commit: &str,
    error: String,
    buffer_release_error: Option<String>,
    authority_sha256: BTreeMap<String, String>,
    authority_observation_errors: BTreeMap<String, String>,
    device: Option<String>,
    host_model: &str,
    host_total_memory_bytes: u64,
    safety_snapshots: Vec<SafetySnapshot>,
) -> Result<(), String> {
    let target_host_model_match = host_model == TARGET_HOST_MODEL;
    let target_host_device_match = device.as_deref() == Some(TARGET_METAL_DEVICE);
    let target_host_memory_match = host_total_memory_bytes == TARGET_HOST_MEMORY_BYTES;
    let report = K4SourceMetalRepeatedFailureReport {
        schema_version: 3,
        semantic: "prismwing_modified_k4_source_bundle_native_metal_47_repeat_failure",
        status: "failed_evidence_preserved",
        decision: "reject_failed_run",
        implementation_commit: implementation_commit.to_owned(),
        source_experiment: SOURCE_EXPERIMENT,
        error,
        buffer_release_error,
        authority_sha256,
        authority_observation_errors,
        device,
        host_model: host_model.to_owned(),
        host_total_memory_bytes,
        target_host_model: TARGET_HOST_MODEL,
        target_metal_device: TARGET_METAL_DEVICE,
        target_host_memory_bytes: TARGET_HOST_MEMORY_BYTES,
        target_host_model_match,
        target_host_device_match,
        target_host_memory_match,
        target_host_match: target_host_model_match
            && target_host_device_match
            && target_host_memory_match,
        safety_snapshots,
        performance_claim: None,
    };
    write_create_new(
        output_path,
        &serde_json::to_vec_pretty(&report).map_err(|error| error.to_string())?,
    )
}

pub fn run_k4_source_metal_repeated_fixture(
    bundle_path: &Path,
    manifest_path: &Path,
    fixture_path: &Path,
    kernel_root: &Path,
    output_path: &Path,
    implementation_commit: &str,
) -> Result<K4SourceMetalRepeatedReport, String> {
    if output_path.exists() {
        return Err(format!("refusing to overwrite {}", output_path.display()));
    }
    if implementation_commit.len() != 40
        || !implementation_commit
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
    {
        return Err("implementation commit must be lowercase 40-hex".to_owned());
    }
    let host_total_memory_bytes = host_total_memory_bytes()?;
    let host_model = host_model()?;
    let mut authority_sha256 = BTreeMap::new();
    let mut authority_observation_errors = BTreeMap::new();
    let mut observed_device = Device::system_default().map(|device| device.name().to_owned());
    let mut safety = match ComponentSafetyMonitor::start_normative() {
        Ok(safety) => safety,
        Err(error) => {
            write_repeated_failure(
                output_path,
                implementation_commit,
                error.clone(),
                None,
                authority_sha256,
                authority_observation_errors,
                observed_device,
                &host_model,
                host_total_memory_bytes,
                Vec::new(),
            )?;
            return Err(error);
        }
    };
    const LAYERS: usize = 47;
    const WARMUPS: usize = 20;
    const SAMPLES: usize = 100;
    const FIXED_NONEXPERT_MS: f64 = 132.370291661 + 6.521666655;
    let attempt = (|| -> Result<K4SourceRepeatedMeasurements, String> {
        let fixture_bytes = match fs::read(fixture_path) {
            Ok(bytes) => {
                authority_sha256.insert("route_fixture".to_owned(), sha256_hex(&bytes));
                Some(bytes)
            }
            Err(error) => {
                authority_observation_errors.insert(
                    "route_fixture".to_owned(),
                    format!("{}: {error}", fixture_path.display()),
                );
                None
            }
        };
        let bundle_preflight_sha256 = match file_sha256(bundle_path) {
            Ok(sha256) => {
                authority_sha256.insert("bundle".to_owned(), sha256.clone());
                Some(sha256)
            }
            Err(error) => {
                authority_observation_errors.insert("bundle".to_owned(), error);
                None
            }
        };
        let manifest_preflight_sha256 = match fs::read(manifest_path) {
            Ok(bytes) => {
                let sha256 = sha256_hex(&bytes);
                authority_sha256.insert("bundle_manifest".to_owned(), sha256.clone());
                Some(sha256)
            }
            Err(error) => {
                authority_observation_errors.insert(
                    "bundle_manifest".to_owned(),
                    format!("{}: {error}", manifest_path.display()),
                );
                None
            }
        };
        for &(name, _) in &PW0478_KERNEL_SHA256 {
            let source_path = kernel_root.join(name);
            let label = format!("kernel/{name}");
            match fs::read(&source_path) {
                Ok(source_bytes) => {
                    authority_sha256.insert(label, sha256_hex(&source_bytes));
                }
                Err(error) => {
                    authority_observation_errors
                        .insert(label, format!("{}: {error}", source_path.display()));
                }
            }
        }
        if !authority_observation_errors.is_empty() {
            return Err(format!(
                "PW-0478 runtime authority observation failed: {}",
                authority_observation_errors
                    .iter()
                    .map(|(label, error)| format!("{label}: {error}"))
                    .collect::<Vec<_>>()
                    .join("; ")
            ));
        }
        let fixture_bytes = fixture_bytes.expect("authority observation covered route fixture");
        let fixture_sha256 = authority_sha256
            .get("route_fixture")
            .expect("authority observation hashed route fixture")
            .clone();
        let bundle_preflight_sha256 =
            bundle_preflight_sha256.expect("authority observation hashed bundle");
        let manifest_preflight_sha256 =
            manifest_preflight_sha256.expect("authority observation hashed bundle manifest");
        let mut authority_mismatches = Vec::new();
        for (label, observed, expected) in [
            ("route fixture", &fixture_sha256, PW0478_FIXTURE_SHA256),
            ("bundle", &bundle_preflight_sha256, PW0478_BUNDLE_SHA256),
            (
                "bundle manifest",
                &manifest_preflight_sha256,
                PW0478_BUNDLE_MANIFEST_SHA256,
            ),
        ] {
            if observed != expected {
                authority_mismatches
                    .push(format!("{label}: observed={observed}, expected={expected}"));
            }
        }
        for &(name, expected) in &PW0478_KERNEL_SHA256 {
            let observed = authority_sha256
                .get(&format!("kernel/{name}"))
                .expect("preflight inserted every kernel hash");
            if observed != expected {
                authority_mismatches.push(format!(
                    "kernel {name}: observed={observed}, expected={expected}"
                ));
            }
        }
        if !authority_mismatches.is_empty() {
            return Err(format!(
                "PW-0478 runtime authority mismatch: {}",
                authority_mismatches.join("; ")
            ));
        }
        let fixture: LayerFixture = serde_json::from_slice(&fixture_bytes)
            .map_err(|error| format!("{}: {error}", fixture_path.display()))?;
        if fixture.semantic != "pw0424_layer28_three_k4_five_source_native_fixture"
            || fixture.layer != 28
            || fixture.input_f32.len() != HIDDEN
            || fixture.candidate_routed_f32.len() != HIDDEN
        {
            return Err("PW-0478 route fixture identity mismatch".to_owned());
        }

        let bundle = K4SourceLayerBundle::open(bundle_path, manifest_path)?;
        authority_sha256.insert("bundle".to_owned(), bundle.bundle_sha256.clone());
        authority_sha256.insert("bundle_manifest".to_owned(), bundle.manifest_sha256.clone());
        if bundle.bundle_sha256 != PW0478_BUNDLE_SHA256
            || bundle.manifest_sha256 != PW0478_BUNDLE_MANIFEST_SHA256
        {
            return Err(format!(
                "PW-0478 bundle authority mismatch: bundle={}, manifest={}",
                bundle.bundle_sha256, bundle.manifest_sha256
            ));
        }

        let runtime = K4SourceMetalRuntime::compile(kernel_root)?;
        observed_device = Some(runtime.device_name.clone());
        for &(name, expected) in &PW0478_KERNEL_SHA256 {
            let observed = runtime
                .kernel_sha256
                .get(name)
                .ok_or_else(|| format!("PW-0478 compiled kernel {name} is missing"))?;
            authority_sha256.insert(format!("kernel/{name}"), observed.clone());
            if observed != expected {
                return Err(format!(
                    "PW-0478 kernel {name} SHA-256 mismatch: observed={observed}, expected={expected}"
                ));
            }
        }
        safety.checkpoint("pw0478_bundle_and_runtime_ready")?;

        let execution = runtime.execute_repeated(
            &bundle,
            &fixture.input_f32,
            &fixture.native_router_experts,
            &fixture.native_router_weights,
            LAYERS,
        )?;
        let mut error_squared = 0.0_f64;
        let mut reference_squared = 0.0_f64;
        let mut maximum_absolute_error = 0.0_f32;
        let mut unequal_count = 0;
        for (&actual, &expected) in execution.output.iter().zip(&fixture.candidate_routed_f32) {
            let error = actual - expected;
            error_squared += f64::from(error) * f64::from(error);
            reference_squared += f64::from(expected) * f64::from(expected);
            maximum_absolute_error = maximum_absolute_error.max(error.abs());
            unequal_count += usize::from(actual.to_bits() != expected.to_bits());
        }
        let relative_l2 = error_squared.sqrt() / reference_squared.sqrt().max(1.0e-30);
        if unequal_count != 0 || relative_l2 != 0.0 {
            return Err(format!(
                "repeated K4/source transaction lost exact parity: unequal={unequal_count}, relative={relative_l2}"
            ));
        }
        safety.checkpoint("pw0478_exact_parity")?;

        for _ in 0..WARMUPS {
            let warm = runtime.execute_repeated(
                &bundle,
                &fixture.input_f32,
                &fixture.native_router_experts,
                &fixture.native_router_weights,
                LAYERS,
            )?;
            if !f32_slices_bit_equal(&warm.output, &execution.output) {
                return Err("repeated K4/source warm execution is not bit-deterministic".to_owned());
            }
        }
        safety.checkpoint("pw0478_warmups_complete")?;

        let mut transaction_wall = Vec::with_capacity(SAMPLES);
        let mut gpu_time = Vec::with_capacity(SAMPLES);
        let mut complete_call_wall = Vec::with_capacity(SAMPLES);
        for _ in 0..SAMPLES {
            let call_started = Instant::now();
            let sample = runtime.execute_repeated(
                &bundle,
                &fixture.input_f32,
                &fixture.native_router_experts,
                &fixture.native_router_weights,
                LAYERS,
            )?;
            complete_call_wall.push(call_started.elapsed().as_secs_f64() * 1000.0);
            transaction_wall.push(sample.wall_ms);
            gpu_time.push(sample.gpu_ms);
            if !f32_slices_bit_equal(&sample.output, &execution.output) {
                return Err(
                    "repeated K4/source sampled execution is not bit-deterministic".to_owned(),
                );
            }
        }
        safety.checkpoint("pw0478_timed_series_complete")?;
        let transaction_wall = timing_distribution(&transaction_wall);
        let gpu_time = timing_distribution(&gpu_time);
        let complete_call_wall = timing_distribution(&complete_call_wall);
        let bundle_sha256 = bundle.bundle_sha256.clone();
        let bundle_bytes = bundle.bundle_bytes;
        let candidate_route_gate_pass = bundle.candidate_route_gate_pass;
        let device = runtime.device_name.clone();
        let compile_ms = runtime.compile_ms;
        drop(execution);
        drop(runtime);
        drop(bundle);
        drop(fixture);
        drop(fixture_bytes);
        Ok(K4SourceRepeatedMeasurements {
            bundle_sha256,
            bundle_bytes,
            device,
            compile_ms,
            transaction_wall,
            gpu_time,
            complete_call_wall,
            relative_l2,
            maximum_absolute_error,
            unequal_count,
            candidate_route_gate_pass,
        })
    })();

    let measurements = match attempt {
        Ok(measurements) => measurements,
        Err(error) => {
            let (safety_snapshots, release_result) = safety.released_preserving();
            let buffer_release_error = release_result.err();
            write_repeated_failure(
                output_path,
                implementation_commit,
                error.clone(),
                buffer_release_error.clone(),
                authority_sha256,
                authority_observation_errors,
                observed_device,
                &host_model,
                host_total_memory_bytes,
                safety_snapshots,
            )?;
            return Err(match buffer_release_error {
                Some(release_error) => {
                    format!("{error}; buffer release safety check also failed: {release_error}")
                }
                None => error,
            });
        }
    };

    let (safety_snapshots, release_result) = safety.released_preserving();
    if let Err(error) = release_result {
        write_repeated_failure(
            output_path,
            implementation_commit,
            error.clone(),
            None,
            authority_sha256,
            authority_observation_errors,
            Some(measurements.device),
            &host_model,
            host_total_memory_bytes,
            safety_snapshots,
        )?;
        return Err(error);
    }

    let raw_routed_complete_call_p90_ms = measurements.complete_call_wall.p90_ms;
    let observed_host_raw_routed_two_tps_budget_pass = raw_routed_budget_pass(
        raw_routed_complete_call_p90_ms,
        RAW_ROUTED_TWO_TPS_BUDGET_MS,
    );
    let observed_host_raw_routed_three_tps_budget_pass = raw_routed_budget_pass(
        raw_routed_complete_call_p90_ms,
        RAW_ROUTED_THREE_TPS_BUDGET_MS,
    );
    let target_host_model_match = host_model == TARGET_HOST_MODEL;
    let target_host_device_match = measurements.device == TARGET_METAL_DEVICE;
    let target_host_memory_match = host_total_memory_bytes == TARGET_HOST_MEMORY_BYTES;
    let target_host_match =
        target_host_model_match && target_host_device_match && target_host_memory_match;
    let (status, decision) = if !target_host_match {
        (
            "non_target_host_implementation_preflight_only",
            "keep_exact_16_gib_apple_m1_run_pending",
        )
    } else if observed_host_raw_routed_two_tps_budget_pass {
        (
            "m1_16g_modified_component_two_tps_necessary_condition_pass",
            "authorize_modified_layer28_endpoint_overlay_causal_slice",
        )
    } else {
        (
            "m1_16g_modified_component_two_tps_necessary_condition_fail",
            "reject_modified_roughly_two_tps_fallback_compute_condition",
        )
    };
    let legacy_m4_component_wall_p90_ms = (measurements.device == "Apple M4 Pro")
        .then_some(measurements.transaction_wall.p90_ms + FIXED_NONEXPERT_MS);
    let legacy_m4_component_gpu_p90_ms = (measurements.device == "Apple M4 Pro")
        .then_some(measurements.gpu_time.p90_ms + FIXED_NONEXPERT_MS);
    let report = K4SourceMetalRepeatedReport {
        schema_version: 3,
        semantic: "prismwing_modified_k4_source_bundle_native_metal_47_repeat_component",
        status,
        decision,
        implementation_commit: implementation_commit.to_owned(),
        source_experiment: SOURCE_EXPERIMENT,
        authority_sha256,
        authority_observation_errors,
        bundle_sha256: measurements.bundle_sha256,
        bundle_bytes: measurements.bundle_bytes,
        device: measurements.device,
        host_model,
        host_total_memory_bytes,
        target_host_model: TARGET_HOST_MODEL,
        target_metal_device: TARGET_METAL_DEVICE,
        target_host_memory_bytes: TARGET_HOST_MEMORY_BYTES,
        target_host_model_match,
        target_host_memory_match,
        target_host_device_match,
        target_host_match,
        target_hypothesis_evaluated: target_host_match,
        modified_endpoint_overlay_slice_authorized: target_host_match
            && observed_host_raw_routed_two_tps_budget_pass,
        compile_ms: measurements.compile_ms,
        repeated_layers: LAYERS,
        warmups: WARMUPS,
        samples: SAMPLES,
        deterministic_output_comparisons: WARMUPS + SAMPLES,
        transaction_wall: measurements.transaction_wall,
        gpu_time: measurements.gpu_time,
        complete_call_wall: measurements.complete_call_wall,
        raw_routed_complete_call_p90_ms,
        raw_routed_two_tps_budget_ms: RAW_ROUTED_TWO_TPS_BUDGET_MS,
        observed_host_raw_routed_two_tps_budget_pass,
        target_raw_routed_two_tps_necessary_condition: target_host_match
            .then_some(observed_host_raw_routed_two_tps_budget_pass),
        raw_routed_three_tps_budget_ms: RAW_ROUTED_THREE_TPS_BUDGET_MS,
        observed_host_raw_routed_three_tps_budget_pass,
        target_raw_routed_three_tps_diagnostic: target_host_match
            .then_some(observed_host_raw_routed_three_tps_budget_pass),
        fixed_nonexpert_ms: FIXED_NONEXPERT_MS,
        fixed_nonexpert_ledger_device: "Apple M4 Pro",
        fixed_nonexpert_ledger_context_only: true,
        legacy_m4_context_component_wall_p90_ms: legacy_m4_component_wall_p90_ms,
        legacy_m4_context_component_wall_tps: legacy_m4_component_wall_p90_ms
            .map(|milliseconds| 1000.0 / milliseconds),
        legacy_m4_context_component_gpu_p90_ms: legacy_m4_component_gpu_p90_ms,
        legacy_m4_context_component_gpu_tps: legacy_m4_component_gpu_p90_ms
            .map(|milliseconds| 1000.0 / milliseconds),
        relative_l2: measurements.relative_l2,
        maximum_absolute_error: measurements.maximum_absolute_error,
        unequal_count: measurements.unequal_count,
        exactness_class: K4_EXACTNESS_CLASS,
        expert_identity_preserved: true,
        source_function_preserved: false,
        identity_substitution_allowed: false,
        candidate_route_gate_pass: measurements.candidate_route_gate_pass,
        safety_snapshots,
        performance_claim: None,
    };
    write_create_new(
        output_path,
        &serde_json::to_vec_pretty(&report).map_err(|error| error.to_string())?,
    )?;
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn raw_routed_budget_is_strict_and_rejects_non_finite_measurements() {
        assert!(raw_routed_budget_pass(
            499.999,
            RAW_ROUTED_TWO_TPS_BUDGET_MS
        ));
        assert!(!raw_routed_budget_pass(500.0, RAW_ROUTED_TWO_TPS_BUDGET_MS));
        assert!(!raw_routed_budget_pass(
            f64::INFINITY,
            RAW_ROUTED_TWO_TPS_BUDGET_MS
        ));
        assert!(!raw_routed_budget_pass(
            f64::NAN,
            RAW_ROUTED_TWO_TPS_BUDGET_MS
        ));
    }

    #[test]
    fn bit_determinism_distinguishes_signed_zero() {
        assert!(f32_slices_bit_equal(&[1.0, -2.0], &[1.0, -2.0]));
        assert!(!f32_slices_bit_equal(&[0.0], &[-0.0]));
        assert!(!f32_slices_bit_equal(&[1.0], &[1.0, 2.0]));
    }
}
