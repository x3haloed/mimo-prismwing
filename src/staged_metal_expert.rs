use super::{
    MappedSafetensors, MappedTensorMetadata, MetalMoeManifest, UniqueJson, ValidatedMappedFp8,
    accelerate_sgemm_right_transposed, decode_f8_e4m3fn, read_f32_file, sha256_hex, sha256_reader,
    validate_mapped_fp8, write_create_new,
};
use crate::text_endpoint::{ComponentSafetyMonitor, SafetySnapshot, component_pytorch_noaux_route};
use metal::{CompileOptions, Device, MTLCommandBufferStatus, MTLResourceOptions, MTLSize};
use serde::Serialize;
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

struct RoutedRowExecution {
    output: Vec<f32>,
    selected: Vec<u32>,
    weights: Vec<f32>,
    minimum_boundary_margin: f32,
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

fn metal_project(
    device: &metal::DeviceRef,
    queue: &metal::CommandQueueRef,
    pipeline: &metal::ComputePipelineStateRef,
    lut: &[f32],
    tensor: &ValidatedMappedFp8<'_>,
    activation: &[f32],
) -> Result<Vec<f32>, String> {
    #[repr(C)]
    struct Shape {
        rows: u32,
        columns: u32,
        block_rows: u32,
        block_columns: u32,
    }
    if activation.len() != tensor.columns || activation.iter().any(|x| !x.is_finite()) {
        return Err("Metal projection input mismatch".to_owned());
    }
    let shape = Shape {
        rows: tensor.rows as u32,
        columns: tensor.columns as u32,
        block_rows: 128,
        block_columns: 128,
    };
    let shared = MTLResourceOptions::StorageModeShared;
    let buffer = |bytes: &[u8]| {
        device.new_buffer_with_data(bytes.as_ptr().cast(), bytes.len() as u64, shared)
    };
    let weight = buffer(tensor.weight.bytes);
    let scale = buffer(tensor.scale.bytes);
    let input_buffer = device.new_buffer_with_data(
        activation.as_ptr().cast(),
        std::mem::size_of_val(activation) as u64,
        shared,
    );
    let output_buffer = device.new_buffer((tensor.rows * 4) as u64, shared);
    let shape_buffer = device.new_buffer_with_data(
        (&shape as *const Shape).cast(),
        std::mem::size_of::<Shape>() as u64,
        shared,
    );
    let lut_buffer = device.new_buffer_with_data(
        lut.as_ptr().cast(),
        std::mem::size_of_val(lut) as u64,
        shared,
    );
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
    command.commit();
    command.wait_until_completed();
    if command.status() != MTLCommandBufferStatus::Completed {
        return Err(format!("Metal projection failed: {:?}", command.status()));
    }
    // SAFETY: completion precedes reading the exactly rows-long shared F32 buffer.
    Ok(unsafe {
        std::slice::from_raw_parts(output_buffer.contents().cast::<f32>(), tensor.rows).to_vec()
    })
}

fn execute_staged_expert(
    device: &metal::DeviceRef,
    queue: &metal::CommandQueueRef,
    pipeline: &metal::ComputePipelineStateRef,
    lut: &[f32],
    projections: [&ValidatedMappedFp8<'_>; 3],
    input: &[f32],
) -> Result<Vec<f32>, String> {
    let [gate, up, down] = projections;
    let staged_input = dynamic_fp8_dequantized(input)?;
    let mut gate_output = metal_project(device, queue, pipeline, lut, gate, &staged_input)?;
    let mut up_output = metal_project(device, queue, pipeline, lut, up, &staged_input)?;
    round_bf16_values(&mut gate_output);
    round_bf16_values(&mut up_output);
    let hidden = gate_output
        .iter()
        .zip(&up_output)
        .map(|(&g, &u)| staged_swiglu(g, u))
        .collect::<Vec<_>>();
    let staged_hidden = dynamic_fp8_dequantized(&hidden)?;
    let mut output = metal_project(device, queue, pipeline, lut, down, &staged_hidden)?;
    round_bf16_values(&mut output);
    if output.iter().any(|x| !x.is_finite()) {
        return Err("staged Metal expert produced non-finite output".to_owned());
    }
    Ok(output)
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

    let execute = || -> Result<(Vec<f32>, f64), String> {
        let start = Instant::now();
        let output = execute_staged_expert(
            &device,
            &queue,
            &pipeline,
            &lut,
            [&gate, &up, &down],
            &input,
        )?;
        Ok((output, start.elapsed().as_secs_f64() * 1000.0))
    };

    let (_, cold_wall_ms) = execute()?;
    for _ in 0..WARMUPS {
        execute()?;
    }
    safety.checkpoint("after_warmups")?;
    let mut wall_ms = Vec::with_capacity(MEASUREMENTS);
    let mut output = Vec::new();
    for _ in 0..MEASUREMENTS {
        let (candidate, elapsed) = execute()?;
        output = candidate;
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
                if weight_file != scale_file {
                    return Err("weight and scale split across artifacts".to_owned());
                }
                let mapping = mappings.get(weight_file).ok_or("unknown expert artifact")?;
                let name = format!("{prefix}.{projection}_proj.weight");
                validate_mapped_fp8(mapping, &name, &format!("{name}_scale_inv"), shape_input)
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
        for (&expert_id, &route_weight) in selected.iter().zip(&weights) {
            let (gate, up, down) = projections
                .get(&expert_id)
                .ok_or_else(|| format!("runtime selected unauthoritative expert {expert_id}"))?;
            let expert_output =
                execute_staged_expert(&device, &queue, &pipeline, &lut, [gate, up, down], &input)?;
            for (destination, value) in output.iter_mut().zip(expert_output) {
                *destination += value * route_weight;
            }
        }
        round_bf16_values(&mut output);
        Ok(RoutedRowExecution {
            output,
            selected,
            weights,
            minimum_boundary_margin,
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
    if selected != frozen_selected {
        return Err(format!("native route order mismatch: {selected:?}"));
    }
    let maximum_route_weight_absolute_error = weights
        .iter()
        .zip(&frozen_weights)
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0_f32, f32::max);
    if maximum_route_weight_absolute_error > 3.0e-8 {
        return Err(format!(
            "native route-weight error {maximum_route_weight_absolute_error}"
        ));
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
        return Err(format!(
            "routed-row parity failed: rel L2 {relative_l2}, max abs {maximum_absolute_error}, BF16 equality {bf16_equality_fraction}"
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
    fn dynamic_fp8_is_deterministic_and_finite() {
        let input = (0..256)
            .map(|x| (x as f32 - 127.0) / 31.0)
            .collect::<Vec<_>>();
        let first = dynamic_fp8_dequantized(&input).expect("valid input");
        let second = dynamic_fp8_dequantized(&input).expect("valid input");
        assert_eq!(first, second);
        assert!(first.iter().all(|x| x.is_finite()));
    }
}
