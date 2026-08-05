use memmap2::{Mmap, MmapOptions};
use serde::de::{self, MapAccess, SeqAccess, Visitor};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
#[cfg(target_os = "macos")]
use std::time::Instant;

const MAX_HEADER_BYTES: u64 = 256 * 1024 * 1024;
const EXPERT_MAGIC: &[u8; 8] = b"PWEXPRT1";
const EXPERT_ALIGNMENT: u64 = 64;

struct UniqueJson(Value);

impl<'de> Deserialize<'de> for UniqueJson {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        struct UniqueVisitor;
        impl<'de> Visitor<'de> for UniqueVisitor {
            type Value = UniqueJson;
            fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
                formatter.write_str("JSON without duplicate object keys")
            }
            fn visit_bool<E>(self, value: bool) -> Result<Self::Value, E> {
                Ok(UniqueJson(Value::Bool(value)))
            }
            fn visit_i64<E>(self, value: i64) -> Result<Self::Value, E> {
                Ok(UniqueJson(Value::Number(value.into())))
            }
            fn visit_u64<E>(self, value: u64) -> Result<Self::Value, E> {
                Ok(UniqueJson(Value::Number(value.into())))
            }
            fn visit_f64<E: de::Error>(self, value: f64) -> Result<Self::Value, E> {
                serde_json::Number::from_f64(value)
                    .map(Value::Number)
                    .map(UniqueJson)
                    .ok_or_else(|| E::custom("non-finite JSON number"))
            }
            fn visit_str<E: de::Error>(self, value: &str) -> Result<Self::Value, E> {
                Ok(UniqueJson(Value::String(value.to_owned())))
            }
            fn visit_string<E>(self, value: String) -> Result<Self::Value, E> {
                Ok(UniqueJson(Value::String(value)))
            }
            fn visit_none<E>(self) -> Result<Self::Value, E> {
                Ok(UniqueJson(Value::Null))
            }
            fn visit_unit<E>(self) -> Result<Self::Value, E> {
                Ok(UniqueJson(Value::Null))
            }
            fn visit_seq<A: SeqAccess<'de>>(
                self,
                mut sequence: A,
            ) -> Result<Self::Value, A::Error> {
                let mut values = Vec::new();
                while let Some(value) = sequence.next_element::<UniqueJson>()? {
                    values.push(value.0);
                }
                Ok(UniqueJson(Value::Array(values)))
            }
            fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Self::Value, A::Error> {
                let mut values = serde_json::Map::new();
                while let Some((key, value)) = map.next_entry::<String, UniqueJson>()? {
                    if values.insert(key.clone(), value.0).is_some() {
                        return Err(de::Error::custom(format!("duplicate JSON key: {key}")));
                    }
                }
                Ok(UniqueJson(Value::Object(values)))
            }
        }
        deserializer.deserialize_any(UniqueVisitor)
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct MappedTensorMetadata {
    pub name: String,
    pub dtype: String,
    pub shape: Vec<u64>,
    pub data_offsets: [u64; 2],
    pub data_bytes: u64,
}

pub struct MappedSafetensors {
    mapping: Mmap,
    payload_start: usize,
    tensors: BTreeMap<String, MappedTensorMetadata>,
}

pub struct MappedTensorView<'a> {
    pub metadata: &'a MappedTensorMetadata,
    pub bytes: &'a [u8],
}

#[derive(Debug, Serialize)]
pub struct MappedTensorInspection {
    pub schema_version: u32,
    pub source_file: String,
    pub source_file_bytes: u64,
    pub tensor: MappedTensorMetadata,
    pub tensor_sha256: String,
    pub bytes_hashed: u64,
    pub mapping: &'static str,
}

#[derive(Debug, Serialize)]
pub struct Fp8GemvReport {
    pub schema_version: u32,
    pub source_file: String,
    pub weight: MappedTensorMetadata,
    pub scale: MappedTensorMetadata,
    pub input_f32: usize,
    pub output_f32: usize,
    pub input_sha256: String,
    pub output_sha256: String,
    pub output_first8: Vec<f32>,
    pub logical_bytes: u64,
    pub batch_size: u32,
    pub concurrency: u32,
    pub implementation: &'static str,
}

#[cfg(target_os = "macos")]
#[derive(Debug, Serialize)]
pub struct MetalFp8GemvReport {
    pub schema_version: u32,
    pub source_file: String,
    pub weight: MappedTensorMetadata,
    pub scale: MappedTensorMetadata,
    pub kernel_file: String,
    pub kernel_function: &'static str,
    pub device: String,
    pub input_f32: usize,
    pub output_f32: usize,
    pub input_sha256: String,
    pub reference_sha256: String,
    pub output_sha256: String,
    pub output_first8: Vec<f32>,
    pub relative_l2: f64,
    pub maximum_absolute_error: f32,
    pub compile_ms: f64,
    pub cold_wall_ms: f64,
    pub warmups: usize,
    pub measurements: usize,
    pub wall_ms: Vec<f64>,
    pub wall_p10_ms: f64,
    pub wall_median_ms: f64,
    pub wall_p90_ms: f64,
    pub readable_baseline_ms: f64,
    pub diagnostic_speedup: f64,
    pub timing_asymmetry: &'static str,
    pub logical_bytes: u64,
    pub batch_size: u32,
    pub concurrency: u32,
    pub accepted_tokens: u32,
    #[serde(rename = "A")]
    pub accepted_per_verification: u32,
    #[serde(rename = "U")]
    pub proposed_per_verification: u32,
    pub cache_state: &'static str,
    pub implementation: &'static str,
}

#[cfg(target_os = "macos")]
#[derive(Debug, Serialize)]
pub struct MetalFp8ExpertReport {
    pub schema_version: u32,
    pub semantic: &'static str,
    pub gate_up_source_file: String,
    pub down_source_file: String,
    pub gate: MappedTensorMetadata,
    pub up: MappedTensorMetadata,
    pub down: MappedTensorMetadata,
    pub kernel_file: String,
    pub fp8_kernel: &'static str,
    pub threadgroup_memory_bytes: u64,
    pub device: String,
    pub input_sha256: String,
    pub reference_sha256: String,
    pub output_sha256: String,
    pub output_first8: Vec<f32>,
    pub relative_l2: f64,
    pub maximum_absolute_error: f32,
    pub swiglu_fixture_maximum_absolute_error: f64,
    pub batch_kernel_fixture_maximum_absolute_error: Option<f32>,
    pub compile_ms: f64,
    pub cold_wall_ms: f64,
    pub warmups: usize,
    pub measurements: usize,
    pub wall_ms: Vec<f64>,
    pub wall_p10_ms: f64,
    pub wall_median_ms: f64,
    pub wall_p90_ms: f64,
    pub per_position_ms: f64,
    pub per_position_speedup_vs_pw0034_batch_one: f64,
    pub median_timing_gate_passed: bool,
    pub per_position_speedup_gate_passed: bool,
    pub idealized_serial_routed_only_tps: f64,
    pub idealized_serial_scope: &'static str,
    pub dispatch_composition: &'static str,
    pub logical_bytes: u64,
    pub batch_size: u32,
    pub concurrency: u32,
    pub accepted_tokens: u32,
    #[serde(rename = "A")]
    pub accepted_per_verification: u32,
    #[serde(rename = "U")]
    pub unique_expert_sets: u32,
    pub cache_state: &'static str,
    pub implementation: &'static str,
}

#[cfg(target_os = "macos")]
#[derive(Debug, Deserialize)]
struct SwiGluFixture {
    schema_version: u32,
    semantic: String,
    gate: Vec<f32>,
    up: Vec<f32>,
    expected_f64: Vec<f64>,
}

#[cfg(target_os = "macos")]
struct MetalExpertConfig {
    batch_size: usize,
    fp8_kernel: &'static str,
    threadgroup_batch_factor: usize,
    partial_values_per_lane: usize,
    timing_limit_ms: f64,
    minimum_per_position_speedup: f64,
}

#[cfg(target_os = "macos")]
#[derive(Debug, Deserialize)]
struct MetalMoeManifest {
    schema_version: u32,
    semantic: String,
    revision: String,
    layer: u32,
    batch_size: usize,
    top_k: usize,
    input_sha256: String,
    reference_sha256: String,
    selected_experts_by_position: Vec<Vec<u32>>,
    route_weights_by_position: Vec<Vec<f32>>,
    real_expert_positions: usize,
    padded_expert_positions: usize,
    experts: Vec<MetalMoeExpertManifest>,
    artifact_sha256: BTreeMap<String, String>,
    scheduling: String,
}

#[cfg(target_os = "macos")]
#[derive(Debug, Deserialize)]
struct MetalMoeExpertManifest {
    expert: u32,
    prefix: String,
    positions: Vec<u32>,
    slots: Vec<u32>,
    route_weights: Vec<f32>,
    tensor_files: BTreeMap<String, String>,
}

#[cfg(target_os = "macos")]
#[derive(Debug, Serialize)]
pub struct MetalFp8MoeReport {
    pub schema_version: u32,
    pub semantic: String,
    pub manifest_sha256: String,
    pub input_sha256: String,
    pub reference_sha256: String,
    pub output_sha256: String,
    pub output_first8: Vec<f32>,
    pub device: String,
    pub kernel_file: String,
    pub expert_kernel: &'static str,
    pub scatter_kernel: &'static str,
    pub layer: u32,
    pub batch_size: usize,
    pub top_k: usize,
    pub unique_experts: usize,
    pub expert_position_counts: BTreeMap<u32, usize>,
    pub real_expert_positions: usize,
    pub padded_expert_positions: usize,
    pub padding_overhead_fraction: f64,
    pub relative_l2: f64,
    pub maximum_absolute_error: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub router_file_sha256: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub maximum_route_weight_absolute_error: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub minimum_topk_boundary_margin: Option<f32>,
    pub scatter_fixture_maximum_absolute_error: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub union_kernel_fixture_maximum_absolute_error: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fused_gate_up_fixture_maximum_absolute_error: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub simdgroup_matrix_fixture_maximum_absolute_error: Option<f32>,
    pub compile_ms: f64,
    pub cold_wall_ms: f64,
    pub warmups: usize,
    pub measurements: usize,
    pub wall_ms: Vec<f64>,
    pub wall_p10_ms: f64,
    pub wall_median_ms: f64,
    pub wall_p90_ms: f64,
    pub median_timing_gate_passed: bool,
    pub routed_only_accepted_tps_diagnostic: f64,
    pub logical_source_and_io_bytes: u64,
    pub resident_buffer_bytes: u64,
    pub batch_concurrency: u32,
    pub accepted_tokens: u32,
    #[serde(rename = "A")]
    pub accepted_per_verification: u32,
    #[serde(rename = "U")]
    pub expert_union_factor: f64,
    pub cache_state: &'static str,
    pub scheduling_limitation: String,
    pub implementation: &'static str,
}

#[cfg(target_os = "macos")]
#[derive(Debug, Serialize)]
struct NativeRouteArtifact {
    schema_version: u32,
    semantic: &'static str,
    layer: u32,
    batch_size: usize,
    top_k: usize,
    selected_experts_by_position: Vec<Vec<u32>>,
    route_weights_by_position: Vec<Vec<f32>>,
}

#[cfg(target_os = "macos")]
struct NativeRoutes {
    selected: Vec<Vec<u32>>,
    weights: Vec<Vec<f32>>,
    minimum_boundary_margin: f32,
}

#[cfg(target_os = "macos")]
fn select_noaux_tc_routes(
    logits: &[f32],
    correction: &[f32],
    batch: usize,
    rows: usize,
    top_k: usize,
) -> Result<NativeRoutes, String> {
    if logits.len() != batch * rows
        || correction.len() != rows
        || top_k == 0
        || top_k >= rows
        || logits.iter().any(|value| !value.is_finite())
        || correction.iter().any(|value| !value.is_finite())
    {
        return Err("invalid noaux-tc router inputs".to_owned());
    }
    let mut selected = Vec::with_capacity(batch);
    let mut route_weights = Vec::with_capacity(batch);
    let mut minimum_boundary_margin = f32::INFINITY;
    for position in 0..batch {
        let scores = logits[position * rows..(position + 1) * rows]
            .iter()
            .map(|logit| 1.0_f32 / (1.0 + (-logit).exp()))
            .collect::<Vec<_>>();
        let corrected = scores
            .iter()
            .zip(correction)
            .map(|(score, bias)| score + bias)
            .collect::<Vec<_>>();
        let mut indices = (0..rows).collect::<Vec<_>>();
        indices.sort_by(|left, right| {
            corrected[*right]
                .total_cmp(&corrected[*left])
                .then(left.cmp(right))
        });
        let margin = corrected[indices[top_k - 1]] - corrected[indices[top_k]];
        if !margin.is_finite() || margin <= 0.0 {
            return Err(format!(
                "noaux-tc top-k boundary is tied at position {position}"
            ));
        }
        minimum_boundary_margin = minimum_boundary_margin.min(margin);
        let chosen = &indices[..top_k];
        let denominator = chosen.iter().map(|index| scores[*index]).sum::<f32>() + 1.0e-20;
        let weights = chosen
            .iter()
            .map(|index| scores[*index] / denominator)
            .collect::<Vec<_>>();
        if weights
            .iter()
            .any(|value| !value.is_finite() || *value < 0.0)
            || (weights.iter().sum::<f32>() - 1.0).abs() > 1.0e-6
        {
            return Err(format!(
                "noaux-tc route weights invalid at position {position}"
            ));
        }
        selected.push(chosen.iter().map(|index| *index as u32).collect());
        route_weights.push(weights);
    }
    Ok(NativeRoutes {
        selected,
        weights: route_weights,
        minimum_boundary_margin,
    })
}

#[cfg(target_os = "macos")]
#[derive(Debug, Serialize)]
pub struct MetalRouterReport {
    pub schema_version: u32,
    pub semantic: &'static str,
    pub router_file_sha256: String,
    pub reference_manifest_sha256: String,
    pub input_sha256: String,
    pub output_sha256: String,
    pub device: String,
    pub kernel_file: String,
    pub kernel_function: &'static str,
    pub threadgroup_memory_bytes: u64,
    pub selected_experts_by_position: Vec<Vec<u32>>,
    pub route_weights_by_position: Vec<Vec<f32>>,
    pub maximum_route_weight_absolute_error: f32,
    pub minimum_topk_boundary_margin: f32,
    pub kernel_fixture_maximum_absolute_error: f32,
    pub compile_ms: f64,
    pub cold_wall_ms: f64,
    pub warmups: usize,
    pub measurements: usize,
    pub wall_ms: Vec<f64>,
    pub wall_p10_ms: f64,
    pub wall_median_ms: f64,
    pub wall_p90_ms: f64,
    pub median_timing_gate_passed: bool,
    pub logical_bytes: u64,
    pub batch_size: usize,
    pub concurrency: u32,
    pub accepted_tokens: u32,
    #[serde(rename = "A")]
    pub accepted_per_verification: u32,
    #[serde(rename = "U")]
    pub expert_union_factor: f64,
    pub cache_state: &'static str,
    pub implementation: &'static str,
}

fn dtype_bytes(dtype: &str) -> Option<u64> {
    match dtype {
        "BOOL" | "U8" | "I8" | "F8_E4M3" | "F8_E4M3FN" | "F8_E5M2" => Some(1),
        "U16" | "I16" | "F16" | "BF16" => Some(2),
        "U32" | "I32" | "F32" => Some(4),
        "U64" | "I64" | "F64" => Some(8),
        _ => None,
    }
}

impl MappedSafetensors {
    pub fn open(path: &Path) -> Result<Self, String> {
        let file = File::open(path).map_err(|error| format!("{}: {error}", path.display()))?;
        let file_bytes = file.metadata().map_err(|error| error.to_string())?.len();
        if file_bytes < 16 {
            return Err("safetensors file is too short".to_owned());
        }
        // SAFETY: the file is opened read-only, the returned API exposes immutable slices,
        // and the mapping owns its lifetime independently of the File handle.
        let mapping = unsafe { MmapOptions::new().map(&file) }
            .map_err(|error| format!("{}: {error}", path.display()))?;
        let header_bytes = u64::from_le_bytes(
            mapping[..8]
                .try_into()
                .map_err(|_| "missing header prefix")?,
        );
        if header_bytes == 0
            || header_bytes > MAX_HEADER_BYTES
            || !header_bytes.is_multiple_of(8)
            || 8_u64
                .checked_add(header_bytes)
                .ok_or("header offset overflow")?
                > file_bytes
        {
            return Err(format!("invalid safetensors header length {header_bytes}"));
        }
        let payload_start_u64 = 8_u64.checked_add(header_bytes).ok_or("payload overflow")?;
        let payload_start =
            usize::try_from(payload_start_u64).map_err(|_| "payload offset does not fit usize")?;
        let header_end = payload_start;
        let mut deserializer = serde_json::Deserializer::from_slice(&mapping[8..header_end]);
        let header = UniqueJson::deserialize(&mut deserializer)
            .map_err(|error| format!("malformed safetensors header: {error}"))?
            .0;
        deserializer
            .end()
            .map_err(|error| format!("trailing header data: {error}"))?;
        let object = require_object(&header, "safetensors header")?;
        let payload_bytes = file_bytes
            .checked_sub(payload_start_u64)
            .ok_or("payload underflow")?;
        let mut tensors = BTreeMap::new();
        let mut ranges = Vec::new();
        for (name, value) in object {
            if name == "__metadata__" {
                if !value.is_object() {
                    return Err("__metadata__ must be an object".to_owned());
                }
                continue;
            }
            let metadata = require_object(value, name)?;
            let dtype = metadata
                .get("dtype")
                .and_then(Value::as_str)
                .ok_or_else(|| format!("{name}: missing dtype"))?;
            let element_bytes =
                dtype_bytes(dtype).ok_or_else(|| format!("{name}: unknown dtype {dtype}"))?;
            let shape = metadata
                .get("shape")
                .and_then(Value::as_array)
                .ok_or_else(|| format!("{name}: missing shape"))?
                .iter()
                .map(|dimension| {
                    dimension
                        .as_u64()
                        .ok_or_else(|| format!("{name}: invalid shape"))
                })
                .collect::<Result<Vec<_>, _>>()?;
            if shape.is_empty() || shape.contains(&0) {
                return Err(format!("{name}: empty or zero tensor shape"));
            }
            let elements = shape.iter().try_fold(1_u64, |total, dimension| {
                total
                    .checked_mul(*dimension)
                    .ok_or_else(|| format!("{name}: shape overflow"))
            })?;
            let expected_bytes = elements
                .checked_mul(element_bytes)
                .ok_or_else(|| format!("{name}: tensor byte overflow"))?;
            let (start, end) = require_u64_pair(
                metadata
                    .get("data_offsets")
                    .ok_or_else(|| format!("{name}: missing offsets"))?,
                name,
            )?;
            let data_bytes = end.checked_sub(start).ok_or("tensor offsets reversed")?;
            if data_bytes != expected_bytes {
                return Err(format!(
                    "{name}: shape/dtype requires {expected_bytes} bytes, got {data_bytes}"
                ));
            }
            if end > payload_bytes {
                return Err(format!("{name}: tensor payload exceeds file"));
            }
            let record = MappedTensorMetadata {
                name: name.clone(),
                dtype: dtype.to_owned(),
                shape,
                data_offsets: [start, end],
                data_bytes,
            };
            if tensors.insert(name.clone(), record).is_some() {
                return Err(format!("duplicate tensor name: {name}"));
            }
            ranges.push((start, end, name.clone()));
        }
        if tensors.is_empty() {
            return Err("safetensors file contains no tensors".to_owned());
        }
        ranges.sort_by_key(|range| (range.0, range.1));
        let mut previous_end = 0_u64;
        for (start, end, name) in ranges {
            if start < previous_end {
                return Err(format!("{name}: overlapping tensor payload"));
            }
            previous_end = end;
        }
        Ok(Self {
            mapping,
            payload_start,
            tensors,
        })
    }

    pub fn tensor(&self, name: &str) -> Result<MappedTensorView<'_>, String> {
        let metadata = self
            .tensors
            .get(name)
            .ok_or_else(|| format!("tensor is absent: {name}"))?;
        let start = self
            .payload_start
            .checked_add(
                usize::try_from(metadata.data_offsets[0])
                    .map_err(|_| "tensor offset does not fit usize")?,
            )
            .ok_or("tensor start overflow")?;
        let end = self
            .payload_start
            .checked_add(
                usize::try_from(metadata.data_offsets[1])
                    .map_err(|_| "tensor offset does not fit usize")?,
            )
            .ok_or("tensor end overflow")?;
        Ok(MappedTensorView {
            metadata,
            bytes: &self.mapping[start..end],
        })
    }

    pub fn tensor_count(&self) -> usize {
        self.tensors.len()
    }
}

pub fn inspect_mapped_tensor(path: &Path, name: &str) -> Result<MappedTensorInspection, String> {
    let mapped = MappedSafetensors::open(path)?;
    let view = mapped.tensor(name)?;
    Ok(MappedTensorInspection {
        schema_version: 1,
        source_file: path
            .file_name()
            .and_then(|name| name.to_str())
            .ok_or("source file name is not UTF-8")?
            .to_owned(),
        source_file_bytes: mapped.mapping.len() as u64,
        tensor: view.metadata.clone(),
        tensor_sha256: sha256_hex(view.bytes),
        bytes_hashed: view.bytes.len() as u64,
        mapping: "read_only_memmap",
    })
}

struct ValidatedMappedFp8<'a> {
    weight: MappedTensorView<'a>,
    scale: MappedTensorView<'a>,
    rows: usize,
    columns: usize,
    scale_columns: usize,
    scales: Vec<f32>,
}

fn validate_mapped_fp8<'a>(
    mapped: &'a MappedSafetensors,
    weight_name: &str,
    scale_name: &str,
    input: &[f32],
) -> Result<ValidatedMappedFp8<'a>, String> {
    let weight = mapped.tensor(weight_name)?;
    let scale = mapped.tensor(scale_name)?;
    validate_fp8_views(weight, scale, input)
}

fn validate_fp8_views<'a>(
    weight: MappedTensorView<'a>,
    scale: MappedTensorView<'a>,
    input: &[f32],
) -> Result<ValidatedMappedFp8<'a>, String> {
    if weight.metadata.dtype != "F8_E4M3" || weight.metadata.shape.len() != 2 {
        return Err("FP8 GEMV weight must be F8_E4M3 rank two".to_owned());
    }
    let rows = usize::try_from(weight.metadata.shape[0]).map_err(|_| "rows do not fit usize")?;
    let columns =
        usize::try_from(weight.metadata.shape[1]).map_err(|_| "columns do not fit usize")?;
    if rows == 0 || columns == 0 || !rows.is_multiple_of(128) || !columns.is_multiple_of(128) {
        return Err("FP8 GEMV dimensions must be nonzero multiples of 128".to_owned());
    }
    if input.len() != columns || input.iter().any(|value| !value.is_finite()) {
        return Err("FP8 GEMV input length or finiteness mismatch".to_owned());
    }
    let scale_rows = rows / 128;
    let scale_columns = columns / 128;
    if scale.metadata.dtype != "F32"
        || scale.metadata.shape != [scale_rows as u64, scale_columns as u64]
        || scale.bytes.len() != scale_rows * scale_columns * 4
    {
        return Err("FP8 GEMV scale grid mismatch".to_owned());
    }
    let scales = scale
        .bytes
        .chunks_exact(4)
        .map(|bytes| f32::from_le_bytes(bytes.try_into().expect("four-byte scale")))
        .collect::<Vec<_>>();
    if scales.iter().any(|value| !value.is_finite()) {
        return Err("FP8 GEMV scale is non-finite".to_owned());
    }
    if let Some(offset) = weight
        .bytes
        .iter()
        .position(|bits| matches!(bits, 0x7f | 0xff))
    {
        return Err(format!("non-finite FP8 weight at byte offset {offset}"));
    }
    Ok(ValidatedMappedFp8 {
        weight,
        scale,
        rows,
        columns,
        scale_columns,
        scales,
    })
}

fn mapped_fp8_gemv(
    mapped: &MappedSafetensors,
    weight_name: &str,
    scale_name: &str,
    input: &[f32],
) -> Result<Vec<f32>, String> {
    let validated = validate_mapped_fp8(mapped, weight_name, scale_name, input)?;
    let ValidatedMappedFp8 {
        weight,
        rows,
        columns,
        scale_columns,
        scales,
        ..
    } = validated;
    let mut output = vec![0.0_f32; rows];
    for (row, destination) in output.iter_mut().enumerate() {
        let row_offset = row
            .checked_mul(columns)
            .ok_or("weight row offset overflow")?;
        let scale_row = row / 128 * scale_columns;
        let mut sum = 0.0_f32;
        for (column, activation) in input.iter().enumerate() {
            let decoded = decode_f8_e4m3fn(weight.bytes[row_offset + column]);
            if !decoded.is_finite() {
                return Err(format!(
                    "non-finite FP8 weight at row {row}, column {column}"
                ));
            }
            sum += decoded * scales[scale_row + column / 128] * activation;
        }
        *destination = sum;
    }
    if output.iter().any(|value| !value.is_finite()) {
        return Err("FP8 GEMV produced non-finite output".to_owned());
    }
    Ok(output)
}

fn read_f32_file(path: &Path, expected: Option<usize>) -> Result<(Vec<u8>, Vec<f32>), String> {
    let bytes = fs::read(path).map_err(|error| format!("{}: {error}", path.display()))?;
    if !bytes.len().is_multiple_of(4) {
        return Err(format!(
            "{} length is not divisible by four",
            path.display()
        ));
    }
    let values = bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("four-byte F32")))
        .collect::<Vec<_>>();
    if expected.is_some_and(|length| values.len() != length) {
        return Err(format!("{} F32 length mismatch", path.display()));
    }
    if values.iter().any(|value| !value.is_finite()) {
        return Err(format!("{} contains non-finite F32", path.display()));
    }
    Ok((bytes, values))
}

fn write_create_new(path: &Path, bytes: &[u8]) -> Result<(), String> {
    if path.exists() {
        return Err(format!("refusing to overwrite {}", path.display()));
    }
    let temporary = path.with_file_name(format!(
        ".{}.tmp.{}",
        path.file_name()
            .and_then(|name| name.to_str())
            .ok_or("output name is not UTF-8")?,
        std::process::id()
    ));
    let write_result = (|| -> Result<(), String> {
        let mut file = OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&temporary)
            .map_err(|error| error.to_string())?;
        file.write_all(bytes)
            .and_then(|_| file.sync_all())
            .map_err(|error| error.to_string())?;
        fs::hard_link(&temporary, path).map_err(|error| error.to_string())?;
        Ok(())
    })();
    let _ = fs::remove_file(&temporary);
    write_result
}

pub fn run_mapped_fp8_gemv(
    source: &Path,
    weight_name: &str,
    scale_name: &str,
    input_path: &Path,
    output_path: &Path,
) -> Result<Fp8GemvReport, String> {
    let mapped = MappedSafetensors::open(source)?;
    let weight = mapped.tensor(weight_name)?.metadata.clone();
    let scale = mapped.tensor(scale_name)?.metadata.clone();
    let (input_bytes, input) = read_f32_file(input_path, None)?;
    let output = mapped_fp8_gemv(&mapped, weight_name, scale_name, &input)?;
    let output_bytes = output
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect::<Vec<_>>();
    write_create_new(output_path, &output_bytes)?;
    let logical_bytes = weight
        .data_bytes
        .checked_add(scale.data_bytes)
        .and_then(|value| value.checked_add(input_bytes.len() as u64))
        .and_then(|value| value.checked_add(output_bytes.len() as u64))
        .ok_or("logical byte count overflow")?;
    Ok(Fp8GemvReport {
        schema_version: 1,
        source_file: source
            .file_name()
            .and_then(|name| name.to_str())
            .ok_or("source file name is not UTF-8")?
            .to_owned(),
        weight,
        scale,
        input_f32: input.len(),
        output_f32: output.len(),
        input_sha256: sha256_hex(&input_bytes),
        output_sha256: sha256_hex(&output_bytes),
        output_first8: output.iter().copied().take(8).collect(),
        logical_bytes,
        batch_size: 1,
        concurrency: 1,
        implementation: "single_thread_readable_mapped_fp8_reference",
    })
}

#[cfg(target_os = "macos")]
pub fn run_metal_mapped_fp8_gemv(
    source: &Path,
    kernel_path: &Path,
    weight_name: &str,
    scale_name: &str,
    input_path: &Path,
    reference_path: &Path,
    output_path: &Path,
) -> Result<MetalFp8GemvReport, String> {
    use metal::{CompileOptions, Device, MTLCommandBufferStatus, MTLResourceOptions, MTLSize};

    const KERNEL_FUNCTION: &str = "block_fp8_gemv_parallel_lut_blocked";
    const LANES: u64 = 64;
    const WARMUPS: usize = 5;
    const MEASUREMENTS: usize = 30;
    const READABLE_BASELINE_MS: f64 = 300.0;

    if output_path.exists() {
        return Err(format!("refusing to overwrite {}", output_path.display()));
    }
    let mapped = MappedSafetensors::open(source)?;
    let (input_bytes, input) = read_f32_file(input_path, None)?;
    let validated = validate_mapped_fp8(&mapped, weight_name, scale_name, &input)?;
    let (reference_bytes, reference) = read_f32_file(reference_path, Some(validated.rows))?;
    let kernel_source = fs::read_to_string(kernel_path)
        .map_err(|error| format!("{}: {error}", kernel_path.display()))?;
    if !kernel_source.contains(&format!("kernel void {KERNEL_FUNCTION}")) {
        return Err(format!("kernel source lacks {KERNEL_FUNCTION}"));
    }

    let device = Device::system_default().ok_or("no Metal device is available")?;
    if device.max_threads_per_threadgroup().width < LANES {
        return Err("Metal device cannot dispatch 64-lane threadgroups".to_owned());
    }
    let compile_options = CompileOptions::new();
    compile_options.set_fast_math_enabled(false);
    let compile_start = Instant::now();
    let library = device
        .new_library_with_source(&kernel_source, &compile_options)
        .map_err(|error| format!("Metal compilation failed: {error}"))?;
    let function = library
        .get_function(KERNEL_FUNCTION, None)
        .map_err(|error| format!("Metal kernel lookup failed: {error}"))?;
    let pipeline = device
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|error| format!("Metal pipeline creation failed: {error}"))?;
    let compile_ms = compile_start.elapsed().as_secs_f64() * 1000.0;
    if pipeline.max_total_threads_per_threadgroup() < LANES {
        return Err("Metal pipeline cannot dispatch 64-lane threadgroups".to_owned());
    }

    #[repr(C)]
    struct GemvShape {
        rows: u32,
        columns: u32,
        block_rows: u32,
        block_columns: u32,
    }
    let shape = GemvShape {
        rows: u32::try_from(validated.rows).map_err(|_| "rows do not fit u32")?,
        columns: u32::try_from(validated.columns).map_err(|_| "columns do not fit u32")?,
        block_rows: 128,
        block_columns: 128,
    };
    let decode_lut = (0_u16..=255)
        .map(|bits| decode_f8_e4m3fn(bits as u8))
        .collect::<Vec<_>>();
    if decode_lut.len() != 256 {
        return Err("FP8 decode LUT is incomplete".to_owned());
    }

    let shared = MTLResourceOptions::StorageModeShared;
    let weight_buffer = device.new_buffer_with_data(
        validated.weight.bytes.as_ptr().cast(),
        validated.weight.bytes.len() as u64,
        shared,
    );
    let scale_buffer = device.new_buffer_with_data(
        validated.scale.bytes.as_ptr().cast(),
        validated.scale.bytes.len() as u64,
        shared,
    );
    let input_buffer = device.new_buffer_with_data(
        input.as_ptr().cast(),
        std::mem::size_of_val(input.as_slice()) as u64,
        shared,
    );
    let output_buffer =
        device.new_buffer((validated.rows * std::mem::size_of::<f32>()) as u64, shared);
    let shape_buffer = device.new_buffer_with_data(
        (&shape as *const GemvShape).cast(),
        std::mem::size_of::<GemvShape>() as u64,
        shared,
    );
    let lut_buffer = device.new_buffer_with_data(
        decode_lut.as_ptr().cast(),
        std::mem::size_of_val(decode_lut.as_slice()) as u64,
        shared,
    );
    let queue = device.new_command_queue();

    let dispatch = || -> Result<f64, String> {
        let start = Instant::now();
        let command_buffer = queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(&weight_buffer), 0);
        encoder.set_buffer(1, Some(&scale_buffer), 0);
        encoder.set_buffer(2, Some(&input_buffer), 0);
        encoder.set_buffer(3, Some(&output_buffer), 0);
        encoder.set_buffer(4, Some(&shape_buffer), 0);
        encoder.set_buffer(5, Some(&lut_buffer), 0);
        encoder.set_threadgroup_memory_length(0, LANES * std::mem::size_of::<f32>() as u64);
        encoder.dispatch_thread_groups(
            MTLSize {
                width: validated.rows as u64,
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
        command_buffer.commit();
        command_buffer.wait_until_completed();
        let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
        if command_buffer.status() != MTLCommandBufferStatus::Completed {
            return Err(format!(
                "Metal command failed with status {:?}",
                command_buffer.status()
            ));
        }
        Ok(elapsed_ms)
    };

    let cold_wall_ms = dispatch()?;
    for _ in 0..WARMUPS {
        dispatch()?;
    }
    let mut wall_ms = Vec::with_capacity(MEASUREMENTS);
    for _ in 0..MEASUREMENTS {
        wall_ms.push(dispatch()?);
    }

    // SAFETY: StorageModeShared exposes a CPU-visible pointer, every submitted command has
    // completed, and the buffer is at least rows * sizeof(f32) bytes long.
    let output = unsafe {
        std::slice::from_raw_parts(output_buffer.contents().cast::<f32>(), validated.rows).to_vec()
    };
    if output.iter().any(|value| !value.is_finite()) {
        return Err("Metal FP8 GEMV produced non-finite output".to_owned());
    }
    let mut squared_error = 0.0_f64;
    let mut squared_reference = 0.0_f64;
    let mut maximum_absolute_error = 0.0_f32;
    for (&candidate, &expected) in output.iter().zip(&reference) {
        let difference = candidate - expected;
        squared_error += f64::from(difference) * f64::from(difference);
        squared_reference += f64::from(expected) * f64::from(expected);
        maximum_absolute_error = maximum_absolute_error.max(difference.abs());
    }
    if squared_reference == 0.0 {
        return Err("reference output has zero L2 norm".to_owned());
    }
    let relative_l2 = (squared_error / squared_reference).sqrt();
    if relative_l2 > 2.0e-5 || maximum_absolute_error > 2.0e-4 {
        return Err(format!(
            "Metal parity failed: relative L2 {relative_l2}, max abs {maximum_absolute_error}"
        ));
    }
    let output_bytes = output
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect::<Vec<_>>();
    write_create_new(output_path, &output_bytes)?;

    let mut ordered = wall_ms.clone();
    ordered.sort_by(f64::total_cmp);
    let percentile = |fraction: f64| -> f64 {
        ordered[((ordered.len() - 1) as f64 * fraction).round() as usize]
    };
    let wall_p10_ms = percentile(0.10);
    let wall_median_ms = percentile(0.50);
    let wall_p90_ms = percentile(0.90);
    let diagnostic_speedup = READABLE_BASELINE_MS / wall_median_ms;
    if wall_median_ms > 5.0 || diagnostic_speedup < 20.0 {
        return Err(format!(
            "Metal timing gate failed: median {wall_median_ms} ms, diagnostic speedup {diagnostic_speedup}x"
        ));
    }
    let logical_bytes = validated
        .weight
        .metadata
        .data_bytes
        .checked_add(validated.scale.metadata.data_bytes)
        .and_then(|value| value.checked_add(input_bytes.len() as u64))
        .and_then(|value| value.checked_add(output_bytes.len() as u64))
        .ok_or("logical byte count overflow")?;
    Ok(MetalFp8GemvReport {
        schema_version: 1,
        source_file: source
            .file_name()
            .and_then(|name| name.to_str())
            .ok_or("source file name is not UTF-8")?
            .to_owned(),
        weight: validated.weight.metadata.clone(),
        scale: validated.scale.metadata.clone(),
        kernel_file: kernel_path
            .file_name()
            .and_then(|name| name.to_str())
            .ok_or("kernel file name is not UTF-8")?
            .to_owned(),
        kernel_function: KERNEL_FUNCTION,
        device: device.name().to_owned(),
        input_f32: input.len(),
        output_f32: output.len(),
        input_sha256: sha256_hex(&input_bytes),
        reference_sha256: sha256_hex(&reference_bytes),
        output_sha256: sha256_hex(&output_bytes),
        output_first8: output.iter().copied().take(8).collect(),
        relative_l2,
        maximum_absolute_error,
        compile_ms,
        cold_wall_ms,
        warmups: WARMUPS,
        measurements: MEASUREMENTS,
        wall_ms,
        wall_p10_ms,
        wall_median_ms,
        wall_p90_ms,
        readable_baseline_ms: READABLE_BASELINE_MS,
        diagnostic_speedup,
        timing_asymmetry: "PW-0032 baseline includes process, mapping, validation, GEMV, fsync, hashing, and JSON; Metal series times serialized command creation, dispatch, and wait against resident application buffers",
        logical_bytes,
        batch_size: 1,
        concurrency: 1,
        accepted_tokens: 0,
        accepted_per_verification: 0,
        proposed_per_verification: 0,
        cache_state: "source OS-cache state uncontrolled; application Metal buffers warm after cold dispatch",
        implementation: "rust_owned_mapped_fp8_metal_blocked_lut_64_lane",
    })
}

#[cfg(target_os = "macos")]
fn run_metal_fp8_expert_configured(
    gate_up_source: &Path,
    down_source: &Path,
    kernel_path: &Path,
    input_path: &Path,
    reference_path: &Path,
    output_path: &Path,
    config: MetalExpertConfig,
) -> Result<MetalFp8ExpertReport, String> {
    use metal::{CompileOptions, Device, MTLCommandBufferStatus, MTLResourceOptions, MTLSize};

    const GATE: &str = "model.layers.43.mlp.experts.32.gate_proj.weight";
    const UP: &str = "model.layers.43.mlp.experts.32.up_proj.weight";
    const DOWN: &str = "model.layers.43.mlp.experts.32.down_proj.weight";
    const SWIGLU_KERNEL: &str = "swiglu_f32";
    const LANES: u64 = 64;
    const WARMUPS: usize = 5;
    const MEASUREMENTS: usize = 30;
    let MetalExpertConfig {
        batch_size,
        fp8_kernel,
        threadgroup_batch_factor,
        partial_values_per_lane,
        timing_limit_ms,
        minimum_per_position_speedup,
    } = config;
    if threadgroup_batch_factor == 0 || partial_values_per_lane == 0 {
        return Err("Metal expert threadgroup configuration is zero".to_owned());
    }
    let threadgroup_memory_bytes = LANES
        .checked_mul(partial_values_per_lane as u64)
        .and_then(|value| value.checked_mul(4))
        .ok_or("Metal expert threadgroup memory overflow")?;

    if output_path.exists() {
        return Err(format!("refusing to overwrite {}", output_path.display()));
    }
    let gate_up_mapping = MappedSafetensors::open(gate_up_source)?;
    let down_mapping = MappedSafetensors::open(down_source)?;
    if !matches!(batch_size, 1 | 8) {
        return Err("complete-expert batch must be one or eight".to_owned());
    }
    let (input_bytes, input) = read_f32_file(input_path, Some(4096 * batch_size))?;
    let gate = validate_mapped_fp8(
        &gate_up_mapping,
        GATE,
        &format!("{GATE}_scale_inv"),
        &input[..4096],
    )?;
    let up = validate_mapped_fp8(
        &gate_up_mapping,
        UP,
        &format!("{UP}_scale_inv"),
        &input[..4096],
    )?;
    let hidden_shape_authority = vec![0.0_f32; 2048];
    let down = validate_mapped_fp8(
        &down_mapping,
        DOWN,
        &format!("{DOWN}_scale_inv"),
        &hidden_shape_authority,
    )?;
    if gate.rows != 2048
        || gate.columns != 4096
        || up.rows != gate.rows
        || up.columns != gate.columns
        || down.rows != 4096
        || down.columns != 2048
    {
        return Err("layer-43/expert-32 projection shape mismatch".to_owned());
    }
    let (reference_bytes, reference) = read_f32_file(reference_path, Some(down.rows * batch_size))?;
    let kernel_source = fs::read_to_string(kernel_path)
        .map_err(|error| format!("{}: {error}", kernel_path.display()))?;
    for function in [fp8_kernel, SWIGLU_KERNEL] {
        if !kernel_source.contains(&format!("kernel void {function}")) {
            return Err(format!("kernel source lacks {function}"));
        }
    }

    let device = Device::system_default().ok_or("no Metal device is available")?;
    if device.max_threads_per_threadgroup().width < LANES {
        return Err("Metal device cannot dispatch 64-lane threadgroups".to_owned());
    }
    let compile_options = CompileOptions::new();
    compile_options.set_fast_math_enabled(false);
    let compile_start = Instant::now();
    let library = device
        .new_library_with_source(&kernel_source, &compile_options)
        .map_err(|error| format!("Metal compilation failed: {error}"))?;
    let fp8_function = library
        .get_function(fp8_kernel, None)
        .map_err(|error| format!("Metal FP8 kernel lookup failed: {error}"))?;
    let swiglu_function = library
        .get_function(SWIGLU_KERNEL, None)
        .map_err(|error| format!("Metal SwiGLU kernel lookup failed: {error}"))?;
    let fp8_pipeline = device
        .new_compute_pipeline_state_with_function(&fp8_function)
        .map_err(|error| format!("Metal FP8 pipeline creation failed: {error}"))?;
    let swiglu_pipeline = device
        .new_compute_pipeline_state_with_function(&swiglu_function)
        .map_err(|error| format!("Metal SwiGLU pipeline creation failed: {error}"))?;
    let compile_ms = compile_start.elapsed().as_secs_f64() * 1000.0;
    if fp8_pipeline.max_total_threads_per_threadgroup() < LANES {
        return Err("Metal FP8 pipeline cannot dispatch 64 lanes".to_owned());
    }

    let shared = MTLResourceOptions::StorageModeShared;
    let queue = device.new_command_queue();
    let batch_kernel_fixture_maximum_absolute_error = if batch_size == 8 {
        #[repr(C)]
        struct FixtureShape {
            rows: u32,
            columns: u32,
            block_rows: u32,
            block_columns: u32,
        }
        let fixture_shape = FixtureShape {
            rows: 128,
            columns: 128,
            block_rows: 128,
            block_columns: 128,
        };
        let legal_codes = [0x00_u8, 0x01, 0x30, 0x38, 0x40, 0x78, 0xb8];
        let fixture_weights = (0..128 * 128)
            .map(|index| legal_codes[(index * 13 + index / 128) % legal_codes.len()])
            .collect::<Vec<_>>();
        let fixture_scale = [0.75_f32];
        let fixture_input = (0..8 * 128)
            .map(|index| ((index * 17 % 23) as f32 - 11.0) * 0.03125)
            .collect::<Vec<_>>();
        let fixture_lut = (0_u16..=255)
            .map(|bits| decode_f8_e4m3fn(bits as u8))
            .collect::<Vec<_>>();
        let fixture_expected = (0..8)
            .flat_map(|position| {
                let weights = &fixture_weights;
                let inputs = &fixture_input;
                (0..128).map(move |row| {
                    let mut sum = 0.0_f32;
                    for column in 0..128 {
                        sum += decode_f8_e4m3fn(weights[row * 128 + column])
                            * fixture_scale[0]
                            * inputs[position * 128 + column];
                    }
                    sum
                })
            })
            .collect::<Vec<_>>();
        let fixture_weight_buffer = device.new_buffer_with_data(
            fixture_weights.as_ptr().cast(),
            fixture_weights.len() as u64,
            shared,
        );
        let fixture_scale_buffer = device.new_buffer_with_data(
            fixture_scale.as_ptr().cast(),
            std::mem::size_of_val(&fixture_scale) as u64,
            shared,
        );
        let fixture_input_buffer = device.new_buffer_with_data(
            fixture_input.as_ptr().cast(),
            std::mem::size_of_val(fixture_input.as_slice()) as u64,
            shared,
        );
        let fixture_output_buffer = device.new_buffer((8 * 128 * 4) as u64, shared);
        let fixture_shape_buffer = device.new_buffer_with_data(
            (&fixture_shape as *const FixtureShape).cast(),
            std::mem::size_of::<FixtureShape>() as u64,
            shared,
        );
        let fixture_lut_buffer = device.new_buffer_with_data(
            fixture_lut.as_ptr().cast(),
            std::mem::size_of_val(fixture_lut.as_slice()) as u64,
            shared,
        );
        let fixture_command = queue.new_command_buffer();
        let fixture_encoder = fixture_command.new_compute_command_encoder();
        fixture_encoder.set_compute_pipeline_state(&fp8_pipeline);
        fixture_encoder.set_buffer(0, Some(&fixture_weight_buffer), 0);
        fixture_encoder.set_buffer(1, Some(&fixture_scale_buffer), 0);
        fixture_encoder.set_buffer(2, Some(&fixture_input_buffer), 0);
        fixture_encoder.set_buffer(3, Some(&fixture_output_buffer), 0);
        fixture_encoder.set_buffer(4, Some(&fixture_shape_buffer), 0);
        fixture_encoder.set_buffer(5, Some(&fixture_lut_buffer), 0);
        fixture_encoder.set_threadgroup_memory_length(0, threadgroup_memory_bytes);
        fixture_encoder.dispatch_thread_groups(
            MTLSize {
                width: (128 * threadgroup_batch_factor) as u64,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: LANES,
                height: 1,
                depth: 1,
            },
        );
        fixture_encoder.end_encoding();
        fixture_command.commit();
        fixture_command.wait_until_completed();
        if fixture_command.status() != MTLCommandBufferStatus::Completed {
            return Err("Metal FP8 GEMM8 fixture command failed".to_owned());
        }
        // SAFETY: the fixture shared buffer is complete and exactly 8 * 128 F32 values long.
        let fixture_actual = unsafe {
            std::slice::from_raw_parts(
                fixture_output_buffer.contents().cast::<f32>(),
                fixture_expected.len(),
            )
        };
        if fixture_actual.iter().any(|value| !value.is_finite()) {
            return Err("Metal FP8 GEMM8 fixture produced non-finite output".to_owned());
        }
        let maximum = fixture_actual
            .iter()
            .zip(&fixture_expected)
            .map(|(&actual, &expected)| (actual - expected).abs())
            .fold(0.0_f32, f32::max);
        if maximum > 2.0e-4 {
            return Err(format!("Metal FP8 GEMM8 fixture failed: max abs {maximum}"));
        }
        Some(maximum)
    } else {
        None
    };
    let swiglu_fixture: SwiGluFixture =
        serde_json::from_str(include_str!("../evals/fixtures/tiny/swiglu-f32.json"))
            .map_err(|error| format!("SwiGLU fixture parse failed: {error}"))?;
    if swiglu_fixture.schema_version != 1
        || swiglu_fixture.semantic != "silu_gate_times_up_f32"
        || swiglu_fixture.gate.len() != swiglu_fixture.up.len()
        || swiglu_fixture.gate.len() != swiglu_fixture.expected_f64.len()
        || swiglu_fixture.gate.is_empty()
        || swiglu_fixture
            .gate
            .iter()
            .chain(&swiglu_fixture.up)
            .any(|value| !value.is_finite())
    {
        return Err("SwiGLU fixture identity or dimensions mismatch".to_owned());
    }
    let fixture_count = u32::try_from(swiglu_fixture.gate.len())
        .map_err(|_| "SwiGLU fixture length does not fit u32")?;
    let fixture_gate = device.new_buffer_with_data(
        swiglu_fixture.gate.as_ptr().cast(),
        std::mem::size_of_val(swiglu_fixture.gate.as_slice()) as u64,
        shared,
    );
    let fixture_up = device.new_buffer_with_data(
        swiglu_fixture.up.as_ptr().cast(),
        std::mem::size_of_val(swiglu_fixture.up.as_slice()) as u64,
        shared,
    );
    let fixture_output = device.new_buffer(
        std::mem::size_of_val(swiglu_fixture.gate.as_slice()) as u64,
        shared,
    );
    let fixture_count_buffer = device.new_buffer_with_data(
        (&fixture_count as *const u32).cast(),
        std::mem::size_of::<u32>() as u64,
        shared,
    );
    let fixture_command = queue.new_command_buffer();
    let fixture_encoder = fixture_command.new_compute_command_encoder();
    fixture_encoder.set_compute_pipeline_state(&swiglu_pipeline);
    fixture_encoder.set_buffer(0, Some(&fixture_gate), 0);
    fixture_encoder.set_buffer(1, Some(&fixture_up), 0);
    fixture_encoder.set_buffer(2, Some(&fixture_output), 0);
    fixture_encoder.set_buffer(3, Some(&fixture_count_buffer), 0);
    fixture_encoder.dispatch_threads(
        MTLSize {
            width: u64::from(fixture_count),
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: u64::from(fixture_count),
            height: 1,
            depth: 1,
        },
    );
    fixture_encoder.end_encoding();
    fixture_command.commit();
    fixture_command.wait_until_completed();
    if fixture_command.status() != MTLCommandBufferStatus::Completed {
        return Err("Metal SwiGLU fixture command failed".to_owned());
    }
    // SAFETY: the shared fixture buffer is complete and exactly fixture_count F32 values long.
    let fixture_actual = unsafe {
        std::slice::from_raw_parts(
            fixture_output.contents().cast::<f32>(),
            fixture_count as usize,
        )
    };
    let swiglu_fixture_maximum_absolute_error = fixture_actual
        .iter()
        .zip(&swiglu_fixture.expected_f64)
        .map(|(&actual, &expected)| (f64::from(actual) - expected).abs())
        .fold(0.0_f64, f64::max);
    if fixture_actual.iter().any(|value| !value.is_finite())
        || swiglu_fixture_maximum_absolute_error > 2.0e-6
    {
        return Err(format!(
            "Metal SwiGLU fixture failed: max abs {swiglu_fixture_maximum_absolute_error}"
        ));
    }

    #[repr(C)]
    struct GemvShape {
        rows: u32,
        columns: u32,
        block_rows: u32,
        block_columns: u32,
    }
    let gate_shape = GemvShape {
        rows: gate.rows as u32,
        columns: gate.columns as u32,
        block_rows: 128,
        block_columns: 128,
    };
    let down_shape = GemvShape {
        rows: down.rows as u32,
        columns: down.columns as u32,
        block_rows: 128,
        block_columns: 128,
    };
    let hidden_count = u32::try_from(gate.rows * batch_size)
        .map_err(|_| "batched expert hidden count does not fit u32")?;
    let decode_lut = (0_u16..=255)
        .map(|bits| decode_f8_e4m3fn(bits as u8))
        .collect::<Vec<_>>();

    let buffer = |bytes: &[u8]| {
        device.new_buffer_with_data(bytes.as_ptr().cast(), bytes.len() as u64, shared)
    };
    let gate_weight_buffer = buffer(gate.weight.bytes);
    let gate_scale_buffer = buffer(gate.scale.bytes);
    let up_weight_buffer = buffer(up.weight.bytes);
    let up_scale_buffer = buffer(up.scale.bytes);
    let down_weight_buffer = buffer(down.weight.bytes);
    let down_scale_buffer = buffer(down.scale.bytes);
    let input_buffer = device.new_buffer_with_data(
        input.as_ptr().cast(),
        std::mem::size_of_val(input.as_slice()) as u64,
        shared,
    );
    let gate_output = device.new_buffer((gate.rows * batch_size * 4) as u64, shared);
    let up_output = device.new_buffer((up.rows * batch_size * 4) as u64, shared);
    let hidden_output = device.new_buffer((gate.rows * batch_size * 4) as u64, shared);
    let final_output = device.new_buffer((down.rows * batch_size * 4) as u64, shared);
    let gate_shape_buffer = device.new_buffer_with_data(
        (&gate_shape as *const GemvShape).cast(),
        std::mem::size_of::<GemvShape>() as u64,
        shared,
    );
    let down_shape_buffer = device.new_buffer_with_data(
        (&down_shape as *const GemvShape).cast(),
        std::mem::size_of::<GemvShape>() as u64,
        shared,
    );
    let hidden_count_buffer = device.new_buffer_with_data(
        (&hidden_count as *const u32).cast(),
        std::mem::size_of::<u32>() as u64,
        shared,
    );
    let lut_buffer = device.new_buffer_with_data(
        decode_lut.as_ptr().cast(),
        std::mem::size_of_val(decode_lut.as_slice()) as u64,
        shared,
    );

    let dispatch = || -> Result<f64, String> {
        let start = Instant::now();
        let command = queue.new_command_buffer();
        let encoder = command.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&fp8_pipeline);
        encoder.set_buffer(0, Some(&gate_weight_buffer), 0);
        encoder.set_buffer(1, Some(&gate_scale_buffer), 0);
        encoder.set_buffer(2, Some(&input_buffer), 0);
        encoder.set_buffer(3, Some(&gate_output), 0);
        encoder.set_buffer(4, Some(&gate_shape_buffer), 0);
        encoder.set_buffer(5, Some(&lut_buffer), 0);
        encoder.set_threadgroup_memory_length(0, threadgroup_memory_bytes);
        encoder.dispatch_thread_groups(
            MTLSize {
                width: (gate.rows * threadgroup_batch_factor) as u64,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: LANES,
                height: 1,
                depth: 1,
            },
        );

        encoder.set_buffer(0, Some(&up_weight_buffer), 0);
        encoder.set_buffer(1, Some(&up_scale_buffer), 0);
        encoder.set_buffer(3, Some(&up_output), 0);
        encoder.dispatch_thread_groups(
            MTLSize {
                width: (up.rows * threadgroup_batch_factor) as u64,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: LANES,
                height: 1,
                depth: 1,
            },
        );

        encoder.set_compute_pipeline_state(&swiglu_pipeline);
        encoder.set_buffer(0, Some(&gate_output), 0);
        encoder.set_buffer(1, Some(&up_output), 0);
        encoder.set_buffer(2, Some(&hidden_output), 0);
        encoder.set_buffer(3, Some(&hidden_count_buffer), 0);
        encoder.dispatch_threads(
            MTLSize {
                width: (gate.rows * batch_size) as u64,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: 256,
                height: 1,
                depth: 1,
            },
        );

        encoder.set_compute_pipeline_state(&fp8_pipeline);
        encoder.set_buffer(0, Some(&down_weight_buffer), 0);
        encoder.set_buffer(1, Some(&down_scale_buffer), 0);
        encoder.set_buffer(2, Some(&hidden_output), 0);
        encoder.set_buffer(3, Some(&final_output), 0);
        encoder.set_buffer(4, Some(&down_shape_buffer), 0);
        encoder.set_buffer(5, Some(&lut_buffer), 0);
        encoder.set_threadgroup_memory_length(0, threadgroup_memory_bytes);
        encoder.dispatch_thread_groups(
            MTLSize {
                width: (down.rows * threadgroup_batch_factor) as u64,
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
        let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
        if command.status() != MTLCommandBufferStatus::Completed {
            return Err(format!(
                "Metal expert command failed: {:?}",
                command.status()
            ));
        }
        Ok(elapsed_ms)
    };

    let cold_wall_ms = dispatch()?;
    for _ in 0..WARMUPS {
        dispatch()?;
    }
    let mut wall_ms = Vec::with_capacity(MEASUREMENTS);
    for _ in 0..MEASUREMENTS {
        wall_ms.push(dispatch()?);
    }
    // SAFETY: the final shared buffer is complete and exactly batch * down.rows F32 values long.
    let output = unsafe {
        std::slice::from_raw_parts(
            final_output.contents().cast::<f32>(),
            down.rows * batch_size,
        )
        .to_vec()
    };
    if output.iter().any(|value| !value.is_finite()) {
        return Err("Metal complete expert produced non-finite output".to_owned());
    }
    let mut squared_error = 0.0_f64;
    let mut squared_reference = 0.0_f64;
    let mut maximum_absolute_error = 0.0_f32;
    for (&candidate, &expected) in output.iter().zip(&reference) {
        let difference = candidate - expected;
        squared_error += f64::from(difference) * f64::from(difference);
        squared_reference += f64::from(expected) * f64::from(expected);
        maximum_absolute_error = maximum_absolute_error.max(difference.abs());
    }
    if squared_reference == 0.0 {
        return Err("complete-expert reference has zero L2 norm".to_owned());
    }
    let relative_l2 = (squared_error / squared_reference).sqrt();
    if relative_l2 > 3.0e-5 || maximum_absolute_error > 2.0e-8 {
        return Err(format!(
            "complete-expert parity failed: relative L2 {relative_l2}, max abs {maximum_absolute_error}"
        ));
    }
    let output_bytes = output
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect::<Vec<_>>();
    write_create_new(output_path, &output_bytes)?;

    let mut ordered = wall_ms.clone();
    ordered.sort_by(f64::total_cmp);
    let percentile = |fraction: f64| -> f64 {
        ordered[((ordered.len() - 1) as f64 * fraction).round() as usize]
    };
    let wall_p10_ms = percentile(0.10);
    let wall_median_ms = percentile(0.50);
    let wall_p90_ms = percentile(0.90);
    let per_position_ms = wall_median_ms / batch_size as f64;
    let per_position_speedup_vs_pw0034_batch_one = 1.020875 / per_position_ms;
    let median_timing_gate_passed = wall_median_ms <= timing_limit_ms;
    let per_position_speedup_gate_passed =
        per_position_speedup_vs_pw0034_batch_one >= minimum_per_position_speedup;
    let logical_bytes = [
        gate.weight.metadata.data_bytes,
        gate.scale.metadata.data_bytes,
        up.weight.metadata.data_bytes,
        up.scale.metadata.data_bytes,
        down.weight.metadata.data_bytes,
        down.scale.metadata.data_bytes,
        input_bytes.len() as u64,
        output_bytes.len() as u64,
    ]
    .into_iter()
    .try_fold(0_u64, |total, value| total.checked_add(value))
    .ok_or("complete-expert logical byte count overflow")?;
    Ok(MetalFp8ExpertReport {
        schema_version: 1,
        semantic: if batch_size == 1 {
            "mimo_layer43_expert32_source_fp8_gate_up_swiglu_down"
        } else {
            "mimo_layer43_expert32_source_fp8_gate_up_swiglu_down_batch8"
        },
        gate_up_source_file: gate_up_source
            .file_name()
            .and_then(|name| name.to_str())
            .ok_or("gate/up source file name is not UTF-8")?
            .to_owned(),
        down_source_file: down_source
            .file_name()
            .and_then(|name| name.to_str())
            .ok_or("down source file name is not UTF-8")?
            .to_owned(),
        gate: gate.weight.metadata.clone(),
        up: up.weight.metadata.clone(),
        down: down.weight.metadata.clone(),
        kernel_file: kernel_path
            .file_name()
            .and_then(|name| name.to_str())
            .ok_or("kernel file name is not UTF-8")?
            .to_owned(),
        fp8_kernel,
        threadgroup_memory_bytes,
        device: device.name().to_owned(),
        input_sha256: sha256_hex(&input_bytes),
        reference_sha256: sha256_hex(&reference_bytes),
        output_sha256: sha256_hex(&output_bytes),
        output_first8: output.iter().copied().take(8).collect(),
        relative_l2,
        maximum_absolute_error,
        swiglu_fixture_maximum_absolute_error,
        batch_kernel_fixture_maximum_absolute_error,
        compile_ms,
        cold_wall_ms,
        warmups: WARMUPS,
        measurements: MEASUREMENTS,
        wall_ms,
        wall_p10_ms,
        wall_median_ms,
        wall_p90_ms,
        per_position_ms,
        per_position_speedup_vs_pw0034_batch_one,
        median_timing_gate_passed,
        per_position_speedup_gate_passed,
        idealized_serial_routed_only_tps: 1000.0 * batch_size as f64
            / (wall_median_ms * 8.0 * 47.0),
        idealized_serial_scope: if batch_size == 1 {
            "one batch-one expert cost repeated serially for eight experts across 47 routed layers; excludes routing, dense spine, attention, logits, storage, batching, MTP, and endpoint overhead"
        } else {
            "one batch-eight expert cost repeated for eight perfectly reused experts across 47 routed layers and divided by eight accepted positions; excludes routing, dense spine, attention, logits, storage, MTP, and endpoint overhead"
        },
        dispatch_composition: if batch_size == 1 {
            "one serialized command buffer: gate FP8 GEMV, up FP8 GEMV, F32 SwiGLU, down FP8 GEMV"
        } else if partial_values_per_lane == 8 {
            "one serialized command buffer: gate shared-weight FP8 GEMM8, up shared-weight FP8 GEMM8, F32 SwiGLU, down shared-weight FP8 GEMM8"
        } else {
            "one serialized command buffer: gate FP8 GEMM8, up FP8 GEMM8, F32 SwiGLU, down FP8 GEMM8"
        },
        logical_bytes,
        batch_size: batch_size as u32,
        concurrency: 1,
        accepted_tokens: 0,
        accepted_per_verification: if batch_size == 8 { 8 } else { 0 },
        unique_expert_sets: if batch_size == 8 { 1 } else { 0 },
        cache_state: "source OS-cache state uncontrolled; source copied into persistent application Metal buffers before timed series",
        implementation: if batch_size == 1 {
            "rust_owned_metal_source_fp8_complete_expert"
        } else if partial_values_per_lane == 8 {
            "rust_owned_metal_source_fp8_complete_expert_batch8_shared_weight"
        } else {
            "rust_owned_metal_source_fp8_complete_expert_batch8"
        },
    })
}

#[cfg(target_os = "macos")]
pub fn run_metal_fp8_expert(
    gate_up_source: &Path,
    down_source: &Path,
    kernel_path: &Path,
    input_path: &Path,
    reference_path: &Path,
    output_path: &Path,
) -> Result<MetalFp8ExpertReport, String> {
    run_metal_fp8_expert_configured(
        gate_up_source,
        down_source,
        kernel_path,
        input_path,
        reference_path,
        output_path,
        MetalExpertConfig {
            batch_size: 1,
            fp8_kernel: "block_fp8_gemv_parallel_lut_blocked",
            threadgroup_batch_factor: 1,
            partial_values_per_lane: 1,
            timing_limit_ms: 3.0,
            minimum_per_position_speedup: 0.0,
        },
    )
}

#[cfg(target_os = "macos")]
pub fn run_metal_fp8_expert_batch8(
    gate_up_source: &Path,
    down_source: &Path,
    kernel_path: &Path,
    input_path: &Path,
    reference_path: &Path,
    output_path: &Path,
) -> Result<MetalFp8ExpertReport, String> {
    run_metal_fp8_expert_configured(
        gate_up_source,
        down_source,
        kernel_path,
        input_path,
        reference_path,
        output_path,
        MetalExpertConfig {
            batch_size: 8,
            fp8_kernel: "block_fp8_gemm8_parallel_lut_blocked",
            threadgroup_batch_factor: 8,
            partial_values_per_lane: 1,
            timing_limit_ms: 4.0,
            minimum_per_position_speedup: 2.0,
        },
    )
}

#[cfg(target_os = "macos")]
pub fn run_metal_fp8_expert_batch8_shared_weight(
    gate_up_source: &Path,
    down_source: &Path,
    kernel_path: &Path,
    input_path: &Path,
    reference_path: &Path,
    output_path: &Path,
) -> Result<MetalFp8ExpertReport, String> {
    run_metal_fp8_expert_configured(
        gate_up_source,
        down_source,
        kernel_path,
        input_path,
        reference_path,
        output_path,
        MetalExpertConfig {
            batch_size: 8,
            fp8_kernel: "block_fp8_gemm8_shared_weight_lut_blocked",
            threadgroup_batch_factor: 1,
            partial_values_per_lane: 8,
            timing_limit_ms: 3.5,
            minimum_per_position_speedup: 2.5,
        },
    )
}

#[cfg(target_os = "macos")]
pub fn run_metal_fp8_moe_block(
    manifest_path: &Path,
    artifact_root: &Path,
    kernel_path: &Path,
    input_path: &Path,
    reference_path: &Path,
    output_path: &Path,
) -> Result<MetalFp8MoeReport, String> {
    run_metal_fp8_moe_block_impl(
        manifest_path,
        artifact_root,
        kernel_path,
        input_path,
        reference_path,
        output_path,
        MoeExecutionMode {
            dynamic_router_path: None,
            union_parallel: false,
            fused_gate_up: false,
            simdgroup_matrix: false,
        },
    )
}

#[cfg(target_os = "macos")]
pub fn run_metal_dynamic_fp8_moe_block(
    manifest_path: &Path,
    artifact_root: &Path,
    router_path: &Path,
    kernel_path: &Path,
    input_path: &Path,
    reference_path: &Path,
    output_path: &Path,
) -> Result<MetalFp8MoeReport, String> {
    run_metal_fp8_moe_block_impl(
        manifest_path,
        artifact_root,
        kernel_path,
        input_path,
        reference_path,
        output_path,
        MoeExecutionMode {
            dynamic_router_path: Some(router_path),
            union_parallel: false,
            fused_gate_up: false,
            simdgroup_matrix: false,
        },
    )
}

#[cfg(target_os = "macos")]
pub fn run_metal_union_parallel_fp8_moe_block(
    manifest_path: &Path,
    artifact_root: &Path,
    router_path: &Path,
    kernel_path: &Path,
    input_path: &Path,
    reference_path: &Path,
    output_path: &Path,
) -> Result<MetalFp8MoeReport, String> {
    run_metal_fp8_moe_block_impl(
        manifest_path,
        artifact_root,
        kernel_path,
        input_path,
        reference_path,
        output_path,
        MoeExecutionMode {
            dynamic_router_path: Some(router_path),
            union_parallel: true,
            fused_gate_up: false,
            simdgroup_matrix: false,
        },
    )
}

#[cfg(target_os = "macos")]
pub fn run_metal_fused_gate_up_fp8_moe_block(
    manifest_path: &Path,
    artifact_root: &Path,
    router_path: &Path,
    kernel_path: &Path,
    input_path: &Path,
    reference_path: &Path,
    output_path: &Path,
) -> Result<MetalFp8MoeReport, String> {
    run_metal_fp8_moe_block_impl(
        manifest_path,
        artifact_root,
        kernel_path,
        input_path,
        reference_path,
        output_path,
        MoeExecutionMode {
            dynamic_router_path: Some(router_path),
            union_parallel: false,
            fused_gate_up: true,
            simdgroup_matrix: false,
        },
    )
}

#[cfg(target_os = "macos")]
pub fn run_metal_simdgroup_matrix_fp8_moe_block(
    manifest_path: &Path,
    artifact_root: &Path,
    router_path: &Path,
    kernel_path: &Path,
    input_path: &Path,
    reference_path: &Path,
    output_path: &Path,
) -> Result<MetalFp8MoeReport, String> {
    run_metal_fp8_moe_block_impl(
        manifest_path,
        artifact_root,
        kernel_path,
        input_path,
        reference_path,
        output_path,
        MoeExecutionMode {
            dynamic_router_path: Some(router_path),
            union_parallel: false,
            fused_gate_up: false,
            simdgroup_matrix: true,
        },
    )
}

#[cfg(target_os = "macos")]
struct MoeExecutionMode<'a> {
    dynamic_router_path: Option<&'a Path>,
    union_parallel: bool,
    fused_gate_up: bool,
    simdgroup_matrix: bool,
}

#[cfg(target_os = "macos")]
fn run_metal_fp8_moe_block_impl(
    manifest_path: &Path,
    artifact_root: &Path,
    kernel_path: &Path,
    input_path: &Path,
    reference_path: &Path,
    output_path: &Path,
    mode: MoeExecutionMode<'_>,
) -> Result<MetalFp8MoeReport, String> {
    use metal::{
        CompileOptions, Device, MTLCommandBufferStatus, MTLGPUFamily, MTLResourceOptions, MTLSize,
        NSRange,
    };
    use std::path::Component;

    const REVISION: &str = "63651580ca774f8504f676040460aed3e1244ac1";
    const EXPERT_KERNEL: &str = "block_fp8_gemm8_shared_weight_lut_blocked";
    const SWIGLU_KERNEL: &str = "swiglu_f32";
    const SCATTER_KERNEL: &str = "route_weighted_scatter_add_f32";
    const ROUTER_KERNEL: &str = "f32_gemm8_shared_weight";
    const UNION_EXPERT_KERNEL: &str = "block_fp8_expert_union_gemm8_shared_weight_lut_blocked";
    const FUSED_GATE_UP_KERNEL: &str = "block_fp8_gemm8_fused_gate_up_lut_blocked";
    const SIMDGROUP_MATRIX_KERNEL: &str = "block_fp8_gemm8_simdgroup_matrix_lut_blocked";
    const ROUTER_SHA256: &str = "12c1579d28b78dd69ec9342eb9d1f378efc5aa3c2f2a28b5ec73578e6a8bbcdd";
    const ROUTER_WEIGHT: &str = "model.layers.43.mlp.gate.weight";
    const ROUTER_BIAS: &str = "model.layers.43.mlp.gate.e_score_correction_bias";
    const LANES: u64 = 64;
    const BATCH: usize = 8;
    const HIDDEN: usize = 4096;
    const INTERMEDIATE: usize = 2048;
    const WARMUPS: usize = 5;
    const MEASUREMENTS: usize = 30;

    if output_path.exists() {
        return Err(format!("refusing to overwrite {}", output_path.display()));
    }
    if (mode.union_parallel || mode.fused_gate_up || mode.simdgroup_matrix)
        && mode.dynamic_router_path.is_none()
    {
        return Err("accelerated MoE mode requires dynamic router authority".to_owned());
    }
    if u8::from(mode.union_parallel)
        + u8::from(mode.fused_gate_up)
        + u8::from(mode.simdgroup_matrix)
        > 1
    {
        return Err("conflicting MoE execution modes".to_owned());
    }
    let manifest_bytes =
        fs::read(manifest_path).map_err(|error| format!("{}: {error}", manifest_path.display()))?;
    let unique: UniqueJson = serde_json::from_slice(&manifest_bytes)
        .map_err(|error| format!("{}: {error}", manifest_path.display()))?;
    let manifest: MetalMoeManifest =
        serde_json::from_value(unique.0).map_err(|error| format!("manifest: {error}"))?;
    if manifest.schema_version != 1
        || manifest.semantic != "mimo_layer43_fixture_scheduled_source_fp8_moe_block"
        || manifest.revision != REVISION
        || manifest.layer != 43
        || manifest.batch_size != BATCH
        || manifest.top_k != 8
        || manifest.real_expert_positions != 64
        || manifest.padded_expert_positions != 72
        || manifest.experts.len() != 9
        || manifest.scheduling != "fixture_static_source_routes"
    {
        return Err("unknown heterogeneous MoE manifest identity".to_owned());
    }
    if manifest.selected_experts_by_position.len() != BATCH
        || manifest.route_weights_by_position.len() != BATCH
        || manifest
            .selected_experts_by_position
            .iter()
            .any(|row| row.len() != manifest.top_k)
        || manifest
            .route_weights_by_position
            .iter()
            .any(|row| row.len() != manifest.top_k || row.iter().any(|value| !value.is_finite()))
    {
        return Err("heterogeneous MoE route matrix mismatch".to_owned());
    }
    for weights in &manifest.route_weights_by_position {
        let sum: f32 = weights.iter().sum();
        if weights.iter().any(|value| *value < 0.0) || (sum - 1.0).abs() > 1.0e-5 {
            return Err("heterogeneous MoE route weights are not normalized".to_owned());
        }
    }
    let mut expert_ids = BTreeSet::new();
    let mut placements = BTreeSet::new();
    let mut counts = Vec::new();
    let mut prior_expert = None;
    for expert in &manifest.experts {
        let count = expert.positions.len();
        if prior_expert.is_some_and(|prior| expert.expert <= prior)
            || !expert_ids.insert(expert.expert)
            || !matches!(count, 3 | 5 | 8)
            || expert.slots.len() != count
            || expert.route_weights.len() != count
            || expert
                .route_weights
                .iter()
                .any(|weight| !weight.is_finite())
            || expert.prefix != format!("model.layers.43.mlp.experts.{}", expert.expert)
        {
            return Err("heterogeneous MoE expert manifest mismatch".to_owned());
        }
        prior_expert = Some(expert.expert);
        counts.push(count);
        for index in 0..count {
            let position = expert.positions[index] as usize;
            let slot = expert.slots[index] as usize;
            if position >= BATCH
                || slot >= manifest.top_k
                || !placements.insert((position, slot))
                || manifest.selected_experts_by_position[position][slot] != expert.expert
                || (manifest.route_weights_by_position[position][slot]
                    - expert.route_weights[index])
                    .abs()
                    > f32::EPSILON
            {
                return Err("heterogeneous MoE placement mismatch".to_owned());
            }
        }
    }
    counts.sort_unstable();
    if counts != [3, 5, 8, 8, 8, 8, 8, 8, 8] || placements.len() != 64 {
        return Err("heterogeneous MoE count distribution mismatch".to_owned());
    }

    let resolve_artifact = |name: &str| -> Result<PathBuf, String> {
        let relative = Path::new(name);
        let mut components = relative.components();
        if !matches!(components.next(), Some(Component::Normal(_)))
            || components.next().is_some()
            || relative.file_name().and_then(|value| value.to_str()) != Some(name)
        {
            return Err(format!("unsafe artifact name: {name}"));
        }
        Ok(artifact_root.join(relative))
    };
    let mut verified_artifacts = BTreeSet::new();
    for (name, expected_hash) in &manifest.artifact_sha256 {
        if expected_hash.len() != 64 || !expected_hash.bytes().all(|byte| byte.is_ascii_hexdigit())
        {
            return Err(format!("invalid artifact SHA-256: {name}"));
        }
        let path = resolve_artifact(name)?;
        let mut file = File::open(&path).map_err(|error| format!("{}: {error}", path.display()))?;
        if sha256_reader(&mut file)? != *expected_hash {
            return Err(format!("artifact SHA-256 mismatch: {name}"));
        }
        verified_artifacts.insert(name.clone());
    }
    let expected_tensor_file_keys = BTreeSet::from([
        "gate_weight",
        "gate_scale",
        "up_weight",
        "up_scale",
        "down_weight",
        "down_scale",
    ]);
    for expert in &manifest.experts {
        if expert
            .tensor_files
            .keys()
            .map(String::as_str)
            .collect::<BTreeSet<_>>()
            != expected_tensor_file_keys
            || expert
                .tensor_files
                .values()
                .any(|name| !verified_artifacts.contains(name))
        {
            return Err("expert references an unauthoritative artifact".to_owned());
        }
    }
    let (input_bytes, input) = read_f32_file(input_path, Some(BATCH * HIDDEN))?;
    let (reference_bytes, reference) = read_f32_file(reference_path, Some(BATCH * HIDDEN))?;
    if sha256_hex(&input_bytes) != manifest.input_sha256
        || sha256_hex(&reference_bytes) != manifest.reference_sha256
    {
        return Err("MoE input or reference hash mismatch".to_owned());
    }

    let dynamic_router_source = if let Some(router_path) = mode.dynamic_router_path {
        let mut router_file = File::open(router_path)
            .map_err(|error| format!("{}: {error}", router_path.display()))?;
        if sha256_reader(&mut router_file)? != ROUTER_SHA256 {
            return Err("layer-43 router artifact SHA-256 mismatch".to_owned());
        }
        let mapped = MappedSafetensors::open(router_path)?;
        let weight = mapped.tensor(ROUTER_WEIGHT)?;
        let bias = mapped.tensor(ROUTER_BIAS)?;
        if weight.metadata.dtype != "F32"
            || weight.metadata.shape != [256, HIDDEN as u64]
            || weight.bytes.len() != 256 * HIDDEN * 4
            || bias.metadata.dtype != "F32"
            || bias.metadata.shape != [256]
            || bias.bytes.len() != 256 * 4
        {
            return Err("layer-43 router tensor layout mismatch".to_owned());
        }
        let correction = bias
            .bytes
            .chunks_exact(4)
            .map(|bytes| f32::from_le_bytes(bytes.try_into().expect("four-byte bias")))
            .collect::<Vec<_>>();
        if correction.iter().any(|value| !value.is_finite()) {
            return Err("router correction bias is non-finite".to_owned());
        }
        Some((weight.bytes.to_vec(), correction))
    } else {
        None
    };

    let kernel_source = fs::read_to_string(kernel_path)
        .map_err(|error| format!("{}: {error}", kernel_path.display()))?;
    let mut required_kernels = vec![EXPERT_KERNEL, SWIGLU_KERNEL, SCATTER_KERNEL];
    if dynamic_router_source.is_some() {
        required_kernels.push(ROUTER_KERNEL);
    }
    if mode.union_parallel {
        required_kernels.push(UNION_EXPERT_KERNEL);
    }
    if mode.fused_gate_up {
        required_kernels.push(FUSED_GATE_UP_KERNEL);
    }
    if mode.simdgroup_matrix {
        required_kernels.push(SIMDGROUP_MATRIX_KERNEL);
    }
    for function in required_kernels {
        if !kernel_source.contains(&format!("kernel void {function}")) {
            return Err(format!("kernel source lacks {function}"));
        }
    }
    let device = Device::system_default().ok_or("no Metal device is available")?;
    if mode.simdgroup_matrix && !device.supports_family(MTLGPUFamily::Apple7) {
        return Err("device lacks Apple GPU family 7 SIMD-group matrices".to_owned());
    }
    if device.max_threads_per_threadgroup().width < LANES {
        return Err("Metal device cannot dispatch 64-lane threadgroups".to_owned());
    }
    let options = CompileOptions::new();
    options.set_fast_math_enabled(false);
    let compile_start = Instant::now();
    let library = device
        .new_library_with_source(&kernel_source, &options)
        .map_err(|error| format!("Metal compilation failed: {error}"))?;
    let expert_function = library
        .get_function(EXPERT_KERNEL, None)
        .map_err(|error| format!("expert kernel: {error}"))?;
    let swiglu_function = library
        .get_function(SWIGLU_KERNEL, None)
        .map_err(|error| format!("SwiGLU kernel: {error}"))?;
    let scatter_function = library
        .get_function(SCATTER_KERNEL, None)
        .map_err(|error| format!("scatter kernel: {error}"))?;
    let expert_pipeline = device
        .new_compute_pipeline_state_with_function(&expert_function)
        .map_err(|error| format!("expert pipeline: {error}"))?;
    let swiglu_pipeline = device
        .new_compute_pipeline_state_with_function(&swiglu_function)
        .map_err(|error| format!("SwiGLU pipeline: {error}"))?;
    let scatter_pipeline = device
        .new_compute_pipeline_state_with_function(&scatter_function)
        .map_err(|error| format!("scatter pipeline: {error}"))?;
    let router_pipeline = if dynamic_router_source.is_some() {
        let function = library
            .get_function(ROUTER_KERNEL, None)
            .map_err(|error| format!("router kernel: {error}"))?;
        Some(
            device
                .new_compute_pipeline_state_with_function(&function)
                .map_err(|error| format!("router pipeline: {error}"))?,
        )
    } else {
        None
    };
    let union_expert_pipeline = if mode.union_parallel {
        let function = library
            .get_function(UNION_EXPERT_KERNEL, None)
            .map_err(|error| format!("union expert kernel: {error}"))?;
        Some(
            device
                .new_compute_pipeline_state_with_function(&function)
                .map_err(|error| format!("union expert pipeline: {error}"))?,
        )
    } else {
        None
    };
    let fused_gate_up_pipeline = if mode.fused_gate_up {
        let function = library
            .get_function(FUSED_GATE_UP_KERNEL, None)
            .map_err(|error| format!("fused gate/up kernel: {error}"))?;
        Some(
            device
                .new_compute_pipeline_state_with_function(&function)
                .map_err(|error| format!("fused gate/up pipeline: {error}"))?,
        )
    } else {
        None
    };
    let simdgroup_matrix_pipeline = if mode.simdgroup_matrix {
        let function = library
            .get_function(SIMDGROUP_MATRIX_KERNEL, None)
            .map_err(|error| format!("SIMD-group matrix kernel: {error}"))?;
        let pipeline = device
            .new_compute_pipeline_state_with_function(&function)
            .map_err(|error| format!("SIMD-group matrix pipeline: {error}"))?;
        if pipeline.thread_execution_width() != 32 {
            return Err(format!(
                "SIMD-group matrix pipeline width is {}, expected 32",
                pipeline.thread_execution_width()
            ));
        }
        Some(pipeline)
    } else {
        None
    };
    let compile_ms = compile_start.elapsed().as_secs_f64() * 1000.0;
    let shared = MTLResourceOptions::StorageModeShared;
    let queue = device.new_command_queue();

    let union_kernel_fixture_maximum_absolute_error =
        if let Some(pipeline) = union_expert_pipeline.as_ref() {
            #[repr(C)]
            struct FixtureShape {
                rows: u32,
                columns: u32,
                block_rows: u32,
                block_columns: u32,
            }
            const FIXTURE_EXPERTS: usize = 2;
            const FIXTURE_ROWS: usize = 128;
            const FIXTURE_COLUMNS: usize = 128;
            let fixture_weights = (0..FIXTURE_EXPERTS * FIXTURE_ROWS * FIXTURE_COLUMNS)
                .map(|index| if index % 3 == 0 { 0xb8_u8 } else { 0x38_u8 })
                .collect::<Vec<_>>();
            let fixture_scales = [0.5_f32, 0.25_f32];
            let fixture_input = (0..FIXTURE_EXPERTS * BATCH * FIXTURE_COLUMNS)
                .map(|index| ((index % 13) as f32 - 6.0) * 0.01)
                .collect::<Vec<_>>();
            let mut fixture_expected = vec![0.0_f32; FIXTURE_EXPERTS * BATCH * FIXTURE_ROWS];
            for expert in 0..FIXTURE_EXPERTS {
                for position in 0..BATCH {
                    for row in 0..FIXTURE_ROWS {
                        let mut sum = 0.0_f32;
                        for column in 0..FIXTURE_COLUMNS {
                            let weight_index = expert * FIXTURE_ROWS * FIXTURE_COLUMNS
                                + row * FIXTURE_COLUMNS
                                + column;
                            sum += decode_f8_e4m3fn(fixture_weights[weight_index])
                                * fixture_scales[expert]
                                * fixture_input[expert * BATCH * FIXTURE_COLUMNS
                                    + position * FIXTURE_COLUMNS
                                    + column];
                        }
                        fixture_expected
                            [expert * BATCH * FIXTURE_ROWS + position * FIXTURE_ROWS + row] = sum;
                    }
                }
            }
            let fixture_shape = FixtureShape {
                rows: FIXTURE_ROWS as u32,
                columns: FIXTURE_COLUMNS as u32,
                block_rows: 128,
                block_columns: 128,
            };
            let fixture_weight_buffer = device.new_buffer_with_data(
                fixture_weights.as_ptr().cast(),
                fixture_weights.len() as u64,
                shared,
            );
            let fixture_scale_buffer = device.new_buffer_with_data(
                fixture_scales.as_ptr().cast(),
                std::mem::size_of_val(&fixture_scales) as u64,
                shared,
            );
            let fixture_input_buffer = device.new_buffer_with_data(
                fixture_input.as_ptr().cast(),
                std::mem::size_of_val(fixture_input.as_slice()) as u64,
                shared,
            );
            let fixture_output_buffer = device.new_buffer(
                std::mem::size_of_val(fixture_expected.as_slice()) as u64,
                shared,
            );
            let fixture_shape_buffer = device.new_buffer_with_data(
                (&fixture_shape as *const FixtureShape).cast(),
                std::mem::size_of::<FixtureShape>() as u64,
                shared,
            );
            let fixture_lut = (0_u16..=255)
                .map(|bits| decode_f8_e4m3fn(bits as u8))
                .collect::<Vec<_>>();
            let fixture_lut_buffer = device.new_buffer_with_data(
                fixture_lut.as_ptr().cast(),
                std::mem::size_of_val(fixture_lut.as_slice()) as u64,
                shared,
            );
            let fixture_command = queue.new_command_buffer();
            let fixture_encoder = fixture_command.new_compute_command_encoder();
            fixture_encoder.set_compute_pipeline_state(pipeline);
            fixture_encoder.set_buffer(0, Some(&fixture_weight_buffer), 0);
            fixture_encoder.set_buffer(1, Some(&fixture_scale_buffer), 0);
            fixture_encoder.set_buffer(2, Some(&fixture_input_buffer), 0);
            fixture_encoder.set_buffer(3, Some(&fixture_output_buffer), 0);
            fixture_encoder.set_buffer(4, Some(&fixture_shape_buffer), 0);
            fixture_encoder.set_buffer(5, Some(&fixture_lut_buffer), 0);
            fixture_encoder.set_threadgroup_memory_length(0, LANES * BATCH as u64 * 4);
            fixture_encoder.dispatch_thread_groups(
                MTLSize {
                    width: FIXTURE_ROWS as u64,
                    height: FIXTURE_EXPERTS as u64,
                    depth: 1,
                },
                MTLSize {
                    width: LANES,
                    height: 1,
                    depth: 1,
                },
            );
            fixture_encoder.end_encoding();
            fixture_command.commit();
            fixture_command.wait_until_completed();
            if fixture_command.status() != MTLCommandBufferStatus::Completed {
                return Err("union expert kernel fixture command failed".to_owned());
            }
            // SAFETY: the completed fixture output matches the allocated expected length.
            let fixture_actual = unsafe {
                std::slice::from_raw_parts(
                    fixture_output_buffer.contents().cast::<f32>(),
                    fixture_expected.len(),
                )
            };
            let maximum_error = fixture_actual
                .iter()
                .zip(&fixture_expected)
                .map(|(&actual, &expected)| (actual - expected).abs())
                .fold(0.0_f32, f32::max);
            if fixture_actual.iter().any(|value| !value.is_finite()) || maximum_error > 2.0e-5 {
                return Err(format!(
                    "union expert kernel fixture failed: {maximum_error}"
                ));
            }
            Some(maximum_error)
        } else {
            None
        };

    let fused_gate_up_fixture_maximum_absolute_error = if let Some(pipeline) =
        fused_gate_up_pipeline.as_ref()
    {
        #[repr(C)]
        struct FixtureShape {
            rows: u32,
            columns: u32,
            block_rows: u32,
            block_columns: u32,
        }
        const ROWS: usize = 128;
        const COLUMNS: usize = 128;
        let gate_weights = (0..ROWS * COLUMNS)
            .map(|index| if index % 3 == 0 { 0xb8_u8 } else { 0x38_u8 })
            .collect::<Vec<_>>();
        let up_weights = (0..ROWS * COLUMNS)
            .map(|index| if index % 5 == 0 { 0x40_u8 } else { 0xb0_u8 })
            .collect::<Vec<_>>();
        let gate_scale = [0.5_f32];
        let up_scale = [0.25_f32];
        let fixture_input = (0..BATCH * COLUMNS)
            .map(|index| ((index % 13) as f32 - 6.0) * 0.01)
            .collect::<Vec<_>>();
        let scalar = |weights: &[u8], scale: f32| {
            let mut result = vec![0.0_f32; BATCH * ROWS];
            for position in 0..BATCH {
                for row in 0..ROWS {
                    let mut sum = 0.0_f32;
                    for column in 0..COLUMNS {
                        sum += decode_f8_e4m3fn(weights[row * COLUMNS + column])
                            * scale
                            * fixture_input[position * COLUMNS + column];
                    }
                    result[position * ROWS + row] = sum;
                }
            }
            result
        };
        let expected_gate = scalar(&gate_weights, gate_scale[0]);
        let expected_up = scalar(&up_weights, up_scale[0]);
        let shape = FixtureShape {
            rows: ROWS as u32,
            columns: COLUMNS as u32,
            block_rows: 128,
            block_columns: 128,
        };
        let make_byte_buffer = |bytes: &[u8]| {
            device.new_buffer_with_data(bytes.as_ptr().cast(), bytes.len() as u64, shared)
        };
        let gate_weight_buffer = make_byte_buffer(&gate_weights);
        let up_weight_buffer = make_byte_buffer(&up_weights);
        let gate_scale_buffer = device.new_buffer_with_data(
            gate_scale.as_ptr().cast(),
            std::mem::size_of_val(&gate_scale) as u64,
            shared,
        );
        let up_scale_buffer = device.new_buffer_with_data(
            up_scale.as_ptr().cast(),
            std::mem::size_of_val(&up_scale) as u64,
            shared,
        );
        let input_buffer = device.new_buffer_with_data(
            fixture_input.as_ptr().cast(),
            std::mem::size_of_val(fixture_input.as_slice()) as u64,
            shared,
        );
        let gate_output_buffer = device.new_buffer((BATCH * ROWS * 4) as u64, shared);
        let up_output_buffer = device.new_buffer((BATCH * ROWS * 4) as u64, shared);
        let shape_buffer = device.new_buffer_with_data(
            (&shape as *const FixtureShape).cast(),
            std::mem::size_of::<FixtureShape>() as u64,
            shared,
        );
        let fixture_lut = (0_u16..=255)
            .map(|bits| decode_f8_e4m3fn(bits as u8))
            .collect::<Vec<_>>();
        let lut_buffer = device.new_buffer_with_data(
            fixture_lut.as_ptr().cast(),
            std::mem::size_of_val(fixture_lut.as_slice()) as u64,
            shared,
        );
        let command = queue.new_command_buffer();
        let encoder = command.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(&gate_weight_buffer), 0);
        encoder.set_buffer(1, Some(&gate_scale_buffer), 0);
        encoder.set_buffer(2, Some(&up_weight_buffer), 0);
        encoder.set_buffer(3, Some(&up_scale_buffer), 0);
        encoder.set_buffer(4, Some(&input_buffer), 0);
        encoder.set_buffer(5, Some(&gate_output_buffer), 0);
        encoder.set_buffer(6, Some(&up_output_buffer), 0);
        encoder.set_buffer(7, Some(&shape_buffer), 0);
        encoder.set_buffer(8, Some(&lut_buffer), 0);
        encoder.set_threadgroup_memory_length(0, LANES * BATCH as u64 * 4);
        encoder.dispatch_thread_groups(
            MTLSize {
                width: (ROWS * 2) as u64,
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
            return Err("fused gate/up fixture command failed".to_owned());
        }
        // SAFETY: both completed outputs contain exactly batch * rows F32 values.
        let (actual_gate, actual_up) = unsafe {
            (
                std::slice::from_raw_parts(
                    gate_output_buffer.contents().cast::<f32>(),
                    BATCH * ROWS,
                ),
                std::slice::from_raw_parts(up_output_buffer.contents().cast::<f32>(), BATCH * ROWS),
            )
        };
        let maximum_error = actual_gate
            .iter()
            .zip(&expected_gate)
            .chain(actual_up.iter().zip(&expected_up))
            .map(|(&actual, &expected)| (actual - expected).abs())
            .fold(0.0_f32, f32::max);
        if maximum_error > 2.0e-5 {
            return Err(format!("fused gate/up fixture failed: {maximum_error}"));
        }
        Some(maximum_error)
    } else {
        None
    };

    let simdgroup_matrix_fixture_maximum_absolute_error =
        if let Some(pipeline) = simdgroup_matrix_pipeline.as_ref() {
            #[repr(C)]
            struct FixtureShape {
                rows: u32,
                columns: u32,
                block_rows: u32,
                block_columns: u32,
            }
            const ROWS: usize = 16;
            const COLUMNS: usize = 256;
            let weights = (0..ROWS * COLUMNS)
                .map(|index| match index % 4 {
                    0 => 0xb8_u8,
                    1 => 0x38_u8,
                    2 => 0x40_u8,
                    _ => 0xb0_u8,
                })
                .collect::<Vec<_>>();
            let scales = [0.5_f32, 0.25_f32];
            let input = (0..BATCH * COLUMNS)
                .map(|index| ((index % 17) as f32 - 8.0) * 0.01)
                .collect::<Vec<_>>();
            let mut expected = vec![0.0_f32; BATCH * ROWS];
            for position in 0..BATCH {
                for row in 0..ROWS {
                    let mut sum = 0.0_f32;
                    for column in 0..COLUMNS {
                        sum += decode_f8_e4m3fn(weights[row * COLUMNS + column])
                            * scales[column / 128]
                            * input[position * COLUMNS + column];
                    }
                    expected[position * ROWS + row] = sum;
                }
            }
            let shape = FixtureShape {
                rows: ROWS as u32,
                columns: COLUMNS as u32,
                block_rows: 128,
                block_columns: 128,
            };
            let weight_buffer =
                device.new_buffer_with_data(weights.as_ptr().cast(), weights.len() as u64, shared);
            let scale_buffer = device.new_buffer_with_data(
                scales.as_ptr().cast(),
                std::mem::size_of_val(&scales) as u64,
                shared,
            );
            let input_buffer = device.new_buffer_with_data(
                input.as_ptr().cast(),
                std::mem::size_of_val(input.as_slice()) as u64,
                shared,
            );
            let output_buffer = device.new_buffer((BATCH * ROWS * 4) as u64, shared);
            let shape_buffer = device.new_buffer_with_data(
                (&shape as *const FixtureShape).cast(),
                std::mem::size_of::<FixtureShape>() as u64,
                shared,
            );
            let fixture_lut = (0_u16..=255)
                .map(|bits| decode_f8_e4m3fn(bits as u8))
                .collect::<Vec<_>>();
            let lut_buffer = device.new_buffer_with_data(
                fixture_lut.as_ptr().cast(),
                std::mem::size_of_val(fixture_lut.as_slice()) as u64,
                shared,
            );
            let command = queue.new_command_buffer();
            let encoder = command.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(pipeline);
            encoder.set_buffer(0, Some(&weight_buffer), 0);
            encoder.set_buffer(1, Some(&scale_buffer), 0);
            encoder.set_buffer(2, Some(&input_buffer), 0);
            encoder.set_buffer(3, Some(&output_buffer), 0);
            encoder.set_buffer(4, Some(&shape_buffer), 0);
            encoder.set_buffer(5, Some(&lut_buffer), 0);
            encoder.set_threadgroup_memory_length(0, 64 * 4);
            encoder.dispatch_thread_groups(
                MTLSize {
                    width: (ROWS / 8) as u64,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: 32,
                    height: 1,
                    depth: 1,
                },
            );
            encoder.end_encoding();
            command.commit();
            command.wait_until_completed();
            if command.status() != MTLCommandBufferStatus::Completed {
                return Err("SIMD-group matrix fixture command failed".to_owned());
            }
            // SAFETY: the completed output contains exactly batch * rows F32 values.
            let actual = unsafe {
                std::slice::from_raw_parts(output_buffer.contents().cast::<f32>(), BATCH * ROWS)
            };
            let maximum_error = actual
                .iter()
                .zip(&expected)
                .map(|(&actual, &expected)| (actual - expected).abs())
                .fold(0.0_f32, f32::max);
            if actual.iter().any(|value| !value.is_finite()) || maximum_error > 2.0e-4 {
                return Err(format!("SIMD-group matrix fixture failed: {maximum_error}"));
            }
            Some(maximum_error)
        } else {
            None
        };

    #[repr(C)]
    struct ScatterShape {
        count: u32,
        width: u32,
    }
    let scatter_fixture_output = [
        1.0_f32, 2.0, 3.0, 4.0, -1.0, -2.0, -3.0, -4.0, 0.5, 1.5, 2.5, 3.5,
    ];
    let scatter_fixture_weights = [0.25_f32, 0.5, 2.0];
    let scatter_fixture_positions = [2_u32, 0, 1];
    let scatter_fixture_shape = ScatterShape { count: 3, width: 4 };
    let scatter_fixture_expected = [
        -0.5_f32, -1.0, -1.5, -2.0, 1.0, 3.0, 5.0, 7.0, 0.25, 0.5, 0.75, 1.0,
    ];
    let fixture_source = device.new_buffer_with_data(
        scatter_fixture_output.as_ptr().cast(),
        std::mem::size_of_val(&scatter_fixture_output) as u64,
        shared,
    );
    let fixture_weights = device.new_buffer_with_data(
        scatter_fixture_weights.as_ptr().cast(),
        std::mem::size_of_val(&scatter_fixture_weights) as u64,
        shared,
    );
    let fixture_positions = device.new_buffer_with_data(
        scatter_fixture_positions.as_ptr().cast(),
        std::mem::size_of_val(&scatter_fixture_positions) as u64,
        shared,
    );
    let fixture_destination = device.new_buffer((12 * 4) as u64, shared);
    let fixture_shape = device.new_buffer_with_data(
        (&scatter_fixture_shape as *const ScatterShape).cast(),
        std::mem::size_of::<ScatterShape>() as u64,
        shared,
    );
    let fixture_command = queue.new_command_buffer();
    let fixture_blit = fixture_command.new_blit_command_encoder();
    fixture_blit.fill_buffer(&fixture_destination, NSRange::new(0, 12 * 4), 0);
    fixture_blit.end_encoding();
    let fixture_encoder = fixture_command.new_compute_command_encoder();
    fixture_encoder.set_compute_pipeline_state(&scatter_pipeline);
    fixture_encoder.set_buffer(0, Some(&fixture_source), 0);
    fixture_encoder.set_buffer(1, Some(&fixture_weights), 0);
    fixture_encoder.set_buffer(2, Some(&fixture_positions), 0);
    fixture_encoder.set_buffer(3, Some(&fixture_destination), 0);
    fixture_encoder.set_buffer(4, Some(&fixture_shape), 0);
    fixture_encoder.dispatch_threads(
        MTLSize {
            width: 12,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: 12,
            height: 1,
            depth: 1,
        },
    );
    fixture_encoder.end_encoding();
    fixture_command.commit();
    fixture_command.wait_until_completed();
    if fixture_command.status() != MTLCommandBufferStatus::Completed {
        return Err("scatter fixture command failed".to_owned());
    }
    // SAFETY: the completed shared fixture destination is exactly 12 F32 values.
    let fixture_actual =
        unsafe { std::slice::from_raw_parts(fixture_destination.contents().cast::<f32>(), 12) };
    let scatter_fixture_maximum_absolute_error = fixture_actual
        .iter()
        .zip(&scatter_fixture_expected)
        .map(|(&actual, &expected)| (actual - expected).abs())
        .fold(0.0_f32, f32::max);
    if scatter_fixture_maximum_absolute_error > 1.0e-6 {
        return Err(format!(
            "scatter fixture failed: {scatter_fixture_maximum_absolute_error}"
        ));
    }

    #[repr(C)]
    struct GemvShape {
        rows: u32,
        columns: u32,
        block_rows: u32,
        block_columns: u32,
    }
    struct ExpertBuffers {
        expert: u32,
        count: usize,
        input: metal::Buffer,
        gate_weight: metal::Buffer,
        gate_scale: metal::Buffer,
        up_weight: metal::Buffer,
        up_scale: metal::Buffer,
        down_weight: metal::Buffer,
        down_scale: metal::Buffer,
        route_weights: metal::Buffer,
        positions: metal::Buffer,
        scatter_shape: metal::Buffer,
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
    let gate_shape_buffer = device.new_buffer_with_data(
        (&gate_shape as *const GemvShape).cast(),
        std::mem::size_of::<GemvShape>() as u64,
        shared,
    );
    let down_shape_buffer = device.new_buffer_with_data(
        (&down_shape as *const GemvShape).cast(),
        std::mem::size_of::<GemvShape>() as u64,
        shared,
    );
    let hidden_count = (BATCH * INTERMEDIATE) as u32;
    let hidden_count_buffer = device.new_buffer_with_data(
        (&hidden_count as *const u32).cast(),
        std::mem::size_of::<u32>() as u64,
        shared,
    );
    let decode_lut = (0_u16..=255)
        .map(|bits| decode_f8_e4m3fn(bits as u8))
        .collect::<Vec<_>>();
    let lut_buffer = device.new_buffer_with_data(
        decode_lut.as_ptr().cast(),
        std::mem::size_of_val(decode_lut.as_slice()) as u64,
        shared,
    );
    struct DynamicRouterBuffers {
        weight: metal::Buffer,
        input: metal::Buffer,
        output: metal::Buffer,
        shape: metal::Buffer,
    }
    let dynamic_router = dynamic_router_source.as_ref().map(|(weight, _)| {
        let shape = GemvShape {
            rows: 256,
            columns: HIDDEN as u32,
            block_rows: 1,
            block_columns: 1,
        };
        DynamicRouterBuffers {
            weight: device.new_buffer_with_data(
                weight.as_ptr().cast(),
                weight.len() as u64,
                shared,
            ),
            input: device.new_buffer_with_data(
                input.as_ptr().cast(),
                std::mem::size_of_val(input.as_slice()) as u64,
                shared,
            ),
            output: device.new_buffer((BATCH * 256 * 4) as u64, shared),
            shape: device.new_buffer_with_data(
                (&shape as *const GemvShape).cast(),
                std::mem::size_of::<GemvShape>() as u64,
                shared,
            ),
        }
    });

    let mut experts = Vec::with_capacity(manifest.experts.len());
    let mut packed_gate_weights = Vec::new();
    let mut packed_gate_scales = Vec::new();
    let mut packed_up_weights = Vec::new();
    let mut packed_up_scales = Vec::new();
    let mut packed_down_weights = Vec::new();
    let mut packed_down_scales = Vec::new();
    let mut logical_source_bytes = 0_u64;
    let mut expert_position_counts = BTreeMap::new();
    for entry in &manifest.experts {
        let open_tensor_file = |key: &str| -> Result<MappedSafetensors, String> {
            let name = entry
                .tensor_files
                .get(key)
                .ok_or_else(|| format!("expert {} lacks {key} authority", entry.expert))?;
            MappedSafetensors::open(&resolve_artifact(name)?)
        };
        let gate_weight_mapping = open_tensor_file("gate_weight")?;
        let gate_scale_mapping = open_tensor_file("gate_scale")?;
        let up_weight_mapping = open_tensor_file("up_weight")?;
        let up_scale_mapping = open_tensor_file("up_scale")?;
        let down_weight_mapping = open_tensor_file("down_weight")?;
        let down_scale_mapping = open_tensor_file("down_scale")?;
        let gate_name = format!("{}.gate_proj.weight", entry.prefix);
        let up_name = format!("{}.up_proj.weight", entry.prefix);
        let down_name = format!("{}.down_proj.weight", entry.prefix);
        let gate = validate_fp8_views(
            gate_weight_mapping.tensor(&gate_name)?,
            gate_scale_mapping.tensor(&format!("{gate_name}_scale_inv"))?,
            &input[..HIDDEN],
        )?;
        let up = validate_fp8_views(
            up_weight_mapping.tensor(&up_name)?,
            up_scale_mapping.tensor(&format!("{up_name}_scale_inv"))?,
            &input[..HIDDEN],
        )?;
        let down = validate_fp8_views(
            down_weight_mapping.tensor(&down_name)?,
            down_scale_mapping.tensor(&format!("{down_name}_scale_inv"))?,
            &vec![0.0; INTERMEDIATE],
        )?;
        if gate.rows != INTERMEDIATE
            || gate.columns != HIDDEN
            || up.rows != INTERMEDIATE
            || up.columns != HIDDEN
            || down.rows != HIDDEN
            || down.columns != INTERMEDIATE
        {
            return Err(format!("expert {} shape mismatch", entry.expert));
        }
        for bytes in [
            gate.weight.metadata.data_bytes,
            gate.scale.metadata.data_bytes,
            up.weight.metadata.data_bytes,
            up.scale.metadata.data_bytes,
            down.weight.metadata.data_bytes,
            down.scale.metadata.data_bytes,
        ] {
            logical_source_bytes = logical_source_bytes
                .checked_add(bytes)
                .ok_or("MoE logical source bytes overflow")?;
        }
        packed_gate_weights.extend_from_slice(gate.weight.bytes);
        packed_gate_scales.extend_from_slice(gate.scale.bytes);
        packed_up_weights.extend_from_slice(up.weight.bytes);
        packed_up_scales.extend_from_slice(up.scale.bytes);
        packed_down_weights.extend_from_slice(down.weight.bytes);
        packed_down_scales.extend_from_slice(down.scale.bytes);
        let make_buffer = |bytes: &[u8]| {
            device.new_buffer_with_data(bytes.as_ptr().cast(), bytes.len() as u64, shared)
        };
        let mut gathered = vec![0.0_f32; BATCH * HIDDEN];
        for (local, position) in entry.positions.iter().enumerate() {
            let source = &input[*position as usize * HIDDEN..(*position as usize + 1) * HIDDEN];
            gathered[local * HIDDEN..(local + 1) * HIDDEN].copy_from_slice(source);
        }
        let mut route_weights = vec![0.0_f32; BATCH];
        route_weights[..entry.route_weights.len()].copy_from_slice(&entry.route_weights);
        let mut positions = vec![0_u32; BATCH];
        positions[..entry.positions.len()].copy_from_slice(&entry.positions);
        let scatter_shape = ScatterShape {
            count: entry.positions.len() as u32,
            width: HIDDEN as u32,
        };
        experts.push(ExpertBuffers {
            expert: entry.expert,
            count: entry.positions.len(),
            input: device.new_buffer_with_data(
                gathered.as_ptr().cast(),
                std::mem::size_of_val(gathered.as_slice()) as u64,
                shared,
            ),
            gate_weight: make_buffer(gate.weight.bytes),
            gate_scale: make_buffer(gate.scale.bytes),
            up_weight: make_buffer(up.weight.bytes),
            up_scale: make_buffer(up.scale.bytes),
            down_weight: make_buffer(down.weight.bytes),
            down_scale: make_buffer(down.scale.bytes),
            route_weights: device.new_buffer_with_data(
                route_weights.as_ptr().cast(),
                std::mem::size_of_val(route_weights.as_slice()) as u64,
                shared,
            ),
            positions: device.new_buffer_with_data(
                positions.as_ptr().cast(),
                std::mem::size_of_val(positions.as_slice()) as u64,
                shared,
            ),
            scatter_shape: device.new_buffer_with_data(
                (&scatter_shape as *const ScatterShape).cast(),
                std::mem::size_of::<ScatterShape>() as u64,
                shared,
            ),
        });
        expert_position_counts.insert(entry.expert, entry.positions.len());
    }

    struct UnionBuffers {
        gate_weight: metal::Buffer,
        gate_scale: metal::Buffer,
        up_weight: metal::Buffer,
        up_scale: metal::Buffer,
        down_weight: metal::Buffer,
        down_scale: metal::Buffer,
        input: metal::Buffer,
        gate_output: metal::Buffer,
        up_output: metal::Buffer,
        hidden_output: metal::Buffer,
        expert_output: metal::Buffer,
        hidden_count: metal::Buffer,
    }
    let union_buffers = if mode.union_parallel {
        let make_buffer = |bytes: &[u8]| {
            device.new_buffer_with_data(bytes.as_ptr().cast(), bytes.len() as u64, shared)
        };
        let union_hidden_count = (experts.len() * BATCH * INTERMEDIATE) as u32;
        Some(UnionBuffers {
            gate_weight: make_buffer(&packed_gate_weights),
            gate_scale: make_buffer(&packed_gate_scales),
            up_weight: make_buffer(&packed_up_weights),
            up_scale: make_buffer(&packed_up_scales),
            down_weight: make_buffer(&packed_down_weights),
            down_scale: make_buffer(&packed_down_scales),
            input: device.new_buffer((experts.len() * BATCH * HIDDEN * 4) as u64, shared),
            gate_output: device
                .new_buffer((experts.len() * BATCH * INTERMEDIATE * 4) as u64, shared),
            up_output: device.new_buffer((experts.len() * BATCH * INTERMEDIATE * 4) as u64, shared),
            hidden_output: device
                .new_buffer((experts.len() * BATCH * INTERMEDIATE * 4) as u64, shared),
            expert_output: device.new_buffer((experts.len() * BATCH * HIDDEN * 4) as u64, shared),
            hidden_count: device.new_buffer_with_data(
                (&union_hidden_count as *const u32).cast(),
                std::mem::size_of::<u32>() as u64,
                shared,
            ),
        })
    } else {
        None
    };

    let gate_output = device.new_buffer((BATCH * INTERMEDIATE * 4) as u64, shared);
    let up_output = device.new_buffer((BATCH * INTERMEDIATE * 4) as u64, shared);
    let hidden_output = device.new_buffer((BATCH * INTERMEDIATE * 4) as u64, shared);
    let expert_output = device.new_buffer((BATCH * HIDDEN * 4) as u64, shared);
    let block_output = device.new_buffer((BATCH * HIDDEN * 4) as u64, shared);
    let dynamic_maximum_route_weight_error = std::cell::Cell::new(0.0_f32);
    let dynamic_minimum_boundary_margin = std::cell::Cell::new(f32::INFINITY);
    let dispatch = || -> Result<f64, String> {
        let start = Instant::now();
        let dynamic_counts = if let (Some(router), Some(pipeline), Some((_, correction))) = (
            dynamic_router.as_ref(),
            router_pipeline.as_ref(),
            dynamic_router_source.as_ref(),
        ) {
            let router_command = queue.new_command_buffer();
            let router_encoder = router_command.new_compute_command_encoder();
            router_encoder.set_compute_pipeline_state(pipeline);
            router_encoder.set_buffer(0, Some(&router.weight), 0);
            router_encoder.set_buffer(1, Some(&router.input), 0);
            router_encoder.set_buffer(2, Some(&router.output), 0);
            router_encoder.set_buffer(3, Some(&router.shape), 0);
            router_encoder.set_threadgroup_memory_length(0, LANES * BATCH as u64 * 4);
            router_encoder.dispatch_thread_groups(
                MTLSize {
                    width: 256,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: LANES,
                    height: 1,
                    depth: 1,
                },
            );
            router_encoder.end_encoding();
            router_command.commit();
            router_command.wait_until_completed();
            if router_command.status() != MTLCommandBufferStatus::Completed {
                return Err("dynamic router command failed".to_owned());
            }
            // SAFETY: the completed shared router output is exactly batch * 256 F32 logits.
            let logits = unsafe {
                std::slice::from_raw_parts(router.output.contents().cast::<f32>(), BATCH * 256)
            };
            let routes = select_noaux_tc_routes(logits, correction, BATCH, 256, 8)?;
            dynamic_minimum_boundary_margin.set(routes.minimum_boundary_margin);
            let mut schedule = BTreeMap::<u32, Vec<(u32, f32)>>::new();
            let mut maximum_route_weight_error = 0.0_f32;
            for position in 0..BATCH {
                let selected_set = routes.selected[position]
                    .iter()
                    .copied()
                    .collect::<BTreeSet<_>>();
                let reference_set = manifest.selected_experts_by_position[position]
                    .iter()
                    .copied()
                    .collect::<BTreeSet<_>>();
                if selected_set != reference_set {
                    return Err(format!(
                        "dynamic router selected set mismatch at position {position}"
                    ));
                }
                for (&expert_id, &weight) in routes.selected[position]
                    .iter()
                    .zip(&routes.weights[position])
                {
                    let reference_slot = manifest.selected_experts_by_position[position]
                        .iter()
                        .position(|value| *value == expert_id)
                        .ok_or("dynamic route expert absent from oracle")?;
                    let route_weight_error = (weight
                        - manifest.route_weights_by_position[position][reference_slot])
                        .abs();
                    maximum_route_weight_error = maximum_route_weight_error.max(route_weight_error);
                    if route_weight_error > 2.0e-6 {
                        return Err("dynamic route weight parity failed".to_owned());
                    }
                    schedule
                        .entry(expert_id)
                        .or_default()
                        .push((position as u32, weight));
                }
            }
            dynamic_maximum_route_weight_error.set(maximum_route_weight_error);
            if schedule.len() != experts.len()
                || schedule.keys().copied().collect::<BTreeSet<_>>() != expert_ids
            {
                return Err("dynamic router selected unknown expert union".to_owned());
            }
            let mut counts = BTreeMap::new();
            for (expert_index, expert) in experts.iter().enumerate() {
                let placements = schedule
                    .get(&expert.expert)
                    .ok_or("dynamic expert schedule absent")?;
                let count = placements.len();
                let mut gathered = vec![0.0_f32; BATCH * HIDDEN];
                let mut weights = [0.0_f32; BATCH];
                let mut positions = [0_u32; BATCH];
                for (local, &(position, weight)) in placements.iter().enumerate() {
                    gathered[local * HIDDEN..(local + 1) * HIDDEN].copy_from_slice(
                        &input[position as usize * HIDDEN..(position as usize + 1) * HIDDEN],
                    );
                    weights[local] = weight;
                    positions[local] = position;
                }
                let scatter_shape = ScatterShape {
                    count: count as u32,
                    width: HIDDEN as u32,
                };
                // SAFETY: each shared buffer was allocated to the exact copied slice size.
                unsafe {
                    let gathered_destination = if let Some(union) = union_buffers.as_ref() {
                        union
                            .input
                            .contents()
                            .cast::<f32>()
                            .add(expert_index * BATCH * HIDDEN)
                    } else {
                        expert.input.contents().cast::<f32>()
                    };
                    std::ptr::copy_nonoverlapping(
                        gathered.as_ptr(),
                        gathered_destination,
                        gathered.len(),
                    );
                    std::ptr::copy_nonoverlapping(
                        weights.as_ptr(),
                        expert.route_weights.contents().cast::<f32>(),
                        weights.len(),
                    );
                    std::ptr::copy_nonoverlapping(
                        positions.as_ptr(),
                        expert.positions.contents().cast::<u32>(),
                        positions.len(),
                    );
                    std::ptr::copy_nonoverlapping(
                        &scatter_shape,
                        expert.scatter_shape.contents().cast::<ScatterShape>(),
                        1,
                    );
                }
                counts.insert(expert.expert, count);
            }
            Some(counts)
        } else {
            None
        };
        let command = queue.new_command_buffer();
        let blit = command.new_blit_command_encoder();
        blit.fill_buffer(
            &block_output,
            NSRange::new(0, (BATCH * HIDDEN * std::mem::size_of::<f32>()) as u64),
            0,
        );
        blit.end_encoding();
        let encoder = command.new_compute_command_encoder();
        if let (Some(union), Some(union_pipeline)) =
            (union_buffers.as_ref(), union_expert_pipeline.as_ref())
        {
            encoder.set_compute_pipeline_state(union_pipeline);
            encoder.set_buffer(0, Some(&union.gate_weight), 0);
            encoder.set_buffer(1, Some(&union.gate_scale), 0);
            encoder.set_buffer(2, Some(&union.input), 0);
            encoder.set_buffer(3, Some(&union.gate_output), 0);
            encoder.set_buffer(4, Some(&gate_shape_buffer), 0);
            encoder.set_buffer(5, Some(&lut_buffer), 0);
            encoder.set_threadgroup_memory_length(0, LANES * BATCH as u64 * 4);
            encoder.dispatch_thread_groups(
                MTLSize {
                    width: INTERMEDIATE as u64,
                    height: experts.len() as u64,
                    depth: 1,
                },
                MTLSize {
                    width: LANES,
                    height: 1,
                    depth: 1,
                },
            );
            encoder.set_buffer(0, Some(&union.up_weight), 0);
            encoder.set_buffer(1, Some(&union.up_scale), 0);
            encoder.set_buffer(3, Some(&union.up_output), 0);
            encoder.dispatch_thread_groups(
                MTLSize {
                    width: INTERMEDIATE as u64,
                    height: experts.len() as u64,
                    depth: 1,
                },
                MTLSize {
                    width: LANES,
                    height: 1,
                    depth: 1,
                },
            );
            encoder.set_compute_pipeline_state(&swiglu_pipeline);
            encoder.set_buffer(0, Some(&union.gate_output), 0);
            encoder.set_buffer(1, Some(&union.up_output), 0);
            encoder.set_buffer(2, Some(&union.hidden_output), 0);
            encoder.set_buffer(3, Some(&union.hidden_count), 0);
            encoder.dispatch_threads(
                MTLSize {
                    width: (experts.len() * BATCH * INTERMEDIATE) as u64,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: 256,
                    height: 1,
                    depth: 1,
                },
            );
            encoder.set_compute_pipeline_state(union_pipeline);
            encoder.set_buffer(0, Some(&union.down_weight), 0);
            encoder.set_buffer(1, Some(&union.down_scale), 0);
            encoder.set_buffer(2, Some(&union.hidden_output), 0);
            encoder.set_buffer(3, Some(&union.expert_output), 0);
            encoder.set_buffer(4, Some(&down_shape_buffer), 0);
            encoder.set_buffer(5, Some(&lut_buffer), 0);
            encoder.set_threadgroup_memory_length(0, LANES * BATCH as u64 * 4);
            encoder.dispatch_thread_groups(
                MTLSize {
                    width: HIDDEN as u64,
                    height: experts.len() as u64,
                    depth: 1,
                },
                MTLSize {
                    width: LANES,
                    height: 1,
                    depth: 1,
                },
            );
            for (expert_index, expert) in experts.iter().enumerate() {
                let expert_count = dynamic_counts
                    .as_ref()
                    .and_then(|counts| counts.get(&expert.expert))
                    .copied()
                    .ok_or("union-parallel expert count absent")?;
                encoder.set_compute_pipeline_state(&scatter_pipeline);
                encoder.set_buffer(
                    0,
                    Some(&union.expert_output),
                    (expert_index * BATCH * HIDDEN * 4) as u64,
                );
                encoder.set_buffer(1, Some(&expert.route_weights), 0);
                encoder.set_buffer(2, Some(&expert.positions), 0);
                encoder.set_buffer(3, Some(&block_output), 0);
                encoder.set_buffer(4, Some(&expert.scatter_shape), 0);
                encoder.dispatch_threads(
                    MTLSize {
                        width: (expert_count * HIDDEN) as u64,
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
        } else {
            for expert in &experts {
                let expert_count = dynamic_counts
                    .as_ref()
                    .and_then(|counts| counts.get(&expert.expert))
                    .copied()
                    .unwrap_or(expert.count);
                if let Some(fused_pipeline) = fused_gate_up_pipeline.as_ref() {
                    encoder.set_compute_pipeline_state(fused_pipeline);
                    encoder.set_buffer(0, Some(&expert.gate_weight), 0);
                    encoder.set_buffer(1, Some(&expert.gate_scale), 0);
                    encoder.set_buffer(2, Some(&expert.up_weight), 0);
                    encoder.set_buffer(3, Some(&expert.up_scale), 0);
                    encoder.set_buffer(4, Some(&expert.input), 0);
                    encoder.set_buffer(5, Some(&gate_output), 0);
                    encoder.set_buffer(6, Some(&up_output), 0);
                    encoder.set_buffer(7, Some(&gate_shape_buffer), 0);
                    encoder.set_buffer(8, Some(&lut_buffer), 0);
                    encoder.set_threadgroup_memory_length(0, LANES * BATCH as u64 * 4);
                    encoder.dispatch_thread_groups(
                        MTLSize {
                            width: (INTERMEDIATE * 2) as u64,
                            height: 1,
                            depth: 1,
                        },
                        MTLSize {
                            width: LANES,
                            height: 1,
                            depth: 1,
                        },
                    );
                } else {
                    let projection_pipeline = simdgroup_matrix_pipeline
                        .as_ref()
                        .unwrap_or(&expert_pipeline);
                    let projection_threads = if mode.simdgroup_matrix { 32 } else { LANES };
                    let projection_groups = if mode.simdgroup_matrix {
                        (INTERMEDIATE / 8) as u64
                    } else {
                        INTERMEDIATE as u64
                    };
                    let projection_threadgroup_bytes = if mode.simdgroup_matrix {
                        64 * 4
                    } else {
                        LANES * BATCH as u64 * 4
                    };
                    encoder.set_compute_pipeline_state(projection_pipeline);
                    encoder.set_buffer(0, Some(&expert.gate_weight), 0);
                    encoder.set_buffer(1, Some(&expert.gate_scale), 0);
                    encoder.set_buffer(2, Some(&expert.input), 0);
                    encoder.set_buffer(3, Some(&gate_output), 0);
                    encoder.set_buffer(4, Some(&gate_shape_buffer), 0);
                    encoder.set_buffer(5, Some(&lut_buffer), 0);
                    encoder.set_threadgroup_memory_length(0, projection_threadgroup_bytes);
                    encoder.dispatch_thread_groups(
                        MTLSize {
                            width: projection_groups,
                            height: 1,
                            depth: 1,
                        },
                        MTLSize {
                            width: projection_threads,
                            height: 1,
                            depth: 1,
                        },
                    );
                    encoder.set_buffer(0, Some(&expert.up_weight), 0);
                    encoder.set_buffer(1, Some(&expert.up_scale), 0);
                    encoder.set_buffer(3, Some(&up_output), 0);
                    encoder.dispatch_thread_groups(
                        MTLSize {
                            width: projection_groups,
                            height: 1,
                            depth: 1,
                        },
                        MTLSize {
                            width: projection_threads,
                            height: 1,
                            depth: 1,
                        },
                    );
                }
                encoder.set_compute_pipeline_state(&swiglu_pipeline);
                encoder.set_buffer(0, Some(&gate_output), 0);
                encoder.set_buffer(1, Some(&up_output), 0);
                encoder.set_buffer(2, Some(&hidden_output), 0);
                encoder.set_buffer(3, Some(&hidden_count_buffer), 0);
                encoder.dispatch_threads(
                    MTLSize {
                        width: (BATCH * INTERMEDIATE) as u64,
                        height: 1,
                        depth: 1,
                    },
                    MTLSize {
                        width: 256,
                        height: 1,
                        depth: 1,
                    },
                );
                let down_pipeline = simdgroup_matrix_pipeline
                    .as_ref()
                    .unwrap_or(&expert_pipeline);
                encoder.set_compute_pipeline_state(down_pipeline);
                encoder.set_buffer(0, Some(&expert.down_weight), 0);
                encoder.set_buffer(1, Some(&expert.down_scale), 0);
                encoder.set_buffer(2, Some(&hidden_output), 0);
                encoder.set_buffer(3, Some(&expert_output), 0);
                encoder.set_buffer(4, Some(&down_shape_buffer), 0);
                encoder.set_buffer(5, Some(&lut_buffer), 0);
                encoder.set_threadgroup_memory_length(
                    0,
                    if mode.simdgroup_matrix {
                        64 * 4
                    } else {
                        LANES * BATCH as u64 * 4
                    },
                );
                encoder.dispatch_thread_groups(
                    MTLSize {
                        width: if mode.simdgroup_matrix {
                            (HIDDEN / 8) as u64
                        } else {
                            HIDDEN as u64
                        },
                        height: 1,
                        depth: 1,
                    },
                    MTLSize {
                        width: if mode.simdgroup_matrix { 32 } else { LANES },
                        height: 1,
                        depth: 1,
                    },
                );
                encoder.set_compute_pipeline_state(&scatter_pipeline);
                encoder.set_buffer(0, Some(&expert_output), 0);
                encoder.set_buffer(1, Some(&expert.route_weights), 0);
                encoder.set_buffer(2, Some(&expert.positions), 0);
                encoder.set_buffer(3, Some(&block_output), 0);
                encoder.set_buffer(4, Some(&expert.scatter_shape), 0);
                encoder.dispatch_threads(
                    MTLSize {
                        width: (expert_count * HIDDEN) as u64,
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
        }
        encoder.end_encoding();
        command.commit();
        command.wait_until_completed();
        let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
        if command.status() != MTLCommandBufferStatus::Completed {
            return Err(format!("Metal MoE command failed: {:?}", command.status()));
        }
        Ok(elapsed_ms)
    };

    let cold_wall_ms = dispatch()?;
    for _ in 0..WARMUPS {
        dispatch()?;
    }
    let mut wall_ms = Vec::with_capacity(MEASUREMENTS);
    for _ in 0..MEASUREMENTS {
        wall_ms.push(dispatch()?);
    }
    // SAFETY: the completed shared block output is exactly batch * hidden F32 values.
    let output = unsafe {
        std::slice::from_raw_parts(block_output.contents().cast::<f32>(), BATCH * HIDDEN).to_vec()
    };
    if output.iter().any(|value| !value.is_finite()) {
        return Err("Metal MoE block produced non-finite output".to_owned());
    }
    let mut squared_error = 0.0_f64;
    let mut squared_reference = 0.0_f64;
    let mut maximum_absolute_error = 0.0_f32;
    for (&candidate, &expected) in output.iter().zip(&reference) {
        let difference = candidate - expected;
        squared_error += f64::from(difference) * f64::from(difference);
        squared_reference += f64::from(expected) * f64::from(expected);
        maximum_absolute_error = maximum_absolute_error.max(difference.abs());
    }
    if squared_reference == 0.0 {
        return Err("MoE reference has zero L2 norm".to_owned());
    }
    let relative_l2 = (squared_error / squared_reference).sqrt();
    if relative_l2 > 4.0e-5 || maximum_absolute_error > 3.0e-8 {
        return Err(format!(
            "MoE parity failed: relative L2 {relative_l2}, max abs {maximum_absolute_error}"
        ));
    }
    let output_bytes = output
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect::<Vec<_>>();
    write_create_new(output_path, &output_bytes)?;
    let mut ordered = wall_ms.clone();
    ordered.sort_by(f64::total_cmp);
    let percentile = |fraction: f64| -> f64 {
        ordered[((ordered.len() - 1) as f64 * fraction).round() as usize]
    };
    let wall_p10_ms = percentile(0.10);
    let wall_median_ms = percentile(0.50);
    let wall_p90_ms = percentile(0.90);
    let median_timing_gate_passed = wall_median_ms
        <= if mode.union_parallel {
            14.0
        } else if mode.fused_gate_up {
            15.5
        } else if mode.simdgroup_matrix {
            14.0
        } else if dynamic_router_source.is_some() {
            20.0
        } else {
            25.0
        };
    let router_source_bytes = dynamic_router_source
        .as_ref()
        .map_or(0_u64, |(weight, correction)| {
            (weight.len() + correction.len() * 4) as u64
        });
    let logical_source_and_io_bytes = logical_source_bytes
        .checked_add(router_source_bytes)
        .and_then(|value| value.checked_add(input_bytes.len() as u64))
        .and_then(|value| value.checked_add(output_bytes.len() as u64))
        .ok_or("MoE logical byte count overflow")?;
    let router_resident_bytes = if dynamic_router_source.is_some() {
        router_source_bytes + (BATCH * HIDDEN * 4 + BATCH * 256 * 4) as u64
    } else {
        0
    };
    let union_resident_bytes = if mode.union_parallel {
        logical_source_bytes + (experts.len() * BATCH * (HIDDEN * 2 + INTERMEDIATE * 3) * 4) as u64
    } else {
        0
    };
    let resident_buffer_bytes = logical_source_bytes
        .checked_add(router_resident_bytes)
        .and_then(|value| value.checked_add(union_resident_bytes))
        .and_then(|value| value.checked_add((manifest.experts.len() * BATCH * HIDDEN * 4) as u64))
        .and_then(|value| value.checked_add((BATCH * INTERMEDIATE * 4 * 3) as u64))
        .and_then(|value| value.checked_add((BATCH * HIDDEN * 4 * 2) as u64))
        .ok_or("MoE resident byte count overflow")?;
    let padding_overhead_fraction = (manifest.padded_expert_positions
        - manifest.real_expert_positions) as f64
        / manifest.real_expert_positions as f64;
    Ok(MetalFp8MoeReport {
        schema_version: 1,
        semantic: if mode.union_parallel {
            "mimo_layer43_native_union_parallel_source_fp8_moe_block".to_owned()
        } else if mode.fused_gate_up {
            "mimo_layer43_native_fused_gate_up_source_fp8_moe_block".to_owned()
        } else if mode.simdgroup_matrix {
            "mimo_layer43_native_simdgroup_matrix_source_fp8_moe_block".to_owned()
        } else if dynamic_router_source.is_some() {
            "mimo_layer43_native_dynamic_source_fp8_moe_block".to_owned()
        } else {
            manifest.semantic
        },
        manifest_sha256: sha256_hex(&manifest_bytes),
        input_sha256: sha256_hex(&input_bytes),
        reference_sha256: sha256_hex(&reference_bytes),
        output_sha256: sha256_hex(&output_bytes),
        output_first8: output.iter().copied().take(8).collect(),
        device: device.name().to_owned(),
        kernel_file: kernel_path
            .file_name()
            .and_then(|name| name.to_str())
            .ok_or("kernel file name is not UTF-8")?
            .to_owned(),
        expert_kernel: EXPERT_KERNEL,
        scatter_kernel: SCATTER_KERNEL,
        layer: manifest.layer,
        batch_size: manifest.batch_size,
        top_k: manifest.top_k,
        unique_experts: manifest.experts.len(),
        expert_position_counts,
        real_expert_positions: manifest.real_expert_positions,
        padded_expert_positions: manifest.padded_expert_positions,
        padding_overhead_fraction,
        relative_l2,
        maximum_absolute_error,
        router_file_sha256: dynamic_router_source.as_ref().map(|_| ROUTER_SHA256),
        maximum_route_weight_absolute_error: dynamic_router_source
            .as_ref()
            .map(|_| dynamic_maximum_route_weight_error.get()),
        minimum_topk_boundary_margin: dynamic_router_source
            .as_ref()
            .map(|_| dynamic_minimum_boundary_margin.get()),
        scatter_fixture_maximum_absolute_error,
        union_kernel_fixture_maximum_absolute_error,
        fused_gate_up_fixture_maximum_absolute_error,
        simdgroup_matrix_fixture_maximum_absolute_error,
        compile_ms,
        cold_wall_ms,
        warmups: WARMUPS,
        measurements: MEASUREMENTS,
        wall_ms,
        wall_p10_ms,
        wall_median_ms,
        wall_p90_ms,
        median_timing_gate_passed,
        routed_only_accepted_tps_diagnostic: BATCH as f64 * 1000.0 / (wall_median_ms * 47.0),
        logical_source_and_io_bytes,
        resident_buffer_bytes,
        batch_concurrency: 1,
        accepted_tokens: 0,
        accepted_per_verification: 8,
        expert_union_factor: manifest.experts.len() as f64 / manifest.top_k as f64,
        cache_state: "selected source artifacts OS-cache state uncontrolled; all model and application buffers resident before timed series",
        scheduling_limitation: if mode.union_parallel {
            "exact fixed layer43 input; dynamic native routes; selected expert union preloaded and packed"
                .to_owned()
        } else if mode.fused_gate_up {
            "exact fixed layer43 input; dynamic native routes; selected expert union preloaded; gate/up fused per expert"
                .to_owned()
        } else if mode.simdgroup_matrix {
            "exact fixed layer43 input; dynamic native routes; selected expert union preloaded; SIMD-group matrix projections"
                .to_owned()
        } else if dynamic_router_source.is_some() {
            "exact fixed layer43 input; dynamic native routes; selected expert union preloaded"
                .to_owned()
        } else {
            manifest.scheduling
        },
        implementation: if mode.union_parallel {
            "rust_owned_metal_dynamic_router_union_parallel_source_fp8_moe_block"
        } else if mode.fused_gate_up {
            "rust_owned_metal_dynamic_router_fused_gate_up_source_fp8_moe_block"
        } else if mode.simdgroup_matrix {
            "rust_owned_metal_dynamic_router_simdgroup_matrix_source_fp8_moe_block"
        } else if dynamic_router_source.is_some() {
            "rust_owned_metal_dynamic_router_source_fp8_moe_block"
        } else {
            "rust_owned_metal_fixture_scheduled_source_fp8_moe_block"
        },
    })
}

#[cfg(target_os = "macos")]
pub fn run_metal_noaux_tc_router(
    router_path: &Path,
    kernel_path: &Path,
    input_path: &Path,
    reference_manifest_path: &Path,
    output_path: &Path,
) -> Result<MetalRouterReport, String> {
    use metal::{CompileOptions, Device, MTLCommandBufferStatus, MTLResourceOptions, MTLSize};

    const ROUTER_SHA256: &str = "12c1579d28b78dd69ec9342eb9d1f378efc5aa3c2f2a28b5ec73578e6a8bbcdd";
    const REFERENCE_SHA256: &str =
        "a2b30be7ab767c754fd4680887420246dcd28d314bff242b144f664a8ff12470";
    const WEIGHT_NAME: &str = "model.layers.43.mlp.gate.weight";
    const BIAS_NAME: &str = "model.layers.43.mlp.gate.e_score_correction_bias";
    const KERNEL_FUNCTION: &str = "f32_gemm8_shared_weight";
    const BATCH: usize = 8;
    const ROWS: usize = 256;
    const COLUMNS: usize = 4096;
    const TOP_K: usize = 8;
    const LANES: u64 = 64;
    const THREADGROUP_MEMORY_BYTES: u64 = LANES * BATCH as u64 * 4;
    const WARMUPS: usize = 5;
    const MEASUREMENTS: usize = 30;

    if output_path.exists() {
        return Err(format!("refusing to overwrite {}", output_path.display()));
    }
    let mut router_file =
        File::open(router_path).map_err(|error| format!("{}: {error}", router_path.display()))?;
    let router_file_sha256 = sha256_reader(&mut router_file)?;
    if router_file_sha256 != ROUTER_SHA256 {
        return Err("layer-43 router artifact SHA-256 mismatch".to_owned());
    }
    let mapped = MappedSafetensors::open(router_path)?;
    let weight = mapped.tensor(WEIGHT_NAME)?;
    let bias = mapped.tensor(BIAS_NAME)?;
    if weight.metadata.dtype != "F32"
        || weight.metadata.shape != [ROWS as u64, COLUMNS as u64]
        || weight.bytes.len() != ROWS * COLUMNS * 4
        || bias.metadata.dtype != "F32"
        || bias.metadata.shape != [ROWS as u64]
        || bias.bytes.len() != ROWS * 4
    {
        return Err("layer-43 router tensor layout mismatch".to_owned());
    }
    let correction = bias
        .bytes
        .chunks_exact(4)
        .map(|bytes| f32::from_le_bytes(bytes.try_into().expect("four-byte bias")))
        .collect::<Vec<_>>();
    if correction.iter().any(|value| !value.is_finite()) {
        return Err("router correction bias is non-finite".to_owned());
    }
    let (input_bytes, input) = read_f32_file(input_path, Some(BATCH * COLUMNS))?;
    let reference_bytes = fs::read(reference_manifest_path)
        .map_err(|error| format!("{}: {error}", reference_manifest_path.display()))?;
    let reference_manifest_sha256 = sha256_hex(&reference_bytes);
    if reference_manifest_sha256 != REFERENCE_SHA256 {
        return Err("router reference manifest SHA-256 mismatch".to_owned());
    }
    let unique: UniqueJson = serde_json::from_slice(&reference_bytes)
        .map_err(|error| format!("router reference manifest: {error}"))?;
    let reference: MetalMoeManifest =
        serde_json::from_value(unique.0).map_err(|error| format!("router reference: {error}"))?;
    if reference.schema_version != 1
        || reference.layer != 43
        || reference.batch_size != BATCH
        || reference.top_k != TOP_K
        || reference.input_sha256 != sha256_hex(&input_bytes)
        || reference.selected_experts_by_position.len() != BATCH
        || reference.route_weights_by_position.len() != BATCH
    {
        return Err("router reference identity mismatch".to_owned());
    }
    let kernel_source = fs::read_to_string(kernel_path)
        .map_err(|error| format!("{}: {error}", kernel_path.display()))?;
    if !kernel_source.contains(&format!("kernel void {KERNEL_FUNCTION}")) {
        return Err(format!("kernel source lacks {KERNEL_FUNCTION}"));
    }
    let device = Device::system_default().ok_or("no Metal device is available")?;
    if device.max_threads_per_threadgroup().width < LANES {
        return Err("Metal device cannot dispatch 64-lane threadgroups".to_owned());
    }
    let options = CompileOptions::new();
    options.set_fast_math_enabled(false);
    let compile_start = Instant::now();
    let library = device
        .new_library_with_source(&kernel_source, &options)
        .map_err(|error| format!("Metal compilation failed: {error}"))?;
    let function = library
        .get_function(KERNEL_FUNCTION, None)
        .map_err(|error| format!("router kernel lookup: {error}"))?;
    let pipeline = device
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|error| format!("router pipeline: {error}"))?;
    let compile_ms = compile_start.elapsed().as_secs_f64() * 1000.0;
    let shared = MTLResourceOptions::StorageModeShared;
    let queue = device.new_command_queue();

    #[repr(C)]
    struct GemvShape {
        rows: u32,
        columns: u32,
        block_rows: u32,
        block_columns: u32,
    }
    let shape = GemvShape {
        rows: ROWS as u32,
        columns: COLUMNS as u32,
        block_rows: 1,
        block_columns: 1,
    };
    let shape_buffer = device.new_buffer_with_data(
        (&shape as *const GemvShape).cast(),
        std::mem::size_of::<GemvShape>() as u64,
        shared,
    );

    let fixture_weights = (0..ROWS * COLUMNS)
        .map(|index| ((index * 13 % 31) as f32 - 15.0) * 0.0005)
        .collect::<Vec<_>>();
    let fixture_input = (0..BATCH * COLUMNS)
        .map(|index| ((index * 17 % 29) as f32 - 14.0) * 0.001)
        .collect::<Vec<_>>();
    let fixture_expected = (0..BATCH)
        .flat_map(|position| {
            let weights = &fixture_weights;
            let inputs = &fixture_input;
            (0..ROWS).map(move |row| {
                let mut sum = 0.0_f32;
                for column in 0..COLUMNS {
                    sum += weights[row * COLUMNS + column] * inputs[position * COLUMNS + column];
                }
                sum
            })
        })
        .collect::<Vec<_>>();
    let fixture_weight_buffer = device.new_buffer_with_data(
        fixture_weights.as_ptr().cast(),
        std::mem::size_of_val(fixture_weights.as_slice()) as u64,
        shared,
    );
    let fixture_input_buffer = device.new_buffer_with_data(
        fixture_input.as_ptr().cast(),
        std::mem::size_of_val(fixture_input.as_slice()) as u64,
        shared,
    );
    let fixture_output_buffer = device.new_buffer((BATCH * ROWS * 4) as u64, shared);
    let fixture_command = queue.new_command_buffer();
    let fixture_encoder = fixture_command.new_compute_command_encoder();
    fixture_encoder.set_compute_pipeline_state(&pipeline);
    fixture_encoder.set_buffer(0, Some(&fixture_weight_buffer), 0);
    fixture_encoder.set_buffer(1, Some(&fixture_input_buffer), 0);
    fixture_encoder.set_buffer(2, Some(&fixture_output_buffer), 0);
    fixture_encoder.set_buffer(3, Some(&shape_buffer), 0);
    fixture_encoder.set_threadgroup_memory_length(0, THREADGROUP_MEMORY_BYTES);
    fixture_encoder.dispatch_thread_groups(
        MTLSize {
            width: ROWS as u64,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: LANES,
            height: 1,
            depth: 1,
        },
    );
    fixture_encoder.end_encoding();
    fixture_command.commit();
    fixture_command.wait_until_completed();
    if fixture_command.status() != MTLCommandBufferStatus::Completed {
        return Err("router kernel fixture command failed".to_owned());
    }
    // SAFETY: the completed shared fixture output is exactly batch * rows F32 values.
    let fixture_actual = unsafe {
        std::slice::from_raw_parts(fixture_output_buffer.contents().cast::<f32>(), BATCH * ROWS)
    };
    let kernel_fixture_maximum_absolute_error = fixture_actual
        .iter()
        .zip(&fixture_expected)
        .map(|(&actual, &expected)| (actual - expected).abs())
        .fold(0.0_f32, f32::max);
    if fixture_actual.iter().any(|value| !value.is_finite())
        || kernel_fixture_maximum_absolute_error > 2.0e-4
    {
        return Err(format!(
            "router kernel fixture failed: {kernel_fixture_maximum_absolute_error}"
        ));
    }

    let weight_buffer = device.new_buffer_with_data(
        weight.bytes.as_ptr().cast(),
        weight.bytes.len() as u64,
        shared,
    );
    let input_buffer = device.new_buffer_with_data(
        input.as_ptr().cast(),
        std::mem::size_of_val(input.as_slice()) as u64,
        shared,
    );
    let output_buffer = device.new_buffer((BATCH * ROWS * 4) as u64, shared);
    let dispatch = || -> Result<f64, String> {
        let start = Instant::now();
        let command = queue.new_command_buffer();
        let encoder = command.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(&weight_buffer), 0);
        encoder.set_buffer(1, Some(&input_buffer), 0);
        encoder.set_buffer(2, Some(&output_buffer), 0);
        encoder.set_buffer(3, Some(&shape_buffer), 0);
        encoder.set_threadgroup_memory_length(0, THREADGROUP_MEMORY_BYTES);
        encoder.dispatch_thread_groups(
            MTLSize {
                width: ROWS as u64,
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
        let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
        if command.status() != MTLCommandBufferStatus::Completed {
            return Err(format!("router command failed: {:?}", command.status()));
        }
        Ok(elapsed_ms)
    };
    let cold_wall_ms = dispatch()?;
    for _ in 0..WARMUPS {
        dispatch()?;
    }
    let mut wall_ms = Vec::with_capacity(MEASUREMENTS);
    for _ in 0..MEASUREMENTS {
        wall_ms.push(dispatch()?);
    }
    // SAFETY: the completed shared output is exactly batch * rows F32 logits.
    let logits = unsafe {
        std::slice::from_raw_parts(output_buffer.contents().cast::<f32>(), BATCH * ROWS).to_vec()
    };
    if logits.iter().any(|value| !value.is_finite()) {
        return Err("router logits are non-finite".to_owned());
    }
    let routes = select_noaux_tc_routes(&logits, &correction, BATCH, ROWS, TOP_K)?;
    let selected_experts_by_position = routes.selected;
    let route_weights_by_position = routes.weights;
    let mut maximum_route_weight_absolute_error = 0.0_f32;
    let mut union = BTreeSet::new();
    for position in 0..BATCH {
        let chosen = &selected_experts_by_position[position];
        let weights = &route_weights_by_position[position];
        let reference_ids = &reference.selected_experts_by_position[position];
        let reference_weights = &reference.route_weights_by_position[position];
        if reference_ids.len() != TOP_K || reference_weights.len() != TOP_K {
            return Err("router reference row mismatch".to_owned());
        }
        let candidate_set = chosen.iter().copied().collect::<BTreeSet<_>>();
        let reference_set = reference_ids.iter().copied().collect::<BTreeSet<_>>();
        if candidate_set != reference_set {
            return Err(format!(
                "router selected set mismatch at position {position}"
            ));
        }
        for (&index, &candidate_weight) in chosen.iter().zip(weights) {
            let reference_slot = reference_ids
                .iter()
                .position(|value| *value == index)
                .ok_or("router reference expert absent")?;
            maximum_route_weight_absolute_error = maximum_route_weight_absolute_error
                .max((candidate_weight - reference_weights[reference_slot]).abs());
            union.insert(index);
        }
    }
    if maximum_route_weight_absolute_error > 2.0e-6 {
        return Err(format!(
            "router weight parity failed: {maximum_route_weight_absolute_error}"
        ));
    }
    let artifact = NativeRouteArtifact {
        schema_version: 1,
        semantic: "mimo_layer43_native_noaux_tc_routes",
        layer: 43,
        batch_size: BATCH,
        top_k: TOP_K,
        selected_experts_by_position: selected_experts_by_position.clone(),
        route_weights_by_position: route_weights_by_position.clone(),
    };
    let output_bytes = serde_json::to_vec(&artifact).map_err(|error| error.to_string())?;
    write_create_new(output_path, &output_bytes)?;
    let mut ordered = wall_ms.clone();
    ordered.sort_by(f64::total_cmp);
    let percentile = |fraction: f64| -> f64 {
        ordered[((ordered.len() - 1) as f64 * fraction).round() as usize]
    };
    let wall_p10_ms = percentile(0.10);
    let wall_median_ms = percentile(0.50);
    let wall_p90_ms = percentile(0.90);
    let logical_bytes = weight
        .metadata
        .data_bytes
        .checked_add(bias.metadata.data_bytes)
        .and_then(|value| value.checked_add(input_bytes.len() as u64))
        .and_then(|value| value.checked_add((BATCH * ROWS * 4) as u64))
        .and_then(|value| value.checked_add(output_bytes.len() as u64))
        .ok_or("router logical byte count overflow")?;
    Ok(MetalRouterReport {
        schema_version: 1,
        semantic: "mimo_layer43_native_noaux_tc_router",
        router_file_sha256,
        reference_manifest_sha256,
        input_sha256: sha256_hex(&input_bytes),
        output_sha256: sha256_hex(&output_bytes),
        device: device.name().to_owned(),
        kernel_file: kernel_path
            .file_name()
            .and_then(|name| name.to_str())
            .ok_or("kernel file name is not UTF-8")?
            .to_owned(),
        kernel_function: KERNEL_FUNCTION,
        threadgroup_memory_bytes: THREADGROUP_MEMORY_BYTES,
        selected_experts_by_position,
        route_weights_by_position,
        maximum_route_weight_absolute_error,
        minimum_topk_boundary_margin: routes.minimum_boundary_margin,
        kernel_fixture_maximum_absolute_error,
        compile_ms,
        cold_wall_ms,
        warmups: WARMUPS,
        measurements: MEASUREMENTS,
        wall_ms,
        wall_p10_ms,
        wall_median_ms,
        wall_p90_ms,
        median_timing_gate_passed: wall_median_ms <= 1.0,
        logical_bytes,
        batch_size: BATCH,
        concurrency: 1,
        accepted_tokens: 0,
        accepted_per_verification: 0,
        expert_union_factor: union.len() as f64 / TOP_K as f64,
        cache_state: "router source OS-cache state uncontrolled; weight and input application buffers warm after cold dispatch",
        implementation: "rust_owned_metal_f32_router_plus_native_noaux_tc_selection",
    })
}

#[derive(Debug, Serialize)]
pub struct TensorRecord {
    pub name: String,
    pub shard: String,
    pub dtype: String,
    pub shape: Vec<u64>,
    pub data_bytes: u64,
    pub category: String,
}

#[derive(Debug, Default, Serialize)]
pub struct CategoryTotal {
    pub tensors: u64,
    pub data_bytes: u64,
}

#[derive(Debug, Serialize)]
pub struct Census {
    pub schema_version: u32,
    pub tensor_count: usize,
    pub tensor_data_bytes: u64,
    pub shard_file_bytes: u64,
    pub header_and_padding_bytes: u64,
    pub categories: BTreeMap<String, CategoryTotal>,
    pub tensors: Vec<TensorRecord>,
}

fn require_object<'a>(
    value: &'a Value,
    context: &str,
) -> Result<&'a serde_json::Map<String, Value>, String> {
    value
        .as_object()
        .ok_or_else(|| format!("{context} must be an object"))
}

fn require_u64_pair(value: &Value, context: &str) -> Result<(u64, u64), String> {
    let values = value
        .as_array()
        .ok_or_else(|| format!("{context} must be an array"))?;
    if values.len() != 2 {
        return Err(format!("{context} must contain two offsets"));
    }
    let start = values[0]
        .as_u64()
        .ok_or_else(|| format!("{context} start is not u64"))?;
    let end = values[1]
        .as_u64()
        .ok_or_else(|| format!("{context} end is not u64"))?;
    if end < start {
        return Err(format!("{context} offsets are reversed"));
    }
    Ok((start, end))
}

pub fn classify_tensor(name: &str) -> &'static str {
    if name.starts_with("visual.") {
        "vision_encoder"
    } else if name.starts_with("audio_encoder.") || name.starts_with("speech_embeddings.") {
        "audio_path"
    } else if name.contains(".mtp.") {
        "mtp"
    } else if name.contains(".mlp.experts.") {
        "routed_experts"
    } else if name.contains(".mlp.gate.weight") {
        "routers"
    } else if name == "model.embed_tokens.weight" {
        "token_embeddings"
    } else if name == "lm_head.weight" {
        "lm_head"
    } else if name.starts_with("model.layers.0.mlp.") {
        "dense_layer_zero"
    } else if name.contains("self_attn")
        || name.contains("layernorm")
        || name == "model.norm.weight"
    {
        "attention_and_norms"
    } else {
        "other_language_or_projector"
    }
}

pub fn build_census(index_path: &Path, checkpoint_dir: &Path) -> Result<Census, String> {
    let index: Value = serde_json::from_reader(
        File::open(index_path).map_err(|error| format!("{}: {error}", index_path.display()))?,
    )
    .map_err(|error| format!("{}: {error}", index_path.display()))?;
    let weight_map = require_object(
        index.get("weight_map").ok_or("index lacks weight_map")?,
        "weight_map",
    )?;
    let mut shard_names = BTreeSet::new();
    for shard in weight_map.values() {
        shard_names.insert(
            shard
                .as_str()
                .ok_or("weight_map shard must be a string")?
                .to_owned(),
        );
    }
    // Standalone safetensors not named by the main index are explicit sources too.
    for standalone in ["model_mtp.safetensors", "audio_tokenizer/model.safetensors"] {
        if checkpoint_dir.join(standalone).exists() {
            shard_names.insert(standalone.to_owned());
        }
    }

    let mut records = Vec::new();
    let mut seen_indexed = BTreeSet::new();
    let mut shard_file_bytes = 0_u64;
    for shard in shard_names {
        let path: PathBuf = checkpoint_dir.join(&shard);
        let mapped = MappedSafetensors::open(&path)?;
        let file_bytes = mapped.mapping.len() as u64;
        shard_file_bytes = shard_file_bytes
            .checked_add(file_bytes)
            .ok_or("shard byte overflow")?;
        for metadata in mapped.tensors.values() {
            let name = &metadata.name;
            if let Some(expected_shard) = weight_map.get(name) {
                if expected_shard.as_str() != Some(&shard) {
                    return Err(format!("{name}: index points to a different shard"));
                }
                if !seen_indexed.insert(name.clone()) {
                    return Err(format!("{name}: tensor appears more than once"));
                }
            } else if shard != "audio_tokenizer/model.safetensors"
                && shard != "model_mtp.safetensors"
            {
                return Err(format!("{name}: tensor is absent from the source index"));
            }
            records.push(TensorRecord {
                category: if shard == "audio_tokenizer/model.safetensors" {
                    "audio_tokenizer".to_owned()
                } else {
                    classify_tensor(name).to_owned()
                },
                name: name.clone(),
                shard: shard.clone(),
                dtype: metadata.dtype.clone(),
                shape: metadata.shape.clone(),
                data_bytes: metadata.data_bytes,
            });
        }
    }
    if seen_indexed.len() != weight_map.len() {
        return Err(format!(
            "index assigns {} tensors but only {} were found",
            weight_map.len(),
            seen_indexed.len()
        ));
    }
    records.sort_by(|left, right| {
        left.name
            .cmp(&right.name)
            .then(left.shard.cmp(&right.shard))
    });
    let tensor_data_bytes = records.iter().try_fold(0_u64, |total, record| {
        total
            .checked_add(record.data_bytes)
            .ok_or("tensor byte overflow")
    })?;
    let mut categories: BTreeMap<String, CategoryTotal> = BTreeMap::new();
    for record in &records {
        let total = categories.entry(record.category.clone()).or_default();
        total.tensors += 1;
        total.data_bytes += record.data_bytes;
    }
    Ok(Census {
        schema_version: 1,
        tensor_count: records.len(),
        tensor_data_bytes,
        shard_file_bytes,
        header_and_padding_bytes: shard_file_bytes
            .checked_sub(tensor_data_bytes)
            .ok_or("tensor bytes exceed shard bytes")?,
        categories,
        tensors: records,
    })
}

pub fn write_census(census: &Census, output: &Path) -> Result<(), String> {
    let mut file =
        File::create(output).map_err(|error| format!("{}: {error}", output.display()))?;
    serde_json::to_writer_pretty(&mut file, census).map_err(|error| error.to_string())?;
    use std::io::Write;
    file.write_all(b"\n").map_err(|error| error.to_string())
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ExpertContainerTensor {
    pub name: String,
    pub dtype: String,
    pub shape: Vec<u64>,
    pub source_data_offsets: [u64; 2],
    pub payload_offset: u64,
    pub data_bytes: u64,
    pub sha256: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ExpertContainerHeader {
    pub schema_version: u32,
    pub source_file: String,
    pub source_file_bytes: u64,
    pub source_sha256: String,
    pub payload_alignment: u64,
    pub tensors: Vec<ExpertContainerTensor>,
}

fn align_up(value: u64, alignment: u64) -> Result<u64, String> {
    if alignment == 0 || !alignment.is_power_of_two() {
        return Err("alignment must be a nonzero power of two".to_owned());
    }
    value
        .checked_add(alignment - 1)
        .map(|sum| sum & !(alignment - 1))
        .ok_or_else(|| "alignment overflow".to_owned())
}

fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn sha256_reader(reader: &mut File) -> Result<String, String> {
    reader
        .seek(SeekFrom::Start(0))
        .map_err(|error| error.to_string())?;
    let mut digest = Sha256::new();
    let mut buffer = vec![0_u8; 8 * 1024 * 1024];
    loop {
        let read = reader
            .read(&mut buffer)
            .map_err(|error| error.to_string())?;
        if read == 0 {
            break;
        }
        digest.update(&buffer[..read]);
    }
    Ok(digest
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect())
}

pub fn repack_expert_container(
    source: &Path,
    output: &Path,
    tensor_names: &[String],
) -> Result<ExpertContainerHeader, String> {
    if tensor_names.is_empty() {
        return Err("at least one tensor is required".to_owned());
    }
    if output.exists() {
        return Err(format!("refusing to overwrite {}", output.display()));
    }
    let mut names = tensor_names.to_vec();
    names.sort();
    if names.windows(2).any(|pair| pair[0] == pair[1]) {
        return Err("duplicate tensor name".to_owned());
    }
    let mapped = MappedSafetensors::open(source)?;
    let source_file_bytes = mapped.mapping.len() as u64;
    let mut source_file = File::open(source).map_err(|error| error.to_string())?;
    let source_sha256 = sha256_reader(&mut source_file)?;
    let mut payload = Vec::new();
    let mut tensors = Vec::new();
    for name in names {
        let view = mapped.tensor(&name)?;
        let [start, end] = view.metadata.data_offsets;
        let data_bytes = view.metadata.data_bytes;
        let payload_offset = align_up(payload.len() as u64, EXPERT_ALIGNMENT)?;
        let padding = usize::try_from(payload_offset - payload.len() as u64)
            .map_err(|_| "payload padding does not fit usize")?;
        payload.resize(payload.len() + padding, 0);
        let tensor_sha256 = sha256_hex(view.bytes);
        payload.extend_from_slice(view.bytes);
        tensors.push(ExpertContainerTensor {
            name,
            dtype: view.metadata.dtype.clone(),
            shape: view.metadata.shape.clone(),
            source_data_offsets: [start, end],
            payload_offset,
            data_bytes,
            sha256: tensor_sha256,
        });
    }
    let header = ExpertContainerHeader {
        schema_version: 1,
        source_file: source
            .file_name()
            .and_then(|name| name.to_str())
            .ok_or("source file name is not UTF-8")?
            .to_owned(),
        source_file_bytes,
        source_sha256,
        payload_alignment: EXPERT_ALIGNMENT,
        tensors,
    };
    let header_bytes = serde_json::to_vec(&header).map_err(|error| error.to_string())?;
    let payload_start = align_up(
        16_u64
            .checked_add(header_bytes.len() as u64)
            .ok_or("container header overflow")?,
        EXPERT_ALIGNMENT,
    )?;
    let temporary = output.with_file_name(format!(
        ".{}.tmp.{}",
        output
            .file_name()
            .and_then(|name| name.to_str())
            .ok_or("output file name is not UTF-8")?,
        std::process::id()
    ));
    let write_result = (|| -> Result<(), String> {
        let mut file = OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&temporary)
            .map_err(|error| error.to_string())?;
        file.write_all(EXPERT_MAGIC)
            .and_then(|_| file.write_all(&(header_bytes.len() as u64).to_le_bytes()))
            .and_then(|_| file.write_all(&header_bytes))
            .map_err(|error| error.to_string())?;
        let current = 16 + header_bytes.len() as u64;
        file.write_all(&vec![0_u8; (payload_start - current) as usize])
            .and_then(|_| file.write_all(&payload))
            .and_then(|_| file.sync_all())
            .map_err(|error| error.to_string())?;
        verify_expert_container(&temporary)?;
        fs::hard_link(&temporary, output).map_err(|error| error.to_string())?;
        Ok(())
    })();
    let _ = fs::remove_file(&temporary);
    write_result?;
    Ok(header)
}

pub fn verify_expert_container(path: &Path) -> Result<ExpertContainerHeader, String> {
    let mut file = File::open(path).map_err(|error| error.to_string())?;
    let file_bytes = file.metadata().map_err(|error| error.to_string())?.len();
    let mut magic = [0_u8; 8];
    file.read_exact(&mut magic)
        .map_err(|error| error.to_string())?;
    if &magic != EXPERT_MAGIC {
        return Err("unknown expert-container magic".to_owned());
    }
    let mut header_length_bytes = [0_u8; 8];
    file.read_exact(&mut header_length_bytes)
        .map_err(|error| error.to_string())?;
    let header_length = u64::from_le_bytes(header_length_bytes);
    if header_length == 0 || header_length > MAX_HEADER_BYTES {
        return Err("invalid expert-container header length".to_owned());
    }
    let mut raw_header = vec![0_u8; header_length as usize];
    file.read_exact(&mut raw_header)
        .map_err(|error| error.to_string())?;
    let header: ExpertContainerHeader =
        serde_json::from_slice(&raw_header).map_err(|error| error.to_string())?;
    if header.schema_version != 1 || header.payload_alignment != EXPERT_ALIGNMENT {
        return Err("unknown expert-container schema".to_owned());
    }
    let payload_start = align_up(16 + header_length, header.payload_alignment)?;
    let mut previous_end = 0_u64;
    let mut names = BTreeSet::new();
    for tensor in &header.tensors {
        if !names.insert(&tensor.name) {
            return Err("duplicate tensor in expert container".to_owned());
        }
        if tensor.payload_offset % header.payload_alignment != 0
            || tensor.payload_offset < previous_end
        {
            return Err("invalid or overlapping expert payload offsets".to_owned());
        }
        let end = tensor
            .payload_offset
            .checked_add(tensor.data_bytes)
            .ok_or("expert payload overflow")?;
        if payload_start
            .checked_add(end)
            .ok_or("file offset overflow")?
            > file_bytes
        {
            return Err("expert payload exceeds container".to_owned());
        }
        let mut data = vec![
            0_u8;
            usize::try_from(tensor.data_bytes)
                .map_err(|_| "tensor does not fit usize")?
        ];
        file.seek(SeekFrom::Start(payload_start + tensor.payload_offset))
            .and_then(|_| file.read_exact(&mut data))
            .map_err(|error| error.to_string())?;
        if sha256_hex(&data) != tensor.sha256 {
            return Err(format!("tensor hash mismatch: {}", tensor.name));
        }
        previous_end = end;
    }
    if payload_start + previous_end != file_bytes {
        return Err("unexpected trailing or missing expert-container bytes".to_owned());
    }
    Ok(header)
}

pub fn read_header_length(path: &Path) -> Result<u64, String> {
    let mut file = File::open(path).map_err(|error| error.to_string())?;
    file.seek(SeekFrom::Start(0))
        .map_err(|error| error.to_string())?;
    let mut prefix = [0_u8; 8];
    file.read_exact(&mut prefix)
        .map_err(|error| error.to_string())?;
    Ok(u64::from_le_bytes(prefix))
}

#[derive(Debug, Deserialize)]
pub struct TinyExpert {
    pub gate: Vec<Vec<f64>>,
    pub up: Vec<Vec<f64>>,
    pub down: Vec<Vec<f64>>,
}

#[derive(Debug, Deserialize)]
pub struct TinyExpected {
    pub topk_indices: Vec<usize>,
    pub topk_weights: Vec<f64>,
    pub output: Vec<f64>,
}

#[derive(Debug, Deserialize)]
pub struct TinyMoeFixture {
    pub schema_version: u32,
    pub semantic: String,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub groups: usize,
    pub topk_group: usize,
    pub top_k: usize,
    pub normalize_topk: bool,
    pub routed_scaling_factor: f64,
    pub inputs: Vec<Vec<f64>>,
    pub router_weight: Vec<Vec<f64>>,
    pub correction_bias: Vec<f64>,
    pub experts: Vec<TinyExpert>,
    pub expected: Vec<TinyExpected>,
}

#[derive(Debug)]
pub struct TinyMoeResult {
    pub topk_indices: Vec<usize>,
    pub topk_weights: Vec<f64>,
    pub output: Vec<f64>,
}

#[derive(Debug, Deserialize)]
pub struct RealFp8Fixture {
    pub schema_version: u32,
    pub semantic: String,
    pub raw_u8: Vec<Vec<u8>>,
    pub decoded_fp8: Vec<Vec<f32>>,
    pub scale_inv: f32,
    pub dequantized: Vec<Vec<f32>>,
}

#[derive(Debug, Deserialize)]
pub struct ExhaustiveFp8Fixture {
    pub schema_version: u32,
    pub semantic: String,
    pub expected_f32_bits: Vec<u32>,
}

#[derive(Debug, Deserialize)]
pub struct RealFp8GemvFixture {
    pub schema_version: u32,
    pub semantic: String,
    pub rows: usize,
    pub columns: usize,
    pub block_columns: usize,
    pub raw_u8: Vec<Vec<u8>>,
    pub scale_inv: Vec<f32>,
    pub input: Vec<f32>,
    pub expected_f32: Vec<f32>,
}

#[derive(Debug, Deserialize)]
pub struct RealInt4GemvFixture {
    pub schema_version: u32,
    pub semantic: String,
    pub rows: usize,
    pub columns: usize,
    pub block_columns: usize,
    pub packed_u8: Vec<Vec<u8>>,
    pub scale: Vec<Vec<f32>>,
    pub input: Vec<f32>,
    pub expected_f32: Vec<f32>,
}

#[derive(Debug, Deserialize)]
pub struct MlxAffineInt4Fixture {
    pub schema_version: u32,
    pub semantic: String,
    pub rows: usize,
    pub columns: usize,
    pub group_size: usize,
    pub bits: usize,
    pub packed_u32: Vec<Vec<u32>>,
    pub scale_f16: Vec<Vec<f32>>,
    pub bias_f16: Vec<Vec<f32>>,
    pub input_f16: Vec<f32>,
    pub expected_manual_f32: Vec<f32>,
}

pub fn decode_f8_e4m3fn(bits: u8) -> f32 {
    let sign = if bits & 0x80 == 0 { 1.0 } else { -1.0 };
    let exponent = i32::from((bits >> 3) & 0x0f);
    let mantissa = i32::from(bits & 0x07);
    if exponent == 0 {
        sign * 2.0_f32.powi(-6) * (mantissa as f32 / 8.0)
    } else if exponent == 15 && mantissa == 7 {
        f32::from_bits(u32::from(bits & 0x80) << 24 | 0x7ff0_0000)
    } else {
        sign * 2.0_f32.powi(exponent - 7) * (1.0 + mantissa as f32 / 8.0)
    }
}

pub fn block_fp8_gemv(
    rows: &[Vec<u8>],
    scales: &[f32],
    block_columns: usize,
    input: &[f32],
) -> Result<Vec<f32>, String> {
    if block_columns == 0
        || input.is_empty()
        || !input.len().is_multiple_of(block_columns)
        || scales.len() != input.len() / block_columns
        || rows.iter().any(|row| row.len() != input.len())
    {
        return Err("invalid block-FP8 GEMV dimensions".to_owned());
    }
    rows.iter()
        .map(|row| {
            let mut sum = 0.0_f32;
            for (column, (bits, activation)) in row.iter().zip(input).enumerate() {
                let scale = scales[column / block_columns];
                sum += decode_f8_e4m3fn(*bits) * scale * activation;
            }
            Ok(sum)
        })
        .collect()
}

fn decode_signed_nibble(nibble: u8) -> f32 {
    let value = nibble & 0x0f;
    if value < 8 {
        f32::from(value)
    } else {
        f32::from(value) - 16.0
    }
}

pub fn group_int4_gemv(
    rows: &[Vec<u8>],
    scales: &[Vec<f32>],
    block_columns: usize,
    input: &[f32],
) -> Result<Vec<f32>, String> {
    if block_columns == 0
        || !block_columns.is_multiple_of(2)
        || input.is_empty()
        || !input.len().is_multiple_of(block_columns)
        || rows.len() != scales.len()
        || rows.iter().any(|row| row.len() * 2 != input.len())
        || scales
            .iter()
            .any(|row| row.len() != input.len() / block_columns)
    {
        return Err("invalid group-INT4 GEMV dimensions".to_owned());
    }
    rows.iter()
        .zip(scales)
        .map(|(row, row_scales)| {
            let mut sum = 0.0_f32;
            for (packed_column, bits) in row.iter().enumerate() {
                let column = packed_column * 2;
                let scale = row_scales[column / block_columns];
                sum += decode_signed_nibble(*bits) * scale * input[column];
                sum += decode_signed_nibble(*bits >> 4) * scale * input[column + 1];
            }
            Ok(sum)
        })
        .collect()
}

pub fn affine_int4_gemv(
    packed_rows: &[Vec<u32>],
    scales: &[Vec<f32>],
    biases: &[Vec<f32>],
    group_size: usize,
    input: &[f32],
) -> Result<Vec<f32>, String> {
    let values_per_word = 8;
    if group_size == 0
        || !group_size.is_multiple_of(values_per_word)
        || input.is_empty()
        || !input.len().is_multiple_of(group_size)
        || packed_rows.len() != scales.len()
        || packed_rows.len() != biases.len()
        || packed_rows
            .iter()
            .any(|row| row.len() * values_per_word != input.len())
        || scales
            .iter()
            .chain(biases)
            .any(|row| row.len() != input.len() / group_size)
    {
        return Err("invalid affine-INT4 GEMV dimensions".to_owned());
    }
    packed_rows
        .iter()
        .zip(scales)
        .zip(biases)
        .map(|((row, row_scales), row_biases)| {
            let mut sum = 0.0_f32;
            for column in 0..input.len() {
                let word = row[column / values_per_word];
                let code = (word >> ((column % values_per_word) * 4)) & 0x0f;
                let group = column / group_size;
                let weight = code as f32 * row_scales[group] + row_biases[group];
                sum += weight * input[column];
            }
            Ok(sum)
        })
        .collect()
}

fn dot(row: &[f64], vector: &[f64]) -> Result<f64, String> {
    if row.len() != vector.len() {
        return Err("linear dimension mismatch".to_owned());
    }
    Ok(row
        .iter()
        .zip(vector)
        .map(|(left, right)| left * right)
        .sum())
}

fn linear(matrix: &[Vec<f64>], vector: &[f64]) -> Result<Vec<f64>, String> {
    matrix.iter().map(|row| dot(row, vector)).collect()
}

fn evaluate_expert(expert: &TinyExpert, input: &[f64]) -> Result<Vec<f64>, String> {
    let gate = linear(&expert.gate, input)?;
    let up = linear(&expert.up, input)?;
    if gate.len() != up.len() {
        return Err("SwiGLU gate/up dimension mismatch".to_owned());
    }
    let activated: Vec<f64> = gate
        .iter()
        .zip(up)
        .map(|(gate_value, up_value)| gate_value / (1.0 + (-gate_value).exp()) * up_value)
        .collect();
    linear(&expert.down, &activated)
}

pub fn evaluate_tiny_moe(fixture: &TinyMoeFixture, input: &[f64]) -> Result<TinyMoeResult, String> {
    if fixture.schema_version != 1 || fixture.semantic != "mimo_v2_noaux_tc_swiglu_moe" {
        return Err("unknown tiny MoE fixture schema or semantic".to_owned());
    }
    if input.len() != fixture.hidden_size
        || fixture.router_weight.len() != fixture.experts.len()
        || fixture.correction_bias.len() != fixture.experts.len()
        || fixture.groups == 0
        || !fixture.experts.len().is_multiple_of(fixture.groups)
        || fixture.topk_group == 0
        || fixture.topk_group > fixture.groups
    {
        return Err("invalid tiny MoE dimensions".to_owned());
    }
    let scores: Vec<f64> = fixture
        .router_weight
        .iter()
        .map(|row| dot(row, input).map(|logit| 1.0 / (1.0 + (-logit).exp())))
        .collect::<Result<_, _>>()?;
    let choice: Vec<f64> = scores
        .iter()
        .zip(&fixture.correction_bias)
        .map(|(score, bias)| score + bias)
        .collect();
    let group_size = fixture.experts.len() / fixture.groups;
    let mut group_scores = Vec::with_capacity(fixture.groups);
    for group in 0..fixture.groups {
        let mut values = choice[group * group_size..(group + 1) * group_size].to_vec();
        values.sort_by(|left, right| right.total_cmp(left));
        group_scores.push(values.iter().take(2).sum::<f64>());
    }
    let mut groups: Vec<usize> = (0..fixture.groups).collect();
    groups.sort_by(|left, right| {
        group_scores[*right]
            .total_cmp(&group_scores[*left])
            .then(left.cmp(right))
    });
    groups.truncate(fixture.topk_group);
    let selected_groups: BTreeSet<usize> = groups.into_iter().collect();
    let mut indices: Vec<usize> = (0..fixture.experts.len())
        .filter(|index| selected_groups.contains(&(index / group_size)))
        .collect();
    indices.sort_by(|left, right| {
        choice[*right]
            .total_cmp(&choice[*left])
            .then(left.cmp(right))
    });
    indices.truncate(fixture.top_k);
    if indices.len() != fixture.top_k {
        return Err("not enough experts after group selection".to_owned());
    }
    let denominator = if fixture.normalize_topk {
        scores
            .iter()
            .enumerate()
            .filter(|(index, _)| indices.contains(index))
            .map(|(_, score)| score)
            .sum::<f64>()
            + 1e-20
    } else {
        1.0
    };
    let weights: Vec<f64> = indices
        .iter()
        .map(|index| scores[*index] / denominator * fixture.routed_scaling_factor)
        .collect();
    let mut output = vec![0.0; fixture.hidden_size];
    for (index, weight) in indices.iter().zip(&weights) {
        let expert_output = evaluate_expert(&fixture.experts[*index], input)?;
        if expert_output.len() != fixture.hidden_size {
            return Err("expert output dimension mismatch".to_owned());
        }
        for (destination, value) in output.iter_mut().zip(expert_output) {
            *destination += weight * value;
        }
    }
    Ok(TinyMoeResult {
        topk_indices: indices,
        topk_weights: weights,
        output,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_safetensors(path: &Path, header: &str, payload: &[u8]) {
        let mut padded = header.as_bytes().to_vec();
        while !padded.len().is_multiple_of(8) {
            padded.push(b' ');
        }
        let mut bytes = (padded.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(&padded);
        bytes.extend_from_slice(payload);
        fs::write(path, bytes).expect("write safetensors fixture");
    }

    fn temporary_test_directory(name: &str) -> PathBuf {
        let path = std::env::temp_dir().join(format!(
            "prismwing-{name}-{}-{}",
            std::process::id(),
            std::thread::current().name().unwrap_or("test")
        ));
        fs::create_dir_all(&path).expect("temporary directory");
        path
    }

    #[test]
    fn tensor_categories_are_explicit() {
        assert_eq!(
            classify_tensor("model.layers.1.mlp.experts.4.up_proj.weight"),
            "routed_experts"
        );
        assert_eq!(classify_tensor("model.layers.3.mlp.gate.weight"), "routers");
        assert_eq!(
            classify_tensor("visual.blocks.0.attn.qkv.weight"),
            "vision_encoder"
        );
        assert_eq!(
            classify_tensor("audio_encoder.layers.0.weight"),
            "audio_path"
        );
    }

    #[test]
    fn mapped_safetensors_returns_exact_immutable_tensor_bytes() {
        let directory = temporary_test_directory("mapped-safetensors-valid");
        let path = directory.join("valid.safetensors");
        write_safetensors(
            &path,
            r#"{"x":{"dtype":"F32","shape":[2],"data_offsets":[0,8]},"y":{"dtype":"U8","shape":[2],"data_offsets":[8,10]}}"#,
            &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        );
        let mapped = MappedSafetensors::open(&path).expect("valid map");
        assert_eq!(mapped.tensor_count(), 2);
        let view = mapped.tensor("x").expect("x tensor");
        assert_eq!(view.metadata.dtype, "F32");
        assert_eq!(view.metadata.shape, vec![2]);
        assert_eq!(view.bytes, &[1, 2, 3, 4, 5, 6, 7, 8]);
        assert!(mapped.tensor("missing").is_err());
        fs::remove_dir_all(directory).expect("remove fixture directory");
    }

    #[test]
    fn mapped_safetensors_rejects_ambiguous_or_invalid_layouts() {
        let directory = temporary_test_directory("mapped-safetensors-invalid");
        let cases = [
            (
                "duplicate",
                r#"{"x":{"dtype":"U8","shape":[1],"data_offsets":[0,1]},"x":{"dtype":"U8","shape":[1],"data_offsets":[1,2]}}"#,
                vec![0, 1],
            ),
            (
                "unknown-dtype",
                r#"{"x":{"dtype":"MYSTERY","shape":[1],"data_offsets":[0,1]}}"#,
                vec![0],
            ),
            (
                "shape-bytes",
                r#"{"x":{"dtype":"F32","shape":[2],"data_offsets":[0,4]}}"#,
                vec![0; 4],
            ),
            (
                "overlap",
                r#"{"x":{"dtype":"U8","shape":[2],"data_offsets":[0,2]},"y":{"dtype":"U8","shape":[2],"data_offsets":[1,3]}}"#,
                vec![0; 3],
            ),
            (
                "truncated",
                r#"{"x":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}"#,
                vec![0; 7],
            ),
            (
                "zero-shape",
                r#"{"x":{"dtype":"U8","shape":[0],"data_offsets":[0,0]}}"#,
                vec![],
            ),
        ];
        for (name, header, payload) in cases {
            let path = directory.join(format!("{name}.safetensors"));
            write_safetensors(&path, header, &payload);
            assert!(MappedSafetensors::open(&path).is_err(), "case {name}");
        }
        fs::remove_dir_all(directory).expect("remove fixture directory");
    }

    #[test]
    fn mapped_fp8_gemv_selects_row_and_column_scale_blocks() {
        let directory = temporary_test_directory("mapped-fp8-gemv");
        let source = directory.join("projection.safetensors");
        let input_path = directory.join("input.f32");
        let output_path = directory.join("output.f32");
        let mut payload = vec![0x38; 256 * 256]; // E4M3 1.0
        for scale in [1.0_f32, 2.0, 3.0, 4.0] {
            payload.extend_from_slice(&scale.to_le_bytes());
        }
        write_safetensors(
            &source,
            r#"{"weight":{"dtype":"F8_E4M3","shape":[256,256],"data_offsets":[0,65536]},"scale":{"dtype":"F32","shape":[2,2],"data_offsets":[65536,65552]}}"#,
            &payload,
        );
        let input = vec![1.0_f32; 256];
        let input_bytes = input
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>();
        fs::write(&input_path, input_bytes).expect("input fixture");
        let mapped = MappedSafetensors::open(&source).expect("mapped fixture");
        let result = mapped_fp8_gemv(&mapped, "weight", "scale", &input).expect("GEMV");
        assert!(result[..128].iter().all(|value| *value == 384.0));
        assert!(result[128..].iter().all(|value| *value == 896.0));
        let report = run_mapped_fp8_gemv(&source, "weight", "scale", &input_path, &output_path)
            .expect("file GEMV");
        assert_eq!(report.output_f32, 256);
        assert!(
            run_mapped_fp8_gemv(&source, "weight", "scale", &input_path, &output_path,)
                .unwrap_err()
                .contains("refusing to overwrite")
        );
        let mut invalid = input;
        invalid[0] = f32::NAN;
        assert!(mapped_fp8_gemv(&mapped, "weight", "scale", &invalid).is_err());
        let short_input = directory.join("short.f32");
        fs::write(&short_input, [0_u8; 3]).expect("short input fixture");
        assert!(
            run_mapped_fp8_gemv(
                &source,
                "weight",
                "scale",
                &short_input,
                &directory.join("short-output.f32"),
            )
            .is_err()
        );

        let nonfinite_source = directory.join("nonfinite-scale.safetensors");
        let mut nonfinite_payload = vec![0x38; 256 * 256];
        for scale in [f32::NAN, 2.0, 3.0, 4.0] {
            nonfinite_payload.extend_from_slice(&scale.to_le_bytes());
        }
        write_safetensors(
            &nonfinite_source,
            r#"{"weight":{"dtype":"F8_E4M3","shape":[256,256],"data_offsets":[0,65536]},"scale":{"dtype":"F32","shape":[2,2],"data_offsets":[65536,65552]}}"#,
            &nonfinite_payload,
        );
        let nonfinite_mapped = MappedSafetensors::open(&nonfinite_source).expect("nonfinite map");
        assert!(mapped_fp8_gemv(&nonfinite_mapped, "weight", "scale", &vec![1.0; 256]).is_err());

        let nonfinite_weight_source = directory.join("nonfinite-weight.safetensors");
        let mut nonfinite_weight_payload = payload.clone();
        nonfinite_weight_payload[123] = 0x7f;
        write_safetensors(
            &nonfinite_weight_source,
            r#"{"weight":{"dtype":"F8_E4M3","shape":[256,256],"data_offsets":[0,65536]},"scale":{"dtype":"F32","shape":[2,2],"data_offsets":[65536,65552]}}"#,
            &nonfinite_weight_payload,
        );
        let nonfinite_weight_mapped =
            MappedSafetensors::open(&nonfinite_weight_source).expect("nonfinite weight map");
        assert!(
            mapped_fp8_gemv(&nonfinite_weight_mapped, "weight", "scale", &vec![1.0; 256])
                .unwrap_err()
                .contains("non-finite FP8 weight")
        );

        let bad_grid_source = directory.join("bad-grid.safetensors");
        write_safetensors(
            &bad_grid_source,
            r#"{"weight":{"dtype":"F8_E4M3","shape":[256,256],"data_offsets":[0,65536]},"scale":{"dtype":"F32","shape":[1,4],"data_offsets":[65536,65552]}}"#,
            &payload,
        );
        let bad_grid_mapped = MappedSafetensors::open(&bad_grid_source).expect("bad grid map");
        assert!(mapped_fp8_gemv(&bad_grid_mapped, "weight", "scale", &vec![1.0; 256]).is_err());
        fs::remove_dir_all(directory).expect("remove fixture directory");
    }

    #[test]
    fn source_derived_tiny_moe_fixture_matches() {
        let fixture: TinyMoeFixture = serde_json::from_str(include_str!(
            "../evals/fixtures/tiny/moe-noaux-tc-swiglu.json"
        ))
        .expect("fixture parses");
        assert_eq!(fixture.inputs.len(), fixture.expected.len());
        for (input, expected) in fixture.inputs.iter().zip(&fixture.expected) {
            let result = evaluate_tiny_moe(&fixture, input).expect("evaluation");
            assert_eq!(result.topk_indices, expected.topk_indices);
            for (actual, wanted) in result.topk_weights.iter().zip(&expected.topk_weights) {
                assert!((actual - wanted).abs() < 1e-14);
            }
            for (actual, wanted) in result.output.iter().zip(&expected.output) {
                assert!((actual - wanted).abs() < 1e-14);
            }
        }
    }

    #[test]
    fn sampled_real_fp8_block_matches() {
        let fixture: RealFp8Fixture = serde_json::from_str(include_str!(
            "../evals/fixtures/real/mtp-gate-fp8-block.json"
        ))
        .expect("fixture parses");
        assert_eq!(fixture.schema_version, 1);
        assert_eq!(fixture.semantic, "safetensors_f8_e4m3fn_block_dequant");
        assert_eq!(fixture.raw_u8.len(), fixture.decoded_fp8.len());
        for ((raw_row, decoded_row), dequantized_row) in fixture
            .raw_u8
            .iter()
            .zip(&fixture.decoded_fp8)
            .zip(&fixture.dequantized)
        {
            assert_eq!(raw_row.len(), decoded_row.len());
            assert_eq!(raw_row.len(), dequantized_row.len());
            for ((bits, expected), expected_dequantized) in
                raw_row.iter().zip(decoded_row).zip(dequantized_row)
            {
                let decoded = decode_f8_e4m3fn(*bits);
                assert_eq!(decoded, *expected);
                assert!((decoded * fixture.scale_inv - expected_dequantized).abs() < 1e-9);
            }
        }
    }

    #[test]
    fn every_fp8_encoding_matches_oracle() {
        let fixture: ExhaustiveFp8Fixture = serde_json::from_str(include_str!(
            "../evals/fixtures/tiny/f8-e4m3fn-all-bytes.json"
        ))
        .expect("fixture parses");
        assert_eq!(fixture.schema_version, 1);
        assert_eq!(fixture.semantic, "f8_e4m3fn_exhaustive_f32_bits");
        assert_eq!(fixture.expected_f32_bits.len(), 256);
        for bits in 0_u8..=u8::MAX {
            assert_eq!(
                decode_f8_e4m3fn(bits).to_bits(),
                fixture.expected_f32_bits[usize::from(bits)],
                "FP8 byte 0x{bits:02x}"
            );
        }
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn swiglu_fixture_has_independent_f64_oracle() {
        let fixture: SwiGluFixture =
            serde_json::from_str(include_str!("../evals/fixtures/tiny/swiglu-f32.json"))
                .expect("SwiGLU fixture parses");
        assert_eq!(fixture.schema_version, 1);
        assert_eq!(fixture.semantic, "silu_gate_times_up_f32");
        assert_eq!(fixture.gate.len(), fixture.up.len());
        assert_eq!(fixture.gate.len(), fixture.expected_f64.len());
        for ((&gate, &up), &expected) in fixture
            .gate
            .iter()
            .zip(&fixture.up)
            .zip(&fixture.expected_f64)
        {
            let gate = f64::from(gate);
            let actual = gate / (1.0 + (-gate).exp()) * f64::from(up);
            assert!((actual - expected).abs() < 1.0e-15);
        }
    }

    #[test]
    fn production_width_real_fp8_gemv_matches() {
        let fixture: RealFp8GemvFixture = serde_json::from_str(include_str!(
            "../evals/fixtures/real/mtp-gate-fp8-gemv.json"
        ))
        .expect("fixture parses");
        assert_eq!(fixture.schema_version, 1);
        assert_eq!(fixture.semantic, "mimo_block_fp8_gemv_slice");
        assert_eq!(fixture.raw_u8.len(), fixture.rows);
        assert_eq!(fixture.input.len(), fixture.columns);
        let actual = block_fp8_gemv(
            &fixture.raw_u8,
            &fixture.scale_inv,
            fixture.block_columns,
            &fixture.input,
        )
        .expect("GEMV");
        assert_eq!(actual.len(), fixture.expected_f32.len());
        for (row, (value, expected)) in actual.iter().zip(&fixture.expected_f32).enumerate() {
            assert!(
                (value - expected).abs() < 2e-7,
                "row {row}: actual {value}, expected {expected}"
            );
        }
    }

    #[test]
    fn block_fp8_gemv_rejects_bad_layouts() {
        let row = vec![vec![0_u8; 4]];
        let input = vec![0.0_f32; 4];
        assert!(block_fp8_gemv(&row, &[1.0], 0, &input).is_err());
        assert!(block_fp8_gemv(&row, &[1.0], 3, &input).is_err());
        assert!(block_fp8_gemv(&row, &[1.0, 1.0], 4, &input).is_err());
        assert!(block_fp8_gemv(&[vec![0_u8; 3]], &[1.0], 4, &input).is_err());
    }

    #[test]
    fn production_width_real_int4_gemv_matches() {
        let fixture: RealInt4GemvFixture = serde_json::from_str(include_str!(
            "../evals/fixtures/real/mtp-gate-int4-gemv.json"
        ))
        .expect("fixture parses");
        assert_eq!(fixture.schema_version, 1);
        assert_eq!(fixture.semantic, "mimo_signed_int4_group128_gemv_slice");
        assert_eq!(fixture.packed_u8.len(), fixture.rows);
        assert_eq!(fixture.input.len(), fixture.columns);
        let actual = group_int4_gemv(
            &fixture.packed_u8,
            &fixture.scale,
            fixture.block_columns,
            &fixture.input,
        )
        .expect("GEMV");
        for (row, (value, expected)) in actual.iter().zip(&fixture.expected_f32).enumerate() {
            assert!(
                (value - expected).abs() < 2e-7,
                "row {row}: actual {value}, expected {expected}"
            );
        }
    }

    #[test]
    fn group_int4_gemv_rejects_bad_layouts() {
        let input = vec![0.0_f32; 4];
        assert!(group_int4_gemv(&[vec![0; 2]], &[vec![1.0]], 0, &input).is_err());
        assert!(group_int4_gemv(&[vec![0; 2]], &[vec![1.0]], 3, &input).is_err());
        assert!(group_int4_gemv(&[vec![0; 1]], &[vec![1.0]], 4, &input).is_err());
        assert!(group_int4_gemv(&[vec![0; 2]], &[vec![]], 4, &input).is_err());
    }

    #[test]
    fn mlx_affine_int4_fixture_has_independent_scalar_oracle() {
        let fixture: MlxAffineInt4Fixture = serde_json::from_str(include_str!(
            "../evals/fixtures/real/mtp-gate-mlx-affine-int4.json"
        ))
        .expect("fixture parses");
        assert_eq!(fixture.schema_version, 1);
        assert_eq!(fixture.semantic, "mlx_affine_int4_group128_gemv_slice");
        assert_eq!(fixture.bits, 4);
        assert_eq!(fixture.packed_u32.len(), fixture.rows);
        assert_eq!(fixture.input_f16.len(), fixture.columns);
        let actual = affine_int4_gemv(
            &fixture.packed_u32,
            &fixture.scale_f16,
            &fixture.bias_f16,
            fixture.group_size,
            &fixture.input_f16,
        )
        .expect("affine GEMV");
        for (row, (value, expected)) in actual.iter().zip(&fixture.expected_manual_f32).enumerate()
        {
            assert!(
                (value - expected).abs() < 2e-7,
                "row {row}: actual {value}, expected {expected}"
            );
        }
    }

    #[test]
    fn real_expert_down_affine_int4_has_independent_scalar_oracle() {
        let fixture: MlxAffineInt4Fixture = serde_json::from_str(include_str!(
            "../evals/fixtures/real/l43-e32-down-mlx-affine-int4.json"
        ))
        .expect("fixture parses");
        assert_eq!(fixture.schema_version, 1);
        assert_eq!(fixture.semantic, "mlx_affine_int4_group128_gemv_slice");
        assert_eq!(fixture.bits, 4);
        assert_eq!(fixture.packed_u32.len(), fixture.rows);
        assert_eq!(fixture.input_f16.len(), fixture.columns);
        let actual = affine_int4_gemv(
            &fixture.packed_u32,
            &fixture.scale_f16,
            &fixture.bias_f16,
            fixture.group_size,
            &fixture.input_f16,
        )
        .expect("affine GEMV");
        for (row, (value, expected)) in actual.iter().zip(&fixture.expected_manual_f32).enumerate()
        {
            assert!(
                (value - expected).abs() < 2e-7,
                "row {row}: actual {value}, expected {expected}"
            );
        }
    }

    #[test]
    fn expert_container_round_trips_and_detects_tampering() {
        let directory = temporary_test_directory("container");
        let source = directory.join("source.safetensors");
        let output = directory.join("expert.pwexpert");
        let header = serde_json::json!({
            "tensor.b": {"dtype": "U8", "shape": [3], "data_offsets": [4, 7]},
            "tensor.a": {"dtype": "U8", "shape": [4], "data_offsets": [0, 4]}
        });
        write_safetensors(&source, &header.to_string(), &[1, 2, 3, 4, 9, 8, 7]);

        let names = vec!["tensor.b".to_owned(), "tensor.a".to_owned()];
        let packed = repack_expert_container(&source, &output, &names).expect("repack");
        assert_eq!(packed.tensors[0].name, "tensor.a");
        assert_eq!(packed.tensors[1].name, "tensor.b");
        let verified = verify_expert_container(&output).expect("verify");
        assert_eq!(verified.tensors.len(), 2);
        assert!(
            repack_expert_container(&source, &output, &names)
                .unwrap_err()
                .contains("refusing to overwrite")
        );

        let mut tampered = OpenOptions::new()
            .read(true)
            .write(true)
            .open(&output)
            .expect("open output");
        let mut raw_length = [0_u8; 8];
        tampered.seek(SeekFrom::Start(8)).expect("seek length");
        tampered.read_exact(&mut raw_length).expect("read length");
        let payload_start =
            align_up(16 + u64::from_le_bytes(raw_length), EXPERT_ALIGNMENT).expect("align");
        tampered
            .seek(SeekFrom::Start(payload_start))
            .and_then(|_| tampered.write_all(&[0xff]))
            .expect("tamper");
        assert!(
            verify_expert_container(&output)
                .unwrap_err()
                .contains("hash mismatch")
        );
        drop(tampered);
        fs::remove_dir_all(directory).expect("temporary cleanup");
    }
}
