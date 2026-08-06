//! PW-0050 bounded, target-faithful slow text endpoint.

use super::{
    MappedSafetensors, MappedTensorView, UniqueJson, ValidatedMappedFp8,
    accelerate_sgemm_right_transposed, decode_bf16_tensor, decode_fp8_matrix_f32, read_f32_file,
    sha256_hex, sha256_reader, stable_rms_inverse, validate_fp8_views, write_create_new,
};
use crate::routed_layer_artifact::{
    RoutedLayerArtifactManifest, RoutedLayerSourceTensor, build_routed_layer_artifact,
    open_routed_layer_artifact,
};
use crate::staged_metal_expert::{
    BoundedMetalExpertOutput, BoundedMetalExpertRuntime, ExpertTomography, NoCopyProjectionBacking,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File};
use std::os::unix::fs::MetadataExt;
use std::path::{Component, Path};
use std::process::Command;
use std::time::Instant;
use tokenizers::Tokenizer;

const REVISION: &str = "63651580ca774f8504f676040460aed3e1244ac1";
const MODEL_LOCK_SHA256: &str = "df8c74e6f9e1cef154aae5881b9042777653206aaff72855f7b1a1340e0d1050";
const HIDDEN: usize = 4096;
const HEADS: usize = 64;
const QK_HEAD_DIM: usize = 192;
const V_HEAD_DIM: usize = 128;
const ROPE_DIM: usize = 64;
const ROUTED_EXPERTS: usize = 256;
const TOP_K: usize = 8;
const MOE_INTERMEDIATE: usize = 2048;
const CHAT_PROMPT: &str = "<|im_start|>system\nYou are MiMo, a helpful AI assistant engineered by Xiaomi.<|im_end|><|im_start|>user\nHello<|im_end|><|im_start|>assistant\n<think></think>";
const CHAT_PROMPT_IDS: [u32; 27] = [
    151_644, 8948, 198, 2610, 525, 20_740, 25_612, 11, 264, 10_950, 15_235, 17_847, 44_936, 553,
    71_449, 13, 151_645, 151_644, 872, 198, 9707, 151_645, 151_644, 77_091, 198, 151_667, 151_668,
];

#[derive(Debug, Deserialize)]
struct EndpointFixture {
    schema_version: u32,
    semantic: String,
    revision: String,
    config_sha256: String,
    index_sha256: String,
    tokenizer_sha256: String,
    tokenizer_config_sha256: String,
    checkpoint_verification_sha256: String,
    prompt_utf8: String,
    add_special_tokens: bool,
    expected_prompt_token_ids: Vec<u32>,
    full_prefix_trace_append_token_ids: Option<Vec<u32>>,
    hosted_reference: Option<HostedReferenceFixture>,
    full_attention_qkv_scale_layout: FullQkvScaleFixture,
    decode: DecodeFixture,
    safety: SafetyFixture,
}

#[derive(Debug, Deserialize)]
struct HostedReferenceFixture {
    provider: String,
    manifest_sha256: String,
    response_sha256: String,
    generated_token_ids: Vec<u32>,
    generated_text: String,
}

#[derive(Debug, Deserialize)]
struct FullQkvScaleFixture {
    weight_shape: [usize; 2],
    scale_shape: [usize; 2],
    query_rows: usize,
    query_scale_rows: usize,
    key_heads: usize,
    key_rows_per_head: usize,
    key_scale_rows_per_head: usize,
    key_scale_row_start: usize,
    value_heads: usize,
    value_rows_per_head: usize,
    value_scale_row_start: usize,
}

#[derive(Debug, Deserialize)]
struct DecodeFixture {
    sampling: String,
    new_tokens: usize,
    batch_size: usize,
    concurrency: usize,
    use_kv_cache: bool,
}

#[derive(Clone, Debug, Deserialize)]
struct SafetyFixture {
    minimum_system_memory_free_percent: u64,
    maximum_process_physical_footprint_bytes: u64,
    maximum_post_phase_physical_footprint_bytes: u64,
    maximum_swap_growth_bytes: u64,
    maximum_new_throttled_pages: u64,
    require_malloc_pressure_relief: bool,
    protect_resident_services: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct ModelConfig {
    model_type: String,
    dtype: String,
    hidden_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    v_head_dim: usize,
    swa_num_attention_heads: usize,
    swa_num_key_value_heads: usize,
    swa_head_dim: usize,
    swa_v_head_dim: usize,
    partial_rotary_factor: f64,
    rope_theta: f64,
    swa_rope_theta: f64,
    sliding_window: usize,
    attention_value_scale: f32,
    add_full_attention_sink_bias: bool,
    add_swa_attention_sink_bias: bool,
    attention_projection_layout: String,
    intermediate_size: usize,
    moe_intermediate_size: usize,
    n_routed_experts: usize,
    num_experts_per_tok: usize,
    n_group: usize,
    topk_group: usize,
    norm_topk_prob: bool,
    routed_scaling_factor: Option<f32>,
    scoring_func: String,
    topk_method: String,
    layernorm_epsilon: f32,
    vocab_size: usize,
    hybrid_layer_pattern: Vec<u8>,
    moe_layer_freq: Vec<u8>,
    tie_word_embeddings: bool,
    quantization_config: QuantizationConfig,
}

#[derive(Debug, Deserialize)]
struct QuantizationConfig {
    activation_scheme: String,
    fmt: String,
    quant_method: String,
    store_dtype: String,
    weight_block_size: [usize; 2],
}

#[derive(Debug, Deserialize)]
struct CheckpointVerification {
    schema_version: u32,
    evidence_class: String,
    complete: bool,
    lock_sha256: String,
    revision: String,
    files: Vec<VerifiedFile>,
}

#[derive(Debug, Deserialize)]
struct VerifiedFile {
    path: String,
    bytes: u64,
    device: u64,
    inode: u64,
    modified_ns: i128,
    sha256: String,
    status: String,
}

#[derive(Debug, Default, Serialize)]
pub struct EndpointLedger {
    pub logical_source_bytes: u64,
    pub actual_process_disk_bytes_read: u64,
    pub peak_resident_bytes: u64,
    pub fp8_matrices_expanded: u64,
    pub bf16_matrices_expanded: u64,
    pub routed_expert_executions: u64,
    pub dynamic_activation_groups: u64,
    pub dynamic_activation_values: u64,
}

#[derive(Debug, Default, Serialize)]
pub struct MetalExpertLedger {
    pub expert_executions: u64,
    pub projection_dispatches: u64,
    pub installed_source_bytes: u64,
    pub released_projection_buffers: u64,
    pub sparse_decoded_weight_bytes: u64,
    pub sparse_repair_counts: [u64; 3],
    #[serde(skip)]
    pub tomography_enabled: bool,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub expert_tomography: Vec<ExpertTomography>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub layer_tomography: Vec<MetalLayerTomography>,
}

#[derive(Debug, Serialize)]
pub struct MetalLayerTomography {
    pub layer: usize,
    pub route_and_schedule_ms: f64,
    pub expert_count: usize,
    pub expert_wall_sum_ms: f64,
    pub layer_final_bf16_round_ms: f64,
    pub routed_mlp_wall_ms: f64,
    pub activity: ProcessActivityDelta,
}

#[derive(Debug, Serialize)]
pub struct LayerRouteTrace {
    pub layer: usize,
    pub attention: &'static str,
    pub cache_length: usize,
    pub selected_experts_by_position: Vec<Vec<u32>>,
    pub route_weights_by_position: Vec<Vec<f32>>,
    #[serde(rename = "U")]
    pub expert_union_factor: f64,
    pub wall_ms: f64,
}

#[derive(Debug, Serialize)]
pub struct DecodeStepReport {
    pub input_token_id: u32,
    pub input_token_ids: Vec<u32>,
    pub output_token_id: u32,
    pub output_token_text: String,
    pub top_logits: Vec<(u32, f32)>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub full_logits: Option<Vec<f32>>,
    pub layer_traces: Vec<LayerRouteTrace>,
    pub wall_ms: f64,
}

#[derive(Debug, Serialize)]
pub struct TextEndpointReport {
    pub schema_version: u32,
    pub semantic: &'static str,
    pub revision: &'static str,
    pub commit: String,
    pub fixture_sha256: String,
    pub model_lock_sha256: &'static str,
    pub checkpoint_verification_sha256: String,
    pub config_sha256: String,
    pub index_sha256: String,
    pub tokenizer_sha256: String,
    pub tokenizer_config_sha256: String,
    pub prompt_utf8: String,
    pub prompt_token_ids: Vec<u32>,
    pub generated_token_ids: Vec<u32>,
    pub generated_text: String,
    pub steps: Vec<DecodeStepReport>,
    pub ledger: EndpointLedger,
    pub safety_snapshots: Vec<SafetySnapshot>,
    pub complete_wall_ms: f64,
    pub batch_size: usize,
    pub concurrency: usize,
    pub accepted_tokens: usize,
    #[serde(rename = "A")]
    pub accepted_per_verification: usize,
    pub cache_state: &'static str,
    pub exactness: &'static str,
    pub performance_claim: Option<String>,
    pub implementation: &'static str,
}

#[derive(Debug, Serialize)]
pub struct NumericalParity {
    pub relative_l2: f64,
    pub maximum_absolute_error: f32,
    pub equal_values: usize,
    pub total_values: usize,
    pub equality_fraction: f64,
    pub passed: bool,
}

#[derive(Debug, Serialize)]
pub struct MetalLayerParity {
    pub layer: usize,
    pub selected_experts_exact: bool,
    pub maximum_route_weight_absolute_error: f32,
    pub final_state: NumericalParity,
}

#[derive(Debug, Serialize)]
pub struct MetalIncrementalTextReport {
    pub schema_version: u32,
    pub semantic: &'static str,
    pub revision: &'static str,
    pub commit: String,
    pub fixture_sha256: String,
    pub checkpoint_verification_sha256: String,
    pub oracle_manifest_sha256: String,
    pub kernel_sha256: String,
    pub kernel_compile_ms: f64,
    pub metal_device: String,
    pub prompt_token_ids: Vec<u32>,
    pub generated_token_ids: Vec<u32>,
    pub generated_text: String,
    pub steps: Vec<DecodeStepReport>,
    pub layer_parity: Vec<MetalLayerParity>,
    pub final_norm_parity: NumericalParity,
    pub logits_parity: NumericalParity,
    pub top20_token_identity: bool,
    pub projected_top20_jsd_nats: f64,
    pub ledger: EndpointLedger,
    pub metal_ledger: MetalExpertLedger,
    pub safety_snapshots: Vec<SafetySnapshot>,
    pub complete_wall_ms: f64,
    pub prefill_wall_ms: f64,
    pub incremental_wall_ms: f64,
    pub speedup_vs_pw0092_repeats: [f64; 2],
    pub timing_gate_passed: bool,
    pub batch_size: usize,
    pub concurrency: usize,
    pub accepted_tokens_in_timed_interval: usize,
    #[serde(rename = "A")]
    pub accepted_per_verification: usize,
    pub cache_state: &'static str,
    pub exactness: &'static str,
    pub performance_claim: Option<String>,
    pub implementation: &'static str,
    pub promotion_gates_passed: bool,
    pub status: &'static str,
}

#[derive(Deserialize)]
struct IncrementalOracleManifest {
    schema_version: u32,
    semantic: String,
    revision: String,
    checkpoint_verification_sha256: String,
    prefill_token_ids: Vec<u32>,
    incremental_input_token_id: u32,
    output_token_id: u32,
    captures: BTreeMap<String, OracleCapture>,
    incremental_layer_traces: Vec<OracleLayerTrace>,
}

#[derive(Deserialize)]
struct OracleCapture {
    file: String,
    shape: Vec<usize>,
    dtype: String,
    sha256: String,
}

#[derive(Deserialize)]
struct OracleLayerTrace {
    layer: usize,
    cache_positions: usize,
    selected_experts_by_position: Vec<Vec<u32>>,
    route_weights_by_position: Vec<Vec<f32>>,
}

#[derive(Deserialize)]
struct Layer4CachedOracleManifest {
    schema_version: u32,
    semantic: String,
    revision: String,
    checkpoint_verification_sha256: String,
    last_layer: usize,
    layer4_captures: BTreeMap<String, OracleCapture>,
    layer4_routes: Layer4OracleRoutes,
    layer4_expert_captures: BTreeMap<String, BTreeMap<String, OracleCapture>>,
}

#[derive(Deserialize)]
struct Layer4OracleRoutes {
    selected_experts: Vec<u32>,
    route_weights: Vec<f32>,
}

#[derive(Debug, Serialize)]
pub struct MetalStageDiagnostic {
    pub stage: &'static str,
    pub parity: NumericalParity,
    pub sparse_repairs: usize,
    pub first_mismatches: Vec<String>,
}

#[derive(Debug, Serialize)]
pub struct MetalExpertDiagnostic {
    pub expert: u32,
    pub stages: Vec<MetalStageDiagnostic>,
}

#[derive(Debug, Serialize)]
pub struct Layer4MetalDiagnosticReport {
    pub schema_version: u32,
    pub semantic: &'static str,
    pub revision: &'static str,
    pub commit: String,
    pub checkpoint_verification_sha256: String,
    pub oracle_manifest_sha256: String,
    pub kernel_sha256: String,
    pub kernel_compile_ms: f64,
    pub metal_device: String,
    pub selected_experts: Vec<u32>,
    pub route_weights: Vec<f32>,
    pub maximum_route_weight_absolute_error: f32,
    pub expert_diagnostics: Vec<MetalExpertDiagnostic>,
    pub routed_parity: NumericalParity,
    pub final_residual_parity: NumericalParity,
    pub metal_ledger: MetalExpertLedger,
    pub endpoint_ledger: EndpointLedger,
    pub safety_snapshots: Vec<SafetySnapshot>,
    pub wall_ms: f64,
    pub batch_size: usize,
    pub concurrency: usize,
    pub accepted_tokens: usize,
    #[serde(rename = "A")]
    pub accepted_per_verification: usize,
    #[serde(rename = "U")]
    pub unique_experts: usize,
    pub performance_claim: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct RoutedLayerArtifactBuildReport {
    pub schema_version: u32,
    pub semantic: &'static str,
    pub revision: &'static str,
    pub commit: String,
    pub checkpoint_verification_sha256: String,
    pub oracle_manifest_sha256: String,
    pub artifact_manifest_sha256: String,
    pub artifact_sha256: String,
    pub artifact_bytes: u64,
    pub page_bytes: usize,
    pub layer: usize,
    pub selected_experts: Vec<u32>,
    pub tensor_records: usize,
    pub construction_wall_ms: f64,
    pub fresh_verification_wall_ms: f64,
    pub safety_snapshots: Vec<SafetySnapshot>,
    pub exactness: &'static str,
    pub performance_claim: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct RoutedLayerArtifactTrial {
    pub repetition: usize,
    pub cache_state: &'static str,
    pub variant: &'static str,
    pub mapping_open_ms: f64,
    pub trusted_tensor_bind_ms: f64,
    pub initial_invalidation_ms: f64,
    pub layer_wall_ms: f64,
    pub final_release_ms: f64,
    pub activity: ProcessActivityDelta,
    pub installed_source_bytes: u64,
    pub sparse_repair_counts: [u64; 3],
    pub expert_tomography: Vec<ExpertTomography>,
    pub expert_diagnostics: Vec<MetalExpertDiagnostic>,
    pub routed_sha256: String,
    pub final_residual_sha256: String,
    pub routed_parity: NumericalParity,
    pub final_residual_parity: NumericalParity,
}

#[derive(Debug, Serialize)]
pub struct RoutedLayerArtifactBenchmarkReport {
    pub schema_version: u32,
    pub semantic: &'static str,
    pub revision: &'static str,
    pub commit: String,
    pub checkpoint_verification_sha256: String,
    pub oracle_manifest_sha256: String,
    pub artifact_manifest_sha256: String,
    pub artifact_sha256: String,
    pub kernel_sha256: String,
    pub kernel_compile_ms: f64,
    pub metal_device: String,
    pub no_copy_probe_ms: f64,
    pub no_copy_probe_passed: bool,
    pub warm_prefault_ms: f64,
    pub warm_prefault_checksum: u64,
    pub selected_experts: Vec<u32>,
    pub route_weights: Vec<f32>,
    pub maximum_route_weight_absolute_error: f32,
    pub trials: Vec<RoutedLayerArtifactTrial>,
    pub safety_snapshots: Vec<SafetySnapshot>,
    pub batch_size: usize,
    pub concurrency: usize,
    pub accepted_tokens: usize,
    #[serde(rename = "A")]
    pub accepted_per_verification: usize,
    #[serde(rename = "U")]
    pub unique_experts: usize,
    pub exactness: &'static str,
    pub performance_claim: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct CaptureRecord {
    pub file: String,
    pub shape: Vec<usize>,
    pub dtype: &'static str,
    pub sha256: String,
}

#[derive(Debug, Serialize)]
pub struct Layer0TraceReport {
    pub schema_version: u32,
    pub semantic: &'static str,
    pub revision: &'static str,
    pub commit: String,
    pub fixture_sha256: String,
    pub checkpoint_verification_sha256: String,
    pub prompt_token_ids: Vec<u32>,
    pub numerics: &'static str,
    pub captures: BTreeMap<String, CaptureRecord>,
    pub ledger: EndpointLedger,
    pub safety_snapshots: Vec<SafetySnapshot>,
    pub complete_wall_ms: f64,
    pub batch_size: usize,
    pub prompt_positions: usize,
    pub concurrency: usize,
    pub accepted_tokens: usize,
    pub performance_claim: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct Layer1RoutingTraceReport {
    pub schema_version: u32,
    pub semantic: &'static str,
    pub revision: &'static str,
    pub commit: String,
    pub fixture_sha256: String,
    pub checkpoint_verification_sha256: String,
    pub prompt_token_ids: Vec<u32>,
    pub source_input_sha256: String,
    pub numerics: &'static str,
    pub captures: BTreeMap<String, CaptureRecord>,
    pub selected_experts_by_position: Vec<Vec<u32>>,
    pub route_weights_by_position: Vec<Vec<f32>>,
    pub ledger: EndpointLedger,
    pub safety_snapshots: Vec<SafetySnapshot>,
    pub complete_wall_ms: f64,
    pub batch_size: usize,
    pub prompt_positions: usize,
    pub concurrency: usize,
    pub accepted_tokens: usize,
    pub performance_claim: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct ExpertScheduleEntry {
    pub expert: u32,
    pub positions: Vec<usize>,
}

#[derive(Debug, Serialize)]
pub struct Layer1ExpertTraceReport {
    pub schema_version: u32,
    pub semantic: &'static str,
    pub revision: &'static str,
    pub commit: String,
    pub fixture_sha256: String,
    pub checkpoint_verification_sha256: String,
    pub prompt_token_ids: Vec<u32>,
    pub source_input_sha256: String,
    pub numerics: &'static str,
    pub captures: BTreeMap<String, CaptureRecord>,
    pub selected_experts_by_position: Vec<Vec<u32>>,
    pub route_weights_by_position: Vec<Vec<f32>>,
    pub expert_schedule: Vec<ExpertScheduleEntry>,
    pub ledger: EndpointLedger,
    pub safety_snapshots: Vec<SafetySnapshot>,
    pub complete_wall_ms: f64,
    pub batch_size: usize,
    pub prompt_positions: usize,
    pub concurrency: usize,
    pub accepted_tokens: usize,
    pub performance_claim: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct FullPrefixTraceReport {
    pub schema_version: u32,
    pub semantic: &'static str,
    pub revision: &'static str,
    pub commit: String,
    pub fixture_sha256: String,
    pub checkpoint_verification_sha256: String,
    pub prompt_token_ids: Vec<u32>,
    pub numerics: &'static str,
    pub captures: BTreeMap<String, CaptureRecord>,
    pub layer_traces: Vec<LayerRouteTrace>,
    pub ledger: EndpointLedger,
    pub safety_snapshots: Vec<SafetySnapshot>,
    pub complete_wall_ms: f64,
    pub batch_size: usize,
    pub prompt_positions: usize,
    pub concurrency: usize,
    pub accepted_tokens: usize,
    pub performance_claim: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct SafetySnapshot {
    pub phase: String,
    pub system_memory_free_percent: u64,
    pub swap_used_bytes: u64,
    pub swap_growth_bytes: u64,
    pub throttled_pages: u64,
    pub new_throttled_pages: u64,
    pub process_physical_footprint_bytes: u64,
    pub process_peak_resident_bytes: u64,
    pub malloc_pressure_relief_bytes: u64,
    pub protected_service_pids: BTreeMap<String, Vec<u32>>,
}

struct SafetyMonitor {
    policy: SafetyFixture,
    baseline_swap_bytes: u64,
    baseline_throttled_pages: u64,
    baseline_services: BTreeSet<String>,
    snapshots: Vec<SafetySnapshot>,
}

pub(crate) struct ComponentSafetyMonitor(SafetyMonitor);

impl ComponentSafetyMonitor {
    pub(crate) fn start_normative() -> Result<Self, String> {
        SafetyMonitor::start(SafetyFixture {
            minimum_system_memory_free_percent: 20,
            maximum_process_physical_footprint_bytes: 8 * 1024 * 1024 * 1024,
            maximum_post_phase_physical_footprint_bytes: 4 * 1024 * 1024 * 1024,
            maximum_swap_growth_bytes: 512 * 1024 * 1024,
            maximum_new_throttled_pages: 0,
            require_malloc_pressure_relief: true,
            protect_resident_services: vec![
                "ChatGPT".to_owned(),
                "WindowServer".to_owned(),
                "nxnode".to_owned(),
                "syncthing".to_owned(),
            ],
        })
        .map(Self)
    }

    pub(crate) fn checkpoint(&mut self, phase: &str) -> Result<(), String> {
        self.0.checkpoint(phase, false)
    }

    pub(crate) fn released(mut self) -> Result<Vec<SafetySnapshot>, String> {
        self.0.checkpoint("buffer_release", true)?;
        Ok(self.0.snapshots)
    }
}

struct RoutedMlpOutput {
    output: Vec<f32>,
    logits: Vec<f32>,
    scores: Vec<f32>,
    selected: Vec<Vec<u32>>,
    weights: Vec<Vec<f32>>,
}

struct RoutingTrace {
    logits: Vec<f32>,
    scores: Vec<f32>,
    selected: Vec<Vec<u32>>,
    weights: Vec<Vec<f32>>,
}

#[derive(Default)]
struct ExpertCaptures {
    schedule: Vec<ExpertScheduleEntry>,
    gate: Vec<f32>,
    up: Vec<f32>,
    swiglu: Vec<f32>,
    down: Vec<f32>,
}

#[derive(Default)]
struct FullPrefixCaptures {
    embedding: Vec<f32>,
    layer_finals: Vec<Vec<f32>>,
    final_norm: Vec<f32>,
}

struct NativeDecodeStep {
    output_token: u32,
    top_logits: Vec<(u32, f32)>,
    full_logits: Vec<f32>,
    traces: Vec<LayerRouteTrace>,
    wall_ms: f64,
}

struct EndpointAuthority {
    fixture_bytes: Vec<u8>,
    fixture: EndpointFixture,
    safety: SafetyMonitor,
    config: ModelConfig,
    tokenizer: Tokenizer,
    prompt_token_ids: Vec<u32>,
    checkpoint: Checkpoint,
    verification_sha256: String,
}

#[derive(Default)]
struct Layer0Captures {
    qkv: Vec<f32>,
    query: Vec<f32>,
    key: Vec<f32>,
    value: Vec<f32>,
    sinks: Vec<f32>,
    attention_scores: Vec<f32>,
    attention_probabilities: Vec<f32>,
    attention: Vec<f32>,
    attention_projection: Vec<f32>,
    gate: Vec<f32>,
    up: Vec<f32>,
    swiglu: Vec<f32>,
    down: Vec<f32>,
}

#[derive(Default)]
struct AttentionHeadTrace {
    scores: Vec<f32>,
    probabilities: Vec<f32>,
}

struct CheckpointTensorSource<'a> {
    view: MappedTensorView<'a>,
    shard: &'a str,
    shard_sha256: &'a str,
    absolute_offsets: [u64; 2],
}

struct Checkpoint {
    weight_map: BTreeMap<String, String>,
    shards: BTreeMap<String, MappedSafetensors>,
    shard_sha256: BTreeMap<String, String>,
}

impl Checkpoint {
    fn open(
        root: &Path,
        index_path: &Path,
        verification: &CheckpointVerification,
    ) -> Result<Self, String> {
        let bytes =
            fs::read(index_path).map_err(|error| format!("{}: {error}", index_path.display()))?;
        let unique: UniqueJson =
            serde_json::from_slice(&bytes).map_err(|error| format!("checkpoint index: {error}"))?;
        let object = unique
            .0
            .as_object()
            .ok_or("checkpoint index must be an object")?;
        let map = object
            .get("weight_map")
            .and_then(Value::as_object)
            .ok_or("checkpoint index lacks weight_map")?;
        let mut weight_map = BTreeMap::new();
        let mut shard_names = BTreeSet::new();
        for (tensor, shard) in map {
            let shard = shard
                .as_str()
                .ok_or_else(|| format!("{tensor}: shard is not a string"))?;
            validate_relative_file(shard)?;
            if weight_map
                .insert(tensor.clone(), shard.to_owned())
                .is_some()
            {
                return Err(format!("duplicate checkpoint tensor: {tensor}"));
            }
            shard_names.insert(shard.to_owned());
        }
        if weight_map.len() != 73_081 || shard_names.len() != 17 {
            return Err("checkpoint index tensor or shard count mismatch".to_owned());
        }
        let verified = verification
            .files
            .iter()
            .map(|record| (record.path.as_str(), record))
            .collect::<BTreeMap<_, _>>();
        let mut shards = BTreeMap::new();
        let mut shard_sha256 = BTreeMap::new();
        for shard in shard_names {
            let record = verified
                .get(shard.as_str())
                .ok_or_else(|| format!("indexed shard absent from verification: {shard}"))?;
            verify_live_identity(root, record)?;
            let mapped = MappedSafetensors::open(&root.join(&shard))?;
            shard_sha256.insert(shard.clone(), record.sha256.clone());
            shards.insert(shard, mapped);
        }
        for (tensor, shard) in &weight_map {
            if !shards
                .get(shard)
                .ok_or_else(|| format!("mapped shard absent: {shard}"))?
                .tensors
                .contains_key(tensor)
            {
                return Err(format!("{tensor}: index tensor absent from {shard}"));
            }
        }
        for (shard, mapped) in &shards {
            for tensor in mapped.tensors.keys() {
                if weight_map.get(tensor) != Some(shard) {
                    return Err(format!(
                        "{tensor}: tensor exists outside its indexed shard {shard}"
                    ));
                }
            }
        }
        Ok(Self {
            weight_map,
            shards,
            shard_sha256,
        })
    }

    fn tensor(&self, name: &str) -> Result<MappedTensorView<'_>, String> {
        let shard = self
            .weight_map
            .get(name)
            .ok_or_else(|| format!("tensor absent from checkpoint index: {name}"))?;
        self.shards
            .get(shard)
            .ok_or_else(|| format!("mapped shard absent: {shard}"))?
            .tensor(name)
    }

    fn shard_for_tensor(&self, name: &str) -> Result<&str, String> {
        self.weight_map
            .get(name)
            .map(String::as_str)
            .ok_or_else(|| format!("tensor absent from checkpoint index: {name}"))
    }

    fn source_tensor(&self, name: &str) -> Result<CheckpointTensorSource<'_>, String> {
        let shard = self.shard_for_tensor(name)?;
        let mapped = self
            .shards
            .get(shard)
            .ok_or_else(|| format!("mapped shard absent: {shard}"))?;
        let view = mapped.tensor(name)?;
        let start = (mapped.payload_start as u64)
            .checked_add(view.metadata.data_offsets[0])
            .ok_or("source tensor absolute start overflow")?;
        let end = (mapped.payload_start as u64)
            .checked_add(view.metadata.data_offsets[1])
            .ok_or("source tensor absolute end overflow")?;
        Ok(CheckpointTensorSource {
            view,
            shard,
            shard_sha256: self
                .shard_sha256
                .get(shard)
                .map(String::as_str)
                .ok_or_else(|| format!("verified shard hash absent: {shard}"))?,
            absolute_offsets: [start, end],
        })
    }

    fn release_file_pages(&self) -> Result<(), String> {
        for (shard, mapped) in &self.shards {
            // SAFETY: the pointer and length describe this live, immutable file mapping.
            // MS_INVALIDATE discards clean cached data and MADV_DONTNEED marks the same
            // address range as cold. Later tensor reads remain valid and fault the
            // authoritative checkpoint bytes back in.
            let invalidate_result = unsafe {
                libc::msync(
                    mapped.mapping.as_ptr().cast_mut().cast(),
                    mapped.mapping.len(),
                    libc::MS_INVALIDATE,
                )
            };
            if invalidate_result != 0 {
                return Err(format!(
                    "{shard}: checkpoint cache invalidation failed: {}",
                    std::io::Error::last_os_error()
                ));
            }
            let advise_result = unsafe {
                libc::madvise(
                    mapped.mapping.as_ptr().cast_mut().cast(),
                    mapped.mapping.len(),
                    libc::MADV_DONTNEED,
                )
            };
            if advise_result != 0 {
                return Err(format!(
                    "{shard}: checkpoint page release failed: {}",
                    std::io::Error::last_os_error()
                ));
            }
        }
        Ok(())
    }
}

#[derive(Default)]
struct LayerKvCache {
    keys: Vec<f32>,
    values: Vec<f32>,
    positions: usize,
    kv_heads: usize,
}

impl LayerKvCache {
    fn validate(&self) -> Result<(), String> {
        if self.positions == 0 {
            if !self.keys.is_empty() || !self.values.is_empty() {
                return Err("empty K/V cache contains data".to_owned());
            }
            return Ok(());
        }
        if self.kv_heads == 0
            || self.keys.len() != self.positions * self.kv_heads * QK_HEAD_DIM
            || self.values.len() != self.positions * self.kv_heads * V_HEAD_DIM
            || self
                .keys
                .iter()
                .chain(&self.values)
                .any(|value| !value.is_finite())
        {
            return Err("retained K/V cache identity mismatch".to_owned());
        }
        Ok(())
    }
}

fn validate_relative_file(name: &str) -> Result<(), String> {
    let path = Path::new(name);
    if path.components().count() != 1
        || !matches!(path.components().next(), Some(Component::Normal(_)))
    {
        return Err(format!("unsafe checkpoint shard path: {name}"));
    }
    Ok(())
}

fn verify_live_identity(root: &Path, record: &VerifiedFile) -> Result<(), String> {
    if record.status != "verified" {
        return Err(format!(
            "{} did not pass checkpoint verification",
            record.path
        ));
    }
    validate_relative_file(&record.path)?;
    let metadata = root
        .join(&record.path)
        .metadata()
        .map_err(|error| format!("{}: {error}", record.path))?;
    let modified_ns =
        i128::from(metadata.mtime()) * 1_000_000_000_i128 + i128::from(metadata.mtime_nsec());
    if metadata.len() != record.bytes
        || metadata.dev() != record.device
        || metadata.ino() != record.inode
        || modified_ns != record.modified_ns
        || record.sha256.len() != 64
    {
        return Err(format!(
            "{} identity changed after checkpoint verification",
            record.path
        ));
    }
    Ok(())
}

fn hash_file(path: &Path) -> Result<String, String> {
    let mut file = File::open(path).map_err(|error| format!("{}: {error}", path.display()))?;
    sha256_reader(&mut file)
}

#[repr(C)]
#[derive(Default)]
struct RusageInfoV2 {
    uuid: [u8; 16],
    user_time: u64,
    system_time: u64,
    pkg_idle_wkups: u64,
    interrupt_wkups: u64,
    pageins: u64,
    wired_size: u64,
    resident_size: u64,
    phys_footprint: u64,
    proc_start_abstime: u64,
    proc_exit_abstime: u64,
    child_user_time: u64,
    child_system_time: u64,
    child_pkg_idle_wkups: u64,
    child_interrupt_wkups: u64,
    child_pageins: u64,
    child_elapsed_abstime: u64,
    diskio_bytesread: u64,
    diskio_byteswritten: u64,
}

#[derive(Clone, Copy, Debug, Default, Serialize)]
pub(crate) struct ProcessActivity {
    pub disk_bytes_read: u64,
    pub pageins: u64,
    pub minor_faults: u64,
    pub major_faults: u64,
    pub user_cpu_us: u64,
    pub system_cpu_us: u64,
}

#[derive(Clone, Copy, Debug, Default, Serialize)]
pub struct ProcessActivityDelta {
    pub disk_bytes_read: u64,
    pub pageins: u64,
    pub minor_faults: i64,
    pub major_faults: i64,
    pub user_cpu_us: u64,
    pub system_cpu_us: u64,
}

impl ProcessActivity {
    pub(crate) fn checked_delta(self, earlier: Self) -> Result<ProcessActivityDelta, String> {
        let delta = |later: u64, before: u64, name: &str| {
            later.checked_sub(before).ok_or_else(|| {
                format!(
                    "process activity counter moved backwards: {name} later={later} before={before}"
                )
            })
        };
        let signed_delta = |later: u64, before: u64, name: &str| {
            let difference = i128::from(later) - i128::from(before);
            i64::try_from(difference)
                .map_err(|_| format!("process activity signed delta does not fit i64: {name}"))
        };
        Ok(ProcessActivityDelta {
            disk_bytes_read: delta(self.disk_bytes_read, earlier.disk_bytes_read, "disk read")?,
            pageins: delta(self.pageins, earlier.pageins, "pageins")?,
            // Darwin's getrusage fault counts can fall across multithreaded
            // Accelerate/Metal boundaries. Preserve the signed observation;
            // do not underflow or silently clamp it.
            minor_faults: signed_delta(self.minor_faults, earlier.minor_faults, "minor faults")?,
            major_faults: signed_delta(self.major_faults, earlier.major_faults, "major faults")?,
            user_cpu_us: delta(self.user_cpu_us, earlier.user_cpu_us, "user CPU")?,
            system_cpu_us: delta(self.system_cpu_us, earlier.system_cpu_us, "system CPU")?,
        })
    }
}

#[link(name = "proc")]
unsafe extern "C" {
    fn proc_pid_rusage(
        pid: libc::c_int,
        flavor: libc::c_int,
        buffer: *mut libc::c_void,
    ) -> libc::c_int;
}

fn process_usage() -> Result<RusageInfoV2, String> {
    let mut usage = RusageInfoV2::default();
    // SAFETY: `usage` has Darwin's rusage_info_v2 layout and is exclusively borrowed.
    let result = unsafe {
        proc_pid_rusage(
            std::process::id() as libc::c_int,
            2,
            (&mut usage as *mut RusageInfoV2).cast(),
        )
    };
    if result != 0 {
        return Err(format!("proc_pid_rusage failed with {result}"));
    }
    Ok(usage)
}

pub(crate) fn process_activity() -> Result<ProcessActivity, String> {
    let process = process_usage()?;
    // SAFETY: Darwin initializes the complete rusage structure for RUSAGE_SELF.
    let mut usage = unsafe { std::mem::zeroed::<libc::rusage>() };
    // SAFETY: `usage` is a valid exclusive output pointer.
    if unsafe { libc::getrusage(libc::RUSAGE_SELF, &mut usage) } != 0 {
        return Err("getrusage(RUSAGE_SELF) failed".to_owned());
    }
    let nonnegative = |value: libc::c_long, name: &str| {
        u64::try_from(value).map_err(|_| format!("negative process activity counter: {name}"))
    };
    let timeval_us = |value: libc::timeval, name: &str| -> Result<u64, String> {
        let seconds = nonnegative(value.tv_sec, name)?;
        let micros = nonnegative(value.tv_usec.into(), name)?;
        seconds
            .checked_mul(1_000_000)
            .and_then(|total| total.checked_add(micros))
            .ok_or_else(|| format!("process CPU counter overflow: {name}"))
    };
    Ok(ProcessActivity {
        disk_bytes_read: process.diskio_bytesread,
        pageins: process.pageins,
        minor_faults: nonnegative(usage.ru_minflt, "minor faults")?,
        major_faults: nonnegative(usage.ru_majflt, "major faults")?,
        user_cpu_us: timeval_us(usage.ru_utime, "user CPU")?,
        system_cpu_us: timeval_us(usage.ru_stime, "system CPU")?,
    })
}

fn process_disk_bytes_read() -> Result<u64, String> {
    Ok(process_usage()?.diskio_bytesread)
}

fn peak_resident_bytes() -> Result<u64, String> {
    // SAFETY: Darwin initializes the complete rusage structure for RUSAGE_SELF.
    let mut usage = unsafe { std::mem::zeroed::<libc::rusage>() };
    // SAFETY: `usage` is a valid exclusive output pointer.
    if unsafe { libc::getrusage(libc::RUSAGE_SELF, &mut usage) } != 0 {
        return Err("getrusage(RUSAGE_SELF) failed".to_owned());
    }
    u64::try_from(usage.ru_maxrss).map_err(|_| "negative peak resident set".to_owned())
}

unsafe extern "C" {
    fn malloc_zone_pressure_relief(zone: *mut libc::c_void, goal: usize) -> usize;
    fn pw_pytorch_topk_unsorted_f32(
        values: *const f32,
        count: usize,
        top_k: usize,
        selected: *mut u32,
    ) -> i32;
}

fn pressure_relief() -> u64 {
    // SAFETY: a null zone asks Darwin to visit every malloc zone; no Rust allocation is
    // accessed while the call runs and a zero goal requests all presently releasable bytes.
    unsafe { malloc_zone_pressure_relief(std::ptr::null_mut(), 0) as u64 }
}

fn release_matrix_transients(checkpoint: &Checkpoint) -> Result<(), String> {
    checkpoint.release_file_pages()?;
    pressure_relief();
    Ok(())
}

fn command_output(program: &str, arguments: &[&str]) -> Result<String, String> {
    let output = Command::new(program)
        .args(arguments)
        .output()
        .map_err(|error| format!("{program}: {error}"))?;
    if !output.status.success() {
        return Err(format!("{program} exited {}", output.status));
    }
    String::from_utf8(output.stdout).map_err(|error| format!("{program}: {error}"))
}

fn system_memory_free_percent() -> Result<u64, String> {
    let output = command_output("/usr/bin/memory_pressure", &["-Q"])?;
    let marker = "System-wide memory free percentage:";
    output
        .lines()
        .find_map(|line| line.trim().strip_prefix(marker))
        .and_then(|value| value.trim().strip_suffix('%'))
        .and_then(|value| value.parse::<u64>().ok())
        .ok_or("memory_pressure output lacks free percentage".to_owned())
}

fn swap_used_bytes() -> Result<u64, String> {
    let output = command_output("/usr/sbin/sysctl", &["-n", "vm.swapusage"])?;
    let used = output
        .split_whitespace()
        .collect::<Vec<_>>()
        .windows(3)
        .find_map(|fields| (fields[0] == "used" && fields[1] == "=").then_some(fields[2]))
        .ok_or("vm.swapusage output lacks used value")?;
    let (number, multiplier) = if let Some(value) = used.strip_suffix('M') {
        (value, 1024.0_f64 * 1024.0)
    } else if let Some(value) = used.strip_suffix('G') {
        (value, 1024.0_f64 * 1024.0 * 1024.0)
    } else {
        return Err("vm.swapusage used value has unknown unit".to_owned());
    };
    let bytes = number
        .parse::<f64>()
        .map_err(|error| format!("vm.swapusage used value: {error}"))?
        * multiplier;
    if !bytes.is_finite() || bytes < 0.0 || bytes > u64::MAX as f64 {
        return Err("vm.swapusage used value is invalid".to_owned());
    }
    Ok(bytes.round() as u64)
}

fn throttled_pages() -> Result<u64, String> {
    let output = command_output("/usr/bin/vm_stat", &[])?;
    output
        .lines()
        .find_map(|line| line.trim().strip_prefix("Pages throttled:"))
        .and_then(|value| value.trim().strip_suffix('.'))
        .and_then(|value| value.trim().parse::<u64>().ok())
        .ok_or("vm_stat output lacks throttled pages".to_owned())
}

fn protected_service_pids(names: &[String]) -> Result<BTreeMap<String, Vec<u32>>, String> {
    let output = command_output("/bin/ps", &["-axo", "pid=,comm="])?;
    let mut result = names
        .iter()
        .cloned()
        .map(|name| (name, Vec::new()))
        .collect::<BTreeMap<_, _>>();
    for line in output.lines() {
        let mut fields = line.trim().splitn(2, char::is_whitespace);
        let Some(pid) = fields.next().and_then(|value| value.parse::<u32>().ok()) else {
            continue;
        };
        let Some(command) = fields.next().map(str::trim) else {
            continue;
        };
        let basename = Path::new(command)
            .file_name()
            .and_then(|value| value.to_str())
            .unwrap_or(command);
        for (name, pids) in &mut result {
            if basename == name {
                pids.push(pid);
            }
        }
    }
    Ok(result)
}

impl SafetyMonitor {
    fn start(policy: SafetyFixture) -> Result<Self, String> {
        let baseline_swap_bytes = swap_used_bytes()?;
        let baseline_throttled_pages = throttled_pages()?;
        let services = protected_service_pids(&policy.protect_resident_services)?;
        let baseline_services = services
            .iter()
            .filter_map(|(name, pids)| (!pids.is_empty()).then_some(name.clone()))
            .collect();
        let mut monitor = Self {
            policy,
            baseline_swap_bytes,
            baseline_throttled_pages,
            baseline_services,
            snapshots: Vec::new(),
        };
        monitor.checkpoint("process_start", false)?;
        Ok(monitor)
    }

    fn checkpoint(&mut self, phase: &str, relieve: bool) -> Result<(), String> {
        let relief = if relieve { pressure_relief() } else { 0 };
        let memory_free = system_memory_free_percent()?;
        let swap = swap_used_bytes()?;
        let throttled = throttled_pages()?;
        let usage = process_usage()?;
        let peak = peak_resident_bytes()?;
        let services = protected_service_pids(&self.policy.protect_resident_services)?;
        let swap_growth = swap.saturating_sub(self.baseline_swap_bytes);
        let new_throttled = throttled.saturating_sub(self.baseline_throttled_pages);
        let snapshot = SafetySnapshot {
            phase: phase.to_owned(),
            system_memory_free_percent: memory_free,
            swap_used_bytes: swap,
            swap_growth_bytes: swap_growth,
            throttled_pages: throttled,
            new_throttled_pages: new_throttled,
            process_physical_footprint_bytes: usage.phys_footprint,
            process_peak_resident_bytes: peak,
            malloc_pressure_relief_bytes: relief,
            protected_service_pids: services.clone(),
        };
        self.snapshots.push(snapshot);
        if memory_free < self.policy.minimum_system_memory_free_percent {
            return Err(format!(
                "safety stop at {phase}: system memory free is {memory_free}%"
            ));
        }
        if usage.phys_footprint > self.policy.maximum_process_physical_footprint_bytes
            || peak > self.policy.maximum_process_physical_footprint_bytes
        {
            return Err(format!(
                "safety stop at {phase}: process footprint limit exceeded (current={}, peak={}, limit={})",
                usage.phys_footprint, peak, self.policy.maximum_process_physical_footprint_bytes
            ));
        }
        if relieve && usage.phys_footprint > self.policy.maximum_post_phase_physical_footprint_bytes
        {
            return Err(format!(
                "safety stop at {phase}: post-phase footprint limit exceeded"
            ));
        }
        if swap_growth > self.policy.maximum_swap_growth_bytes {
            return Err(format!(
                "safety stop at {phase}: swap growth limit exceeded"
            ));
        }
        if new_throttled > self.policy.maximum_new_throttled_pages {
            return Err(format!("safety stop at {phase}: VM throttling observed"));
        }
        for name in &self.baseline_services {
            if services.get(name).is_none_or(Vec::is_empty) {
                return Err(format!(
                    "safety stop at {phase}: resident service {name} disappeared"
                ));
            }
        }
        Ok(())
    }
}

fn validate_fixture(fixture: &EndpointFixture) -> Result<(), String> {
    let hosted_identity = fixture.hosted_reference.as_ref().is_some_and(|hosted| {
        hosted.provider == "Parasail"
            && hosted.manifest_sha256
                == "f9c5dd42a76e0eb87581fa427fe03c69ad32903c5711e5078a002ab7514732ea"
            && hosted.response_sha256
                == "e5a8956f3a7985e1ac3d5396c7bc9fe73bc77c6451eb2225c8df7c8973e3212d"
            && hosted.generated_token_ids == [9707, 0]
            && hosted.generated_text == "Hello!"
    });
    let raw_identity = fixture.schema_version == 1
        && fixture.semantic == "mimo_v2_5_target_faithful_raw_text_incremental_decode"
        && fixture.prompt_utf8 == "Hello"
        && fixture.expected_prompt_token_ids == [9707]
        && fixture.full_prefix_trace_append_token_ids.is_none()
        && fixture.hosted_reference.is_none();
    let chat_identity = fixture.schema_version == 2
        && fixture.semantic == "mimo_v2_5_target_faithful_chat_prefill_incremental_decode"
        && fixture.prompt_utf8 == CHAT_PROMPT
        && fixture.expected_prompt_token_ids == CHAT_PROMPT_IDS
        && fixture.full_prefix_trace_append_token_ids.is_none()
        && hosted_identity;
    let trace_identity = fixture.schema_version == 3
        && fixture.semantic == "mimo_v2_5_target_faithful_whole_sequence_trace"
        && fixture.prompt_utf8 == CHAT_PROMPT
        && fixture.expected_prompt_token_ids == CHAT_PROMPT_IDS
        && fixture.full_prefix_trace_append_token_ids.as_deref() == Some(&[264])
        && hosted_identity;
    if (!raw_identity && !chat_identity && !trace_identity)
        || fixture.revision != REVISION
        || fixture.add_special_tokens
        || fixture.full_attention_qkv_scale_layout.weight_shape != [13_568, 4096]
        || fixture.full_attention_qkv_scale_layout.scale_shape != [108, 32]
        || fixture.full_attention_qkv_scale_layout.query_rows != 12_288
        || fixture.full_attention_qkv_scale_layout.query_scale_rows != 96
        || fixture.full_attention_qkv_scale_layout.key_heads != 4
        || fixture.full_attention_qkv_scale_layout.key_rows_per_head != 192
        || fixture
            .full_attention_qkv_scale_layout
            .key_scale_rows_per_head
            != 2
        || fixture.full_attention_qkv_scale_layout.key_scale_row_start != 96
        || fixture.full_attention_qkv_scale_layout.value_heads != 4
        || fixture.full_attention_qkv_scale_layout.value_rows_per_head != 128
        || fixture
            .full_attention_qkv_scale_layout
            .value_scale_row_start
            != 104
        || fixture.decode.sampling != "greedy"
        || fixture.decode.new_tokens != 2
        || fixture.decode.batch_size != 1
        || fixture.decode.concurrency != 1
        || !fixture.decode.use_kv_cache
        || fixture.safety.minimum_system_memory_free_percent != 20
        || fixture.safety.maximum_process_physical_footprint_bytes != 8 * 1024 * 1024 * 1024
        || fixture.safety.maximum_post_phase_physical_footprint_bytes != 4 * 1024 * 1024 * 1024
        || fixture.safety.maximum_swap_growth_bytes != 512 * 1024 * 1024
        || fixture.safety.maximum_new_throttled_pages != 0
        || !fixture.safety.require_malloc_pressure_relief
        || fixture.safety.protect_resident_services
            != ["ChatGPT", "WindowServer", "nxnode", "syncthing"]
    {
        return Err("unknown slow text endpoint fixture identity".to_owned());
    }
    Ok(())
}

fn validate_slow_endpoint_fixture(fixture: &EndpointFixture) -> Result<(), String> {
    if fixture.full_prefix_trace_append_token_ids.is_some() {
        return Err("slow endpoint rejects full-prefix trace-only fixture".to_owned());
    }
    Ok(())
}

fn full_prefix_trace_tokens(
    fixture: &EndpointFixture,
    prompt_token_ids: &[u32],
) -> Result<Vec<u32>, String> {
    if !matches!(fixture.schema_version, 2 | 3) || prompt_token_ids != CHAT_PROMPT_IDS {
        return Err("full-prefix trace requires the frozen chat fixture".to_owned());
    }
    let mut tokens = prompt_token_ids.to_vec();
    if let Some(appended) = fixture.full_prefix_trace_append_token_ids.as_ref() {
        tokens.extend(appended);
    }
    Ok(tokens)
}

fn validate_config(config: &ModelConfig) -> Result<(), String> {
    let full_layers = config
        .hybrid_layer_pattern
        .iter()
        .enumerate()
        .filter_map(|(layer, &kind)| (kind == 0).then_some(layer))
        .collect::<Vec<_>>();
    if config.model_type != "mimo_v2"
        || config.dtype != "bfloat16"
        || config.hidden_size != HIDDEN
        || config.num_hidden_layers != 48
        || config.num_attention_heads != HEADS
        || config.num_key_value_heads != 4
        || config.head_dim != QK_HEAD_DIM
        || config.v_head_dim != V_HEAD_DIM
        || config.swa_num_attention_heads != HEADS
        || config.swa_num_key_value_heads != 8
        || config.swa_head_dim != QK_HEAD_DIM
        || config.swa_v_head_dim != V_HEAD_DIM
        || (config.partial_rotary_factor * QK_HEAD_DIM as f64) as usize != ROPE_DIM
        || config.rope_theta != 10_000_000.0
        || config.swa_rope_theta != 10_000.0
        || config.sliding_window != 128
        || config.attention_value_scale != 0.707
        || config.add_full_attention_sink_bias
        || !config.add_swa_attention_sink_bias
        || config.attention_projection_layout != "fused_qkv"
        || config.intermediate_size != 16_384
        || config.moe_intermediate_size != MOE_INTERMEDIATE
        || config.n_routed_experts != ROUTED_EXPERTS
        || config.num_experts_per_tok != TOP_K
        || config.n_group != 1
        || config.topk_group != 1
        || !config.norm_topk_prob
        || config.routed_scaling_factor.is_some()
        || config.scoring_func != "sigmoid"
        || config.topk_method != "noaux_tc"
        || config.layernorm_epsilon != 1.0e-5
        || config.vocab_size != 152_576
        || config.hybrid_layer_pattern.len() != 48
        || config.moe_layer_freq.len() != 48
        || config.moe_layer_freq[0] != 0
        || config.moe_layer_freq[1..].iter().any(|&value| value != 1)
        || config.tie_word_embeddings
        || config.quantization_config.activation_scheme != "dynamic"
        || config.quantization_config.fmt != "e4m3"
        || config.quantization_config.quant_method != "fp8"
        || config.quantization_config.store_dtype != "fp8"
        || config.quantization_config.weight_block_size != [128, 128]
        || full_layers != [0, 5, 11, 17, 23, 29, 35, 41, 47]
    {
        return Err("pinned MiMo text config identity mismatch".to_owned());
    }
    Ok(())
}

fn decode_f32(view: MappedTensorView<'_>, shape: &[u64]) -> Result<Vec<f32>, String> {
    if view.metadata.dtype != "F32" || view.metadata.shape != shape {
        return Err(format!("{} F32 tensor layout mismatch", view.metadata.name));
    }
    let values = view
        .bytes
        .chunks_exact(4)
        .map(|bytes| f32::from_le_bytes(bytes.try_into().expect("four-byte F32")))
        .collect::<Vec<_>>();
    if values.iter().any(|value| !value.is_finite()) {
        return Err(format!("{} contains non-finite F32", view.metadata.name));
    }
    Ok(values)
}

fn bf16_vector(
    checkpoint: &Checkpoint,
    name: &str,
    count: usize,
    ledger: &mut EndpointLedger,
) -> Result<Vec<f32>, String> {
    let view = checkpoint.tensor(name)?;
    ledger.logical_source_bytes = ledger
        .logical_source_bytes
        .checked_add(view.metadata.data_bytes)
        .ok_or("logical byte ledger overflow")?;
    decode_bf16_tensor(view, count)
}

fn f32_vector(
    checkpoint: &Checkpoint,
    name: &str,
    shape: &[u64],
    ledger: &mut EndpointLedger,
) -> Result<Vec<f32>, String> {
    let view = checkpoint.tensor(name)?;
    ledger.logical_source_bytes = ledger
        .logical_source_bytes
        .checked_add(view.metadata.data_bytes)
        .ok_or("logical byte ledger overflow")?;
    decode_f32(view, shape)
}

fn rms_norm(
    values: &[f32],
    rows: usize,
    weights: &[f32],
    epsilon: f32,
) -> Result<Vec<f32>, String> {
    if values.len() != rows * HIDDEN || weights.len() != HIDDEN {
        return Err("RMSNorm shape mismatch".to_owned());
    }
    let mut output = vec![0.0_f32; values.len()];
    for row in 0..rows {
        let source = &values[row * HIDDEN..(row + 1) * HIDDEN];
        let inverse = stable_rms_inverse(source, epsilon)?;
        for column in 0..HIDDEN {
            let normalized = round_bf16(source[column] * inverse);
            output[row * HIDDEN + column] = round_bf16(normalized * weights[column]);
        }
    }
    Ok(output)
}

fn round_bf16(value: f32) -> f32 {
    let bits = value.to_bits();
    if bits & 0x7f80_0000 == 0x7f80_0000 {
        if bits & 0x007f_ffff == 0 {
            return value;
        }
        let payload = ((bits >> 16) as u16) | 0x0040;
        return f32::from_bits(u32::from(payload) << 16);
    }
    let rounding_bias = 0x7fff + ((bits >> 16) & 1);
    f32::from_bits(bits.wrapping_add(rounding_bias) & 0xffff_0000)
}

fn round_bf16_values(values: &mut [f32]) {
    for value in values {
        *value = round_bf16(*value);
    }
}

struct DynamicFp8Activations {
    dequantized: Vec<f32>,
    scales: Vec<f32>,
    encoded: Vec<u8>,
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
    let mut best_distance = f32::INFINITY;
    for candidate in 0_u8..=0x7e {
        let decoded = super::decode_f8_e4m3fn(candidate);
        let distance = (magnitude - decoded).abs();
        if distance < best_distance
            || (distance == best_distance && candidate & 1 == 0 && best & 1 != 0)
        {
            best = candidate;
            best_distance = distance;
        }
    }
    Ok(best | if value.is_sign_negative() { 0x80 } else { 0 })
}

fn dynamic_fp8_activations(
    input: &[f32],
    rows: usize,
    columns: usize,
) -> Result<DynamicFp8Activations, String> {
    const GROUP: usize = 128;
    const EPSILON: f32 = 1.0e-10;
    const FP8_MAX: f32 = 448.0;
    if rows == 0
        || columns == 0
        || !columns.is_multiple_of(GROUP)
        || input.len() != rows * columns
        || input.iter().any(|value| !value.is_finite())
    {
        return Err("dynamic FP8 activation shape or value mismatch".to_owned());
    }
    let mut dequantized = Vec::with_capacity(input.len());
    let mut encoded = Vec::with_capacity(input.len());
    let mut scales = Vec::with_capacity(rows * columns / GROUP);
    for row in input.chunks_exact(columns) {
        for group in row.chunks_exact(GROUP) {
            let absmax = group
                .iter()
                .map(|value| value.abs())
                .fold(0.0_f32, f32::max)
                .max(EPSILON);
            let scale = absmax / FP8_MAX;
            if !scale.is_finite() || scale <= 0.0 {
                return Err("dynamic FP8 activation scale is invalid".to_owned());
            }
            scales.push(scale);
            for &value in group {
                let byte = encode_f8_e4m3fn((value / scale).clamp(-FP8_MAX, FP8_MAX))?;
                encoded.push(byte);
                dequantized.push(super::decode_f8_e4m3fn(byte) * scale);
            }
        }
    }
    Ok(DynamicFp8Activations {
        dequantized,
        scales,
        encoded,
    })
}

fn fp8_linear(
    checkpoint: &Checkpoint,
    weight_name: &str,
    input: &[f32],
    rows: usize,
    columns: usize,
    output_columns: usize,
    ledger: &mut EndpointLedger,
) -> Result<Vec<f32>, String> {
    if input.len() != rows * columns || rows == 0 {
        return Err(format!("{weight_name}: FP8 linear input shape mismatch"));
    }
    let output = {
        let weight = checkpoint.tensor(weight_name)?;
        let scale_name = format!("{weight_name}_scale_inv");
        let scale = checkpoint.tensor(&scale_name)?;
        let validated = validate_fp8_views(weight, scale, &input[..columns])?;
        if validated.rows != output_columns || validated.columns != columns {
            return Err(format!("{weight_name}: FP8 linear weight shape mismatch"));
        }
        ledger.logical_source_bytes = ledger
            .logical_source_bytes
            .checked_add(validated.weight.metadata.data_bytes + validated.scale.metadata.data_bytes)
            .ok_or("logical byte ledger overflow")?;
        ledger.fp8_matrices_expanded += 1;
        let quantized = dynamic_fp8_activations(input, rows, columns)?;
        ledger.dynamic_activation_groups += quantized.scales.len() as u64;
        ledger.dynamic_activation_values += quantized.encoded.len() as u64;
        let decoded = decode_fp8_matrix_f32(&validated);
        let mut output = accelerate_sgemm_right_transposed(
            &quantized.dequantized,
            &decoded,
            rows,
            output_columns,
            columns,
        )?;
        round_bf16_values(&mut output);
        output
    };
    release_matrix_transients(checkpoint)?;
    Ok(output)
}

fn full_qkv_scale_row(weight_row: usize) -> Result<usize, String> {
    const Q_ROWS: usize = HEADS * QK_HEAD_DIM;
    const K_ROWS: usize = 4 * QK_HEAD_DIM;
    const V_ROWS: usize = 4 * V_HEAD_DIM;
    if weight_row < Q_ROWS {
        return Ok(weight_row / 128);
    }
    if weight_row < Q_ROWS + K_ROWS {
        let local = weight_row - Q_ROWS;
        let head = local / QK_HEAD_DIM;
        let dimension = local % QK_HEAD_DIM;
        return Ok(96 + head * 2 + dimension / 128);
    }
    if weight_row < Q_ROWS + K_ROWS + V_ROWS {
        return Ok(104 + (weight_row - Q_ROWS - K_ROWS) / V_HEAD_DIM);
    }
    Err("full-QKV weight row is out of range".to_owned())
}

fn full_qkv_linear(
    checkpoint: &Checkpoint,
    weight_name: &str,
    input: &[f32],
    rows: usize,
    ledger: &mut EndpointLedger,
) -> Result<Vec<f32>, String> {
    const OUTPUT_ROWS: usize = HEADS * QK_HEAD_DIM + 4 * QK_HEAD_DIM + 4 * V_HEAD_DIM;
    if rows == 0 || input.len() != rows * HIDDEN {
        return Err(format!("{weight_name}: full-QKV input shape mismatch"));
    }
    let output = {
        let weight = checkpoint.tensor(weight_name)?;
        let scale_name = format!("{weight_name}_scale_inv");
        let scale = checkpoint.tensor(&scale_name)?;
        if weight.metadata.dtype != "F8_E4M3"
            || weight.metadata.shape != [OUTPUT_ROWS as u64, HIDDEN as u64]
            || scale.metadata.dtype != "F32"
            || scale.metadata.shape != [108, 32]
            || scale.bytes.len() != 108 * 32 * 4
        {
            return Err(format!("{weight_name}: unknown full-QKV FP8 scale layout"));
        }
        if let Some(offset) = weight
            .bytes
            .iter()
            .position(|bits| matches!(bits, 0x7f | 0xff))
        {
            return Err(format!(
                "{weight_name}: non-finite FP8 weight at byte offset {offset}"
            ));
        }
        let scales = scale
            .bytes
            .chunks_exact(4)
            .map(|bytes| f32::from_le_bytes(bytes.try_into().expect("four-byte F32 scale")))
            .collect::<Vec<_>>();
        if scales.iter().any(|value| !value.is_finite()) {
            return Err(format!("{weight_name}: full-QKV scale is non-finite"));
        }
        let mut decoded = Vec::with_capacity(weight.bytes.len());
        for weight_row in 0..OUTPUT_ROWS {
            let scale_row = full_qkv_scale_row(weight_row)?;
            for column in 0..HIDDEN {
                decoded.push(
                    super::decode_f8_e4m3fn(weight.bytes[weight_row * HIDDEN + column])
                        * scales[scale_row * 32 + column / 128],
                );
            }
        }
        ledger.logical_source_bytes = ledger
            .logical_source_bytes
            .checked_add(weight.metadata.data_bytes + scale.metadata.data_bytes)
            .ok_or("logical byte ledger overflow")?;
        ledger.fp8_matrices_expanded += 1;
        let quantized = dynamic_fp8_activations(input, rows, HIDDEN)?;
        ledger.dynamic_activation_groups += quantized.scales.len() as u64;
        ledger.dynamic_activation_values += quantized.encoded.len() as u64;
        let mut output = accelerate_sgemm_right_transposed(
            &quantized.dequantized,
            &decoded,
            rows,
            OUTPUT_ROWS,
            HIDDEN,
        )?;
        round_bf16_values(&mut output);
        output
    };
    release_matrix_transients(checkpoint)?;
    Ok(output)
}

fn bf16_linear(
    checkpoint: &Checkpoint,
    weight_name: &str,
    input: &[f32],
    rows: usize,
    columns: usize,
    output_columns: usize,
    ledger: &mut EndpointLedger,
) -> Result<Vec<f32>, String> {
    if input.len() != rows * columns || rows == 0 {
        return Err(format!("{weight_name}: BF16 linear input shape mismatch"));
    }
    let output = {
        let view = checkpoint.tensor(weight_name)?;
        if view.metadata.dtype != "BF16"
            || view.metadata.shape != [output_columns as u64, columns as u64]
        {
            return Err(format!("{weight_name}: BF16 linear weight shape mismatch"));
        }
        ledger.logical_source_bytes = ledger
            .logical_source_bytes
            .checked_add(view.metadata.data_bytes)
            .ok_or("logical byte ledger overflow")?;
        ledger.bf16_matrices_expanded += 1;
        let decoded = decode_bf16_tensor(view, output_columns * columns)?;
        let mut output =
            accelerate_sgemm_right_transposed(input, &decoded, rows, output_columns, columns)?;
        round_bf16_values(&mut output);
        output
    };
    release_matrix_transients(checkpoint)?;
    Ok(output)
}

fn bf16_last_row_linear(
    checkpoint: &Checkpoint,
    weight_name: &str,
    input: &[f32],
    columns: usize,
    output_columns: usize,
    ledger: &mut EndpointLedger,
) -> Result<Vec<f32>, String> {
    if input.len() != columns || columns == 0 {
        return Err(format!("{weight_name}: BF16 last-row input shape mismatch"));
    }
    let output = {
        let view = checkpoint.tensor(weight_name)?;
        if view.metadata.dtype != "BF16"
            || view.metadata.shape != [output_columns as u64, columns as u64]
        {
            return Err(format!(
                "{weight_name}: BF16 last-row weight shape mismatch"
            ));
        }
        ledger.logical_source_bytes = ledger
            .logical_source_bytes
            .checked_add(view.metadata.data_bytes)
            .ok_or("logical byte ledger overflow")?;
        ledger.bf16_matrices_expanded += 1;
        let decoded = decode_bf16_tensor(view, output_columns * columns)?;
        let mut output =
            accelerate_sgemm_right_transposed(input, &decoded, 1, output_columns, columns)?;
        round_bf16_values(&mut output);
        output
    };
    release_matrix_transients(checkpoint)?;
    Ok(output)
}

fn f32_linear(
    checkpoint: &Checkpoint,
    weight_name: &str,
    input: &[f32],
    rows: usize,
    columns: usize,
    output_columns: usize,
    ledger: &mut EndpointLedger,
) -> Result<Vec<f32>, String> {
    if input.len() != rows * columns || rows == 0 {
        return Err(format!("{weight_name}: F32 linear input shape mismatch"));
    }
    let view = checkpoint.tensor(weight_name)?;
    let decoded = decode_f32(view, &[output_columns as u64, columns as u64])?;
    ledger.logical_source_bytes = ledger
        .logical_source_bytes
        .checked_add((decoded.len() * 4) as u64)
        .ok_or("logical byte ledger overflow")?;
    accelerate_sgemm_right_transposed(input, &decoded, rows, output_columns, columns)
}

fn apply_rope(values: &mut [f32], heads: usize, position: usize, theta: f64) {
    for head in 0..heads {
        let offset = head * QK_HEAD_DIM;
        for pair in 0..ROPE_DIM / 2 {
            let angle = position as f64 / theta.powf(2.0 * pair as f64 / ROPE_DIM as f64);
            let cosine = round_bf16(angle.cos() as f32);
            let sine = round_bf16(angle.sin() as f32);
            let first = values[offset + pair];
            let second = values[offset + pair + ROPE_DIM / 2];
            let first_cosine = round_bf16(first * cosine);
            let second_sine = round_bf16(second * sine);
            let second_cosine = round_bf16(second * cosine);
            let first_sine = round_bf16(first * sine);
            values[offset + pair] = round_bf16(first_cosine - second_sine);
            values[offset + pair + ROPE_DIM / 2] = round_bf16(second_cosine + first_sine);
        }
    }
}

fn attention_softmax(scores: &[f32], bf16_output: bool) -> Result<Vec<f32>, String> {
    if scores.is_empty() || scores.iter().any(|value| !value.is_finite()) {
        return Err("attention softmax requires finite scores".to_owned());
    }
    let maximum = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let centered = scores
        .iter()
        .map(|score| {
            if bf16_output {
                round_bf16(*score - maximum)
            } else {
                *score - maximum
            }
        })
        .collect::<Vec<_>>();
    if bf16_output {
        return pytorch_arm_softmax_f32(&centered).map(|mut probabilities| {
            round_bf16_values(&mut probabilities);
            probabilities
        });
    }
    let mut probabilities = centered.iter().map(|score| score.exp()).collect::<Vec<_>>();
    let denominator = probabilities.iter().sum::<f32>();
    if !denominator.is_finite() || denominator <= 0.0 {
        return Err("attention softmax denominator is invalid".to_owned());
    }
    for probability in &mut probabilities {
        *probability /= denominator;
    }
    Ok(probabilities)
}

fn pytorch_arm_softmax_f32(centered: &[f32]) -> Result<Vec<f32>, String> {
    if centered.is_empty() || centered.iter().any(|value| !value.is_finite()) {
        return Err("PyTorch ARM softmax requires finite scores".to_owned());
    }
    let mut exponentials = centered
        .iter()
        .map(|value| sleef_expf_u10(*value))
        .collect::<Vec<_>>();
    let denominator = if exponentials.len() < 4 {
        exponentials[1..]
            .iter()
            .fold(exponentials[0], |sum, value| sum + value)
    } else {
        let full = exponentials.len() - exponentials.len() % 4;
        let mut lanes = [
            exponentials[0],
            exponentials[1],
            exponentials[2],
            exponentials[3],
        ];
        for chunk in exponentials[4..full].chunks_exact(4) {
            for lane in 0..4 {
                lanes[lane] += chunk[lane];
            }
        }
        for (lane, value) in exponentials[full..].iter().enumerate() {
            lanes[lane] += value;
        }
        (lanes[0] + lanes[2]) + (lanes[1] + lanes[3])
    };
    if !denominator.is_finite() || denominator <= 0.0 {
        return Err("PyTorch ARM softmax denominator is invalid".to_owned());
    }
    let inverse = 1.0_f32 / denominator;
    for exponential in &mut exponentials {
        *exponential *= inverse;
    }
    Ok(exponentials)
}

#[cfg(test)]
fn causal_attention_head(
    query: &[f32],
    keys: &[&[f32]],
    values: &[&[f32]],
    scale: f32,
    sink: Option<f32>,
) -> Result<Vec<f32>, String> {
    causal_attention_head_with_dtype(query, keys, values, scale, sink, false, None)
}

fn causal_attention_head_bf16(
    query: &[f32],
    keys: &[&[f32]],
    values: &[&[f32]],
    scale: f32,
    sink: Option<f32>,
) -> Result<Vec<f32>, String> {
    causal_attention_head_with_dtype(query, keys, values, scale, sink, true, None)
}

fn pytorch_bf16_four_lane_dot_f32(left: &[f32], right: &[f32]) -> f32 {
    debug_assert_eq!(left.len(), right.len());
    let mut partials = [0.0_f32; 4];
    let complete = left.len() / 4 * 4;
    for index in (0..complete).step_by(4) {
        partials[0] += left[index] * right[index];
        partials[1] += left[index + 1] * right[index + 1];
        partials[2] += left[index + 2] * right[index + 2];
        partials[3] += left[index + 3] * right[index + 3];
    }
    for index in complete..left.len() {
        partials[0] += left[index] * right[index];
    }
    partials[0] += partials[1];
    partials[0] += partials[2];
    partials[0] += partials[3];
    partials[0]
}

fn pytorch_bf16_specialized_vector_dot_f32(left: &[f32], right: &[f32]) -> f32 {
    debug_assert_eq!(left.len(), right.len());
    let mut accumulators = [[0.0_f32; 4]; 8];
    let complete_blocks = left.len() / 32 * 32;
    for block in (0..complete_blocks).step_by(32) {
        for (register, accumulator) in accumulators.iter_mut().enumerate() {
            for (lane, value) in accumulator.iter_mut().enumerate() {
                let index = block + register * 4 + lane;
                *value += left[index] * right[index];
            }
        }
    }
    for offset in [4, 2, 1] {
        for register in 0..offset {
            let source = accumulators[offset + register];
            for (target, source) in accumulators[register].iter_mut().zip(source) {
                *target += source;
            }
        }
    }
    let mut reduced =
        (accumulators[0][0] + accumulators[0][1]) + (accumulators[0][2] + accumulators[0][3]);
    let complete_vectors = left.len() / 8 * 8;
    let mut tail = [0.0_f32; 4];
    for block in (complete_blocks..complete_vectors).step_by(8) {
        for lane in 0..4 {
            tail[lane] += left[block + lane] * right[block + lane];
            tail[lane] += left[block + 4 + lane] * right[block + 4 + lane];
        }
    }
    reduced += (tail[0] + tail[1]) + (tail[2] + tail[3]);
    for index in complete_vectors..left.len() {
        reduced += left[index] * right[index];
    }
    reduced
}

fn causal_attention_head_with_dtype(
    query: &[f32],
    keys: &[&[f32]],
    values: &[&[f32]],
    scale: f32,
    sink: Option<f32>,
    bf16_boundaries: bool,
    trace: Option<&mut AttentionHeadTrace>,
) -> Result<Vec<f32>, String> {
    if query.is_empty()
        || keys.is_empty()
        || keys.len() != values.len()
        || keys.iter().any(|key| key.len() != query.len())
        || values.iter().any(|value| value.is_empty())
        || values.iter().any(|value| value.len() != values[0].len())
    {
        return Err("causal attention head shape mismatch".to_owned());
    }
    let mut scores = keys
        .iter()
        .map(|key| {
            let dot = if bf16_boundaries {
                pytorch_bf16_specialized_vector_dot_f32(query, key)
            } else {
                query
                    .iter()
                    .zip(*key)
                    .map(|(left, right)| left * right)
                    .sum::<f32>()
            };
            if bf16_boundaries {
                round_bf16(round_bf16(dot) * scale)
            } else {
                dot * scale
            }
        })
        .collect::<Vec<_>>();
    if let Some(sink) = sink {
        scores.push(sink);
    }
    let probabilities = attention_softmax(&scores, bf16_boundaries)?;
    if let Some(trace) = trace {
        let maximum = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        trace.scores = scores
            .iter()
            .map(|score| {
                if bf16_boundaries {
                    round_bf16(*score - maximum)
                } else {
                    *score - maximum
                }
            })
            .collect();
        trace.probabilities = probabilities.clone();
    }
    let mut output = vec![0.0_f32; values[0].len()];
    if bf16_boundaries {
        let mut value_column = vec![0.0_f32; values.len()];
        for dimension in 0..output.len() {
            for (destination, value) in value_column.iter_mut().zip(values) {
                *destination = value[dimension];
            }
            output[dimension] =
                pytorch_bf16_four_lane_dot_f32(&probabilities[..values.len()], &value_column);
        }
    } else {
        for (position, value) in values.iter().enumerate() {
            for (destination, source) in output.iter_mut().zip(*value) {
                *destination += probabilities[position] * source;
            }
        }
    }
    if bf16_boundaries {
        round_bf16_values(&mut output);
    }
    Ok(output)
}

#[allow(clippy::too_many_arguments)]
fn attention(
    checkpoint: &Checkpoint,
    config: &ModelConfig,
    layer: usize,
    normalized: &[f32],
    rows: usize,
    cache: &mut LayerKvCache,
    ledger: &mut EndpointLedger,
    mut captures: Option<&mut Layer0Captures>,
) -> Result<Vec<f32>, String> {
    cache.validate()?;
    let is_swa = config.hybrid_layer_pattern[layer] == 1;
    let kv_heads = if is_swa { 8 } else { 4 };
    if cache.positions != 0 && cache.kv_heads != kv_heads {
        return Err(format!("layer {layer}: K/V head authority changed"));
    }
    let q_size = HEADS * QK_HEAD_DIM;
    let k_size = kv_heads * QK_HEAD_DIM;
    let v_size = kv_heads * V_HEAD_DIM;
    let qkv_rows = q_size + k_size + v_size;
    let prefix = format!("model.layers.{layer}.self_attn");
    let qkv_name = format!("{prefix}.qkv_proj.weight");
    let qkv = if is_swa {
        fp8_linear(
            checkpoint, &qkv_name, normalized, rows, HIDDEN, qkv_rows, ledger,
        )
    } else {
        full_qkv_linear(checkpoint, &qkv_name, normalized, rows, ledger)
    }?;
    if let Some(captures) = captures.as_deref_mut() {
        captures.qkv = qkv.clone();
    }
    let prior = cache.positions;
    let theta = if is_swa {
        config.swa_rope_theta
    } else {
        config.rope_theta
    };
    let mut queries = vec![0.0_f32; rows * q_size];
    for row in 0..rows {
        let source = &qkv[row * qkv_rows..(row + 1) * qkv_rows];
        let query = &mut queries[row * q_size..(row + 1) * q_size];
        query.copy_from_slice(&source[..q_size]);
        apply_rope(query, HEADS, prior + row, theta);
        let mut key = source[q_size..q_size + k_size].to_vec();
        apply_rope(&mut key, kv_heads, prior + row, theta);
        cache.keys.extend(key);
        cache.values.extend(
            source[q_size + k_size..]
                .iter()
                .map(|value| round_bf16(value * config.attention_value_scale)),
        );
    }
    cache.positions += rows;
    cache.kv_heads = kv_heads;
    cache.validate()?;
    if let Some(captures) = captures.as_deref_mut() {
        captures.query = queries.clone();
        captures.key = cache.keys.clone();
        captures.value = cache.values.clone();
    }
    let sinks = if is_swa {
        Some(bf16_vector(
            checkpoint,
            &format!("{prefix}.attention_sink_bias"),
            HEADS,
            ledger,
        )?)
    } else {
        None
    };
    if let (Some(captures), Some(sinks)) = (captures.as_deref_mut(), sinks.as_ref()) {
        captures.sinks = sinks.clone();
    }
    let mut result = vec![0.0_f32; rows * HEADS * V_HEAD_DIM];
    let scale = 1.0_f32 / (QK_HEAD_DIM as f32).sqrt();
    let kv_groups = HEADS / kv_heads;
    for row in 0..rows {
        let end = prior + row + 1;
        let start = if is_swa {
            end.saturating_sub(config.sliding_window)
        } else {
            0
        };
        for head in 0..HEADS {
            let kv_head = head / kv_groups;
            let query = &queries
                [row * q_size + head * QK_HEAD_DIM..row * q_size + (head + 1) * QK_HEAD_DIM];
            let mut keys = Vec::with_capacity(end - start);
            let mut values = Vec::with_capacity(end - start);
            for position in start..end {
                let key_offset = (position * kv_heads + kv_head) * QK_HEAD_DIM;
                keys.push(&cache.keys[key_offset..key_offset + QK_HEAD_DIM]);
                let value_offset = (position * kv_heads + kv_head) * V_HEAD_DIM;
                values.push(&cache.values[value_offset..value_offset + V_HEAD_DIM]);
            }
            let mut head_trace = AttentionHeadTrace::default();
            let head_output = if captures.is_some() {
                causal_attention_head_with_dtype(
                    query,
                    &keys,
                    &values,
                    scale,
                    sinks.as_ref().map(|values| values[head]),
                    true,
                    Some(&mut head_trace),
                )?
            } else {
                causal_attention_head_bf16(
                    query,
                    &keys,
                    &values,
                    scale,
                    sinks.as_ref().map(|values| values[head]),
                )?
            };
            if let Some(captures) = captures.as_deref_mut() {
                captures.attention_scores.extend(head_trace.scores);
                captures
                    .attention_probabilities
                    .extend(head_trace.probabilities);
            }
            let destination = &mut result[row * HEADS * V_HEAD_DIM + head * V_HEAD_DIM
                ..row * HEADS * V_HEAD_DIM + (head + 1) * V_HEAD_DIM];
            destination.copy_from_slice(&head_output);
        }
    }
    if let Some(captures) = captures.as_deref_mut() {
        captures.attention = result.clone();
    }
    let projected = bf16_linear(
        checkpoint,
        &format!("{prefix}.o_proj.weight"),
        &result,
        rows,
        HEADS * V_HEAD_DIM,
        HIDDEN,
        ledger,
    )?;
    if let Some(captures) = captures {
        captures.attention_projection = projected.clone();
    }
    Ok(projected)
}

fn dense_mlp(
    checkpoint: &Checkpoint,
    input: &[f32],
    rows: usize,
    ledger: &mut EndpointLedger,
    captures: Option<&mut Layer0Captures>,
) -> Result<Vec<f32>, String> {
    let prefix = "model.layers.0.mlp";
    let gate = fp8_linear(
        checkpoint,
        &format!("{prefix}.gate_proj.weight"),
        input,
        rows,
        HIDDEN,
        16_384,
        ledger,
    )?;
    let up = fp8_linear(
        checkpoint,
        &format!("{prefix}.up_proj.weight"),
        input,
        rows,
        HIDDEN,
        16_384,
        ledger,
    )?;
    let activated = gate
        .iter()
        .zip(&up)
        .map(|(&gate, &up)| {
            let silu = round_bf16(gate / (1.0 + (-gate).exp()));
            round_bf16(silu * up)
        })
        .collect::<Vec<_>>();
    let down = fp8_linear(
        checkpoint,
        &format!("{prefix}.down_proj.weight"),
        &activated,
        rows,
        16_384,
        HIDDEN,
        ledger,
    )?;
    if let Some(captures) = captures {
        captures.gate = gate;
        captures.up = up;
        captures.swiglu = activated;
        captures.down = down.clone();
    }
    Ok(down)
}

fn route_mlp(
    checkpoint: &Checkpoint,
    layer: usize,
    input: &[f32],
    rows: usize,
    ledger: &mut EndpointLedger,
) -> Result<RoutingTrace, String> {
    let prefix = format!("model.layers.{layer}.mlp");
    let logits = f32_linear(
        checkpoint,
        &format!("{prefix}.gate.weight"),
        input,
        rows,
        HIDDEN,
        ROUTED_EXPERTS,
        ledger,
    )?;
    let correction = f32_vector(
        checkpoint,
        &format!("{prefix}.gate.e_score_correction_bias"),
        &[ROUTED_EXPERTS as u64],
        ledger,
    )?;
    let scores = logits
        .iter()
        .map(|&logit| pytorch_sigmoid_f32(logit))
        .collect::<Vec<_>>();
    let (selected, weights) = pytorch_noaux_routes(&scores, &correction, rows)?;
    Ok(RoutingTrace {
        logits,
        scores,
        selected,
        weights,
    })
}

fn pytorch_sum_eight(values: &[f32; TOP_K]) -> f32 {
    let lanes = [
        values[0] + values[4],
        values[1] + values[5],
        values[2] + values[6],
        values[3] + values[7],
    ];
    lanes.iter().fold(0.0_f32, |sum, value| sum + value)
}

type PytorchRouteRows = (Vec<Vec<u32>>, Vec<Vec<f32>>);

fn pytorch_noaux_routes(
    scores: &[f32],
    correction: &[f32],
    rows: usize,
) -> Result<PytorchRouteRows, String> {
    if scores.len() != rows * ROUTED_EXPERTS
        || correction.len() != ROUTED_EXPERTS
        || scores.iter().any(|value| !value.is_finite())
        || correction.iter().any(|value| !value.is_finite())
    {
        return Err("invalid PyTorch noaux-tc inputs".to_owned());
    }
    let mut selected_rows = Vec::with_capacity(rows);
    let mut weight_rows = Vec::with_capacity(rows);
    for position in 0..rows {
        let row = &scores[position * ROUTED_EXPERTS..(position + 1) * ROUTED_EXPERTS];
        let corrected = row
            .iter()
            .zip(correction)
            .map(|(score, bias)| score + bias)
            .collect::<Vec<_>>();
        let mut selected = [0_u32; TOP_K];
        // SAFETY: both buffers have the lengths supplied to the C++ bridge.
        let result = unsafe {
            pw_pytorch_topk_unsorted_f32(
                corrected.as_ptr(),
                corrected.len(),
                TOP_K,
                selected.as_mut_ptr(),
            )
        };
        if result != 0
            || selected
                .iter()
                .any(|expert| *expert as usize >= ROUTED_EXPERTS)
            || selected.iter().copied().collect::<BTreeSet<_>>().len() != TOP_K
        {
            return Err(format!("PyTorch top-k failed at position {position}"));
        }
        let selected_set = selected.iter().copied().collect::<BTreeSet<_>>();
        let boundary = selected
            .iter()
            .map(|expert| corrected[*expert as usize])
            .fold(f32::INFINITY, f32::min);
        let rejected = corrected
            .iter()
            .enumerate()
            .filter(|(expert, _)| !selected_set.contains(&(*expert as u32)))
            .map(|(_, value)| *value)
            .fold(f32::NEG_INFINITY, f32::max);
        if !boundary.is_finite() || boundary <= rejected {
            return Err(format!(
                "PyTorch top-k boundary tied at position {position}"
            ));
        }
        let chosen = selected.map(|expert| row[expert as usize]);
        let denominator = pytorch_sum_eight(&chosen) + 1.0e-20;
        let weights = selected
            .iter()
            .map(|expert| row[*expert as usize] / denominator)
            .collect::<Vec<_>>();
        let weight_sum = weights.iter().copied().sum::<f32>();
        if weights
            .iter()
            .any(|value| !value.is_finite() || *value < 0.0)
            || (weight_sum - 1.0).abs() > 1.0e-6
        {
            return Err(format!(
                "PyTorch route weights invalid at position {position}"
            ));
        }
        selected_rows.push(selected.to_vec());
        weight_rows.push(weights);
    }
    Ok((selected_rows, weight_rows))
}

pub(crate) fn component_pytorch_noaux_route(
    logits: &[f32],
    correction: &[f32],
) -> Result<(Vec<u32>, Vec<f32>, f32), String> {
    if logits.len() != ROUTED_EXPERTS {
        return Err("component router logit shape mismatch".to_owned());
    }
    let scores = logits
        .iter()
        .map(|value| pytorch_sigmoid_f32(*value))
        .collect::<Vec<_>>();
    let (selected_rows, weight_rows) = pytorch_noaux_routes(&scores, correction, 1)?;
    let selected = selected_rows.into_iter().next().ok_or("missing route")?;
    let weights = weight_rows
        .into_iter()
        .next()
        .ok_or("missing route weights")?;
    let selected_set = selected.iter().copied().collect::<BTreeSet<_>>();
    let corrected = scores
        .iter()
        .zip(correction)
        .map(|(score, bias)| score + bias)
        .collect::<Vec<_>>();
    let boundary = selected
        .iter()
        .map(|expert| corrected[*expert as usize])
        .fold(f32::INFINITY, f32::min);
    let rejected = corrected
        .iter()
        .enumerate()
        .filter(|(expert, _)| !selected_set.contains(&(*expert as u32)))
        .map(|(_, value)| *value)
        .fold(f32::NEG_INFINITY, f32::max);
    Ok((selected, weights, boundary - rejected))
}

fn sleef_expf_u10(d: f32) -> f32 {
    if d < -104.0 {
        return 0.0;
    }
    if d > 100.0 {
        return f32::INFINITY;
    }
    let q = (d * std::f32::consts::LOG2_E).round_ties_even() as i32;
    let qf = q as f32;
    let mut s = qf.mul_add(-0.693_145_75_f32, d);
    s = qf.mul_add(-1.428_606_8e-6_f32, s);
    let mut u = 0.000_198_527_62_f32;
    u = u.mul_add(s, 0.001_393_043_6_f32);
    u = u.mul_add(s, 0.008_333_361_f32);
    u = u.mul_add(s, 0.041_666_485_f32);
    u = u.mul_add(s, 0.166_666_67_f32);
    u = u.mul_add(s, 0.5);
    u = (s * s).mul_add(u, s) + 1.0;
    let first_q = q >> 1;
    let second_q = q - first_q;
    let first_exponent =
        u32::try_from(first_q + 127).expect("first SLEEF exponent factor is normal") << 23;
    let second_exponent =
        u32::try_from(second_q + 127).expect("second SLEEF exponent factor is normal") << 23;
    (u * f32::from_bits(first_exponent)) * f32::from_bits(second_exponent)
}

fn pytorch_sigmoid_f32(value: f32) -> f32 {
    1.0 / (1.0 + sleef_expf_u10(-value))
}

fn routed_mlp(
    checkpoint: &Checkpoint,
    layer: usize,
    input: &[f32],
    rows: usize,
    ledger: &mut EndpointLedger,
) -> Result<RoutedMlpOutput, String> {
    routed_mlp_traced(
        checkpoint,
        layer,
        input,
        rows,
        ledger,
        None,
        &mut |_| Ok(()),
    )
}

fn require_one_row_metal_experts(rows: usize) -> Result<(), String> {
    if rows != 1 {
        return Err(format!(
            "bounded Metal routed experts require exactly one row, got {rows}"
        ));
    }
    Ok(())
}

fn routed_mlp_metal(
    checkpoint: &Checkpoint,
    layer: usize,
    input: &[f32],
    rows: usize,
    ledger: &mut EndpointLedger,
    metal_ledger: &mut MetalExpertLedger,
    runtime: &BoundedMetalExpertRuntime,
) -> Result<RoutedMlpOutput, String> {
    let routed_wall_started = Instant::now();
    let routed_activity_started = metal_ledger
        .tomography_enabled
        .then(process_activity)
        .transpose()?;
    require_one_row_metal_experts(rows)?;
    if input.len() != HIDDEN {
        return Err("bounded Metal routed expert input shape mismatch".to_owned());
    }
    let prefix = format!("model.layers.{layer}.mlp");
    let route_started = Instant::now();
    let routing = route_mlp(checkpoint, layer, input, rows, ledger)?;
    let mut schedule: BTreeMap<u32, Vec<(usize, f32)>> = BTreeMap::new();
    for slot in 0..TOP_K {
        schedule
            .entry(routing.selected[0][slot])
            .or_default()
            .push((0, routing.weights[0][slot]));
    }
    if schedule.len() != TOP_K || schedule.values().any(|placements| placements.len() != 1) {
        return Err("bounded Metal route did not contain eight unique experts".to_owned());
    }
    let route_and_schedule_ms = route_started.elapsed().as_secs_f64() * 1000.0;
    let mut output = vec![0.0_f32; HIDDEN];
    let down_shape_authority = vec![0.0_f32; MOE_INTERMEDIATE];
    let expert_tomography_start = metal_ledger.expert_tomography.len();
    for (expert, placements) in schedule {
        let expert_wall_started = Instant::now();
        let expert_activity_started = metal_ledger
            .tomography_enabled
            .then(process_activity)
            .transpose()?;
        let expert_prefix = format!("{prefix}.experts.{expert}");
        let validation_started = Instant::now();
        let tensor = |projection: &str| -> Result<_, String> {
            let name = format!("{expert_prefix}.{projection}_proj.weight");
            validate_fp8_views(
                checkpoint.tensor(&name)?,
                checkpoint.tensor(&format!("{name}_scale_inv"))?,
                if projection == "down" {
                    // Validation needs only authoritative input width and finiteness;
                    // the runtime derives the actual staged hidden vector internally.
                    &down_shape_authority
                } else {
                    input
                },
            )
        };
        let gate = tensor("gate")?;
        let up = tensor("up")?;
        let down = tensor("down")?;
        let tensor_lookup_validation_ms = validation_started.elapsed().as_secs_f64() * 1000.0;
        if (gate.rows, gate.columns) != (MOE_INTERMEDIATE, HIDDEN)
            || (up.rows, up.columns) != (MOE_INTERMEDIATE, HIDDEN)
            || (down.rows, down.columns) != (HIDDEN, MOE_INTERMEDIATE)
        {
            return Err(format!(
                "layer {layer} expert {expert}: projection shape mismatch"
            ));
        }
        let mut execution = if metal_ledger.tomography_enabled {
            runtime.execute_profiled(layer, expert, [&gate, &up, &down], input)?
        } else {
            runtime.execute([&gate, &up, &down], input)?
        };
        let weight = placements[0].1;
        let scatter_started = Instant::now();
        for (destination, value) in output.iter_mut().zip(&execution.down) {
            *destination += *value * weight;
        }
        let weighted_scatter_ms = scatter_started.elapsed().as_secs_f64() * 1000.0;
        ledger.logical_source_bytes = ledger
            .logical_source_bytes
            .checked_add(execution.installed_source_bytes)
            .ok_or("logical byte ledger overflow")?;
        ledger.dynamic_activation_groups += 80;
        ledger.dynamic_activation_values += 10_240;
        ledger.routed_expert_executions += 1;
        metal_ledger.expert_executions += 1;
        metal_ledger.projection_dispatches += 3;
        metal_ledger.installed_source_bytes = metal_ledger
            .installed_source_bytes
            .checked_add(execution.installed_source_bytes)
            .ok_or("Metal installed-byte ledger overflow")?;
        metal_ledger.released_projection_buffers += 3;
        metal_ledger.sparse_decoded_weight_bytes = metal_ledger
            .sparse_decoded_weight_bytes
            .checked_add(execution.sparse_decoded_weight_bytes)
            .ok_or("Metal sparse-byte ledger overflow")?;
        for (total, count) in metal_ledger
            .sparse_repair_counts
            .iter_mut()
            .zip(execution.sparse_repair_counts)
        {
            *total += count as u64;
        }
        let release_started = Instant::now();
        release_matrix_transients(checkpoint)?;
        let matrix_transient_release_ms = release_started.elapsed().as_secs_f64() * 1000.0;
        if let Some(mut tomography) = execution.tomography.take() {
            tomography.source_shards = ["gate", "up", "down"]
                .iter()
                .map(|projection| {
                    checkpoint
                        .shard_for_tensor(&format!("{expert_prefix}.{projection}_proj.weight"))
                        .map(str::to_owned)
                })
                .collect::<Result<Vec<_>, _>>()?;
            tomography.tensor_lookup_validation_ms = tensor_lookup_validation_ms;
            tomography.weighted_scatter_ms = weighted_scatter_ms;
            tomography.matrix_transient_release_ms = matrix_transient_release_ms;
            tomography.wall_ms = expert_wall_started.elapsed().as_secs_f64() * 1000.0;
            tomography.activity = process_activity()?.checked_delta(
                expert_activity_started.ok_or("missing expert activity baseline")?,
            )?;
            metal_ledger.expert_tomography.push(tomography);
        }
    }
    let final_round_started = Instant::now();
    round_bf16_values(&mut output);
    let layer_final_bf16_round_ms = final_round_started.elapsed().as_secs_f64() * 1000.0;
    if let Some(before) = routed_activity_started {
        let expert_wall_sum_ms = metal_ledger.expert_tomography[expert_tomography_start..]
            .iter()
            .map(|record| record.wall_ms)
            .sum();
        metal_ledger.layer_tomography.push(MetalLayerTomography {
            layer,
            route_and_schedule_ms,
            expert_count: metal_ledger.expert_tomography.len() - expert_tomography_start,
            expert_wall_sum_ms,
            layer_final_bf16_round_ms,
            routed_mlp_wall_ms: routed_wall_started.elapsed().as_secs_f64() * 1000.0,
            activity: process_activity()?.checked_delta(before)?,
        });
    }
    Ok(RoutedMlpOutput {
        output,
        logits: routing.logits,
        scores: routing.scores,
        selected: routing.selected,
        weights: routing.weights,
    })
}

#[allow(clippy::too_many_arguments)]
fn routed_mlp_traced(
    checkpoint: &Checkpoint,
    layer: usize,
    input: &[f32],
    rows: usize,
    ledger: &mut EndpointLedger,
    mut captures: Option<&mut ExpertCaptures>,
    expert_completed: &mut dyn FnMut(u32) -> Result<(), String>,
) -> Result<RoutedMlpOutput, String> {
    let prefix = format!("model.layers.{layer}.mlp");
    let routing = route_mlp(checkpoint, layer, input, rows, ledger)?;
    let mut schedule: BTreeMap<u32, Vec<(usize, f32)>> = BTreeMap::new();
    for position in 0..rows {
        for slot in 0..TOP_K {
            schedule
                .entry(routing.selected[position][slot])
                .or_default()
                .push((position, routing.weights[position][slot]));
        }
    }
    let mut output = vec![0.0_f32; rows * HIDDEN];
    for (expert, placements) in schedule {
        let mut gathered = vec![0.0_f32; placements.len() * HIDDEN];
        for (local, &(position, _)) in placements.iter().enumerate() {
            gathered[local * HIDDEN..(local + 1) * HIDDEN]
                .copy_from_slice(&input[position * HIDDEN..(position + 1) * HIDDEN]);
        }
        let expert_prefix = format!("{prefix}.experts.{expert}");
        let gate = fp8_linear(
            checkpoint,
            &format!("{expert_prefix}.gate_proj.weight"),
            &gathered,
            placements.len(),
            HIDDEN,
            MOE_INTERMEDIATE,
            ledger,
        )?;
        let up = fp8_linear(
            checkpoint,
            &format!("{expert_prefix}.up_proj.weight"),
            &gathered,
            placements.len(),
            HIDDEN,
            MOE_INTERMEDIATE,
            ledger,
        )?;
        let activated = gate
            .iter()
            .zip(&up)
            .map(|(&gate, &up)| {
                let silu = round_bf16(gate / (1.0 + (-gate).exp()));
                round_bf16(silu * up)
            })
            .collect::<Vec<_>>();
        let projected = fp8_linear(
            checkpoint,
            &format!("{expert_prefix}.down_proj.weight"),
            &activated,
            placements.len(),
            MOE_INTERMEDIATE,
            HIDDEN,
            ledger,
        )?;
        if let Some(captures) = captures.as_deref_mut() {
            captures.schedule.push(ExpertScheduleEntry {
                expert,
                positions: placements.iter().map(|(position, _)| *position).collect(),
            });
            captures.gate.extend_from_slice(&gate);
            captures.up.extend_from_slice(&up);
            captures.swiglu.extend_from_slice(&activated);
            captures.down.extend_from_slice(&projected);
        }
        for (local, &(position, weight)) in placements.iter().enumerate() {
            for column in 0..HIDDEN {
                output[position * HIDDEN + column] += projected[local * HIDDEN + column] * weight;
            }
        }
        ledger.routed_expert_executions += 1;
        expert_completed(expert)?;
    }
    round_bf16_values(&mut output);
    Ok(RoutedMlpOutput {
        output,
        logits: routing.logits,
        scores: routing.scores,
        selected: routing.selected,
        weights: routing.weights,
    })
}

fn expert_union_factor(unique_experts: usize, positions: usize) -> Result<f64, String> {
    if positions == 0 {
        return Err("expert union factor requires at least one position".to_owned());
    }
    Ok(unique_experts as f64 / positions as f64)
}

fn embedding(
    checkpoint: &Checkpoint,
    token_ids: &[u32],
    ledger: &mut EndpointLedger,
) -> Result<Vec<f32>, String> {
    let view = checkpoint.tensor("model.embed_tokens.weight")?;
    if view.metadata.dtype != "BF16"
        || view.metadata.shape != [152_576, HIDDEN as u64]
        || token_ids.iter().any(|&token| token as usize >= 152_576)
    {
        return Err("embedding tensor or token identity mismatch".to_owned());
    }
    let mut output = Vec::with_capacity(token_ids.len() * HIDDEN);
    for &token in token_ids {
        let start = token as usize * HIDDEN * 2;
        ledger.logical_source_bytes = ledger
            .logical_source_bytes
            .checked_add((HIDDEN * 2) as u64)
            .ok_or("logical byte ledger overflow")?;
        output.extend(
            view.bytes[start..start + HIDDEN * 2]
                .chunks_exact(2)
                .map(|bytes| {
                    f32::from_bits(
                        u32::from(u16::from_le_bytes(bytes.try_into().expect("two-byte BF16")))
                            << 16,
                    )
                }),
        );
    }
    if output.iter().any(|value| !value.is_finite()) {
        return Err("embedding contains non-finite values".to_owned());
    }
    Ok(output)
}

fn top_logits(logits: &[f32], count: usize) -> Result<Vec<(u32, f32)>, String> {
    if logits.len() != 152_576 || logits.iter().any(|value| !value.is_finite()) {
        return Err("LM-head logits are invalid".to_owned());
    }
    let mut indices = (0..logits.len()).collect::<Vec<_>>();
    indices.sort_by(|left, right| {
        logits[*right]
            .total_cmp(&logits[*left])
            .then(left.cmp(right))
    });
    Ok(indices
        .into_iter()
        .take(count)
        .map(|index| (index as u32, logits[index]))
        .collect())
}

#[allow(clippy::too_many_arguments)]
fn decode_step(
    checkpoint: &Checkpoint,
    config: &ModelConfig,
    token_ids: &[u32],
    caches: &mut [LayerKvCache],
    ledger: &mut EndpointLedger,
    safety: &mut SafetyMonitor,
    mut full_captures: Option<&mut FullPrefixCaptures>,
    mut metal: Option<(&BoundedMetalExpertRuntime, &mut MetalExpertLedger)>,
) -> Result<NativeDecodeStep, String> {
    let started = Instant::now();
    if token_ids.is_empty() {
        return Err("decode step requires at least one token".to_owned());
    }
    let rows = token_ids.len();
    let mut hidden = embedding(checkpoint, token_ids, ledger)?;
    if let Some(captures) = full_captures.as_deref_mut() {
        captures.embedding = hidden.clone();
    }
    let mut traces = Vec::with_capacity(48);
    if caches.len() != 48 {
        return Err("text endpoint requires exactly 48 K/V caches".to_owned());
    }
    for (layer, cache) in caches.iter_mut().enumerate() {
        let layer_started = Instant::now();
        let input_norm = bf16_vector(
            checkpoint,
            &format!("model.layers.{layer}.input_layernorm.weight"),
            HIDDEN,
            ledger,
        )?;
        let normalized = rms_norm(&hidden, rows, &input_norm, config.layernorm_epsilon)?;
        let attention_output = attention(
            checkpoint,
            config,
            layer,
            &normalized,
            rows,
            cache,
            ledger,
            None,
        )?;
        let post_attention = hidden
            .iter()
            .zip(attention_output)
            .map(|(&residual, projected)| round_bf16(residual + projected))
            .collect::<Vec<_>>();
        let post_norm = bf16_vector(
            checkpoint,
            &format!("model.layers.{layer}.post_attention_layernorm.weight"),
            HIDDEN,
            ledger,
        )?;
        let moe_input = rms_norm(&post_attention, rows, &post_norm, config.layernorm_epsilon)?;
        let (mlp, selected, weights) = if layer == 0 {
            (
                dense_mlp(checkpoint, &moe_input, rows, ledger, None)?,
                Vec::new(),
                Vec::new(),
            )
        } else {
            let routed = if let Some((runtime, metal_ledger)) = metal.as_mut() {
                routed_mlp_metal(
                    checkpoint,
                    layer,
                    &moe_input,
                    rows,
                    ledger,
                    metal_ledger,
                    runtime,
                )?
            } else {
                routed_mlp(checkpoint, layer, &moe_input, rows, ledger)?
            };
            (routed.output, routed.selected, routed.weights)
        };
        hidden = post_attention
            .iter()
            .zip(mlp)
            .map(|(&residual, projected)| round_bf16(residual + projected))
            .collect();
        if let Some(captures) = full_captures.as_deref_mut() {
            captures.layer_finals.push(hidden.clone());
        }
        let unique = selected
            .iter()
            .flatten()
            .copied()
            .collect::<BTreeSet<_>>()
            .len();
        traces.push(LayerRouteTrace {
            layer,
            attention: if config.hybrid_layer_pattern[layer] == 1 {
                "sliding_window_128"
            } else {
                "full"
            },
            cache_length: cache.positions,
            selected_experts_by_position: selected,
            route_weights_by_position: weights,
            expert_union_factor: if layer == 0 {
                0.0
            } else {
                expert_union_factor(unique, rows)?
            },
            wall_ms: layer_started.elapsed().as_secs_f64() * 1000.0,
        });
        checkpoint.release_file_pages()?;
        safety.checkpoint(&format!("layer_{layer}_complete"), true)?;
    }
    let final_norm = bf16_vector(checkpoint, "model.norm.weight", HIDDEN, ledger)?;
    let normalized = rms_norm(&hidden, rows, &final_norm, config.layernorm_epsilon)?;
    if let Some(captures) = full_captures {
        captures.final_norm = normalized.clone();
    }
    let last_logits = bf16_last_row_linear(
        checkpoint,
        "lm_head.weight",
        &normalized[(rows - 1) * HIDDEN..rows * HIDDEN],
        HIDDEN,
        config.vocab_size,
        ledger,
    )?;
    let top = top_logits(&last_logits, 20)?;
    checkpoint.release_file_pages()?;
    safety.checkpoint("lm_head_complete", true)?;
    let token = top[0].0;
    Ok(NativeDecodeStep {
        output_token: token,
        top_logits: top,
        full_logits: last_logits,
        traces,
        wall_ms: started.elapsed().as_secs_f64() * 1000.0,
    })
}

fn open_endpoint_authority(
    checkpoint_root: &Path,
    model_lock_path: &Path,
    verification_path: &Path,
    fixture_path: &Path,
) -> Result<EndpointAuthority, String> {
    if hash_file(model_lock_path)? != MODEL_LOCK_SHA256 {
        return Err("model lock SHA-256 mismatch".to_owned());
    }
    let fixture_bytes =
        fs::read(fixture_path).map_err(|error| format!("{}: {error}", fixture_path.display()))?;
    let fixture: EndpointFixture = serde_json::from_slice(&fixture_bytes)
        .map_err(|error| format!("endpoint fixture: {error}"))?;
    validate_fixture(&fixture)?;
    let mut safety = SafetyMonitor::start(fixture.safety.clone())?;
    for (name, expected) in [
        ("config.json", fixture.config_sha256.as_str()),
        (
            "model.safetensors.index.json",
            fixture.index_sha256.as_str(),
        ),
        ("tokenizer.json", fixture.tokenizer_sha256.as_str()),
        (
            "tokenizer_config.json",
            fixture.tokenizer_config_sha256.as_str(),
        ),
    ] {
        if hash_file(&checkpoint_root.join(name))? != expected {
            return Err(format!("{name} SHA-256 mismatch"));
        }
    }
    let verification_sha256 = hash_file(verification_path)?;
    if verification_sha256 != fixture.checkpoint_verification_sha256 {
        return Err("checkpoint verification SHA-256 mismatch".to_owned());
    }
    let verification: CheckpointVerification = serde_json::from_reader(
        File::open(verification_path)
            .map_err(|error| format!("{}: {error}", verification_path.display()))?,
    )
    .map_err(|error| format!("checkpoint verification: {error}"))?;
    if verification.schema_version != 1
        || verification.evidence_class != "local_checkpoint_lock_verification"
        || !verification.complete
        || verification.lock_sha256 != MODEL_LOCK_SHA256
        || verification.revision != REVISION
    {
        return Err("checkpoint verification identity mismatch".to_owned());
    }
    let config: ModelConfig = serde_json::from_reader(
        File::open(checkpoint_root.join("config.json")).map_err(|error| error.to_string())?,
    )
    .map_err(|error| format!("model config: {error}"))?;
    validate_config(&config)?;
    let tokenizer = Tokenizer::from_file(checkpoint_root.join("tokenizer.json"))
        .map_err(|error| format!("tokenizer load: {error}"))?;
    let encoded = tokenizer
        .encode(fixture.prompt_utf8.clone(), fixture.add_special_tokens)
        .map_err(|error| format!("tokenizer encode: {error}"))?;
    let prompt_token_ids = encoded.get_ids().to_vec();
    if prompt_token_ids != fixture.expected_prompt_token_ids
        || tokenizer
            .decode(&prompt_token_ids, false)
            .map_err(|error| format!("tokenizer decode: {error}"))?
            != fixture.prompt_utf8
    {
        return Err("tokenizer prompt identity mismatch".to_owned());
    }
    let checkpoint = Checkpoint::open(
        checkpoint_root,
        &checkpoint_root.join("model.safetensors.index.json"),
        &verification,
    )?;
    safety.checkpoint("checkpoint_open", true)?;
    Ok(EndpointAuthority {
        fixture_bytes,
        fixture,
        safety,
        config,
        tokenizer,
        prompt_token_ids,
        checkpoint,
        verification_sha256,
    })
}

#[allow(clippy::too_many_arguments)]
pub fn run_slow_text_endpoint(
    checkpoint_root: &Path,
    model_lock_path: &Path,
    verification_path: &Path,
    fixture_path: &Path,
    output_path: &Path,
    commit: &str,
) -> Result<TextEndpointReport, String> {
    if output_path.exists() {
        return Err(format!("refusing to overwrite {}", output_path.display()));
    }
    let complete_started = Instant::now();
    let disk_bytes_read_before = process_disk_bytes_read()?;
    let EndpointAuthority {
        fixture_bytes,
        fixture,
        mut safety,
        config,
        tokenizer,
        prompt_token_ids,
        checkpoint,
        verification_sha256,
    } = open_endpoint_authority(
        checkpoint_root,
        model_lock_path,
        verification_path,
        fixture_path,
    )?;
    validate_slow_endpoint_fixture(&fixture)?;
    let mut caches = (0..48).map(|_| LayerKvCache::default()).collect::<Vec<_>>();
    let mut ledger = EndpointLedger::default();
    let mut input_token_ids = prompt_token_ids.clone();
    let mut generated = Vec::with_capacity(fixture.decode.new_tokens);
    let mut steps = Vec::with_capacity(fixture.decode.new_tokens);
    for _ in 0..fixture.decode.new_tokens {
        let step = decode_step(
            &checkpoint,
            &config,
            &input_token_ids,
            &mut caches,
            &mut ledger,
            &mut safety,
            None,
            None,
        )?;
        let output_token_text = tokenizer
            .decode(&[step.output_token], false)
            .map_err(|error| format!("tokenizer output decode: {error}"))?;
        steps.push(DecodeStepReport {
            input_token_id: *input_token_ids
                .last()
                .ok_or("endpoint input token sequence is empty")?,
            input_token_ids: input_token_ids.clone(),
            output_token_id: step.output_token,
            output_token_text,
            top_logits: step.top_logits,
            full_logits: (fixture.schema_version == 2).then_some(step.full_logits),
            layer_traces: step.traces,
            wall_ms: step.wall_ms,
        });
        generated.push(step.output_token);
        input_token_ids = vec![step.output_token];
        safety.checkpoint(&format!("token_{}_accepted", generated.len()), true)?;
    }
    let expected_cache_positions = prompt_token_ids.len() + fixture.decode.new_tokens - 1;
    if caches
        .iter()
        .any(|cache| cache.positions != expected_cache_positions || cache.validate().is_err())
    {
        return Err(format!(
            "incremental K/V cache did not retain {expected_cache_positions} positions"
        ));
    }
    let generated_text = tokenizer
        .decode(&generated, false)
        .map_err(|error| format!("tokenizer generated decode: {error}"))?;
    ledger.actual_process_disk_bytes_read = process_disk_bytes_read()?
        .checked_sub(disk_bytes_read_before)
        .ok_or("process disk byte counter moved backwards")?;
    ledger.peak_resident_bytes = peak_resident_bytes()?;
    let report = TextEndpointReport {
        schema_version: fixture.schema_version,
        semantic: if fixture.schema_version == 1 {
            "mimo_v2_5_target_faithful_slow_text_endpoint"
        } else {
            "mimo_v2_5_target_faithful_slow_chat_endpoint"
        },
        revision: REVISION,
        commit: commit.to_owned(),
        fixture_sha256: sha256_hex(&fixture_bytes),
        model_lock_sha256: MODEL_LOCK_SHA256,
        checkpoint_verification_sha256: verification_sha256,
        config_sha256: fixture.config_sha256,
        index_sha256: fixture.index_sha256,
        tokenizer_sha256: fixture.tokenizer_sha256,
        tokenizer_config_sha256: fixture.tokenizer_config_sha256,
        prompt_utf8: fixture.prompt_utf8,
        prompt_token_ids,
        generated_token_ids: generated,
        generated_text,
        steps,
        ledger,
        safety_snapshots: safety.snapshots,
        complete_wall_ms: complete_started.elapsed().as_secs_f64() * 1000.0,
        batch_size: 1,
        concurrency: 1,
        accepted_tokens: 2,
        accepted_per_verification: 1,
        cache_state: "cold process; verified source mmap; one matrix expanded at a time; retained per-layer K/V",
        exactness: "L0 source weights and routes; dynamic per-token-group E4M3FN activations; source-authorized BF16 tensor boundaries with readable FP32 accumulation",
        performance_claim: None,
        implementation: "single_rust_authority_tokenizers_mmap_accelerate_dynamic_activation_source_fp8_bf16_boundaries",
    };
    let report_bytes = serde_json::to_vec_pretty(&report).map_err(|error| error.to_string())?;
    write_create_new(output_path, &report_bytes)?;
    Ok(report)
}

fn numerical_parity(actual: &[f32], expected: &[f32]) -> Result<NumericalParity, String> {
    if actual.len() != expected.len()
        || actual.is_empty()
        || actual
            .iter()
            .chain(expected)
            .any(|value| !value.is_finite())
    {
        return Err("numerical parity shape or finiteness mismatch".to_owned());
    }
    let mut squared_error = 0.0_f64;
    let mut squared_reference = 0.0_f64;
    let mut maximum_absolute_error = 0.0_f32;
    let mut equal_values = 0_usize;
    for (&candidate, &reference) in actual.iter().zip(expected) {
        let difference = candidate - reference;
        squared_error += f64::from(difference).powi(2);
        squared_reference += f64::from(reference).powi(2);
        maximum_absolute_error = maximum_absolute_error.max(difference.abs());
        equal_values += usize::from(candidate.to_bits() == reference.to_bits());
    }
    if squared_reference == 0.0 {
        return Err("numerical parity reference has zero L2 norm".to_owned());
    }
    let relative_l2 = (squared_error / squared_reference).sqrt();
    let equality_fraction = equal_values as f64 / actual.len() as f64;
    Ok(NumericalParity {
        relative_l2,
        maximum_absolute_error,
        equal_values,
        total_values: actual.len(),
        equality_fraction,
        passed: relative_l2 <= 5.0e-4
            && maximum_absolute_error <= 2.0e-2
            && equality_fraction >= 0.99,
    })
}

fn read_oracle_capture(
    manifest_path: &Path,
    capture: &OracleCapture,
    expected_values: usize,
) -> Result<Vec<f32>, String> {
    if capture.shape.iter().product::<usize>() != expected_values
        || !matches!(capture.dtype.as_str(), "BF16_widened_F32" | "F32")
        || Path::new(&capture.file)
            .file_name()
            .and_then(|name| name.to_str())
            != Some(capture.file.as_str())
    {
        return Err("incremental oracle capture metadata mismatch".to_owned());
    }
    let path = manifest_path.with_file_name(&capture.file);
    let (bytes, values) = read_f32_file(&path, Some(expected_values))?;
    if sha256_hex(&bytes) != capture.sha256 {
        return Err(format!(
            "incremental oracle capture hash mismatch: {}",
            capture.file
        ));
    }
    Ok(values)
}

fn projected_top20_jsd(reference: &[f32], candidate: &[f32]) -> Result<(bool, f64), String> {
    let reference_top = top_logits(reference, 20)?;
    let candidate_top = top_logits(candidate, 20)?;
    let top20_identity = reference_top
        .iter()
        .map(|(token, _)| token)
        .eq(candidate_top.iter().map(|(token, _)| token));
    let reference_max = reference.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let candidate_max = candidate.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let reference_sum = reference
        .iter()
        .map(|value| f64::from(*value - reference_max).exp())
        .sum::<f64>();
    let candidate_sum = candidate
        .iter()
        .map(|value| f64::from(*value - candidate_max).exp())
        .sum::<f64>();
    if !reference_sum.is_finite() || !candidate_sum.is_finite() {
        return Err("incremental logit normalization is non-finite".to_owned());
    }
    let mut reference_projection = Vec::with_capacity(21);
    let mut candidate_projection = Vec::with_capacity(21);
    for (token, _) in &reference_top {
        let index = *token as usize;
        reference_projection
            .push(f64::from(reference[index] - reference_max).exp() / reference_sum);
        candidate_projection
            .push(f64::from(candidate[index] - candidate_max).exp() / candidate_sum);
    }
    reference_projection.push((1.0 - reference_projection.iter().sum::<f64>()).max(0.0));
    candidate_projection.push((1.0 - candidate_projection.iter().sum::<f64>()).max(0.0));
    let jsd = reference_projection
        .iter()
        .zip(&candidate_projection)
        .map(|(&reference, &candidate)| {
            let midpoint = (reference + candidate) * 0.5;
            let contribution = |value: f64| {
                if value == 0.0 {
                    0.0
                } else {
                    value * (value / midpoint).ln()
                }
            };
            0.5 * (contribution(reference) + contribution(candidate))
        })
        .sum();
    Ok((top20_identity, jsd))
}

#[allow(clippy::too_many_arguments)]
pub fn run_metal_incremental_text_endpoint(
    checkpoint_root: &Path,
    model_lock_path: &Path,
    verification_path: &Path,
    fixture_path: &Path,
    oracle_manifest_path: &Path,
    kernel_path: &Path,
    output_path: &Path,
    commit: &str,
) -> Result<MetalIncrementalTextReport, String> {
    run_metal_incremental(
        checkpoint_root,
        model_lock_path,
        verification_path,
        fixture_path,
        oracle_manifest_path,
        kernel_path,
        output_path,
        commit,
        false,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn run_weight_install_tomography(
    checkpoint_root: &Path,
    model_lock_path: &Path,
    verification_path: &Path,
    fixture_path: &Path,
    oracle_manifest_path: &Path,
    kernel_path: &Path,
    output_path: &Path,
    commit: &str,
) -> Result<MetalIncrementalTextReport, String> {
    run_metal_incremental(
        checkpoint_root,
        model_lock_path,
        verification_path,
        fixture_path,
        oracle_manifest_path,
        kernel_path,
        output_path,
        commit,
        true,
    )
}

#[allow(clippy::too_many_arguments)]
fn run_metal_incremental(
    checkpoint_root: &Path,
    model_lock_path: &Path,
    verification_path: &Path,
    fixture_path: &Path,
    oracle_manifest_path: &Path,
    kernel_path: &Path,
    output_path: &Path,
    commit: &str,
    tomography_enabled: bool,
) -> Result<MetalIncrementalTextReport, String> {
    const ORACLE_SHA256: &str = "75b4a5799bcc7dc898643c266d42a00b52c75be0f1fe1682ef253ce8fe4287a8";
    const PREFILL_LOGITS_SHA256: &str =
        "c43be0909487235bddfe6e0de69aa42a98339faf43cd6b77d6ef4b5f1a853cab";
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
    let complete_started = Instant::now();
    let disk_bytes_read_before = process_disk_bytes_read()?;
    let EndpointAuthority {
        fixture_bytes,
        fixture,
        mut safety,
        config,
        tokenizer,
        prompt_token_ids,
        checkpoint,
        verification_sha256,
    } = open_endpoint_authority(
        checkpoint_root,
        model_lock_path,
        verification_path,
        fixture_path,
    )?;
    validate_slow_endpoint_fixture(&fixture)?;
    if fixture.decode.new_tokens != 2 {
        return Err("Metal incremental endpoint requires the frozen two-token fixture".to_owned());
    }
    let oracle_bytes = fs::read(oracle_manifest_path)
        .map_err(|error| format!("{}: {error}", oracle_manifest_path.display()))?;
    let oracle_manifest_sha256 = sha256_hex(&oracle_bytes);
    if oracle_manifest_sha256 != ORACLE_SHA256 {
        return Err("PW-0095 oracle manifest SHA-256 mismatch".to_owned());
    }
    let oracle: IncrementalOracleManifest = serde_json::from_slice(&oracle_bytes)
        .map_err(|error| format!("incremental oracle manifest: {error}"))?;
    if oracle.schema_version != 1
        || oracle.semantic != "mimo_pytorch_incremental_cache_oracle"
        || oracle.revision != REVISION
        || oracle.checkpoint_verification_sha256 != verification_sha256
        || oracle.prefill_token_ids != prompt_token_ids
        || oracle.incremental_input_token_id != 264
        || oracle.output_token_id != 13
        || oracle.incremental_layer_traces.len() != 48
    {
        return Err("PW-0095 oracle authority mismatch".to_owned());
    }
    let runtime = BoundedMetalExpertRuntime::compile(kernel_path)?;
    safety.checkpoint("metal_compile_complete", true)?;
    let mut caches = (0..48).map(|_| LayerKvCache::default()).collect::<Vec<_>>();
    let mut ledger = EndpointLedger::default();
    let prefill = decode_step(
        &checkpoint,
        &config,
        &prompt_token_ids,
        &mut caches,
        &mut ledger,
        &mut safety,
        None,
        None,
    )?;
    let prefill_bytes = prefill
        .full_logits
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect::<Vec<_>>();
    if prefill.output_token != 264
        || sha256_hex(&prefill_bytes) != PREFILL_LOGITS_SHA256
        || caches.iter().any(|cache| cache.positions != 27)
    {
        return Err("CPU prefill diverged from PW-0092 authority".to_owned());
    }
    safety.checkpoint("token_1_accepted", true)?;
    let mut captures = FullPrefixCaptures::default();
    let mut metal_ledger = MetalExpertLedger {
        tomography_enabled,
        ..MetalExpertLedger::default()
    };
    let incremental = decode_step(
        &checkpoint,
        &config,
        &[264],
        &mut caches,
        &mut ledger,
        &mut safety,
        Some(&mut captures),
        Some((&runtime, &mut metal_ledger)),
    )?;
    if caches
        .iter()
        .any(|cache| cache.positions != 28 || cache.validate().is_err())
        || captures.layer_finals.len() != 48
        || metal_ledger.expert_executions != 376
        || metal_ledger.projection_dispatches != 1_128
        || metal_ledger.released_projection_buffers != 1_128
    {
        return Err("Metal incremental causal/accounting gate failed".to_owned());
    }
    let mut layer_parity = Vec::with_capacity(48);
    let mut all_layer_parity_passed = true;
    for layer in 0..48 {
        let oracle_trace = &oracle.incremental_layer_traces[layer];
        let actual_trace = &incremental.traces[layer];
        if oracle_trace.layer != layer
            || actual_trace.layer != layer
            || oracle_trace.cache_positions != actual_trace.cache_length
        {
            return Err(format!(
                "layer {layer}: incremental trace identity mismatch"
            ));
        }
        let selected_experts_exact =
            oracle_trace.selected_experts_by_position == actual_trace.selected_experts_by_position;
        let maximum_route_weight_absolute_error = oracle_trace
            .route_weights_by_position
            .iter()
            .flatten()
            .zip(actual_trace.route_weights_by_position.iter().flatten())
            .map(|(expected, actual)| (expected - actual).abs())
            .fold(0.0_f32, f32::max);
        let capture = oracle
            .captures
            .get(&format!("layer_{layer:02}_incremental_final"))
            .ok_or_else(|| format!("layer {layer}: missing incremental oracle capture"))?;
        let expected = read_oracle_capture(oracle_manifest_path, capture, HIDDEN)?;
        let final_state = numerical_parity(&captures.layer_finals[layer], &expected)?;
        all_layer_parity_passed &= selected_experts_exact
            && maximum_route_weight_absolute_error <= 5.0e-4
            && final_state.passed;
        layer_parity.push(MetalLayerParity {
            layer,
            selected_experts_exact,
            maximum_route_weight_absolute_error,
            final_state,
        });
    }
    let expected_final_norm = read_oracle_capture(
        oracle_manifest_path,
        oracle
            .captures
            .get("incremental_final_norm")
            .ok_or("missing incremental final-norm oracle capture")?,
        HIDDEN,
    )?;
    let final_norm_parity = numerical_parity(&captures.final_norm, &expected_final_norm)?;
    let expected_logits = read_oracle_capture(
        oracle_manifest_path,
        oracle
            .captures
            .get("incremental_last_logits")
            .ok_or("missing incremental logit oracle capture")?,
        config.vocab_size,
    )?;
    let logits_parity = numerical_parity(&incremental.full_logits, &expected_logits)?;
    let (top20_token_identity, projected_top20_jsd_nats) =
        projected_top20_jsd(&expected_logits, &incremental.full_logits)?;
    let speedup_vs_pw0092_repeats = [
        158_521.015 / incremental.wall_ms,
        158_614.709 / incremental.wall_ms,
    ];
    let timing_gate_passed = incremental.wall_ms <= 20_000.0
        && speedup_vs_pw0092_repeats
            .iter()
            .all(|speedup| *speedup >= 5.0);
    checkpoint.release_file_pages()?;
    safety.checkpoint("candidate_buffers_released", true)?;
    let promotion_gates_passed = incremental.output_token == 13
        && all_layer_parity_passed
        && final_norm_parity.passed
        && logits_parity.passed
        && timing_gate_passed;
    if !promotion_gates_passed && !tomography_enabled {
        let minimum_free = safety
            .snapshots
            .iter()
            .map(|snapshot| snapshot.system_memory_free_percent)
            .min()
            .ok_or("missing failed-run safety snapshot")?;
        let maximum_peak = safety
            .snapshots
            .iter()
            .map(|snapshot| snapshot.process_peak_resident_bytes)
            .max()
            .ok_or("missing failed-run safety snapshot")?;
        let post_release = safety
            .snapshots
            .last()
            .ok_or("missing failed-run release snapshot")?
            .process_physical_footprint_bytes;
        let maximum_swap_growth = safety
            .snapshots
            .iter()
            .map(|snapshot| snapshot.swap_growth_bytes)
            .max()
            .ok_or("missing failed-run swap snapshot")?;
        let maximum_new_throttled = safety
            .snapshots
            .iter()
            .map(|snapshot| snapshot.new_throttled_pages)
            .max()
            .ok_or("missing failed-run throttle snapshot")?;
        let first_failed_layer = layer_parity.iter().find(|parity| {
            !parity.selected_experts_exact
                || parity.maximum_route_weight_absolute_error > 5.0e-4
                || !parity.final_state.passed
        });
        return Err(format!(
            "Metal incremental final gate failed: token {}, layers pass {}, first failed layer {:?}, norm pass {}, logits pass {}, wall {} ms, speedups {:?}, safety min-free {}%, peak {}, post-release {}, swap-growth {}, new-throttled {}",
            incremental.output_token,
            all_layer_parity_passed,
            first_failed_layer.map(|parity| (
                parity.layer,
                parity.selected_experts_exact,
                parity.maximum_route_weight_absolute_error,
                parity.final_state.relative_l2,
                parity.final_state.maximum_absolute_error,
                parity.final_state.equality_fraction,
            )),
            final_norm_parity.passed,
            logits_parity.passed,
            incremental.wall_ms,
            speedup_vs_pw0092_repeats,
            minimum_free,
            maximum_peak,
            post_release,
            maximum_swap_growth,
            maximum_new_throttled,
        ));
    }
    if incremental.output_token != 13
        || !all_layer_parity_passed
        || !final_norm_parity.passed
        || !logits_parity.passed
        || !timing_gate_passed
    {
        safety.checkpoint("tomography_candidate_rejected", true)?;
    } else {
        safety.checkpoint("token_2_accepted", true)?;
    }
    safety.checkpoint("final_release", true)?;
    ledger.actual_process_disk_bytes_read = process_disk_bytes_read()?
        .checked_sub(disk_bytes_read_before)
        .ok_or("process disk byte counter moved backwards")?;
    ledger.peak_resident_bytes = peak_resident_bytes()?;
    let prefill_text = tokenizer
        .decode(&[prefill.output_token], false)
        .map_err(|error| format!("tokenizer output decode: {error}"))?;
    let incremental_text = tokenizer
        .decode(&[incremental.output_token], false)
        .map_err(|error| format!("tokenizer output decode: {error}"))?;
    let generated_token_ids = vec![prefill.output_token, incremental.output_token];
    let generated_text = tokenizer
        .decode(&generated_token_ids, false)
        .map_err(|error| format!("tokenizer generated decode: {error}"))?;
    let prefill_wall_ms = prefill.wall_ms;
    let incremental_wall_ms = incremental.wall_ms;
    let steps = vec![
        DecodeStepReport {
            input_token_id: *prompt_token_ids.last().ok_or("empty prompt")?,
            input_token_ids: prompt_token_ids.clone(),
            output_token_id: prefill.output_token,
            output_token_text: prefill_text,
            top_logits: prefill.top_logits,
            full_logits: Some(prefill.full_logits),
            layer_traces: prefill.traces,
            wall_ms: prefill_wall_ms,
        },
        DecodeStepReport {
            input_token_id: 264,
            input_token_ids: vec![264],
            output_token_id: incremental.output_token,
            output_token_text: incremental_text,
            top_logits: incremental.top_logits,
            full_logits: Some(incremental.full_logits),
            layer_traces: incremental.traces,
            wall_ms: incremental_wall_ms,
        },
    ];
    let report = MetalIncrementalTextReport {
        schema_version: 1,
        semantic: if tomography_enabled {
            "mimo_v2_5_bounded_metal_incremental_weight_install_tomography"
        } else {
            "mimo_v2_5_target_faithful_bounded_metal_incremental_text_endpoint"
        },
        revision: REVISION,
        commit: commit.to_owned(),
        fixture_sha256: sha256_hex(&fixture_bytes),
        checkpoint_verification_sha256: verification_sha256,
        oracle_manifest_sha256,
        kernel_sha256: runtime.kernel_sha256.clone(),
        kernel_compile_ms: runtime.compile_ms,
        metal_device: runtime.device_name.clone(),
        prompt_token_ids,
        generated_token_ids,
        generated_text,
        steps,
        layer_parity,
        final_norm_parity,
        logits_parity,
        top20_token_identity,
        projected_top20_jsd_nats,
        ledger,
        metal_ledger,
        safety_snapshots: safety.snapshots,
        complete_wall_ms: complete_started.elapsed().as_secs_f64() * 1000.0,
        prefill_wall_ms,
        incremental_wall_ms,
        speedup_vs_pw0092_repeats,
        timing_gate_passed,
        batch_size: 1,
        concurrency: 1,
        accepted_tokens_in_timed_interval: usize::from(promotion_gates_passed),
        accepted_per_verification: 1,
        cache_state: "cold process and verified SSD mmap; CPU prefill; retained K/V; warm process-local Metal pipeline; bounded copied expert tensors released per projection",
        exactness: "L3 bounded arithmetic approximation: source weights/routes and value-derived sparse BF16 midpoint repair",
        performance_claim: None,
        implementation: "single_rust_authority_retained_kv_cpu_attention_bounded_source_fp8_metal_experts_sparse_bf16_repair",
        promotion_gates_passed,
        status: if promotion_gates_passed {
            "promotion_gates_passed"
        } else {
            "diagnostic_complete_candidate_gates_failed"
        },
    };
    let report_bytes = serde_json::to_vec_pretty(&report).map_err(|error| error.to_string())?;
    write_create_new(output_path, &report_bytes)?;
    Ok(report)
}

fn host_page_bytes() -> Result<usize, String> {
    // SAFETY: `_SC_PAGESIZE` is a side-effect-free process query.
    let value = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    let page = usize::try_from(value).map_err(|_| "host page-size query failed")?;
    if page == 0 || !page.is_power_of_two() {
        return Err(format!("invalid host page size {page}"));
    }
    Ok(page)
}

fn read_layer4_oracle(
    oracle_manifest_path: &Path,
    verification_sha256: &str,
) -> Result<(Vec<u8>, String, Layer4CachedOracleManifest), String> {
    const ORACLE_SHA256: &str = "9c96d85e45832abdccd3be2325db993749579a904469d1862c8f3437cafab86d";
    let bytes = fs::read(oracle_manifest_path)
        .map_err(|error| format!("{}: {error}", oracle_manifest_path.display()))?;
    let hash = sha256_hex(&bytes);
    if hash != ORACLE_SHA256 {
        return Err("PW-0101 oracle manifest SHA-256 mismatch".to_owned());
    }
    let oracle: Layer4CachedOracleManifest = serde_json::from_slice(&bytes)
        .map_err(|error| format!("layer-4 cached oracle: {error}"))?;
    if oracle.schema_version != 1
        || oracle.semantic != "mimo_pytorch_layer4_partial_cached_oracle"
        || oracle.revision != REVISION
        || oracle.checkpoint_verification_sha256 != verification_sha256
        || oracle.last_layer != 4
        || oracle.layer4_routes.selected_experts.len() != TOP_K
        || oracle.layer4_routes.route_weights.len() != TOP_K
        || oracle
            .layer4_routes
            .selected_experts
            .iter()
            .copied()
            .collect::<BTreeSet<_>>()
            .len()
            != TOP_K
    {
        return Err("PW-0101 oracle authority mismatch".to_owned());
    }
    Ok((bytes, hash, oracle))
}

fn verify_artifact_source_authority(
    manifest: &RoutedLayerArtifactManifest,
    checkpoint: &Checkpoint,
) -> Result<(), String> {
    for record in &manifest.tensors {
        let expected_name = if record.role == "weight" {
            format!(
                "model.layers.{}.mlp.experts.{}.{}_proj.weight",
                manifest.layer, record.expert, record.projection
            )
        } else if record.role == "scale" {
            format!(
                "model.layers.{}.mlp.experts.{}.{}_proj.weight_scale_inv",
                manifest.layer, record.expert, record.projection
            )
        } else {
            return Err("artifact source record has an unknown role".to_owned());
        };
        let source = checkpoint.source_tensor(&expected_name)?;
        if record.artifact_metadata.name != expected_name
            || record.source_shard != source.shard
            || record.source_shard_sha256 != source.shard_sha256
            || record.source_absolute_offsets != source.absolute_offsets
            || record.artifact_metadata.dtype != source.view.metadata.dtype
            || record.artifact_metadata.shape != source.view.metadata.shape
            || record.artifact_metadata.data_bytes != source.view.metadata.data_bytes
            || record.source_tensor_sha256 != sha256_hex(source.view.bytes)
        {
            return Err(format!(
                "{}: artifact-to-checkpoint source binding mismatch",
                record.artifact_metadata.name
            ));
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn build_layer4_metal_ready_artifact(
    checkpoint_root: &Path,
    model_lock_path: &Path,
    verification_path: &Path,
    fixture_path: &Path,
    oracle_manifest_path: &Path,
    artifact_path: &Path,
    artifact_manifest_path: &Path,
    output_path: &Path,
    commit: &str,
) -> Result<RoutedLayerArtifactBuildReport, String> {
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
    let EndpointAuthority {
        fixture,
        mut safety,
        checkpoint,
        verification_sha256,
        ..
    } = open_endpoint_authority(
        checkpoint_root,
        model_lock_path,
        verification_path,
        fixture_path,
    )?;
    validate_slow_endpoint_fixture(&fixture)?;
    let (_, oracle_manifest_sha256, oracle) =
        read_layer4_oracle(oracle_manifest_path, &verification_sha256)?;
    safety.checkpoint("artifact_authorities_open", true)?;
    let page_bytes = host_page_bytes()?;
    let mut sources = Vec::with_capacity(TOP_K * 6);
    for &expert in &oracle.layer4_routes.selected_experts {
        for projection in ["gate", "up", "down"] {
            for role in ["weight", "scale"] {
                let tensor_name = if role == "weight" {
                    format!("model.layers.4.mlp.experts.{expert}.{projection}_proj.weight")
                } else {
                    format!(
                        "model.layers.4.mlp.experts.{expert}.{projection}_proj.weight_scale_inv"
                    )
                };
                let source = checkpoint.source_tensor(&tensor_name)?;
                sources.push(RoutedLayerSourceTensor {
                    expert,
                    projection,
                    role,
                    source_shard: source.shard,
                    source_shard_sha256: source.shard_sha256,
                    source_absolute_offsets: source.absolute_offsets,
                    metadata: source.view.metadata,
                    bytes: source.view.bytes,
                });
            }
        }
    }
    if sources.len() != TOP_K * 6 {
        return Err("layer-4 artifact source count mismatch".to_owned());
    }
    let construction_started = Instant::now();
    let manifest = build_routed_layer_artifact(
        artifact_path,
        artifact_manifest_path,
        REVISION,
        commit,
        &verification_sha256,
        &oracle_manifest_sha256,
        4,
        page_bytes,
        oracle.layer4_routes.selected_experts.clone(),
        &sources,
    )?;
    let construction_wall_ms = construction_started.elapsed().as_secs_f64() * 1000.0;
    drop(sources);
    safety.checkpoint("artifact_constructed", true)?;
    let verification_started = Instant::now();
    let verified = open_routed_layer_artifact(artifact_path, artifact_manifest_path, true)?;
    let fresh_verification_wall_ms = verification_started.elapsed().as_secs_f64() * 1000.0;
    if verified.manifest.revision != REVISION
        || verified.manifest.commit != commit
        || verified.manifest.checkpoint_verification_sha256 != verification_sha256
        || verified.manifest.oracle_manifest_sha256 != oracle_manifest_sha256
        || verified.manifest.layer != 4
        || verified.manifest.page_bytes != page_bytes
        || verified.manifest.selected_experts != oracle.layer4_routes.selected_experts
        || verified.manifest.tensors.len() != TOP_K * 6
    {
        return Err("fresh artifact authority verification mismatch".to_owned());
    }
    verify_artifact_source_authority(&verified.manifest, &checkpoint)?;
    let artifact_manifest_sha256 = verified.manifest_sha256.clone();
    drop(verified);
    safety.checkpoint("artifact_fresh_mapping_released", true)?;
    let report = RoutedLayerArtifactBuildReport {
        schema_version: 1,
        semantic: "mimo_v2_5_l1_page_stable_layer4_artifact_build",
        revision: REVISION,
        commit: commit.to_owned(),
        checkpoint_verification_sha256: verification_sha256,
        oracle_manifest_sha256,
        artifact_manifest_sha256,
        artifact_sha256: manifest.artifact_sha256,
        artifact_bytes: manifest.artifact_bytes,
        page_bytes,
        layer: 4,
        selected_experts: manifest.selected_experts,
        tensor_records: manifest.tensors.len(),
        construction_wall_ms,
        fresh_verification_wall_ms,
        safety_snapshots: safety.snapshots,
        exactness: "L1 function-preserving lossless page-aligned runtime layout",
        performance_claim: None,
    };
    write_create_new(
        output_path,
        &serde_json::to_vec_pretty(&report).map_err(|error| error.to_string())?,
    )?;
    Ok(report)
}

struct ArtifactExpertBindings<'a> {
    expert: u32,
    gate: ValidatedMappedFp8<'a>,
    up: ValidatedMappedFp8<'a>,
    down: ValidatedMappedFp8<'a>,
    backing: [[usize; 2]; 3],
}

fn diagnose_layer4_expert(
    oracle_manifest_path: &Path,
    oracle: &Layer4CachedOracleManifest,
    expert: u32,
    execution: &BoundedMetalExpertOutput,
) -> Result<MetalExpertDiagnostic, String> {
    let captures = oracle
        .layer4_expert_captures
        .get(&expert.to_string())
        .ok_or_else(|| format!("missing layer-4 expert {expert} oracle captures"))?;
    let mut stages = Vec::with_capacity(4);
    for (stage, actual, pre_round, repairs) in [
        (
            "gate",
            execution.gate.as_slice(),
            Some(execution.gate_pre_round.as_slice()),
            execution.sparse_repair_counts[0],
        ),
        (
            "up",
            execution.up.as_slice(),
            Some(execution.up_pre_round.as_slice()),
            execution.sparse_repair_counts[1],
        ),
        ("swiglu", execution.swiglu.as_slice(), None, 0),
        (
            "down",
            execution.down.as_slice(),
            Some(execution.down_pre_round.as_slice()),
            execution.sparse_repair_counts[2],
        ),
    ] {
        let expected = read_oracle_capture(
            oracle_manifest_path,
            captures
                .get(stage)
                .ok_or_else(|| format!("missing expert {expert} {stage} capture"))?,
            actual.len(),
        )?;
        stages.push(metal_stage_diagnostic(
            stage, actual, &expected, pre_round, repairs,
        )?);
    }
    Ok(MetalExpertDiagnostic { expert, stages })
}

fn f32_values_sha256(values: &[f32]) -> String {
    let bytes = values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect::<Vec<_>>();
    sha256_hex(&bytes)
}

#[allow(clippy::too_many_arguments)]
fn finish_layer4_artifact_trial(
    repetition: usize,
    cache_state: &'static str,
    variant: &'static str,
    mapping_open_ms: f64,
    trusted_tensor_bind_ms: f64,
    initial_invalidation_ms: f64,
    layer_started: Instant,
    activity_before: ProcessActivity,
    final_release_ms: f64,
    installed_source_bytes: u64,
    sparse_repair_counts: [u64; 3],
    expert_tomography: Vec<ExpertTomography>,
    expert_diagnostics: Vec<MetalExpertDiagnostic>,
    routed: Vec<f32>,
    post_attention: &[f32],
    expected_routed: &[f32],
    expected_final: &[f32],
) -> Result<RoutedLayerArtifactTrial, String> {
    let mut final_residual = post_attention
        .iter()
        .zip(&routed)
        .map(|(&residual, &projected)| residual + projected)
        .collect::<Vec<_>>();
    round_bf16_values(&mut final_residual);
    Ok(RoutedLayerArtifactTrial {
        repetition,
        cache_state,
        variant,
        mapping_open_ms,
        trusted_tensor_bind_ms,
        initial_invalidation_ms,
        layer_wall_ms: layer_started.elapsed().as_secs_f64() * 1000.0,
        final_release_ms,
        activity: process_activity()?.checked_delta(activity_before)?,
        installed_source_bytes,
        sparse_repair_counts,
        expert_tomography,
        expert_diagnostics,
        routed_sha256: f32_values_sha256(&routed),
        final_residual_sha256: f32_values_sha256(&final_residual),
        routed_parity: numerical_parity(&routed, expected_routed)?,
        final_residual_parity: numerical_parity(&final_residual, expected_final)?,
    })
}

#[allow(clippy::too_many_arguments)]
fn run_layer4_control_trial(
    repetition: usize,
    cache_state: &'static str,
    checkpoint: &Checkpoint,
    runtime: &BoundedMetalExpertRuntime,
    oracle_manifest_path: &Path,
    oracle: &Layer4CachedOracleManifest,
    selected_sorted: &[u32],
    weight_by_expert: &BTreeMap<u32, f32>,
    moe_input: &[f32],
    post_attention: &[f32],
    expected_routed: &[f32],
    expected_final: &[f32],
) -> Result<RoutedLayerArtifactTrial, String> {
    let invalidation_started = Instant::now();
    if cache_state == "cold" {
        checkpoint.release_file_pages()?;
        pressure_relief();
    }
    let initial_invalidation_ms = invalidation_started.elapsed().as_secs_f64() * 1000.0;
    let layer_started = Instant::now();
    let activity_before = process_activity()?;
    let mut routed = vec![0.0_f32; HIDDEN];
    let down_shape_authority = vec![0.0_f32; MOE_INTERMEDIATE];
    let mut expert_tomography = Vec::with_capacity(TOP_K);
    let mut expert_diagnostics = Vec::with_capacity(TOP_K);
    let mut installed_source_bytes = 0_u64;
    let mut sparse_repair_counts = [0_u64; 3];
    for &expert in selected_sorted {
        let expert_started = Instant::now();
        let expert_activity_before = process_activity()?;
        let prefix = format!("model.layers.4.mlp.experts.{expert}");
        let validation_started = Instant::now();
        let tensor = |projection: &str| -> Result<_, String> {
            let name = format!("{prefix}.{projection}_proj.weight");
            validate_fp8_views(
                checkpoint.tensor(&name)?,
                checkpoint.tensor(&format!("{name}_scale_inv"))?,
                if projection == "down" {
                    &down_shape_authority
                } else {
                    moe_input
                },
            )
        };
        let gate = tensor("gate")?;
        let up = tensor("up")?;
        let down = tensor("down")?;
        let tensor_lookup_validation_ms = validation_started.elapsed().as_secs_f64() * 1000.0;
        let mut execution = runtime.execute_profiled(4, expert, [&gate, &up, &down], moe_input)?;
        let scatter_started = Instant::now();
        for (destination, value) in routed.iter_mut().zip(&execution.down) {
            *destination += *value * weight_by_expert[&expert];
        }
        let weighted_scatter_ms = scatter_started.elapsed().as_secs_f64() * 1000.0;
        let release_started = Instant::now();
        release_matrix_transients(checkpoint)?;
        let matrix_transient_release_ms = release_started.elapsed().as_secs_f64() * 1000.0;
        let mut tomography = execution
            .tomography
            .take()
            .ok_or("control trial lacks expert tomography")?;
        tomography.source_shards = ["gate", "up", "down"]
            .iter()
            .map(|projection| {
                checkpoint
                    .shard_for_tensor(&format!("{prefix}.{projection}_proj.weight"))
                    .map(str::to_owned)
            })
            .collect::<Result<Vec<_>, _>>()?;
        tomography.tensor_lookup_validation_ms = tensor_lookup_validation_ms;
        tomography.weighted_scatter_ms = weighted_scatter_ms;
        tomography.matrix_transient_release_ms = matrix_transient_release_ms;
        tomography.wall_ms = expert_started.elapsed().as_secs_f64() * 1000.0;
        tomography.activity = process_activity()?.checked_delta(expert_activity_before)?;
        installed_source_bytes = installed_source_bytes
            .checked_add(execution.installed_source_bytes)
            .ok_or("control installed-source ledger overflow")?;
        for (total, count) in sparse_repair_counts
            .iter_mut()
            .zip(execution.sparse_repair_counts)
        {
            *total += count as u64;
        }
        expert_diagnostics.push(diagnose_layer4_expert(
            oracle_manifest_path,
            oracle,
            expert,
            &execution,
        )?);
        expert_tomography.push(tomography);
    }
    round_bf16_values(&mut routed);
    finish_layer4_artifact_trial(
        repetition,
        cache_state,
        "C0_copied_global_release",
        0.0,
        0.0,
        initial_invalidation_ms,
        layer_started,
        activity_before,
        0.0,
        installed_source_bytes,
        sparse_repair_counts,
        expert_tomography,
        expert_diagnostics,
        routed,
        post_attention,
        expected_routed,
        expected_final,
    )
}

#[allow(clippy::too_many_arguments)]
fn run_layer4_artifact_trial(
    repetition: usize,
    cache_state: &'static str,
    no_copy: bool,
    artifact_path: &Path,
    artifact_manifest_path: &Path,
    runtime: &BoundedMetalExpertRuntime,
    oracle_manifest_path: &Path,
    oracle: &Layer4CachedOracleManifest,
    selected_sorted: &[u32],
    weight_by_expert: &BTreeMap<u32, f32>,
    moe_input: &[f32],
    post_attention: &[f32],
    expected_routed: &[f32],
    expected_final: &[f32],
) -> Result<RoutedLayerArtifactTrial, String> {
    let open_started = Instant::now();
    let artifact = open_routed_layer_artifact(artifact_path, artifact_manifest_path, false)?;
    let mapping_open_ms = open_started.elapsed().as_secs_f64() * 1000.0;
    let invalidation_started = Instant::now();
    if cache_state == "cold" {
        artifact.invalidate_pages()?;
        pressure_relief();
    }
    let initial_invalidation_ms = invalidation_started.elapsed().as_secs_f64() * 1000.0;
    let layer_started = Instant::now();
    let activity_before = process_activity()?;
    let bind_started = Instant::now();
    let down_shape_authority = vec![0.0_f32; MOE_INTERMEDIATE];
    let mut bindings = Vec::with_capacity(TOP_K);
    for &expert in selected_sorted {
        let (gate, gate_backing) = artifact.validated_fp8(expert, "gate", moe_input)?;
        let (up, up_backing) = artifact.validated_fp8(expert, "up", moe_input)?;
        let (down, down_backing) = artifact.validated_fp8(expert, "down", &down_shape_authority)?;
        bindings.push(ArtifactExpertBindings {
            expert,
            gate,
            up,
            down,
            backing: [gate_backing, up_backing, down_backing],
        });
    }
    let trusted_tensor_bind_ms = bind_started.elapsed().as_secs_f64() * 1000.0;
    let mut routed = vec![0.0_f32; HIDDEN];
    let mut expert_tomography = Vec::with_capacity(TOP_K);
    let mut expert_diagnostics = Vec::with_capacity(TOP_K);
    let mut installed_source_bytes = 0_u64;
    let mut sparse_repair_counts = [0_u64; 3];
    for binding in &bindings {
        let expert_started = Instant::now();
        let expert_activity_before = process_activity()?;
        let projections = [&binding.gate, &binding.up, &binding.down];
        let mut execution = if no_copy {
            runtime.execute_profiled_no_copy(
                4,
                binding.expert,
                projections,
                binding.backing.map(|lengths| NoCopyProjectionBacking {
                    weight_region_bytes: lengths[0],
                    scale_region_bytes: lengths[1],
                    page_bytes: artifact.manifest.page_bytes,
                }),
                moe_input,
            )?
        } else {
            runtime.execute_profiled(4, binding.expert, projections, moe_input)?
        };
        let scatter_started = Instant::now();
        for (destination, value) in routed.iter_mut().zip(&execution.down) {
            *destination += *value * weight_by_expert[&binding.expert];
        }
        let weighted_scatter_ms = scatter_started.elapsed().as_secs_f64() * 1000.0;
        let mut tomography = execution
            .tomography
            .take()
            .ok_or("artifact trial lacks expert tomography")?;
        tomography.source_shards = vec![artifact.manifest.artifact_file.clone(); 3];
        tomography.tensor_lookup_validation_ms = 0.0;
        tomography.weighted_scatter_ms = weighted_scatter_ms;
        tomography.matrix_transient_release_ms = 0.0;
        tomography.wall_ms = expert_started.elapsed().as_secs_f64() * 1000.0;
        tomography.activity = process_activity()?.checked_delta(expert_activity_before)?;
        installed_source_bytes = installed_source_bytes
            .checked_add(execution.installed_source_bytes)
            .ok_or("artifact installed-source ledger overflow")?;
        for (total, count) in sparse_repair_counts
            .iter_mut()
            .zip(execution.sparse_repair_counts)
        {
            *total += count as u64;
        }
        expert_diagnostics.push(diagnose_layer4_expert(
            oracle_manifest_path,
            oracle,
            binding.expert,
            &execution,
        )?);
        expert_tomography.push(tomography);
    }
    round_bf16_values(&mut routed);
    drop(bindings);
    let release_started = Instant::now();
    if cache_state == "cold" {
        artifact.invalidate_pages()?;
    }
    drop(artifact);
    pressure_relief();
    let final_release_ms = release_started.elapsed().as_secs_f64() * 1000.0;
    finish_layer4_artifact_trial(
        repetition,
        cache_state,
        if no_copy {
            "C2_artifact_no_copy"
        } else {
            "C1_artifact_copied"
        },
        mapping_open_ms,
        trusted_tensor_bind_ms,
        initial_invalidation_ms,
        layer_started,
        activity_before,
        final_release_ms,
        installed_source_bytes,
        sparse_repair_counts,
        expert_tomography,
        expert_diagnostics,
        routed,
        post_attention,
        expected_routed,
        expected_final,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn benchmark_layer4_metal_ready_artifact(
    checkpoint_root: &Path,
    model_lock_path: &Path,
    verification_path: &Path,
    fixture_path: &Path,
    oracle_manifest_path: &Path,
    artifact_path: &Path,
    artifact_manifest_path: &Path,
    kernel_path: &Path,
    output_path: &Path,
    commit: &str,
) -> Result<RoutedLayerArtifactBenchmarkReport, String> {
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
    let EndpointAuthority {
        fixture,
        mut safety,
        checkpoint,
        verification_sha256,
        ..
    } = open_endpoint_authority(
        checkpoint_root,
        model_lock_path,
        verification_path,
        fixture_path,
    )?;
    validate_slow_endpoint_fixture(&fixture)?;
    let (_, oracle_manifest_sha256, oracle) =
        read_layer4_oracle(oracle_manifest_path, &verification_sha256)?;
    let verified = open_routed_layer_artifact(artifact_path, artifact_manifest_path, true)?;
    if verified.manifest.revision != REVISION
        || verified.manifest.checkpoint_verification_sha256 != verification_sha256
        || verified.manifest.oracle_manifest_sha256 != oracle_manifest_sha256
        || verified.manifest.layer != 4
        || verified.manifest.page_bytes != host_page_bytes()?
        || verified.manifest.selected_experts != oracle.layer4_routes.selected_experts
        || verified.manifest.tensors.len() != TOP_K * 6
    {
        return Err("benchmark artifact authority mismatch".to_owned());
    }
    verify_artifact_source_authority(&verified.manifest, &checkpoint)?;
    let artifact_manifest_sha256 = verified.manifest_sha256.clone();
    let artifact_sha256 = verified.manifest.artifact_sha256.clone();
    safety.checkpoint("artifact_full_verification_complete", true)?;
    let runtime = BoundedMetalExpertRuntime::compile(kernel_path)?;
    let (probe_region, probe_page_bytes) = verified.no_copy_probe_region()?;
    let no_copy_probe_ms = runtime.probe_no_copy_mapping(probe_region, probe_page_bytes)?;
    drop(verified);
    safety.checkpoint("no_copy_probe_mapping_released", true)?;
    let moe_input = read_oracle_capture(
        oracle_manifest_path,
        oracle
            .layer4_captures
            .get("moe_input")
            .ok_or("missing layer-4 MoE input capture")?,
        HIDDEN,
    )?;
    let post_attention = read_oracle_capture(
        oracle_manifest_path,
        oracle
            .layer4_captures
            .get("post_attention")
            .ok_or("missing layer-4 post-attention capture")?,
        HIDDEN,
    )?;
    let expected_routed = read_oracle_capture(
        oracle_manifest_path,
        oracle
            .layer4_captures
            .get("routed")
            .ok_or("missing layer-4 routed capture")?,
        HIDDEN,
    )?;
    let expected_final = read_oracle_capture(
        oracle_manifest_path,
        oracle
            .layer4_captures
            .get("final")
            .ok_or("missing layer-4 final capture")?,
        HIDDEN,
    )?;
    let mut route_ledger = EndpointLedger::default();
    let routing = route_mlp(&checkpoint, 4, &moe_input, 1, &mut route_ledger)?;
    if routing.selected.len() != 1
        || routing.weights.len() != 1
        || routing.selected[0] != oracle.layer4_routes.selected_experts
    {
        return Err("layer-4 artifact benchmark route mismatch".to_owned());
    }
    let maximum_route_weight_absolute_error = routing.weights[0]
        .iter()
        .zip(&oracle.layer4_routes.route_weights)
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0_f32, f32::max);
    if maximum_route_weight_absolute_error > 3.0e-8 {
        return Err("layer-4 artifact benchmark route-weight mismatch".to_owned());
    }
    let weight_by_expert = routing.selected[0]
        .iter()
        .copied()
        .zip(routing.weights[0].iter().copied())
        .collect::<BTreeMap<_, _>>();
    let selected_sorted = weight_by_expert.keys().copied().collect::<Vec<_>>();
    let mut trials = Vec::with_capacity(18);
    let rotated_orders = [[0_u8, 1, 2], [1_u8, 2, 0], [2_u8, 0, 1]];
    let mut warm_prefault_ms = 0.0;
    let mut warm_prefault_checksum = 0_u64;
    for &cache_state in &["cold", "warm"] {
        if cache_state == "warm" {
            let warm_started = Instant::now();
            let artifact =
                open_routed_layer_artifact(artifact_path, artifact_manifest_path, false)?;
            warm_prefault_checksum = artifact.prefault_pages();
            drop(artifact);
            warm_prefault_ms = warm_started.elapsed().as_secs_f64() * 1000.0;
            safety.checkpoint("artifact_warm_prefault_complete", true)?;
        }
        for (repetition, order) in rotated_orders.iter().enumerate() {
            for variant in order {
                let trial = match variant {
                    0 => run_layer4_control_trial(
                        repetition,
                        cache_state,
                        &checkpoint,
                        &runtime,
                        oracle_manifest_path,
                        &oracle,
                        &selected_sorted,
                        &weight_by_expert,
                        &moe_input,
                        &post_attention,
                        &expected_routed,
                        &expected_final,
                    )?,
                    1 | 2 => run_layer4_artifact_trial(
                        repetition,
                        cache_state,
                        *variant == 2,
                        artifact_path,
                        artifact_manifest_path,
                        &runtime,
                        oracle_manifest_path,
                        &oracle,
                        &selected_sorted,
                        &weight_by_expert,
                        &moe_input,
                        &post_attention,
                        &expected_routed,
                        &expected_final,
                    )?,
                    _ => return Err("invalid interleaved trial variant".to_owned()),
                };
                let expected_installed = 201_375_744_u64;
                if trial.installed_source_bytes != expected_installed
                    || trial.expert_diagnostics.len() != TOP_K
                    || trial.expert_tomography.len() != TOP_K
                {
                    return Err("layer-4 artifact trial accounting mismatch".to_owned());
                }
                trials.push(trial);
                safety.checkpoint(
                    &format!("{cache_state}_repetition_{repetition}_variant_{variant}_released"),
                    true,
                )?;
            }
        }
    }
    let first_routed = trials
        .first()
        .map(|trial| trial.routed_sha256.as_str())
        .ok_or("artifact benchmark produced no trials")?;
    let first_final = trials[0].final_residual_sha256.as_str();
    let first_expert_diagnostics =
        serde_json::to_vec(&trials[0].expert_diagnostics).map_err(|error| error.to_string())?;
    if trials.iter().any(|trial| {
        trial.routed_sha256 != first_routed
            || trial.final_residual_sha256 != first_final
            || serde_json::to_vec(&trial.expert_diagnostics)
                .map(|bytes| bytes != first_expert_diagnostics)
                .unwrap_or(true)
    }) {
        return Err("artifact benchmark cross-variant correctness mismatch".to_owned());
    }
    checkpoint.release_file_pages()?;
    pressure_relief();
    safety.checkpoint("benchmark_final_release", true)?;
    let report = RoutedLayerArtifactBenchmarkReport {
        schema_version: 1,
        semantic: "mimo_v2_5_layer4_page_stable_copied_no_copy_benchmark",
        revision: REVISION,
        commit: commit.to_owned(),
        checkpoint_verification_sha256: verification_sha256,
        oracle_manifest_sha256,
        artifact_manifest_sha256,
        artifact_sha256,
        kernel_sha256: runtime.kernel_sha256.clone(),
        kernel_compile_ms: runtime.compile_ms,
        metal_device: runtime.device_name.clone(),
        no_copy_probe_ms,
        no_copy_probe_passed: true,
        warm_prefault_ms,
        warm_prefault_checksum,
        selected_experts: routing.selected[0].clone(),
        route_weights: routing.weights[0].clone(),
        maximum_route_weight_absolute_error,
        trials,
        safety_snapshots: safety.snapshots,
        batch_size: 1,
        concurrency: 1,
        accepted_tokens: 0,
        accepted_per_verification: 0,
        unique_experts: TOP_K,
        exactness: "L1 storage/layout with unchanged rejected L3 Metal arithmetic candidate",
        performance_claim: None,
    };
    write_create_new(
        output_path,
        &serde_json::to_vec_pretty(&report).map_err(|error| error.to_string())?,
    )?;
    Ok(report)
}

fn metal_stage_diagnostic(
    stage: &'static str,
    actual: &[f32],
    expected: &[f32],
    pre_round: Option<&[f32]>,
    sparse_repairs: usize,
) -> Result<MetalStageDiagnostic, String> {
    let parity = numerical_parity(actual, expected)?;
    let first_mismatches = actual
        .iter()
        .zip(expected)
        .enumerate()
        .filter(|(_, (candidate, oracle))| candidate.to_bits() != oracle.to_bits())
        .take(8)
        .map(|(index, (candidate, oracle))| {
            let pre = pre_round.map_or(*candidate, |values| values[index]);
            let midpoint_distance = (pre.to_bits() & 0xffff).abs_diff(0x8000);
            format!(
                "index={index},actual={:#010x},expected={:#010x},pre={:#010x},midpoint_distance={midpoint_distance}",
                candidate.to_bits(),
                oracle.to_bits(),
                pre.to_bits(),
            )
        })
        .collect();
    Ok(MetalStageDiagnostic {
        stage,
        parity,
        sparse_repairs,
        first_mismatches,
    })
}

#[allow(clippy::too_many_arguments)]
pub fn run_layer4_metal_diagnostic(
    checkpoint_root: &Path,
    model_lock_path: &Path,
    verification_path: &Path,
    fixture_path: &Path,
    oracle_manifest_path: &Path,
    kernel_path: &Path,
    output_path: &Path,
    commit: &str,
) -> Result<Layer4MetalDiagnosticReport, String> {
    const ORACLE_SHA256: &str = "9c96d85e45832abdccd3be2325db993749579a904469d1862c8f3437cafab86d";
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
    let started = Instant::now();
    let EndpointAuthority {
        fixture,
        mut safety,
        checkpoint,
        verification_sha256,
        ..
    } = open_endpoint_authority(
        checkpoint_root,
        model_lock_path,
        verification_path,
        fixture_path,
    )?;
    validate_slow_endpoint_fixture(&fixture)?;
    let oracle_bytes = fs::read(oracle_manifest_path)
        .map_err(|error| format!("{}: {error}", oracle_manifest_path.display()))?;
    let oracle_manifest_sha256 = sha256_hex(&oracle_bytes);
    if oracle_manifest_sha256 != ORACLE_SHA256 {
        return Err("PW-0101 oracle manifest SHA-256 mismatch".to_owned());
    }
    let oracle: Layer4CachedOracleManifest = serde_json::from_slice(&oracle_bytes)
        .map_err(|error| format!("layer-4 cached oracle: {error}"))?;
    if oracle.schema_version != 1
        || oracle.semantic != "mimo_pytorch_layer4_partial_cached_oracle"
        || oracle.revision != REVISION
        || oracle.checkpoint_verification_sha256 != verification_sha256
        || oracle.last_layer != 4
        || oracle.layer4_routes.selected_experts.len() != TOP_K
        || oracle.layer4_routes.route_weights.len() != TOP_K
    {
        return Err("PW-0101 oracle authority mismatch".to_owned());
    }
    let moe_input = read_oracle_capture(
        oracle_manifest_path,
        oracle
            .layer4_captures
            .get("moe_input")
            .ok_or("missing layer-4 MoE input capture")?,
        HIDDEN,
    )?;
    let post_attention = read_oracle_capture(
        oracle_manifest_path,
        oracle
            .layer4_captures
            .get("post_attention")
            .ok_or("missing layer-4 post-attention capture")?,
        HIDDEN,
    )?;
    let expected_routed = read_oracle_capture(
        oracle_manifest_path,
        oracle
            .layer4_captures
            .get("routed")
            .ok_or("missing layer-4 routed capture")?,
        HIDDEN,
    )?;
    let expected_final = read_oracle_capture(
        oracle_manifest_path,
        oracle
            .layer4_captures
            .get("final")
            .ok_or("missing layer-4 final capture")?,
        HIDDEN,
    )?;
    let runtime = BoundedMetalExpertRuntime::compile(kernel_path)?;
    safety.checkpoint("metal_compile_complete", true)?;
    let mut endpoint_ledger = EndpointLedger::default();
    let routing = route_mlp(&checkpoint, 4, &moe_input, 1, &mut endpoint_ledger)?;
    if routing.selected.len() != 1
        || routing.weights.len() != 1
        || routing.selected[0] != oracle.layer4_routes.selected_experts
    {
        return Err("layer-4 Metal diagnostic route mismatch".to_owned());
    }
    let maximum_route_weight_absolute_error = routing.weights[0]
        .iter()
        .zip(&oracle.layer4_routes.route_weights)
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0_f32, f32::max);
    if maximum_route_weight_absolute_error > 3.0e-8 {
        return Err(format!(
            "layer-4 Metal diagnostic route-weight error {maximum_route_weight_absolute_error}"
        ));
    }
    let route_weights = routing.weights[0].clone();
    let selected_experts = routing.selected[0].clone();
    let weight_by_expert = selected_experts
        .iter()
        .copied()
        .zip(route_weights.iter().copied())
        .collect::<BTreeMap<_, _>>();
    let mut metal_ledger = MetalExpertLedger {
        tomography_enabled: true,
        ..MetalExpertLedger::default()
    };
    let mut routed = vec![0.0_f32; HIDDEN];
    let down_shape_authority = vec![0.0_f32; MOE_INTERMEDIATE];
    let mut expert_diagnostics = Vec::with_capacity(TOP_K);
    for expert in weight_by_expert.keys().copied() {
        let expert_wall_started = Instant::now();
        let expert_activity_started = process_activity()?;
        let prefix = format!("model.layers.4.mlp.experts.{expert}");
        let validation_started = Instant::now();
        let tensor = |projection: &str| -> Result<_, String> {
            let name = format!("{prefix}.{projection}_proj.weight");
            validate_fp8_views(
                checkpoint.tensor(&name)?,
                checkpoint.tensor(&format!("{name}_scale_inv"))?,
                if projection == "down" {
                    &down_shape_authority
                } else {
                    &moe_input
                },
            )
        };
        let gate = tensor("gate")?;
        let up = tensor("up")?;
        let down = tensor("down")?;
        let tensor_lookup_validation_ms = validation_started.elapsed().as_secs_f64() * 1000.0;
        let mut execution = runtime.execute_profiled(4, expert, [&gate, &up, &down], &moe_input)?;
        let scatter_started = Instant::now();
        for (destination, value) in routed.iter_mut().zip(&execution.down) {
            *destination += *value * weight_by_expert[&expert];
        }
        let weighted_scatter_ms = scatter_started.elapsed().as_secs_f64() * 1000.0;
        let release_started = Instant::now();
        checkpoint.release_file_pages()?;
        let matrix_transient_release_ms = release_started.elapsed().as_secs_f64() * 1000.0;
        let mut tomography = execution
            .tomography
            .take()
            .ok_or("layer-4 diagnostic lacks expert tomography")?;
        tomography.source_shards = ["gate", "up", "down"]
            .iter()
            .map(|projection| {
                checkpoint
                    .shard_for_tensor(&format!("{prefix}.{projection}_proj.weight"))
                    .map(str::to_owned)
            })
            .collect::<Result<Vec<_>, _>>()?;
        tomography.tensor_lookup_validation_ms = tensor_lookup_validation_ms;
        tomography.weighted_scatter_ms = weighted_scatter_ms;
        tomography.matrix_transient_release_ms = matrix_transient_release_ms;
        tomography.wall_ms = expert_wall_started.elapsed().as_secs_f64() * 1000.0;
        tomography.activity = process_activity()?.checked_delta(expert_activity_started)?;
        metal_ledger.expert_tomography.push(tomography);
        let captures = oracle
            .layer4_expert_captures
            .get(&expert.to_string())
            .ok_or_else(|| format!("missing layer-4 expert {expert} oracle captures"))?;
        let mut stages = Vec::with_capacity(4);
        for (stage, actual, pre_round, repairs) in [
            (
                "gate",
                execution.gate.as_slice(),
                Some(execution.gate_pre_round.as_slice()),
                execution.sparse_repair_counts[0],
            ),
            (
                "up",
                execution.up.as_slice(),
                Some(execution.up_pre_round.as_slice()),
                execution.sparse_repair_counts[1],
            ),
            ("swiglu", execution.swiglu.as_slice(), None, 0),
            (
                "down",
                execution.down.as_slice(),
                Some(execution.down_pre_round.as_slice()),
                execution.sparse_repair_counts[2],
            ),
        ] {
            let expected = read_oracle_capture(
                oracle_manifest_path,
                captures
                    .get(stage)
                    .ok_or_else(|| format!("missing expert {expert} {stage} capture"))?,
                actual.len(),
            )?;
            stages.push(metal_stage_diagnostic(
                stage, actual, &expected, pre_round, repairs,
            )?);
        }
        endpoint_ledger.logical_source_bytes = endpoint_ledger
            .logical_source_bytes
            .checked_add(execution.installed_source_bytes)
            .ok_or("layer-4 logical byte ledger overflow")?;
        endpoint_ledger.routed_expert_executions += 1;
        endpoint_ledger.dynamic_activation_groups += 80;
        endpoint_ledger.dynamic_activation_values += 10_240;
        metal_ledger.expert_executions += 1;
        metal_ledger.projection_dispatches += 3;
        metal_ledger.installed_source_bytes += execution.installed_source_bytes;
        metal_ledger.released_projection_buffers += 3;
        metal_ledger.sparse_decoded_weight_bytes += execution.sparse_decoded_weight_bytes;
        for (total, count) in metal_ledger
            .sparse_repair_counts
            .iter_mut()
            .zip(execution.sparse_repair_counts)
        {
            *total += count as u64;
        }
        expert_diagnostics.push(MetalExpertDiagnostic { expert, stages });
        safety.checkpoint(&format!("expert_{expert}_released"), true)?;
    }
    round_bf16_values(&mut routed);
    let routed_parity = numerical_parity(&routed, &expected_routed)?;
    let mut final_residual = post_attention
        .iter()
        .zip(&routed)
        .map(|(&residual, &projected)| residual + projected)
        .collect::<Vec<_>>();
    round_bf16_values(&mut final_residual);
    let final_residual_parity = numerical_parity(&final_residual, &expected_final)?;
    checkpoint.release_file_pages()?;
    safety.checkpoint("final_release", true)?;
    endpoint_ledger.peak_resident_bytes = peak_resident_bytes()?;
    let report = Layer4MetalDiagnosticReport {
        schema_version: 1,
        semantic: "mimo_layer4_exact_input_bounded_metal_divergence_diagnostic",
        revision: REVISION,
        commit: commit.to_owned(),
        checkpoint_verification_sha256: verification_sha256,
        oracle_manifest_sha256,
        kernel_sha256: runtime.kernel_sha256.clone(),
        kernel_compile_ms: runtime.compile_ms,
        metal_device: runtime.device_name.clone(),
        selected_experts,
        route_weights,
        maximum_route_weight_absolute_error,
        expert_diagnostics,
        routed_parity,
        final_residual_parity,
        metal_ledger,
        endpoint_ledger,
        safety_snapshots: safety.snapshots,
        wall_ms: started.elapsed().as_secs_f64() * 1000.0,
        batch_size: 1,
        concurrency: 1,
        accepted_tokens: 0,
        accepted_per_verification: 0,
        unique_experts: TOP_K,
        performance_claim: None,
    };
    let report_bytes = serde_json::to_vec_pretty(&report).map_err(|error| error.to_string())?;
    write_create_new(output_path, &report_bytes)?;
    Ok(report)
}

fn write_capture(
    output_dir: &Path,
    name: &str,
    shape: &[usize],
    values: &[f32],
) -> Result<CaptureRecord, String> {
    write_capture_typed(output_dir, name, shape, values, "BF16_widened_F32")
}

fn write_capture_typed(
    output_dir: &Path,
    name: &str,
    shape: &[usize],
    values: &[f32],
    dtype: &'static str,
) -> Result<CaptureRecord, String> {
    let expected = shape
        .iter()
        .try_fold(1_usize, |product, value| product.checked_mul(*value))
        .ok_or("capture shape overflow")?;
    if expected != values.len() || values.iter().any(|value| !value.is_finite()) {
        return Err(format!("{name}: capture shape or value mismatch"));
    }
    let bytes = values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect::<Vec<_>>();
    let file = format!("{name}.f32");
    write_create_new(&output_dir.join(&file), &bytes)?;
    Ok(CaptureRecord {
        file,
        shape: shape.to_vec(),
        dtype,
        sha256: sha256_hex(&bytes),
    })
}

fn execute_dense_layer0(
    checkpoint: &Checkpoint,
    config: &ModelConfig,
    prompt_token_ids: &[u32],
    ledger: &mut EndpointLedger,
    captures: &mut Layer0Captures,
) -> Result<Vec<f32>, String> {
    let rows = prompt_token_ids.len();
    let hidden = embedding(checkpoint, prompt_token_ids, ledger)?;
    let input_norm_weight = bf16_vector(
        checkpoint,
        "model.layers.0.input_layernorm.weight",
        HIDDEN,
        ledger,
    )?;
    let normalized = rms_norm(&hidden, rows, &input_norm_weight, config.layernorm_epsilon)?;
    let mut cache = LayerKvCache::default();
    let attention_output = attention(
        checkpoint,
        config,
        0,
        &normalized,
        rows,
        &mut cache,
        ledger,
        Some(captures),
    )?;
    let post_attention = hidden
        .iter()
        .zip(&attention_output)
        .map(|(&residual, &projected)| round_bf16(residual + projected))
        .collect::<Vec<_>>();
    let post_norm_weight = bf16_vector(
        checkpoint,
        "model.layers.0.post_attention_layernorm.weight",
        HIDDEN,
        ledger,
    )?;
    let post_attention_norm = rms_norm(
        &post_attention,
        rows,
        &post_norm_weight,
        config.layernorm_epsilon,
    )?;
    let down = dense_mlp(
        checkpoint,
        &post_attention_norm,
        rows,
        ledger,
        Some(captures),
    )?;
    let final_hidden = post_attention
        .iter()
        .zip(&down)
        .map(|(&residual, &projected)| round_bf16(residual + projected))
        .collect();
    Ok(final_hidden)
}

#[allow(clippy::too_many_arguments)]
pub fn run_real_layer0_trace(
    checkpoint_root: &Path,
    model_lock_path: &Path,
    verification_path: &Path,
    fixture_path: &Path,
    output_dir: &Path,
    commit: &str,
) -> Result<Layer0TraceReport, String> {
    if output_dir.exists() {
        return Err(format!("refusing to overwrite {}", output_dir.display()));
    }
    let complete_started = Instant::now();
    let disk_bytes_read_before = process_disk_bytes_read()?;
    let EndpointAuthority {
        fixture_bytes,
        fixture,
        mut safety,
        config,
        tokenizer: _,
        prompt_token_ids,
        checkpoint,
        verification_sha256,
    } = open_endpoint_authority(
        checkpoint_root,
        model_lock_path,
        verification_path,
        fixture_path,
    )?;
    if fixture.schema_version != 2 || prompt_token_ids != CHAT_PROMPT_IDS {
        return Err("layer-0 trace requires the frozen chat fixture".to_owned());
    }
    fs::create_dir(output_dir).map_err(|error| format!("{}: {error}", output_dir.display()))?;
    let rows = prompt_token_ids.len();
    let mut ledger = EndpointLedger::default();
    let hidden = embedding(&checkpoint, &prompt_token_ids, &mut ledger)?;
    safety.checkpoint("embedding_complete", true)?;
    let input_norm_weight = bf16_vector(
        &checkpoint,
        "model.layers.0.input_layernorm.weight",
        HIDDEN,
        &mut ledger,
    )?;
    let normalized = rms_norm(&hidden, rows, &input_norm_weight, config.layernorm_epsilon)?;
    safety.checkpoint("input_norm_complete", true)?;
    let mut internal = Layer0Captures::default();
    let mut cache = LayerKvCache::default();
    let attention_output = attention(
        &checkpoint,
        &config,
        0,
        &normalized,
        rows,
        &mut cache,
        &mut ledger,
        Some(&mut internal),
    )?;
    safety.checkpoint("attention_complete", true)?;
    let post_attention = hidden
        .iter()
        .zip(&attention_output)
        .map(|(&residual, &projected)| round_bf16(residual + projected))
        .collect::<Vec<_>>();
    let post_norm_weight = bf16_vector(
        &checkpoint,
        "model.layers.0.post_attention_layernorm.weight",
        HIDDEN,
        &mut ledger,
    )?;
    let post_attention_norm = rms_norm(
        &post_attention,
        rows,
        &post_norm_weight,
        config.layernorm_epsilon,
    )?;
    safety.checkpoint("post_attention_norm_complete", true)?;
    let down = dense_mlp(
        &checkpoint,
        &post_attention_norm,
        rows,
        &mut ledger,
        Some(&mut internal),
    )?;
    let final_hidden = post_attention
        .iter()
        .zip(&down)
        .map(|(&residual, &projected)| round_bf16(residual + projected))
        .collect::<Vec<_>>();
    checkpoint.release_file_pages()?;
    safety.checkpoint("layer_0_complete", true)?;

    let mut captures = BTreeMap::new();
    for (name, shape, values) in [
        ("embedding", vec![rows, HIDDEN], hidden.as_slice()),
        ("input_norm", vec![rows, HIDDEN], normalized.as_slice()),
        ("qkv", vec![rows, 13_568], internal.qkv.as_slice()),
        (
            "query",
            vec![rows, HEADS, QK_HEAD_DIM],
            internal.query.as_slice(),
        ),
        ("key", vec![rows, 4, QK_HEAD_DIM], internal.key.as_slice()),
        (
            "value",
            vec![rows, 4, V_HEAD_DIM],
            internal.value.as_slice(),
        ),
        (
            "attention_scores",
            vec![HEADS * rows * (rows + 1) / 2],
            internal.attention_scores.as_slice(),
        ),
        (
            "attention_probabilities",
            vec![HEADS * rows * (rows + 1) / 2],
            internal.attention_probabilities.as_slice(),
        ),
        (
            "attention",
            vec![rows, HEADS, V_HEAD_DIM],
            internal.attention.as_slice(),
        ),
        (
            "attention_projection",
            vec![rows, HIDDEN],
            internal.attention_projection.as_slice(),
        ),
        (
            "post_attention",
            vec![rows, HIDDEN],
            post_attention.as_slice(),
        ),
        (
            "post_attention_norm",
            vec![rows, HIDDEN],
            post_attention_norm.as_slice(),
        ),
        ("gate", vec![rows, 16_384], internal.gate.as_slice()),
        ("up", vec![rows, 16_384], internal.up.as_slice()),
        ("swiglu", vec![rows, 16_384], internal.swiglu.as_slice()),
        ("down", vec![rows, HIDDEN], internal.down.as_slice()),
        ("final", vec![rows, HIDDEN], final_hidden.as_slice()),
    ] {
        captures.insert(
            name.to_owned(),
            write_capture(output_dir, name, &shape, values)?,
        );
    }
    safety.checkpoint("captures_written", true)?;
    ledger.actual_process_disk_bytes_read = process_disk_bytes_read()?
        .checked_sub(disk_bytes_read_before)
        .ok_or("process disk byte counter moved backwards")?;
    ledger.peak_resident_bytes = peak_resident_bytes()?;
    let report = Layer0TraceReport {
        schema_version: 1,
        semantic: "mimo_real_layer0_bf16_dynamic_fp8_rust_trace",
        revision: REVISION,
        commit: commit.to_owned(),
        fixture_sha256: sha256_hex(&fixture_bytes),
        checkpoint_verification_sha256: verification_sha256,
        prompt_token_ids,
        numerics: "dynamic_fp8_e4m3fn_per_token_group_128_bf16_boundaries",
        captures,
        ledger,
        safety_snapshots: safety.snapshots,
        complete_wall_ms: complete_started.elapsed().as_secs_f64() * 1000.0,
        batch_size: 1,
        prompt_positions: rows,
        concurrency: 1,
        accepted_tokens: 0,
        performance_claim: None,
    };
    let bytes = serde_json::to_vec_pretty(&report).map_err(|error| error.to_string())?;
    write_create_new(&output_dir.join("manifest.json"), &bytes)?;
    Ok(report)
}

#[allow(clippy::too_many_arguments)]
pub fn run_real_layer1_routing_trace(
    checkpoint_root: &Path,
    model_lock_path: &Path,
    verification_path: &Path,
    fixture_path: &Path,
    output_dir: &Path,
    commit: &str,
) -> Result<Layer1RoutingTraceReport, String> {
    if output_dir.exists() {
        return Err(format!("refusing to overwrite {}", output_dir.display()));
    }
    let complete_started = Instant::now();
    let disk_bytes_read_before = process_disk_bytes_read()?;
    let EndpointAuthority {
        fixture_bytes,
        fixture,
        mut safety,
        config,
        tokenizer: _,
        prompt_token_ids,
        checkpoint,
        verification_sha256,
    } = open_endpoint_authority(
        checkpoint_root,
        model_lock_path,
        verification_path,
        fixture_path,
    )?;
    if fixture.schema_version != 2 || prompt_token_ids != CHAT_PROMPT_IDS {
        return Err("layer-1 routing trace requires the frozen chat fixture".to_owned());
    }
    fs::create_dir(output_dir).map_err(|error| format!("{}: {error}", output_dir.display()))?;
    let rows = prompt_token_ids.len();
    let mut ledger = EndpointLedger::default();
    let mut layer0_captures = Layer0Captures::default();
    let incoming = execute_dense_layer0(
        &checkpoint,
        &config,
        &prompt_token_ids,
        &mut ledger,
        &mut layer0_captures,
    )?;
    checkpoint.release_file_pages()?;
    safety.checkpoint("layer_0_complete", true)?;
    let incoming_bytes = incoming
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect::<Vec<_>>();
    let input_norm_weight = bf16_vector(
        &checkpoint,
        "model.layers.1.input_layernorm.weight",
        HIDDEN,
        &mut ledger,
    )?;
    let normalized = rms_norm(
        &incoming,
        rows,
        &input_norm_weight,
        config.layernorm_epsilon,
    )?;
    let mut captures_internal = Layer0Captures::default();
    let mut cache = LayerKvCache::default();
    let attention_output = attention(
        &checkpoint,
        &config,
        1,
        &normalized,
        rows,
        &mut cache,
        &mut ledger,
        Some(&mut captures_internal),
    )?;
    safety.checkpoint("layer_1_attention_complete", true)?;
    let post_attention = incoming
        .iter()
        .zip(&attention_output)
        .map(|(&residual, &projected)| round_bf16(residual + projected))
        .collect::<Vec<_>>();
    let post_norm_weight = bf16_vector(
        &checkpoint,
        "model.layers.1.post_attention_layernorm.weight",
        HIDDEN,
        &mut ledger,
    )?;
    let post_attention_norm = rms_norm(
        &post_attention,
        rows,
        &post_norm_weight,
        config.layernorm_epsilon,
    )?;
    let routing = route_mlp(&checkpoint, 1, &post_attention_norm, rows, &mut ledger)?;
    checkpoint.release_file_pages()?;
    safety.checkpoint("layer_1_routing_complete", true)?;

    let attention_states = HEADS * rows * (rows + 3) / 2;
    let mut captures = BTreeMap::new();
    for (name, shape, values) in [
        ("incoming", vec![rows, HIDDEN], incoming.as_slice()),
        ("input_norm", vec![rows, HIDDEN], normalized.as_slice()),
        ("qkv", vec![rows, 14_848], captures_internal.qkv.as_slice()),
        (
            "query",
            vec![rows, HEADS, QK_HEAD_DIM],
            captures_internal.query.as_slice(),
        ),
        (
            "key",
            vec![rows, 8, QK_HEAD_DIM],
            captures_internal.key.as_slice(),
        ),
        (
            "value",
            vec![rows, 8, V_HEAD_DIM],
            captures_internal.value.as_slice(),
        ),
        ("sinks", vec![HEADS], captures_internal.sinks.as_slice()),
        (
            "attention_scores",
            vec![attention_states],
            captures_internal.attention_scores.as_slice(),
        ),
        (
            "attention_probabilities",
            vec![attention_states],
            captures_internal.attention_probabilities.as_slice(),
        ),
        (
            "attention",
            vec![rows, HEADS, V_HEAD_DIM],
            captures_internal.attention.as_slice(),
        ),
        (
            "attention_projection",
            vec![rows, HIDDEN],
            captures_internal.attention_projection.as_slice(),
        ),
        (
            "post_attention",
            vec![rows, HIDDEN],
            post_attention.as_slice(),
        ),
        (
            "post_attention_norm",
            vec![rows, HIDDEN],
            post_attention_norm.as_slice(),
        ),
    ] {
        captures.insert(
            name.to_owned(),
            write_capture(output_dir, name, &shape, values)?,
        );
    }
    captures.insert(
        "router_logits".to_owned(),
        write_capture_typed(
            output_dir,
            "router_logits",
            &[rows, ROUTED_EXPERTS],
            &routing.logits,
            "F32",
        )?,
    );
    captures.insert(
        "router_scores".to_owned(),
        write_capture_typed(
            output_dir,
            "router_scores",
            &[rows, ROUTED_EXPERTS],
            &routing.scores,
            "F32",
        )?,
    );
    safety.checkpoint("captures_written", true)?;
    ledger.actual_process_disk_bytes_read = process_disk_bytes_read()?
        .checked_sub(disk_bytes_read_before)
        .ok_or("process disk byte counter moved backwards")?;
    ledger.peak_resident_bytes = peak_resident_bytes()?;
    let report = Layer1RoutingTraceReport {
        schema_version: 1,
        semantic: "mimo_real_layer1_attention_to_routing_rust_trace",
        revision: REVISION,
        commit: commit.to_owned(),
        fixture_sha256: sha256_hex(&fixture_bytes),
        checkpoint_verification_sha256: verification_sha256,
        prompt_token_ids,
        source_input_sha256: sha256_hex(&incoming_bytes),
        numerics: "dynamic_fp8_e4m3fn_per_token_group_128_bf16_boundaries",
        captures,
        selected_experts_by_position: routing.selected,
        route_weights_by_position: routing.weights,
        ledger,
        safety_snapshots: safety.snapshots,
        complete_wall_ms: complete_started.elapsed().as_secs_f64() * 1000.0,
        batch_size: 1,
        prompt_positions: rows,
        concurrency: 1,
        accepted_tokens: 0,
        performance_claim: None,
    };
    let bytes = serde_json::to_vec_pretty(&report).map_err(|error| error.to_string())?;
    write_create_new(&output_dir.join("manifest.json"), &bytes)?;
    Ok(report)
}

#[allow(clippy::too_many_arguments)]
pub fn run_real_layer1_expert_trace(
    checkpoint_root: &Path,
    model_lock_path: &Path,
    verification_path: &Path,
    fixture_path: &Path,
    output_dir: &Path,
    commit: &str,
) -> Result<Layer1ExpertTraceReport, String> {
    if output_dir.exists() {
        return Err(format!("refusing to overwrite {}", output_dir.display()));
    }
    let complete_started = Instant::now();
    let disk_bytes_read_before = process_disk_bytes_read()?;
    let EndpointAuthority {
        fixture_bytes,
        fixture,
        mut safety,
        config,
        tokenizer: _,
        prompt_token_ids,
        checkpoint,
        verification_sha256,
    } = open_endpoint_authority(
        checkpoint_root,
        model_lock_path,
        verification_path,
        fixture_path,
    )?;
    if fixture.schema_version != 2 || prompt_token_ids != CHAT_PROMPT_IDS {
        return Err("layer-1 expert trace requires the frozen chat fixture".to_owned());
    }
    fs::create_dir(output_dir).map_err(|error| format!("{}: {error}", output_dir.display()))?;
    let rows = prompt_token_ids.len();
    let mut ledger = EndpointLedger::default();
    let mut layer0_captures = Layer0Captures::default();
    let incoming = execute_dense_layer0(
        &checkpoint,
        &config,
        &prompt_token_ids,
        &mut ledger,
        &mut layer0_captures,
    )?;
    checkpoint.release_file_pages()?;
    safety.checkpoint("layer_0_complete", true)?;
    let incoming_bytes = incoming
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect::<Vec<_>>();
    let input_norm_weight = bf16_vector(
        &checkpoint,
        "model.layers.1.input_layernorm.weight",
        HIDDEN,
        &mut ledger,
    )?;
    let normalized = rms_norm(
        &incoming,
        rows,
        &input_norm_weight,
        config.layernorm_epsilon,
    )?;
    let mut cache = LayerKvCache::default();
    let attention_output = attention(
        &checkpoint,
        &config,
        1,
        &normalized,
        rows,
        &mut cache,
        &mut ledger,
        None,
    )?;
    safety.checkpoint("layer_1_attention_complete", true)?;
    let post_attention = incoming
        .iter()
        .zip(&attention_output)
        .map(|(&residual, &projected)| round_bf16(residual + projected))
        .collect::<Vec<_>>();
    let post_norm_weight = bf16_vector(
        &checkpoint,
        "model.layers.1.post_attention_layernorm.weight",
        HIDDEN,
        &mut ledger,
    )?;
    let post_attention_norm = rms_norm(
        &post_attention,
        rows,
        &post_norm_weight,
        config.layernorm_epsilon,
    )?;
    safety.checkpoint("layer_1_routing_input_complete", true)?;
    let mut expert_captures = ExpertCaptures::default();
    let mut expert_completed = |expert| {
        checkpoint.release_file_pages()?;
        safety.checkpoint(&format!("layer_1_expert_{expert}_complete"), true)
    };
    let routed = routed_mlp_traced(
        &checkpoint,
        1,
        &post_attention_norm,
        rows,
        &mut ledger,
        Some(&mut expert_captures),
        &mut expert_completed,
    )?;
    let final_hidden = post_attention
        .iter()
        .zip(&routed.output)
        .map(|(&residual, &projected)| round_bf16(residual + projected))
        .collect::<Vec<_>>();
    checkpoint.release_file_pages()?;
    safety.checkpoint("layer_1_complete", true)?;

    let placements = rows * TOP_K;
    if expert_captures
        .schedule
        .iter()
        .map(|entry| entry.positions.len())
        .sum::<usize>()
        != placements
        || expert_captures.gate.len() != placements * MOE_INTERMEDIATE
        || expert_captures.up.len() != placements * MOE_INTERMEDIATE
        || expert_captures.swiglu.len() != placements * MOE_INTERMEDIATE
        || expert_captures.down.len() != placements * HIDDEN
    {
        return Err("layer-1 expert capture shape mismatch".to_owned());
    }
    let mut captures = BTreeMap::new();
    for (name, shape, values) in [
        (
            "moe_input",
            vec![rows, HIDDEN],
            post_attention_norm.as_slice(),
        ),
        (
            "expert_gate",
            vec![placements, MOE_INTERMEDIATE],
            expert_captures.gate.as_slice(),
        ),
        (
            "expert_up",
            vec![placements, MOE_INTERMEDIATE],
            expert_captures.up.as_slice(),
        ),
        (
            "expert_swiglu",
            vec![placements, MOE_INTERMEDIATE],
            expert_captures.swiglu.as_slice(),
        ),
        (
            "expert_down",
            vec![placements, HIDDEN],
            expert_captures.down.as_slice(),
        ),
        (
            "routed_output",
            vec![rows, HIDDEN],
            routed.output.as_slice(),
        ),
        ("final", vec![rows, HIDDEN], final_hidden.as_slice()),
    ] {
        captures.insert(
            name.to_owned(),
            write_capture(output_dir, name, &shape, values)?,
        );
    }
    safety.checkpoint("captures_written", true)?;
    ledger.actual_process_disk_bytes_read = process_disk_bytes_read()?
        .checked_sub(disk_bytes_read_before)
        .ok_or("process disk byte counter moved backwards")?;
    ledger.peak_resident_bytes = peak_resident_bytes()?;
    let report = Layer1ExpertTraceReport {
        schema_version: 1,
        semantic: "mimo_real_layer1_selected_experts_rust_trace",
        revision: REVISION,
        commit: commit.to_owned(),
        fixture_sha256: sha256_hex(&fixture_bytes),
        checkpoint_verification_sha256: verification_sha256,
        prompt_token_ids,
        source_input_sha256: sha256_hex(&incoming_bytes),
        numerics: "dynamic_fp8_e4m3fn_per_token_group_128_bf16_boundaries",
        captures,
        selected_experts_by_position: routed.selected,
        route_weights_by_position: routed.weights,
        expert_schedule: expert_captures.schedule,
        ledger,
        safety_snapshots: safety.snapshots,
        complete_wall_ms: complete_started.elapsed().as_secs_f64() * 1000.0,
        batch_size: 1,
        prompt_positions: rows,
        concurrency: 1,
        accepted_tokens: 0,
        performance_claim: None,
    };
    let bytes = serde_json::to_vec_pretty(&report).map_err(|error| error.to_string())?;
    write_create_new(&output_dir.join("manifest.json"), &bytes)?;
    Ok(report)
}

#[allow(clippy::too_many_arguments)]
pub fn run_real_layer2_trace(
    checkpoint_root: &Path,
    model_lock_path: &Path,
    verification_path: &Path,
    fixture_path: &Path,
    output_dir: &Path,
    commit: &str,
) -> Result<Layer1ExpertTraceReport, String> {
    run_real_routed_layer_trace(
        checkpoint_root,
        model_lock_path,
        verification_path,
        fixture_path,
        output_dir,
        commit,
        2,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn run_real_layer4_trace(
    checkpoint_root: &Path,
    model_lock_path: &Path,
    verification_path: &Path,
    fixture_path: &Path,
    output_dir: &Path,
    commit: &str,
) -> Result<Layer1ExpertTraceReport, String> {
    run_real_routed_layer_trace(
        checkpoint_root,
        model_lock_path,
        verification_path,
        fixture_path,
        output_dir,
        commit,
        4,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn run_real_layer7_trace(
    checkpoint_root: &Path,
    model_lock_path: &Path,
    verification_path: &Path,
    fixture_path: &Path,
    output_dir: &Path,
    commit: &str,
) -> Result<Layer1ExpertTraceReport, String> {
    run_real_routed_layer_trace(
        checkpoint_root,
        model_lock_path,
        verification_path,
        fixture_path,
        output_dir,
        commit,
        7,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn run_real_routed_layer_trace(
    checkpoint_root: &Path,
    model_lock_path: &Path,
    verification_path: &Path,
    fixture_path: &Path,
    output_dir: &Path,
    commit: &str,
    target_layer: usize,
) -> Result<Layer1ExpertTraceReport, String> {
    let semantic = match target_layer {
        2 => "mimo_real_layer2_complete_rust_trace",
        4 => "mimo_real_layer4_complete_rust_trace",
        7 => "mimo_real_layer7_complete_rust_trace",
        11 => "mimo_real_layer11_complete_rust_trace",
        13 => "mimo_real_layer13_complete_rust_trace",
        14 => "mimo_real_layer14_complete_rust_trace",
        19 => "mimo_real_layer19_complete_rust_trace",
        29 => "mimo_real_layer29_complete_rust_trace",
        34 => "mimo_real_layer34_complete_rust_trace",
        _ => return Err(format!("unsupported routed trace layer {target_layer}")),
    };
    if output_dir.exists() {
        return Err(format!("refusing to overwrite {}", output_dir.display()));
    }
    let complete_started = Instant::now();
    let disk_bytes_read_before = process_disk_bytes_read()?;
    let EndpointAuthority {
        fixture_bytes,
        fixture,
        mut safety,
        config,
        tokenizer: _,
        prompt_token_ids,
        checkpoint,
        verification_sha256,
    } = open_endpoint_authority(
        checkpoint_root,
        model_lock_path,
        verification_path,
        fixture_path,
    )?;
    if fixture.schema_version != 2 || prompt_token_ids != CHAT_PROMPT_IDS {
        return Err("real routed-layer trace requires the frozen chat fixture".to_owned());
    }
    fs::create_dir(output_dir).map_err(|error| format!("{}: {error}", output_dir.display()))?;
    let rows = prompt_token_ids.len();
    let mut ledger = EndpointLedger::default();
    let mut layer0_captures = Layer0Captures::default();
    let mut hidden = execute_dense_layer0(
        &checkpoint,
        &config,
        &prompt_token_ids,
        &mut ledger,
        &mut layer0_captures,
    )?;
    safety.checkpoint("layer_0_complete", true)?;
    for layer in 1..target_layer {
        let prefix = format!("model.layers.{layer}");
        let input_norm = bf16_vector(
            &checkpoint,
            &format!("{prefix}.input_layernorm.weight"),
            HIDDEN,
            &mut ledger,
        )?;
        let normalized = rms_norm(&hidden, rows, &input_norm, config.layernorm_epsilon)?;
        let mut cache = LayerKvCache::default();
        let attention_output = attention(
            &checkpoint,
            &config,
            layer,
            &normalized,
            rows,
            &mut cache,
            &mut ledger,
            None,
        )?;
        let post_attention = hidden
            .iter()
            .zip(attention_output)
            .map(|(&residual, projected)| round_bf16(residual + projected))
            .collect::<Vec<_>>();
        let post_norm = bf16_vector(
            &checkpoint,
            &format!("{prefix}.post_attention_layernorm.weight"),
            HIDDEN,
            &mut ledger,
        )?;
        let moe_input = rms_norm(&post_attention, rows, &post_norm, config.layernorm_epsilon)?;
        let mlp = routed_mlp(&checkpoint, layer, &moe_input, rows, &mut ledger)?;
        hidden = post_attention
            .iter()
            .zip(mlp.output)
            .map(|(&residual, projected)| round_bf16(residual + projected))
            .collect();
        checkpoint.release_file_pages()?;
        safety.checkpoint(&format!("layer_{layer}_complete"), true)?;
    }
    let incoming_bytes = hidden
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect::<Vec<_>>();

    let target_prefix = format!("model.layers.{target_layer}");
    let input_norm = bf16_vector(
        &checkpoint,
        &format!("{target_prefix}.input_layernorm.weight"),
        HIDDEN,
        &mut ledger,
    )?;
    let normalized = rms_norm(&hidden, rows, &input_norm, config.layernorm_epsilon)?;
    let mut attention_captures = Layer0Captures::default();
    let mut cache = LayerKvCache::default();
    let attention_output = attention(
        &checkpoint,
        &config,
        target_layer,
        &normalized,
        rows,
        &mut cache,
        &mut ledger,
        Some(&mut attention_captures),
    )?;
    safety.checkpoint(&format!("layer_{target_layer}_attention_complete"), true)?;
    let post_attention = hidden
        .iter()
        .zip(&attention_output)
        .map(|(&residual, &projected)| round_bf16(residual + projected))
        .collect::<Vec<_>>();
    let post_norm = bf16_vector(
        &checkpoint,
        &format!("{target_prefix}.post_attention_layernorm.weight"),
        HIDDEN,
        &mut ledger,
    )?;
    let moe_input = rms_norm(&post_attention, rows, &post_norm, config.layernorm_epsilon)?;
    let mut expert_captures = ExpertCaptures::default();
    let mut expert_completed = |expert| {
        checkpoint.release_file_pages()?;
        safety.checkpoint(
            &format!("layer_{target_layer}_expert_{expert}_complete"),
            true,
        )
    };
    let routed = routed_mlp_traced(
        &checkpoint,
        target_layer,
        &moe_input,
        rows,
        &mut ledger,
        Some(&mut expert_captures),
        &mut expert_completed,
    )?;
    let final_hidden = post_attention
        .iter()
        .zip(&routed.output)
        .map(|(&residual, &projected)| round_bf16(residual + projected))
        .collect::<Vec<_>>();
    checkpoint.release_file_pages()?;
    safety.checkpoint(&format!("layer_{target_layer}_complete"), true)?;
    let placements = rows * TOP_K;
    let target_is_swa = config.hybrid_layer_pattern[target_layer] == 1;
    let target_kv_heads = if target_is_swa { 8 } else { 4 };
    let target_qkv_rows =
        HEADS * QK_HEAD_DIM + target_kv_heads * QK_HEAD_DIM + target_kv_heads * V_HEAD_DIM;
    let attention_states = if target_is_swa {
        HEADS * rows * (rows + 3) / 2
    } else {
        HEADS * rows * (rows + 1) / 2
    };
    let mut captures = BTreeMap::new();
    for (name, shape, values) in [
        ("incoming", vec![rows, HIDDEN], hidden.as_slice()),
        ("input_norm", vec![rows, HIDDEN], normalized.as_slice()),
        (
            "qkv",
            vec![rows, target_qkv_rows],
            attention_captures.qkv.as_slice(),
        ),
        (
            "query",
            vec![rows, HEADS, QK_HEAD_DIM],
            attention_captures.query.as_slice(),
        ),
        (
            "key",
            vec![rows, target_kv_heads, QK_HEAD_DIM],
            attention_captures.key.as_slice(),
        ),
        (
            "value",
            vec![rows, target_kv_heads, V_HEAD_DIM],
            attention_captures.value.as_slice(),
        ),
        (
            "sinks",
            vec![if target_is_swa { HEADS } else { 0 }],
            attention_captures.sinks.as_slice(),
        ),
        (
            "attention_scores",
            vec![attention_states],
            attention_captures.attention_scores.as_slice(),
        ),
        (
            "attention_probabilities",
            vec![attention_states],
            attention_captures.attention_probabilities.as_slice(),
        ),
        (
            "attention",
            vec![rows, HEADS, V_HEAD_DIM],
            attention_captures.attention.as_slice(),
        ),
        (
            "attention_projection",
            vec![rows, HIDDEN],
            attention_captures.attention_projection.as_slice(),
        ),
        (
            "post_attention",
            vec![rows, HIDDEN],
            post_attention.as_slice(),
        ),
        ("moe_input", vec![rows, HIDDEN], moe_input.as_slice()),
        (
            "expert_gate",
            vec![placements, MOE_INTERMEDIATE],
            expert_captures.gate.as_slice(),
        ),
        (
            "expert_up",
            vec![placements, MOE_INTERMEDIATE],
            expert_captures.up.as_slice(),
        ),
        (
            "expert_swiglu",
            vec![placements, MOE_INTERMEDIATE],
            expert_captures.swiglu.as_slice(),
        ),
        (
            "expert_down",
            vec![placements, HIDDEN],
            expert_captures.down.as_slice(),
        ),
        (
            "routed_output",
            vec![rows, HIDDEN],
            routed.output.as_slice(),
        ),
        ("final", vec![rows, HIDDEN], final_hidden.as_slice()),
    ] {
        captures.insert(
            name.to_owned(),
            write_capture(output_dir, name, &shape, values)?,
        );
    }
    captures.insert(
        "router_logits".to_owned(),
        write_capture_typed(
            output_dir,
            "router_logits",
            &[rows, ROUTED_EXPERTS],
            &routed.logits,
            "F32",
        )?,
    );
    captures.insert(
        "router_scores".to_owned(),
        write_capture_typed(
            output_dir,
            "router_scores",
            &[rows, ROUTED_EXPERTS],
            &routed.scores,
            "F32",
        )?,
    );
    safety.checkpoint("captures_written", true)?;
    ledger.actual_process_disk_bytes_read =
        process_disk_bytes_read()?.saturating_sub(disk_bytes_read_before);
    ledger.peak_resident_bytes = peak_resident_bytes()?;
    let report = Layer1ExpertTraceReport {
        schema_version: 1,
        semantic,
        revision: REVISION,
        commit: commit.to_owned(),
        fixture_sha256: sha256_hex(&fixture_bytes),
        checkpoint_verification_sha256: verification_sha256,
        prompt_token_ids,
        source_input_sha256: sha256_hex(&incoming_bytes),
        numerics: "dynamic_fp8_e4m3fn_per_token_group_128_bf16_boundaries",
        captures,
        selected_experts_by_position: routed.selected,
        route_weights_by_position: routed.weights,
        expert_schedule: expert_captures.schedule,
        ledger,
        safety_snapshots: safety.snapshots,
        complete_wall_ms: complete_started.elapsed().as_secs_f64() * 1000.0,
        batch_size: 1,
        prompt_positions: rows,
        concurrency: 1,
        accepted_tokens: 0,
        performance_claim: None,
    };
    let bytes = serde_json::to_vec_pretty(&report).map_err(|error| error.to_string())?;
    write_create_new(&output_dir.join("manifest.json"), &bytes)?;
    Ok(report)
}

#[allow(clippy::too_many_arguments)]
pub fn run_full_prefix_trace(
    checkpoint_root: &Path,
    model_lock_path: &Path,
    verification_path: &Path,
    fixture_path: &Path,
    output_dir: &Path,
    commit: &str,
) -> Result<FullPrefixTraceReport, String> {
    if output_dir.exists() {
        return Err(format!("refusing to overwrite {}", output_dir.display()));
    }
    let complete_started = Instant::now();
    let disk_bytes_read_before = process_disk_bytes_read()?;
    let EndpointAuthority {
        fixture_bytes,
        fixture,
        mut safety,
        config,
        tokenizer: _,
        prompt_token_ids,
        checkpoint,
        verification_sha256,
    } = open_endpoint_authority(
        checkpoint_root,
        model_lock_path,
        verification_path,
        fixture_path,
    )?;
    let prompt_token_ids = full_prefix_trace_tokens(&fixture, &prompt_token_ids)?;
    fs::create_dir(output_dir).map_err(|error| format!("{}: {error}", output_dir.display()))?;
    let rows = prompt_token_ids.len();
    let mut caches = (0..48).map(|_| LayerKvCache::default()).collect::<Vec<_>>();
    let mut ledger = EndpointLedger::default();
    let mut internal = FullPrefixCaptures::default();
    let step = decode_step(
        &checkpoint,
        &config,
        &prompt_token_ids,
        &mut caches,
        &mut ledger,
        &mut safety,
        Some(&mut internal),
        None,
    )?;
    if internal.embedding.len() != rows * HIDDEN
        || internal.layer_finals.len() != 48
        || internal
            .layer_finals
            .iter()
            .any(|values| values.len() != rows * HIDDEN)
        || internal.final_norm.len() != rows * HIDDEN
        || step.full_logits.len() != config.vocab_size
    {
        return Err("full-prefix capture shape mismatch".to_owned());
    }
    let mut captures = BTreeMap::new();
    captures.insert(
        "embedding".to_owned(),
        write_capture(
            output_dir,
            "embedding",
            &[rows, HIDDEN],
            &internal.embedding,
        )?,
    );
    for (layer, values) in internal.layer_finals.iter().enumerate() {
        let name = format!("layer_{layer:02}_final");
        captures.insert(
            name.clone(),
            write_capture(output_dir, &name, &[rows, HIDDEN], values)?,
        );
    }
    captures.insert(
        "final_norm".to_owned(),
        write_capture(
            output_dir,
            "final_norm",
            &[rows, HIDDEN],
            &internal.final_norm,
        )?,
    );
    captures.insert(
        "last_logits".to_owned(),
        write_capture_typed(
            output_dir,
            "last_logits",
            &[config.vocab_size],
            &step.full_logits,
            "F32",
        )?,
    );
    safety.checkpoint("captures_written", true)?;
    ledger.actual_process_disk_bytes_read = process_disk_bytes_read()?
        .checked_sub(disk_bytes_read_before)
        .ok_or("process disk byte counter moved backwards")?;
    ledger.peak_resident_bytes = peak_resident_bytes()?;
    let report = FullPrefixTraceReport {
        schema_version: 1,
        semantic: "mimo_full_prefix_layer_final_rust_trace",
        revision: REVISION,
        commit: commit.to_owned(),
        fixture_sha256: sha256_hex(&fixture_bytes),
        checkpoint_verification_sha256: verification_sha256,
        prompt_token_ids,
        numerics: "dynamic_fp8_e4m3fn_per_token_group_128_bf16_boundaries",
        captures,
        layer_traces: step.traces,
        ledger,
        safety_snapshots: safety.snapshots,
        complete_wall_ms: complete_started.elapsed().as_secs_f64() * 1000.0,
        batch_size: 1,
        prompt_positions: rows,
        concurrency: 1,
        accepted_tokens: 0,
        performance_claim: None,
    };
    let bytes = serde_json::to_vec_pretty(&report).map_err(|error| error.to_string())?;
    write_create_new(&output_dir.join("manifest.json"), &bytes)?;
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn retained_cache_rejects_corruption() {
        let valid = LayerKvCache {
            keys: vec![0.0; 4 * QK_HEAD_DIM],
            values: vec![0.0; 4 * V_HEAD_DIM],
            positions: 1,
            kv_heads: 4,
        };
        assert!(valid.validate().is_ok());
        let mut corrupted = valid;
        corrupted.keys.pop();
        assert!(corrupted.validate().is_err());
    }

    #[test]
    fn unsafe_checkpoint_shard_paths_fail_closed() {
        assert!(validate_relative_file("model.safetensors").is_ok());
        assert!(validate_relative_file("../model.safetensors").is_err());
        assert!(validate_relative_file("nested/model.safetensors").is_err());
    }

    #[test]
    fn darwin_monitor_parsers_observe_live_safe_values() {
        assert!(system_memory_free_percent().is_ok_and(|value| value <= 100));
        assert!(swap_used_bytes().is_ok());
        assert!(throttled_pages().is_ok());
        assert!(process_usage().is_ok());
        assert!(peak_resident_bytes().is_ok_and(|value| value > 0));
    }

    #[test]
    fn full_qkv_scale_layout_preserves_head_boundaries() {
        assert_eq!(full_qkv_scale_row(0), Ok(0));
        assert_eq!(full_qkv_scale_row(12_287), Ok(95));
        assert_eq!(full_qkv_scale_row(12_288), Ok(96));
        assert_eq!(full_qkv_scale_row(12_415), Ok(96));
        assert_eq!(full_qkv_scale_row(12_416), Ok(97));
        assert_eq!(full_qkv_scale_row(12_479), Ok(97));
        assert_eq!(full_qkv_scale_row(12_480), Ok(98));
        assert_eq!(full_qkv_scale_row(13_055), Ok(103));
        assert_eq!(full_qkv_scale_row(13_056), Ok(104));
        assert_eq!(full_qkv_scale_row(13_567), Ok(107));
        assert!(full_qkv_scale_row(13_568).is_err());
        let used = (0..13_568)
            .map(|row| full_qkv_scale_row(row).expect("valid full-QKV row"))
            .collect::<BTreeSet<_>>();
        assert_eq!(used, (0..108).collect());
    }

    #[test]
    fn expert_union_factor_normalizes_by_positions() {
        assert_eq!(expert_union_factor(56, 8), Ok(7.0));
        assert_eq!(expert_union_factor(8, 1), Ok(8.0));
        assert!(expert_union_factor(0, 0).is_err());
    }

    #[test]
    fn frozen_chat_prefill_fixture_is_exact() {
        let fixture: EndpointFixture = serde_json::from_str(include_str!(
            "../evals/fixtures/real/pw0052-chat-endpoint.json"
        ))
        .expect("valid frozen chat fixture");
        assert!(validate_fixture(&fixture).is_ok());
        assert_eq!(fixture.prompt_utf8, CHAT_PROMPT);
        assert_eq!(fixture.expected_prompt_token_ids, CHAT_PROMPT_IDS);
    }

    #[test]
    fn whole_sequence_trace_fixture_appends_one_frozen_token() {
        let fixture: EndpointFixture = serde_json::from_str(include_str!(
            "../evals/fixtures/real/pw0094-rust-whole-sequence.json"
        ))
        .expect("valid whole-sequence fixture");
        assert!(validate_fixture(&fixture).is_ok());
        assert_eq!(fixture.expected_prompt_token_ids, CHAT_PROMPT_IDS);
        assert_eq!(
            fixture.full_prefix_trace_append_token_ids.as_deref(),
            Some(&[264][..])
        );
        assert_eq!(
            full_prefix_trace_tokens(&fixture, &CHAT_PROMPT_IDS).expect("trace tokens"),
            [CHAT_PROMPT_IDS.as_slice(), &[264]].concat()
        );
        assert!(validate_slow_endpoint_fixture(&fixture).is_err());
    }

    #[test]
    fn causal_prefill_rows_match_tiny_fixture() {
        let fixture: Value = serde_json::from_str(include_str!(
            "../evals/fixtures/tiny/pw0052-causal-prefill.json"
        ))
        .expect("valid causal prefill fixture");
        assert_eq!(fixture["schema_version"], 1);
        assert_eq!(fixture["semantic"], "causal_prefill_attention_rows");
        for case in fixture["cases"].as_array().expect("cases") {
            let query = serde_json::from_value::<Vec<f32>>(case["query"].clone()).expect("query");
            let keys = serde_json::from_value::<Vec<Vec<f32>>>(case["keys"].clone()).expect("keys");
            let values =
                serde_json::from_value::<Vec<Vec<f32>>>(case["values"].clone()).expect("values");
            let key_views = keys.iter().map(Vec::as_slice).collect::<Vec<_>>();
            let value_views = values.iter().map(Vec::as_slice).collect::<Vec<_>>();
            let sink = case["sink"].as_f64().map(|value| value as f32);
            let expected =
                serde_json::from_value::<Vec<f32>>(case["expected"].clone()).expect("expected");
            let actual = causal_attention_head(
                &query,
                &key_views,
                &value_views,
                case["scale"].as_f64().expect("scale") as f32,
                sink,
            )
            .expect("valid causal attention");
            assert_eq!(actual.len(), expected.len());
            for (actual, expected) in actual.iter().zip(expected) {
                assert!((actual - expected).abs() <= 1.0e-6);
            }
        }
    }

    #[test]
    fn dynamic_fp8_activations_match_pytorch_bytes() {
        fn flatten_i64(value: &Value, output: &mut Vec<i64>) {
            if let Some(values) = value.as_array() {
                for value in values {
                    flatten_i64(value, output);
                }
            } else {
                output.push(value.as_i64().expect("integer fixture value"));
            }
        }

        let fixture: Value = serde_json::from_str(include_str!(
            "../evals/fixtures/tiny/pw0053-dynamic-fp8-activation.json"
        ))
        .expect("valid dynamic activation fixture");
        assert_eq!(
            fixture["semantic"],
            "dynamic_fp8_e4m3fn_per_token_group_128"
        );
        let mut input_bits = Vec::new();
        flatten_i64(&fixture["input_f32_bits"], &mut input_bits);
        let input = input_bits
            .into_iter()
            .map(|bits| f32::from_bits((bits as i32) as u32))
            .collect::<Vec<_>>();
        let actual = dynamic_fp8_activations(&input, 2, 256).expect("valid activations");

        let mut scale_bits = Vec::new();
        flatten_i64(&fixture["scale_f32_bits"], &mut scale_bits);
        assert_eq!(
            actual
                .scales
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            scale_bits
                .into_iter()
                .map(|bits| (bits as i32) as u32)
                .collect::<Vec<_>>()
        );
        let mut encoded = Vec::new();
        flatten_i64(&fixture["encoded_u8"], &mut encoded);
        assert_eq!(
            actual.encoded,
            encoded
                .into_iter()
                .map(|value| value as u8)
                .collect::<Vec<_>>()
        );
        let mut dequantized_bits = Vec::new();
        flatten_i64(&fixture["dequantized_f32_bits"], &mut dequantized_bits);
        assert_eq!(
            actual
                .dequantized
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            dequantized_bits
                .into_iter()
                .map(|bits| (bits as i32) as u32)
                .collect::<Vec<_>>()
        );

        assert!(dynamic_fp8_activations(&[], 0, 128).is_err());
        assert!(dynamic_fp8_activations(&[0.0; 127], 1, 127).is_err());
        let mut nonfinite = [0.0_f32; 128];
        nonfinite[3] = f32::NAN;
        assert!(dynamic_fp8_activations(&nonfinite, 1, 128).is_err());
    }

    #[test]
    fn bf16_rounding_matches_pytorch_payloads() {
        let fixture: Value = serde_json::from_str(include_str!(
            "../evals/fixtures/tiny/pw0054-bf16-boundary.json"
        ))
        .expect("valid BF16 boundary fixture");
        assert_eq!(fixture["semantic"], "f32_to_bf16_rne");
        let input = fixture["input_f32_bits"].as_array().expect("input bits");
        let expected_bf16 = fixture["bf16_u16"].as_array().expect("BF16 bits");
        let expected_widened = fixture["widened_f32_bits"]
            .as_array()
            .expect("widened bits");
        assert_eq!(input.len(), expected_bf16.len());
        assert_eq!(input.len(), expected_widened.len());
        for ((input, expected_bf16), expected_widened) in
            input.iter().zip(expected_bf16).zip(expected_widened)
        {
            let input_bits = (input.as_i64().expect("signed F32 bits") as i32) as u32;
            let actual = round_bf16(f32::from_bits(input_bits));
            let expected_bf16 = expected_bf16.as_u64().expect("BF16 payload") as u16;
            let expected_widened =
                (expected_widened.as_i64().expect("signed widened bits") as i32) as u32;
            if f32::from_bits(input_bits).is_finite() {
                assert_eq!((actual.to_bits() >> 16) as u16, expected_bf16);
                assert_eq!(actual.to_bits(), expected_widened);
            } else if f32::from_bits(input_bits).is_nan() {
                assert!(actual.is_nan());
                assert_eq!(
                    actual.is_sign_negative(),
                    f32::from_bits(input_bits).is_sign_negative()
                );
            } else {
                assert_eq!(actual.to_bits(), input_bits);
            }
        }

        for case in fixture["attention_cases"]
            .as_array()
            .expect("attention cases")
        {
            let query =
                serde_json::from_value::<Vec<f32>>(case["query"].clone()).expect("BF16 query");
            let keys =
                serde_json::from_value::<Vec<Vec<f32>>>(case["keys"].clone()).expect("BF16 keys");
            let values = serde_json::from_value::<Vec<Vec<f32>>>(case["values"].clone())
                .expect("BF16 values");
            let key_views = keys.iter().map(Vec::as_slice).collect::<Vec<_>>();
            let value_views = values.iter().map(Vec::as_slice).collect::<Vec<_>>();
            let sink = case["sink"].as_f64().map(|value| value as f32);
            let actual = causal_attention_head_bf16(
                &query,
                &key_views,
                &value_views,
                case["scale"].as_f64().expect("scale") as f32,
                sink,
            )
            .expect("valid BF16 attention");
            let expected = case["expected_bf16_u16"]
                .as_array()
                .expect("expected BF16 output");
            assert_eq!(actual.len(), expected.len());
            for (actual, expected) in actual.iter().zip(expected) {
                assert_eq!(
                    (actual.to_bits() >> 16) as u16,
                    expected.as_u64().expect("BF16 payload") as u16
                );
            }
        }
    }

    #[test]
    fn bf16_text_rope_matches_pytorch_operation_staging() {
        fn flatten_u16(value: &Value, output: &mut Vec<u16>) {
            if let Some(values) = value.as_array() {
                for value in values {
                    flatten_u16(value, output);
                }
            } else {
                output.push(value.as_u64().expect("BF16 payload") as u16);
            }
        }

        let fixture: Value =
            serde_json::from_str(include_str!("../evals/fixtures/tiny/pw0055-bf16-rope.json"))
                .expect("valid BF16 RoPE fixture");
        assert_eq!(fixture["semantic"], "mimo_text_rope_bf16_operation_staging");
        for case in fixture["cases"].as_array().expect("RoPE cases") {
            assert_eq!(case["heads"], 2);
            assert_eq!(case["head_dim"], QK_HEAD_DIM);
            assert_eq!(case["rope_dim"], ROPE_DIM);
            let mut input_bits = Vec::new();
            flatten_u16(&case["input_bf16_u16"], &mut input_bits);
            let mut values = input_bits
                .iter()
                .map(|bits| f32::from_bits(u32::from(*bits) << 16))
                .collect::<Vec<_>>();
            let original = values.clone();
            apply_rope(
                &mut values,
                2,
                case["position"].as_u64().expect("position") as usize,
                case["theta"].as_f64().expect("theta"),
            );
            let mut expected = Vec::new();
            flatten_u16(&case["output_bf16_u16"], &mut expected);
            assert_eq!(values.len(), expected.len());
            assert_eq!(
                values
                    .iter()
                    .map(|value| (value.to_bits() >> 16) as u16)
                    .collect::<Vec<_>>(),
                expected
            );
            for head in 0..2 {
                assert_eq!(
                    &values[head * QK_HEAD_DIM + ROPE_DIM..(head + 1) * QK_HEAD_DIM],
                    &original[head * QK_HEAD_DIM + ROPE_DIM..(head + 1) * QK_HEAD_DIM]
                );
            }
        }
    }

    #[test]
    fn pytorch_arm_softmax_matches_pytorch_payloads() {
        let fixture: Value = serde_json::from_str(include_str!(
            "../evals/fixtures/tiny/pw0057-vforce-softmax.json"
        ))
        .expect("valid PyTorch ARM softmax fixture");
        assert_eq!(fixture["semantic"], "pytorch_f32_softmax_to_bf16");
        for case in fixture["cases"].as_array().expect("softmax cases") {
            let score_bits = case["score_bf16_u16"].as_array().expect("score bits");
            let scores = score_bits
                .iter()
                .map(|value| {
                    f32::from_bits(u32::from(value.as_u64().expect("BF16 score") as u16) << 16)
                })
                .collect::<Vec<_>>();
            let actual = attention_softmax(&scores, true).expect("valid BF16 softmax");
            let expected = case["probability_bf16_u16"]
                .as_array()
                .expect("probability bits")
                .iter()
                .map(|value| value.as_u64().expect("BF16 probability") as u16)
                .collect::<Vec<_>>();
            assert_eq!(
                actual
                    .iter()
                    .map(|value| (value.to_bits() >> 16) as u16)
                    .collect::<Vec<_>>(),
                expected
            );
            if let Some(expected_f32) = case["probability_f32_u32"].as_array() {
                let maximum = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let centered = scores
                    .iter()
                    .map(|score| round_bf16(*score - maximum))
                    .collect::<Vec<_>>();
                assert_eq!(
                    pytorch_arm_softmax_f32(&centered)
                        .expect("valid F32 softmax")
                        .iter()
                        .map(|value| value.to_bits())
                        .collect::<Vec<_>>(),
                    expected_f32
                        .iter()
                        .map(|value| value.as_u64().expect("F32 probability") as u32)
                        .collect::<Vec<_>>()
                );
            }
        }
        let row: Value = serde_json::from_str(include_str!(
            "../evals/fixtures/tiny/pw0085-pytorch-arm-horizontal-softmax.json"
        ))
        .expect("valid discriminating ARM horizontal softmax fixture");
        assert_eq!(
            row["semantic"],
            "pytorch_aarch64_f32_softmax_horizontal_order"
        );
        let scores = row["score_bf16_u16"]
            .as_array()
            .expect("score bits")
            .iter()
            .map(|value| {
                f32::from_bits(u32::from(value.as_u64().expect("BF16 score") as u16) << 16)
            })
            .collect::<Vec<_>>();
        assert_eq!(
            scores
                .iter()
                .map(|value| sleef_expf_u10(*value).to_bits())
                .collect::<Vec<_>>(),
            row["exponential_f32_u32"]
                .as_array()
                .expect("exponential bits")
                .iter()
                .map(|value| value.as_u64().expect("F32 exponential") as u32)
                .collect::<Vec<_>>()
        );
        let probabilities = pytorch_arm_softmax_f32(&scores).expect("valid F32 softmax");
        assert_eq!(
            probabilities
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            row["probability_f32_u32"]
                .as_array()
                .expect("probability bits")
                .iter()
                .map(|value| value.as_u64().expect("F32 probability") as u32)
                .collect::<Vec<_>>()
        );
        assert_eq!(
            attention_softmax(&scores, true)
                .expect("valid BF16 softmax")
                .iter()
                .map(|value| (value.to_bits() >> 16) as u16)
                .collect::<Vec<_>>(),
            row["probability_bf16_u16"]
                .as_array()
                .expect("BF16 probability bits")
                .iter()
                .map(|value| value.as_u64().expect("BF16 probability") as u16)
                .collect::<Vec<_>>()
        );
        assert!(attention_softmax(&[], true).is_err());
        assert!(attention_softmax(&[f32::NAN], true).is_err());
    }

    #[test]
    fn pytorch_bf16_dot_matches_four_lane_source_order() {
        let fixture: Value = serde_json::from_str(include_str!(
            "../evals/fixtures/tiny/pw0070-pytorch-bf16-dot.json"
        ))
        .expect("valid PyTorch BF16 dot fixture");
        assert_eq!(
            fixture["semantic"],
            "pytorch_aarch64_bf16_dot_four_lane_order"
        );
        let query = fixture["query_bf16_u16"]
            .as_array()
            .expect("query bits")
            .iter()
            .map(|value| {
                f32::from_bits(u32::from(value.as_u64().expect("BF16 query") as u16) << 16)
            })
            .collect::<Vec<_>>();
        let key = fixture["key_bf16_u16"]
            .as_array()
            .expect("key bits")
            .iter()
            .map(|value| f32::from_bits(u32::from(value.as_u64().expect("BF16 key") as u16) << 16))
            .collect::<Vec<_>>();
        assert_eq!(query.len(), QK_HEAD_DIM);
        assert_eq!(key.len(), QK_HEAD_DIM);

        let source_order = pytorch_bf16_four_lane_dot_f32(&query, &key);
        assert_eq!(
            source_order.to_bits(),
            fixture["source_four_lane_dot_f32_u32"]
                .as_u64()
                .expect("source-order dot bits") as u32
        );
        let forward = query
            .iter()
            .zip(&key)
            .map(|(left, right)| left * right)
            .sum::<f32>();
        assert_eq!(
            forward.to_bits(),
            fixture["forward_dot_f32_u32"]
                .as_u64()
                .expect("forward dot bits") as u32
        );
        assert_ne!(
            round_bf16(source_order).to_bits(),
            round_bf16(forward).to_bits()
        );
        assert_eq!(
            (round_bf16(source_order).to_bits() >> 16) as u16,
            fixture["dot_bf16_u16"].as_u64().expect("BF16 dot bits") as u16
        );
        let scale = f32::from_bits(fixture["scale_f32_u32"].as_u64().expect("scale bits") as u32);
        let score = round_bf16(round_bf16(source_order) * scale);
        assert_eq!(
            (score.to_bits() >> 16) as u16,
            fixture["scaled_score_bf16_u16"]
                .as_u64()
                .expect("scaled BF16 score bits") as u16
        );
        let maximum = f32::from_bits(
            u32::from(
                fixture["row_maximum_bf16_u16"]
                    .as_u64()
                    .expect("row maximum bits") as u16,
            ) << 16,
        );
        assert_eq!(
            (round_bf16(score - maximum).to_bits() >> 16) as u16,
            fixture["centered_score_bf16_u16"]
                .as_u64()
                .expect("centered score bits") as u16
        );
    }

    #[test]
    fn pytorch_bf16_specialized_vector_dot_matches_source_order() {
        fn bits(value: &Value, name: &str) -> Vec<f32> {
            value
                .as_array()
                .unwrap_or_else(|| panic!("{name} bits"))
                .iter()
                .map(|value| {
                    f32::from_bits(u32::from(value.as_u64().expect("BF16 payload") as u16) << 16)
                })
                .collect()
        }

        let fixture: Value = serde_json::from_str(include_str!(
            "../evals/fixtures/tiny/pw0073-pytorch-bf16-vector-dot.json"
        ))
        .expect("valid specialized BF16 vector-dot fixture");
        assert_eq!(
            fixture["semantic"],
            "pytorch_aarch64_bf16_specialized_vector_dot_order"
        );
        let query = bits(&fixture["query_bf16_u16"], "query");
        let key = bits(&fixture["key_bf16_u16"], "key");
        assert_eq!(query.len(), QK_HEAD_DIM);
        assert_eq!(key.len(), QK_HEAD_DIM);

        let specialized = pytorch_bf16_specialized_vector_dot_f32(&query, &key);
        let four_lane = pytorch_bf16_four_lane_dot_f32(&query, &key);
        assert_eq!(
            specialized.to_bits(),
            fixture["source_specialized_vector_dot_f32_u32"]
                .as_u64()
                .expect("specialized dot bits") as u32
        );
        assert_eq!(
            four_lane.to_bits(),
            fixture["four_lane_dot_f32_u32"]
                .as_u64()
                .expect("four-lane dot bits") as u32
        );
        assert_ne!(
            round_bf16(specialized).to_bits(),
            round_bf16(four_lane).to_bits()
        );
        assert_eq!(
            (round_bf16(specialized).to_bits() >> 16) as u16,
            fixture["dot_bf16_u16"].as_u64().expect("BF16 dot bits") as u16
        );
        let scale = f32::from_bits(fixture["scale_f32_u32"].as_u64().expect("scale bits") as u32);
        let score = round_bf16(round_bf16(specialized) * scale);
        assert_eq!(
            (score.to_bits() >> 16) as u16,
            fixture["scaled_score_bf16_u16"]
                .as_u64()
                .expect("scaled score bits") as u16
        );
        let maximum = f32::from_bits(
            u32::from(
                fixture["row_maximum_bf16_u16"]
                    .as_u64()
                    .expect("row maximum bits") as u16,
            ) << 16,
        );
        assert_eq!(
            (round_bf16(score - maximum).to_bits() >> 16) as u16,
            fixture["centered_score_bf16_u16"]
                .as_u64()
                .expect("centered score bits") as u16
        );
    }

    #[test]
    fn pytorch_bf16_specialized_swa_score_dot_matches_source_order() {
        fn bits(value: &Value, name: &str) -> Vec<f32> {
            value
                .as_array()
                .unwrap_or_else(|| panic!("{name} bits"))
                .iter()
                .map(|value| {
                    f32::from_bits(u32::from(value.as_u64().expect("BF16 payload") as u16) << 16)
                })
                .collect()
        }

        let fixture: Value = serde_json::from_str(include_str!(
            "../evals/fixtures/tiny/pw0082-pytorch-bf16-swa-score-dot.json"
        ))
        .expect("valid specialized SWA score-dot fixture");
        assert_eq!(
            fixture["semantic"],
            "pytorch_aarch64_bf16_specialized_swa_score_dot_order"
        );
        let query = bits(&fixture["query_bf16_u16"], "query");
        let key = bits(&fixture["key_bf16_u16"], "key");
        let specialized = pytorch_bf16_specialized_vector_dot_f32(&query, &key);
        let four_lane = pytorch_bf16_four_lane_dot_f32(&query, &key);
        assert_eq!(
            specialized.to_bits(),
            fixture["source_specialized_vector_dot_f32_u32"]
                .as_u64()
                .expect("specialized dot bits") as u32
        );
        assert_eq!(
            four_lane.to_bits(),
            fixture["four_lane_dot_f32_u32"]
                .as_u64()
                .expect("four-lane dot bits") as u32
        );
        assert_ne!(
            round_bf16(specialized).to_bits(),
            round_bf16(four_lane).to_bits()
        );
        assert_eq!(
            (round_bf16(specialized).to_bits() >> 16) as u16,
            fixture["dot_bf16_u16"].as_u64().expect("BF16 dot bits") as u16
        );
        let scale = f32::from_bits(fixture["scale_f32_u32"].as_u64().expect("scale bits") as u32);
        let score = round_bf16(round_bf16(specialized) * scale);
        let maximum = f32::from_bits(
            u32::from(
                fixture["row_maximum_bf16_u16"]
                    .as_u64()
                    .expect("row maximum bits") as u16,
            ) << 16,
        );
        assert_eq!(
            (round_bf16(score - maximum).to_bits() >> 16) as u16,
            fixture["centered_score_bf16_u16"]
                .as_u64()
                .expect("centered score bits") as u16
        );
    }

    #[test]
    fn pytorch_bf16_attention_value_dot_matches_source_order() {
        fn bits(value: &Value, name: &str) -> Vec<f32> {
            value
                .as_array()
                .unwrap_or_else(|| panic!("{name} bits"))
                .iter()
                .map(|value| {
                    f32::from_bits(u32::from(value.as_u64().expect("BF16 payload") as u16) << 16)
                })
                .collect()
        }

        let fixture: Value = serde_json::from_str(include_str!(
            "../evals/fixtures/tiny/pw0076-pytorch-bf16-attention-value-dot.json"
        ))
        .expect("valid BF16 attention-value dot fixture");
        assert_eq!(
            fixture["semantic"],
            "pytorch_aarch64_bf16_attention_value_dot_order"
        );
        let probability = bits(&fixture["probability_bf16_u16"], "probability");
        let value = bits(&fixture["value_bf16_u16"], "value");
        assert_eq!(probability.len(), 25);
        assert_eq!(value.len(), 25);
        let specialized = pytorch_bf16_specialized_vector_dot_f32(&probability, &value);
        let generic = pytorch_bf16_four_lane_dot_f32(&probability, &value);
        let forward = probability
            .iter()
            .zip(&value)
            .map(|(left, right)| left * right)
            .sum::<f32>();
        assert_eq!(
            specialized.to_bits(),
            fixture["source_specialized_vector_dot_f32_u32"]
                .as_u64()
                .expect("specialized dot bits") as u32
        );
        assert_eq!(generic.to_bits(), specialized.to_bits());
        assert_eq!(
            forward.to_bits(),
            fixture["forward_dot_f32_u32"]
                .as_u64()
                .expect("forward dot bits") as u32
        );
        assert_ne!(
            round_bf16(specialized).to_bits(),
            round_bf16(forward).to_bits()
        );
        assert_eq!(
            (round_bf16(specialized).to_bits() >> 16) as u16,
            fixture["dot_bf16_u16"].as_u64().expect("BF16 dot bits") as u16
        );

        let discriminating: Value = serde_json::from_str(include_str!(
            "../evals/fixtures/tiny/pw0088-pytorch-bf16-attention-value-gemm.json"
        ))
        .expect("valid discriminating BF16 attention-value GEMM fixture");
        assert_eq!(
            discriminating["semantic"],
            "pytorch_aarch64_bf16_attention_value_gemm_order"
        );
        let probability = bits(&discriminating["probability_bf16_u16"], "probability");
        let value = bits(&discriminating["value_bf16_u16"], "value");
        let generic = pytorch_bf16_four_lane_dot_f32(&probability, &value);
        let specialized = pytorch_bf16_specialized_vector_dot_f32(&probability, &value);
        assert_eq!(
            generic.to_bits(),
            discriminating["source_generic_four_lane_f32_u32"]
                .as_u64()
                .expect("generic dot bits") as u32
        );
        assert_eq!(
            specialized.to_bits(),
            discriminating["source_specialized_vector_f32_u32"]
                .as_u64()
                .expect("specialized dot bits") as u32
        );
        assert_ne!(
            round_bf16(generic).to_bits(),
            round_bf16(specialized).to_bits()
        );
        assert_eq!(
            (round_bf16(generic).to_bits() >> 16) as u16,
            discriminating["matrix_result_bf16_u16"]
                .as_u64()
                .expect("matrix result bits") as u16
        );
    }

    #[test]
    fn pytorch_bf16_lm_head_dot_matches_source_order() {
        fn bits(value: &Value, name: &str) -> Vec<f32> {
            value
                .as_array()
                .unwrap_or_else(|| panic!("{name} bits"))
                .iter()
                .map(|value| {
                    f32::from_bits(u32::from(value.as_u64().expect("BF16 payload") as u16) << 16)
                })
                .collect()
        }

        let fixture: Value = serde_json::from_str(include_str!(
            "../evals/fixtures/tiny/pw0090-pytorch-bf16-lm-head-dot.json"
        ))
        .expect("valid BF16 LM-head dot fixture");
        assert_eq!(
            fixture["semantic"],
            "pytorch_aarch64_bf16_lm_head_specialized_dot"
        );
        let input = bits(&fixture["input_bf16_u16"], "input");
        let weight = bits(&fixture["weight_bf16_u16"], "weight");
        assert_eq!(input.len(), HIDDEN);
        assert_eq!(weight.len(), HIDDEN);
        let specialized = pytorch_bf16_specialized_vector_dot_f32(&input, &weight);
        let generic = pytorch_bf16_four_lane_dot_f32(&input, &weight);
        let forward = input
            .iter()
            .zip(&weight)
            .map(|(left, right)| left * right)
            .sum::<f32>();
        assert_eq!(
            specialized.to_bits(),
            fixture["source_specialized_vector_f32_u32"]
                .as_u64()
                .expect("specialized bits") as u32
        );
        assert_eq!(
            generic.to_bits(),
            fixture["source_generic_four_lane_f32_u32"]
                .as_u64()
                .expect("generic bits") as u32
        );
        assert_eq!(
            forward.to_bits(),
            fixture["forward_f32_u32"].as_u64().expect("forward bits") as u32
        );
        let expected = fixture["pytorch_dot_bf16_u16"]
            .as_u64()
            .expect("PyTorch BF16 bits") as u16;
        assert_eq!((round_bf16(specialized).to_bits() >> 16) as u16, expected);
        assert_ne!(
            fixture["pw0089_rust_logit_bf16_u16"]
                .as_u64()
                .expect("prior Rust BF16 bits") as u16,
            expected
        );
    }

    #[test]
    fn pytorch_router_sigmoid_matches_real_vectorized_payloads() {
        let fixture: Value = serde_json::from_str(include_str!(
            "../evals/fixtures/tiny/pw0062-router-sigmoid.json"
        ))
        .expect("valid router sigmoid fixture");
        assert_eq!(fixture["semantic"], "pytorch_vectorized_f32_sigmoid");
        let logits = fixture["logit_f32_u32"].as_array().expect("logits");
        let expected = fixture["score_f32_u32"].as_array().expect("scores");
        assert_eq!(logits.len(), expected.len());
        for (logit, score) in logits.iter().zip(expected) {
            assert_eq!(
                pytorch_sigmoid_f32(f32::from_bits(logit.as_u64().expect("logit bits") as u32))
                    .to_bits(),
                score.as_u64().expect("score bits") as u32
            );
        }
    }

    #[test]
    fn sleef_exp_and_sigmoid_match_subnormal_boundaries() {
        let fixture: Value = serde_json::from_str(include_str!(
            "../evals/fixtures/tiny/pw0068-sleef-boundaries.json"
        ))
        .expect("valid SLEEF boundary fixture");
        assert_eq!(
            fixture["semantic"],
            "pytorch_sleef_u10_exp_and_sigmoid_boundaries"
        );
        let inputs = fixture["input_f32_u32"].as_array().expect("inputs");
        let exponentials = fixture["exp_f32_u32"].as_array().expect("exponentials");
        let sigmoids = fixture["sigmoid_of_negated_input_f32_u32"]
            .as_array()
            .expect("sigmoids");
        assert_eq!(inputs.len(), exponentials.len());
        assert_eq!(inputs.len(), sigmoids.len());
        for ((input, exponential), sigmoid) in inputs.iter().zip(exponentials).zip(sigmoids) {
            let value = f32::from_bits(input.as_u64().expect("input bits") as u32);
            assert_eq!(
                sleef_expf_u10(value).to_bits(),
                exponential.as_u64().expect("exponential bits") as u32
            );
            assert_eq!(
                pytorch_sigmoid_f32(-value).to_bits(),
                sigmoid.as_u64().expect("sigmoid bits") as u32
            );
        }
    }

    #[test]
    fn pytorch_unsorted_topk_matches_real_route_order() {
        let fixture: Value = serde_json::from_str(include_str!(
            "../evals/fixtures/tiny/pw0063-topk-order.json"
        ))
        .expect("valid top-k fixture");
        let corrected = fixture["corrected_f32_u32"]
            .as_array()
            .expect("corrected scores")
            .iter()
            .map(|value| f32::from_bits(value.as_u64().expect("score bits") as u32))
            .collect::<Vec<_>>();
        let mut selected = [0_u32; TOP_K];
        // SAFETY: the fixture and output arrays provide the declared lengths.
        assert_eq!(
            unsafe {
                pw_pytorch_topk_unsorted_f32(
                    corrected.as_ptr(),
                    corrected.len(),
                    TOP_K,
                    selected.as_mut_ptr(),
                )
            },
            0
        );
        let expected = fixture["selected_experts"]
            .as_array()
            .expect("selected experts")
            .iter()
            .map(|value| value.as_u64().expect("expert") as u32)
            .collect::<Vec<_>>();
        assert_eq!(selected.as_slice(), expected);
    }

    #[test]
    fn bounded_metal_endpoint_rejects_non_incremental_rows() {
        assert!(require_one_row_metal_experts(0).is_err());
        assert_eq!(require_one_row_metal_experts(1), Ok(()));
        assert!(require_one_row_metal_experts(2).is_err());
    }

    #[test]
    fn incremental_numerical_gate_is_fail_closed() {
        let exact = numerical_parity(&[1.0, -2.0, 3.0], &[1.0, -2.0, 3.0]).expect("exact parity");
        assert!(exact.passed);
        assert_eq!(exact.equality_fraction, 1.0);
        let drifted = numerical_parity(&[1.0, -2.0, 3.1], &[1.0, -2.0, 3.0]).expect("drift parity");
        assert!(!drifted.passed);
        assert!(numerical_parity(&[1.0], &[]).is_err());
    }

    #[test]
    fn projected_top20_distribution_is_identity_on_equal_logits() {
        let mut logits = vec![-20.0_f32; 152_576];
        for (index, value) in logits.iter_mut().take(20).enumerate() {
            *value = 20.0 - index as f32;
        }
        let (identity, jsd) = projected_top20_jsd(&logits, &logits).expect("projected JSD");
        assert!(identity);
        assert_eq!(jsd, 0.0);
    }

    #[test]
    fn process_activity_deltas_preserve_fault_regressions_and_fail_on_cumulative_regression() {
        let before = ProcessActivity {
            disk_bytes_read: 10,
            pageins: 20,
            minor_faults: 30,
            major_faults: 40,
            user_cpu_us: 50,
            system_cpu_us: 60,
        };
        let after = ProcessActivity {
            disk_bytes_read: 11,
            pageins: 22,
            minor_faults: 33,
            major_faults: 44,
            user_cpu_us: 55,
            system_cpu_us: 66,
        };
        let delta = after.checked_delta(before).expect("monotonic counters");
        assert_eq!(delta.disk_bytes_read, 1);
        assert_eq!(delta.pageins, 2);
        assert_eq!(delta.minor_faults, 3);
        assert_eq!(delta.major_faults, 4);
        assert_eq!(delta.user_cpu_us, 5);
        assert_eq!(delta.system_cpu_us, 6);
        let fault_regression = ProcessActivity {
            minor_faults: 23,
            major_faults: 39,
            ..after
        }
        .checked_delta(before)
        .expect("fault counters are signed observations");
        assert_eq!(fault_regression.minor_faults, -7);
        assert_eq!(fault_regression.major_faults, -1);
        assert!(before.checked_delta(after).is_err());
    }
}
