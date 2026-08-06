//! PW-0050 bounded, target-faithful slow text endpoint.

use super::{
    MappedSafetensors, MappedTensorView, UniqueJson, accelerate_sgemm_right_transposed,
    decode_bf16_tensor, decode_fp8_matrix_f32, select_noaux_tc_routes, sha256_hex, sha256_reader,
    stable_rms_inverse, validate_fp8_views, write_create_new,
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

struct Checkpoint {
    weight_map: BTreeMap<String, String>,
    shards: BTreeMap<String, MappedSafetensors>,
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
        for shard in shard_names {
            let record = verified
                .get(shard.as_str())
                .ok_or_else(|| format!("indexed shard absent from verification: {shard}"))?;
            verify_live_identity(root, record)?;
            let mapped = MappedSafetensors::open(&root.join(&shard))?;
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
        Ok(Self { weight_map, shards })
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
    fn vvexpf(output: *mut f32, input: *const f32, count: *const i32);
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
    let raw_identity = fixture.schema_version == 1
        && fixture.semantic == "mimo_v2_5_target_faithful_raw_text_incremental_decode"
        && fixture.prompt_utf8 == "Hello"
        && fixture.expected_prompt_token_ids == [9707]
        && fixture.hosted_reference.is_none();
    let chat_identity = fixture.schema_version == 2
        && fixture.semantic == "mimo_v2_5_target_faithful_chat_prefill_incremental_decode"
        && fixture.prompt_utf8 == CHAT_PROMPT
        && fixture.expected_prompt_token_ids == CHAT_PROMPT_IDS
        && fixture.hosted_reference.as_ref().is_some_and(|hosted| {
            hosted.provider == "Parasail"
                && hosted.manifest_sha256
                    == "f9c5dd42a76e0eb87581fa427fe03c69ad32903c5711e5078a002ab7514732ea"
                && hosted.response_sha256
                    == "e5a8956f3a7985e1ac3d5396c7bc9fe73bc77c6451eb2225c8df7c8973e3212d"
                && hosted.generated_token_ids == [9707, 0]
                && hosted.generated_text == "Hello!"
        });
    if (!raw_identity && !chat_identity)
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
    let mut probabilities = vec![0.0_f32; centered.len()];
    if bf16_output {
        let count = i32::try_from(centered.len()).map_err(|_| "softmax length exceeds i32")?;
        // SAFETY: input and output are disjoint initialized buffers of `count` F32 values.
        unsafe { vvexpf(probabilities.as_mut_ptr(), centered.as_ptr(), &count) };
    } else {
        for (probability, score) in probabilities.iter_mut().zip(&centered) {
            *probability = score.exp();
        }
    }
    let denominator = probabilities.iter().sum::<f32>();
    if !denominator.is_finite() || denominator <= 0.0 {
        return Err("attention softmax denominator is invalid".to_owned());
    }
    for probability in &mut probabilities {
        *probability = if bf16_output {
            round_bf16(*probability / denominator)
        } else {
            *probability / denominator
        };
    }
    Ok(probabilities)
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
            let dot = query
                .iter()
                .zip(*key)
                .map(|(left, right)| left * right)
                .sum::<f32>();
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
    for (position, value) in values.iter().enumerate() {
        for (destination, source) in output.iter_mut().zip(*value) {
            *destination += probabilities[position] * source;
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
    let routes = select_noaux_tc_routes(&logits, &correction, rows, ROUTED_EXPERTS, TOP_K)?;
    let scores = logits
        .iter()
        .map(|logit| 1.0_f32 / (1.0 + (-logit).exp()))
        .collect();
    Ok(RoutingTrace {
        logits,
        scores,
        selected: routes.selected,
        weights: routes.weights,
    })
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

fn decode_step(
    checkpoint: &Checkpoint,
    config: &ModelConfig,
    token_ids: &[u32],
    caches: &mut [LayerKvCache],
    ledger: &mut EndpointLedger,
    safety: &mut SafetyMonitor,
    mut full_captures: Option<&mut FullPrefixCaptures>,
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
            let routed = routed_mlp(checkpoint, layer, &moe_input, rows, ledger)?;
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
    let logits = bf16_linear(
        checkpoint,
        "lm_head.weight",
        &normalized,
        rows,
        HIDDEN,
        config.vocab_size,
        ledger,
    )?;
    let last_logits = logits[(rows - 1) * config.vocab_size..rows * config.vocab_size].to_vec();
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
        return Err("layer-2 trace requires the frozen chat fixture".to_owned());
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
    let input_norm = bf16_vector(
        &checkpoint,
        "model.layers.1.input_layernorm.weight",
        HIDDEN,
        &mut ledger,
    )?;
    let normalized = rms_norm(&hidden, rows, &input_norm, config.layernorm_epsilon)?;
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
    let post_attention = hidden
        .iter()
        .zip(attention_output)
        .map(|(&residual, projected)| round_bf16(residual + projected))
        .collect::<Vec<_>>();
    let post_norm = bf16_vector(
        &checkpoint,
        "model.layers.1.post_attention_layernorm.weight",
        HIDDEN,
        &mut ledger,
    )?;
    let moe_input = rms_norm(&post_attention, rows, &post_norm, config.layernorm_epsilon)?;
    let layer1_mlp = routed_mlp(&checkpoint, 1, &moe_input, rows, &mut ledger)?;
    hidden = post_attention
        .iter()
        .zip(layer1_mlp.output)
        .map(|(&residual, projected)| round_bf16(residual + projected))
        .collect();
    checkpoint.release_file_pages()?;
    safety.checkpoint("layer_1_complete", true)?;
    let incoming_bytes = hidden
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect::<Vec<_>>();

    let input_norm = bf16_vector(
        &checkpoint,
        "model.layers.2.input_layernorm.weight",
        HIDDEN,
        &mut ledger,
    )?;
    let normalized = rms_norm(&hidden, rows, &input_norm, config.layernorm_epsilon)?;
    let mut attention_captures = Layer0Captures::default();
    let mut cache = LayerKvCache::default();
    let attention_output = attention(
        &checkpoint,
        &config,
        2,
        &normalized,
        rows,
        &mut cache,
        &mut ledger,
        Some(&mut attention_captures),
    )?;
    safety.checkpoint("layer_2_attention_complete", true)?;
    let post_attention = hidden
        .iter()
        .zip(&attention_output)
        .map(|(&residual, &projected)| round_bf16(residual + projected))
        .collect::<Vec<_>>();
    let post_norm = bf16_vector(
        &checkpoint,
        "model.layers.2.post_attention_layernorm.weight",
        HIDDEN,
        &mut ledger,
    )?;
    let moe_input = rms_norm(&post_attention, rows, &post_norm, config.layernorm_epsilon)?;
    let mut expert_captures = ExpertCaptures::default();
    let mut expert_completed = |expert| {
        checkpoint.release_file_pages()?;
        safety.checkpoint(&format!("layer_2_expert_{expert}_complete"), true)
    };
    let routed = routed_mlp_traced(
        &checkpoint,
        2,
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
    safety.checkpoint("layer_2_complete", true)?;
    let placements = rows * TOP_K;
    let attention_states = HEADS * rows * (rows + 3) / 2;
    let mut captures = BTreeMap::new();
    for (name, shape, values) in [
        ("incoming", vec![rows, HIDDEN], hidden.as_slice()),
        ("input_norm", vec![rows, HIDDEN], normalized.as_slice()),
        ("qkv", vec![rows, 14_848], attention_captures.qkv.as_slice()),
        (
            "query",
            vec![rows, HEADS, QK_HEAD_DIM],
            attention_captures.query.as_slice(),
        ),
        (
            "key",
            vec![rows, 8, QK_HEAD_DIM],
            attention_captures.key.as_slice(),
        ),
        (
            "value",
            vec![rows, 8, V_HEAD_DIM],
            attention_captures.value.as_slice(),
        ),
        ("sinks", vec![HEADS], attention_captures.sinks.as_slice()),
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
        semantic: "mimo_real_layer2_complete_rust_trace",
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
    if fixture.schema_version != 2 || prompt_token_ids != CHAT_PROMPT_IDS {
        return Err("full-prefix trace requires the frozen chat fixture".to_owned());
    }
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
    fn vforce_softmax_matches_pytorch_bf16_payloads() {
        let fixture: Value = serde_json::from_str(include_str!(
            "../evals/fixtures/tiny/pw0057-vforce-softmax.json"
        ))
        .expect("valid vForce softmax fixture");
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
        }
        assert!(attention_softmax(&[], true).is_err());
        assert!(attention_softmax(&[f32::NAN], true).is_err());
    }
}
