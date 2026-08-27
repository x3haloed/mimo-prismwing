use memmap2::{Mmap, MmapOptions};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::path::Path;

const ALIGNMENT: u64 = 16 * 1024;
const LEGACY_LAYER: usize = 28;
const TOTAL_EXPERT_COUNT: usize = 8;
pub(crate) const K4_EXACTNESS_CLASS: &str = "L3_modified_weights";
const K4_ROLES: [&str; 7] = [
    "packed",
    "left_sign",
    "right_sign",
    "global_scale",
    "row_scale",
    "correction_left",
    "correction_right",
];
const SOURCE_ROLES: [&str; 6] = [
    "gate_weight",
    "gate_scales",
    "up_weight",
    "up_scales",
    "down_weight",
    "down_scales",
];

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct BundlePayload {
    pub(crate) offset: u64,
    pub(crate) bytes: u64,
    pub(crate) sha256: String,
    pub(crate) alignment: u64,
    #[serde(default)]
    pub(crate) dtype: Option<String>,
    #[serde(default)]
    pub(crate) shape: Option<Vec<usize>>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct K4ProjectionRecord {
    pub(crate) rows: usize,
    pub(crate) columns: usize,
    pub(crate) rank: usize,
    pub(crate) payloads: BTreeMap<String, BundlePayload>,
    pub(crate) source_manifest_sha256: String,
}

#[derive(Debug, Deserialize)]
pub(crate) struct BundleExpertRecord {
    pub(crate) expert: u32,
    pub(crate) format: String,
    #[serde(default)]
    pub(crate) projections: BTreeMap<String, K4ProjectionRecord>,
    #[serde(default)]
    pub(crate) payloads: BTreeMap<String, BundlePayload>,
    #[serde(default)]
    pub(crate) source_fixture_sha256: Option<String>,
}

#[derive(Debug, Deserialize)]
struct RouteAuthority {
    experts: Vec<u32>,
    weights: Vec<f32>,
    candidate_relative_l2: f64,
    #[serde(default = "default_route_threshold")]
    maximum_relative_l2: f64,
    correctness_qualified: bool,
}

fn default_route_threshold() -> f64 {
    0.01
}

fn bundle_contract_supported(
    schema_version: u32,
    semantic: &str,
    layer: usize,
    k4_count: usize,
    source_count: usize,
) -> bool {
    (schema_version == 1
        && semantic == "prismwing_identity_preserving_k4_source_layer_bundle_v1"
        && layer == LEGACY_LAYER
        && matches!((k4_count, source_count), (5, 3) | (3, 5)))
        || (schema_version == 2
            && semantic == "prismwing_mixed_k4_source_layer_bundle_v2"
            && layer < 48
            && matches!((k4_count, source_count), (5, 3) | (4, 4) | (3, 5)))
}

#[derive(Debug, Deserialize)]
struct BundleManifest {
    schema_version: u32,
    experiment_id: String,
    semantic: String,
    layer: usize,
    alignment_bytes: u64,
    bundle_bytes: u64,
    logical_end_bytes: u64,
    bundle_sha256: String,
    tlut: BundlePayload,
    records: Vec<BundleExpertRecord>,
    k4_experts: Vec<u32>,
    source_experts: Vec<u32>,
    route_authority: RouteAuthority,
    identity_policy: String,
    spec_sha256: String,
    claims_excluded: Vec<String>,
}

pub(crate) struct K4SourceLayerBundle {
    pub(crate) mapping: Mmap,
    pub(crate) manifest_sha256: String,
    pub(crate) records: BTreeMap<u32, BundleExpertRecord>,
    pub(crate) tlut: BundlePayload,
    pub(crate) bundle_sha256: String,
    pub(crate) bundle_bytes: u64,
    pub(crate) layer: usize,
    pub(crate) k4_experts: Vec<u32>,
    pub(crate) source_experts: Vec<u32>,
    pub(crate) candidate_route_gate_pass: bool,
    pub(crate) route_candidate_relative_l2: f64,
}

fn sha256(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn payload_slice<'a>(mapping: &'a [u8], payload: &BundlePayload) -> Result<&'a [u8], String> {
    let start = usize::try_from(payload.offset).map_err(|error| error.to_string())?;
    let length = usize::try_from(payload.bytes).map_err(|error| error.to_string())?;
    let end = start
        .checked_add(length)
        .ok_or("K4/source payload extent overflow")?;
    mapping
        .get(start..end)
        .ok_or_else(|| "K4/source payload exceeds bundle".to_owned())
}

fn validate_payload(
    mapping: &[u8],
    payload: &BundlePayload,
    extents: &mut Vec<(u64, u64)>,
) -> Result<(), String> {
    if payload.bytes == 0
        || payload.alignment != ALIGNMENT
        || !payload.offset.is_multiple_of(ALIGNMENT)
        || payload.sha256.len() != 64
    {
        return Err("K4/source payload layout mismatch".to_owned());
    }
    let end = payload
        .offset
        .checked_add(payload.bytes)
        .ok_or("K4/source payload extent overflow")?;
    if extents
        .iter()
        .any(|&(left, right)| payload.offset < right && left < end)
    {
        return Err("K4/source payload extents overlap".to_owned());
    }
    let bytes = payload_slice(mapping, payload)?;
    if sha256(bytes) != payload.sha256 {
        return Err("K4/source payload hash mismatch".to_owned());
    }
    extents.push((payload.offset, end));
    Ok(())
}

impl K4SourceLayerBundle {
    pub(crate) fn open(bundle_path: &Path, manifest_path: &Path) -> Result<Self, String> {
        let manifest_bytes = std::fs::read(manifest_path)
            .map_err(|error| format!("{}: {error}", manifest_path.display()))?;
        let manifest_sha256 = sha256(&manifest_bytes);
        let manifest: BundleManifest = serde_json::from_slice(&manifest_bytes)
            .map_err(|error| format!("{}: {error}", manifest_path.display()))?;
        let file = File::open(bundle_path)
            .map_err(|error| format!("{}: {error}", bundle_path.display()))?;
        let file_bytes = file.metadata().map_err(|error| error.to_string())?.len();
        // SAFETY: the file remains immutable for this read-only mapping's lifetime.
        let mapping =
            unsafe { MmapOptions::new().map(&file) }.map_err(|error| error.to_string())?;
        if !bundle_contract_supported(
            manifest.schema_version,
            &manifest.semantic,
            manifest.layer,
            manifest.k4_experts.len(),
            manifest.source_experts.len(),
        ) || !manifest.experiment_id.starts_with("PW-")
            || manifest.alignment_bytes != ALIGNMENT
            || manifest.bundle_bytes != file_bytes
            || manifest.bundle_bytes != mapping.len() as u64
            || manifest.logical_end_bytes > manifest.bundle_bytes
            || manifest.bundle_sha256 != sha256(&mapping)
            || manifest
                .k4_experts
                .iter()
                .copied()
                .collect::<BTreeSet<_>>()
                .len()
                != manifest.k4_experts.len()
            || manifest
                .source_experts
                .iter()
                .copied()
                .collect::<BTreeSet<_>>()
                .len()
                != manifest.source_experts.len()
            || !manifest
                .k4_experts
                .iter()
                .copied()
                .collect::<BTreeSet<_>>()
                .is_disjoint(
                    &manifest
                        .source_experts
                        .iter()
                        .copied()
                        .collect::<BTreeSet<_>>(),
                )
            || manifest.identity_policy
                != "selected expert IDs must match bundle records exactly; no substitution"
            || manifest.spec_sha256.len() != 64
            || manifest.claims_excluded.is_empty()
        {
            return Err("K4/source bundle authority mismatch".to_owned());
        }
        if manifest.route_authority.experts.len() != TOTAL_EXPERT_COUNT
            || manifest.route_authority.weights.len() != TOTAL_EXPERT_COUNT
            || manifest
                .route_authority
                .weights
                .iter()
                .any(|value| !value.is_finite() || *value < 0.0)
            || !manifest.route_authority.candidate_relative_l2.is_finite()
            || manifest.route_authority.candidate_relative_l2 < 0.0
            || !manifest.route_authority.maximum_relative_l2.is_finite()
            || manifest.route_authority.maximum_relative_l2 <= 0.0
            || manifest.route_authority.correctness_qualified
                != (manifest.route_authority.candidate_relative_l2
                    <= manifest.route_authority.maximum_relative_l2)
            || manifest
                .route_authority
                .experts
                .iter()
                .copied()
                .collect::<BTreeSet<_>>()
                != manifest
                    .k4_experts
                    .iter()
                    .copied()
                    .chain(manifest.source_experts.iter().copied())
                    .collect::<BTreeSet<_>>()
        {
            return Err("K4/source route authority mismatch".to_owned());
        }
        let mut extents = Vec::new();
        validate_payload(&mapping, &manifest.tlut, &mut extents)?;
        if manifest.tlut.bytes != 4096 {
            return Err("K4/source TLUT layout mismatch".to_owned());
        }
        let mut records = BTreeMap::new();
        for record in manifest.records {
            let expert = record.expert;
            if records.contains_key(&expert) {
                return Err("duplicate K4/source expert identity".to_owned());
            }
            if manifest.k4_experts.contains(&expert) {
                if record.format != "qtip_k4_ldlq"
                    || record
                        .projections
                        .keys()
                        .map(String::as_str)
                        .collect::<BTreeSet<_>>()
                        != ["gate", "up", "down"].into_iter().collect()
                    || !record.payloads.is_empty()
                {
                    return Err("K4 expert record layout mismatch".to_owned());
                }
                for (name, projection) in &record.projections {
                    let expected = if name == "down" {
                        (4096, 2048)
                    } else {
                        (2048, 4096)
                    };
                    if (projection.rows, projection.columns) != expected
                        || projection.rank != 1
                        || projection.source_manifest_sha256.len() != 64
                        || projection
                            .payloads
                            .keys()
                            .map(String::as_str)
                            .collect::<BTreeSet<_>>()
                            != K4_ROLES.into_iter().collect()
                    {
                        return Err("K4 projection record mismatch".to_owned());
                    }
                    for payload in projection.payloads.values() {
                        validate_payload(&mapping, payload, &mut extents)?;
                    }
                }
            } else if manifest.source_experts.contains(&expert) {
                if record.format != "source_fp8_e4m3_block128"
                    || !record.projections.is_empty()
                    || record
                        .source_fixture_sha256
                        .as_deref()
                        .is_none_or(|hash| hash.len() != 64)
                    || record
                        .payloads
                        .keys()
                        .map(String::as_str)
                        .collect::<BTreeSet<_>>()
                        != SOURCE_ROLES.into_iter().collect()
                {
                    return Err("source expert record layout mismatch".to_owned());
                }
                for (role, payload) in &record.payloads {
                    let weight = role.ends_with("_weight");
                    let expected_shape = match role.as_str() {
                        "gate_weight" | "up_weight" => vec![2048, 4096],
                        "down_weight" => vec![4096, 2048],
                        "gate_scales" | "up_scales" => vec![16, 32],
                        "down_scales" => vec![32, 16],
                        _ => unreachable!(),
                    };
                    if payload.dtype.as_deref() != Some(if weight { "F8_E4M3" } else { "F32" })
                        || payload.shape.as_ref() != Some(&expected_shape)
                    {
                        return Err("source expert payload type mismatch".to_owned());
                    }
                    validate_payload(&mapping, payload, &mut extents)?;
                }
            } else {
                return Err("unknown K4/source expert identity".to_owned());
            }
            records.insert(expert, record);
        }
        if records.keys().copied().collect::<BTreeSet<_>>()
            != manifest
                .k4_experts
                .iter()
                .copied()
                .chain(manifest.source_experts.iter().copied())
                .collect::<BTreeSet<_>>()
        {
            return Err("K4/source bundle identity coverage mismatch".to_owned());
        }
        Ok(Self {
            mapping,
            manifest_sha256,
            records,
            tlut: manifest.tlut,
            bundle_sha256: manifest.bundle_sha256,
            bundle_bytes: manifest.bundle_bytes,
            layer: manifest.layer,
            k4_experts: manifest.k4_experts,
            source_experts: manifest.source_experts,
            candidate_route_gate_pass: manifest.route_authority.correctness_qualified,
            route_candidate_relative_l2: manifest.route_authority.candidate_relative_l2,
        })
    }
}

#[derive(Debug, Serialize)]
pub struct K4SourceBundleVerificationReport {
    pub schema_version: u32,
    pub semantic: &'static str,
    pub commit: String,
    pub bundle_sha256: String,
    pub bundle_bytes: u64,
    pub layer: usize,
    pub expert_ids: Vec<u32>,
    pub k4_experts: Vec<u32>,
    pub source_experts: Vec<u32>,
    pub route_candidate_relative_l2: f64,
    pub payloads: usize,
    pub payload_bytes: u64,
    pub exactness_class: &'static str,
    pub expert_identity_preserved: bool,
    pub source_function_preserved: bool,
    pub identity_substitution_allowed: bool,
    pub candidate_route_gate_pass: bool,
    pub status: &'static str,
}

pub fn verify_k4_source_layer_bundle(
    bundle_path: &Path,
    manifest_path: &Path,
    output_path: &Path,
    commit: &str,
) -> Result<K4SourceBundleVerificationReport, String> {
    if output_path.exists() {
        return Err(format!("refusing to overwrite {}", output_path.display()));
    }
    if commit.len() != 40
        || !commit
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err("implementation commit must be lowercase 40-hex".to_owned());
    }
    let bundle = K4SourceLayerBundle::open(bundle_path, manifest_path)?;
    let payloads = 1 + bundle
        .records
        .values()
        .map(|record| {
            record.payloads.len()
                + record
                    .projections
                    .values()
                    .map(|projection| projection.payloads.len())
                    .sum::<usize>()
        })
        .sum::<usize>();
    let payload_bytes = bundle.tlut.bytes
        + bundle
            .records
            .values()
            .flat_map(|record| {
                record.payloads.values().chain(
                    record
                        .projections
                        .values()
                        .flat_map(|projection| projection.payloads.values()),
                )
            })
            .map(|payload| payload.bytes)
            .sum::<u64>();
    let report = K4SourceBundleVerificationReport {
        schema_version: 1,
        semantic: "prismwing_k4_source_layer_bundle_rust_readback",
        commit: commit.to_owned(),
        bundle_sha256: bundle.bundle_sha256,
        bundle_bytes: bundle.bundle_bytes,
        layer: bundle.layer,
        expert_ids: bundle.records.keys().copied().collect(),
        k4_experts: bundle.k4_experts.clone(),
        source_experts: bundle.source_experts.clone(),
        route_candidate_relative_l2: bundle.route_candidate_relative_l2,
        payloads,
        payload_bytes,
        exactness_class: K4_EXACTNESS_CLASS,
        expert_identity_preserved: true,
        source_function_preserved: false,
        identity_substitution_allowed: false,
        candidate_route_gate_pass: bundle.candidate_route_gate_pass,
        status: if bundle.candidate_route_gate_pass {
            "modified_k4_source_candidate_bundle_verified"
        } else {
            "modified_k4_source_bundle_verified_without_route_gate"
        },
    };
    crate::write_create_new(
        output_path,
        &serde_json::to_vec_pretty(&report).map_err(|error| error.to_string())?,
    )?;
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn supported_identity_counts_are_disjoint_and_complete() {
        let k4 = [188, 93, 199, 248, 252]
            .into_iter()
            .collect::<BTreeSet<_>>();
        let source = [117, 114, 41].into_iter().collect::<BTreeSet<_>>();
        assert!(k4.is_disjoint(&source));
        assert_eq!(k4.len() + source.len(), 8);
        assert_eq!(K4_EXACTNESS_CLASS, "L3_modified_weights");
    }

    #[test]
    fn schema_two_adds_layer_generic_four_four_without_weakening_legacy() {
        let v1 = "prismwing_identity_preserving_k4_source_layer_bundle_v1";
        let v2 = "prismwing_mixed_k4_source_layer_bundle_v2";
        assert!(bundle_contract_supported(1, v1, 28, 3, 5));
        assert!(!bundle_contract_supported(1, v1, 4, 4, 4));
        assert!(bundle_contract_supported(2, v2, 4, 4, 4));
        assert!(bundle_contract_supported(2, v2, 47, 5, 3));
        assert!(!bundle_contract_supported(2, v2, 48, 4, 4));
        assert!(!bundle_contract_supported(2, v1, 4, 4, 4));
        assert!(!bundle_contract_supported(2, v2, 4, 2, 6));
    }
}
