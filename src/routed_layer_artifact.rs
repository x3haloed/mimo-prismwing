use super::{
    MappedTensorMetadata, MappedTensorView, UniqueJson, ValidatedMappedFp8, sha256_hex,
    validate_prevalidated_fp8_views, write_create_new,
};
use memmap2::{Mmap, MmapOptions};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::fs::{self, File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::Path;

pub(crate) const ARTIFACT_SCHEMA_VERSION: u32 = 1;
pub(crate) const ARTIFACT_SEMANTIC: &str = "mimo_v2_5_l1_page_stable_routed_layer";

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct RoutedLayerArtifactManifest {
    pub(crate) schema_version: u32,
    pub(crate) semantic: String,
    pub(crate) revision: String,
    pub(crate) commit: String,
    pub(crate) checkpoint_verification_sha256: String,
    pub(crate) oracle_manifest_sha256: String,
    pub(crate) layer: usize,
    pub(crate) page_bytes: usize,
    pub(crate) artifact_file: String,
    pub(crate) artifact_bytes: u64,
    pub(crate) artifact_sha256: String,
    pub(crate) selected_experts: Vec<u32>,
    pub(crate) tensors: Vec<RoutedLayerArtifactTensor>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct RoutedLayerArtifactTensor {
    pub(crate) expert: u32,
    pub(crate) projection: String,
    pub(crate) role: String,
    pub(crate) source_shard: String,
    pub(crate) source_shard_sha256: String,
    pub(crate) source_absolute_offsets: [u64; 2],
    pub(crate) source_tensor_sha256: String,
    pub(crate) artifact_metadata: MappedTensorMetadata,
    pub(crate) artifact_region_bytes: u64,
    pub(crate) artifact_tensor_sha256: String,
}

pub(crate) struct RoutedLayerSourceTensor<'a> {
    pub(crate) expert: u32,
    pub(crate) projection: &'static str,
    pub(crate) role: &'static str,
    pub(crate) source_shard: &'a str,
    pub(crate) source_shard_sha256: &'a str,
    pub(crate) source_absolute_offsets: [u64; 2],
    pub(crate) metadata: &'a MappedTensorMetadata,
    pub(crate) bytes: &'a [u8],
}

pub(crate) struct RoutedLayerArtifact {
    pub(crate) manifest: RoutedLayerArtifactManifest,
    pub(crate) manifest_sha256: String,
    mapping: Mmap,
}

fn align_up(value: u64, alignment: u64) -> Result<u64, String> {
    if alignment == 0 || !alignment.is_power_of_two() {
        return Err("artifact alignment must be a nonzero power of two".to_owned());
    }
    value
        .checked_add(alignment - 1)
        .map(|sum| sum & !(alignment - 1))
        .ok_or_else(|| "artifact alignment overflow".to_owned())
}

fn validate_name(value: &str, label: &str) -> Result<(), String> {
    if value.is_empty()
        || value == "."
        || value == ".."
        || value.contains('/')
        || value.contains('\\')
        || value.as_bytes().contains(&0)
    {
        return Err(format!("invalid {label}: {value:?}"));
    }
    Ok(())
}

fn is_lower_hex(value: &str, bytes: usize) -> bool {
    value.len() == bytes * 2
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn tensor_expected_bytes(metadata: &MappedTensorMetadata) -> Result<u64, String> {
    let element_bytes = match metadata.dtype.as_str() {
        "F8_E4M3" => 1_u64,
        "F32" => 4_u64,
        other => {
            return Err(format!(
                "{}: unsupported artifact dtype {other}",
                metadata.name
            ));
        }
    };
    if metadata.shape.is_empty() || metadata.shape.contains(&0) {
        return Err(format!("{}: empty artifact tensor shape", metadata.name));
    }
    metadata
        .shape
        .iter()
        .try_fold(element_bytes, |bytes, dimension| {
            bytes
                .checked_mul(*dimension)
                .ok_or_else(|| format!("{}: artifact tensor shape overflow", metadata.name))
        })
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn build_routed_layer_artifact(
    artifact_path: &Path,
    manifest_path: &Path,
    revision: &str,
    commit: &str,
    checkpoint_verification_sha256: &str,
    oracle_manifest_sha256: &str,
    layer: usize,
    page_bytes: usize,
    selected_experts: Vec<u32>,
    sources: &[RoutedLayerSourceTensor<'_>],
) -> Result<RoutedLayerArtifactManifest, String> {
    if artifact_path.exists() || manifest_path.exists() {
        return Err("refusing to overwrite routed-layer artifact output".to_owned());
    }
    if page_bytes == 0 || !page_bytes.is_power_of_two() {
        return Err("host page size must be a nonzero power of two".to_owned());
    }
    if sources.is_empty() || selected_experts.is_empty() {
        return Err("routed-layer artifact source set is empty".to_owned());
    }
    let artifact_file = artifact_path
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or("artifact file name is not UTF-8")?
        .to_owned();
    validate_name(&artifact_file, "artifact file name")?;
    let temporary =
        artifact_path.with_file_name(format!(".{artifact_file}.tmp.{}", std::process::id()));
    let build_result = (|| -> Result<RoutedLayerArtifactManifest, String> {
        let file = OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&temporary)
            .map_err(|error| format!("{}: {error}", temporary.display()))?;
        let mut writer = BufWriter::with_capacity(8 * 1024 * 1024, file);
        let mut artifact_digest = Sha256::new();
        let zeros = vec![0_u8; page_bytes];
        let mut offset = 0_u64;
        let mut records = Vec::with_capacity(sources.len());
        let mut identities = BTreeSet::new();
        for source in sources {
            if !selected_experts.contains(&source.expert)
                || !matches!(source.projection, "gate" | "up" | "down")
                || !matches!(source.role, "weight" | "scale")
                || !identities.insert((source.expert, source.projection, source.role))
            {
                return Err("invalid or duplicate routed-layer source identity".to_owned());
            }
            if source.metadata.data_bytes != source.bytes.len() as u64
                || tensor_expected_bytes(source.metadata)? != source.bytes.len() as u64
                || source.source_absolute_offsets[1].checked_sub(source.source_absolute_offsets[0])
                    != Some(source.bytes.len() as u64)
            {
                return Err(format!(
                    "{}: source metadata or byte range mismatch",
                    source.metadata.name
                ));
            }
            validate_name(source.source_shard, "source shard")?;
            let region_bytes = align_up(source.bytes.len() as u64, page_bytes as u64)?;
            if !offset.is_multiple_of(page_bytes as u64) {
                return Err("artifact writer lost page alignment".to_owned());
            }
            let end = offset
                .checked_add(source.bytes.len() as u64)
                .ok_or("artifact tensor end overflow")?;
            let region_end = offset
                .checked_add(region_bytes)
                .ok_or("artifact region end overflow")?;
            let tensor_sha256 = sha256_hex(source.bytes);
            writer
                .write_all(source.bytes)
                .map_err(|error| format!("artifact payload write: {error}"))?;
            artifact_digest.update(source.bytes);
            let mut padding = region_end - end;
            while padding > 0 {
                let chunk = usize::try_from(padding.min(zeros.len() as u64))
                    .map_err(|_| "padding chunk does not fit usize")?;
                writer
                    .write_all(&zeros[..chunk])
                    .map_err(|error| format!("artifact padding write: {error}"))?;
                artifact_digest.update(&zeros[..chunk]);
                padding -= chunk as u64;
            }
            records.push(RoutedLayerArtifactTensor {
                expert: source.expert,
                projection: source.projection.to_owned(),
                role: source.role.to_owned(),
                source_shard: source.source_shard.to_owned(),
                source_shard_sha256: source.source_shard_sha256.to_owned(),
                source_absolute_offsets: source.source_absolute_offsets,
                source_tensor_sha256: tensor_sha256.clone(),
                artifact_metadata: MappedTensorMetadata {
                    name: source.metadata.name.clone(),
                    dtype: source.metadata.dtype.clone(),
                    shape: source.metadata.shape.clone(),
                    data_offsets: [offset, end],
                    data_bytes: source.metadata.data_bytes,
                },
                artifact_region_bytes: region_bytes,
                artifact_tensor_sha256: tensor_sha256,
            });
            offset = region_end;
        }
        writer
            .flush()
            .and_then(|_| writer.get_ref().sync_all())
            .map_err(|error| format!("artifact sync: {error}"))?;
        fs::hard_link(&temporary, artifact_path)
            .map_err(|error| format!("artifact publish: {error}"))?;
        Ok(RoutedLayerArtifactManifest {
            schema_version: ARTIFACT_SCHEMA_VERSION,
            semantic: ARTIFACT_SEMANTIC.to_owned(),
            revision: revision.to_owned(),
            commit: commit.to_owned(),
            checkpoint_verification_sha256: checkpoint_verification_sha256.to_owned(),
            oracle_manifest_sha256: oracle_manifest_sha256.to_owned(),
            layer,
            page_bytes,
            artifact_file,
            artifact_bytes: offset,
            artifact_sha256: artifact_digest
                .finalize()
                .iter()
                .map(|byte| format!("{byte:02x}"))
                .collect(),
            selected_experts,
            tensors: records,
        })
    })();
    let _ = fs::remove_file(&temporary);
    let manifest = build_result?;
    let bytes = serde_json::to_vec_pretty(&manifest).map_err(|error| error.to_string())?;
    if let Err(error) = write_create_new(manifest_path, &bytes) {
        return Err(format!(
            "artifact was published at {} but manifest publication failed: {error}",
            artifact_path.display()
        ));
    }
    Ok(manifest)
}

pub(crate) fn open_routed_layer_artifact(
    artifact_path: &Path,
    manifest_path: &Path,
    verify_payload_hashes: bool,
) -> Result<RoutedLayerArtifact, String> {
    let manifest_bytes =
        fs::read(manifest_path).map_err(|error| format!("{}: {error}", manifest_path.display()))?;
    let unique: UniqueJson = serde_json::from_slice(&manifest_bytes)
        .map_err(|error| format!("routed-layer artifact manifest: {error}"))?;
    let manifest: RoutedLayerArtifactManifest = serde_json::from_value(unique.0)
        .map_err(|error| format!("routed-layer artifact manifest: {error}"))?;
    validate_name(&manifest.artifact_file, "artifact file name")?;
    if artifact_path.file_name().and_then(|name| name.to_str())
        != Some(manifest.artifact_file.as_str())
    {
        return Err("artifact path does not match manifest file identity".to_owned());
    }
    let file = File::open(artifact_path)
        .map_err(|error| format!("{}: {error}", artifact_path.display()))?;
    let file_bytes = file.metadata().map_err(|error| error.to_string())?.len();
    if file_bytes != manifest.artifact_bytes {
        return Err("routed-layer artifact file length mismatch".to_owned());
    }
    // SAFETY: the artifact is immutable for this process and only immutable slices
    // are exposed. The mapping owns the underlying VM region after File is dropped.
    let mapping = unsafe { MmapOptions::new().map(&file) }
        .map_err(|error| format!("{}: {error}", artifact_path.display()))?;
    validate_manifest_and_mapping(&manifest, &mapping, verify_payload_hashes)?;
    Ok(RoutedLayerArtifact {
        manifest,
        manifest_sha256: sha256_hex(&manifest_bytes),
        mapping,
    })
}

fn validate_manifest_and_mapping(
    manifest: &RoutedLayerArtifactManifest,
    mapping: &[u8],
    verify_payload_hashes: bool,
) -> Result<(), String> {
    if manifest.schema_version != ARTIFACT_SCHEMA_VERSION
        || manifest.semantic != ARTIFACT_SEMANTIC
        || !is_lower_hex(&manifest.commit, 20)
        || !is_lower_hex(&manifest.checkpoint_verification_sha256, 32)
        || !is_lower_hex(&manifest.oracle_manifest_sha256, 32)
        || !is_lower_hex(&manifest.artifact_sha256, 32)
        || manifest.page_bytes == 0
        || !manifest.page_bytes.is_power_of_two()
        || manifest.artifact_bytes != mapping.len() as u64
        || manifest.selected_experts.is_empty()
        || manifest
            .selected_experts
            .iter()
            .copied()
            .collect::<BTreeSet<_>>()
            .len()
            != manifest.selected_experts.len()
        || manifest.tensors.is_empty()
    {
        return Err("routed-layer artifact top-level authority mismatch".to_owned());
    }
    let mut expected_offset = 0_u64;
    let mut identities = BTreeSet::new();
    let mut tensor_names = BTreeSet::new();
    for record in &manifest.tensors {
        let metadata = &record.artifact_metadata;
        if !manifest.selected_experts.contains(&record.expert)
            || !matches!(record.projection.as_str(), "gate" | "up" | "down")
            || !matches!(record.role.as_str(), "weight" | "scale")
            || !identities.insert((
                record.expert,
                record.projection.as_str(),
                record.role.as_str(),
            ))
            || !tensor_names.insert(metadata.name.as_str())
            || validate_name(&record.source_shard, "source shard").is_err()
            || !is_lower_hex(&record.source_shard_sha256, 32)
            || !is_lower_hex(&record.source_tensor_sha256, 32)
            || !is_lower_hex(&record.artifact_tensor_sha256, 32)
            || metadata.data_offsets[0] != expected_offset
            || !expected_offset.is_multiple_of(manifest.page_bytes as u64)
            || record.artifact_region_bytes == 0
            || !record
                .artifact_region_bytes
                .is_multiple_of(manifest.page_bytes as u64)
            || metadata.data_offsets[1].checked_sub(metadata.data_offsets[0])
                != Some(metadata.data_bytes)
            || tensor_expected_bytes(metadata)? != metadata.data_bytes
            || record.source_absolute_offsets[1].checked_sub(record.source_absolute_offsets[0])
                != Some(metadata.data_bytes)
            || record.source_tensor_sha256 != record.artifact_tensor_sha256
        {
            return Err(format!("{}: invalid artifact tensor record", metadata.name));
        }
        let region_end = expected_offset
            .checked_add(record.artifact_region_bytes)
            .ok_or("artifact record region overflow")?;
        if metadata.data_offsets[1] > region_end || region_end > mapping.len() as u64 {
            return Err(format!("{}: artifact record exceeds file", metadata.name));
        }
        let start = usize::try_from(metadata.data_offsets[0])
            .map_err(|_| "artifact start does not fit usize")?;
        let end = usize::try_from(metadata.data_offsets[1])
            .map_err(|_| "artifact end does not fit usize")?;
        let padded_end =
            usize::try_from(region_end).map_err(|_| "artifact padded end does not fit usize")?;
        if mapping[end..padded_end].iter().any(|byte| *byte != 0) {
            return Err(format!("{}: artifact padding is not zero", metadata.name));
        }
        if verify_payload_hashes
            && sha256_hex(&mapping[start..end]) != record.artifact_tensor_sha256
        {
            return Err(format!("{}: artifact tensor hash mismatch", metadata.name));
        }
        expected_offset = region_end;
    }
    if expected_offset != manifest.artifact_bytes {
        return Err("artifact records do not cover the complete file".to_owned());
    }
    if verify_payload_hashes && sha256_hex(mapping) != manifest.artifact_sha256 {
        return Err("complete routed-layer artifact hash mismatch".to_owned());
    }
    Ok(())
}

impl RoutedLayerArtifact {
    fn record(
        &self,
        expert: u32,
        projection: &str,
        role: &str,
    ) -> Result<&RoutedLayerArtifactTensor, String> {
        let mut matching = self.manifest.tensors.iter().filter(|record| {
            record.expert == expert && record.projection == projection && record.role == role
        });
        let record = matching
            .next()
            .ok_or_else(|| format!("artifact lacks expert {expert} {projection} {role}"))?;
        if matching.next().is_some() {
            return Err("artifact tensor identity is ambiguous".to_owned());
        }
        Ok(record)
    }

    fn view<'a>(
        &'a self,
        record: &'a RoutedLayerArtifactTensor,
    ) -> Result<MappedTensorView<'a>, String> {
        let start = usize::try_from(record.artifact_metadata.data_offsets[0])
            .map_err(|_| "artifact tensor start does not fit usize")?;
        let end = usize::try_from(record.artifact_metadata.data_offsets[1])
            .map_err(|_| "artifact tensor end does not fit usize")?;
        Ok(MappedTensorView {
            metadata: &record.artifact_metadata,
            bytes: &self.mapping[start..end],
        })
    }

    pub(crate) fn validated_fp8<'a>(
        &'a self,
        expert: u32,
        projection: &str,
        input: &[f32],
    ) -> Result<(ValidatedMappedFp8<'a>, [usize; 2]), String> {
        let weight = self.record(expert, projection, "weight")?;
        let scale = self.record(expert, projection, "scale")?;
        let backing = [
            usize::try_from(weight.artifact_region_bytes)
                .map_err(|_| "weight backing length does not fit usize")?,
            usize::try_from(scale.artifact_region_bytes)
                .map_err(|_| "scale backing length does not fit usize")?,
        ];
        Ok((
            validate_prevalidated_fp8_views(self.view(weight)?, self.view(scale)?, input)?,
            backing,
        ))
    }

    pub(crate) fn no_copy_probe_region(&self) -> Result<(&[u8], usize), String> {
        let record = self
            .manifest
            .tensors
            .first()
            .ok_or("artifact has no probe region")?;
        let start = usize::try_from(record.artifact_metadata.data_offsets[0])
            .map_err(|_| "probe start does not fit usize")?;
        let region = usize::try_from(record.artifact_region_bytes)
            .map_err(|_| "probe region does not fit usize")?;
        let end = start.checked_add(region).ok_or("probe end overflow")?;
        Ok((&self.mapping[start..end], self.manifest.page_bytes))
    }

    pub(crate) fn prefault_pages(&self) -> u64 {
        self.mapping
            .chunks(self.manifest.page_bytes)
            .fold(0_u64, |checksum, page| {
                // SAFETY: every chunk is nonempty and belongs to this live immutable
                // mapping. Volatile reads ensure the warm-up faults every VM page.
                checksum.wrapping_add(u64::from(unsafe { std::ptr::read_volatile(page.as_ptr()) }))
            })
    }

    pub(crate) fn invalidate_pages(&self) -> Result<(), String> {
        // SAFETY: this is a live immutable file mapping. Both calls only discard
        // clean cached pages; later reads fault authoritative artifact bytes back.
        if unsafe {
            libc::msync(
                self.mapping.as_ptr().cast_mut().cast(),
                self.mapping.len(),
                libc::MS_INVALIDATE,
            )
        } != 0
        {
            return Err(format!(
                "artifact cache invalidation failed: {}",
                std::io::Error::last_os_error()
            ));
        }
        // SAFETY: identical live mapping and advisory-only range.
        if unsafe {
            libc::madvise(
                self.mapping.as_ptr().cast_mut().cast(),
                self.mapping.len(),
                libc::MADV_DONTNEED,
            )
        } != 0
        {
            return Err(format!(
                "artifact MADV_DONTNEED failed: {}",
                std::io::Error::last_os_error()
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture_manifest() -> (RoutedLayerArtifactManifest, Vec<u8>) {
        let page = 16_usize;
        let mut data = vec![0_u8; 32];
        data[..4].copy_from_slice(&[1, 2, 3, 4]);
        data[16..20].copy_from_slice(&1.0_f32.to_le_bytes());
        let weight_hash = sha256_hex(&data[..4]);
        let scale_hash = sha256_hex(&data[16..20]);
        let records = vec![
            RoutedLayerArtifactTensor {
                expert: 7,
                projection: "gate".to_owned(),
                role: "weight".to_owned(),
                source_shard: "source.safetensors".to_owned(),
                source_shard_sha256: "a".repeat(64),
                source_absolute_offsets: [8, 12],
                source_tensor_sha256: weight_hash.clone(),
                artifact_metadata: MappedTensorMetadata {
                    name: "weight".to_owned(),
                    dtype: "F8_E4M3".to_owned(),
                    shape: vec![2, 2],
                    data_offsets: [0, 4],
                    data_bytes: 4,
                },
                artifact_region_bytes: 16,
                artifact_tensor_sha256: weight_hash,
            },
            RoutedLayerArtifactTensor {
                expert: 7,
                projection: "gate".to_owned(),
                role: "scale".to_owned(),
                source_shard: "source.safetensors".to_owned(),
                source_shard_sha256: "a".repeat(64),
                source_absolute_offsets: [12, 16],
                source_tensor_sha256: scale_hash.clone(),
                artifact_metadata: MappedTensorMetadata {
                    name: "scale".to_owned(),
                    dtype: "F32".to_owned(),
                    shape: vec![1],
                    data_offsets: [16, 20],
                    data_bytes: 4,
                },
                artifact_region_bytes: 16,
                artifact_tensor_sha256: scale_hash,
            },
        ];
        let manifest = RoutedLayerArtifactManifest {
            schema_version: ARTIFACT_SCHEMA_VERSION,
            semantic: ARTIFACT_SEMANTIC.to_owned(),
            revision: "revision".to_owned(),
            commit: "0".repeat(40),
            checkpoint_verification_sha256: "b".repeat(64),
            oracle_manifest_sha256: "c".repeat(64),
            layer: 4,
            page_bytes: page,
            artifact_file: "artifact.bin".to_owned(),
            artifact_bytes: data.len() as u64,
            artifact_sha256: sha256_hex(&data),
            selected_experts: vec![7],
            tensors: records,
        };
        (manifest, data)
    }

    #[test]
    fn artifact_validation_fails_closed_on_payload_padding_and_layout_mutations() {
        let (manifest, data) = fixture_manifest();
        validate_manifest_and_mapping(&manifest, &data, true).expect("valid fixture");
        for (name, mutate) in [("payload", 0_usize), ("padding", 5_usize)] {
            let mut changed = data.clone();
            changed[mutate] ^= 1;
            assert!(
                validate_manifest_and_mapping(&manifest, &changed, true).is_err(),
                "{name} mutation must fail"
            );
        }
        let mut overlap = manifest.clone();
        overlap.tensors[1].artifact_metadata.data_offsets = [0, 4];
        assert!(validate_manifest_and_mapping(&overlap, &data, true).is_err());
        let mut alignment = manifest.clone();
        alignment.tensors[1].artifact_metadata.data_offsets = [15, 19];
        assert!(validate_manifest_and_mapping(&alignment, &data, true).is_err());
        let mut schema = manifest.clone();
        schema.schema_version += 1;
        assert!(validate_manifest_and_mapping(&schema, &data, true).is_err());
        assert!(validate_manifest_and_mapping(&manifest, &data[..31], true).is_err());
    }
}
