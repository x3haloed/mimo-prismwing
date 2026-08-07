use crate::routed_layer_artifact::{RoutedLayerArtifactManifest, open_routed_layer_artifact};
use crate::text_endpoint::{
    ComponentSafetyMonitor, ProcessActivityDelta, SafetySnapshot, process_activity,
};
use metal::{Buffer, Device, MTLResourceOptions};
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::fs::File;
use std::os::fd::AsRawFd;
use std::path::Path;
use std::time::Instant;

const ARTIFACT_SHA256: &str = "fac61c2cfad4b00248c96a52b68360fecd39e2c912e6ffd6643e3f06ade00d21";
const MANIFEST_SHA256: &str = "40179385a571a19b135a4740122744ae3d8ea2c97ef265ac20968296e98822b8";
const ARTIFACT_BYTES: usize = 201_719_808;
const EXPERTS: usize = 8;
const SLOT_ALIGNMENT: usize = 2 * 1024 * 1024;
const WORKER_COUNTS: [usize; 4] = [1, 2, 4, 8];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct ExpertExtent {
    expert: u32,
    offset: usize,
    bytes: usize,
}

#[derive(Debug, Serialize)]
pub struct PreadExpertRead {
    pub expert: u32,
    pub slot: usize,
    pub source_offset: u64,
    pub requested_bytes: u64,
    pub returned_bytes: u64,
    pub pread_calls: usize,
    pub wall_ms: f64,
}

#[derive(Debug, Serialize)]
pub struct PreadExpertAcquisitionTrial {
    pub repetition: usize,
    pub cache_state: &'static str,
    pub workers: usize,
    pub transfer_wall_ms: f64,
    pub integrity_ms: f64,
    pub requested_bytes: u64,
    pub returned_bytes: u64,
    pub pread_calls: usize,
    pub slot_stream_sha256: String,
    pub expert_reads: Vec<PreadExpertRead>,
    pub activity: ProcessActivityDelta,
}

#[derive(Debug, Serialize)]
pub struct PreadExpertAcquisitionReport {
    pub schema_version: u32,
    pub semantic: &'static str,
    pub commit: String,
    pub artifact_manifest_sha256: String,
    pub artifact_sha256: String,
    pub artifact_bytes: u64,
    pub expert_stride_bytes: u64,
    pub expert_count: usize,
    pub selected_experts: Vec<u32>,
    pub slot_alignment_bytes: u64,
    pub slot_capacity_bytes: u64,
    pub slot_buffer_pointer_identity: Vec<bool>,
    pub slot_buffer_lengths: Vec<u64>,
    pub metal_device: String,
    pub worker_counts: Vec<usize>,
    pub warm_prefault_ms: f64,
    pub warm_prefault_checksum: u64,
    pub trials: Vec<PreadExpertAcquisitionTrial>,
    pub safety_snapshots: Vec<SafetySnapshot>,
    pub batch_size: usize,
    pub concurrency: usize,
    pub accepted_tokens: usize,
    #[serde(rename = "A")]
    pub accepted_per_verification: usize,
    #[serde(rename = "U")]
    pub unique_experts: usize,
    pub performance_claim: Option<String>,
}

struct AlignedMetalSlots {
    pointers: Vec<*mut libc::c_void>,
    buffers: Vec<Buffer>,
    bytes: usize,
}

impl AlignedMetalSlots {
    fn new(device: &Device, count: usize, bytes: usize) -> Result<Self, String> {
        if count == 0 || bytes == 0 || !bytes.is_multiple_of(16 * 1024) {
            return Err("invalid aligned expert-slot dimensions".to_owned());
        }
        let mut slots = Self {
            pointers: Vec::with_capacity(count),
            buffers: Vec::with_capacity(count),
            bytes,
        };
        for _ in 0..count {
            let mut pointer = std::ptr::null_mut();
            // SAFETY: posix_memalign writes one allocation pointer on success.
            let result = unsafe { libc::posix_memalign(&mut pointer, SLOT_ALIGNMENT, bytes) };
            if result != 0 || pointer.is_null() {
                return Err(format!("posix_memalign failed with {result}"));
            }
            let buffer = device.new_buffer_with_bytes_no_copy(
                pointer.cast_const(),
                bytes as u64,
                MTLResourceOptions::StorageModeShared,
                None,
            );
            if buffer.contents() != pointer || buffer.length() != bytes as u64 {
                // SAFETY: this pointer was just allocated and Metal did not accept
                // the required identity. The buffer drops before this function exits.
                unsafe { libc::free(pointer) };
                return Err("Metal no-copy slot lost pointer identity or length".to_owned());
            }
            slots.pointers.push(pointer);
            slots.buffers.push(buffer);
        }
        Ok(slots)
    }

    fn stream_sha256(&self) -> String {
        let mut digest = Sha256::new();
        for pointer in &self.pointers {
            // SAFETY: every slot owns `bytes` initialized bytes after a successful trial.
            let bytes = unsafe { std::slice::from_raw_parts(pointer.cast::<u8>(), self.bytes) };
            digest.update(bytes);
        }
        digest
            .finalize()
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect()
    }
}

impl Drop for AlignedMetalSlots {
    fn drop(&mut self) {
        self.buffers.clear();
        for pointer in self.pointers.drain(..) {
            // SAFETY: each pointer was returned once by posix_memalign and is no
            // longer retained by a Metal buffer after clearing `buffers`.
            unsafe { libc::free(pointer) };
        }
    }
}

fn derive_expert_extents(
    manifest: &RoutedLayerArtifactManifest,
) -> Result<Vec<ExpertExtent>, String> {
    if manifest.selected_experts.len() != EXPERTS || manifest.tensors.len() != EXPERTS * 6 {
        return Err("PW-0136 artifact does not contain eight complete experts".to_owned());
    }
    let expected_roles = [
        ("gate", "weight"),
        ("gate", "scale"),
        ("up", "weight"),
        ("up", "scale"),
        ("down", "weight"),
        ("down", "scale"),
    ];
    let mut extents = Vec::with_capacity(EXPERTS);
    let mut global_cursor = 0_usize;
    for (&expert, records) in manifest
        .selected_experts
        .iter()
        .zip(manifest.tensors.chunks_exact(6))
    {
        let start = global_cursor;
        for (record, expected) in records.iter().zip(expected_roles) {
            let offset = usize::try_from(record.artifact_metadata.data_offsets[0])
                .map_err(|_| "PW-0136 record offset does not fit usize")?;
            let region = usize::try_from(record.artifact_region_bytes)
                .map_err(|_| "PW-0136 record extent does not fit usize")?;
            if record.expert != expert
                || (record.projection.as_str(), record.role.as_str()) != expected
                || offset != global_cursor
            {
                return Err("PW-0136 expert extent identity or contiguity mismatch".to_owned());
            }
            global_cursor = global_cursor
                .checked_add(region)
                .ok_or("PW-0136 expert extent overflow")?;
        }
        extents.push(ExpertExtent {
            expert,
            offset: start,
            bytes: global_cursor - start,
        });
    }
    let stride = extents
        .first()
        .ok_or("PW-0136 has no expert extents")?
        .bytes;
    if global_cursor != ARTIFACT_BYTES
        || extents.iter().any(|extent| {
            extent.bytes != stride
                || !extent.offset.is_multiple_of(manifest.page_bytes)
                || !extent.bytes.is_multiple_of(manifest.page_bytes)
        })
    {
        return Err("PW-0136 fixed-stride coverage mismatch".to_owned());
    }
    Ok(extents)
}

fn execute_trial(
    repetition: usize,
    cache_state: &'static str,
    workers: usize,
    fd: libc::c_int,
    extents: &[ExpertExtent],
    slots: &AlignedMetalSlots,
) -> Result<PreadExpertAcquisitionTrial, String> {
    if !WORKER_COUNTS.contains(&workers) || extents.len() != slots.pointers.len() {
        return Err("PW-0136 invalid trial dimensions".to_owned());
    }
    let addresses = slots
        .pointers
        .iter()
        .map(|pointer| *pointer as usize)
        .collect::<Vec<_>>();
    let activity_before = process_activity()?;
    let transfer_started = Instant::now();
    let worker_results = std::thread::scope(|scope| {
        let mut handles = Vec::with_capacity(workers);
        for worker in 0..workers {
            let addresses = &addresses;
            handles.push(
                scope.spawn(move || -> Result<Vec<PreadExpertRead>, String> {
                    let mut local = Vec::new();
                    for slot in (worker..extents.len()).step_by(workers) {
                        let extent = extents[slot];
                        let started = Instant::now();
                        let mut filled = 0_usize;
                        let mut calls = 0_usize;
                        while filled < extent.bytes {
                            // SAFETY: each worker writes only its disjoint slot; the
                            // destination and source range both remain valid.
                            let count = unsafe {
                                libc::pread(
                                    fd,
                                    (addresses[slot] as *mut u8).add(filled).cast(),
                                    extent.bytes - filled,
                                    (extent.offset + filled) as libc::off_t,
                                )
                            };
                            calls += 1;
                            if count < 0 {
                                return Err(format!(
                                    "pread expert {} failed: {}",
                                    extent.expert,
                                    std::io::Error::last_os_error()
                                ));
                            }
                            if count == 0 {
                                return Err(format!("pread expert {} returned EOF", extent.expert));
                            }
                            filled = filled
                                .checked_add(count as usize)
                                .ok_or("PW-0136 pread byte count overflow")?;
                        }
                        local.push(PreadExpertRead {
                            expert: extent.expert,
                            slot,
                            source_offset: extent.offset as u64,
                            requested_bytes: extent.bytes as u64,
                            returned_bytes: filled as u64,
                            pread_calls: calls,
                            wall_ms: started.elapsed().as_secs_f64() * 1000.0,
                        });
                    }
                    Ok(local)
                }),
            );
        }
        let mut reads = Vec::with_capacity(extents.len());
        for handle in handles {
            reads.extend(
                handle
                    .join()
                    .map_err(|_| "PW-0136 pread worker panicked".to_owned())??,
            );
        }
        Ok::<_, String>(reads)
    })?;
    let transfer_wall_ms = transfer_started.elapsed().as_secs_f64() * 1000.0;
    let activity = process_activity()?.checked_delta(activity_before)?;
    let mut expert_reads = worker_results;
    expert_reads.sort_by_key(|read| read.slot);
    if expert_reads.len() != extents.len()
        || expert_reads.iter().zip(extents).any(|(read, extent)| {
            read.expert != extent.expert
                || read.requested_bytes != extent.bytes as u64
                || read.returned_bytes != extent.bytes as u64
        })
    {
        return Err("PW-0136 expert read ledger mismatch".to_owned());
    }
    let integrity_started = Instant::now();
    let slot_stream_sha256 = slots.stream_sha256();
    let integrity_ms = integrity_started.elapsed().as_secs_f64() * 1000.0;
    if slot_stream_sha256 != ARTIFACT_SHA256 {
        return Err("PW-0136 slot stream differs from authenticated artifact".to_owned());
    }
    Ok(PreadExpertAcquisitionTrial {
        repetition,
        cache_state,
        workers,
        transfer_wall_ms,
        integrity_ms,
        requested_bytes: expert_reads.iter().map(|read| read.requested_bytes).sum(),
        returned_bytes: expert_reads.iter().map(|read| read.returned_bytes).sum(),
        pread_calls: expert_reads.iter().map(|read| read.pread_calls).sum(),
        slot_stream_sha256,
        expert_reads,
        activity,
    })
}

pub fn benchmark_pread_expert_acquisition(
    artifact_path: &Path,
    artifact_manifest_path: &Path,
    output_path: &Path,
    commit: &str,
) -> Result<PreadExpertAcquisitionReport, String> {
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
    let artifact = open_routed_layer_artifact(artifact_path, artifact_manifest_path, true)?;
    if artifact.manifest_sha256 != MANIFEST_SHA256
        || artifact.manifest.artifact_sha256 != ARTIFACT_SHA256
        || artifact.manifest.artifact_bytes != ARTIFACT_BYTES as u64
    {
        return Err("PW-0136 artifact authority mismatch".to_owned());
    }
    let extents = derive_expert_extents(&artifact.manifest)?;
    let stride = extents[0].bytes;
    safety.checkpoint("pread_artifact_verified")?;
    let device = Device::system_default().ok_or("no Metal device is available")?;
    let slots = AlignedMetalSlots::new(&device, EXPERTS, stride)?;
    let pointer_identity = slots
        .buffers
        .iter()
        .zip(&slots.pointers)
        .map(|(buffer, pointer)| buffer.contents() == *pointer)
        .collect::<Vec<_>>();
    let buffer_lengths = slots
        .buffers
        .iter()
        .map(|buffer| buffer.length())
        .collect::<Vec<_>>();
    if slots
        .bytes
        .checked_mul(EXPERTS)
        .is_none_or(|bytes| bytes >= 1024_usize.pow(3))
        || pointer_identity.iter().any(|identity| !identity)
        || buffer_lengths.iter().any(|length| *length != stride as u64)
    {
        return Err("PW-0136 slot capacity or Metal binding mismatch".to_owned());
    }
    safety.checkpoint("pread_slots_allocated")?;
    let file = File::open(artifact_path)
        .map_err(|error| format!("{}: {error}", artifact_path.display()))?;
    let mut trials = Vec::with_capacity(24);
    let orders = [[1_usize, 2, 4, 8], [8_usize, 4, 2, 1], [2_usize, 8, 1, 4]];
    for (repetition, order) in orders.iter().enumerate() {
        for &workers in order {
            artifact.invalidate_pages()?;
            let trial = execute_trial(
                repetition,
                "cold",
                workers,
                file.as_raw_fd(),
                &extents,
                &slots,
            )?;
            if trial.requested_bytes != ARTIFACT_BYTES as u64
                || trial.returned_bytes != ARTIFACT_BYTES as u64
                || trial.slot_stream_sha256 != ARTIFACT_SHA256
            {
                return Err("PW-0136 cold trial accounting mismatch".to_owned());
            }
            trials.push(trial);
            safety.checkpoint(&format!(
                "pread_cold_repetition_{repetition}_workers_{workers}"
            ))?;
        }
    }
    let warm_started = Instant::now();
    let warm_prefault_checksum = artifact.prefault_pages();
    let warm_prefault_ms = warm_started.elapsed().as_secs_f64() * 1000.0;
    safety.checkpoint("pread_warm_prefault_complete")?;
    for (repetition, order) in orders.iter().enumerate() {
        for &workers in order {
            let trial = execute_trial(
                repetition,
                "warm",
                workers,
                file.as_raw_fd(),
                &extents,
                &slots,
            )?;
            if trial.requested_bytes != ARTIFACT_BYTES as u64
                || trial.returned_bytes != ARTIFACT_BYTES as u64
                || trial.slot_stream_sha256 != ARTIFACT_SHA256
            {
                return Err("PW-0136 warm trial accounting mismatch".to_owned());
            }
            trials.push(trial);
            safety.checkpoint(&format!(
                "pread_warm_repetition_{repetition}_workers_{workers}"
            ))?;
        }
    }
    drop(file);
    drop(slots);
    artifact.invalidate_pages()?;
    drop(artifact);
    let safety_snapshots = safety.released()?;
    let report = PreadExpertAcquisitionReport {
        schema_version: 1,
        semantic: "mimo_v2_5_layer4_page_aligned_pread_expert_slot_acquisition",
        commit: commit.to_owned(),
        artifact_manifest_sha256: MANIFEST_SHA256.to_owned(),
        artifact_sha256: ARTIFACT_SHA256.to_owned(),
        artifact_bytes: ARTIFACT_BYTES as u64,
        expert_stride_bytes: stride as u64,
        expert_count: EXPERTS,
        selected_experts: extents.iter().map(|extent| extent.expert).collect(),
        slot_alignment_bytes: SLOT_ALIGNMENT as u64,
        slot_capacity_bytes: (stride * EXPERTS) as u64,
        slot_buffer_pointer_identity: pointer_identity,
        slot_buffer_lengths: buffer_lengths,
        metal_device: device.name().to_owned(),
        worker_counts: WORKER_COUNTS.to_vec(),
        warm_prefault_ms,
        warm_prefault_checksum,
        trials,
        safety_snapshots,
        batch_size: 1,
        concurrency: 1,
        accepted_tokens: 0,
        accepted_per_verification: 0,
        unique_experts: EXPERTS,
        performance_claim: None,
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
    use crate::MappedTensorMetadata;
    use crate::routed_layer_artifact::RoutedLayerArtifactTensor;

    fn record(
        expert: u32,
        projection: &str,
        role: &str,
        offset: u64,
        region_bytes: u64,
    ) -> RoutedLayerArtifactTensor {
        RoutedLayerArtifactTensor {
            expert,
            projection: projection.to_owned(),
            role: role.to_owned(),
            source_shard: "source.safetensors".to_owned(),
            source_shard_sha256: "a".repeat(64),
            source_absolute_offsets: [offset, offset + 1],
            source_tensor_sha256: "b".repeat(64),
            artifact_metadata: MappedTensorMetadata {
                name: format!("{expert}.{projection}.{role}"),
                dtype: "F8_E4M3".to_owned(),
                shape: vec![1],
                data_offsets: [offset, offset + 1],
                data_bytes: 1,
            },
            artifact_region_bytes: region_bytes,
            artifact_tensor_sha256: "b".repeat(64),
        }
    }

    fn fixture_manifest() -> RoutedLayerArtifactManifest {
        let experts = (0..8).collect::<Vec<_>>();
        let identities = [
            ("gate", "weight", 8_388_608),
            ("gate", "scale", 16_384),
            ("up", "weight", 8_388_608),
            ("up", "scale", 16_384),
            ("down", "weight", 8_388_608),
            ("down", "scale", 16_384),
        ];
        let mut offset = 0_u64;
        let mut tensors = Vec::new();
        for &expert in &experts {
            for (projection, role, region_bytes) in identities {
                tensors.push(record(expert, projection, role, offset, region_bytes));
                offset += region_bytes;
            }
        }
        RoutedLayerArtifactManifest {
            schema_version: 1,
            semantic: "test".to_owned(),
            revision: "test".to_owned(),
            commit: "a".repeat(40),
            checkpoint_verification_sha256: "a".repeat(64),
            oracle_manifest_sha256: "a".repeat(64),
            layer: 4,
            page_bytes: 16 * 1024,
            artifact_file: "test.bin".to_owned(),
            artifact_bytes: ARTIFACT_BYTES as u64,
            artifact_sha256: ARTIFACT_SHA256.to_owned(),
            selected_experts: experts,
            tensors,
        }
    }

    #[test]
    fn extents_cover_fixed_complete_experts() {
        let extents = derive_expert_extents(&fixture_manifest()).unwrap();
        assert_eq!(extents.len(), 8);
        assert_eq!(extents[0].bytes, 25_214_976);
        assert_eq!(extents[7].offset + extents[7].bytes, ARTIFACT_BYTES);
    }

    #[test]
    fn extents_reject_projection_reordering() {
        let mut manifest = fixture_manifest();
        manifest.tensors.swap(0, 1);
        assert!(derive_expert_extents(&manifest).is_err());
    }
}
