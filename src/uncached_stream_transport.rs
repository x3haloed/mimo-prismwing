use crate::routed_layer_artifact::{RoutedLayerArtifactTensor, open_routed_layer_artifact};
use crate::text_endpoint::{
    ComponentSafetyMonitor, ProcessActivityDelta, SafetySnapshot, process_activity,
};
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fs::File;
use std::os::fd::AsRawFd;
use std::path::Path;
use std::time::Instant;

const ARTIFACT_SHA256: &str = "fac61c2cfad4b00248c96a52b68360fecd39e2c912e6ffd6643e3f06ade00d21";
const MANIFEST_SHA256: &str = "40179385a571a19b135a4740122744ae3d8ea2c97ef265ac20968296e98822b8";
const PAGE_BYTES: u64 = 16 * 1024;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct AlignedReadPlan {
    physical_offset: u64,
    physical_bytes: usize,
    logical_offset: usize,
    logical_bytes: usize,
}

fn aligned_read_plan(
    offset: u64,
    bytes: u64,
    file_bytes: u64,
    page_bytes: u64,
) -> Result<AlignedReadPlan, String> {
    if bytes == 0 || page_bytes == 0 || !page_bytes.is_power_of_two() {
        return Err("invalid uncached read range or page size".to_owned());
    }
    let logical_end = offset
        .checked_add(bytes)
        .ok_or("uncached logical range overflow")?;
    if logical_end > file_bytes {
        return Err("uncached logical range exceeds EOF".to_owned());
    }
    let physical_offset = offset & !(page_bytes - 1);
    let rounded_end = logical_end
        .checked_add(page_bytes - 1)
        .map(|end| end & !(page_bytes - 1))
        .ok_or("uncached physical range overflow")?
        .min(file_bytes);
    let physical_bytes = usize::try_from(rounded_end - physical_offset)
        .map_err(|_| "uncached physical length does not fit usize")?;
    let logical_offset = usize::try_from(offset - physical_offset)
        .map_err(|_| "uncached logical offset does not fit usize")?;
    let logical_bytes =
        usize::try_from(bytes).map_err(|_| "uncached logical length does not fit usize")?;
    if logical_offset
        .checked_add(logical_bytes)
        .is_none_or(|end| end > physical_bytes)
    {
        return Err("uncached widened range does not contain logical range".to_owned());
    }
    Ok(AlignedReadPlan {
        physical_offset,
        physical_bytes,
        logical_offset,
        logical_bytes,
    })
}

trait PositionalRead {
    fn read_at(&mut self, destination: &mut [u8], offset: u64) -> Result<usize, String>;
}

struct FdReader(libc::c_int);

impl PositionalRead for FdReader {
    fn read_at(&mut self, destination: &mut [u8], offset: u64) -> Result<usize, String> {
        // SAFETY: destination is a valid exclusive slice and pread does not alter
        // the descriptor offset. The checked plan constrains offset to the file.
        let read = unsafe {
            libc::pread(
                self.0,
                destination.as_mut_ptr().cast(),
                destination.len(),
                offset as libc::off_t,
            )
        };
        if read < 0 {
            Err(format!("pread failed: {}", std::io::Error::last_os_error()))
        } else {
            Ok(read as usize)
        }
    }
}

fn read_plan_exact<R: PositionalRead>(
    reader: &mut R,
    plan: AlignedReadPlan,
    destination: &mut [u8],
) -> Result<usize, String> {
    if destination.len() < plan.physical_bytes {
        return Err("uncached destination is smaller than widened range".to_owned());
    }
    let mut filled = 0;
    let mut calls = 0;
    while filled < plan.physical_bytes {
        let read = reader.read_at(
            &mut destination[filled..plan.physical_bytes],
            plan.physical_offset + filled as u64,
        )?;
        calls += 1;
        if read == 0 {
            return Err("uncached read reached EOF before widened range completed".to_owned());
        }
        filled = filled
            .checked_add(read)
            .ok_or("uncached read byte count overflow")?;
    }
    Ok(calls)
}

struct AlignedBuffer {
    pointer: *mut u8,
    capacity: usize,
}

impl AlignedBuffer {
    fn new(capacity: usize) -> Result<Self, String> {
        if capacity == 0 {
            return Err("uncached buffer capacity is zero".to_owned());
        }
        let mut pointer = std::ptr::null_mut();
        // SAFETY: posix_memalign writes one allocation pointer on success.
        let result = unsafe { libc::posix_memalign(&mut pointer, PAGE_BYTES as usize, capacity) };
        if result != 0 || pointer.is_null() {
            return Err(format!("uncached posix_memalign failed with {result}"));
        }
        Ok(Self {
            pointer: pointer.cast(),
            capacity,
        })
    }

    fn bytes_mut(&mut self) -> &mut [u8] {
        // SAFETY: the allocation owns capacity bytes exclusively for self's lifetime.
        unsafe { std::slice::from_raw_parts_mut(self.pointer, self.capacity) }
    }
}

impl Drop for AlignedBuffer {
    fn drop(&mut self) {
        // SAFETY: pointer was returned once by posix_memalign and is freed once.
        unsafe { libc::free(self.pointer.cast()) };
    }
}

#[derive(Debug, Serialize)]
pub struct UncachedTransportTrial {
    pub repetition: usize,
    pub order: usize,
    pub scope: &'static str,
    pub transport: &'static str,
    pub cache_state: &'static str,
    pub transfer_wall_ms: f64,
    pub integrity_wall_ms: f64,
    pub complete_trial_wall_ms: f64,
    pub records: usize,
    pub logical_bytes: u64,
    pub widened_bytes: u64,
    pub read_amplification: f64,
    pub pread_calls: usize,
    pub logical_stream_sha256: String,
    pub activity: ProcessActivityDelta,
}

#[derive(Debug, Serialize)]
pub struct UncachedStreamTransportReport {
    pub schema_version: u32,
    pub semantic: &'static str,
    pub commit: String,
    pub artifact_manifest_sha256: String,
    pub artifact_sha256: String,
    pub checkpoint_root: String,
    pub page_bytes: u64,
    pub f_nocache_value: libc::c_int,
    pub f_rdahead_value: libc::c_int,
    pub nocache_enabled: bool,
    pub automatic_readahead_disabled: bool,
    pub maximum_buffer_bytes: u64,
    pub trials: Vec<UncachedTransportTrial>,
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

fn set_transport_flags(file: &File, nocache: bool) -> Result<(bool, bool), String> {
    let fd = file.as_raw_fd();
    // SAFETY: fcntl receives a live descriptor and documented Darwin integer flags.
    if unsafe { libc::fcntl(fd, libc::F_NOCACHE, libc::c_int::from(nocache)) } == -1 {
        return Err(format!(
            "F_NOCACHE failed: {}",
            std::io::Error::last_os_error()
        ));
    }
    let readahead_disabled = if nocache {
        // F_RDAHEAD=0 is part of the isolated candidate; the control retains the
        // system default rather than changing two policies on both sides.
        if unsafe { libc::fcntl(fd, libc::F_RDAHEAD, 0) } == -1 {
            return Err(format!(
                "F_RDAHEAD failed: {}",
                std::io::Error::last_os_error()
            ));
        }
        true
    } else {
        false
    };
    Ok((nocache, readahead_disabled))
}

fn invalidate_plan(file: &File, plan: AlignedReadPlan) -> Result<(), String> {
    // SAFETY: the file remains live through the mapping, and the plan is page
    // aligned and bounded by its verified file length.
    let mapping = unsafe {
        memmap2::MmapOptions::new()
            .offset(plan.physical_offset)
            .len(plan.physical_bytes)
            .map(file)
    }
    .map_err(|error| format!("cold-range mmap failed: {error}"))?;
    let invalidate = unsafe {
        libc::msync(
            mapping.as_ptr().cast_mut().cast(),
            mapping.len(),
            libc::MS_INVALIDATE,
        )
    };
    if invalidate != 0 {
        return Err(format!(
            "cold-range MS_INVALIDATE failed: {}",
            std::io::Error::last_os_error()
        ));
    }
    let advise = unsafe {
        libc::madvise(
            mapping.as_ptr().cast_mut().cast(),
            mapping.len(),
            libc::MADV_DONTNEED,
        )
    };
    if advise != 0 {
        return Err(format!(
            "cold-range MADV_DONTNEED failed: {}",
            std::io::Error::last_os_error()
        ));
    }
    Ok(())
}

fn expected_stream_sha256(
    artifact: &crate::routed_layer_artifact::RoutedLayerArtifact,
    records: &[RoutedLayerArtifactTensor],
) -> Result<String, String> {
    let mut digest = Sha256::new();
    for record in records {
        digest.update(artifact.record_bytes(record)?);
    }
    Ok(format!("{:x}", digest.finalize()))
}

struct TrialIdentity<'a> {
    checkpoint_root: &'a Path,
    records: &'a [RoutedLayerArtifactTensor],
    expected_sha256: &'a str,
    repetition: usize,
    order: usize,
    scope: &'static str,
    nocache: bool,
}

fn execute_trial(
    identity: TrialIdentity<'_>,
    buffer: &mut AlignedBuffer,
) -> Result<(UncachedTransportTrial, bool, bool), String> {
    let TrialIdentity {
        checkpoint_root,
        records,
        expected_sha256,
        repetition,
        order,
        scope,
        nocache,
    } = identity;
    let mut files = BTreeMap::new();
    let mut plans = Vec::with_capacity(records.len());
    let mut logical_bytes = 0_u64;
    let mut widened_bytes = 0_u64;
    for record in records {
        if !files.contains_key(&record.source_shard) {
            let file = File::open(checkpoint_root.join(&record.source_shard))
                .map_err(|error| format!("{} open: {error}", record.source_shard))?;
            files.insert(record.source_shard.clone(), file);
        }
        let file = files
            .get(&record.source_shard)
            .ok_or("uncached source descriptor disappeared")?;
        let file_bytes = file
            .metadata()
            .map_err(|error| format!("{} metadata: {error}", record.source_shard))?
            .len();
        let bytes = record.source_absolute_offsets[1] - record.source_absolute_offsets[0];
        let plan = aligned_read_plan(
            record.source_absolute_offsets[0],
            bytes,
            file_bytes,
            PAGE_BYTES,
        )?;
        if plan.physical_bytes > buffer.capacity {
            return Err("uncached widened range exceeds declared buffer".to_owned());
        }
        logical_bytes += bytes;
        widened_bytes += plan.physical_bytes as u64;
        plans.push((record, plan));
    }
    for (record, plan) in &plans {
        invalidate_plan(&files[&record.source_shard], *plan)?;
    }
    let mut flags = (nocache, nocache);
    for file in files.values() {
        flags = set_transport_flags(file, nocache)?;
    }
    let activity_before = process_activity()?;
    let trial_started = Instant::now();
    let mut transfer_wall_ms = 0.0;
    let mut integrity_wall_ms = 0.0;
    let mut digest = Sha256::new();
    let mut pread_calls = 0;
    for (record, plan) in plans {
        let file = &files[&record.source_shard];
        let mut reader = FdReader(file.as_raw_fd());
        let transfer_started = Instant::now();
        pread_calls += read_plan_exact(&mut reader, plan, buffer.bytes_mut())?;
        transfer_wall_ms += transfer_started.elapsed().as_secs_f64() * 1000.0;
        let integrity_started = Instant::now();
        let logical_end = plan.logical_offset + plan.logical_bytes;
        let logical = &buffer.bytes_mut()[plan.logical_offset..logical_end];
        if format!("{:x}", Sha256::digest(logical)) != record.source_tensor_sha256 {
            return Err(format!(
                "{}: uncached payload hash mismatch",
                record.artifact_metadata.name
            ));
        }
        digest.update(logical);
        integrity_wall_ms += integrity_started.elapsed().as_secs_f64() * 1000.0;
    }
    let complete_trial_wall_ms = trial_started.elapsed().as_secs_f64() * 1000.0;
    let activity = process_activity()?.checked_delta(activity_before)?;
    let logical_stream_sha256 = format!("{:x}", digest.finalize());
    if logical_stream_sha256 != expected_sha256 {
        return Err("uncached logical stream differs from artifact authority".to_owned());
    }
    Ok((
        UncachedTransportTrial {
            repetition,
            order,
            scope,
            transport: if nocache {
                "f_nocache_pread"
            } else {
                "cacheable_pread_control"
            },
            cache_state: "cold after range MS_INVALIDATE plus MADV_DONTNEED",
            transfer_wall_ms,
            integrity_wall_ms,
            complete_trial_wall_ms,
            records: records.len(),
            logical_bytes,
            widened_bytes,
            read_amplification: widened_bytes as f64 / logical_bytes as f64,
            pread_calls,
            logical_stream_sha256,
            activity,
        },
        flags.0,
        flags.1,
    ))
}

pub fn benchmark_uncached_stream_transport(
    checkpoint_root: &Path,
    artifact_path: &Path,
    artifact_manifest_path: &Path,
    output_path: &Path,
    commit: &str,
) -> Result<UncachedStreamTransportReport, String> {
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
    let artifact = open_routed_layer_artifact(artifact_path, artifact_manifest_path, true)?;
    if artifact.manifest_sha256 != MANIFEST_SHA256
        || artifact.manifest.artifact_sha256 != ARTIFACT_SHA256
        || artifact.manifest.page_bytes as u64 != PAGE_BYTES
        || artifact.manifest.selected_experts.len() != 8
        || artifact.manifest.tensors.len() != 48
    {
        return Err("PW-0213 artifact authority mismatch".to_owned());
    }
    let object_records = artifact.manifest.tensors[..6].to_vec();
    let layer_records = artifact.manifest.tensors.clone();
    let maximum_buffer_bytes = layer_records
        .iter()
        .map(|record| {
            let file_bytes = std::fs::metadata(checkpoint_root.join(&record.source_shard))
                .map_err(|error| error.to_string())?
                .len();
            aligned_read_plan(
                record.source_absolute_offsets[0],
                record.source_absolute_offsets[1] - record.source_absolute_offsets[0],
                file_bytes,
                PAGE_BYTES,
            )
            .map(|plan| plan.physical_bytes)
        })
        .collect::<Result<Vec<_>, String>>()?
        .into_iter()
        .max()
        .ok_or("PW-0213 has no source records")?;
    let mut buffer = AlignedBuffer::new(maximum_buffer_bytes)?;
    safety.checkpoint("uncached_buffer_allocated")?;
    let mut trials = Vec::new();
    let mut nocache_enabled = true;
    let mut automatic_readahead_disabled = true;
    for (scope, records) in [
        ("one_expert_object", object_records.as_slice()),
        ("complete_routed_layer", layer_records.as_slice()),
    ] {
        let expected = expected_stream_sha256(&artifact, records)?;
        let orders = [[false, true], [true, false], [false, true]];
        for (repetition, order) in orders.iter().enumerate() {
            for (position, &nocache) in order.iter().enumerate() {
                let (trial, enabled, disabled) = execute_trial(
                    TrialIdentity {
                        checkpoint_root,
                        records,
                        expected_sha256: &expected,
                        repetition,
                        order: position,
                        scope,
                        nocache,
                    },
                    &mut buffer,
                )?;
                if nocache {
                    nocache_enabled &= enabled;
                    automatic_readahead_disabled &= disabled;
                }
                safety.checkpoint(&format!(
                    "{scope}_repetition_{repetition}_position_{position}"
                ))?;
                trials.push(trial);
            }
        }
    }
    drop(buffer);
    drop(artifact);
    let safety_snapshots = safety.released()?;
    let report = UncachedStreamTransportReport {
        schema_version: 1,
        semantic: "mimo_v2_5_real_checkpoint_page_aligned_f_nocache_transport",
        commit: commit.to_owned(),
        artifact_manifest_sha256: MANIFEST_SHA256.to_owned(),
        artifact_sha256: ARTIFACT_SHA256.to_owned(),
        checkpoint_root: checkpoint_root.display().to_string(),
        page_bytes: PAGE_BYTES,
        f_nocache_value: libc::F_NOCACHE,
        f_rdahead_value: libc::F_RDAHEAD,
        nocache_enabled,
        automatic_readahead_disabled,
        maximum_buffer_bytes: maximum_buffer_bytes as u64,
        trials,
        safety_snapshots,
        batch_size: 1,
        concurrency: 1,
        accepted_tokens: 0,
        accepted_per_verification: 0,
        unique_experts: 8,
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

    struct ScriptedReader {
        source: Vec<u8>,
        maximum_read: usize,
        fail: bool,
    }

    impl PositionalRead for ScriptedReader {
        fn read_at(&mut self, destination: &mut [u8], offset: u64) -> Result<usize, String> {
            if self.fail {
                return Err("injected I/O failure".to_owned());
            }
            let offset = offset as usize;
            if offset >= self.source.len() {
                return Ok(0);
            }
            let count = destination
                .len()
                .min(self.maximum_read)
                .min(self.source.len() - offset);
            destination[..count].copy_from_slice(&self.source[offset..offset + count]);
            Ok(count)
        }
    }

    #[test]
    fn planning_covers_unaligned_adjacent_and_duplicate_ranges() {
        let a = aligned_read_plan(3, 5, 32, 4).unwrap();
        let b = aligned_read_plan(8, 4, 32, 4).unwrap();
        let duplicate = aligned_read_plan(3, 5, 32, 4).unwrap();
        assert_eq!(
            a,
            AlignedReadPlan {
                physical_offset: 0,
                physical_bytes: 8,
                logical_offset: 3,
                logical_bytes: 5
            }
        );
        assert_eq!(b.physical_offset, 8);
        assert_eq!(a, duplicate);
    }

    #[test]
    fn exact_reader_retries_short_reads() {
        let plan = aligned_read_plan(3, 5, 16, 4).unwrap();
        let mut reader = ScriptedReader {
            source: (0..16).collect(),
            maximum_read: 3,
            fail: false,
        };
        let mut destination = vec![0; 8];
        assert_eq!(
            read_plan_exact(&mut reader, plan, &mut destination).unwrap(),
            3
        );
        assert_eq!(&destination[3..8], &[3, 4, 5, 6, 7]);
    }

    #[test]
    fn planning_and_reader_fail_closed_at_eof_and_io_error() {
        assert!(aligned_read_plan(14, 3, 16, 4).is_err());
        let plan = aligned_read_plan(8, 4, 16, 4).unwrap();
        let mut eof = ScriptedReader {
            source: vec![0; 8],
            maximum_read: 4,
            fail: false,
        };
        let mut destination = vec![0; 4];
        assert!(read_plan_exact(&mut eof, plan, &mut destination).is_err());
        let mut failure = ScriptedReader {
            source: vec![0; 16],
            maximum_read: 4,
            fail: true,
        };
        assert!(read_plan_exact(&mut failure, plan, &mut destination).is_err());
    }
}
