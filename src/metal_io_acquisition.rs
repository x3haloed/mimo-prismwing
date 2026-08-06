use crate::routed_layer_artifact::{
    RoutedLayerArtifact, RoutedLayerArtifactTensor, open_routed_layer_artifact,
};
use crate::text_endpoint::{
    ComponentSafetyMonitor, ProcessActivityDelta, SafetySnapshot, process_activity,
};
use foreign_types::ForeignType;
use metal::{Device, MTLResourceOptions};
use objc::rc::StrongPtr;
use objc::runtime::Object;
use objc::{class, msg_send, sel, sel_impl};
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::path::Path;
use std::time::Instant;

const IO_STATUS_COMPLETE: isize = 3;
const IO_QUEUE_CONCURRENT: isize = 0;
const IO_QUEUE_SERIAL: isize = 1;
const IO_PRIORITY_NORMAL: isize = 1;
const NS_UTF8_STRING_ENCODING: usize = 4;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct TransferRange {
    record_index: usize,
    source_offset: usize,
    destination_offset: usize,
    bytes: usize,
}

#[derive(Debug, Serialize)]
pub struct MetalIoTrial {
    pub repetition: usize,
    pub cache_state: &'static str,
    pub command_buffers: usize,
    pub queue_create_ms: f64,
    pub file_handle_create_ms: f64,
    pub destination_allocate_ms: f64,
    pub destination_initialize_ms: f64,
    pub command_buffer_create_ms: f64,
    pub command_encode_ms: f64,
    pub commit_call_ms: f64,
    pub synchronous_wait_ms: f64,
    pub transfer_wall_ms: f64,
    pub integrity_ms: f64,
    pub explicit_release_ms: f64,
    pub encoded_records: usize,
    pub encoded_bytes: u64,
    pub destination_capacity_bytes: u64,
    pub complete_statuses: Vec<isize>,
    pub destination_records_sha256: String,
    pub activity: ProcessActivityDelta,
}

#[derive(Debug, Serialize)]
pub struct MetalIoAcquisitionReport {
    pub schema_version: u32,
    pub semantic: &'static str,
    pub commit: String,
    pub artifact_manifest_sha256: String,
    pub artifact_sha256: String,
    pub artifact_bytes: u64,
    pub encoded_bytes_per_trial: u64,
    pub record_count: usize,
    pub metal_device: String,
    pub selector_probe_passed: bool,
    pub exact_offset_probe_passed: bool,
    pub exact_offset_probe_ms: f64,
    pub warm_prefault_ms: f64,
    pub warm_prefault_checksum: u64,
    pub trials: Vec<MetalIoTrial>,
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

struct MetalIoRuntime {
    device: Device,
}

struct QueueObjects {
    descriptor: StrongPtr,
    queue: StrongPtr,
}

fn partition_ranges(
    records: &[RoutedLayerArtifactTensor],
    command_buffers: usize,
    artifact_bytes: usize,
) -> Result<Vec<Vec<TransferRange>>, String> {
    if records.is_empty() || !(1..=records.len()).contains(&command_buffers) {
        return Err("invalid Metal I/O partition dimensions".to_owned());
    }
    let mut ranges = Vec::with_capacity(records.len());
    let mut encoded_bytes = 0_usize;
    let mut previous_end = 0_usize;
    for (record_index, record) in records.iter().enumerate() {
        let source_offset = usize::try_from(record.artifact_metadata.data_offsets[0])
            .map_err(|_| "Metal I/O source offset does not fit usize")?;
        let bytes = usize::try_from(record.artifact_metadata.data_bytes)
            .map_err(|_| "Metal I/O byte count does not fit usize")?;
        let end = source_offset
            .checked_add(bytes)
            .ok_or("Metal I/O transfer range overflow")?;
        if bytes == 0 || end > artifact_bytes {
            return Err("Metal I/O transfer range exceeds artifact".to_owned());
        }
        if source_offset < previous_end {
            return Err("Metal I/O transfer ranges overlap or reorder".to_owned());
        }
        previous_end = end;
        encoded_bytes = encoded_bytes
            .checked_add(bytes)
            .ok_or("Metal I/O encoded byte total overflow")?;
        ranges.push(TransferRange {
            record_index,
            source_offset,
            destination_offset: source_offset,
            bytes,
        });
    }
    let base = records.len() / command_buffers;
    let remainder = records.len() % command_buffers;
    let mut partitions = Vec::with_capacity(command_buffers);
    let mut cursor = 0;
    for partition in 0..command_buffers {
        let length = base + usize::from(partition < remainder);
        partitions.push(ranges[cursor..cursor + length].to_vec());
        cursor += length;
    }
    if cursor != records.len()
        || partitions.iter().flatten().count() != records.len()
        || encoded_bytes
            != records
                .iter()
                .map(|record| record.artifact_metadata.data_bytes as usize)
                .sum::<usize>()
    {
        return Err("Metal I/O partition accounting mismatch".to_owned());
    }
    Ok(partitions)
}

fn object_error(error: *mut Object, context: &str) -> String {
    if error.is_null() {
        return format!("{context}: Objective-C API returned nil without NSError");
    }
    // SAFETY: NSError owns an NSString localizedDescription and UTF8String is
    // valid for the duration of this call.
    unsafe {
        let description: *mut Object = msg_send![error, localizedDescription];
        let bytes: *const std::ffi::c_char = msg_send![description, UTF8String];
        if bytes.is_null() {
            format!("{context}: NSError lacks UTF-8 description")
        } else {
            format!(
                "{context}: {}",
                std::ffi::CStr::from_ptr(bytes).to_string_lossy()
            )
        }
    }
}

fn ns_file_url(path: &Path) -> Result<StrongPtr, String> {
    let path = path
        .to_str()
        .ok_or("Metal I/O artifact path is not UTF-8")?;
    // SAFETY: the Objective-C initializers are checked for nil and StrongPtr
    // preserves both objects through URL construction.
    unsafe {
        let allocated: *mut Object = msg_send![class!(NSString), alloc];
        let string: *mut Object = msg_send![allocated,
            initWithBytes:path.as_ptr()
            length:path.len()
            encoding:NS_UTF8_STRING_ENCODING
        ];
        if string.is_null() {
            return Err("failed to construct Metal I/O NSString path".to_owned());
        }
        let string = StrongPtr::new(string);
        let allocated_url: *mut Object = msg_send![class!(NSURL), alloc];
        let url: *mut Object = msg_send![allocated_url, initFileURLWithPath:*string];
        if url.is_null() {
            return Err("failed to construct Metal I/O file URL".to_owned());
        }
        Ok(StrongPtr::new(url))
    }
}

impl MetalIoRuntime {
    fn new() -> Result<Self, String> {
        let device = Device::system_default().ok_or("no Metal device is available")?;
        let object = device.as_ptr().cast::<Object>();
        // SAFETY: respondsToSelector is valid for every Objective-C object.
        let queue_available: bool = unsafe {
            msg_send![object, respondsToSelector:sel!(newIOCommandQueueWithDescriptor:error:)]
        };
        let file_available: bool =
            unsafe { msg_send![object, respondsToSelector:sel!(newIOFileHandleWithURL:error:)] };
        if !queue_available || !file_available {
            return Err("Metal I/O selectors are unavailable on this device".to_owned());
        }
        Ok(Self { device })
    }

    fn queue(&self, command_buffers: usize) -> Result<QueueObjects, String> {
        // SAFETY: all selector arguments match the installed Metal SDK. New
        // methods return +1 objects captured by StrongPtr.
        unsafe {
            let descriptor: *mut Object = msg_send![class!(MTLIOCommandQueueDescriptor), new];
            if descriptor.is_null() {
                return Err("failed to create Metal I/O queue descriptor".to_owned());
            }
            let descriptor = StrongPtr::new(descriptor);
            let _: () = msg_send![*descriptor, setMaxCommandBufferCount:command_buffers];
            let _: () = msg_send![*descriptor, setMaxCommandsInFlight:command_buffers];
            let _: () = msg_send![*descriptor, setPriority:IO_PRIORITY_NORMAL];
            let queue_type = if command_buffers == 1 {
                IO_QUEUE_SERIAL
            } else {
                IO_QUEUE_CONCURRENT
            };
            let _: () = msg_send![*descriptor, setType:queue_type];
            let mut error: *mut Object = std::ptr::null_mut();
            let queue: *mut Object = msg_send![self.device.as_ptr().cast::<Object>(),
                newIOCommandQueueWithDescriptor:*descriptor error:&mut error
            ];
            if queue.is_null() {
                return Err(object_error(error, "Metal I/O queue creation"));
            }
            Ok(QueueObjects {
                descriptor,
                queue: StrongPtr::new(queue),
            })
        }
    }

    fn file_handle(&self, artifact_path: &Path) -> Result<(StrongPtr, StrongPtr), String> {
        let url = ns_file_url(artifact_path)?;
        // SAFETY: selector availability was checked at construction and the URL
        // remains retained beside the returned file handle.
        unsafe {
            let mut error: *mut Object = std::ptr::null_mut();
            let handle: *mut Object = msg_send![self.device.as_ptr().cast::<Object>(),
                newIOFileHandleWithURL:*url error:&mut error
            ];
            if handle.is_null() {
                return Err(object_error(error, "Metal I/O file-handle creation"));
            }
            Ok((url, StrongPtr::new(handle)))
        }
    }

    fn execute(
        &self,
        repetition: usize,
        cache_state: &'static str,
        artifact_path: &Path,
        artifact: &RoutedLayerArtifact,
        command_buffers: usize,
    ) -> Result<MetalIoTrial, String> {
        let partitions = partition_ranges(
            &artifact.manifest.tensors,
            command_buffers,
            artifact.manifest.artifact_bytes as usize,
        )?;
        let queue_started = Instant::now();
        let queue = self.queue(command_buffers)?;
        let queue_create_ms = queue_started.elapsed().as_secs_f64() * 1000.0;
        let handle_started = Instant::now();
        let (url, file_handle) = self.file_handle(artifact_path)?;
        let file_handle_create_ms = handle_started.elapsed().as_secs_f64() * 1000.0;
        let allocate_started = Instant::now();
        let destination = self.device.new_buffer(
            artifact.manifest.artifact_bytes,
            MTLResourceOptions::StorageModeShared,
        );
        let destination_allocate_ms = allocate_started.elapsed().as_secs_f64() * 1000.0;
        let initialize_started = Instant::now();
        // SAFETY: the shared buffer exposes artifact_bytes writable bytes.
        unsafe {
            std::ptr::write_bytes(
                destination.contents().cast::<u8>(),
                0xa5,
                artifact.manifest.artifact_bytes as usize,
            );
        }
        let destination_initialize_ms = initialize_started.elapsed().as_secs_f64() * 1000.0;
        let command_started = Instant::now();
        let mut commands = Vec::with_capacity(command_buffers);
        for _ in 0..command_buffers {
            // SAFETY: commandBuffer returns a valid autoreleased object owned by
            // the live queue; retain it through completion.
            let command: *mut Object = unsafe { msg_send![*queue.queue, commandBuffer] };
            if command.is_null() {
                return Err("Metal I/O queue returned a nil command buffer".to_owned());
            }
            commands.push(unsafe { StrongPtr::retain(command) });
        }
        let command_buffer_create_ms = command_started.elapsed().as_secs_f64() * 1000.0;
        let encode_started = Instant::now();
        for (command, ranges) in commands.iter().zip(&partitions) {
            for range in ranges {
                // SAFETY: manifest validation and partition_ranges prove all
                // source and destination ranges are in bounds and nonoverlapping.
                unsafe {
                    let _: () = msg_send![**command,
                        loadBuffer:destination.as_ptr().cast::<Object>()
                        offset:range.destination_offset
                        size:range.bytes
                        sourceHandle:*file_handle
                        sourceHandleOffset:range.source_offset
                    ];
                }
            }
        }
        let command_encode_ms = encode_started.elapsed().as_secs_f64() * 1000.0;
        let activity_before = process_activity()?;
        let transfer_started = Instant::now();
        let commit_started = Instant::now();
        for command in &commands {
            unsafe {
                let _: () = msg_send![**command, commit];
            }
        }
        let commit_call_ms = commit_started.elapsed().as_secs_f64() * 1000.0;
        let wait_started = Instant::now();
        for command in &commands {
            unsafe {
                let _: () = msg_send![**command, waitUntilCompleted];
            }
        }
        let synchronous_wait_ms = wait_started.elapsed().as_secs_f64() * 1000.0;
        let transfer_wall_ms = transfer_started.elapsed().as_secs_f64() * 1000.0;
        let activity = process_activity()?.checked_delta(activity_before)?;
        let mut complete_statuses = Vec::with_capacity(commands.len());
        for command in &commands {
            let status: isize = unsafe { msg_send![**command, status] };
            complete_statuses.push(status);
            if status != IO_STATUS_COMPLETE {
                let error: *mut Object = unsafe { msg_send![**command, error] };
                return Err(object_error(error, &format!("Metal I/O status {status}")));
            }
        }
        let integrity_started = Instant::now();
        let mut digest = Sha256::new();
        for range in partitions.iter().flatten() {
            let record = &artifact.manifest.tensors[range.record_index];
            let expected = artifact.record_bytes(record)?;
            // SAFETY: the command is complete and the range lies within the
            // live shared destination buffer.
            let actual = unsafe {
                std::slice::from_raw_parts(
                    destination
                        .contents()
                        .cast::<u8>()
                        .add(range.destination_offset),
                    range.bytes,
                )
            };
            if actual != expected {
                return Err(format!(
                    "{}: Metal I/O destination bytes mismatch",
                    record.artifact_metadata.name
                ));
            }
            digest.update(actual);
        }
        let destination_records_sha256 = digest
            .finalize()
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect();
        let integrity_ms = integrity_started.elapsed().as_secs_f64() * 1000.0;
        let encoded_bytes = partitions
            .iter()
            .flatten()
            .map(|range| range.bytes as u64)
            .sum();
        let encoded_records = partitions.iter().flatten().count();
        let release_started = Instant::now();
        drop(commands);
        drop(destination);
        drop(file_handle);
        drop(url);
        drop(queue.queue);
        drop(queue.descriptor);
        let explicit_release_ms = release_started.elapsed().as_secs_f64() * 1000.0;
        Ok(MetalIoTrial {
            repetition,
            cache_state,
            command_buffers,
            queue_create_ms,
            file_handle_create_ms,
            destination_allocate_ms,
            destination_initialize_ms,
            command_buffer_create_ms,
            command_encode_ms,
            commit_call_ms,
            synchronous_wait_ms,
            transfer_wall_ms,
            integrity_ms,
            explicit_release_ms,
            encoded_records,
            encoded_bytes,
            destination_capacity_bytes: artifact.manifest.artifact_bytes,
            complete_statuses,
            destination_records_sha256,
            activity,
        })
    }

    fn exact_offset_probe(
        &self,
        artifact_path: &Path,
        artifact: &RoutedLayerArtifact,
    ) -> Result<f64, String> {
        let records = &artifact.manifest.tensors;
        if records.len() < 2 {
            return Err("Metal I/O probe needs two records".to_owned());
        }
        let queue = self.queue(2)?;
        let (_url, handle) = self.file_handle(artifact_path)?;
        let output = self
            .device
            .new_buffer(8192, MTLResourceOptions::StorageModeShared);
        let started = Instant::now();
        let mut commands = Vec::new();
        for (slot, record) in records.iter().skip(1).take(2).enumerate() {
            let source = record.artifact_metadata.data_offsets[0] as usize;
            let bytes = usize::try_from(record.artifact_metadata.data_bytes)
                .map_err(|_| "probe byte count does not fit usize")?
                .min(4096);
            let command: *mut Object = unsafe { msg_send![*queue.queue, commandBuffer] };
            if command.is_null() || bytes == 0 {
                return Err("Metal I/O exact-offset probe setup failed".to_owned());
            }
            let command = unsafe { StrongPtr::retain(command) };
            unsafe {
                let _: () = msg_send![*command,
                    loadBuffer:output.as_ptr().cast::<Object>()
                    offset:slot * 4096
                    size:bytes
                    sourceHandle:*handle
                    sourceHandleOffset:source
                ];
                let _: () = msg_send![*command, commit];
            }
            commands.push((command, record, slot, bytes));
        }
        for (command, _, _, _) in &commands {
            unsafe {
                let _: () = msg_send![**command, waitUntilCompleted];
            }
            let status: isize = unsafe { msg_send![**command, status] };
            if status != IO_STATUS_COMPLETE {
                return Err(format!("Metal I/O exact-offset probe status {status}"));
            }
        }
        for (_, record, slot, bytes) in &commands {
            let expected = &artifact.record_bytes(record)?[..*bytes];
            let actual = unsafe {
                std::slice::from_raw_parts(output.contents().cast::<u8>().add(slot * 4096), *bytes)
            };
            if actual != expected {
                return Err("Metal I/O exact-offset probe mismatch".to_owned());
            }
        }
        Ok(started.elapsed().as_secs_f64() * 1000.0)
    }
}

pub fn benchmark_metal_io_acquisition(
    artifact_path: &Path,
    artifact_manifest_path: &Path,
    output_path: &Path,
    commit: &str,
) -> Result<MetalIoAcquisitionReport, String> {
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
    if artifact.manifest.artifact_bytes != 201_719_808
        || artifact.manifest.tensors.len() != 48
        || artifact
            .manifest
            .tensors
            .iter()
            .map(|record| record.artifact_metadata.data_bytes)
            .sum::<u64>()
            != 201_375_744
    {
        return Err("Metal I/O artifact authority mismatch".to_owned());
    }
    safety.checkpoint("metal_io_artifact_verified")?;
    let runtime = MetalIoRuntime::new()?;
    let exact_offset_probe_ms = runtime.exact_offset_probe(artifact_path, &artifact)?;
    artifact.invalidate_pages()?;
    safety.checkpoint("metal_io_exact_offset_probe_released")?;
    let mut trials = Vec::with_capacity(18);
    let mut warm_prefault_ms = 0.0;
    let mut warm_prefault_checksum = 0_u64;
    for &cache_state in &["cold", "warm"] {
        if cache_state == "warm" {
            let started = Instant::now();
            warm_prefault_checksum = artifact.prefault_pages();
            warm_prefault_ms = started.elapsed().as_secs_f64() * 1000.0;
            safety.checkpoint("metal_io_warm_prefault_complete")?;
        }
        for repetition in 0..3 {
            let order = if repetition % 2 == 0 {
                [1_usize, 2, 3]
            } else {
                [3_usize, 2, 1]
            };
            for command_buffers in order {
                if cache_state == "cold" {
                    artifact.invalidate_pages()?;
                }
                let trial = runtime.execute(
                    repetition,
                    cache_state,
                    artifact_path,
                    &artifact,
                    command_buffers,
                )?;
                if trial.encoded_records != 48
                    || trial.encoded_bytes != 201_375_744
                    || trial.destination_capacity_bytes != 201_719_808
                    || trial.complete_statuses != vec![IO_STATUS_COMPLETE; command_buffers]
                {
                    return Err("Metal I/O trial accounting mismatch".to_owned());
                }
                trials.push(trial);
                safety.checkpoint(&format!(
                    "metal_io_{cache_state}_repetition_{repetition}_commands_{command_buffers}_released"
                ))?;
            }
        }
    }
    let hashes = trials
        .iter()
        .map(|trial| trial.destination_records_sha256.as_str())
        .collect::<BTreeSet<_>>();
    if hashes.len() != 1 {
        return Err("Metal I/O configurations produced different record bytes".to_owned());
    }
    let safety_snapshots = safety.released()?;
    let report = MetalIoAcquisitionReport {
        schema_version: 1,
        semantic: "mimo_v2_5_layer4_metal_io_acquisition_bound",
        commit: commit.to_owned(),
        artifact_manifest_sha256: artifact.manifest_sha256.clone(),
        artifact_sha256: artifact.manifest.artifact_sha256.clone(),
        artifact_bytes: artifact.manifest.artifact_bytes,
        encoded_bytes_per_trial: 201_375_744,
        record_count: 48,
        metal_device: runtime.device.name().to_owned(),
        selector_probe_passed: true,
        exact_offset_probe_passed: true,
        exact_offset_probe_ms,
        warm_prefault_ms,
        warm_prefault_checksum,
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
    use crate::MappedTensorMetadata;

    fn record(index: usize, start: u64, bytes: u64) -> RoutedLayerArtifactTensor {
        RoutedLayerArtifactTensor {
            expert: index as u32,
            projection: "gate".to_owned(),
            role: "weight".to_owned(),
            source_shard: "source.safetensors".to_owned(),
            source_shard_sha256: "a".repeat(64),
            source_absolute_offsets: [start, start + bytes],
            source_tensor_sha256: "b".repeat(64),
            artifact_metadata: MappedTensorMetadata {
                name: format!("tensor-{index}"),
                dtype: "F8_E4M3".to_owned(),
                shape: vec![bytes],
                data_offsets: [start, start + bytes],
                data_bytes: bytes,
            },
            artifact_region_bytes: bytes,
            artifact_tensor_sha256: "b".repeat(64),
        }
    }

    #[test]
    fn partitions_are_contiguous_balanced_and_preserve_exact_slots() {
        let records = (0..5)
            .map(|index| record(index, (index * 16) as u64, 4))
            .collect::<Vec<_>>();
        let partitions = partition_ranges(&records, 3, 80).expect("valid partition");
        assert_eq!(
            partitions.iter().map(Vec::len).collect::<Vec<_>>(),
            [2, 2, 1]
        );
        assert_eq!(
            partitions
                .iter()
                .flatten()
                .map(|range| (
                    range.record_index,
                    range.source_offset,
                    range.destination_offset
                ))
                .collect::<Vec<_>>(),
            [
                (0, 0, 0),
                (1, 16, 16),
                (2, 32, 32),
                (3, 48, 48),
                (4, 64, 64)
            ]
        );
    }

    #[test]
    fn partitions_fail_closed_on_overlap_bounds_and_bad_counts() {
        let valid = vec![record(0, 0, 8), record(1, 16, 8)];
        assert!(partition_ranges(&valid, 0, 32).is_err());
        assert!(partition_ranges(&valid, 3, 32).is_err());
        assert!(partition_ranges(&valid, 1, 20).is_err());
        let overlap = vec![record(0, 0, 8), record(1, 4, 8)];
        assert!(partition_ranges(&overlap, 1, 32).is_err());
    }
}
