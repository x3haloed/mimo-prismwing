//! Fail-closed ownership and Darwin pressure handling for declared residency.
//!
//! This module deliberately does not raise the runtime safety ceiling.  It is
//! the independently testable safety substrate that must be connected to a
//! real declared cache before the conditional high-residency mode can exist.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};
use std::ffi::{c_char, c_ulong, c_void};
use std::fs;
use std::path::Path;
use std::sync::{Arc, Mutex};

pub const MAXIMUM_DECLARED_RESIDENCY_BYTES: u64 = 12 * 1024 * 1024 * 1024;
pub const PRESSURE_NORMAL: u64 = 1;
pub const PRESSURE_WARNING: u64 = 2;
pub const PRESSURE_CRITICAL: u64 = 4;

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct DeclaredResidentObject {
    pub identity: String,
    pub bytes: u64,
    pub tensor_metadata_sha256: String,
    pub lifetime: String,
    pub warning_eviction_order: u64,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct DeclaredResidencyManifest {
    pub capacity_bytes: u64,
    pub selected_bytes: u64,
    pub unallocated_bytes: u64,
    pub persistent_lifetime: String,
    pub warning_eviction_order: String,
    pub critical_pressure_action: String,
    pub objects: Vec<DeclaredResidentObject>,
}

impl DeclaredResidencyManifest {
    pub fn from_offline_report(path: &Path) -> Result<Self, String> {
        let bytes = fs::read(path)
            .map_err(|error| format!("failed to read {}: {error}", path.display()))?;
        let report: Value = serde_json::from_slice(&bytes)
            .map_err(|error| format!("invalid JSON in {}: {error}", path.display()))?;
        let manifest = report
            .get("residency_manifest")
            .ok_or_else(|| "offline report lacks residency_manifest".to_owned())?;
        let parsed: Self = serde_json::from_value(manifest.clone())
            .map_err(|error| format!("invalid residency_manifest: {error}"))?;
        parsed.validate()?;
        Ok(parsed)
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.capacity_bytes != MAXIMUM_DECLARED_RESIDENCY_BYTES {
            return Err(format!(
                "declared capacity must equal {MAXIMUM_DECLARED_RESIDENCY_BYTES}, got {}",
                self.capacity_bytes
            ));
        }
        let selected = self.objects.iter().try_fold(0_u64, |total, object| {
            total
                .checked_add(object.bytes)
                .ok_or_else(|| "declared object byte sum overflowed".to_owned())
        })?;
        if selected != self.selected_bytes {
            return Err(format!(
                "declared object bytes {selected} do not match selected_bytes {}",
                self.selected_bytes
            ));
        }
        if self.selected_bytes.checked_add(self.unallocated_bytes) != Some(self.capacity_bytes) {
            return Err("selected_bytes + unallocated_bytes must equal capacity_bytes".to_owned());
        }
        let expected_lifetime = canonical_lifetime(&self.persistent_lifetime);
        let mut identities = BTreeSet::new();
        let mut orders = BTreeSet::new();
        for object in &self.objects {
            if object.identity.is_empty() || !identities.insert(object.identity.as_str()) {
                return Err(format!(
                    "empty or duplicate resident identity: {}",
                    object.identity
                ));
            }
            if object.bytes == 0 {
                return Err(format!("zero-byte resident object: {}", object.identity));
            }
            if object.tensor_metadata_sha256.len() != 64
                || !object
                    .tensor_metadata_sha256
                    .bytes()
                    .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
            {
                return Err(format!("invalid lowercase SHA-256 for {}", object.identity));
            }
            if canonical_lifetime(&object.lifetime) != expected_lifetime {
                return Err(format!("lifetime mismatch for {}", object.identity));
            }
            if !orders.insert(object.warning_eviction_order) {
                return Err(format!(
                    "duplicate warning eviction order {}",
                    object.warning_eviction_order
                ));
            }
        }
        let expected_orders: BTreeSet<u64> = (1..=self.objects.len() as u64).collect();
        if orders != expected_orders {
            return Err("warning eviction orders must be contiguous from 1".to_owned());
        }
        if !self
            .warning_eviction_order
            .starts_with("ascending warning_eviction_order")
        {
            return Err("manifest must declare ascending warning eviction".to_owned());
        }
        if !self.critical_pressure_action.starts_with("stop") {
            return Err("critical pressure action must be stop".to_owned());
        }
        Ok(())
    }
}

fn canonical_lifetime(value: &str) -> String {
    value
        .replace('_', " ")
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

struct InstalledResident {
    identity: String,
    bytes: u64,
    warning_eviction_order: u64,
    payload: Box<dyn Send>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct PressureEvent {
    pub mask: u64,
    pub resident_bytes_before: u64,
    pub resident_bytes_after: u64,
    pub evicted_identities: Vec<String>,
    pub growth_stopped: bool,
}

struct ResidencyState {
    installed: BTreeMap<String, InstalledResident>,
    resident_bytes: u64,
    growth_stopped: bool,
    events: Vec<PressureEvent>,
}

pub struct PressureResidencyController {
    manifest: DeclaredResidencyManifest,
    declarations: BTreeMap<String, DeclaredResidentObject>,
    state: Mutex<ResidencyState>,
}

impl PressureResidencyController {
    pub fn new(manifest: DeclaredResidencyManifest) -> Result<Arc<Self>, String> {
        manifest.validate()?;
        let declarations = manifest
            .objects
            .iter()
            .cloned()
            .map(|object| (object.identity.clone(), object))
            .collect();
        Ok(Arc::new(Self {
            manifest,
            declarations,
            state: Mutex::new(ResidencyState {
                installed: BTreeMap::new(),
                resident_bytes: 0,
                growth_stopped: false,
                events: Vec::new(),
            }),
        }))
    }

    pub fn manifest(&self) -> &DeclaredResidencyManifest {
        &self.manifest
    }

    pub fn install<T: Send + 'static>(
        &self,
        identity: &str,
        actual_bytes: u64,
        actual_tensor_metadata_sha256: &str,
        payload: T,
    ) -> Result<(), String> {
        let declaration = self
            .declarations
            .get(identity)
            .ok_or_else(|| format!("resident object is not declared: {identity}"))?;
        if declaration.bytes != actual_bytes {
            return Err(format!(
                "resident byte mismatch for {identity}: declared {}, actual {actual_bytes}",
                declaration.bytes
            ));
        }
        if declaration.tensor_metadata_sha256 != actual_tensor_metadata_sha256 {
            return Err(format!("resident metadata hash mismatch for {identity}"));
        }
        let mut state = self
            .state
            .lock()
            .map_err(|_| "residency lock poisoned".to_owned())?;
        if state.growth_stopped {
            return Err("resident growth stopped by critical memory pressure".to_owned());
        }
        if state.installed.contains_key(identity) {
            return Err(format!("resident object already installed: {identity}"));
        }
        let next_bytes = state
            .resident_bytes
            .checked_add(actual_bytes)
            .ok_or_else(|| "resident byte count overflowed".to_owned())?;
        if next_bytes > self.manifest.selected_bytes {
            return Err("installed resident bytes exceed declared selected_bytes".to_owned());
        }
        state.installed.insert(
            identity.to_owned(),
            InstalledResident {
                identity: identity.to_owned(),
                bytes: actual_bytes,
                warning_eviction_order: declaration.warning_eviction_order,
                payload: Box::new(payload),
            },
        );
        state.resident_bytes = next_bytes;
        Ok(())
    }

    pub fn resident_bytes(&self) -> Result<u64, String> {
        Ok(self
            .state
            .lock()
            .map_err(|_| "residency lock poisoned".to_owned())?
            .resident_bytes)
    }

    pub fn events(&self) -> Result<Vec<PressureEvent>, String> {
        Ok(self
            .state
            .lock()
            .map_err(|_| "residency lock poisoned".to_owned())?
            .events
            .clone())
    }

    pub fn growth_stopped(&self) -> Result<bool, String> {
        Ok(self
            .state
            .lock()
            .map_err(|_| "residency lock poisoned".to_owned())?
            .growth_stopped)
    }

    pub fn handle_pressure_mask(&self, mask: u64) -> Result<(), String> {
        let invalid =
            mask & !(PRESSURE_NORMAL | PRESSURE_WARNING | PRESSURE_CRITICAL) != 0 || mask == 0;
        // An event outside the documented Darwin mask is itself a safety
        // failure: evict and stop growth before reporting it.
        let effective_mask = if invalid { PRESSURE_CRITICAL } else { mask };
        let must_evict = effective_mask & (PRESSURE_WARNING | PRESSURE_CRITICAL) != 0;
        let critical = effective_mask & PRESSURE_CRITICAL != 0;
        let (mut removed, event) = {
            let mut state = self
                .state
                .lock()
                .map_err(|_| "residency lock poisoned".to_owned())?;
            let before = state.resident_bytes;
            if critical {
                state.growth_stopped = true;
            }
            let mut removed = Vec::new();
            if must_evict {
                removed.extend(std::mem::take(&mut state.installed).into_values());
                removed.sort_by_key(|object| object.warning_eviction_order);
                state.resident_bytes = 0;
            }
            let event = PressureEvent {
                mask: effective_mask,
                resident_bytes_before: before,
                resident_bytes_after: state.resident_bytes,
                evicted_identities: removed
                    .iter()
                    .map(|object| object.identity.clone())
                    .collect(),
                growth_stopped: state.growth_stopped,
            };
            state.events.push(event.clone());
            (removed, event)
        };
        // Payload destruction is outside the controller lock so destructors cannot deadlock it.
        for object in removed.drain(..) {
            debug_assert!(object.bytes > 0);
            drop(object.payload);
        }
        if must_evict {
            debug_assert_eq!(event.resident_bytes_after, 0);
        }
        if invalid {
            Err(format!("unknown Darwin pressure mask: {mask}"))
        } else {
            Ok(())
        }
    }
}

#[repr(C)]
struct DispatchObject {
    _private: [u8; 0],
}

type DispatchSource = *mut DispatchObject;
type DispatchQueue = *mut DispatchObject;

unsafe extern "C" {
    static _dispatch_source_type_memorypressure: c_char;
    fn dispatch_queue_create(label: *const c_char, attributes: *const c_void) -> DispatchQueue;
    fn dispatch_source_create(
        source_type: *const c_void,
        handle: usize,
        mask: c_ulong,
        queue: DispatchQueue,
    ) -> DispatchSource;
    fn dispatch_set_context(object: DispatchSource, context: *mut c_void);
    fn dispatch_source_set_event_handler_f(
        source: DispatchSource,
        handler: unsafe extern "C" fn(*mut c_void),
    );
    fn dispatch_source_get_data(source: DispatchSource) -> c_ulong;
    fn dispatch_activate(object: DispatchSource);
    fn dispatch_source_cancel(source: DispatchSource);
    fn dispatch_sync_f(
        queue: DispatchQueue,
        context: *mut c_void,
        work: unsafe extern "C" fn(*mut c_void),
    );
    fn dispatch_release(object: *mut c_void);
}

struct ObserverContext {
    controller: Arc<PressureResidencyController>,
    source: DispatchSource,
}

pub struct DarwinMemoryPressureObserver {
    source: DispatchSource,
    queue: DispatchQueue,
    context: *mut ObserverContext,
}

unsafe extern "C" fn pressure_event(context: *mut c_void) {
    // SAFETY: context remains owned by the observer until after cancel + queue drain.
    let context = unsafe { &*(context.cast::<ObserverContext>()) };
    let mask = unsafe { dispatch_source_get_data(context.source) } as u64;
    let _ = context.controller.handle_pressure_mask(mask);
}

unsafe extern "C" fn drain_marker(_context: *mut c_void) {}

impl DarwinMemoryPressureObserver {
    pub fn start(controller: Arc<PressureResidencyController>) -> Result<Self, String> {
        static QUEUE_LABEL: &[u8] = b"org.prismwing.memory-pressure\0";
        let queue = unsafe { dispatch_queue_create(QUEUE_LABEL.as_ptr().cast(), std::ptr::null()) };
        if queue.is_null() {
            return Err("failed to create Darwin memory-pressure queue".to_owned());
        }
        let source_type = std::ptr::addr_of!(_dispatch_source_type_memorypressure).cast();
        let source = unsafe {
            dispatch_source_create(
                source_type,
                0,
                (PRESSURE_NORMAL | PRESSURE_WARNING | PRESSURE_CRITICAL) as c_ulong,
                queue,
            )
        };
        if source.is_null() {
            return Err("failed to create Darwin memory-pressure dispatch source".to_owned());
        }
        let context = Box::into_raw(Box::new(ObserverContext { controller, source }));
        unsafe {
            dispatch_set_context(source, context.cast());
            dispatch_source_set_event_handler_f(source, pressure_event);
            dispatch_activate(source);
        }
        Ok(Self {
            source,
            queue,
            context,
        })
    }
}

impl Drop for DarwinMemoryPressureObserver {
    fn drop(&mut self) {
        unsafe {
            dispatch_source_cancel(self.source);
            dispatch_sync_f(self.queue, std::ptr::null_mut(), drain_marker);
            dispatch_release(self.source.cast());
            dispatch_release(self.queue.cast());
            drop(Box::from_raw(self.context));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn manifest() -> DeclaredResidencyManifest {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/pressure_residency_manifest.json");
        DeclaredResidencyManifest::from_offline_report(&path).unwrap()
    }

    struct DropCounter(Arc<AtomicUsize>);
    impl Drop for DropCounter {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }

    #[test]
    fn validates_the_canonical_offline_manifest() {
        let path = Path::new("/Volumes/Elements/mimo-prismwing/evidence/PW-0207/offline-001.json");
        if path.exists() {
            let parsed = DeclaredResidencyManifest::from_offline_report(path).unwrap();
            assert_eq!(parsed.objects.len(), 592);
            assert_eq!(parsed.selected_bytes, 12_878_375_808);
        }
    }

    #[test]
    fn rejects_non_total_eviction_order() {
        let mut invalid = manifest();
        invalid.objects[1].warning_eviction_order = 1;
        assert!(
            invalid
                .validate()
                .unwrap_err()
                .contains("duplicate warning eviction")
        );
    }

    #[test]
    fn warning_immediately_evicts_owned_payloads_in_declared_order() {
        let controller = PressureResidencyController::new(manifest()).unwrap();
        let drops = Arc::new(AtomicUsize::new(0));
        controller
            .install("second", 5, &"1".repeat(64), DropCounter(drops.clone()))
            .unwrap();
        controller
            .install("first", 3, &"0".repeat(64), DropCounter(drops.clone()))
            .unwrap();
        controller.handle_pressure_mask(PRESSURE_WARNING).unwrap();
        assert_eq!(controller.resident_bytes().unwrap(), 0);
        assert_eq!(drops.load(Ordering::SeqCst), 2);
        let events = controller.events().unwrap();
        assert_eq!(events[0].evicted_identities, vec!["first", "second"]);
        assert!(!events[0].growth_stopped);
    }

    #[test]
    fn critical_pressure_evicts_and_permanently_stops_growth() {
        let controller = PressureResidencyController::new(manifest()).unwrap();
        controller
            .install("first", 3, &"0".repeat(64), vec![0_u8; 3])
            .unwrap();
        controller.handle_pressure_mask(PRESSURE_CRITICAL).unwrap();
        assert!(controller.growth_stopped().unwrap());
        assert!(
            controller
                .install("second", 5, &"1".repeat(64), vec![0_u8; 5])
                .unwrap_err()
                .contains("stopped")
        );
    }

    #[test]
    fn normal_pressure_retains_payloads_and_unknown_pressure_fails_closed() {
        let controller = PressureResidencyController::new(manifest()).unwrap();
        controller
            .install("first", 3, &"0".repeat(64), vec![0_u8; 3])
            .unwrap();
        controller.handle_pressure_mask(PRESSURE_NORMAL).unwrap();
        assert_eq!(controller.resident_bytes().unwrap(), 3);
        assert!(controller.handle_pressure_mask(8).is_err());
        assert_eq!(controller.resident_bytes().unwrap(), 0);
        assert!(controller.growth_stopped().unwrap());
    }

    #[test]
    fn live_darwin_observer_can_start_and_drain() {
        let controller = PressureResidencyController::new(manifest()).unwrap();
        drop(DarwinMemoryPressureObserver::start(controller).unwrap());
    }
}
