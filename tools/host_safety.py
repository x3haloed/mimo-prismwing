#!/usr/bin/env python3
"""Fail-closed shared-host safety monitoring for bounded model walks."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import ctypes
import gc
import os
from pathlib import Path
import platform
import resource
import subprocess
from typing import Protocol, Sequence


GIB = 1024**3
MIB = 1024**2
DEFAULT_PROTECTED_SERVICES = ("ChatGPT", "WindowServer", "nxnode", "syncthing")


class HostSafetyViolation(RuntimeError):
    """Raised immediately when a normative shared-host stop is reached."""


@dataclass(frozen=True)
class HostSafetyPolicy:
    minimum_system_memory_free_percent: int = 10
    maximum_process_physical_footprint_bytes: int = 13 * GIB
    maximum_post_release_physical_footprint_bytes: int = 12 * GIB
    maximum_swap_growth_bytes: int = 0
    maximum_new_throttled_pages: int = 0
    protected_services: tuple[str, ...] = DEFAULT_PROTECTED_SERVICES


@dataclass(frozen=True)
class HostReading:
    system_memory_free_percent: int
    swap_used_bytes: int
    throttled_pages: int
    process_resident_bytes: int
    process_physical_footprint_bytes: int
    process_peak_resident_bytes: int
    protected_service_pids: dict[str, list[int]]
    process_disk_bytes_read: int = 0
    process_disk_bytes_written: int = 0


@dataclass(frozen=True)
class HostSafetySnapshot:
    phase: str
    release_boundary: bool
    released_resources: tuple[str, ...]
    system_memory_free_percent: int
    swap_used_bytes: int
    swap_growth_bytes: int
    throttled_pages: int
    new_throttled_pages: int
    process_resident_bytes: int
    process_physical_footprint_bytes: int
    process_peak_resident_bytes: int
    allocator_pressure_relief_bytes: int
    protected_service_pids: dict[str, list[int]]
    process_disk_bytes_read: int = 0
    process_disk_bytes_written: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


class HostProbe(Protocol):
    def read(self, protected_services: Sequence[str]) -> HostReading: ...

    def allocator_pressure_relief(self) -> int: ...


class _RusageInfoV2(ctypes.Structure):
    _fields_ = [
        ("uuid", ctypes.c_uint8 * 16),
        ("user_time", ctypes.c_uint64),
        ("system_time", ctypes.c_uint64),
        ("pkg_idle_wkups", ctypes.c_uint64),
        ("interrupt_wkups", ctypes.c_uint64),
        ("pageins", ctypes.c_uint64),
        ("wired_size", ctypes.c_uint64),
        ("resident_size", ctypes.c_uint64),
        ("phys_footprint", ctypes.c_uint64),
        ("proc_start_abstime", ctypes.c_uint64),
        ("proc_exit_abstime", ctypes.c_uint64),
        ("child_user_time", ctypes.c_uint64),
        ("child_system_time", ctypes.c_uint64),
        ("child_pkg_idle_wkups", ctypes.c_uint64),
        ("child_interrupt_wkups", ctypes.c_uint64),
        ("child_pageins", ctypes.c_uint64),
        ("child_elapsed_abstime", ctypes.c_uint64),
        ("diskio_bytesread", ctypes.c_uint64),
        ("diskio_byteswritten", ctypes.c_uint64),
    ]


class DarwinHostProbe:
    """Read the authoritative Darwin counters required by TARGET.md Gate 8."""

    def __init__(self) -> None:
        if platform.system() != "Darwin":
            raise RuntimeError("Darwin host-safety counters are required")
        self._libproc = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
        self._libproc.proc_pid_rusage.argtypes = (
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p,
        )
        self._libproc.proc_pid_rusage.restype = ctypes.c_int
        self._libsystem = ctypes.CDLL(None, use_errno=True)
        self._libsystem.malloc_zone_pressure_relief.argtypes = (ctypes.c_void_p, ctypes.c_size_t)
        self._libsystem.malloc_zone_pressure_relief.restype = ctypes.c_size_t

    @staticmethod
    def _command(*arguments: str) -> str:
        return subprocess.run(
            arguments, check=True, text=True, capture_output=True
        ).stdout

    def _system_memory_free_percent(self) -> int:
        marker = "System-wide memory free percentage:"
        for line in self._command("/usr/bin/memory_pressure", "-Q").splitlines():
            if marker in line:
                return int(line.split(marker, 1)[1].strip().removesuffix("%"))
        raise RuntimeError("memory_pressure output lacks free percentage")

    def _swap_used_bytes(self) -> int:
        fields = self._command("/usr/sbin/sysctl", "-n", "vm.swapusage").split()
        try:
            value = fields[fields.index("used") + 2]
        except (ValueError, IndexError) as error:
            raise RuntimeError("vm.swapusage output lacks used value") from error
        multipliers = {"K": 1024, "M": MIB, "G": GIB}
        if not value or value[-1] not in multipliers:
            raise RuntimeError("vm.swapusage used value has unknown unit")
        return round(float(value[:-1]) * multipliers[value[-1]])

    def _throttled_pages(self) -> int:
        for line in self._command("/usr/bin/vm_stat").splitlines():
            if line.strip().startswith("Pages throttled:"):
                return int(line.split(":", 1)[1].strip().removesuffix("."))
        raise RuntimeError("vm_stat output lacks throttled pages")

    def _process_usage(self) -> _RusageInfoV2:
        usage = _RusageInfoV2()
        result = self._libproc.proc_pid_rusage(os.getpid(), 2, ctypes.byref(usage))
        if result != 0:
            errno = ctypes.get_errno()
            raise RuntimeError(f"proc_pid_rusage failed: result={result}, errno={errno}")
        return usage

    def _protected_service_pids(self, names: Sequence[str]) -> dict[str, list[int]]:
        result = {name: [] for name in names}
        for line in self._command("/bin/ps", "-axo", "pid=,comm=").splitlines():
            fields = line.strip().split(None, 1)
            if len(fields) != 2:
                continue
            basename = Path(fields[1]).name
            if basename in result:
                result[basename].append(int(fields[0]))
        return result

    def read(self, protected_services: Sequence[str]) -> HostReading:
        usage = self._process_usage()
        peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        if peak < 0:
            raise RuntimeError("getrusage returned a negative peak RSS")
        return HostReading(
            system_memory_free_percent=self._system_memory_free_percent(),
            swap_used_bytes=self._swap_used_bytes(),
            throttled_pages=self._throttled_pages(),
            process_resident_bytes=int(usage.resident_size),
            process_physical_footprint_bytes=int(usage.phys_footprint),
            process_peak_resident_bytes=peak,
            protected_service_pids=self._protected_service_pids(protected_services),
            process_disk_bytes_read=int(usage.diskio_bytesread),
            process_disk_bytes_written=int(usage.diskio_byteswritten),
        )

    def allocator_pressure_relief(self) -> int:
        return int(self._libsystem.malloc_zone_pressure_relief(None, 0))


@dataclass
class HostSafetyMonitor:
    probe: HostProbe = field(default_factory=DarwinHostProbe)
    policy: HostSafetyPolicy = field(default_factory=HostSafetyPolicy)
    snapshots: list[HostSafetySnapshot] = field(default_factory=list, init=False)
    _baseline_swap_bytes: int = field(init=False)
    _baseline_throttled_pages: int = field(init=False)
    _baseline_services: set[str] = field(init=False)

    def __post_init__(self) -> None:
        baseline = self.probe.read(self.policy.protected_services)
        self._baseline_swap_bytes = baseline.swap_used_bytes
        self._baseline_throttled_pages = baseline.throttled_pages
        self._baseline_services = {
            name for name, pids in baseline.protected_service_pids.items() if pids
        }
        self._record_and_enforce("process_start", baseline, False, (), 0)

    def checkpoint(self, phase: str) -> HostSafetySnapshot:
        reading = self.probe.read(self.policy.protected_services)
        return self._record_and_enforce(phase, reading, False, (), 0)

    def release_checkpoint(
        self, phase: str, released_resources: Sequence[str]
    ) -> HostSafetySnapshot:
        if not released_resources or any(not name.strip() for name in released_resources):
            raise ValueError("release checkpoint requires named released resources")
        gc.collect()
        relief = self.probe.allocator_pressure_relief()
        reading = self.probe.read(self.policy.protected_services)
        return self._record_and_enforce(
            phase, reading, True, tuple(released_resources), relief
        )

    def _record_and_enforce(
        self,
        phase: str,
        reading: HostReading,
        release_boundary: bool,
        released_resources: tuple[str, ...],
        relief: int,
    ) -> HostSafetySnapshot:
        if not phase.strip():
            raise ValueError("safety phase must be named")
        swap_growth = max(0, reading.swap_used_bytes - self._baseline_swap_bytes)
        new_throttled = max(
            0, reading.throttled_pages - self._baseline_throttled_pages
        )
        snapshot = HostSafetySnapshot(
            phase=phase,
            release_boundary=release_boundary,
            released_resources=released_resources,
            system_memory_free_percent=reading.system_memory_free_percent,
            swap_used_bytes=reading.swap_used_bytes,
            swap_growth_bytes=swap_growth,
            throttled_pages=reading.throttled_pages,
            new_throttled_pages=new_throttled,
            process_resident_bytes=reading.process_resident_bytes,
            process_physical_footprint_bytes=reading.process_physical_footprint_bytes,
            process_peak_resident_bytes=reading.process_peak_resident_bytes,
            allocator_pressure_relief_bytes=relief,
            protected_service_pids=reading.protected_service_pids,
            process_disk_bytes_read=reading.process_disk_bytes_read,
            process_disk_bytes_written=reading.process_disk_bytes_written,
        )
        self.snapshots.append(snapshot)

        policy = self.policy
        if reading.system_memory_free_percent < policy.minimum_system_memory_free_percent:
            raise HostSafetyViolation(f"safety stop at {phase}: system memory free")
        if (
            reading.process_physical_footprint_bytes
            > policy.maximum_process_physical_footprint_bytes
            or reading.process_peak_resident_bytes
            > policy.maximum_process_physical_footprint_bytes
        ):
            raise HostSafetyViolation(f"safety stop at {phase}: process memory ceiling")
        if (
            release_boundary
            and reading.process_physical_footprint_bytes
            > policy.maximum_post_release_physical_footprint_bytes
        ):
            raise HostSafetyViolation(f"safety stop at {phase}: post-release footprint")
        if swap_growth > policy.maximum_swap_growth_bytes:
            raise HostSafetyViolation(f"safety stop at {phase}: swap growth")
        if new_throttled > policy.maximum_new_throttled_pages:
            raise HostSafetyViolation(f"safety stop at {phase}: VM throttling")
        for name in self._baseline_services:
            if not reading.protected_service_pids.get(name):
                raise HostSafetyViolation(
                    f"safety stop at {phase}: protected service {name} disappeared"
                )
        return snapshot

    def evidence(self) -> list[dict]:
        return [snapshot.to_dict() for snapshot in self.snapshots]
