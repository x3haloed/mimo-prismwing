import unittest

from tools.host_safety import (
    GIB,
    MIB,
    HostReading,
    HostSafetyPolicy,
    HostSafetyMonitor,
    HostSafetyViolation,
)


def reading(
    *,
    free=80,
    swap=100 * MIB,
    throttled=0,
    resident=100 * MIB,
    footprint=100 * MIB,
    peak=100 * MIB,
    services=None,
):
    return HostReading(
        system_memory_free_percent=free,
        swap_used_bytes=swap,
        throttled_pages=throttled,
        process_resident_bytes=resident,
        process_physical_footprint_bytes=footprint,
        process_peak_resident_bytes=peak,
        protected_service_pids=services or {"ChatGPT": [1], "WindowServer": [2]},
    )


class FakeProbe:
    def __init__(self, *readings):
        self.readings = list(readings)
        self.relief_calls = 0

    def read(self, protected_services):
        if not self.readings:
            raise RuntimeError("counter unavailable")
        return self.readings.pop(0)

    def allocator_pressure_relief(self):
        self.relief_calls += 1
        return 1234


class HostSafetyTests(unittest.TestCase):
    def test_normative_defaults_fail_closed_until_high_residency_is_realized(self):
        policy = HostSafetyPolicy()
        self.assertEqual(policy.minimum_system_memory_free_percent, 10)
        self.assertEqual(policy.maximum_process_physical_footprint_bytes, 8 * GIB)
        self.assertEqual(
            policy.maximum_post_release_physical_footprint_bytes, 4 * GIB
        )
        self.assertEqual(policy.maximum_swap_growth_bytes, 0)

    def test_in_phase_and_post_release_limits_are_distinct(self):
        probe = FakeProbe(
            reading(), reading(footprint=4 * GIB), reading(footprint=4 * GIB + 1)
        )
        monitor = HostSafetyMonitor(probe=probe)
        monitor.checkpoint("in_flight")
        with self.assertRaisesRegex(HostSafetyViolation, "post-release footprint"):
            monitor.release_checkpoint("released", ["draft weights"])
        self.assertEqual(probe.relief_calls, 1)
        self.assertEqual(monitor.snapshots[-1].released_resources, ("draft weights",))
        self.assertEqual(monitor.snapshots[-1].allocator_pressure_relief_bytes, 1234)

    def test_peak_overshoot_fails_after_expensive_operation(self):
        monitor = HostSafetyMonitor(probe=FakeProbe(reading(), reading(peak=8 * GIB + 1)))
        with self.assertRaisesRegex(HostSafetyViolation, "process memory ceiling"):
            monitor.checkpoint("layer_0_complete")

    def test_swap_throttle_and_service_stops(self):
        cases = (
            (reading(swap=100 * MIB + 1), "swap growth"),
            (reading(throttled=1), "VM throttling"),
            (
                reading(services={"ChatGPT": [], "WindowServer": [2]}),
                "protected service ChatGPT",
            ),
        )
        for candidate, message in cases:
            with self.subTest(message=message):
                monitor = HostSafetyMonitor(probe=FakeProbe(reading(), candidate))
                with self.assertRaisesRegex(HostSafetyViolation, message):
                    monitor.checkpoint("candidate")

    def test_unreadable_counter_fails_closed(self):
        monitor = HostSafetyMonitor(probe=FakeProbe(reading()))
        with self.assertRaisesRegex(RuntimeError, "counter unavailable"):
            monitor.checkpoint("missing_counter")

    def test_only_baseline_resident_services_are_required(self):
        absent = {"ChatGPT": [], "WindowServer": [2]}
        monitor = HostSafetyMonitor(
            probe=FakeProbe(reading(services=absent), reading(services=absent))
        )
        monitor.checkpoint("still_healthy")

    def test_release_requires_named_resources(self):
        monitor = HostSafetyMonitor(probe=FakeProbe(reading()))
        with self.assertRaisesRegex(ValueError, "named released resources"):
            monitor.release_checkpoint("released", [])


if __name__ == "__main__":
    unittest.main()
