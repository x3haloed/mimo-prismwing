import unittest

from tools.analyze_million_token_openrouter_reference import validate_safety


def snapshot(phase, *, release=False, free=50, physical=100, peak=200, swap=0, throttled=0):
    return {
        "phase": phase,
        "release_boundary": release,
        "system_memory_free_percent": free,
        "process_physical_footprint_bytes": physical,
        "process_peak_resident_bytes": peak,
        "swap_growth_bytes": swap,
        "new_throttled_pages": throttled,
        "protected_service_pids": {"ChatGPT": [1], "WindowServer": [2]},
    }


class MillionTokenReferenceAnalysisTests(unittest.TestCase):
    def test_safety_summary_requires_release_and_preserves_extrema(self):
        rows = [snapshot("start"), snapshot("end", release=True, free=45, physical=80, peak=300)]
        result = validate_safety(rows)
        self.assertEqual(result["minimum_system_memory_free_percent"], 45)
        self.assertEqual(result["maximum_process_peak_resident_bytes"], 300)
        self.assertTrue(result["protected_services_stable"])

    def test_safety_fails_closed_on_service_and_pressure(self):
        rows = [snapshot("start"), snapshot("end", release=True)]
        rows[-1]["protected_service_pids"]["ChatGPT"] = []
        with self.assertRaisesRegex(ValueError, "service"):
            validate_safety(rows)
        rows = [snapshot("start"), snapshot("end", release=True, free=19)]
        with self.assertRaisesRegex(ValueError, "memory-free"):
            validate_safety(rows)


if __name__ == "__main__":
    unittest.main()
