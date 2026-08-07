import unittest

from tools.analyze_group_local_gptq_three_expert_control import _safety_summary


class GroupLocalGptqAnalysisTests(unittest.TestCase):
    def test_safety_requires_a_release_boundary(self):
        snapshot = {
            "release_boundary": False,
            "system_memory_free_percent": 80,
            "process_peak_resident_bytes": 1,
            "process_physical_footprint_bytes": 1,
            "swap_growth_bytes": 0,
            "new_throttled_pages": 0,
            "protected_service_pids": {"service": [1]},
        }
        with self.assertRaises(ValueError):
            _safety_summary([snapshot])


if __name__ == "__main__":
    unittest.main()
