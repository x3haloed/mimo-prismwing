import unittest

from tools.summarize_pw0318_decode_transaction import safety_extrema


class Pw0318SummaryTests(unittest.TestCase):
    def test_safety_extrema_requires_release_and_preserves_worst_values(self):
        reports = [
            {
                "safety_snapshots": [
                    {
                        "release_boundary": False,
                        "system_memory_free_percent": 70,
                        "process_peak_resident_bytes": 10,
                        "process_physical_footprint_bytes": 8,
                        "swap_growth_bytes": 0,
                        "new_throttled_pages": 0,
                    },
                    {
                        "release_boundary": True,
                        "system_memory_free_percent": 68,
                        "process_peak_resident_bytes": 12,
                        "process_physical_footprint_bytes": 4,
                        "swap_growth_bytes": 0,
                        "new_throttled_pages": 0,
                    },
                ]
            }
        ]
        self.assertEqual(
            safety_extrema(reports),
            {
                "minimum_system_memory_free_percent": 68,
                "maximum_process_peak_resident_bytes": 12,
                "maximum_process_physical_footprint_bytes": 8,
                "maximum_swap_growth_bytes": 0,
                "maximum_new_throttled_pages": 0,
            },
        )

    def test_safety_extrema_rejects_missing_release(self):
        with self.assertRaises(ValueError):
            safety_extrema(
                [
                    {
                        "safety_snapshots": [
                            {
                                "release_boundary": False,
                                "system_memory_free_percent": 70,
                                "process_peak_resident_bytes": 1,
                                "process_physical_footprint_bytes": 1,
                                "swap_growth_bytes": 0,
                                "new_throttled_pages": 0,
                            }
                        ]
                    }
                ]
            )


if __name__ == "__main__":
    unittest.main()
