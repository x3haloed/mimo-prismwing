import unittest

from tools.analyze_pressure_elastic_residency import (
    select_static_residents,
    solve_attributed_rates,
)


class PressureElasticResidencyTests(unittest.TestCase):
    def test_two_phase_attribution_recovers_positive_rates(self):
        shared, expert = solve_attributed_rates(20, 10, 10, 30, 45.0, 65.0)
        self.assertAlmostEqual(shared, 1.4)
        self.assertAlmostEqual(expert, 1.7)

    def test_static_selection_ranks_stall_avoided_per_byte_and_skips_oversize(self):
        rows = [
            {
                "identity": "oversize",
                "bytes": 11,
                "avoided_logical_read_bytes": 100,
                "avoided_stall_ms": 100.0,
                "avoided_stall_ms_per_resident_byte": 10.0,
            },
            {
                "identity": "best",
                "bytes": 6,
                "avoided_logical_read_bytes": 6,
                "avoided_stall_ms": 12.0,
                "avoided_stall_ms_per_resident_byte": 2.0,
            },
            {
                "identity": "second",
                "bytes": 4,
                "avoided_logical_read_bytes": 4,
                "avoided_stall_ms": 4.0,
                "avoided_stall_ms_per_resident_byte": 1.0,
            },
            {
                "identity": "no_reuse",
                "bytes": 1,
                "avoided_logical_read_bytes": 0,
                "avoided_stall_ms": 0.0,
                "avoided_stall_ms_per_resident_byte": 0.0,
            },
        ]
        self.assertEqual(
            [row["identity"] for row in select_static_residents(rows, 10)],
            ["best", "second"],
        )

    def test_singular_or_nonpositive_attribution_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "singular"):
            solve_attributed_rates(1, 1, 1, 1, 1.0, 1.0)
        with self.assertRaisesRegex(ValueError, "non-positive"):
            solve_attributed_rates(10, 1, 1, 10, 1.0, 100.0)


if __name__ == "__main__":
    unittest.main()
