import unittest

import numpy as np

from tools.analyze_pw0325_prismwing1_envelope import (
    SOURCE_BYTES,
    balanced_order,
    category_aggregate,
    conservative_cached_bytes,
    nearest_rank,
    window_identity_records,
)


class Pw0325Prismwing1EnvelopeTests(unittest.TestCase):
    def test_conservative_cache_retains_one_source_record_of_uncertainty(self):
        total = 10 * SOURCE_BYTES
        self.assertEqual(
            conservative_cached_bytes(total, 4 * SOURCE_BYTES),
            7 * SOURCE_BYTES,
        )

    def test_balanced_order_uses_canonical_index_for_exact_tie(self):
        counts = np.asarray([[1, 1], [1, 1]], dtype=np.int64)
        order, remaining = balanced_order(counts, np.asarray([20.0, 20.0]), byte_reduction=10)
        self.assertEqual(order, [0, 1])
        np.testing.assert_array_equal(remaining, np.zeros(2))

    def test_balanced_order_prioritizes_shared_then_underserved_reduction(self):
        counts = np.asarray([[2, 0, 1], [0, 2, 1]], dtype=np.int64)
        order, remaining = balanced_order(counts, np.asarray([20.0, 10.0]), byte_reduction=10)
        self.assertEqual(order, [2, 0])
        np.testing.assert_array_equal(remaining, np.zeros(2))

    def test_category_aggregate_uses_total_tokens_over_total_wall(self):
        windows = [
            {"category": "code", "accepted_tokens": 2, "bytes": 10},
            {"category": "code", "accepted_tokens": 8, "bytes": 90},
        ]
        row = category_aggregate(windows, "bytes")["code"]
        self.assertEqual(row["accepted_tokens"], 10)
        self.assertEqual(row["bytes_after_cache"], 100)
        self.assertAlmostEqual(row["optimistic_accepted_tps"], 10 / row["storage_wall_seconds"])

    def test_nearest_rank_p10_for_32_windows_is_fourth_lowest(self):
        self.assertEqual(nearest_rank(list(range(32)), 0.10), 3)

    def test_window_identity_records_are_sorted_and_representation_explicit(self):
        records = window_identity_records(
            {(2, 7), (1, 9), (1, 3)},
            {(2, 7)},
        )
        self.assertEqual(
            records,
            [
                {"layer": 1, "expert": 3, "representation": "source_fp8"},
                {"layer": 1, "expert": 9, "representation": "source_fp8"},
                {"layer": 2, "expert": 7, "representation": "k4"},
            ],
        )


if __name__ == "__main__":
    unittest.main()
