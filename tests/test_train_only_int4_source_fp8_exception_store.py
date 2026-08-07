import unittest

import numpy as np

from tools.run_train_only_int4_source_fp8_exception_store import (
    FRACTIONS,
    _gate,
    dense_correction,
    exception_count,
    physical_ledger,
    rank_exception_groups,
    source_scale_block_count,
    train_second_moment,
)


class SourceFp8ExceptionStoreTests(unittest.TestCase):
    def test_ranking_uses_train_moment_and_is_deterministic(self):
        source = np.zeros((2, 256), dtype=np.float16)
        int4 = np.zeros_like(source)
        source[0, 0] = 2
        source[1, 128] = 3
        moment = np.ones(256, dtype=np.float32)
        order, scores = rank_exception_groups(source, int4, moment)
        self.assertEqual(order.tolist(), [3, 0, 1, 2])
        self.assertEqual(scores[3], 9)
        moment[128:] = 0
        changed, _ = rank_exception_groups(source, int4, moment)
        self.assertEqual(changed[0], 0)

    def test_dense_correction_restores_only_selected_group(self):
        source = np.arange(512, dtype=np.float16).reshape(2, 256)
        int4 = np.zeros_like(source)
        correction = dense_correction(source, int4, np.array([2], dtype=np.uint32))
        self.assertTrue(np.array_equal(correction[1, :128], source[1, :128]))
        self.assertEqual(np.count_nonzero(correction[0]), 0)
        self.assertEqual(np.count_nonzero(correction[1, 128:]), 0)

    def test_source_scale_blocks_share_across_rows_in_same_128_block(self):
        selected = np.array([0, 1, 2, 3], dtype=np.uint32)
        self.assertEqual(source_scale_block_count(selected, 256), 2)

    def test_six_percent_fits_frozen_physical_envelope(self):
        ledger = physical_ledger(0.06)
        self.assertEqual(ledger["selected_groups_per_projection"], 3933)
        self.assertLessEqual(ledger["combined_to_source_ratio"], 0.60)
        self.assertLessEqual(ledger["correction_to_source_expert_mac_ratio"], 0.10)
        self.assertEqual(exception_count(65_536, 0.01), 656)

    def test_gate_selects_smallest_strict_fraction(self):
        reports = []
        for layer in (4, 24, 46):
            fractions = {}
            for fraction in FRACTIONS:
                error = 0.009 if fraction >= 0.04 else 0.03
                fractions[str(fraction)] = {
                    "routed_output_metrics": {
                        "relative_l2": error,
                        "maximum_row_relative_l2": error,
                        "squared_error": error**2,
                        "expected_squared_norm": 1.0,
                    }
                }
            reports.append({"layer": layer, "fractions": fractions})
        gate = _gate(reports)
        self.assertEqual(gate["smallest_strict_fraction"], 0.04)
        self.assertIsNone(gate["smallest_near_miss_fraction"])

    def test_invalid_moment_fails_closed(self):
        source = np.zeros((1, 128), dtype=np.float16)
        with self.assertRaises(ValueError):
            rank_exception_groups(source, source, np.array([-1] * 128, dtype=np.float32))

    def test_validation_mutation_cannot_change_training_moment(self):
        values = np.arange(168 * 4, dtype=np.float32).reshape(168, 4)
        before = train_second_moment(values, [0, 3, 111])
        values[112:] = -999_999
        after = train_second_moment(values, [0, 3, 111])
        self.assertTrue(np.array_equal(before, after))
        with self.assertRaises(ValueError):
            train_second_moment(values, [112])


if __name__ == "__main__":
    unittest.main()
