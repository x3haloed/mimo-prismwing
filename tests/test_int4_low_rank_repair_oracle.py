import unittest

import numpy as np

from tools.run_int4_low_rank_repair_oracle import (
    apply_low_rank_repair,
    fit_low_rank_repair,
    physical_ledger,
)


class Int4LowRankRepairOracleTests(unittest.TestCase):
    def test_rank_two_repair_recovers_two_row_residual(self):
        inputs = np.zeros((2, 4096), dtype=np.float32)
        inputs[0, 0] = 1.0
        inputs[1, 1] = 1.0
        residual = np.zeros((2, 4096), dtype=np.float32)
        residual[0, 2] = 0.5
        residual[1, 3] = -0.25
        factors = fit_low_rank_repair(inputs, residual, 2)
        actual = apply_low_rank_repair(inputs, np.zeros_like(residual), factors)
        self.assertTrue(np.array_equal(actual, residual))

    def test_rank_one_is_no_better_than_rank_two_on_rank_two_residual(self):
        inputs = np.zeros((2, 4096), dtype=np.float32)
        inputs[0, 0] = inputs[1, 1] = 1.0
        residual = np.zeros((2, 4096), dtype=np.float32)
        residual[0, 2] = residual[1, 3] = 1.0
        rank_one = apply_low_rank_repair(
            inputs, np.zeros_like(residual), fit_low_rank_repair(inputs, residual, 1)
        )
        rank_two = apply_low_rank_repair(
            inputs, np.zeros_like(residual), fit_low_rank_repair(inputs, residual, 2)
        )
        self.assertGreaterEqual(
            np.linalg.norm(rank_one - residual), np.linalg.norm(rank_two - residual)
        )

    def test_rank56_physical_ledger_stays_inside_frozen_envelope(self):
        ledger = physical_ledger(56)
        self.assertLess(ledger["combined_to_source_layer_bank_ratio"], 0.60)
        self.assertLess(ledger["repair_to_source_expert_mac_ratio"], 0.05)
        self.assertEqual(ledger["low_rank_factor_bytes_per_layer"], 234_881_024)

    def test_invalid_fit_and_rank_fail_closed(self):
        with self.assertRaises(ValueError):
            fit_low_rank_repair(np.ones((1, 4)), np.ones((1, 4096)), 1)
        with self.assertRaises(ValueError):
            physical_ledger(0)


if __name__ == "__main__":
    unittest.main()
