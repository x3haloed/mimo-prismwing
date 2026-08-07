import unittest
from pathlib import Path
import tempfile

import numpy as np

from tools.run_routed_residual_subspace_oracle import (
    RANKS,
    metrics,
    passes,
    physical_ledger,
    projection,
    route_slices,
    select_rank,
)
from tools.run_best_rank_real_expert_control import load_capture


class RoutedResidualSubspaceOracleTests(unittest.TestCase):
    def test_centered_projection_and_rank_selection_do_not_require_holdout(self):
        mean = np.array([[1.0, 2.0, 3.0]])
        basis = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        values = np.array([[2.0, 4.0, 9.0], [3.0, 5.0, 8.0]])
        actual = projection(mean, basis, values)
        self.assertTrue(np.array_equal(actual[:, :2], values[:, :2]))
        self.assertTrue(np.array_equal(actual[:, 2], np.array([3.0, 3.0])))
        reports = {
            str(rank): {
                "aggregate_relative_l2": 0.02 if rank < 64 else 0.005,
                "slices": {
                    "a": {"positions": 1, "relative_l2": 0.01},
                    "b": {"positions": 0, "relative_l2": None},
                },
            }
            for rank in RANKS
        }
        self.assertEqual(select_rank(reports), 64)
        self.assertTrue(passes(reports["64"]))

    def test_nested_basis_error_is_monotonic_and_empty_slice_is_explicit(self):
        mean = np.zeros((1, 3))
        values = np.array([[1.0, 2.0, 3.0], [-1.0, 1.0, 2.0]])
        basis1 = np.array([[1.0, 0.0, 0.0]])
        basis2 = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        first = projection(mean, basis1, values)
        second = projection(mean, basis2, values)
        self.assertLessEqual(np.linalg.norm(second - values), np.linalg.norm(first - values))
        report = metrics(values, second, [112, 113], {"empty": [], "all": [112, 113]})
        self.assertEqual(report["slices"]["empty"], {"positions": 0, "relative_l2": None})

    def test_route_slices_preserve_training_unseen_positions(self):
        routes = [[1, 2, 3, 4, 5, 6, 7, 8] for _ in range(224)]
        routes[112] = [1, 2, 3, 4, 5, 6, 7, 99]
        slices = route_slices(routes, [112, 113])
        self.assertEqual(slices["touches_training_unseen_expert"], [112])
        self.assertEqual(slices["all_experts_seen_in_training"], [113])
        expected = np.ones((2, 2))
        report = metrics(expected, expected, [112, 113], slices)
        self.assertEqual(report["aggregate_relative_l2"], 0.0)
        self.assertEqual(report["slices"]["touches_training_unseen_expert"]["positions"], 1)

    def test_physical_ledger_is_exact_and_bounded(self):
        ledger = physical_ledger(111)
        self.assertEqual(ledger["f32_mean_and_basis_bytes"], 1_835_008)
        self.assertEqual(ledger["oracle_output_synthesis_multiplications"], 454_656)
        self.assertTrue(ledger["byte_gate_passed"])
        self.assertTrue(ledger["multiplication_gate_passed"])

    def test_capture_hash_mismatch_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "capture.f32"
            path.write_bytes(np.zeros((2, 2), dtype="<f4").tobytes())
            record = {
                "file": path.name,
                "shape": [2, 2],
                "dtype": "BF16_widened_F32",
                "bytes": 16,
                "sha256": "0" * 64,
            }
            with self.assertRaisesRegex(ValueError, "capture authority mismatch"):
                load_capture(path.parent, record)


if __name__ == "__main__":
    unittest.main()
