import unittest

from tools.run_train_only_int4_low_rank_repair import _gate


def _metrics(relative_l2, maximum_row_relative_l2):
    return {
        "relative_l2": relative_l2,
        "maximum_row_relative_l2": maximum_row_relative_l2,
        "squared_error": relative_l2**2,
        "expected_squared_norm": 1.0,
    }


def _report(layer, baseline, affine, rank32, row, fallback=0):
    return {
        "layer": layer,
        "baseline": {"validation": _metrics(baseline, baseline)},
        "affine": {"validation": _metrics(affine, affine)},
        "rank32": {"validation": _metrics(rank32, row)},
        "coverage": {
            "validation_coverage_fraction": 0.99 if fallback else 1.0,
            "validation_identity_fallback_placements": fallback,
        },
    }


class TrainOnlyRepairAnalysisTests(unittest.TestCase):
    def test_rejection_gate_detects_overfit_and_incomplete_coverage(self):
        reports = [
            _report(4, 0.04, 0.05, 0.17, 0.57),
            _report(24, 0.12, 0.07, 0.10, 0.34, fallback=1),
            _report(46, 0.15, 0.10, 0.09, 0.12),
        ]
        gate = _gate(reports)
        self.assertFalse(gate["strict_pass"])
        self.assertFalse(gate["near_miss"])
        self.assertFalse(gate["nested_validation_monotonic"])
        self.assertFalse(gate["complete_validation_coverage"])

    def test_strict_gate_remains_available_to_validator(self):
        reports = [
            _report(layer, 0.03, 0.015, 0.009, 0.04)
            for layer in (4, 24, 46)
        ]
        gate = _gate(reports)
        self.assertTrue(gate["strict_pass"])
        self.assertFalse(gate["near_miss"])


if __name__ == "__main__":
    unittest.main()
