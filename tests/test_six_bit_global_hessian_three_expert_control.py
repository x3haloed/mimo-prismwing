import unittest

import numpy as np

from tools.run_six_bit_global_hessian_three_expert_control import (
    MAXIMUM_CODE,
    PACKED_BYTES,
    PACKED_RATIO,
    _gate,
    affine_six_bit_grid,
    physical_ledger,
    reconstruct_six_bit_grid,
)


def _report(layer, candidate=0.015, prior=0.04):
    return {
        "layer": layer,
        "expert": 1,
        "six_bit_validation_metrics": {
            "relative_l2": candidate,
            "maximum_row_relative_l2": 0.04,
        },
        "six_bit_train_metrics": {"relative_l2": 0.01},
        "six_bit_rtn_train_metrics": {"relative_l2": 0.02},
        "prior_candidate_validation_metrics": {"relative_l2": prior},
        "four_bit_control_reproduced": True,
        "code_domain_valid": True,
    }


class SixBitGlobalHessianTests(unittest.TestCase):
    def test_affine_grid_uses_64_levels_and_reconstructs(self):
        values = np.linspace(-1, 1, 256, dtype=np.float32).reshape(2, 128)
        scales, biases, quantized, codes = affine_six_bit_grid(values)
        self.assertLessEqual(int(codes.max()), MAXIMUM_CODE)
        self.assertEqual(int(codes.min()), 0)
        self.assertTrue(
            np.array_equal(quantized, reconstruct_six_bit_grid(codes, scales, biases))
        )

    def test_physical_ledger_is_exact(self):
        ledger = physical_ledger()
        self.assertEqual(ledger["packed_bytes_per_expert"], PACKED_BYTES)
        self.assertEqual(ledger["full_routed_bank_bytes"], 236_558_745_600)
        self.assertAlmostEqual(ledger["packed_to_source_ratio"], PACKED_RATIO)
        self.assertLess(PACKED_RATIO, 0.80)
        self.assertEqual(ledger["additional_runtime_macs"], 0)

    def test_gate_requires_every_expert_and_five_bit_improvement(self):
        reports = [_report(4), _report(24), _report(46)]
        self.assertTrue(_gate(reports)["passes"])
        reports[1]["six_bit_validation_metrics"]["relative_l2"] = 0.021
        self.assertFalse(_gate(reports)["passes"])
        reports[1]["six_bit_validation_metrics"]["relative_l2"] = 0.015
        reports[2]["prior_candidate_validation_metrics"]["relative_l2"] = 0.01
        self.assertFalse(_gate(reports)["passes"])


if __name__ == "__main__":
    unittest.main()
