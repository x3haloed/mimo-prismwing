import unittest

import numpy as np

from tools.run_five_bit_global_hessian_three_expert_control import (
    MAXIMUM_CODE,
    PACKED_BYTES,
    PACKED_RATIO,
    _gate,
    affine_nbit_grid,
    physical_ledger,
    reconstruct_nbit_grid,
)


def _report(layer, candidate=0.015, prior=0.06):
    return {
        "layer": layer,
        "expert": 1,
        "five_bit_validation_metrics": {
            "relative_l2": candidate,
            "maximum_row_relative_l2": 0.04,
        },
        "five_bit_train_metrics": {"relative_l2": 0.01},
        "five_bit_rtn_train_metrics": {"relative_l2": 0.02},
        "pw0138_four_bit_validation_metrics": {"relative_l2": prior},
        "four_bit_control_reproduced": True,
        "code_domain_valid": True,
    }


class FiveBitGlobalHessianTests(unittest.TestCase):
    def test_affine_grid_uses_32_levels_and_reconstructs(self):
        values = np.linspace(-1, 1, 256, dtype=np.float32).reshape(2, 128)
        scales, biases, quantized, codes = affine_nbit_grid(values)
        self.assertLessEqual(int(codes.max()), MAXIMUM_CODE)
        self.assertEqual(int(codes.min()), 0)
        self.assertTrue(
            np.array_equal(quantized, reconstruct_nbit_grid(codes, scales, biases))
        )
        self.assertEqual(scales.dtype, np.float16)
        self.assertEqual(biases.dtype, np.float16)

    def test_physical_ledger_is_exact(self):
        ledger = physical_ledger()
        self.assertEqual(ledger["packed_bytes_per_expert"], PACKED_BYTES)
        self.assertAlmostEqual(ledger["packed_to_source_ratio"], PACKED_RATIO)
        self.assertLess(PACKED_RATIO, 0.70)
        self.assertEqual(ledger["additional_runtime_macs"], 0)

    def test_gate_requires_every_expert_and_prior_improvement(self):
        reports = [_report(4), _report(24), _report(46)]
        self.assertTrue(_gate(reports)["passes"])
        reports[1]["five_bit_validation_metrics"]["relative_l2"] = 0.021
        self.assertFalse(_gate(reports)["passes"])
        reports[1]["five_bit_validation_metrics"]["relative_l2"] = 0.015
        reports[2]["pw0138_four_bit_validation_metrics"]["relative_l2"] = 0.01
        self.assertFalse(_gate(reports)["passes"])


if __name__ == "__main__":
    unittest.main()

