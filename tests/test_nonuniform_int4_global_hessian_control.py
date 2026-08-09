import unittest

import numpy as np

from tools.run_nonuniform_int4_global_hessian_control import (
    LLOYD_ITERATIONS,
    MAXIMUM_CODE,
    PACKED_BYTES,
    PACKED_RATIO,
    _gate,
    global_hessian_codebook_fixed_grid,
    nonuniform_int4_grid,
    physical_ledger,
    reconstruct_codebook_grid,
)


def _report(layer, candidate=0.015, prior=0.04):
    return {
        "layer": layer,
        "expert": 1,
        "nonuniform_int4_validation_metrics": {
            "relative_l2": candidate,
            "maximum_row_relative_l2": 0.04,
        },
        "nonuniform_int4_train_metrics": {"relative_l2": 0.01},
        "nonuniform_int4_rtn_train_metrics": {"relative_l2": 0.02},
        "prior_candidate_validation_metrics": {"relative_l2": prior},
        "four_bit_control_reproduced": True,
        "code_domain_valid": True,
    }


class NonuniformInt4ControlTests(unittest.TestCase):
    def test_codebooks_are_deterministic_f16_and_reconstruct(self):
        values = np.linspace(-3, 2, 512, dtype=np.float32).reshape(4, 128)
        first = nonuniform_int4_grid(values)
        second = nonuniform_int4_grid(values)
        self.assertEqual(LLOYD_ITERATIONS, 8)
        self.assertEqual(first[0].dtype, np.float16)
        self.assertLessEqual(int(first[3].max()), MAXIMUM_CODE)
        self.assertTrue(np.array_equal(first[0], second[0]))
        self.assertTrue(np.array_equal(first[3], second[3]))
        self.assertTrue(
            np.array_equal(first[2], reconstruct_codebook_grid(first[3], first[0]))
        )

    def test_ties_choose_lowest_code_and_empty_clusters_remain_finite(self):
        values = np.zeros((2, 128), dtype=np.float32)
        values[1, 64:] = 1
        codebooks, _, quantized, codes = nonuniform_int4_grid(values)
        self.assertTrue(np.isfinite(codebooks).all())
        self.assertTrue(np.all(codes[0] == 0))
        self.assertTrue(np.array_equal(quantized, values.astype(np.float16)))

    def test_physical_ledger_is_exact(self):
        ledger = physical_ledger()
        self.assertEqual(ledger["packed_bytes_per_expert"], PACKED_BYTES)
        self.assertEqual(ledger["full_routed_bank_bytes"], 227_096_395_776)
        self.assertAlmostEqual(ledger["packed_to_source_ratio"], PACKED_RATIO)
        self.assertLessEqual(PACKED_RATIO, 0.75)
        self.assertEqual(ledger["additional_runtime_macs"], 0)

    def test_global_hessian_assignment_stays_on_codebooks_across_blocks(self):
        generator = np.random.default_rng(149)
        weight = generator.normal(size=(3, 256)).astype(np.float32)
        activations = generator.normal(size=(24, 256)).astype(np.float32)
        codebooks, unused, _, _ = nonuniform_int4_grid(weight)
        candidate, codes, diagnostics = global_hessian_codebook_fixed_grid(
            weight, activations, codebooks, unused
        )
        self.assertTrue(
            np.array_equal(candidate, reconstruct_codebook_grid(codes, codebooks))
        )
        self.assertGreater(diagnostics["cross_block_update_l2"], 0)
        self.assertEqual(diagnostics["block_count"], 2)
        self.assertEqual(diagnostics["maximum_code"], MAXIMUM_CODE)

    def test_gate_requires_every_expert_and_six_bit_improvement(self):
        reports = [_report(4), _report(24), _report(46)]
        self.assertTrue(_gate(reports)["passes"])
        reports[1]["nonuniform_int4_validation_metrics"]["relative_l2"] = 0.021
        self.assertFalse(_gate(reports)["passes"])
        reports[1]["nonuniform_int4_validation_metrics"]["relative_l2"] = 0.015
        reports[2]["prior_candidate_validation_metrics"]["relative_l2"] = 0.01
        self.assertFalse(_gate(reports)["passes"])


if __name__ == "__main__":
    unittest.main()
