import unittest

import numpy as np

from tools.run_train_only_awq_int4_expert_rescaling import (
    ALPHAS,
    activation_mean_abs,
    awq_scale,
    physical_ledger,
    transform_expert_weights,
    transform_reconstruction_error,
)


class AwqExpertRescalingTests(unittest.TestCase):
    def test_zero_alpha_is_identity(self):
        scale = awq_scale(np.array([0.0, 1.0, 9.0], dtype=np.float32), 0.0)
        self.assertTrue(np.array_equal(scale, np.ones(3, dtype=np.float32)))

    def test_scale_matches_official_family_and_is_finite(self):
        mean = np.array([1.0, 4.0, 16.0], dtype=np.float32)
        scale = awq_scale(mean, 0.5)
        self.assertTrue(np.allclose(scale, np.array([0.5, 1.0, 2.0], dtype=np.float32)))
        self.assertTrue(np.isfinite(scale).all())

    def test_exact_transform_preserves_swiglu_function(self):
        rng = np.random.default_rng(7)
        weights = {
            "gate": rng.normal(size=(4, 8)).astype(np.float32),
            "up": rng.normal(size=(4, 8)).astype(np.float32),
            "down": rng.normal(size=(8, 4)).astype(np.float32),
        }
        input_scale = np.exp(rng.normal(size=8)).astype(np.float32)
        hidden_scale = np.exp(rng.normal(size=4)).astype(np.float32)
        transformed = transform_expert_weights(weights, input_scale, hidden_scale)
        self.assertLess(
            transform_reconstruction_error(weights, transformed, input_scale, hidden_scale),
            1e-7,
        )
        # A small slice gates the non-homogeneous SiLU placement explicitly.
        x = rng.normal(size=(2, 8))
        gate = rng.normal(size=(4, 8))
        up = rng.normal(size=(4, 8))
        down = rng.normal(size=(8, 4))
        sx = np.exp(rng.normal(size=8))
        sh = np.exp(rng.normal(size=4))
        original_gate = x @ gate.T
        original_up = x @ up.T
        original = (1 / (1 + np.exp(-original_gate)) * original_gate * original_up) @ down.T
        scaled_gate = (x / sx) @ (gate * sx).T
        scaled_up = (x / sx) @ (up * sx / sh[:, None]).T
        scaled = (1 / (1 + np.exp(-scaled_gate)) * scaled_gate * scaled_up) @ (down * sh).T
        self.assertTrue(np.allclose(original, scaled, rtol=1e-12, atol=1e-12))

    def test_validation_mutation_cannot_change_training_mean(self):
        values = np.arange(168 * 4, dtype=np.float32).reshape(168, 4)
        before = activation_mean_abs(values[:112])
        values[112:] = -999_999
        after = activation_mean_abs(values[:112])
        self.assertTrue(np.array_equal(before, after))

    def test_physical_ledger_passes_contract(self):
        ledger = physical_ledger()
        self.assertEqual(ledger["conservative_f16_scale_bytes_per_expert"], 12_288)
        self.assertLessEqual(ledger["combined_to_source_ratio"], 0.60)
        self.assertLessEqual(ledger["runtime_elementwise_to_source_expert_mac_ratio"], 0.01)
        self.assertEqual(ALPHAS[0], 0.0)
        self.assertEqual(ALPHAS[-1], 0.95)


if __name__ == "__main__":
    unittest.main()
