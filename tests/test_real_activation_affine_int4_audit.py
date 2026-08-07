import unittest

import mlx.core as mx
import numpy as np

from tools.run_real_activation_affine_int4_audit import (
    _validation_gate,
    error_metrics,
    quantized_projection,
    reconstruct_routed,
    validate_routes,
)


class RealActivationAffineInt4AuditTests(unittest.TestCase):
    def test_tiny_affine_projection_matches_dequantized_oracle(self):
        weight = (mx.arange(64 * 128, dtype=mx.float32).reshape(64, 128) / 4096).astype(mx.float16)
        values = (mx.arange(2 * 128, dtype=mx.float32).reshape(2, 128) / 1024).astype(mx.float16)
        arrays = mx.quantize(weight, group_size=128, bits=4, mode="affine")
        actual = quantized_projection(values, arrays, 4)
        decoded = mx.dequantize(*arrays, group_size=128, bits=4, mode="affine")
        expected = values @ decoded.T
        mx.eval(actual, expected)
        actual_np = np.asarray(actual).astype(np.float32)
        expected_np = np.asarray(expected).astype(np.float32)
        self.assertLess(np.linalg.norm(actual_np - expected_np) / np.linalg.norm(expected_np), 2e-4)
        self.assertLessEqual(np.max(np.abs(actual_np - expected_np)), 0.03125)

    def test_route_reconstruction_applies_f32_weights_then_bf16_boundary(self):
        selected = [[0, 1, 2, 3, 4, 5, 6, 7] for _ in range(224)]
        weights = [[0.125] * 8 for _ in range(224)]
        schedule = [
            {"expert": expert, "positions": list(range(224))} for expert in range(8)
        ]
        authority = {
            "selected_experts_by_position": selected,
            "route_weights_by_position": weights,
            "expert_schedule": schedule,
        }
        expert_down = np.ones((1792, 4096), dtype=np.float32)
        validate_routes(authority)
        result = reconstruct_routed(expert_down, authority, 0, 2)
        self.assertTrue(np.array_equal(result, np.ones((2, 4096), dtype=np.float32)))

    def test_error_metrics_and_validation_gate(self):
        expected = np.ones((2, 4), dtype=np.float32)
        actual = expected * 1.01
        metrics = error_metrics(actual, expected)
        self.assertAlmostEqual(metrics["relative_l2"], 0.009999990463256836)
        rows = [
            {
                "layer": layer,
                "bits": 4,
                "routed_output_metrics": {
                    **metrics,
                    "relative_l2": 0.009,
                    "maximum_row_relative_l2": 0.04,
                },
                "packed_to_source_ratio": 0.54,
            }
            for layer in (4, 24, 46)
        ]
        gate = _validation_gate(rows)
        self.assertTrue(gate["passes"])
        rows[0]["routed_output_metrics"]["maximum_row_relative_l2"] = 0.051
        self.assertFalse(_validation_gate(rows)["passes"])

    def test_invalid_bits_and_route_cardinality_fail_closed(self):
        with self.assertRaises(ValueError):
            quantized_projection(mx.zeros((1, 128)), (mx.zeros((1,)),) * 3, 6)
        with self.assertRaises(ValueError):
            validate_routes({"selected_experts_by_position": [], "route_weights_by_position": []})


if __name__ == "__main__":
    unittest.main()
