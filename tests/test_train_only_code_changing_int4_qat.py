import unittest

import mlx.core as mx
import numpy as np

from tools.run_train_only_code_changing_int4_qat import (
    _gate,
    materialize_qat_weights,
    quantized_code_ste,
    train_code_offsets,
)


def _grid(rows, groups):
    codes = np.arange(rows * groups * 128, dtype=np.uint8).reshape(rows, groups * 128) % 16
    return {
        "codes": codes,
        "scales": np.full((rows, groups), 0.001, dtype=np.float16),
        "biases": np.full((rows, groups), -0.007, dtype=np.float16),
    }


def _report(layer, candidate=0.04, prior=0.06):
    return {
        "layer": layer,
        "expert": 1,
        "initial_train_metrics": {"relative_l2": 0.05},
        "qat_train_metrics": {"relative_l2": 0.04},
        "qat_validation_metrics": {
            "relative_l2": candidate,
            "maximum_row_relative_l2": 0.07,
        },
        "pw0139_validation_metrics": {"relative_l2": prior},
        "training": {"loss_decreased": True},
        "changed_codes": {"gate": 1, "up": 1, "down": 1},
        "code_totals": {"gate": 10, "up": 10, "down": 10},
        "code_domain_valid": True,
        "grid_metadata_unchanged": True,
    }


class CodeChangingInt4QatTests(unittest.TestCase):
    def test_straight_through_forward_rounds_and_gradient_moves(self):
        initial = mx.array([0.0, 7.0, 15.0])
        offsets = mx.array([-1.0, 0.6, 2.0])
        np.testing.assert_array_equal(
            np.asarray(quantized_code_ste(initial, offsets)), np.array([0.0, 8.0, 15.0])
        )
        gradient = mx.grad(lambda value: mx.sum(quantized_code_ste(initial, value)))(offsets)
        np.testing.assert_array_equal(np.asarray(gradient), np.ones(3, dtype=np.float32))

    def test_materialization_changes_codes_and_preserves_f16_metadata(self):
        grids = {"gate": _grid(2, 1), "up": _grid(2, 1), "down": _grid(4, 1)}
        offsets = {
            f"{name}_offset": np.full(grid["codes"].shape, 0.6, dtype=np.float32)
            for name, grid in grids.items()
        }
        _, final, changed = materialize_qat_weights(grids, offsets)
        self.assertTrue(all(value > 0 for value in changed.values()))
        for name in grids:
            self.assertTrue(np.array_equal(final[name]["scales"], grids[name]["scales"]))
            self.assertTrue(np.array_equal(final[name]["biases"], grids[name]["biases"]))
            self.assertTrue(np.all(final[name]["codes"] <= 15))

    def test_tiny_training_moves_latents(self):
        grids = {"gate": _grid(128, 1), "up": _grid(128, 1), "down": _grid(4, 1)}
        inputs = np.linspace(-0.1, 0.1, 256, dtype=np.float32).reshape(2, 128)
        targets = np.full((2, 4), 0.25, dtype=np.float32)
        offsets, diagnostics = train_code_offsets(inputs, targets, grids, steps=1)
        self.assertTrue(any(np.any(value != 0) for value in offsets.values()))
        self.assertEqual([row["step"] for row in diagnostics["loss_history"]], [0, 1])

    def test_gate_requires_deep_quality_and_code_changes(self):
        reports = [_report(4, 0.02, 0.02), _report(24), _report(46)]
        self.assertTrue(_gate(reports)["passes"])
        reports[1]["qat_validation_metrics"]["relative_l2"] = 0.051
        self.assertFalse(_gate(reports)["passes"])
        reports[1]["qat_validation_metrics"]["relative_l2"] = 0.04
        reports[2]["changed_codes"] = {"gate": 0, "up": 0, "down": 0}
        self.assertFalse(_gate(reports)["passes"])


if __name__ == "__main__":
    unittest.main()

