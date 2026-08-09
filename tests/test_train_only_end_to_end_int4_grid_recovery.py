import unittest

import numpy as np

from tools.run_train_only_end_to_end_int4_grid_recovery import (
    _gate,
    materialize_recovered_weights,
    partition_positions,
    train_fixed_code_grids,
)


def _grid(rows, groups):
    codes = np.arange(rows * groups * 128, dtype=np.uint8).reshape(rows, groups * 128) % 16
    return {
        "codes": codes,
        "scales": np.full((rows, groups), 0.001, dtype=np.float32),
        "biases": np.full((rows, groups), -0.007, dtype=np.float32),
    }


def _report(layer, candidate=0.04, prior=0.06):
    return {
        "layer": layer,
        "expert": 1,
        "initial_train_metrics": {"relative_l2": 0.05},
        "recovered_train_metrics": {"relative_l2": 0.04},
        "recovered_validation_metrics": {
            "relative_l2": candidate,
            "maximum_row_relative_l2": 0.07,
        },
        "pw0139_validation_metrics": {"relative_l2": prior},
        "training": {"loss_decreased": True},
        "codes_unchanged": True,
    }


class TrainOnlyGridRecoveryTests(unittest.TestCase):
    def test_partition_keeps_holdout_sealed(self):
        train, validation = partition_positions([0, 111, 112, 167, 168, 223])
        self.assertEqual(train, [0, 1])
        self.assertEqual(validation, [2, 3])
        with self.assertRaises(ValueError):
            partition_positions([224])

    def test_zero_delta_reconstructs_f16_grid_and_preserves_codes(self):
        grids = {"gate": _grid(2, 1), "up": _grid(2, 1), "down": _grid(4, 1)}
        parameters = {}
        for name, grid in grids.items():
            parameters[f"{name}_log_scale"] = np.zeros_like(grid["scales"])
            parameters[f"{name}_bias_delta"] = np.zeros_like(grid["biases"])
        weights, final = materialize_recovered_weights(grids, parameters)
        for name in grids:
            self.assertTrue(np.array_equal(final[name]["codes"], grids[name]["codes"]))
            expected = (
                grids[name]["codes"].astype(np.float32)
                * np.repeat(final[name]["scales"].astype(np.float32), 128, axis=1)
                + np.repeat(final[name]["biases"].astype(np.float32), 128, axis=1)
            ).astype(np.float16)
            self.assertTrue(np.array_equal(weights[name], expected))
            self.assertEqual(final[name]["scales"].dtype, np.float16)
            self.assertEqual(final[name]["biases"].dtype, np.float16)

    def test_gate_requires_deep_improvement_and_fixed_codes(self):
        reports = [_report(4, 0.02, 0.02), _report(24), _report(46)]
        self.assertTrue(_gate(reports)["passes"])
        reports[1]["recovered_validation_metrics"]["relative_l2"] = 0.051
        self.assertFalse(_gate(reports)["passes"])
        reports[1]["recovered_validation_metrics"]["relative_l2"] = 0.04
        reports[2]["codes_unchanged"] = False
        self.assertFalse(_gate(reports)["passes"])

    def test_training_moves_parameters_without_mutating_codes(self):
        grids = {"gate": _grid(128, 1), "up": _grid(128, 1), "down": _grid(4, 1)}
        original_codes = {name: grid["codes"].copy() for name, grid in grids.items()}
        inputs = np.linspace(-0.1, 0.1, 256, dtype=np.float32).reshape(2, 128)
        targets = np.full((2, 4), 0.25, dtype=np.float32)
        parameters, diagnostics = train_fixed_code_grids(
            inputs, targets, grids, steps=1
        )
        self.assertTrue(diagnostics["loss_decreased"] or diagnostics["loss_history"][-1]["loss"] != diagnostics["loss_history"][0]["loss"])
        self.assertTrue(any(np.any(value != 0) for value in parameters.values()))
        for name in grids:
            self.assertTrue(np.array_equal(grids[name]["codes"], original_codes[name]))


if __name__ == "__main__":
    unittest.main()
