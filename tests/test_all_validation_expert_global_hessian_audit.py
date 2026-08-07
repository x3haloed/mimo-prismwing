import unittest

from tools.run_all_validation_expert_global_hessian_audit import _validation_gate


def _projection():
    return {
        "candidate_calibration_metrics": {"relative_l2": 0.01},
        "baseline_calibration_metrics": {"relative_l2": 0.02},
    }


def _layer(layer, relative_l2=0.01, row=0.02):
    return {
        "layer": layer,
        "validation_placements": 448,
        "pw0138_control_reproduced": True,
        "routed_output_metrics": {
            "relative_l2": relative_l2,
            "maximum_row_relative_l2": row,
            "squared_error": relative_l2**2,
            "expected_squared_norm": 1.0,
        },
        "expert_reports": [
            {
                "validation_placements": 448,
                "projection_reports": {name: _projection() for name in ("gate", "up", "down")},
            }
        ],
    }


class AllValidationExpertAuditTests(unittest.TestCase):
    def test_gate_requires_layer_and_aggregate_bounds(self):
        reports = [_layer(4, 0.005), _layer(24, 0.005), _layer(46, 0.005)]
        self.assertTrue(_validation_gate(reports)["passes"])
        reports[2]["routed_output_metrics"]["relative_l2"] = 0.021
        reports[2]["routed_output_metrics"]["squared_error"] = 0.021**2
        self.assertFalse(_validation_gate(reports)["passes"])

    def test_gate_requires_projection_improvement(self):
        reports = [_layer(4, 0.005), _layer(24, 0.005), _layer(46, 0.005)]
        reports[0]["expert_reports"][0]["projection_reports"]["gate"][
            "candidate_calibration_metrics"
        ]["relative_l2"] = 0.03
        self.assertFalse(_validation_gate(reports)["passes"])


if __name__ == "__main__":
    unittest.main()
