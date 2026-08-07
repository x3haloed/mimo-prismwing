import unittest

from tools.run_pooled_calibration_low_count_gptq import _gate


def _report(candidate, prior):
    projection = {
        "candidate_calibration_metrics": {"relative_l2": 0.01},
        "baseline_calibration_metrics": {"relative_l2": 0.02},
    }
    return {
        "layer": 24,
        "expert": 39,
        "pooled_validation_metrics": {
            "relative_l2": candidate,
            "maximum_row_relative_l2": 0.1,
        },
        "pw0139_routed_calibration_validation_metrics": {"relative_l2": prior},
        "projection_reports": {name: dict(projection) for name in ("gate", "up", "down")},
    }


class PooledCalibrationTests(unittest.TestCase):
    def test_gate_requires_every_expert_to_improve_and_clear_bound(self):
        self.assertTrue(_gate([_report(0.07, 0.12)])["passes"])
        self.assertFalse(_gate([_report(0.081, 0.12)])["passes"])
        self.assertFalse(_gate([_report(0.07, 0.06)])["passes"])


if __name__ == "__main__":
    unittest.main()
