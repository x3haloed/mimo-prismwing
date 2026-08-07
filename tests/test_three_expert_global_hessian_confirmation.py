import unittest

from tools.run_three_expert_global_hessian_confirmation import _gate


def _report(layer, expert, candidate):
    return {
        "layer": layer,
        "expert": expert,
        "dense_control_train": {"relative_l2": 0.2},
        "dense_control_validation": {"relative_l2": 0.16},
        "global_gptq_train": {"relative_l2": 0.05},
        "global_gptq_validation": {
            "relative_l2": candidate,
            "maximum_row_relative_l2": 0.09,
        },
        "pw0135_group_local_validation": {"relative_l2": 0.08},
    }


class ThreeExpertGlobalHessianTests(unittest.TestCase):
    def test_gate_requires_every_expert(self):
        reports = [_report(4, 96, 0.04), _report(24, 22, 0.06), _report(46, 28, 0.05)]
        self.assertTrue(_gate(reports)["passes"])
        reports[1]["global_gptq_validation"]["relative_l2"] = 0.081
        self.assertFalse(_gate(reports)["passes"])

    def test_gate_rejects_regression_from_group_local(self):
        reports = [_report(4, 96, 0.079)]
        reports[0]["pw0135_group_local_validation"]["relative_l2"] = 0.07
        gate = _gate(reports)
        self.assertFalse(gate["passes"])
        self.assertFalse(gate["experts"][0]["conditions"]["no_worse_than_pw0135"])


if __name__ == "__main__":
    unittest.main()
