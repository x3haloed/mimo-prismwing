import unittest

from tools.run_train_only_code_qat_optimizer_preflight import (
    LEARNING_RATES,
    _gate,
    select_trial,
)


def _trial(rate, error, changed=100):
    return {
        "learning_rate": rate,
        "train_metrics": {"relative_l2": error},
        "training": {"loss_decreased": True},
        "changed_codes": {"gate": changed, "up": 0, "down": 0},
        "code_totals": {"gate": 10000, "up": 10000, "down": 10000},
        "code_domain_valid": True,
        "grid_metadata_unchanged": True,
    }


class CodeQatOptimizerPreflightTests(unittest.TestCase):
    def test_selection_uses_train_error_then_lower_rate(self):
        trials = [_trial(rate, 0.02) for rate in LEARNING_RATES]
        self.assertEqual(select_trial(trials)["learning_rate"], LEARNING_RATES[0])
        trials[-1]["train_metrics"]["relative_l2"] = 0.01
        self.assertEqual(select_trial(trials)["learning_rate"], LEARNING_RATES[-1])
        self.assertEqual(
            select_trial([_trial(0.02, 0.01)], (0.02,))["learning_rate"], 0.02
        )

    def test_gate_requires_improvement_and_bounded_code_changes(self):
        initial = {"relative_l2": 0.04}
        selected = _trial(LEARNING_RATES[0], 0.029)
        self.assertTrue(_gate(initial, selected)["passes"])
        selected["train_metrics"]["relative_l2"] = 0.031
        self.assertFalse(_gate(initial, selected)["passes"])
        selected = _trial(LEARNING_RATES[0], 0.029, changed=2000)
        self.assertFalse(_gate(initial, selected)["passes"])


if __name__ == "__main__":
    unittest.main()
