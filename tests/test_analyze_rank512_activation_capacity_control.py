import unittest
from pathlib import Path

from tools.analyze_rank512_activation_capacity_control import analyze


class Rank512ActivationCapacityControlAnalysisTests(unittest.TestCase):
    def test_real_control_validates_the_capacity_rejection(self):
        result = analyze(
            Path("/Users/chad/Models/mimo-prismwing/evidence/PW-0125/run-001.json")
        )
        self.assertTrue(result["evidence_valid"])
        self.assertFalse(result["experiment_passed"])
        self.assertGreater(result["capacity_ratios"]["validation_to_rank768_fitted"], 1.25)
        self.assertLessEqual(result["capacity_ratios"]["holdout_to_rank768_fitted"], 1.25)
        self.assertGreater(
            result["expert_output_comparisons"]["validation"]["relative_error_reduction"],
            0.60,
        )
        self.assertIsNone(result["performance_claim"])


if __name__ == "__main__":
    unittest.main()
