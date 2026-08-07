import unittest
from pathlib import Path

from tools.analyze_rank768_activation_weighted_expert_pilot import analyze


class Rank768ActivationWeightedExpertPilotAnalysisTests(unittest.TestCase):
    def test_real_pilot_closes_holdout_and_safety_gates(self):
        result = analyze(
            Path("/Users/chad/Models/mimo-prismwing/evidence/PW-0121/run-001.json")
        )
        self.assertTrue(result["gates_passed"])
        self.assertGreater(
            result["expert_output_comparisons"]["pilot_holdout"][
                "relative_error_reduction"
            ],
            0.44,
        )
        self.assertEqual(
            result["projection_summary"]["gate"]["selected_step"], 50
        )
        self.assertIsNone(result["performance_claim"])


if __name__ == "__main__":
    unittest.main()
