import unittest
from pathlib import Path

from tools.analyze_rank768_activation_weighted_expert_pilot import (
    PW0122_ANALYSIS_SPEC,
    analyze,
)


class Layer46Rank768ActivationWeightedPilotAnalysisTests(unittest.TestCase):
    def test_real_late_layer_pilot_closes_holdout_and_safety_gates(self):
        result = analyze(
            Path("/Users/chad/Models/mimo-prismwing/evidence/PW-0122/run-001.json"),
            PW0122_ANALYSIS_SPEC,
        )
        self.assertTrue(result["gates_passed"])
        self.assertGreater(
            result["expert_output_comparisons"]["pilot_holdout"][
                "relative_error_reduction"
            ],
            0.47,
        )
        self.assertEqual(result["projection_summary"]["down"]["selected_step"], 70)
        self.assertIsNone(result["performance_claim"])


if __name__ == "__main__":
    unittest.main()
