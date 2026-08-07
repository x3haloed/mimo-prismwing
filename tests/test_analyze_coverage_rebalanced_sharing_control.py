import unittest
from pathlib import Path

from tools.analyze_coverage_rebalanced_sharing_control import analyze


class CoverageRebalancedSharingControlAnalysisTests(unittest.TestCase):
    def test_real_control_rejects_the_coverage_scarcity_explanation(self):
        result = analyze(
            Path("/Users/chad/Models/mimo-prismwing/evidence/PW-0124/run-001.json")
        )
        self.assertTrue(result["evidence_valid"])
        self.assertFalse(result["experiment_passed"])
        self.assertEqual(result["failed_projection_gates"], ["gate", "up", "down"])
        self.assertIn(57, result["failed_projection_experts"])
        self.assertGreater(result["projection_summary"]["up"]["fifth_expert_ratio"], 6.0)
        self.assertTrue(result["all_holdout_improvement_gates_passed"])
        self.assertTrue(result["physical_eligibility"]["byte_gate_passed"])
        self.assertIsNone(result["performance_claim"])


if __name__ == "__main__":
    unittest.main()
