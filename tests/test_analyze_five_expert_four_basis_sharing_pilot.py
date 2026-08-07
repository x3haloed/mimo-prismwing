import unittest
from pathlib import Path

from tools.analyze_five_expert_four_basis_sharing_pilot import analyze


class FiveExpertFourBasisSharingPilotAnalysisTests(unittest.TestCase):
    def test_real_pilot_validates_the_forced_sharing_rejection(self):
        result = analyze(
            Path("/Users/chad/Models/mimo-prismwing/evidence/PW-0123/run-001.json")
        )
        self.assertTrue(result["evidence_valid"])
        self.assertFalse(result["experiment_passed"])
        self.assertEqual(result["failed_projection_gates"], ["gate", "up", "down"])
        self.assertEqual(result["failed_holdout_experts"], [57])
        self.assertGreater(result["projection_summary"]["up"]["fifth_expert_ratio"], 5.0)
        self.assertTrue(result["physical_eligibility"]["byte_gate_passed"])
        self.assertIsNone(result["performance_claim"])


if __name__ == "__main__":
    unittest.main()
