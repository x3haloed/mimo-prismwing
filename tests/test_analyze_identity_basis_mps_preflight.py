import unittest
from pathlib import Path

from tools.analyze_identity_basis_mps_preflight import analyze


class IdentityBasisMpsPreflightAnalysisTests(unittest.TestCase):
    def test_real_preflight_closes_optimizer_and_safety_gates(self):
        result = analyze(
            Path("/Users/chad/Models/mimo-prismwing/evidence/PW-0118/run-001.json")
        )
        self.assertTrue(result["gates_passed"])
        self.assertGreater(result["relative_loss_reduction"], 0.70)
        self.assertEqual(result["maximum_mps_current_allocated_bytes"], 1_342_505_216)
        self.assertIsNone(result["performance_claim"])


if __name__ == "__main__":
    unittest.main()
