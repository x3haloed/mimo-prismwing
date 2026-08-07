import unittest
from pathlib import Path

from tools.analyze_shared_basis_feasibility import analyze


class SharedBasisFeasibilityTests(unittest.TestCase):
    def test_envelope_rejects_unchanged_down_and_selects_deeper_forms(self):
        result = analyze(
            Path("/Users/chad/Models/mimo-prismwing/evidence/PW-0113/run-001.json")
        )
        self.assertAlmostEqual(
            result["published_mobe_applicability"]["unchanged_down_floor_ratio"],
            1 / 3,
        )
        self.assertFalse(
            result["published_mobe_applicability"]["physical_gate_passed"]
        )
        self.assertTrue(result["all_projection_family_has_physically_eligible_configuration"])
        selected = result["selected_activation_audit_configurations"]
        self.assertEqual(
            [(row["selection_role"], row["rank"], row["basis_count"]) for row in selected],
            [("rank_heavy", 768, 4), ("balanced", 512, 8), ("basis_heavy", 128, 32)],
        )
        self.assertTrue(all(row["eligible"] for row in selected))
        self.assertIsNone(result["performance_claim"])


if __name__ == "__main__":
    unittest.main()
