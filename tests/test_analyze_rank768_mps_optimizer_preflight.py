import unittest
from pathlib import Path

from tools.analyze_rank768_mps_optimizer_preflight import analyze


class Rank768MpsOptimizerPreflightAnalysisTests(unittest.TestCase):
    def test_real_preflight_validates_the_bounded_rejection(self):
        result = analyze(
            Path("/Users/chad/Models/mimo-prismwing/evidence/PW-0120/run-001.json")
        )
        self.assertTrue(result["evidence_valid"])
        self.assertFalse(result["experiment_passed"])
        self.assertTrue(result["live_safety_preserved"])
        self.assertFalse(result["release_gate_passed"])
        self.assertEqual(
            result["memory_by_phase"]["optimizer_and_parameters_released"][
                "current_allocated_bytes"
            ],
            0,
        )
        self.assertIsNone(result["performance_claim"])


if __name__ == "__main__":
    unittest.main()
