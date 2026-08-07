import unittest
from pathlib import Path

from tools.analyze_best_rank_real_expert_control import analyze


class BestRankRealExpertControlAnalysisTests(unittest.TestCase):
    def test_real_control_closes_correctness_and_safety_gates(self):
        result = analyze(
            Path("/Users/chad/Models/mimo-prismwing/evidence/PW-0119/run-001.json")
        )
        self.assertTrue(result["gates_passed"])
        self.assertTrue(result["all_source_oracles_bit_exact"])
        self.assertGreater(
            result["expert_output_relative_l2_range_by_layer_and_rank"]["24"]["768"][
                "minimum_relative_l2"
            ],
            0.70,
        )
        self.assertIsNone(result["performance_claim"])


if __name__ == "__main__":
    unittest.main()
