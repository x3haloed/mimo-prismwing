import unittest
from pathlib import Path

from tools.analyze_routed_residual_subspace_oracle import analyze


class RoutedResidualSubspaceOracleAnalysisTests(unittest.TestCase):
    def test_real_oracle_validates_rejection_without_unsealing_holdout(self):
        result = analyze(
            Path("/Users/chad/Models/mimo-prismwing/evidence/PW-0126/run-001.json")
        )
        self.assertTrue(result["evidence_valid"])
        self.assertFalse(result["experiment_passed"])
        self.assertTrue(result["holdout_remained_sealed"])
        self.assertEqual([row["layer"] for row in result["layer_summaries"]], [4, 24, 46])
        self.assertGreater(
            result["layer_summaries"][-1]["validation_rank111"][
                "aggregate_relative_l2"
            ],
            0.38,
        )
        self.assertIsNone(result["performance_claim"])


if __name__ == "__main__":
    unittest.main()
