import math
import unittest

from tools.analyze_global_attention_sparsity_trace import (
    EXPECTED_OBSERVATIONS,
    FRACTIONS,
    GLOBAL_LAYERS,
    summarize_candidates,
)


class GlobalAttentionSparsityTraceTests(unittest.TestCase):
    def test_summary_uses_norm_aggregation_nearest_rank_and_layer_boundaries(self) -> None:
        observations = []
        for layer in GLOBAL_LAYERS:
            for index in range(2):
                candidates = []
                for fraction in FRACTIONS:
                    relative = 0.01 * (index + 1)
                    candidates.append(
                        {
                            "retained_fraction": fraction,
                            "retained_probability_mass": 0.8 + index * 0.1,
                            "reference_l2": 2.0,
                            "candidate_l2": 2.0 - relative,
                            "error_l2": 2.0 * relative,
                            "relative_l2": relative,
                            "maximum_absolute_error": relative / 2,
                            "bit_exact_values": 0,
                            "total_values": 128,
                        }
                    )
                observations.append({"layer": layer, "candidates": candidates})
        summary = summarize_candidates(observations, 3)
        self.assertEqual(summary["observations"], 18)
        self.assertAlmostEqual(summary["aggregate_relative_l2"], math.sqrt(0.00025))
        self.assertEqual(summary["head_query_relative_l2_p50"], 0.01)
        self.assertEqual(summary["head_query_relative_l2_p99"], 0.02)
        self.assertEqual(summary["retained_probability_mass_p01"], 0.8)
        self.assertEqual(summary["retained_probability_mass_p50"], 0.8)
        self.assertEqual(set(summary["layers"]), {str(layer) for layer in GLOBAL_LAYERS})
        self.assertTrue(
            all(row["observations"] == 2 for row in summary["layers"].values())
        )

    def test_contract_size_is_exact(self) -> None:
        self.assertEqual(EXPECTED_OBSERVATIONS, 8_640)
        self.assertEqual(FRACTIONS[3], 0.2)
        self.assertEqual(FRACTIONS[4], 0.21056139043683178)


if __name__ == "__main__":
    unittest.main()
