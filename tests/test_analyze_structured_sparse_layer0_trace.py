import unittest

from tools.analyze_structured_sparse_layer0_trace import (
    SAMPLES,
    nearest_rank,
    summarize_best_pair_per_head_query_oracle,
    summarize_pair,
    validate_ledger_and_timing,
    validate_safety,
)


class StructuredSparseLayer0AnalyzerTests(unittest.TestCase):
    def test_ledger_counts_the_qkv_matrix_but_not_the_rms_vector(self):
        raw = {
            "ledger": {
                "fp8_matrices_expanded": 1,
                "bf16_matrices_expanded": 0,
                "dynamic_activation_values": 65_536 * 4096,
                "actual_process_disk_bytes_read": 1,
                "peak_resident_bytes": 1,
            },
            "complete_wall_ms": 1.0,
        }
        validate_ledger_and_timing(raw)
        raw["ledger"]["bf16_matrices_expanded"] = 1
        with self.assertRaisesRegex(ValueError, "ledger or timing"):
            validate_ledger_and_timing(raw)

    def test_nearest_rank_is_conservative(self):
        self.assertEqual(nearest_rank([4.0, 1.0, 3.0, 2.0], 0.50), 2.0)
        self.assertEqual(nearest_rank([4.0, 1.0, 3.0, 2.0], 0.99), 4.0)

    def test_summary_aggregates_energy_by_position_and_band(self):
        observations = []
        for position in SAMPLES:
            band = (
                "early"
                if position in (63, 127, 255)
                else "final_question"
                if position >= 65_509
                else "interval"
            )
            for head in range(2):
                candidates = []
                for index in range(5):
                    error = 0.005 * (index + 1)
                    candidates.append(
                        {
                            "relative_l2": error,
                            "error_l2": error,
                            "maximum_absolute_error": error,
                            "selected_positions": min(position + 1, 830 + index),
                            "bit_exact_values": 0,
                            "total_values": 128,
                        }
                    )
                observations.append(
                    {
                        "absolute_query_position": position,
                        "band": band,
                        "reference_l2": 1.0,
                        "candidates": candidates,
                    }
                )
        summary = summarize_pair(observations, 0)
        self.assertAlmostEqual(summary["aggregate_relative_l2"], 0.005)
        self.assertAlmostEqual(summary["maximum_position_aggregate_relative_l2"], 0.005)
        self.assertTrue(summary["passes"])
        oracle = summarize_best_pair_per_head_query_oracle(observations)
        self.assertAlmostEqual(oracle["aggregate_relative_l2"], 0.005)
        self.assertEqual(oracle["pair_choice_counts"][0]["observations"], len(observations))
        self.assertTrue(oracle["passes"])

    def test_per_head_query_oracle_chooses_each_minimum_and_can_fail(self):
        observations = []
        for position in SAMPLES:
            band = "early" if position < 256 else "final_question" if position >= 65_509 else "interval"
            for head in range(2):
                best = head
                candidates = []
                for index in range(5):
                    error = 0.06 if index == best else 0.20 + index * 0.01
                    candidates.append(
                        {
                            "relative_l2": error,
                            "error_l2": error,
                            "maximum_absolute_error": error,
                            "selected_positions": min(position + 1, 830 + index),
                            "bit_exact_values": 0,
                            "total_values": 128,
                        }
                    )
                observations.append(
                    {
                        "absolute_query_position": position,
                        "band": band,
                        "reference_l2": 1.0,
                        "candidates": candidates,
                    }
                )
        oracle = summarize_best_pair_per_head_query_oracle(observations)
        self.assertEqual(oracle["pair_choice_counts"][0]["observations"], len(SAMPLES))
        self.assertEqual(oracle["pair_choice_counts"][1]["observations"], len(SAMPLES))
        self.assertAlmostEqual(oracle["aggregate_relative_l2"], 0.06)
        self.assertFalse(oracle["passes"])

    def test_safety_requires_rss_release_and_service_health(self):
        base = {
            "release_boundary": True,
            "system_memory_free_percent": 50,
            "process_resident_bytes": 100,
            "process_physical_footprint_bytes": 100,
            "process_peak_resident_bytes": 100,
            "swap_growth_bytes": 0,
            "new_throttled_pages": 0,
            "protected_service_pids": {"ChatGPT": [1]},
        }
        phases = ["process_start"]
        phases += [f"pw0176_qkv_chunk_{index:03}_released" for index in range(1, 65)]
        phases += [f"pw0176_selector_head_{index:02}_released" for index in range(64)]
        phases += [
            "pw0176_fixture_and_checkpoint_authenticated",
            "pw0176_qkv_weight_decoded",
            "pw0176_qkv_projection_complete",
            "pw0176_observations_complete",
            "pw0176_checkpoint_and_buffers_released",
            "pw0176_final_service_health",
        ]
        snapshots = [dict(base, phase=phase) for phase in phases]
        snapshots[0]["release_boundary"] = False
        summary = validate_safety(snapshots)
        self.assertTrue(summary["final_services_healthy"])
        snapshots[2]["process_resident_bytes"] = 9 * 1024**3
        with self.assertRaisesRegex(ValueError, "Gate-8"):
            validate_safety(snapshots)


if __name__ == "__main__":
    unittest.main()
