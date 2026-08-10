import math
import unittest

from tools.analyze_global_attention_sparsity_trace import (
    CORPUS_SHA256,
    EXPECTED_OBSERVATIONS,
    FRACTIONS,
    GLOBAL_LAYERS,
    INPUT_SHA256,
    PW0157_SHA256,
    REVISION,
    ROUTES_SHA256,
    SAMPLE_POSITIONS,
    VERIFICATION_SHA256,
    _validate_raw,
    summarize_candidates,
    validate_safety,
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

    def test_raw_identity_requires_corpus_and_checkpoint_receipt_hashes(self) -> None:
        raw = {
            "schema_version": 1,
            "semantic": "mimo_target_faithful_global_attention_sparsity_shadow_trace",
            "revision": REVISION,
            "fixture_sha256": CORPUS_SHA256,
            "checkpoint_verification_sha256": VERIFICATION_SHA256,
            "pw0157_prefix512_sha256": PW0157_SHA256,
            "traced_prefix_positions": 512,
            "input_token_ids_sha256": INPUT_SHA256,
            "layer_routes_sha256": ROUTES_SHA256,
            "observed_global_layers": list(GLOBAL_LAYERS),
            "sampled_absolute_query_positions": list(SAMPLE_POSITIONS),
            "observed_heads_per_sample": 64,
            "retained_fractions": list(FRACTIONS),
            "batch_size": 1,
            "concurrency": 1,
            "accepted_tokens": 0,
            "performance_claim": None,
            "exactness": "target_faithful_source_state_with_noncausal_L3_shadow_only",
            "candidate_numerics": "source_bf16_probabilities_f32_retained_mass_and_renormalization_source_four_lane_f32_reduction_final_bf16",
            "complete_wall_ms": 1.0,
            "ledger": {"actual_process_disk_bytes_read": 0, "peak_resident_bytes": 1},
            "observations": [],
        }
        with self.assertRaisesRegex(ValueError, "raw observation count mismatch"):
            _validate_raw(raw)
        raw["fixture_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "raw trace identity mismatch"):
            _validate_raw(raw)
        raw["fixture_sha256"] = CORPUS_SHA256
        raw["checkpoint_verification_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "raw trace identity mismatch"):
            _validate_raw(raw)
        raw["checkpoint_verification_sha256"] = VERIFICATION_SHA256
        raw["batch_size"] = 2
        with self.assertRaisesRegex(ValueError, "raw trace identity mismatch"):
            _validate_raw(raw)
        raw["batch_size"] = 1
        raw["complete_wall_ms"] = float("nan")
        with self.assertRaisesRegex(ValueError, "raw trace timing or ledger mismatch"):
            _validate_raw(raw)

    def test_gate8_enforces_peak_rss_as_well_as_footprint(self) -> None:
        snapshots = []
        for phase in ("process_start", "walk_complete", "checkpoint_released", "final_service_health"):
            snapshots.append(
                {
                    "phase": phase,
                    "system_memory_free_percent": 70,
                    "process_physical_footprint_bytes": 1024,
                    "process_peak_resident_bytes": 2048,
                    "swap_growth_bytes": 0,
                    "new_throttled_pages": 0,
                    "protected_service_pids": {"WindowServer": [1]},
                }
            )
        validate_safety(snapshots)
        snapshots[1]["process_peak_resident_bytes"] = 8 * 1024**3 + 1
        with self.assertRaisesRegex(ValueError, "Gate-8 safety violation"):
            validate_safety(snapshots)


if __name__ == "__main__":
    unittest.main()
