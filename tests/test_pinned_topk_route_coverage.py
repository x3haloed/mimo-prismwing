import unittest

from tools.analyze_pinned_topk_route_coverage import (
    KV_CACHE_BYTES_PER_POSITION,
    compare_route_prefix,
    compare_kv_release_control,
    compact_sha256,
    validate_route_rows,
)


def trace(layer: int, positions: int, selected: list[list[int]]) -> dict:
    routed = layer != 0
    return {
        "layer": layer,
        "attention": "full" if layer in (0, 5, 11, 17, 23, 29, 35, 41, 47) else "sliding_window_128",
        "cache_length": positions,
        "selected_experts_by_position": selected if routed else [],
        "route_weights_by_position": [[0.125] * 8 for _ in selected] if routed else [],
        "expert_union_factor": None,
        "wall_ms": 1.0,
    }


class PinnedTopkRouteCoverageTests(unittest.TestCase):
    def test_kv_release_byte_ledger_covers_all_pinned_attention_layers(self) -> None:
        self.assertEqual(KV_CACHE_BYTES_PER_POSITION, 445_440)

    def test_kv_release_control_ignores_only_timing(self) -> None:
        control = [trace(layer, 1, [list(range(8))]) for layer in range(48)]
        candidate = [dict(row) for row in control]
        for row in candidate:
            row["wall_ms"] = 9.0
        self.assertTrue(compare_kv_release_control(control, candidate)["exact"])
        candidate[7] = dict(candidate[7])
        candidate[7]["selected_experts_by_position"] = [list(range(1, 9))]
        with self.assertRaisesRegex(ValueError, "changed route semantics"):
            compare_kv_release_control(control, candidate)

    def test_rust_compatible_token_hash_is_compact_json(self) -> None:
        self.assertEqual(
            compact_sha256([1, 20, 300]),
            "789be4f68ce5c3bc0e01d0e19e3a12e82496b528eec3f76bdc798fa1842463e6",
        )

    def test_route_prefix_comparison_quantifies_shape_sensitivity(self) -> None:
        row_a = list(range(8))
        row_b = list(range(8, 16))
        earlier = [trace(layer, 1, [row_a]) for layer in range(48)]
        later = [trace(layer, 2, [row_a, row_b]) for layer in range(48)]
        comparison = compare_route_prefix(earlier, later, 1)
        self.assertEqual(comparison["exact_selected_order_fraction"], 1.0)
        self.assertEqual(comparison["exact_weight_row_fraction"], 1.0)
        later[17]["route_weights_by_position"][0][0] += 0.001
        later[23]["selected_experts_by_position"][0] = list(range(1, 9))
        comparison = compare_route_prefix(earlier, later, 1)
        self.assertEqual(comparison["exact_selected_order_rows"], 46)
        self.assertEqual(comparison["exact_selected_set_rows"], 46)
        self.assertEqual(comparison["exact_weight_rows"], 46)
        self.assertEqual(comparison["exact_expert_weight_mapping_rows"], 45)
        self.assertAlmostEqual(
            comparison["maximum_common_expert_weight_absolute_difference"], 0.001
        )
        self.assertEqual(
            comparison["first_selected_order_divergence"], {"layer": 23, "position": 0}
        )
        self.assertEqual(comparison["first_weight_divergence"], {"layer": 17, "position": 0})

    def test_route_validation_recomputes_distinct_layer_experts(self) -> None:
        rows = [list(range(8)), list(range(4, 12))]
        traces = [trace(layer, 2, rows) for layer in range(48)]
        manifest = {"layer_traces": traces}
        observed = validate_route_rows(manifest, 2)
        self.assertEqual(len(observed), 47 * 12)
        traces[7]["selected_experts_by_position"][0][7] = 0
        with self.assertRaisesRegex(ValueError, "invalid routed expert"):
            validate_route_rows(manifest, 2)


if __name__ == "__main__":
    unittest.main()
