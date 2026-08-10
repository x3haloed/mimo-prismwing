import unittest

from tools.analyze_prompt_calibrated_p100_hbm_cache import (
    EXPERT_BYTES,
    expert_cache_capacity,
    kv_capacity_bytes,
    prompt_frequency_cache,
    storage_scenarios,
)


def tiny_config():
    return {
        "hybrid_layer_pattern": [0] * 8 + [1] * 40,
        "num_key_value_heads": 4,
        "head_dim": 192,
        "v_head_dim": 128,
        "swa_num_key_value_heads": 8,
        "swa_head_dim": 192,
        "swa_v_head_dim": 128,
        "sliding_window_size": 128,
    }


class PromptCalibratedP100HbmCacheTests(unittest.TestCase):
    def test_8k_kv_ledger_counts_full_and_sliding_layers(self):
        ledger = kv_capacity_bytes(tiny_config())
        self.assertEqual(ledger["full_attention_layers"], 8)
        self.assertEqual(ledger["sliding_window_layers"], 40)
        self.assertEqual(ledger["total_bytes"], 190_054_400)

    def test_aggregate_hbm_capacity_uses_only_complete_experts(self):
        ledger = expert_cache_capacity(12_814_555_472, 190_054_400)
        self.assertEqual(ledger["complete_expert_slots"], 661)
        self.assertEqual(ledger["expert_cache_bytes"], 661 * EXPERT_BYTES)
        self.assertLess(ledger["unallocated_tail_bytes"], EXPERT_BYTES)

    def test_prompt_frequency_cache_never_learns_from_suffix(self):
        routes = []
        for layer in range(47):
            prompt = [[layer % 8] * 8 for _ in range(87)]
            suffix = [[(layer + 1) % 8] * 8 for _ in range(137)]
            routes.append(prompt + suffix)
        # The synthetic duplicate rows are acceptable to this pure policy test.
        result = prompt_frequency_cache(routes, 47)
        self.assertEqual(result["suffix_union_hits"], 0)
        self.assertEqual(result["suffix_union_misses"], 47)

    def test_one_fast_lane_cannot_reach_formal_target_even_perfectly(self):
        rows = storage_scenarios(10_672_914_432)
        one = next(
            row
            for row in rows
            if row["lanes"] == 1
            and row["granted_nameplate_bytes_per_second_per_lane"] == 3.5e9
        )
        four = next(
            row
            for row in rows
            if row["lanes"] == 4
            and row["granted_nameplate_bytes_per_second_per_lane"] == 3.5e9
        )
        self.assertFalse(one["targets"]["50.0"]["possible_with_A_at_most_q"])
        self.assertEqual(four["targets"]["34.3"]["minimum_integer_A"], 34)
        self.assertEqual(four["targets"]["50.0"]["minimum_integer_A"], 49)

    def test_capacity_fails_when_no_complete_expert_fits(self):
        with self.assertRaisesRegex(ValueError, "no complete expert"):
            expert_cache_capacity(29_000_000_000, 1_000_000_000)


if __name__ == "__main__":
    unittest.main()
