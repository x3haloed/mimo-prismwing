import copy
import unittest

from tools.analyze_million_context_p100_attention_bound import (
    attention_ledger,
    kv_ledger,
    peak_ceiling,
)


def pinned_geometry():
    return {
        "hybrid_layer_pattern": [
            0 if layer in {0, 5, 11, 17, 23, 29, 35, 41, 47} else 1
            for layer in range(48)
        ],
        "num_attention_heads": 64,
        "num_key_value_heads": 4,
        "head_dim": 192,
        "v_head_dim": 128,
        "swa_num_key_value_heads": 8,
        "swa_head_dim": 192,
        "swa_v_head_dim": 128,
        "sliding_window_size": 128,
    }


class MillionContextP100AttentionBoundTests(unittest.TestCase):
    def test_exact_attention_arithmetic(self):
        ledger = attention_ledger(pinned_geometry())
        self.assertEqual(ledger["causal_pairs_per_global_layer"], 500_000_500_000)
        self.assertEqual(ledger["causal_pairs_per_sliding_layer"], 127_991_872)
        self.assertEqual(ledger["global_attention_flops"], 184_320_184_320_000_000)
        self.assertEqual(ledger["sliding_window_attention_flops"], 204_459_336_007_680)
        self.assertEqual(ledger["mandatory_attention_flops"], 184_524_643_656_007_680)

    def test_advertised_fp16_ceiling_still_exceeds_gate(self):
        mandatory = attention_ledger(pinned_geometry())["mandatory_attention_flops"]
        ceiling = peak_ceiling(mandatory, 37.4e12, "two_p100_fp16")
        self.assertAlmostEqual(ceiling["attention_only_floor_seconds"], 4933.8140014975315)
        self.assertFalse(ceiling["passes_1800_second_complete_prefill_gate"])

    def test_exact_bf16_kv_and_generous_streaming_capacity(self):
        ledger = kv_ledger(pinned_geometry())
        self.assertEqual(ledger["total_kv_bytes"], 23_065_559_040)
        self.assertEqual(ledger["full_reservation_over_hbm_bytes"], 6_221_107_536)
        self.assertEqual(ledger["free_streaming_non_routed_complete_expert_slots"], 261)
        self.assertEqual(ledger["free_streaming_non_routed_unallocated_tail_bytes"], 23_564_288)

    def test_changed_layer_schedule_fails_closed(self):
        config = copy.deepcopy(pinned_geometry())
        config["hybrid_layer_pattern"][0], config["hybrid_layer_pattern"][1] = 1, 0
        with self.assertRaisesRegex(ValueError, "global attention schedule"):
            attention_ledger(config)

    def test_changed_geometry_fails_closed(self):
        config = copy.deepcopy(pinned_geometry())
        config["num_attention_heads"] = 32
        with self.assertRaisesRegex(ValueError, "attention geometry"):
            attention_ledger(config)


if __name__ == "__main__":
    unittest.main()
