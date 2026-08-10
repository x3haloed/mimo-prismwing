import copy
import unittest

from tools.analyze_structured_sparse_prefill import (
    EXPECTED_CEILING,
    bounded_causal_pairs,
    configuration_work,
    continuation_ceiling,
)


def fixture_configuration():
    return [
        {str(head): ["vertical_and_slash", 30, 800, 0.99] for head in range(32)}
        for _layer in range(40)
    ]


def fixture_predecessors():
    pw0158 = {
        "evidence_class": "pw0158_million_context_two_p100_attention_ceiling",
        "attention_work_ledger": {
            "global_attention_flops": 184_320_184_320_000_000,
            "sliding_window_attention_flops": 204_459_336_007_680,
        },
    }
    pw0161 = {
        "evidence_class": "pw0161_volta_32gb_complete_system_envelope",
        "positions": 1_000_000,
        "arithmetic": [
            {
                "id": "v100_pcie_32gb",
                "modes": [
                    {
                        "mode": "direct_fp32_control",
                        "mandatory_matrix_plus_attention_flops": 214_165_790_024_007_680,
                        "granted_concurrent_epyc_flops_per_second": 742_400_000_000,
                    }
                ],
            }
        ],
    }
    return pw0158, pw0161


class StructuredSparsePrefillAuditTests(unittest.TestCase):
    def test_bounded_causal_pairs(self):
        self.assertEqual(bounded_causal_pairs(4, 2), 7)
        self.assertEqual(bounded_causal_pairs(4, 10), 10)

    def test_vertical_slash_fixture_is_structurally_below_ceiling(self):
        work = configuration_work(fixture_configuration())
        self.assertEqual(work["head_records"], 1280)
        self.assertLess(
            work["favorable_effective_global_attention_work_fraction_with_index_qk"],
            EXPECTED_CEILING,
        )
        self.assertFalse(work["overlap_deduplicated"])

    def test_malformed_or_foreign_pattern_fails_closed(self):
        config = fixture_configuration()
        config[0]["0"][0] = "block_sparse"
        with self.assertRaisesRegex(ValueError, "unsupported or malformed"):
            configuration_work(config)

    def test_continuation_ceiling_reproduces(self):
        ceiling = continuation_ceiling(*fixture_predecessors())
        self.assertAlmostEqual(
            ceiling["maximum_global_attention_work_fraction"], EXPECTED_CEILING, places=15
        )

    def test_predecessor_drift_fails_closed(self):
        pw0158, pw0161 = fixture_predecessors()
        changed = copy.deepcopy(pw0161)
        changed["arithmetic"][0]["modes"][0]["granted_concurrent_epyc_flops_per_second"] += 1
        with self.assertRaisesRegex(ValueError, "PW-0161"):
            continuation_ceiling(pw0158, changed)


if __name__ == "__main__":
    unittest.main()
