import math
import unittest

from tools.analyze_fp8_symbol_census import (
    decode_e4m3fn,
    full_expert_analysis,
    greedy_codebook,
    relative_l2,
)


class AnalyzeFp8SymbolCensusTests(unittest.TestCase):
    def test_e4m3fn_reference_values(self):
        self.assertEqual(decode_e4m3fn(0x00), 0.0)
        self.assertEqual(decode_e4m3fn(0x38), 1.0)
        self.assertEqual(decode_e4m3fn(0x7E), 448.0)
        self.assertTrue(math.isnan(decode_e4m3fn(0x7F)))

    def test_greedy_codebook_preserves_supported_symbols(self):
        counts = [0] * 256
        counts[0x38] = 10
        counts[0x40] = 5
        codebook = greedy_codebook(counts, 2)
        self.assertEqual(relative_l2(counts, codebook), 0.0)

    def test_full_expert_analysis_requires_and_aggregates_complete_projections(self):
        counts = [0] * 256
        counts[0x38] = 128 * 128
        block = {
            "symbol_counts": counts,
            "affine6_rtn_squared_error": 1.0,
            "reference_squared_sum": float(128 * 128),
        }
        samples = []
        for projection in ("gate_proj", "up_proj", "down_proj"):
            samples.append({
                "tensor": f"model.layers.4.mlp.experts.96.{projection}.weight",
                "row_block": 0,
                "shape": [128, 128],
                "fetched_bytes": 128 * 128,
                "scale_values": [2.0],
                "blocks": [block],
            })
        result = full_expert_analysis({"samples": samples})
        self.assertEqual(len(result["projection_records"]), 3)
        self.assertEqual(len(result["expert_records"]), 1)
        self.assertEqual(result["expert_records"][0]["fp8_subset_6bit_relative_l2"], 0.0)
        self.assertAlmostEqual(
            result["expert_records"][0]["affine6_rtn_relative_l2"],
            1.0 / math.sqrt(128 * 128),
        )


if __name__ == "__main__":
    unittest.main()
