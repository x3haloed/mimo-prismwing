import unittest

from tools.remote_fp8_symbol_census import (
    BLOCK,
    analyze_block,
    analyze_row_tile,
    parse_sample,
)


class RemoteFp8SymbolCensusTests(unittest.TestCase):
    def test_constant_block_has_exact_palette_and_zero_entropy(self):
        result = analyze_block(bytes([17]) * (BLOCK * BLOCK))
        self.assertEqual(result["distinct_symbols"], 1)
        self.assertEqual(result["symbol_counts"][17], BLOCK * BLOCK)
        self.assertEqual(result["entropy_bits_per_weight"], 0.0)
        self.assertEqual(result["exact_palette_bytes"]["4"], 8193)
        self.assertEqual(result["exact_split_bits_per_weight"], 5)
        self.assertEqual(result["exponent_top_coverage"]["7"], 1.0)

    def test_full_alphabet_rejects_sub_byte_exact_palettes(self):
        valid = bytes(value for value in range(256) if value not in (0x7F, 0xFF))
        result = analyze_block((valid * 65)[: BLOCK * BLOCK])
        self.assertEqual(result["distinct_symbols"], 254)
        self.assertGreater(result["entropy_bits_per_weight"], 7.9)
        self.assertEqual(result["distinct_exponents"], 16)
        self.assertEqual(result["exact_split_bits_per_weight"], 8)
        self.assertTrue(all(value is None for value in result["exact_palette_bytes"].values()))

    def test_row_tile_splits_columns_into_quantization_blocks(self):
        codes = list(range(127)) + [128]
        rows = [bytes([codes[row]]) * (BLOCK * 2) for row in range(BLOCK)]
        result = analyze_row_tile(b"".join(rows), BLOCK * 2)
        self.assertEqual([item["column_block"] for item in result], [0, 1])
        self.assertEqual([item["distinct_symbols"] for item in result], [128, 128])

    def test_sample_uses_final_colon_for_row_block(self):
        self.assertEqual(parse_sample("model.layers.1.weight:3"), ("model.layers.1.weight", 3))


if __name__ == "__main__":
    unittest.main()
