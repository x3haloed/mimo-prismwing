import unittest

from tools.merge_fp8_symbol_censuses import merge


class MergeFp8SymbolCensusesTests(unittest.TestCase):
    def record(self, row_block):
        return {
            "schema_version": 1,
            "evidence_class": "pinned_remote_deterministic_fp8_row_tile_samples",
            "repository": "example/model",
            "revision": "abc",
            "network_bytes": 10,
            "sample_count": 1,
            "quantization_block_count": 1,
            "samples": [{
                "tensor": "model.layers.0.mlp.experts.0.gate_proj.weight",
                "row_block": row_block,
                "blocks": [{}],
            }],
        }

    def test_merge_sums_ledger_and_sorts_samples(self):
        result = merge([self.record(1), self.record(0)])
        self.assertEqual(result["network_bytes"], 20)
        self.assertEqual(result["sample_count"], 2)
        self.assertEqual(result["quantization_block_count"], 2)
        self.assertEqual([sample["row_block"] for sample in result["samples"]], [0, 1])

    def test_merge_rejects_duplicate_samples(self):
        with self.assertRaises(ValueError):
            merge([self.record(0), self.record(0)])


if __name__ == "__main__":
    unittest.main()
