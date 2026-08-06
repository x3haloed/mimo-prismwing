import unittest

import torch

from tools.generate_incremental_cache_oracle import (
    LayerCache,
    apply_rope_at,
    validate_cache,
    visible_start,
)
from tools.generate_real_layer0_bf16_oracle import apply_rope


class IncrementalCacheOracleTests(unittest.TestCase):
    def test_absolute_rope_matches_last_whole_sequence_row(self):
        values = torch.arange(28 * 2 * 192, dtype=torch.float32).reshape(28, 2, 192)
        values = (values / 8192).to(torch.bfloat16)
        expected = apply_rope(values, 10_000.0)[-1:]
        actual = apply_rope_at(values[-1:], 10_000.0, 27)
        self.assertTrue(torch.equal(actual, expected))

    def test_visibility_and_cache_authority_fail_closed(self):
        self.assertEqual(visible_start(False, 28), 0)
        self.assertEqual(visible_start(True, 28), 0)
        self.assertEqual(visible_start(True, 129), 1)
        cache = LayerCache(
            torch.zeros((28, 8, 192), dtype=torch.bfloat16),
            torch.zeros((28, 8, 128), dtype=torch.bfloat16),
        )
        validate_cache(cache, 28, 8)
        cache.keys[0, 0, 0] = float("nan")
        with self.assertRaises(ValueError):
            validate_cache(cache, 28, 8)
        with self.assertRaises(ValueError):
            visible_start(True, 0)


if __name__ == "__main__":
    unittest.main()
