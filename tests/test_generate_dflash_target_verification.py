import unittest

import torch

from tools.generate_dflash_target_verification import (
    apply_rope_positions,
    jacobi_successor_block,
    kv_ledger,
)


class TargetVerificationTests(unittest.TestCase):
    def test_jacobi_successor_preserves_anchor_and_shifts_posterior(self):
        proposed = [264, 10, 11, 12, 13, 14, 15, 16]
        posterior = [20, 21, 22, 23, 24, 25, 26, 27]
        self.assertEqual(
            jacobi_successor_block(proposed, posterior),
            [264, 20, 21, 22, 23, 24, 25, 26],
        )

    def test_block_rope_preserves_unrotated_tail_and_uses_absolute_positions(self):
        values = torch.arange(2 * 3 * 192, dtype=torch.float32).reshape(2, 3, 192).to(torch.bfloat16)
        actual = apply_rope_positions(values, 10000.0, 0)
        self.assertTrue(torch.equal(actual[0], values[0]))
        self.assertTrue(torch.equal(actual[:, :, 64:], values[:, :, 64:]))
        self.assertFalse(torch.equal(actual[1, :, :64], values[1, :, :64]))

    def test_kv_ledger_increases_for_prefix_visibility(self):
        no_prefix = kv_ledger(8)
        with_prefix = kv_ledger(8, 27)
        self.assertEqual(no_prefix["cache_write_bytes"], with_prefix["cache_write_bytes"])
        self.assertGreater(
            with_prefix["attention_cache_read_bytes"], no_prefix["attention_cache_read_bytes"]
        )


if __name__ == "__main__":
    unittest.main()
