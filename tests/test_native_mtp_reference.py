import math
import unittest

import torch

from tools.native_mtp_reference import (
    last_row_attention_core,
    q4_proposal_block,
    rotate_mtp_input_ids,
)


class NativeMtpReferenceTests(unittest.TestCase):
    def test_non_chain_input_rotation_and_q4_block(self):
        ids = [10, 11, 12, 13]
        self.assertEqual(rotate_mtp_input_ids(ids, 21), [11, 12, 13, 21])
        self.assertEqual(q4_proposal_block(13, [21, 22, 23]), [13, 21, 22, 23])

    def test_last_row_matches_full_causal_reference(self):
        torch.manual_seed(17)
        rows = 5
        q = torch.randn((rows, 64, 192), dtype=torch.float32).to(torch.bfloat16)
        k = torch.randn((rows, 8, 192), dtype=torch.float32).to(torch.bfloat16)
        v = torch.randn((rows, 8, 128), dtype=torch.float32).to(torch.bfloat16)
        sinks = torch.randn((64,), dtype=torch.float32).to(torch.bfloat16)
        expected = torch.empty((1, 64, 128), dtype=torch.bfloat16)
        scale = 1.0 / math.sqrt(192)
        for head in range(64):
            kv_head = head // 8
            scores = (q[-1, head] @ k[:, kv_head].T) * scale
            scores = torch.cat((scores, sinks[head:head + 1]))
            scores = scores - scores.max()
            probabilities = torch.softmax(scores, dim=-1, dtype=torch.float32).to(torch.bfloat16)
            expected[0, head] = probabilities[:-1] @ v[:, kv_head]
        self.assertTrue(torch.equal(last_row_attention_core(q, k, v, sinks), expected))

    def test_invalid_schedule_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "exactly three"):
            q4_proposal_block(1, [2, 3])
        with self.assertRaisesRegex(ValueError, "nonempty"):
            rotate_mtp_input_ids([], 1)


if __name__ == "__main__":
    unittest.main()
