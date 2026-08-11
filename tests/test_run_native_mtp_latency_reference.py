import unittest

import torch

from tools.dflash_semantics import verify_greedy_block
from tools.run_native_mtp_latency_reference import endpoint_committed_tokens


class NativeMtpEndpointAccountingTests(unittest.TestCase):
    def test_full_convergence_excludes_known_anchor(self):
        verification = verify_greedy_block(
            torch.tensor([[10, 20, 30, 40]], dtype=torch.long),
            torch.tensor([[20, 30, 40, 50]], dtype=torch.long),
        )
        self.assertEqual(verification.accepted_length_a, 4)
        self.assertEqual(endpoint_committed_tokens(verification), 3)

    def test_immediate_rejection_commits_correction(self):
        verification = verify_greedy_block(
            torch.tensor([[10, 20, 30, 40]], dtype=torch.long),
            torch.tensor([[99, 30, 40, 50]], dtype=torch.long),
        )
        self.assertEqual(verification.accepted_length_a, 1)
        self.assertEqual(endpoint_committed_tokens(verification), 1)


if __name__ == "__main__":
    unittest.main()
