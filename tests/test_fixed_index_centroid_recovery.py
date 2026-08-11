import unittest
import torch

from tools.run_fixed_index_centroid_recovery import torch_decode


class FixedIndexCentroidRecoveryTest(unittest.TestCase):
    def test_fixed_indices_propagate_finite_centroid_gradients(self):
        indices = torch.tensor([[0, 1], [1, 0]], dtype=torch.int64)
        codebooks = torch.nn.Parameter(torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]))
        before = torch_decode(indices, codebooks).detach().clone()
        loss = torch.sum(torch_decode(indices, codebooks) ** 2)
        loss.backward()
        self.assertTrue(torch.isfinite(codebooks.grad).all())
        with torch.no_grad(): codebooks -= 0.01 * codebooks.grad
        self.assertFalse(torch.equal(before, torch_decode(indices, codebooks)))


if __name__ == "__main__": unittest.main()
