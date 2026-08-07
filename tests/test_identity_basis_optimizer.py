import unittest

import torch

from tools.run_identity_basis_mps_preflight import IdentityBasis


class IdentityBasisOptimizerTests(unittest.TestCase):
    def test_tiny_cpu_forward_and_gradient_update(self):
        torch.manual_seed(260118)
        model = IdentityBasis(2, 3, 2, 2, 4, device=torch.device("cpu"))
        expert_ids = torch.tensor([0, 1])
        target = torch.tensor(
            [[[0.1, -0.2, 0.3, 0.4]] * 3, [[-0.1, 0.2, -0.3, 0.1]] * 3]
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        before = torch.nn.functional.mse_loss(model.tile(expert_ids, 3, 4), target)
        optimizer.zero_grad(set_to_none=True)
        before.backward()
        self.assertIsNotNone(model.a.grad)
        self.assertIsNotNone(model.b.grad)
        self.assertIsNotNone(model.alpha.grad)
        optimizer.step()
        after = torch.nn.functional.mse_loss(model.tile(expert_ids, 3, 4), target)
        self.assertTrue(torch.isfinite(after))
        self.assertLess(float(after.detach()), float(before.detach()))


if __name__ == "__main__":
    unittest.main()
