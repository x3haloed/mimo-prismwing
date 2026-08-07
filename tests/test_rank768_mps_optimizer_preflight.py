import unittest

import torch

from tools.run_identity_basis_mps_preflight import IdentityBasis
from tools.run_rank768_mps_optimizer_preflight import (
    BASES,
    PARAMETER_VALUES,
    RANK,
    SEMANTIC_ADAM_BYTES,
)


class Rank768MpsOptimizerPreflightTests(unittest.TestCase):
    def test_production_accounting_and_tiny_rank768_equation(self):
        self.assertEqual(RANK, 768)
        self.assertEqual(BASES, 4)
        self.assertEqual(PARAMETER_VALUES, 415_237_120)
        self.assertEqual(SEMANTIC_ADAM_BYTES, 6_643_793_920)
        torch.manual_seed(260120)
        model = IdentityBasis(2, 3, RANK, BASES, 4, device=torch.device("cpu"))
        expert_ids = torch.tensor([1])
        coefficients = torch.softmax(model.alpha[expert_ids], dim=-1)
        expected = torch.matmul(
            model.a[expert_ids],
            torch.einsum("em,mrc->erc", coefficients, model.b),
        )
        self.assertTrue(torch.equal(model.tile(expert_ids, 3, 4), expected))
        target = torch.full((1, 3, 4), 0.1)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        before = torch.nn.functional.mse_loss(model.tile(expert_ids, 3, 4), target)
        optimizer.zero_grad(set_to_none=True)
        before.backward()
        optimizer.step()
        after = torch.nn.functional.mse_loss(model.tile(expert_ids, 3, 4), target)
        self.assertLess(float(after.detach()), float(before.detach()))


if __name__ == "__main__":
    unittest.main()
