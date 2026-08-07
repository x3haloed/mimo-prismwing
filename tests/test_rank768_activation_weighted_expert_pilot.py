import unittest

import numpy as np
import torch

from tools.run_rank768_activation_weighted_expert_pilot import (
    balanced_factors,
    factor_expert,
    normalized_mse,
)


class Rank768ActivationWeightedExpertPilotTests(unittest.TestCase):
    def test_balanced_factors_and_complete_expert_equation(self):
        diagonal = np.linspace(0.5, 1.5, 128, dtype=np.float32)
        matrix = np.diag(diagonal)
        factors = balanced_factors(np.linalg.svd(matrix, full_matrices=False), 128)
        self.assertTrue(np.allclose(factors[0] @ factors[1], matrix, atol=1e-6))
        expert = {"gate": factors, "up": factors, "down": factors}
        values = torch.linspace(-1.0, 1.0, 128).reshape(1, 128).to(torch.bfloat16)
        output = factor_expert(expert, values)
        self.assertEqual(tuple(output.shape), (1, 128))
        self.assertTrue(torch.isfinite(output.float()).all())

    def test_normalized_loss_improves_under_one_update(self):
        torch.manual_seed(260121)
        left = torch.nn.Parameter(torch.randn(8, 4) * 0.01)
        right = torch.nn.Parameter(torch.randn(4, 16) * 0.01)
        inputs = torch.randn(6, 16)
        targets = torch.randn(6, 8)
        optimizer = torch.optim.Adam([left, right], lr=0.01)
        before = normalized_mse((inputs @ right.T) @ left.T, targets)
        optimizer.zero_grad(set_to_none=True)
        before.backward()
        optimizer.step()
        after = normalized_mse((inputs @ right.T) @ left.T, targets)
        self.assertLess(float(after.detach()), float(before.detach()))


if __name__ == "__main__":
    unittest.main()
