import unittest

import numpy as np
import torch

from tools.run_five_expert_four_basis_sharing_pilot import (
    BASES,
    D,
    EXPERTS,
    P,
    RANK,
    _equal_expert_loss,
    _full_bank_ledger,
    _select_fifth_basis_from_train,
    _shared_predict,
)


class FiveExpertFourBasisSharingPilotTests(unittest.TestCase):
    def test_topology_forces_sharing_and_equation_matches_reference(self):
        self.assertGreater(len(EXPERTS), BASES)
        values = len(EXPERTS) * P * RANK + BASES * RANK * D + len(EXPERTS) * BASES
        self.assertEqual(values, 20_447_252)
        self.assertEqual(values * 4 * 4, 327_156_032)
        torch.manual_seed(260123)
        a = torch.randn(5, 3, 2, requires_grad=True)
        b = torch.randn(4, 2, 6, requires_grad=True)
        coefficients = torch.randn(5, 4, requires_grad=True)
        inputs = torch.randn(7, 6)
        actual = _shared_predict("gate", 4, inputs, a, b, coefficients)
        combined = sum(coefficients[4, index] * b[index] for index in range(4))
        expected = inputs @ combined.T @ a[4].T
        self.assertTrue(torch.allclose(actual, expected))
        down_values = torch.randn(7, 3)
        down_actual = _shared_predict("down", 4, down_values, a, b, coefficients)
        down_expected = down_values @ a[4] @ combined
        self.assertTrue(torch.allclose(down_actual, down_expected))
        loss = actual.square().mean()
        loss.backward()
        self.assertGreater(float(coefficients.grad.abs().sum()), 0.0)

    def test_equal_expert_loss_does_not_weight_row_count(self):
        predictions = [torch.ones(1, 2), torch.full((10, 2), 2.0)]
        targets = [torch.full((1, 2), 2.0), torch.ones(10, 2)]
        actual = _equal_expert_loss(predictions, targets)
        expected = torch.tensor((0.25 + 1.0) / 2.0)
        self.assertTrue(torch.allclose(actual, expected))

    def test_fifth_basis_selection_reads_train_rows_only(self):
        left = np.eye(2, dtype=np.float32)
        bases = np.stack(
            [
                np.eye(2, dtype=np.float32),
                np.asarray([[2.0, 0.0], [0.0, 2.0]], dtype=np.float32),
            ]
        )
        inputs = torch.eye(2)
        targets = torch.eye(2)
        selected, losses = _select_fifth_basis_from_train(
            "gate", left, bases, inputs, targets, [0]
        )
        changed_targets = targets.clone()
        changed_targets[1] = 1000.0
        changed, changed_losses = _select_fifth_basis_from_train(
            "gate", left, bases, inputs, changed_targets, [0]
        )
        self.assertEqual(selected, 0)
        self.assertEqual(changed, selected)
        self.assertEqual(changed_losses, losses)

    def test_full_bank_physical_hypothesis_closes_frozen_algebra_gates(self):
        ledger = _full_bank_ledger()
        self.assertTrue(ledger["byte_gate_passed"])
        self.assertTrue(ledger["multiplication_gate_passed"])
        self.assertLess(ledger["candidate_to_source_byte_ratio"], 0.20)
        self.assertEqual(ledger["candidate_to_source_multiplication_ratio"], 0.3753662109375)


if __name__ == "__main__":
    unittest.main()
