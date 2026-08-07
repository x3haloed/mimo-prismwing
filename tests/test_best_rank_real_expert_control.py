import unittest

import numpy as np
import torch

from tools.generate_real_layer1_expert_oracle import dynamic_input
from tools.run_best_rank_real_expert_control import parity, svd_control


class BestRankRealExpertControlTests(unittest.TestCase):
    def test_full_rank_gate_and_down_orientations_reconstruct(self):
        matrix = np.diag(np.linspace(0.5, 1.5, 128, dtype=np.float32))
        decomposition = np.linalg.svd(matrix, full_matrices=False)
        gate_input = torch.linspace(-2.0, 2.0, 128).reshape(1, 128).to(torch.bfloat16)
        gate = svd_control(decomposition, gate_input, 128, down=False)
        gate_expected = (
            dynamic_input(gate_input) @ torch.from_numpy(matrix).T
        ).to(torch.bfloat16)
        self.assertEqual(parity(gate, gate_expected)["relative_l2"], 0.0)
        down_input = torch.linspace(1.0, -1.0, 128).reshape(1, 128).to(torch.bfloat16)
        down = svd_control(decomposition, down_input, 128, down=True)
        down_expected = (
            dynamic_input(down_input) @ torch.from_numpy(matrix)
        ).to(torch.bfloat16)
        self.assertEqual(parity(down, down_expected)["relative_l2"], 0.0)


if __name__ == "__main__":
    unittest.main()
