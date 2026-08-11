import unittest
import numpy as np
import torch
from tools.run_weight_aware_activation_sparsity import sparsify_rows

class ActivationSparsityTest(unittest.TestCase):
    def test_exact_cardinality_stable_ties_and_finite_output(self):
        values=torch.tensor([[1,1,2,2],[1,1,1,1]],dtype=torch.bfloat16)
        actual=sparsify_rows(values,np.ones(4,dtype=np.float32),.5).float().numpy()
        self.assertEqual(np.count_nonzero(actual,axis=1).tolist(),[2,2])
        self.assertEqual(actual[1].tolist(),[1,1,0,0])
        self.assertTrue(np.isfinite(actual).all())

if __name__ == "__main__": unittest.main()
