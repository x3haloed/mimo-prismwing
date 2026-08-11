import unittest

import numpy as np

from tools.run_pw0304_polar_row_probe import factor_rows, signed_hadamard


class PW0215PolarRowProbeTests(unittest.TestCase):
    def test_signed_hadamard_preserves_inner_products(self):
        rng = np.random.default_rng(1)
        left = rng.normal(size=(5, 16)).astype(np.float32)
        right = rng.normal(size=(7, 16)).astype(np.float32)
        signs = rng.choice(np.array([-1, 1], dtype=np.float32), size=16)
        expected = left @ right.T
        actual = signed_hadamard(left, signs) @ signed_hadamard(right, signs).T
        np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-6)

    def test_factorization_has_one_root_and_n_minus_one_angles(self):
        rows = np.arange(32, dtype=np.float32).reshape(2, 16) - 8
        roots, levels = factor_rows(rows)
        self.assertEqual(roots.shape, (2,))
        self.assertEqual([x.shape[1] for x in levels], [8, 4, 2, 1])
        self.assertEqual(sum(x.shape[1] for x in levels), 15)
        np.testing.assert_allclose(roots, np.linalg.norm(rows, axis=1), rtol=1e-6)


if __name__ == "__main__":
    unittest.main()
