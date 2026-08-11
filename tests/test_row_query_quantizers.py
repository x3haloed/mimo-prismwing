import unittest

try:
    import numpy as np
except ModuleNotFoundError:
    np = None

if np is not None:
    from tools.row_query_quantizers import (
        estimate_mse_inner_products,
        estimate_covariance_inner_products,
        estimate_turbo_prod_inner_products,
        fit_block_covariance_transform,
        fwht_rows,
        inverse_rotate_rows,
        lloyd_max_normal_centroids,
        quantize_rotated_rows,
        quantize_covariance_rows,
        quantize_covariance_shared_block_codebooks,
        rotate_rows,
        seeded_signs,
        turbo_prod_quantize_rows,
    )


@unittest.skipIf(np is None, "NumPy execution toolchain is not installed on this host")
class RowQueryQuantizerTests(unittest.TestCase):
    def test_lloyd_max_grid_is_symmetric_and_ordered(self):
        centroids = lloyd_max_normal_centroids(3)
        self.assertTrue(np.all(np.diff(centroids) > 0))
        np.testing.assert_allclose(centroids, -centroids[::-1], atol=1e-12)

    def test_hadamard_rotation_preserves_inner_products(self):
        values = np.arange(24, dtype=np.float64).reshape(3, 8) - 4
        signs = seeded_signs(8, 17)
        rotated = rotate_rows(values, signs)
        np.testing.assert_allclose(rotated @ rotated.T, values @ values.T, atol=1e-10)
        np.testing.assert_allclose(fwht_rows(fwht_rows(values)), values, atol=1e-10)

    def test_mse_estimator_matches_reconstructed_row_dot(self):
        rng = np.random.default_rng(3)
        rows = rng.normal(size=(5, 16))
        queries = rng.normal(size=(4, 16))
        signs = seeded_signs(16, 9)
        quantized = quantize_rotated_rows(rows, 4, signs)
        reconstructed = inverse_rotate_rows(quantized["reconstructed_rotated"], signs)
        np.testing.assert_allclose(
            estimate_mse_inner_products(queries, quantized, signs),
            queries @ reconstructed.T,
            atol=1e-10,
        )

    def test_turbo_prod_shapes_and_term_identity(self):
        rng = np.random.default_rng(4)
        rows = rng.normal(size=(6, 32))
        queries = rng.normal(size=(7, 32))
        signs = seeded_signs(32, 11)
        qjl_signs = seeded_signs(32, 13)
        quantized = turbo_prod_quantize_rows(rows, signs, qjl_signs)
        total, base, correction = estimate_turbo_prod_inner_products(
            queries, quantized, signs, qjl_signs
        )
        self.assertEqual(total.shape, (7, 6))
        np.testing.assert_allclose(total, base + correction)

    def test_covariance_transform_estimator_shapes(self):
        rng = np.random.default_rng(6)
        train = rng.normal(size=(20, 16))
        rows = rng.normal(size=(5, 16))
        queries = rng.normal(size=(7, 16))
        transform = fit_block_covariance_transform(train, block=8)
        quantized = quantize_covariance_rows(rows, transform, bits=4)
        result = estimate_covariance_inner_products(queries, quantized, transform)
        self.assertEqual(result.shape, (7, 5))
        shared = quantize_covariance_shared_block_codebooks(rows, transform, levels=4)
        shared_result = estimate_covariance_inner_products(queries, shared, transform)
        self.assertEqual(shared_result.shape, (7, 5))


if __name__ == "__main__":
    unittest.main()
