import unittest

import numpy as np

from tools.generate_mtp_attention_fixture import (
    affine_reconstruct_rotated,
    dequant_affine8,
    dequant4,
    quantize_affine8,
    quantize4,
    wht,
)


class MtpAttentionFixtureTests(unittest.TestCase):
    def test_wht_inverse_and_turbo4_layout_are_deterministic(self) -> None:
        signs1 = np.where(np.arange(128) % 3 == 0, -1, 1).astype(np.float32)
        signs2 = np.where(np.arange(128) % 5 == 0, -1, 1).astype(np.float32)
        values = np.linspace(-2, 3, 256, dtype=np.float32)
        rotated = wht(values, False, signs1, signs2)
        restored = wht(rotated, True, signs1, signs2)
        np.testing.assert_allclose(restored, values, rtol=2e-6, atol=2e-6)
        first = quantize4(values, signs1, signs2)
        second = quantize4(values, signs1, signs2)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 136)
        self.assertTrue(np.isfinite(dequant4(first, 256)).all())

    def test_affine_wht_reconstruction_is_finite_and_improves_with_bits(self) -> None:
        signs1 = np.where(np.arange(128) % 3 == 0, -1, 1).astype(np.float32)
        signs2 = np.where(np.arange(128) % 5 == 0, -1, 1).astype(np.float32)
        values = np.linspace(-2, 3, 256, dtype=np.float32)
        rotated = wht(values, False, signs1, signs2)
        errors = []
        for bits in (4, 5, 6, 8):
            first = affine_reconstruct_rotated(values, bits, signs1, signs2)
            second = affine_reconstruct_rotated(values, bits, signs1, signs2)
            np.testing.assert_array_equal(first, second)
            self.assertTrue(np.isfinite(first).all())
            errors.append(np.linalg.norm(first - rotated))
        self.assertTrue(all(left > right for left, right in zip(errors, errors[1:])))
        with self.assertRaises(ValueError):
            affine_reconstruct_rotated(values, 3, signs1, signs2)

    def test_affine8_packing_has_explicit_scale_and_signed_codes(self) -> None:
        signs1 = np.where(np.arange(128) % 3 == 0, -1, 1).astype(np.float32)
        signs2 = np.where(np.arange(128) % 5 == 0, -1, 1).astype(np.float32)
        values = np.linspace(-2, 3, 256, dtype=np.float32)
        payload = quantize_affine8(values, signs1, signs2)
        self.assertEqual(len(payload), 260)
        reconstructed = dequant_affine8(payload, 256)
        expected = affine_reconstruct_rotated(values, 8, signs1, signs2)
        np.testing.assert_array_equal(reconstructed, expected)
        zeros = quantize_affine8(np.zeros(128, np.float32), signs1, signs2)
        self.assertEqual(zeros, bytes(130))
        with self.assertRaises(ValueError):
            dequant_affine8(payload[:-1], 256)


if __name__ == "__main__":
    unittest.main()
