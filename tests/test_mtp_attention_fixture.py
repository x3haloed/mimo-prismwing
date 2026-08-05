import unittest

import numpy as np

from tools.generate_mtp_attention_fixture import dequant4, quantize4, wht


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


if __name__ == "__main__":
    unittest.main()
