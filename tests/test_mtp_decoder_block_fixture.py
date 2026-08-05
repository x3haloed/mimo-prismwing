import unittest

import numpy as np

from tools.generate_mtp_decoder_block_fixture import install_scales, rms_norm


class MtpDecoderBlockFixtureTests(unittest.TestCase):
    def test_rms_norm_and_scale_installation_are_explicit(self) -> None:
        values = np.arange(16, dtype=np.float32).reshape(2, 8) - 4
        weight = np.linspace(0.5, 1.5, 8, dtype=np.float32)
        normalized = rms_norm(values, weight)
        expected = values / np.sqrt(np.mean(values * values, axis=-1, keepdims=True) + 1e-5) * weight
        np.testing.assert_allclose(normalized, expected, rtol=2e-7, atol=2e-7)

        raw = np.ones((256, 128), np.float32)
        scales = np.array([[1], [2]], np.float32)
        installed = install_scales(raw, scales)
        self.assertTrue(np.all(installed[:128] == 1))
        self.assertTrue(np.all(installed[128:] == 2))
        with self.assertRaises(ValueError):
            install_scales(raw, np.ones((1, 1), np.float32))


if __name__ == "__main__":
    unittest.main()
