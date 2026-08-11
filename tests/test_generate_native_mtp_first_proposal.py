import unittest

from tools.generate_native_mtp_first_proposal import (
    PW0206_DECODE_SHA256,
    PW0206_PREFIX_SHA256,
    expected_mtp_inventory,
)


class NativeMTPProposalTests(unittest.TestCase):
    def test_corrected_authorities_are_hash_locked_separately_from_pw0103(self):
        self.assertEqual(len(PW0206_PREFIX_SHA256), 64)
        self.assertEqual(len(PW0206_DECODE_SHA256), 64)
        self.assertNotEqual(PW0206_PREFIX_SHA256, PW0206_DECODE_SHA256)

    def test_inventory_covers_three_exact_dense_mtp_layers(self):
        inventory = expected_mtp_inventory()
        self.assertEqual(len(inventory), 48)
        self.assertEqual(
            inventory["model.mtp.layers.0.self_attn.qkv_proj.weight"],
            ("F8_E4M3", (14848, 4096)),
        )
        self.assertEqual(
            inventory["model.mtp.layers.2.eh_proj.weight"],
            ("BF16", (4096, 8192)),
        )


if __name__ == "__main__":
    unittest.main()
