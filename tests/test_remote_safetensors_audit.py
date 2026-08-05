import unittest

from tools.remote_safetensors_audit import sample_offsets


class RemoteSafetensorsAuditTests(unittest.TestCase):
    def test_sample_offsets_cover_start_middle_and_end(self):
        self.assertEqual(sample_offsets(100, 10), [0, 45, 90])

    def test_duplicate_offsets_are_removed_for_small_payload(self):
        self.assertEqual(sample_offsets(10, 10), [0])

    def test_invalid_sample_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "sample must fit"):
            sample_offsets(10, 11)


if __name__ == "__main__":
    unittest.main()
