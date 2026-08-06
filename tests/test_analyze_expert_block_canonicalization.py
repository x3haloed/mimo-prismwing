import unittest

from tools.analyze_expert_block_canonicalization import (
    BLOCK_BYTES,
    components,
    extract_column_block,
    extract_row_block,
    reconstruct_tensors,
    scatter_column_block,
    xor_bytes,
)


class ExpertBlockCanonicalizationTests(unittest.TestCase):
    def test_xor_is_exactly_reversible_and_length_checked(self):
        left = bytes(range(256)) * (BLOCK_BYTES // 256) + bytes(range(BLOCK_BYTES % 256))
        right = bytes(reversed(range(256))) * (BLOCK_BYTES // 256) + bytes(
            reversed(range(256 - BLOCK_BYTES % 256, 256))
        )
        residual = xor_bytes(left, right)
        self.assertEqual(xor_bytes(residual, right), left)
        with self.assertRaises(ValueError):
            xor_bytes(b"a", b"bb")

    def test_components_fail_closed_on_wrong_block_size(self):
        with self.assertRaises(ValueError):
            components(b"\0" * (BLOCK_BYTES - 1))

    def test_inverse_reconstruction_rejects_wrong_block_count(self):
        with self.assertRaises(ValueError):
            reconstruct_tensors([])

    def test_tiny_row_and_column_blocks_round_trip_without_aliasing(self):
        rows = memoryview(bytes(range(24)))
        self.assertEqual(extract_row_block(rows, 1, 2, 4), bytes(range(8, 16)))
        matrix = memoryview(bytes(range(24)))
        block = extract_column_block(matrix, 1, 4, 6, 2, 1)
        self.assertEqual(block, bytes([2, 3, 8, 9, 14, 15, 20, 21]))
        destination = bytearray(24)
        scatter_column_block(destination, memoryview(block), 1, 4, 6, 2, 1)
        self.assertEqual(
            destination,
            bytes([0, 0, 2, 3, 0, 0, 0, 0, 8, 9, 0, 0, 0, 0, 14, 15, 0, 0, 0, 0, 20, 21, 0, 0]),
        )
        with self.assertRaises(ValueError):
            extract_column_block(matrix, 3, 4, 6, 2, 1)


if __name__ == "__main__":
    unittest.main()
