import hashlib
from pathlib import Path
import struct
import unittest

from tools.prepare_pw0176_fixture import (
    TARGET_TOKENS,
    build_fixture_payload,
    sample_positions,
)


CHECKPOINT = Path("/Users/chad/Models/mimo-prismwing/checkpoints/MiMo-V2.5-63651580")


class PreparePw0176FixtureTests(unittest.TestCase):
    def test_sample_schedule_is_complete_disjoint_and_frozen(self):
        positions = sample_positions()
        self.assertEqual(len(positions), 24)
        self.assertEqual(positions, sorted(set(positions)))
        self.assertEqual(positions[:3], [63, 127, 255])
        self.assertEqual(
            positions[-6:], [65_509, 65_515, 65_520, 65_525, 65_530, 65_535]
        )
        self.assertTrue(all(0 <= position < TARGET_TOKENS for position in positions))

    def test_payload_is_exact_little_endian_authority(self):
        if not (CHECKPOINT / "tokenizer.json").is_file():
            self.skipTest("pinned tokenizer is not installed")
        generation, payload = build_fixture_payload(CHECKPOINT)
        self.assertEqual(len(payload), TARGET_TOKENS * 4)
        self.assertEqual(hashlib.sha256(payload).hexdigest(), generation["token_ids_sha256"])
        first = struct.unpack_from("<I", payload)[0]
        last = struct.unpack_from("<I", payload, len(payload) - 4)[0]
        self.assertEqual(first, generation["first_256_token_ids"][0])
        self.assertEqual(last, generation["last_256_token_ids"][-1])
        self.assertLess(generation["needle_token_offset"], 256)
        self.assertGreaterEqual(generation["question_token_offset"], TARGET_TOKENS - 256)


if __name__ == "__main__":
    unittest.main()
