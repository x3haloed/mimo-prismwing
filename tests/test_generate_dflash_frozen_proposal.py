from pathlib import Path
import tempfile
import unittest

import torch
from transformers import Qwen3Config

from tools.generate_dflash_frozen_proposal import (
    configure_sglang_full_head_rope,
    tensor_capture,
    verified_file,
)


class FrozenProposalTests(unittest.TestCase):
    def test_sglang_adapter_is_narrow_and_explicit(self):
        config = Qwen3Config(head_dim=128)
        config.partial_rotary_factor = 0.5
        record = configure_sglang_full_head_rope(config)
        self.assertEqual(config.partial_rotary_factor, 1.0)
        self.assertEqual(record["rotary_dim"], 128)
        config.partial_rotary_factor = 0.25
        with self.assertRaisesRegex(ValueError, "exported partial-RoPE"):
            configure_sglang_full_head_rope(config)

    def test_verified_file_requires_complete_stat_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "value"
            path.write_bytes(b"abc")
            stat = path.stat()
            record = {
                "status": "verified",
                "bytes": stat.st_size,
                "device": stat.st_dev,
                "inode": stat.st_ino,
                "modified_ns": stat.st_mtime_ns,
            }
            verified_file(path, record)
            record["bytes"] += 1
            with self.assertRaisesRegex(ValueError, "identity changed"):
                verified_file(path, record)

    def test_tensor_capture_is_exclusive_and_hashes_widened_values(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            record = tensor_capture(root, "hidden", torch.tensor([[1.0]], dtype=torch.bfloat16))
            self.assertEqual(record["shape"], [1, 1])
            self.assertEqual(record["dtype"], "BF16_widened_F32")
            self.assertEqual(len((root / "hidden.f32").read_bytes()), 4)
            with self.assertRaises(FileExistsError):
                tensor_capture(root, "hidden", torch.tensor([[1.0]], dtype=torch.bfloat16))


if __name__ == "__main__":
    unittest.main()
