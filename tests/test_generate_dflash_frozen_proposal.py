import hashlib
from pathlib import Path
import tempfile
import unittest

import torch
from transformers import Qwen3Config

from tools.generate_dflash_frozen_proposal import (
    assemble_block_noise_embeddings,
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
                "sha256": hashlib.sha256(b"abc").hexdigest(),
            }
            verified_file(path, record)
            record["device"] += 1
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

    def test_exported_mask_replaces_only_masked_positions(self):
        anchor = torch.tensor([[1, 2, 3, 4]], dtype=torch.bfloat16)
        target_mask = torch.tensor([[0.5, 0, 0, 0]], dtype=torch.bfloat16)
        exported_mask = torch.ones((1, 4), dtype=torch.bfloat16)
        noise, record = assemble_block_noise_embeddings(
            anchor, target_mask, exported_mask
        )
        self.assertEqual(noise.shape, (1, 8, 4))
        torch.testing.assert_close(noise[0, 0], anchor[0])
        torch.testing.assert_close(noise[0, 1:], exported_mask.expand(7, -1))
        self.assertTrue(record["exported_mask_embedding_used"])
        self.assertGreater(
            record["comparison_to_base_target_row"]["relative_l2_to_target_row"],
            1,
        )

    def test_exported_mask_validation_fails_closed(self):
        anchor = torch.ones((1, 4), dtype=torch.bfloat16)
        target_mask = torch.ones((1, 4), dtype=torch.bfloat16)
        with self.assertRaisesRegex(ValueError, "tensor identity mismatch"):
            assemble_block_noise_embeddings(
                anchor, target_mask, torch.ones((1, 5), dtype=torch.bfloat16)
            )


if __name__ == "__main__":
    unittest.main()
