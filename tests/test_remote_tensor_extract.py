import json
from pathlib import Path
import struct
import tempfile
import unittest

from safetensors import safe_open

from tools.remote_tensor_extract import materialize, materialize_local
from tools.extract_selected_experts import group_by_shard, selected_tensor_names


class RemoteTensorExtractTests(unittest.TestCase):
    def test_selected_expert_names_are_complete_and_grouped_by_index(self):
        names = selected_tensor_names(2, [7])
        self.assertEqual(len(names), 6)
        index = {"weight_map": {name: f"shard-{offset % 2}" for offset, name in enumerate(names)}}
        grouped = group_by_shard(index, names)
        self.assertEqual(set(grouped), {"shard-0", "shard-1"})
        self.assertEqual(sum(map(len, grouped.values())), 6)

    def test_selected_tensors_are_lossless_and_fail_closed(self):
        header = {
            "tensor.b": {"dtype": "U8", "shape": [3], "data_offsets": [4, 7]},
            "tensor.a": {"dtype": "U8", "shape": [4], "data_offsets": [0, 4]},
        }
        raw_header = json.dumps(header, separators=(",", ":")).encode()
        source = struct.pack("<Q", len(raw_header)) + raw_header + bytes([1, 2, 3, 4, 9, 8, 7])
        lock = {
            "schema_version": 1,
            "repository": "owner/model",
            "revision": "a" * 40,
            "files": [{"path": "model.safetensors", "bytes": len(source), "sha256": "b" * 64}],
        }

        def fetch(repository, revision, path, start, end):
            self.assertEqual((repository, revision, path), ("owner/model", "a" * 40, "model.safetensors"))
            return source[start : end + 1]

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "slice.safetensors"
            result = materialize(
                lock, "model.safetensors", output, ["tensor.b", "tensor.a"], fetch
            )
            with safe_open(output, framework="pt", device="cpu") as tensors:
                self.assertEqual(tensors.get_tensor("tensor.a").tolist(), [1, 2, 3, 4])
                self.assertEqual(tensors.get_tensor("tensor.b").tolist(), [9, 8, 7])
            self.assertEqual([item["name"] for item in result["tensors"]], ["tensor.a", "tensor.b"])
            with self.assertRaisesRegex(ValueError, "refusing to overwrite"):
                materialize(lock, "model.safetensors", output, ["tensor.a"], fetch)

    def test_local_extraction_requires_unchanged_verified_source(self):
        header = {
            "tensor.a": {"dtype": "U8", "shape": [4], "data_offsets": [0, 4]},
        }
        raw_header = json.dumps(header, separators=(",", ":")).encode()
        source = struct.pack("<Q", len(raw_header)) + raw_header + bytes([1, 2, 3, 4])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = root / "model.safetensors"
            source_path.write_bytes(source)
            source_stat = source_path.stat()
            source_hash = __import__("hashlib").sha256(source).hexdigest()
            lock = {
                "schema_version": 1,
                "repository": "owner/model",
                "revision": "a" * 40,
                "files": [
                    {"path": source_path.name, "bytes": len(source), "sha256": source_hash}
                ],
            }
            verification = {
                "schema_version": 1,
                "evidence_class": "local_checkpoint_lock_verification",
                "repository": "owner/model",
                "revision": "a" * 40,
                "complete": True,
                "files": [
                    {
                        "path": source_path.name,
                        "status": "verified",
                        "bytes": len(source),
                        "sha256": source_hash,
                        "device": source_stat.st_dev,
                        "inode": source_stat.st_ino,
                        "modified_ns": source_stat.st_mtime_ns,
                    }
                ],
            }
            output = root / "slice.safetensors"
            result = materialize_local(
                lock, verification, root, source_path.name, output, ["tensor.a"]
            )
            self.assertEqual(
                result["evidence_class"],
                "pinned_local_verified_lossless_tensor_ranges",
            )
            with safe_open(output, framework="pt", device="cpu") as tensors:
                self.assertEqual(tensors.get_tensor("tensor.a").tolist(), [1, 2, 3, 4])

            source_path.write_bytes(source[:-1] + b"x")
            with self.assertRaisesRegex(ValueError, "identity changed"):
                materialize_local(
                    lock,
                    verification,
                    root,
                    source_path.name,
                    root / "second.safetensors",
                    ["tensor.a"],
                )


if __name__ == "__main__":
    unittest.main()
