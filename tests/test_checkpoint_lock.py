import hashlib
from pathlib import Path
import tempfile
import unittest

from tools.checkpoint_lock import sha256_file, verify_lock
from tools.openrouter_reference import canonical_json


class CheckpointLockTests(unittest.TestCase):
    def test_partial_and_complete_verification(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            payload = b"checkpoint"
            (root / "one.bin").write_bytes(payload)
            lock = {
                "schema_version": 1,
                "file_count": 2,
                "total_bytes": len(payload) + 4,
                "files": [
                    {"path": "one.bin", "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()},
                    {"path": "two.bin", "bytes": 4, "sha256": hashlib.sha256(b"more").hexdigest()},
                ],
            }
            lock_path = root / "lock.json"
            lock_path.write_bytes(canonical_json(lock))
            result = verify_lock(lock_path, root, require_complete=False)
            self.assertFalse(result["complete"])
            self.assertEqual(result["verified_files"], 1)
            self.assertEqual(result["missing_files"], ["two.bin"])
            self.assertEqual(result["files"][0]["status"], "verified")
            self.assertEqual(result["files"][1], {"path": "two.bin", "status": "missing"})
            with self.assertRaisesRegex(ValueError, "incomplete"):
                verify_lock(lock_path, root, require_complete=True)

    def test_complete_verification_manifest_is_hash_bound(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            payload = b"checkpoint"
            (root / "one.bin").write_bytes(payload)
            lock = {
                "schema_version": 1,
                "repository": "owner/model",
                "revision": "a" * 40,
                "file_count": 1,
                "total_bytes": len(payload),
                "files": [
                    {
                        "path": "one.bin",
                        "bytes": len(payload),
                        "sha256": hashlib.sha256(payload).hexdigest(),
                    }
                ],
            }
            lock_path = root / "lock.json"
            encoded_lock = canonical_json(lock)
            lock_path.write_bytes(encoded_lock)
            result = verify_lock(lock_path, root, require_complete=True)
            self.assertTrue(result["complete"])
            self.assertEqual(result["repository"], "owner/model")
            self.assertEqual(result["revision"], "a" * 40)
            self.assertEqual(result["lock_sha256"], hashlib.sha256(encoded_lock).hexdigest())
            self.assertEqual(result["files"][0]["sha256"], hashlib.sha256(payload).hexdigest())

    def test_sha256_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "value"
            path.write_bytes(b"value")
            self.assertEqual(sha256_file(path), hashlib.sha256(b"value").hexdigest())


if __name__ == "__main__":
    unittest.main()
