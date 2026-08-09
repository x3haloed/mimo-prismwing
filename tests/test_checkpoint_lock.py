import hashlib
from pathlib import Path
import tempfile
import unittest

from tools.checkpoint_lock import sha256_file, validate_verified_install_file, verify_lock
from tools.openrouter_reference import canonical_json


class CheckpointLockTests(unittest.TestCase):
    def test_runtime_identity_allows_only_transient_device_drift(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "payload"
            path.write_bytes(b"checkpoint")
            stat = path.stat()
            record = {
                "status": "verified",
                "bytes": stat.st_size,
                "sha256": hashlib.sha256(b"checkpoint").hexdigest(),
                "device": stat.st_dev + 1,
                "inode": stat.st_ino,
                "modified_ns": stat.st_mtime_ns,
            }
            result = validate_verified_install_file(path, record)
            self.assertTrue(result["device_changed"])
            self.assertEqual(result["current_device"], stat.st_dev)
            for field, value in (
                ("bytes", stat.st_size + 1),
                ("inode", stat.st_ino + 1),
                ("modified_ns", stat.st_mtime_ns + 1),
                ("status", "unverified"),
                ("sha256", "missing"),
            ):
                with self.subTest(field=field):
                    corrupted = dict(record)
                    corrupted[field] = value
                    with self.assertRaisesRegex(ValueError, "identity changed"):
                        validate_verified_install_file(path, corrupted)
            directory = Path(temporary) / "directory"
            directory.mkdir()
            with self.assertRaisesRegex(ValueError, "not regular"):
                validate_verified_install_file(directory, record)

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
