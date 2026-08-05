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
            verify_lock(lock_path, root, require_complete=False)
            with self.assertRaisesRegex(ValueError, "incomplete"):
                verify_lock(lock_path, root, require_complete=True)

    def test_sha256_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "value"
            path.write_bytes(b"value")
            self.assertEqual(sha256_file(path), hashlib.sha256(b"value").hexdigest())


if __name__ == "__main__":
    unittest.main()
