import hashlib
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from tools.verify_source_lock import verify_source_lock


class SourceLockTests(unittest.TestCase):
    def test_verifies_revision_and_content_and_fails_on_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            subprocess.run(["git", "-C", str(root), "config", "user.email", "fixture@example.test"], check=True)
            subprocess.run(["git", "-C", str(root), "config", "user.name", "Fixture"], check=True)
            payload = root / "source.c"
            payload.write_text("int fixture = 1;\n")
            subprocess.run(["git", "-C", str(root), "add", "source.c"], check=True)
            subprocess.run(["git", "-C", str(root), "commit", "-qm", "fixture"], check=True)
            revision = subprocess.run(
                ["git", "-C", str(root), "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            digest = hashlib.sha256(payload.read_bytes()).hexdigest()
            lock_path = root / "lock.json"
            lock_path.write_text(json.dumps({
                "schema_version": 1,
                "revision": revision,
                "files": {"source.c": digest},
            }))

            result = verify_source_lock(lock_path, root)
            self.assertEqual(result["verified_file_count"], 1)

            payload.write_text("int fixture = 2;\n")
            with self.assertRaisesRegex(ValueError, "source hash mismatch"):
                verify_source_lock(lock_path, root)


if __name__ == "__main__":
    unittest.main()
