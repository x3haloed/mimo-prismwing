import json
from pathlib import Path
import tempfile
import unittest

from tools.reproduce_pw0311_k4_expert import (
    compare_projection_directory,
    select_reference_slot,
    sha256_file,
)


class Pw0311K4ExpertReproductionTests(unittest.TestCase):
    def test_reference_slot_fails_closed_on_unknown_or_duplicate_expert(self):
        projection_reports = {name: {} for name in ("gate", "up", "down")}
        report = {"slots": [{"expert": 114, "projection_reports": projection_reports}]}
        self.assertEqual(select_reference_slot(report, 114)["expert"], 114)
        with self.assertRaisesRegex(ValueError, "outside the authenticated"):
            select_reference_slot(report, 117)
        report["slots"].append({"expert": 114, "projection_reports": projection_reports})
        with self.assertRaisesRegex(ValueError, "exactly one"):
            select_reference_slot(report, 114)

    def test_projection_tree_requires_every_byte_to_match(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference = root / "reference"
            candidate = root / "candidate"
            reference.mkdir()
            candidate.mkdir()
            fixture = b'{"fixture":true}\n'
            payload = b"payload"
            manifest = {
                "fixture": {"file": "fixture.json"},
                "files": {"packed": {"file": "packed.u16le"}},
            }
            manifest_bytes = (json.dumps(manifest, indent=2) + "\n").encode()
            for target in (reference, candidate):
                (target / "manifest.json").write_bytes(manifest_bytes)
                (target / "fixture.json").write_bytes(fixture)
                (target / "packed.u16le").write_bytes(payload)
            expected = {"manifest_sha256": sha256_file(reference / "manifest.json")}
            result = compare_projection_directory(candidate, reference, expected)
            self.assertEqual(result["files_bit_exact"], 3)
            self.assertEqual(result["bytes_bit_exact"], len(manifest_bytes) + len(fixture) + len(payload))
            (candidate / "packed.u16le").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "payload mismatch"):
                compare_projection_directory(candidate, reference, expected)


if __name__ == "__main__":
    unittest.main()
