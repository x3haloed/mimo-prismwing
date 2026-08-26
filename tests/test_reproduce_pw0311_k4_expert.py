import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from tools.reproduce_pw0311_k4_expert import (
    array_sha256,
    compare_projection_directory,
    panel_prefix,
    select_reference_slot,
    sha256_file,
)


class Pw0311K4ExpertReproductionTests(unittest.TestCase):
    def test_array_authority_includes_dtype_and_shape(self):
        value = np.asarray([[1.0, 2.0]], dtype=np.float32)
        self.assertEqual(
            array_sha256(value),
            "7a164e75acdd3efe475392a0e54792396713720bce9617d4015cd2a251ca2880",
        )
        self.assertNotEqual(array_sha256(value), array_sha256(value.reshape(2)))
        self.assertNotEqual(array_sha256(value), array_sha256(value.astype(np.float64)))

    def test_reference_slot_fails_closed_on_unknown_or_duplicate_expert(self):
        projection_reports = {name: {} for name in ("gate", "up", "down")}
        report = {"slots": [{"expert": 114, "projection_reports": projection_reports}]}
        self.assertEqual(select_reference_slot(report, 114)["expert"], 114)
        with self.assertRaisesRegex(ValueError, "outside the authenticated"):
            select_reference_slot(report, 117)
        report["slots"].append({"expert": 114, "projection_reports": projection_reports})
        with self.assertRaisesRegex(ValueError, "exactly one"):
            select_reference_slot(report, 114)

    def test_panel_prefix_replays_only_authenticated_predecessors(self):
        self.assertEqual(panel_prefix(114, True), ())
        self.assertEqual(panel_prefix(188, True), (114,))
        self.assertEqual(panel_prefix(41, True), (114, 188, 93, 199, 248))
        self.assertEqual(panel_prefix(41, False), ())
        with self.assertRaisesRegex(ValueError, "outside the authenticated"):
            panel_prefix(117, True)

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
