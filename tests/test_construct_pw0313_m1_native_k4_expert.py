import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from tools.construct_pw0313_m1_native_k4_expert import (
    classify_projection,
    compare_payload_trees,
    deterministic_tree_manifest,
    metric,
)


class Pw0313M1NativeK4Tests(unittest.TestCase):
    def _projection(self, root: Path, packed: bytes = b"packed") -> Path:
        root.mkdir()
        (root / "packed.u16le").write_bytes(packed)
        (root / "fixture.json").write_text("{}\n")
        manifest = {
            "fixture": {"file": "fixture.json"},
            "files": {"packed": {"file": "packed.u16le"}},
        }
        (root / "manifest.json").write_text(json.dumps(manifest) + "\n")
        return root

    def test_payload_and_semantic_classification_are_separate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference = self._projection(root / "reference")
            candidate = self._projection(root / "candidate")
            payload = compare_payload_trees(candidate, reference)
            self.assertTrue(payload["payload_identical"])
            self.assertEqual(classify_projection(payload, {"relative_l2": 1.0}), "payload_identical")

            (candidate / "packed.u16le").write_bytes(b"alias!")
            payload = compare_payload_trees(candidate, reference)
            self.assertFalse(payload["payload_identical"])
            self.assertEqual(classify_projection(payload, {"relative_l2": 0.0}), "semantic_alias")
            self.assertEqual(classify_projection(payload, {"relative_l2": 1e-6}), "numerical_drift")

    def test_metric_fails_closed_on_shape_and_reports_relative_error(self):
        self.assertEqual(metric(np.ones(2, dtype=np.float32), np.ones(2, dtype=np.float32))["relative_l2"], 0.0)
        self.assertAlmostEqual(
            metric(np.ones(2, dtype=np.float32), np.asarray([2.0, 1.0], dtype=np.float32))["relative_l2"],
            1.0 / np.sqrt(2.0),
        )
        with self.assertRaisesRegex(ValueError, "shape mismatch"):
            metric(np.ones(2), np.ones(3))

    def test_repeat_manifest_excludes_nondeterministic_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "payload").write_bytes(b"stable")
            (root / "construction.json").write_bytes(b"timing changes")
            manifest = deterministic_tree_manifest(root)
            self.assertEqual([row["path"] for row in manifest["files"]], ["payload"])


if __name__ == "__main__":
    unittest.main()
