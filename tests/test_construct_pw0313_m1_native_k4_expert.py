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
    substitute_exact_frozen_route,
)


class Pw0313M1NativeK4Tests(unittest.TestCase):
    def _projection(self, root: Path, packed: bytes = b"packed") -> Path:
        root.mkdir()
        payloads = {
            "packed": ("packed.u16le", packed),
            "left_sign": ("left-sign.i8", b"left"),
            "right_sign": ("right-sign.i8", b"right"),
            "global_scale": ("global-scale.f32le", b"scale"),
            "row_scale": ("row-scale.f16le", b"rows"),
            "correction_left": ("correction-left.f16le", b"cleft"),
            "correction_right": ("correction-right.f16le", b"cright"),
        }
        for filename, content in payloads.values():
            (root / filename).write_bytes(content)
        (root / "fixture.json").write_text("{}\n")
        manifest = {
            "fixture": {"file": "fixture.json"},
            "files": {key: {"file": filename} for key, (filename, _) in payloads.items()},
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
            self.assertTrue(payload["all_files_identical"])
            self.assertEqual(classify_projection(payload, {"relative_l2": 1.0}), "payload_identical")

            (candidate / "packed.u16le").write_bytes(b"alias!")
            payload = compare_payload_trees(candidate, reference)
            self.assertFalse(payload["payload_identical"])
            self.assertFalse(payload["all_files_identical"])
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

    def test_exact_route_substitution_does_not_require_lost_assembler(self):
        route = np.asarray([[1.0, 2.0]], dtype=np.float32)
        output = np.asarray([[3.0, 4.0]], dtype=np.float32)
        replaced = substitute_exact_frozen_route(route, output, output.copy())
        np.testing.assert_array_equal(replaced, route)
        with self.assertRaisesRegex(ValueError, "new authenticated route assembler"):
            substitute_exact_frozen_route(route, output, output + np.float32(1e-3))


if __name__ == "__main__":
    unittest.main()
