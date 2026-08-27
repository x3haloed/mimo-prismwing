import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from tools.construct_pw0314_layer4_k4 import (
    HIDDEN,
    ROWS,
    SOURCE_SHARD,
    SOURCE_SHARD_SHA256,
    TOP_K,
    metrics_pass,
    partition_metrics,
    reconstruct_route,
    selected_rows,
    verify_full_checkpoint_install,
)


class Pw0314Layer4K4Tests(unittest.TestCase):
    def test_selected_rows_preserve_position_slot_and_weight(self):
        row = {
            "selected_experts_by_position": [[1, 64], [64, 2], [3, 4]],
            "route_weights_by_position": [[0.4, 0.6], [0.7, 0.3], [0.5, 0.5]],
            "expert_schedule": [
                {"expert": 1, "positions": [0]},
                {"expert": 2, "positions": [1]},
                {"expert": 3, "positions": [2]},
                {"expert": 4, "positions": [2]},
                {"expert": 64, "positions": [1, 0]},
            ],
        }
        positions, slots, weights, offsets = selected_rows(row)
        np.testing.assert_array_equal(positions, [1, 0])
        np.testing.assert_array_equal(slots, [0, 1])
        np.testing.assert_allclose(weights, [0.7, 0.6])
        np.testing.assert_array_equal(offsets, [4, 5])

    def test_route_reconstruction_applies_float32_sum_then_bf16_boundary(self):
        values = np.zeros((ROWS * TOP_K, HIDDEN), dtype=np.float32)
        values[:ROWS, :] = 2.0
        layer_row = {
            "selected_experts_by_position": [[0] + list(range(1, TOP_K)) for _ in range(ROWS)],
            "route_weights_by_position": [[0.25] + [0.0] * (TOP_K - 1) for _ in range(ROWS)],
            "expert_schedule": [{"expert": 0, "positions": list(range(ROWS))}]
            + [{"expert": expert, "positions": list(range(ROWS))} for expert in range(1, TOP_K)],
        }
        result = reconstruct_route(values, layer_row, lambda value: np.asarray(value, dtype=np.float32))
        np.testing.assert_array_equal(result, np.full((ROWS, HIDDEN), 0.5, dtype=np.float32))

    def test_partition_gate_checks_aggregate_and_worst_row(self):
        reference = np.ones((2, 4), dtype=np.float32)
        candidate = reference.copy()
        candidate[0, 0] += np.float32(0.01)
        row = partition_metrics(reference, candidate)
        self.assertGreater(row["maximum_row_relative_l2"], 0.0)
        self.assertTrue(metrics_pass({"slice": row}))
        row["maximum_row_relative_l2"] = 0.05
        self.assertFalse(metrics_pass({"slice": row}))

    def test_receipt_preflight_rejects_unmapped_or_reidentified_shard(self):
        import tools.construct_pw0314_layer4_k4 as module

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shard = root / SOURCE_SHARD
            shard.write_bytes(b"source")
            names = [
                f"model.layers.4.mlp.experts.64.{name}_proj.weight{suffix}"
                for name in ("gate", "up", "down")
                for suffix in ("", "_scale_inv")
            ]
            index = root / "model.safetensors.index.json"
            index.write_text(json.dumps({"weight_map": {name: SOURCE_SHARD for name in names}}))
            receipt = root / "receipt.json"
            rows = []
            for path, digest in ((index, "index"), (shard, SOURCE_SHARD_SHA256)):
                stat = path.stat()
                rows.append({
                    "path": path.name,
                    "bytes": stat.st_size,
                    "inode": stat.st_ino,
                    "modified_ns": stat.st_mtime_ns,
                    "device": stat.st_dev,
                    "sha256": digest,
                    "status": "verified",
                })
            receipt.write_text(json.dumps({
                "schema_version": 1,
                "evidence_class": "local_checkpoint_lock_verification",
                "complete": True,
                "missing_files": [],
                "revision": module.CHECKPOINT_REVISION,
                "lock_sha256": "a" * 64,
                "files": rows,
            }))
            original = (
                module.CHECKPOINT_RECEIPT_SHA256,
                module.CHECKPOINT_INDEX_SHA256,
                module.sha256_file,
            )
            module.CHECKPOINT_RECEIPT_SHA256 = "receipt"
            module.CHECKPOINT_INDEX_SHA256 = "index"
            module.sha256_file = lambda path: "receipt" if path == receipt else "index"
            try:
                result = verify_full_checkpoint_install(root, receipt)
                self.assertEqual(result["source_shard_sha256_from_receipt"], SOURCE_SHARD_SHA256)
                shard.write_bytes(b"changed identity")
                with self.assertRaisesRegex(ValueError, "installed file identity mismatch"):
                    verify_full_checkpoint_install(root, receipt)
            finally:
                module.CHECKPOINT_RECEIPT_SHA256, module.CHECKPOINT_INDEX_SHA256, module.sha256_file = original


if __name__ == "__main__":
    unittest.main()
