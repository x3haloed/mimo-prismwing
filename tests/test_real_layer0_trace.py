import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from tools.compare_real_layer0_traces import ORDER, compare
from tools.generate_real_layer0_bf16_oracle import PROMPT_IDS, REVISION


class RealLayer0TraceTests(unittest.TestCase):
    def write_trace(self, root: Path, semantic: str, delta: float = 0.0) -> Path:
        captures = {}
        for index, name in enumerate(ORDER):
            values = np.asarray([1.0, 2.0 + (delta if index == 2 else 0.0)], dtype="<f4")
            payload = values.tobytes()
            path = root / f"{name}.f32"
            path.write_bytes(payload)
            captures[name] = {"file": path.name, "shape": [2],
                              "dtype": "BF16_widened_F32",
                              "sha256": hashlib.sha256(payload).hexdigest()}
        manifest = {"schema_version": 1, "semantic": semantic, "revision": REVISION,
                    "prompt_token_ids": PROMPT_IDS,
                    "checkpoint_verification_sha256": "verification",
                    "numerics": "dynamic_fp8_e4m3fn_per_token_group_128_bf16_boundaries",
                    "captures": captures}
        path = root / "manifest.json"
        path.write_text(json.dumps(manifest))
        return path

    def test_exact_traces_clear_layer(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            oracle_root, rust_root = root / "oracle", root / "rust"
            oracle_root.mkdir(); rust_root.mkdir()
            oracle = self.write_trace(oracle_root, "mimo_real_layer0_bf16_dynamic_fp8_oracle")
            rust = self.write_trace(rust_root, "mimo_real_layer0_bf16_dynamic_fp8_rust_trace")
            result = compare(oracle, rust)
            self.assertTrue(result["layer0_provisionally_cleared"])
            self.assertIsNone(result["first_failure"])

    def test_comparison_localizes_and_rejects_tampering(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            oracle_root, rust_root = root / "oracle", root / "rust"
            oracle_root.mkdir(); rust_root.mkdir()
            oracle = self.write_trace(oracle_root, "mimo_real_layer0_bf16_dynamic_fp8_oracle")
            rust = self.write_trace(rust_root, "mimo_real_layer0_bf16_dynamic_fp8_rust_trace", 1.0)
            result = compare(oracle, rust)
            self.assertEqual(result["first_failure"]["capture"], "qkv")
            (rust_root / "qkv.f32").write_bytes(b"tampered")
            with self.assertRaises(ValueError):
                compare(oracle, rust)


if __name__ == "__main__":
    unittest.main()
