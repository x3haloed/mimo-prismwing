import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from tools.compare_real_layer1_routing_traces import NUMERICS, ORDER, compare
from tools.generate_real_layer0_bf16_oracle import PROMPT_IDS, REVISION


class RealLayer1RoutingTraceTests(unittest.TestCase):
    def write_trace(self, root: Path, semantic: str, *, qkv_delta: float = 0.0,
                    reverse_routes: bool = False, route_weight_delta: float = 0.0) -> Path:
        captures = {}
        for name in ORDER:
            values = np.asarray([1.0, 2.0 + (qkv_delta if name == "qkv" else 0.0)], dtype="<f4")
            payload = values.tobytes()
            path = root / f"{name}.f32"
            path.write_bytes(payload)
            captures[name] = {"file": path.name, "shape": [2],
                              "dtype": "F32" if name.startswith("router_") else "BF16_widened_F32",
                              "sha256": hashlib.sha256(payload).hexdigest()}
        experts = list(range(8))
        weights = [0.01 * (index + 1) for index in range(8)]
        if reverse_routes:
            experts.reverse(); weights.reverse()
        weights[experts.index(0)] += route_weight_delta
        manifest = {"schema_version": 1, "semantic": semantic, "revision": REVISION,
                    "prompt_token_ids": PROMPT_IDS, "checkpoint_verification_sha256": "verification",
                    "source_input_sha256": "source", "numerics": NUMERICS, "captures": captures,
                    "selected_experts_by_position": [experts],
                    "route_weights_by_position": [weights]}
        path = root / "manifest.json"
        path.write_text(json.dumps(manifest))
        return path

    def test_exact_routes_clear_despite_order(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); oracle_root = root / "oracle"; rust_root = root / "rust"
            oracle_root.mkdir(); rust_root.mkdir()
            oracle = self.write_trace(oracle_root, "mimo_real_layer1_attention_to_routing_oracle")
            rust = self.write_trace(rust_root, "mimo_real_layer1_attention_to_routing_rust_trace",
                                    reverse_routes=True)
            result = compare(oracle, rust)
            self.assertTrue(result["layer1_routing_provisionally_cleared"])
            self.assertEqual(result["routing"]["maximum_weight_error_by_expert"], 0.0)

    def test_localizes_capture_and_route_weight_failures(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); oracle_root = root / "oracle"; rust_root = root / "rust"
            oracle_root.mkdir(); rust_root.mkdir()
            oracle = self.write_trace(oracle_root, "mimo_real_layer1_attention_to_routing_oracle")
            rust = self.write_trace(rust_root, "mimo_real_layer1_attention_to_routing_rust_trace",
                                    qkv_delta=1.0, route_weight_delta=1e-4)
            result = compare(oracle, rust)
            self.assertEqual(result["first_failure"]["capture"], "qkv")
            self.assertGreater(result["routing"]["maximum_weight_error_by_expert"], 5e-7)
            (rust_root / "router_scores.f32").write_bytes(b"tampered")
            with self.assertRaises(ValueError):
                compare(oracle, rust)


if __name__ == "__main__":
    unittest.main()
