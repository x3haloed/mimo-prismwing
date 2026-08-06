import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from tools.compare_real_layer1_expert_traces import NUMERICS, ORDER, compare
from tools.generate_real_layer0_bf16_oracle import PROMPT_IDS, REVISION


class RealLayer1ExpertTraceTests(unittest.TestCase):
    def write_trace(self, root: Path, semantic: str, delta: float = 0.0) -> Path:
        captures = {}
        for name in ORDER:
            values = np.asarray([1.0, 2.0 + (delta if name == "expert_down" else 0.0)], dtype="<f4")
            payload = values.tobytes(); path = root / f"{name}.f32"; path.write_bytes(payload)
            captures[name] = {"file": path.name, "shape": [2], "dtype": "BF16_widened_F32",
                              "sha256": hashlib.sha256(payload).hexdigest()}
        selected = [[(position + slot) % 28 for slot in range(8)] for position in range(27)]
        weights = [[0.01 * (slot + 1) for slot in range(8)] for _ in range(27)]
        positions_by_expert = {expert: [] for expert in range(28)}
        for position, experts in enumerate(selected):
            for expert in experts:
                positions_by_expert[expert].append(position)
        schedule = [{"expert": expert, "positions": positions}
                    for expert, positions in positions_by_expert.items()]
        manifest = {"schema_version": 1, "semantic": semantic, "revision": REVISION,
                    "prompt_token_ids": PROMPT_IDS, "checkpoint_verification_sha256": "verification",
                    "source_input_sha256": "source", "numerics": NUMERICS, "captures": captures,
                    "selected_experts_by_position": selected, "route_weights_by_position": weights,
                    "expert_schedule": schedule}
        path = root / "manifest.json"; path.write_text(json.dumps(manifest)); return path

    def test_exact_trace_clears(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); oracle_root = root / "oracle"; rust_root = root / "rust"
            oracle_root.mkdir(); rust_root.mkdir()
            oracle = self.write_trace(oracle_root, "mimo_real_layer1_selected_experts_oracle")
            rust = self.write_trace(rust_root, "mimo_real_layer1_selected_experts_rust_trace")
            self.assertTrue(compare(oracle, rust)["layer1_experts_provisionally_cleared"])

    def test_localizes_and_rejects_tampering(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); oracle_root = root / "oracle"; rust_root = root / "rust"
            oracle_root.mkdir(); rust_root.mkdir()
            oracle = self.write_trace(oracle_root, "mimo_real_layer1_selected_experts_oracle")
            rust = self.write_trace(rust_root, "mimo_real_layer1_selected_experts_rust_trace", 1.0)
            self.assertEqual(compare(oracle, rust)["first_failure"]["capture"], "expert_down")
            (rust_root / "expert_down.f32").write_bytes(b"tampered")
            with self.assertRaises(ValueError):
                compare(oracle, rust)


if __name__ == "__main__":
    unittest.main()
