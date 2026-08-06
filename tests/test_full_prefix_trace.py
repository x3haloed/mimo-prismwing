import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from tools.compare_full_prefix_traces import NUMERICS, ORDER, compare
from tools.generate_full_prefix_oracle import load_token_fixture
from tools.generate_real_layer0_bf16_oracle import PROMPT_IDS, REVISION


class FullPrefixTraceTests(unittest.TestCase):
    def test_whole_sequence_fixture_extends_frozen_prompt_once(self):
        fixture = Path("evals/fixtures/real/pw0093-whole-sequence.json")
        tokens, digest = load_token_fixture(fixture)
        self.assertEqual(tokens, [*PROMPT_IDS, 264])
        self.assertEqual(len(digest), 64)
        with tempfile.TemporaryDirectory() as temporary:
            changed = json.loads(fixture.read_text())
            changed["whole_sequence_token_ids"][-1] = 13
            path = Path(temporary) / "changed.json"
            path.write_text(json.dumps(changed))
            with self.assertRaises(ValueError):
                load_token_fixture(path)

    def write_trace(self, root: Path, semantic: str, delta_name: str | None = None) -> Path:
        captures = {}
        for name in ORDER:
            values = np.zeros(10_000, dtype="<f4") if name == "last_logits" else np.asarray(
                [1.0, 2.0 + (1.0 if name == delta_name else 0.0)], dtype="<f4")
            payload = values.tobytes(); path = root / f"{name}.f32"; path.write_bytes(payload)
            captures[name] = {"file": path.name, "shape": list(values.shape),
                              "dtype": "F32" if name == "last_logits" else "BF16_widened_F32",
                              "sha256": hashlib.sha256(payload).hexdigest()}
        selected = [[list(range(8)) for _ in range(27)] for _ in range(47)]
        weights = [[[0.125] * 8 for _ in range(27)] for _ in range(47)]
        traces = [{"layer": 0, "attention": "full", "selected_experts_by_position": [],
                   "route_weights_by_position": []}]
        for layer in range(1, 48):
            traces.append({"layer": layer,
                           "attention": "full" if layer % 6 == 5 else "sliding_window_128",
                           "selected_experts_by_position": selected[layer - 1],
                           "route_weights_by_position": weights[layer - 1]})
        manifest = {"schema_version": 1, "semantic": semantic, "revision": REVISION,
                    "prompt_token_ids": PROMPT_IDS, "checkpoint_verification_sha256": "verification",
                    "numerics": NUMERICS, "captures": captures, "layer_traces": traces}
        path = root / "manifest.json"; path.write_text(json.dumps(manifest)); return path

    def test_exact_trace_clears_and_difference_localizes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); oracle_root = root / "oracle"; rust_root = root / "rust"
            oracle_root.mkdir(); rust_root.mkdir()
            oracle = self.write_trace(oracle_root, "mimo_full_prefix_layer_final_oracle")
            rust = self.write_trace(rust_root, "mimo_full_prefix_layer_final_rust_trace")
            self.assertTrue(compare(oracle, rust)["full_prefix_provisionally_cleared"])
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); oracle_root = root / "oracle"; rust_root = root / "rust"
            oracle_root.mkdir(); rust_root.mkdir()
            oracle = self.write_trace(oracle_root, "mimo_full_prefix_layer_final_oracle")
            rust = self.write_trace(rust_root, "mimo_full_prefix_layer_final_rust_trace",
                                    "layer_03_final")
            self.assertEqual(compare(oracle, rust)["first_failure"], "layer_03_final")
            (rust_root / "layer_03_final.f32").write_bytes(b"tampered")
            with self.assertRaises(ValueError): compare(oracle, rust)


if __name__ == "__main__": unittest.main()
