import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from tools.generate_base_decoder_layer_fixture import (
    attention_query,
    noaux_tc_route,
    rope,
    validate_semantic_fixture,
)
from tools.generate_base_decoder_layer_moe_fixture import (
    build_schedule,
    load_f32_artifact,
    validate_extraction_authority,
)


class BaseDecoderLayerFixtureTests(unittest.TestCase):
    def test_pw0209_full_width_fixture_is_accepted(self):
        fixture = json.loads(
            Path("evals/fixtures/real/pw0209-layer43-context128-full-width.json").read_text()
        )
        validate_semantic_fixture(fixture)

    def test_semantic_fixture_locks_real_layer_shape(self):
        fixture = json.loads(
            Path("evals/fixtures/real/base-layer43-context128.json").read_text()
        )
        self.assertEqual(fixture["layer"], 43)
        self.assertEqual(fixture["context"], 128)
        self.assertEqual(fixture["query_count"], 8)
        self.assertEqual(fixture["parameters"]["rope_dim"], 64)
        self.assertEqual(fixture["parameters"]["sliding_window"], 128)
        self.assertEqual(len(fixture["tensors"]), 8)
        validate_semantic_fixture(fixture)

    def test_semantic_fixture_rejects_wrong_revision_source_position_and_shape(self):
        fixture = json.loads(
            Path("evals/fixtures/real/base-layer43-context128.json").read_text()
        )
        mutations = (
            ("revision", "wrong", "semantic fixture"),
            ("source_file", "model_pp0_ep0_shard0.safetensors", "source authority"),
            ("query_start", 119, "semantic fixture"),
        )
        for key, value, message in mutations:
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, message):
                candidate = copy.deepcopy(fixture)
                candidate[key] = value
                validate_semantic_fixture(candidate)
        candidate = copy.deepcopy(fixture)
        candidate["tensors"]["model.layers.43.self_attn.qkv_proj.weight"]["shape"] = [1, 1]
        with self.assertRaisesRegex(ValueError, "tensor shape"):
            validate_semantic_fixture(candidate)

    def test_attention_artifact_rejects_wrong_input_hash(self):
        payload = np.arange(8, dtype="<f4").tobytes()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "input.f32").write_bytes(payload)
            record = {
                "file": "input.f32",
                "sha256": hashlib.sha256(payload + b"wrong").hexdigest(),
                "shape": [2, 4],
            }
            with self.assertRaisesRegex(ValueError, "artifact hash mismatch"):
                load_f32_artifact(root, record, (2, 4))

    def test_partial_rope_preserves_unrotated_tail_and_norm(self):
        values = np.arange(2 * 192, dtype=np.float32).reshape(2, 192) / 100
        rotated = rope(values, position=17, rope_dim=64, theta=10_000)
        np.testing.assert_array_equal(rotated[:, 64:], values[:, 64:])
        np.testing.assert_allclose(
            np.linalg.norm(rotated[:, :64], axis=1),
            np.linalg.norm(values[:, :64], axis=1),
            rtol=2e-6,
            atol=2e-6,
        )
        with self.assertRaisesRegex(ValueError, "RoPE"):
            rope(values, position=0, rope_dim=63, theta=10_000)

    def test_sink_probability_adds_zero_value_mass(self):
        query = np.zeros((64, 192), dtype=np.float32)
        keys = np.zeros((1, 8, 192), dtype=np.float32)
        values = np.ones((1, 8, 128), dtype=np.float32)
        sinks = np.zeros(64, dtype=np.float32)
        output = attention_query(query, keys, values, sinks)
        np.testing.assert_array_equal(output, np.full((64, 128), 0.5, np.float32))

    def test_noaux_tc_uses_corrected_choice_and_uncorrected_weights(self):
        hidden = np.ones((1, 4), dtype=np.float32)
        router = np.zeros((256, 4), dtype=np.float32)
        correction = np.arange(256, dtype=np.float32) / 256
        selected, weights, margin = noaux_tc_route(hidden, router, correction, 8)
        self.assertEqual(set(selected[0]), set(range(248, 256)))
        np.testing.assert_allclose(weights, np.full((1, 8), 0.125, np.float32))
        self.assertGreater(margin, 0)

    def test_dynamic_schedule_preserves_every_route_placement(self):
        selected = np.arange(64, dtype=np.int64).reshape(8, 8) % 9
        for row in range(8):
            selected[row] = (np.arange(8) + row) % 9
        weights = np.full((8, 8), 0.125, dtype=np.float32)
        schedule = build_schedule(selected, weights)
        self.assertEqual(sum(len(item["positions"]) for item in schedule.values()), 64)
        self.assertEqual(set(schedule), set(range(9)))
        with self.assertRaisesRegex(ValueError, "selected expert row"):
            duplicate = selected.copy()
            duplicate[0, 1] = duplicate[0, 0]
            build_schedule(duplicate, weights)

    def test_extraction_rejects_wrong_route_and_expert_authority(self):
        schedule = {1: {"positions": [0], "slots": [0], "route_weights": [1.0]}}
        extraction = {
            "schema_version": 1,
            "revision": "63651580ca774f8504f676040460aed3e1244ac1",
            "layer": 43,
            "experts": [1],
            "source_slices": [
                {
                    "output_file": "expert.safetensors",
                    "evidence_class": "pinned_local_verified_lossless_tensor_ranges",
                }
            ],
        }
        validate_extraction_authority(extraction, schedule)
        wrong_expert = copy.deepcopy(extraction)
        wrong_expert["experts"] = [2]
        with self.assertRaisesRegex(ValueError, "extraction identity"):
            validate_extraction_authority(wrong_expert, schedule)
        wrong_authority = copy.deepcopy(extraction)
        wrong_authority["source_slices"][0]["evidence_class"] = "unverified"
        with self.assertRaisesRegex(ValueError, "verification authority"):
            validate_extraction_authority(wrong_authority, schedule)


if __name__ == "__main__":
    unittest.main()
