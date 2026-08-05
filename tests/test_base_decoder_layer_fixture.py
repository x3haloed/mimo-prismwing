import json
from pathlib import Path
import unittest

import numpy as np

from tools.generate_base_decoder_layer_fixture import attention_query, noaux_tc_route, rope


class BaseDecoderLayerFixtureTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
