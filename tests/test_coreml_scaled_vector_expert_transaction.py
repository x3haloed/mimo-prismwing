import json
from pathlib import Path
import unittest

import numpy as np

from tools.run_coreml_scaled_vector_expert_transaction import normalize_projection, scaled_linear


class CoreMlScaledVectorExpertTransactionTest(unittest.TestCase):
    def test_tiny_scaled_projection_is_algebraically_exact(self):
        fixture_path = Path(__file__).parents[1] / "evals/fixtures/tiny/pw0177-scaled-projection.json"
        fixture = json.loads(fixture_path.read_text())
        values = np.asarray(fixture["input"], dtype=np.float32)
        weight = np.asarray(fixture["weight"], dtype=np.float32)
        normalized, scale = normalize_projection(weight)
        np.testing.assert_array_equal(scale, np.asarray(fixture["expected_scale"], dtype=np.float32))
        np.testing.assert_array_equal(normalized, np.asarray(fixture["expected_normalized_weight"], dtype=np.float32))
        np.testing.assert_array_equal(scaled_linear(values, normalized, scale), np.asarray(fixture["expected_output"], dtype=np.float32))
        np.testing.assert_array_equal(scaled_linear(values, normalized, scale), values @ weight.T)


if __name__ == "__main__":
    unittest.main()
