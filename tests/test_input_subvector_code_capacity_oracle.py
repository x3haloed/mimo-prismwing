import json
from pathlib import Path
import unittest

import numpy as np

from tools.run_input_subvector_code_capacity_oracle import decode_input_subvectors


class InputSubvectorCodeCapacityOracleTest(unittest.TestCase):
    def test_tiny_decode_and_ledger(self):
        fixture = json.loads((Path(__file__).parents[1] / "evals/fixtures/tiny/pw0178-input-subvector-code.json").read_text())
        codebooks = np.asarray(fixture["codebooks"], dtype=np.float16)
        indices = np.asarray(fixture["indices"], dtype=np.uint8)
        actual = decode_input_subvectors(indices, codebooks)
        np.testing.assert_array_equal(actual, np.asarray(fixture["expected_weight"], dtype=np.float16))
        self.assertEqual(indices.nbytes, fixture["expected_index_bytes"])
        self.assertEqual(codebooks.nbytes, fixture["expected_codebook_f16_bytes"])


if __name__ == "__main__":
    unittest.main()
