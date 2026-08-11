import json
from pathlib import Path
import unittest

import numpy as np

from tools.run_subvector_low_rank_residual_oracle import add_low_rank


class SubvectorLowRankResidualOracleTest(unittest.TestCase):
    def test_tiny_low_rank_residual(self):
        fixture = json.loads((Path(__file__).parents[1] / "evals/fixtures/tiny/pw0179-low-rank-residual.json").read_text())
        actual = add_low_rank(np.asarray(fixture["core"]), np.asarray(fixture["left"]), np.asarray(fixture["right"]))
        np.testing.assert_array_equal(actual, np.asarray(fixture["expected"], dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
