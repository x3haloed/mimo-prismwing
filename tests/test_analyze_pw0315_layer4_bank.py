import unittest

import numpy as np

from tools.analyze_pw0315_layer4_bank import (
    replace_expert_outputs,
    stable_projections,
)


class Pw0315Layer4BankAnalyzerTests(unittest.TestCase):
    def test_replacement_uses_expert_major_offsets_for_multiple_identities(self):
        layer_row = {
            "selected_experts_by_position": [[64, 96], [96, 31]],
            "route_weights_by_position": [[0.4, 0.6], [0.7, 0.3]],
            "expert_schedule": [
                {"expert": 31, "positions": [1]},
                {"expert": 64, "positions": [0]},
                {"expert": 96, "positions": [0, 1]},
            ],
        }
        source = np.arange(8, dtype=np.float32).reshape(4, 2)
        after_64 = replace_expert_outputs(
            source, layer_row, 64, np.asarray([[20.0, 21.0]], dtype=np.float32)
        )
        after_96 = replace_expert_outputs(
            after_64,
            layer_row,
            96,
            np.asarray([[30.0, 31.0], [40.0, 41.0]], dtype=np.float32),
        )
        np.testing.assert_array_equal(
            after_96,
            [[0.0, 1.0], [20.0, 21.0], [30.0, 31.0], [40.0, 41.0]],
        )

    def test_stable_projection_comparison_ignores_only_timing(self):
        projections = {
            "gate": {
                "candidate_array_sha256": "abc",
                "quantization_seconds": 1.0,
            }
        }
        self.assertEqual(
            stable_projections(projections),
            {"gate": {"candidate_array_sha256": "abc"}},
        )


if __name__ == "__main__":
    unittest.main()
