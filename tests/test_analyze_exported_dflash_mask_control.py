import unittest

import numpy as np

from tools.analyze_exported_dflash_mask_control import token_rank


class ExportedMaskAnalysisTests(unittest.TestCase):
    def test_token_rank_uses_strictly_greater_logits(self):
        logits = np.zeros((152576,), dtype=np.float32)
        logits[11] = 8.75
        logits[13] = 6.53125
        logits[198] = 7.03125
        logits[3837] = 6.75
        result = token_rank(logits, 13)
        self.assertEqual(result["greedy_token_id"], 11)
        self.assertEqual(result["rank"], 4)
        self.assertEqual(result["gap_from_greedy"], 2.21875)

    def test_token_rank_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "identity mismatch"):
            token_rank(np.zeros((3,), dtype=np.float32), 0)
        logits = np.zeros((152576,), dtype=np.float32)
        logits[0] = np.nan
        with self.assertRaisesRegex(ValueError, "identity mismatch"):
            token_rank(logits, 0)


if __name__ == "__main__":
    unittest.main()
