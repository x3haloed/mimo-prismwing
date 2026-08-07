import unittest

from tools.analyze_pread_expert_acquisition import median


class PreadExpertAcquisitionAnalysisTests(unittest.TestCase):
    def test_median_preserves_odd_middle(self):
        self.assertEqual(median([3.0, 1.0, 2.0]), 2.0)

    def test_median_rejects_empty(self):
        with self.assertRaises(ValueError):
            median([])


if __name__ == "__main__":
    unittest.main()
