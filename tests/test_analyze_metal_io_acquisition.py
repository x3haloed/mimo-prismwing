import unittest

from tools.analyze_metal_io_acquisition import median


class MetalIoAcquisitionAnalysisTests(unittest.TestCase):
    def test_median_requires_exactly_three_trials(self):
        self.assertEqual(median([3.0, 1.0, 2.0]), 2.0)
        with self.assertRaises(ValueError):
            median([1.0, 2.0])


if __name__ == "__main__":
    unittest.main()
