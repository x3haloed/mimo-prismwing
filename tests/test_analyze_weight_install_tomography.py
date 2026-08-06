import unittest

from tools.analyze_weight_install_tomography import distribution


class WeightInstallTomographyAnalysisTests(unittest.TestCase):
    def test_distribution_is_deterministic(self):
        summary = distribution([5.0, 1.0, 3.0, 2.0, 4.0])
        self.assertEqual(summary["count"], 5)
        self.assertEqual(summary["sum_ms"], 15.0)
        self.assertEqual(summary["p10_ms"], 1.0)
        self.assertEqual(summary["median_ms"], 3.0)
        self.assertEqual(summary["p90_ms"], 5.0)
        self.assertEqual(summary["maximum_ms"], 5.0)

    def test_distribution_rejects_empty_input(self):
        with self.assertRaises(ValueError):
            distribution([])


if __name__ == "__main__":
    unittest.main()
