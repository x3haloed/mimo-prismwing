import unittest

from tools.analyze_uncached_verifier_pilot import median


class UncachedVerifierPilotAnalysisTests(unittest.TestCase):
    def test_median_averages_two_controls(self):
        self.assertEqual(median([4.0, 2.0]), 3.0)

    def test_median_rejects_empty(self):
        with self.assertRaises(ValueError):
            median([])


if __name__ == "__main__":
    unittest.main()
