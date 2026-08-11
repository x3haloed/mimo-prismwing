import unittest

from tools.analyze_uncached_stream_transport import median


class UncachedStreamTransportAnalysisTests(unittest.TestCase):
    def test_median_preserves_odd_middle(self):
        self.assertEqual(median([3.0, 1.0, 2.0]), 2.0)

    def test_median_rejects_empty(self):
        with self.assertRaises(ValueError):
            median([])


if __name__ == "__main__":
    unittest.main()
