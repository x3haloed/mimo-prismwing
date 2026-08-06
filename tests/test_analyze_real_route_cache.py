import unittest

from tools.analyze_real_route_cache import belady_hits, lfu_hits, lru_hits


class CachePolicyTests(unittest.TestCase):
    def test_belady_is_exact_on_small_trace(self):
        accesses = ["a", "b", "c", "a", "b", "c"]
        self.assertEqual(belady_hits(accesses, 2), 2)
        self.assertGreaterEqual(belady_hits(accesses, 2), lru_hits(accesses, 2))
        self.assertGreaterEqual(belady_hits(accesses, 2), lfu_hits(accesses, 2))

    def test_full_capacity_hits_every_noncompulsory_access(self):
        accesses = ["a", "b", "a", "c", "b", "a"]
        expected = len(accesses) - len(set(accesses))
        self.assertEqual(lru_hits(accesses, 3), expected)
        self.assertEqual(lfu_hits(accesses, 3), expected)
        self.assertEqual(belady_hits(accesses, 3), expected)


if __name__ == "__main__":
    unittest.main()
