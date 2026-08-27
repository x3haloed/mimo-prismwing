import unittest
from tools.analyze_pw0320_hybrid_byte_floor import K4_BYTES, SOURCE_BYTES, oracle_cached_bytes, window_metrics

class Pw0320Tests(unittest.TestCase):
    def test_oracle_cache_removes_only_complete_largest_records(self):
        self.assertEqual(oracle_cached_bytes([10, 6, 6], 12), 10)
    def test_window_accounts_unique_formats_and_accepted_tokens(self):
        ids={(1,1),(1,2),(2,3)}
        row=window_metrics(ids,{(1,1)},4,0)
        self.assertEqual(row['uncached_bytes'],K4_BYTES+2*SOURCE_BYTES)
        self.assertEqual(row['bytes_per_accepted_token'],row['uncached_bytes']/4)
        self.assertEqual(row['unique_k4_identities'],1)
        self.assertEqual(row['unique_source_identities'],2)

if __name__ == '__main__': unittest.main()
