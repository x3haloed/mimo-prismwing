import unittest
from tools.analyze_pw0321_stitched_horizon import group_metric
from tools.analyze_pw0320_hybrid_byte_floor import K4_BYTES,SOURCE_BYTES

class Pw0321Tests(unittest.TestCase):
    def test_group_metric_preserves_structural_and_observed_acceptance(self):
        ids={(layer,expert) for layer in range(2) for expert in range(256)}
        selected={(0,expert) for expert in range(256)}
        row=group_metric(ids,selected,7,16)
        self.assertEqual(row['observed_a_sum'],7)
        self.assertEqual(row['structural_a'],16)
        self.assertGreater(row['bytes_after_oracle_cache'],0)
        self.assertEqual(row['unique_identities'],512)
        self.assertGreater(row['structural_optimistic_tps'],row['observed_sum_optimistic_tps'])

if __name__=='__main__': unittest.main()
