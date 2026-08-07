import unittest

from tools.run_int4_low_rank_repair_oracle import physical_ledger


class Int4LowRankRepairAnalysisTests(unittest.TestCase):
    def test_selected_rank32_physical_constants(self):
        ledger = physical_ledger(32)
        self.assertEqual(ledger["low_rank_factor_bytes_per_layer"], 134_217_728)
        self.assertAlmostEqual(ledger["combined_to_source_layer_bank_ratio"], 0.5525994630217232)
        self.assertAlmostEqual(ledger["repair_to_source_expert_mac_ratio"], 1 / 96)


if __name__ == "__main__":
    unittest.main()
