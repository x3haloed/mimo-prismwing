import unittest

from tools.run_train_only_awq_int4_expert_rescaling import physical_ledger


class AwqExpertRescalingAnalysisTests(unittest.TestCase):
    def test_physical_ledger_is_compact(self):
        ledger = physical_ledger()
        self.assertAlmostEqual(ledger["combined_to_source_ratio"], 0.5316084940200146)
        self.assertLess(ledger["runtime_elementwise_to_source_expert_mac_ratio"], 0.001)


if __name__ == "__main__":
    unittest.main()
