import unittest

from tools.run_train_only_int4_source_fp8_exception_store import FRACTIONS, physical_ledger


class SourceFp8ExceptionAnalysisTests(unittest.TestCase):
    def test_physical_curve_is_monotonic_and_six_percent_is_last_admissible(self):
        ledgers = [physical_ledger(fraction) for fraction in FRACTIONS]
        self.assertTrue(all(
            left["combined_bytes_per_expert"] < right["combined_bytes_per_expert"]
            for left, right in zip(ledgers, ledgers[1:])
        ))
        self.assertLessEqual(ledgers[-1]["combined_to_source_ratio"], 0.60)
        self.assertGreater(physical_ledger(0.07)["combined_to_source_ratio"], 0.60)


if __name__ == "__main__":
    unittest.main()
