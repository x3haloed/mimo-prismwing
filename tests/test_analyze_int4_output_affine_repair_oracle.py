import unittest

from tools.analyze_int4_output_affine_repair_oracle import (
    BIAS_REPAIR_BYTES_PER_LAYER,
    FULL_AFFINE_REPAIR_BYTES_PER_LAYER,
)


class Int4OutputAffineRepairAnalysisTests(unittest.TestCase):
    def test_repair_storage_ledger(self):
        self.assertEqual(BIAS_REPAIR_BYTES_PER_LAYER, 2_097_152)
        self.assertEqual(FULL_AFFINE_REPAIR_BYTES_PER_LAYER, 4_194_304)
        self.assertEqual(FULL_AFFINE_REPAIR_BYTES_PER_LAYER, 2 * BIAS_REPAIR_BYTES_PER_LAYER)


if __name__ == "__main__":
    unittest.main()
