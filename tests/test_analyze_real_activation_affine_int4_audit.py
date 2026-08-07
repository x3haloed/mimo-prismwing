import unittest

from tools.analyze_real_activation_affine_int4_audit import EXPECTED_PACKED_BYTES
from tools.run_real_activation_affine_int4_audit import SOURCE_EXPERT_BYTES


class RealActivationAffineInt4AnalysisTests(unittest.TestCase):
    def test_frozen_packed_ratios_match_runtime_artifacts(self):
        self.assertAlmostEqual(
            EXPECTED_PACKED_BYTES[4] / SOURCE_EXPERT_BYTES,
            0.5311203319502075,
        )
        self.assertAlmostEqual(
            EXPECTED_PACKED_BYTES[8] / SOURCE_EXPERT_BYTES,
            1.0309982914327556,
        )
        self.assertLess(EXPECTED_PACKED_BYTES[4] / SOURCE_EXPERT_BYTES, 0.60)
        self.assertGreater(EXPECTED_PACKED_BYTES[8] / SOURCE_EXPERT_BYTES, 1.0)


if __name__ == "__main__":
    unittest.main()
