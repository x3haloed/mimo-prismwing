import unittest

from tools.run_rank768_activation_weighted_expert_pilot import (
    P,
    D,
    PW0121_SPEC,
    PW0122_SPEC,
    PW0125_SPEC,
)


class Rank512ActivationCapacityControlTests(unittest.TestCase):
    def test_rank_and_authority_are_injected_without_changing_prior_specs(self):
        self.assertEqual(PW0121_SPEC.rank, 768)
        self.assertEqual(PW0122_SPEC.rank, 768)
        self.assertEqual(PW0125_SPEC.rank, 512)
        self.assertEqual(PW0121_SPEC.balanced_initialization_tolerance, 5e-6)
        self.assertEqual(PW0122_SPEC.balanced_initialization_tolerance, 5e-6)
        self.assertEqual(PW0125_SPEC.balanced_initialization_tolerance, 1e-5)
        self.assertEqual(PW0125_SPEC.seed, 260125)
        self.assertEqual(PW0125_SPEC.layer, PW0122_SPEC.layer)
        self.assertEqual(PW0125_SPEC.expert, PW0122_SPEC.expert)
        self.assertEqual(PW0125_SPEC.partition_counts, PW0122_SPEC.partition_counts)
        self.assertEqual(P * PW0125_SPEC.rank + PW0125_SPEC.rank * D, 3_145_728)
        self.assertEqual((P * PW0125_SPEC.rank + PW0125_SPEC.rank * D) * 16, 50_331_648)
        self.assertLess(PW0125_SPEC.validation_maximum, 0.75 * 0.6730991256068856)
        self.assertLess(PW0125_SPEC.holdout_maximum, 0.75 * 0.6568507915821798)


if __name__ == "__main__":
    unittest.main()
