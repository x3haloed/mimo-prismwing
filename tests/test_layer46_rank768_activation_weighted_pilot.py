import unittest

from tools.run_rank768_activation_weighted_expert_pilot import PW0121_SPEC, PW0122_SPEC


class Layer46Rank768ActivationWeightedPilotTests(unittest.TestCase):
    def test_experiment_specs_are_explicit_and_distinct(self):
        self.assertEqual((PW0121_SPEC.layer, PW0121_SPEC.expert), (24, 23))
        self.assertEqual((PW0122_SPEC.layer, PW0122_SPEC.expert), (46, 28))
        self.assertEqual(
            PW0122_SPEC.partition_counts,
            {"train": 100, "validation": 56, "pilot_holdout": 56},
        )
        self.assertEqual(PW0122_SPEC.seed, 260122)
        self.assertEqual(PW0122_SPEC.validation_maximum, 0.4292476011983385)
        self.assertEqual(PW0122_SPEC.holdout_maximum, 0.4093612798639559)
        self.assertNotEqual(PW0121_SPEC.evidence_class, PW0122_SPEC.evidence_class)


if __name__ == "__main__":
    unittest.main()
