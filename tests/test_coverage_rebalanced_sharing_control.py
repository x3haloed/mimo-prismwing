import unittest

from tools.run_five_expert_four_basis_sharing_pilot import (
    PW0123_SPEC,
    PW0124_SPEC,
    build_partitions,
)


class CoverageRebalancedSharingControlTests(unittest.TestCase):
    def test_rebalanced_spec_preserves_holdout_and_exact_fifth_counts(self):
        development = list(range(73))
        holdout = list(range(168, 224))
        positions = development + holdout
        partitions = build_partitions(positions, PW0124_SPEC)
        self.assertEqual(len(partitions["train"]), 58)
        self.assertEqual(len(partitions["validation"]), 15)
        self.assertEqual(len(partitions["pilot_holdout"]), 56)
        self.assertEqual(
            [positions[index] for index in partitions["pilot_holdout"]], holdout
        )
        changed_positions = development + list(range(1000, 1056))
        changed = build_partitions(changed_positions, PW0124_SPEC)
        self.assertEqual(changed["train"], partitions["train"])
        self.assertEqual(changed["validation"], partitions["validation"])

    def test_specs_preserve_distinct_authority_and_frozen_counts(self):
        self.assertNotEqual(PW0123_SPEC.evidence_class, PW0124_SPEC.evidence_class)
        self.assertNotEqual(PW0123_SPEC.parent_analysis_sha256, PW0124_SPEC.parent_analysis_sha256)
        self.assertEqual(
            PW0124_SPEC.expected_counts[57],
            {"train": 58, "validation": 15, "pilot_holdout": 56},
        )
        self.assertEqual(PW0124_SPEC.seed, 260124)


if __name__ == "__main__":
    unittest.main()
