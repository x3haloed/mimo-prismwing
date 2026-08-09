import unittest

from tools.analyze_wide_proposer_prerequisite import (
    expected_accepted_positions,
    linear_block_shape,
    solve_conditional_match_probability,
    tree_shape,
)


class WideProposerPrerequisiteTests(unittest.TestCase):
    def test_required_50_tps_probability_reconstructs_a(self):
        probability = solve_conditional_match_probability(137, 125)
        self.assertAlmostEqual(probability, 0.9986313247790672)
        self.assertAlmostEqual(expected_accepted_positions(137, probability), 125)

    def test_published_dflash_width_cannot_cross_owned_host_requirements(self):
        shape = linear_block_shape(16, 137)
        self.assertEqual(shape["maximum_A_per_target_transaction"], 16)
        self.assertEqual(shape["target_transactions_to_span_q_positions"], 9)
        self.assertFalse(shape["can_reach_q137_34_3_requirement_in_one_transaction"])
        self.assertFalse(shape["can_reach_q137_50_requirement_in_one_transaction"])

    def test_supplied_width_needs_eighteen_distinct_target_transactions(self):
        shape = linear_block_shape(8, 137)
        self.assertEqual(shape["target_transactions_to_span_q_positions"], 18)

    def test_tree_node_budget_is_nearly_linear_at_50_tps(self):
        shape = tree_shape(137, 125)
        self.assertEqual(shape["minimum_root_to_leaf_depth_including_anchor"], 125)
        self.assertEqual(shape["maximum_off_path_nodes"], 12)

    def test_probability_domain_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "in \\[0, 1\\]"):
            expected_accepted_positions(16, 1.01)
        with self.assertRaisesRegex(ValueError, "in \\[1, q\\]"):
            solve_conditional_match_probability(16, 17)


if __name__ == "__main__":
    unittest.main()
