import unittest

from tools.analyze_pw0324_onboard_closure import (
    no_fast_branch_has_full_capability_promotion,
    required_uniform_reduction,
)


class Pw0324Tests(unittest.TestCase):
    def test_required_uniform_reduction_accounts_for_accepted_tokens(self):
        result = required_uniform_reduction(1000, 4, 100, 2)
        self.assertEqual(result["allowed_bytes"], 200)
        self.assertEqual(result["additional_reduction_factor"], 5)
        self.assertEqual(result["maximum_remaining_fraction"], 0.2)
        self.assertEqual(result["required_reduction_fraction"], 0.8)

    def test_required_uniform_reduction_fails_closed(self):
        with self.assertRaises(ValueError):
            required_uniform_reduction(1000, 0, 100, 2)

    def test_full_capability_condition_is_derived(self):
        portfolio = {"candidate": {"survives_two_tps_closure": False}}
        self.assertTrue(no_fast_branch_has_full_capability_promotion(
            {"status": "conditional; not_general_default"}, portfolio
        ))
        portfolio["candidate"]["survives_two_tps_closure"] = True
        self.assertFalse(no_fast_branch_has_full_capability_promotion(
            {"status": "conditional; not_general_default"}, portfolio
        ))
        self.assertFalse(no_fast_branch_has_full_capability_promotion(
            {"status": "full_capability production"}, {}
        ))


if __name__ == "__main__":
    unittest.main()
