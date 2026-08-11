import unittest

from tools.analyze_cost_adaptive_verification_horizon import (
    accepted_tokens,
    aggregate,
    best_fixed_q,
    optimal_ratio_choices,
)


class CostAdaptiveVerificationHorizonTests(unittest.TestCase):
    def test_acceptance_counts_correction_and_full_convergence(self):
        proposal = [10, 11, 12, 13, 14, 15, 16, 17]
        posterior = [11, 12, 99, 14, 15, 16, 17, 18]
        self.assertEqual(accepted_tokens(proposal, posterior, 2), 1)
        self.assertEqual(accepted_tokens(proposal, posterior, 3), 2)
        self.assertEqual(accepted_tokens(proposal, posterior, 8), 3)
        self.assertEqual(accepted_tokens(proposal, proposal[1:] + [18], 8), 7)

    def test_best_fixed_q_uses_aggregate_tps(self):
        row = {
            "q": {
                str(q): {
                    "A": q - 1,
                    "modeled_wall_ms": 100.0 * q,
                    "expert_units": q,
                }
                for q in range(2, 9)
            }
        }
        q, result = best_fixed_q([row])
        self.assertEqual(q, 8)
        self.assertEqual(result["A"], 7)

    def test_aggregate_rejects_empty_or_mismatched_choices(self):
        with self.assertRaises(ValueError):
            aggregate([], [])
        with self.assertRaises(ValueError):
            aggregate([{"q": {}}], [])

    def test_ratio_oracle_optimizes_aggregate_not_local_rates(self):
        rows = [
            {
                "q": {
                    str(q): {
                        "A": 1 if q < 8 else 2,
                        "modeled_wall_ms": 1.0 if q < 8 else 100.0,
                        "expert_units": q,
                    }
                    for q in range(2, 9)
                }
            },
            {
                "q": {
                    str(q): {
                        "A": 1 if q < 8 else 100,
                        "modeled_wall_ms": 100.0 if q < 8 else 101.0,
                        "expert_units": q,
                    }
                    for q in range(2, 9)
                }
            },
        ]
        choices, result = optimal_ratio_choices(rows)
        self.assertEqual(choices, [2, 8])
        self.assertGreater(result["modeled_tps"], 900.0)


if __name__ == "__main__":
    unittest.main()
