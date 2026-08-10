import unittest

from tools.analyze_approximate_speculative_suffix_reuse import (
    adjudicate,
    paper_observation,
    validate_predecessors,
)


class ApproximateSpeculativeSuffixReuseTests(unittest.TestCase):
    def test_request_budget_is_not_misread_as_proposal_depth(self) -> None:
        observation = paper_observation()
        self.assertEqual(observation["selected_request_regret_budget_B"], 8)
        self.assertEqual(observation["draft_tokens"], 7)
        self.assertEqual(observation["maximum_granted_accepted_path"], 8)

    def test_released_asd_fails_structure_and_declared_fidelity_coverage(self) -> None:
        result = adjudicate(paper_observation())
        self.assertFalse(result["structural_gate_passes"])
        self.assertEqual(result["structural_shortfall_tokens"], 48)
        self.assertFalse(result["declared_L3_gate_evidence_complete"])
        self.assertFalse(result["released_configuration_passes"])
        self.assertIn("hosted_top20_logprob_gate_reported", result["missing_declared_L3_evidence"])

    def test_scaled_horizon_alone_does_not_waive_fidelity(self) -> None:
        observation = paper_observation()
        observation["draft_tokens"] = 55
        observation["maximum_granted_accepted_path"] = 56
        with self.assertRaisesRegex(ValueError, "structural transcription"):
            adjudicate(observation)

    def test_predecessor_minimum_A_fails_closed(self) -> None:
        target = "USD $500 total; mean Jensen-Shannon divergence at most 0.01; L3 — Bounded approximation"
        pw0170 = {
            "storage_scenarios": [
                {
                    "lanes": 4,
                    "granted_nameplate_bytes_per_second_per_lane": 3_500_000_000.0,
                    "targets": {
                        "34.3": {"minimum_integer_A": 56},
                        "50.0": {"minimum_integer_A": 81},
                    },
                }
            ]
        }
        pw0173 = {
            "decision": "reject_audited_released_configurations_as_direct_pw0170_proposer;retain_only_unproven_new_mimo_specific_q137_branch"
        }
        validate_predecessors(target, pw0170, pw0173)
        pw0170["storage_scenarios"][0]["targets"]["34.3"]["minimum_integer_A"] = 55
        with self.assertRaisesRegex(ValueError, "minimum A"):
            validate_predecessors(target, pw0170, pw0173)

    def test_unfavorable_behavior_is_preserved(self) -> None:
        observation = paper_observation()
        self.assertTrue(observation["reported_hash_divergence_above_95_percent_on_named_primary_tasks"])
        self.assertEqual(observation["worst_reported_task_score_point_change_percentage_points"], -1.52)
        self.assertFalse(observation["hosted_top20_logprob_gate_reported"])


if __name__ == "__main__":
    unittest.main()
