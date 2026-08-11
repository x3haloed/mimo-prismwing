import unittest

from tools.audit_native_mtp_window_corpus import (
    audit_transaction,
    protected_baseline_survived,
    verification_union,
)


def routed_layer(layer: int, expert_offset: int = 0):
    selected = [
        [((expert_offset + row * 8 + column) % 256) for column in range(8)]
        for row in range(8)
    ]
    return {
        "layer": layer,
        "selected_experts_by_position": selected,
        "route_weights_by_position": [[0.125] * 8 for _ in range(8)],
        "U": len({expert for row in selected for expert in row}) / 8,
    }


class NativeMtpWindowCorpusAuditTests(unittest.TestCase):
    def test_protected_process_addition_is_safe_but_baseline_loss_fails(self):
        start = {"protected_service_pids": {"WindowServer": [10], "nxnode": [20]}}
        additive = {
            "protected_service_pids": {"WindowServer": [10], "nxnode": [20, 21]}
        }
        missing = {"protected_service_pids": {"WindowServer": [10], "nxnode": [21]}}
        self.assertTrue(protected_baseline_survived([start, additive]))
        self.assertFalse(protected_baseline_survived([start, missing]))

    def test_union_is_rederived_from_exact_route_rows(self):
        trace = [
            {
                "layer": 0,
                "selected_experts_by_position": [],
                "route_weights_by_position": [],
                "U": 0.0,
            }
        ] + [routed_layer(layer, layer) for layer in range(1, 48)]
        self.assertEqual(verification_union(trace), 8.0)
        trace[20]["U"] = 7.0
        with self.assertRaisesRegex(ValueError, "layer U"):
            verification_union(trace)

    def test_transaction_fails_closed_on_unauthorized_output(self):
        trace = [
            {
                "layer": 0,
                "selected_experts_by_position": [],
                "route_weights_by_position": [],
                "U": 0.0,
            }
        ] + [routed_layer(layer, layer) for layer in range(1, 48)]
        transaction = {
            "index": 0,
            "proposal_token_ids": list(range(8)),
            "posterior_token_ids": list(range(8, 16)),
            "verifier_authorized_token_ids": [8, 9],
            "emitted_token_ids": [8, 10],
            "retained_proposal_rows": 2,
            "proposal_layer_traces": [[{}] * 48 for _ in range(7)],
            "verification_layer_traces": trace,
            "U": 8.0,
        }
        progress = {
            "phase": "transaction_complete",
            "transaction": 0,
            "emitted_tokens": 2,
            "retained_proposal_rows": 2,
            "U": 8.0,
        }
        with self.assertRaisesRegex(ValueError, "verifier authority"):
            audit_transaction(transaction, progress, 0)


if __name__ == "__main__":
    unittest.main()
