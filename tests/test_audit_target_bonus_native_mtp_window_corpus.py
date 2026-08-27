import unittest

from tools.audit_native_mtp_window_corpus import audit_target_bonus_transaction
from tests.test_audit_native_mtp_window_corpus import routed_layer


def trace():
    return [
        {
            "layer": 0,
            "selected_experts_by_position": [],
            "route_weights_by_position": [],
            "U": 0.0,
        }
    ] + [routed_layer(layer, layer) for layer in range(1, 48)]


def transaction(*, emitted=None, retained=8, verifier_retained=8):
    emitted = list(range(1, 9)) if emitted is None else emitted
    return {
        "index": 0,
        "proposal_token_ids": list(range(8)),
        "posterior_token_ids": list(range(1, 9)),
        "verifier_authorized_token_ids": list(range(1, 9)),
        "emitted_token_ids": emitted,
        "verifier_retained_proposal_rows": verifier_retained,
        "retained_proposal_rows": retained,
        "proposal_converged": True,
        "proposal_layer_traces": [[{}] * 48 for _ in range(7)],
        "verification_layer_traces": trace(),
        "U": 8.0,
    }


def progress(*, accepted=8, retained=8):
    return {
        "phase": "transaction_complete",
        "transaction": 0,
        "emitted_tokens": accepted,
        "retained_proposal_rows": retained,
        "U": 8.0,
    }


class TargetBonusNativeMtpAuditTests(unittest.TestCase):
    def test_full_match_accepts_q_tokens_and_q_verifier_rows(self):
        result = audit_target_bonus_transaction(transaction(), progress(), 0)
        self.assertEqual(result["A"], 8)

    def test_terminal_clipping_retains_only_observable_prefix(self):
        result = audit_target_bonus_transaction(
            transaction(emitted=[1, 2, 3], retained=3),
            progress(accepted=3, retained=3),
            0,
        )
        self.assertEqual(result["A"], 3)

    def test_wrong_bonus_or_verifier_retention_fails_closed(self):
        wrong_bonus = transaction()
        wrong_bonus["verifier_authorized_token_ids"][-1] = 99
        wrong_bonus["emitted_token_ids"][-1] = 99
        with self.assertRaisesRegex(ValueError, "target bonus"):
            audit_target_bonus_transaction(wrong_bonus, progress(), 0)
        with self.assertRaisesRegex(ValueError, "verifier retention"):
            audit_target_bonus_transaction(
                transaction(verifier_retained=7), progress(), 0
            )


if __name__ == "__main__":
    unittest.main()
