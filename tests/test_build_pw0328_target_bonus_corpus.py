import unittest

from tools.build_pw0328_target_bonus_corpus import (
    HIDDEN_ROW_BYTES,
    WINDOW_BYTES,
    mtp_history_binding,
    selected_primary_transactions,
    target_hidden_binding,
    transaction_semantics,
    validate_aggregate_byte_ledgers,
)


def full_match_transaction(index=0, emitted=None, retained=None):
    authorized = list(range(1, 9))
    emitted = authorized if emitted is None else emitted
    retained = len(emitted) if retained is None else retained
    return {
        "index": index,
        "proposal_token_ids": list(range(8)),
        "posterior_token_ids": authorized,
        "verifier_authorized_token_ids": authorized,
        "emitted_token_ids": emitted,
        "verifier_retained_proposal_rows": 8,
        "retained_proposal_rows": retained,
        "proposal_converged": True,
    }


class Pw0328TargetBonusCorpusTests(unittest.TestCase):
    def test_transaction_zero_binds_complete_prefill_history(self):
        report = {
            "prompt_token_ids": [10, 11],
            "generated_token_ids": [12, 13],
            "transactions": [
                {
                    "index": 0,
                    "proposal_token_ids": [12],
                    "emitted_token_ids": [13],
                    "retained_proposal_rows": 1,
                }
            ],
        }
        hidden = target_hidden_binding(report, 0, "prefill.f32", "verify.f32")
        history = mtp_history_binding(report, 0, "prefill.f32", "verify.f32")
        self.assertEqual(hidden["target_hidden_source"], "prefill")
        self.assertIsNone(hidden["target_hidden_source_transaction_index"])
        self.assertEqual(hidden["target_hidden_source_row"], 1)
        self.assertEqual(hidden["target_hidden_byte_offset"], HIDDEN_ROW_BYTES)
        self.assertEqual(history["target_input_token_ids"], [10, 11])
        self.assertEqual(history["mtp_layer0_input_token_ids"], [11, 12])
        self.assertEqual(history["target_hidden_rows"], 2)
        self.assertEqual(history["target_hidden_segments"], [
            {
                "source": "prefill",
                "file": "prefill.f32",
                "byte_offset": 0,
                "byte_length": 2 * HIDDEN_ROW_BYTES,
                "rows": 2,
            }
        ])

    def test_later_transaction_binds_preceding_retained_verifier_row(self):
        report = {
            "prompt_token_ids": [10, 11],
            "generated_token_ids": [12, 13, 14],
            "transactions": [
                {
                    "index": 0,
                    "proposal_token_ids": [12],
                    "emitted_token_ids": [13, 14],
                    "retained_proposal_rows": 2,
                },
                {
                    "index": 1,
                    "proposal_token_ids": [14],
                    "emitted_token_ids": [15],
                    "retained_proposal_rows": 1,
                },
            ],
        }
        hidden = target_hidden_binding(report, 1, "prefill.f32", "verify.f32")
        history = mtp_history_binding(report, 1, "prefill.f32", "verify.f32")
        self.assertEqual(hidden["target_hidden_source"], "verifier_transaction")
        self.assertEqual(hidden["target_hidden_source_transaction_index"], 0)
        self.assertEqual(hidden["target_hidden_source_row"], 1)
        self.assertEqual(hidden["target_hidden_byte_offset"], HIDDEN_ROW_BYTES)
        self.assertEqual(history["target_input_token_ids"], [10, 11, 12, 13])
        self.assertEqual(history["mtp_layer0_input_token_ids"], [11, 12, 13, 14])
        self.assertEqual(history["target_hidden_rows"], 4)
        self.assertEqual(history["target_hidden_segments"][1]["rows"], 2)
        self.assertEqual(history["target_hidden_segments"][1]["byte_offset"], 0)
        self.assertEqual(WINDOW_BYTES, 8 * HIDDEN_ROW_BYTES)

    def test_terminal_clip_preserves_full_verifier_authority_A(self):
        result = transaction_semantics(
            full_match_transaction(emitted=[1], retained=1), index=0, terminal=True
        )
        self.assertEqual(result["A"], 8)
        self.assertEqual(result["observable_A"], 1)
        self.assertEqual(result["verifier_retained_proposal_rows"], 8)
        self.assertEqual(result["retained_proposal_rows"], 1)
        self.assertEqual(result["next_anchor_token_id"], 8)

    def test_mismatch_uses_unchanged_first_correction(self):
        transaction = full_match_transaction()
        transaction.update(
            {
                "posterior_token_ids": [1, 2, 99, 4, 5, 6, 7, 8],
                "verifier_authorized_token_ids": [1, 2, 99],
                "emitted_token_ids": [1, 2, 99],
                "verifier_retained_proposal_rows": 3,
                "retained_proposal_rows": 3,
                "proposal_converged": False,
            }
        )
        result = transaction_semantics(transaction, index=0, terminal=False)
        self.assertEqual(result["A"], 3)
        self.assertEqual(result["next_anchor_token_id"], 99)
        self.assertFalse(result["proposal_converged"])

    def test_fixed64_primary_selection_is_transactions_zero_through_seven(self):
        report = {"transactions": [{"index": index} for index in range(10)]}
        selected = selected_primary_transactions(report)
        self.assertEqual([row["index"] for row in selected], list(range(8)))

    def test_nonterminal_clipping_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "nonterminal"):
            transaction_semantics(
                full_match_transaction(emitted=[1], retained=1), index=0, terminal=False
            )

    def test_token_and_retention_types_fail_closed(self):
        invalid_token = full_match_transaction()
        invalid_token["proposal_token_ids"][-1] = 152_576
        with self.assertRaisesRegex(ValueError, "proposal tokens"):
            transaction_semantics(invalid_token, index=0, terminal=False)
        invalid_retention = full_match_transaction(retained=True)
        with self.assertRaisesRegex(ValueError, "observable retention"):
            transaction_semantics(invalid_retention, index=0, terminal=False)

    def test_aggregate_byte_ledgers_close_against_complete_report(self):
        transactions = [
            {"logical_source_bytes": 10, "process_disk_bytes_read": 11},
            {"logical_source_bytes": 20, "process_disk_bytes_read": 21},
        ]
        validate_aggregate_byte_ledgers(
            transactions,
            {"logical_source_bytes": 31, "process_disk_bytes_read": 33},
            category="ordinary",
        )
        with self.assertRaisesRegex(ValueError, "aggregate transaction/report"):
            validate_aggregate_byte_ledgers(
                transactions,
                {"logical_source_bytes": 29, "process_disk_bytes_read": 33},
                category="ordinary",
            )


if __name__ == "__main__":
    unittest.main()
