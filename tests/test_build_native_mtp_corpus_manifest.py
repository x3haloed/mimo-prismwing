import unittest
import subprocess
import sys
from pathlib import Path

from tools.build_native_mtp_corpus_manifest import (
    FIRST_MTP_EVALUABLE_TRANSACTION,
    HIDDEN_ROW_BYTES,
    WINDOW_BYTES,
    mtp_history_binding,
    mtp_target_hidden_binding,
    route_counts,
)


class NativeMtpCorpusManifestTests(unittest.TestCase):
    def test_complete_mtp_history_pairs_shifted_tokens_with_all_target_hiddens(self):
        report = {
            "prompt_token_ids": [10, 11],
            "generated_token_ids": [12, 13, 14],
            "transactions": [
                {
                    "index": 0,
                    "retained_proposal_rows": 1,
                    "emitted_token_ids": [13],
                    "proposal_token_ids": [12, 99],
                },
                {
                    "index": 1,
                    "retained_proposal_rows": 1,
                    "emitted_token_ids": [14],
                    "proposal_token_ids": [13, 98],
                },
            ],
        }
        binding = mtp_history_binding(report, 1, "prefill.f32", "verify.f32")
        self.assertEqual(binding["target_input_token_ids"], [10, 11, 12])
        self.assertEqual(binding["mtp_layer0_input_token_ids"], [11, 12, 13])
        self.assertEqual(binding["target_hidden_rows"], 3)
        self.assertEqual(binding["target_hidden_segments"][0]["rows"], 2)
        self.assertEqual(binding["target_hidden_segments"][1]["rows"], 1)

    def test_mtp_anchor_uses_preceding_retained_target_hidden_row(self):
        report = {
            "transactions": [
                {"index": 0, "retained_proposal_rows": 3},
                {"index": 1, "retained_proposal_rows": 7},
            ]
        }
        self.assertEqual(
            mtp_target_hidden_binding(report, 1),
            {
                "target_hidden_source_transaction_index": 0,
                "target_hidden_source_row": 2,
                "target_hidden_byte_offset": 2 * HIDDEN_ROW_BYTES,
                "target_hidden_byte_length": HIDDEN_ROW_BYTES,
            },
        )
        with self.assertRaisesRegex(ValueError, "transaction zero"):
            mtp_target_hidden_binding(report, 0)
        self.assertEqual(WINDOW_BYTES, 8 * HIDDEN_ROW_BYTES)

    def test_cli_imports_when_executed_by_path(self):
        repo = Path(__file__).resolve().parents[1]
        result = subprocess.run(
            [sys.executable, str(repo / "tools" / "build_native_mtp_corpus_manifest.py"), "--help"],
            cwd=repo,
            text=True,
            capture_output=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("evidence_root", result.stdout)

    def test_route_counts_are_layer_qualified_and_first_eight_only(self):
        transactions = []
        for transaction in range(9):
            traces = [{"layer": 0, "selected_experts_by_position": []}]
            traces.append(
                {
                    "layer": 1,
                    "selected_experts_by_position": [[transaction, 7]],
                }
            )
            transactions.append({"verification_layer_traces": traces})
        counts = route_counts({"transactions": transactions})
        self.assertEqual(FIRST_MTP_EVALUABLE_TRANSACTION, 1)
        self.assertEqual(counts[(1, 7)], 9)
        self.assertNotIn((1, 0), counts)
        self.assertEqual(counts[(1, 8)], 1)
        self.assertNotIn((0, 7), counts)


if __name__ == "__main__":
    unittest.main()
