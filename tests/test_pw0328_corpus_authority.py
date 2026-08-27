import copy
from pathlib import Path
import tempfile
import unittest

from tools.pw0328_corpus_authority import (
    CATEGORIES,
    PW0328_MANIFEST_PATH,
    PW0328_MANIFEST_SHA256,
    authorized_q1_rows,
    authenticate_pw0328_corpus,
    reconstruct_verification_routes,
    sha256_file,
    target_bonus_commit,
    validate_manifest_window,
    validate_primary_window_sequence,
)


REPO = Path(__file__).resolve().parents[1]


def q8_route_fixture():
    traces = [
        {
            "layer": 0,
            "selected_experts_by_position": [],
            "route_weights_by_position": [],
            "U": 0.0,
        }
    ]
    for layer in range(1, 48):
        traces.append(
            {
                "layer": layer,
                "selected_experts_by_position": [
                    list(range(position * 8, position * 8 + 8))
                    for position in range(8)
                ],
                "route_weights_by_position": [[0.125] * 8 for _ in range(8)],
                "U": 8.0,
            }
        )
    return traces


def proposal_route_fixture():
    traces = [
        {
            "layer": 0,
            "selected_experts_by_position": [],
            "route_weights_by_position": [],
            "U": 0.0,
        }
    ]
    for layer in range(1, 48):
        traces.append(
            {
                "layer": layer,
                "selected_experts_by_position": [list(range(8))],
                "route_weights_by_position": [[0.125] * 8],
                "U": 8.0,
            }
        )
    return traces


class Pw0328CorpusAuthorityTests(unittest.TestCase):
    def test_target_bonus_full_match_and_first_correction(self):
        full = target_bonus_commit(list(range(8)), list(range(1, 9)))
        self.assertEqual(full["authorized"], list(range(1, 9)))
        self.assertEqual(full["retained_proposal_rows"], 8)
        self.assertEqual(full["next_anchor_token_id"], 8)
        self.assertTrue(full["proposal_converged"])

        mismatch = target_bonus_commit(
            list(range(8)), [1, 2, 99, 4, 5, 6, 7, 8]
        )
        self.assertEqual(mismatch["authorized"], [1, 2, 99])
        self.assertEqual(mismatch["retained_proposal_rows"], 3)
        self.assertEqual(mismatch["next_anchor_token_id"], 99)
        self.assertFalse(mismatch["proposal_converged"])

    def test_q8_routes_reconstruct_union_identity_and_both_views(self):
        result = reconstruct_verification_routes(q8_route_fixture(), label="fixture")
        self.assertEqual(result["U"], 8.0)
        self.assertEqual(result["unique_identities"], 47 * 64)
        self.assertEqual(len(result["per_layer_q8"]), 47)
        self.assertEqual(len(result["all_q8_rows"]), 8)
        self.assertEqual(result["per_layer_q8"][0]["union_size"], 64)
        self.assertEqual(result["per_layer_q8"][0]["U"], 8.0)
        self.assertEqual(
            result["per_layer_q8"][0]["identities"][0],
            {"layer": 1, "expert": 0},
        )
        q1 = authorized_q1_rows(result["all_q8_rows"], 3)
        self.assertEqual([row["position"] for row in q1], [0, 1, 2])
        self.assertEqual(len(q1[0]["layers"]), 47)
        self.assertEqual(q1[0]["layers"][0], {"layer": 1, "experts": list(range(8))})

    def test_q8_route_identity_failures_are_closed(self):
        duplicate = q8_route_fixture()
        duplicate[1]["selected_experts_by_position"][0][-1] = 0
        with self.assertRaisesRegex(ValueError, "expert row identity"):
            reconstruct_verification_routes(duplicate, label="duplicate")

        with self.assertRaisesRegex(ValueError, "q8 route row count"):
            reconstruct_verification_routes(
                proposal_route_fixture(), label="proposal-is-not-verifier"
            )

    def test_q1_view_uses_only_first_full_authorized_rows(self):
        rows = reconstruct_verification_routes(
            q8_route_fixture(), label="fixture"
        )["all_q8_rows"]
        projected = authorized_q1_rows(rows, 7)
        self.assertEqual([row["position"] for row in projected], list(range(7)))
        self.assertNotIn(7, [row["position"] for row in projected])
        for invalid_a in (0, 9):
            with self.assertRaises(ValueError):
                authorized_q1_rows(rows, invalid_a)

    def test_category_and_window_order_is_frozen(self):
        rows = [
            {
                "corpus_index": index,
                "category": CATEGORIES[index // 8],
                "transaction_index": index % 8,
            }
            for index in range(32)
        ]
        validate_primary_window_sequence(rows)
        reordered = copy.deepcopy(rows)
        reordered[7], reordered[8] = reordered[8], reordered[7]
        with self.assertRaisesRegex(ValueError, "category/window order"):
            validate_primary_window_sequence(reordered)

    def test_manifest_replay_distinguishes_full_A_from_observable_clip(self):
        row = {
            "verifier_authorized_token_ids": list(range(1, 9)),
            "observable_emitted_token_ids": list(range(1, 8)),
            "A": 8,
            "observable_A": 7,
        }
        validate_manifest_window(row, copy.deepcopy(row), index=31)
        clipped_a = copy.deepcopy(row)
        clipped_a["A"] = 7
        with self.assertRaisesRegex(ValueError, "replay"):
            validate_manifest_window(row, clipped_a, index=31)
        suffix = copy.deepcopy(row)
        suffix["observable_emitted_token_ids"] = list(range(2, 9))
        suffix["observable_A"] = 7
        with self.assertRaisesRegex(ValueError, "replay"):
            validate_manifest_window(row, suffix, index=31)

    @unittest.skipUnless(PW0328_MANIFEST_PATH.is_file(), "canonical PW-0328 corpus unavailable")
    def test_canonical_manifest_hash_drift_fails_before_replay(self):
        self.assertEqual(sha256_file(PW0328_MANIFEST_PATH), PW0328_MANIFEST_SHA256)
        with tempfile.TemporaryDirectory() as directory:
            changed = Path(directory) / "manifest.json"
            changed.write_bytes(PW0328_MANIFEST_PATH.read_bytes() + b"\n")
            with self.assertRaisesRegex(ValueError, "manifest SHA-256 mismatch"):
                authenticate_pw0328_corpus(REPO, changed)

    @unittest.skipUnless(PW0328_MANIFEST_PATH.is_file(), "canonical PW-0328 corpus unavailable")
    def test_canonical_corpus_replays_all_bound_authorities(self):
        result = authenticate_pw0328_corpus(REPO)
        self.assertEqual(result["manifest_sha256"], PW0328_MANIFEST_SHA256)
        self.assertEqual(result["artifact_count"], 24)
        self.assertEqual(len(result["windows"]), 32)
        self.assertEqual(len(result["q1_events"]), 232)
        self.assertEqual(result["control"]["sum_A"], 232)
        self.assertEqual(result["control"]["sum_observable_A"], 231)
        self.assertEqual(
            result["q1_events"][-1]["authorized_token_id"],
            result["windows"][-1]["verifier_authorized_token_ids"][-1],
        )
        terminal = result["windows"][-1]
        self.assertEqual(terminal["A"], 8)
        self.assertEqual(terminal["observable_A"], 7)
        self.assertEqual(
            [row["position"] for row in terminal["authorized_q1_rows"]],
            list(range(8)),
        )


if __name__ == "__main__":
    unittest.main()
