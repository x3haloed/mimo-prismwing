import unittest

from tools.analyze_pw0326_target_bonus import (
    RUST_FIXTURES,
    commit_fixture,
    parse_python_test_summary,
    parse_rust_test_summary,
    require_named_rust_fixtures,
)


class Pw0326TargetBonusTests(unittest.TestCase):
    def test_full_match_emits_suffix_and_target_bonus(self):
        result = commit_fixture([41, 42, 43, 44], [42, 43, 44, 45])
        self.assertEqual(result["emitted"], [42, 43, 44, 45])
        self.assertEqual(result["retained_proposal_rows"], 4)
        self.assertEqual(result["next_anchor"], 45)
        self.assertTrue(result["proposal_converged"])

    def test_mismatch_control_is_unchanged(self):
        result = commit_fixture(
            [264, 13, 15, 13, 15, 15, 15, 15],
            [13, 15, 13, 15, 481, 13, 15, 15],
        )
        self.assertEqual(result["emitted"], [13, 15, 13, 15, 481])
        self.assertEqual(result["retained_proposal_rows"], 5)
        self.assertEqual(result["next_anchor"], 481)
        self.assertFalse(result["proposal_converged"])

    def test_invalid_widths_fail_closed(self):
        with self.assertRaises(ValueError):
            commit_fixture([1], [2])
        with self.assertRaises(ValueError):
            commit_fixture([1, 2], [2, 3, 4])

    def test_complete_library_summary_rejects_filtered_or_failed_tests(self):
        good = "test result: ok. 119 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out"
        self.assertEqual(parse_rust_test_summary(good)["passed"], 119)
        with self.assertRaises(ValueError):
            parse_rust_test_summary(
                "test result: ok. 4 passed; 0 failed; 0 ignored; 0 measured; 115 filtered out"
            )

    def test_named_rust_fixtures_and_python_suite_are_required(self):
        output = "\n".join(f"test text_endpoint::tests::{name} ... ok" for name in RUST_FIXTURES)
        self.assertEqual(require_named_rust_fixtures(output), list(RUST_FIXTURES))
        with self.assertRaises(ValueError):
            require_named_rust_fixtures(output.replace(RUST_FIXTURES[0], "missing"))
        self.assertEqual(
            parse_python_test_summary("Ran 12 tests in 0.01s\n\nOK"),
            {"tests": 12, "status": "OK"},
        )


if __name__ == "__main__":
    unittest.main()
