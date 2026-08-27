import json
from pathlib import Path
import tempfile
import unittest

from tools.analyze_pw0313_m1_native_k4 import analyze


class Pw0313AnalysisTests(unittest.TestCase):
    def _report(self, path: Path, expert: int, passed: bool) -> Path:
        semantic = {
            "gates": {"pass": passed},
            "expert_output": {"m1_vs_m4": {"relative_l2": 0.0 if passed else 0.006}},
        }
        report = {
            "schema_version": 1,
            "experiment_id": "PW-0313",
            "revision": "m1-native-k4-v1",
            "expert": expert,
            "commit": "a" * 40,
            "failure": None,
            "status": "pass" if passed else "failed",
            "complete_seconds": 1.0,
            "peak_rss_bytes": 100,
            "deterministic_tree": {"files": [{"path": "payload", "sha256": "b" * 64}], "total_bytes": 1},
            "projections": {"gate": {"classification": "payload_identical"}},
            "semantic": semantic,
            "safety_snapshots": [
                {
                    "release_boundary": True,
                    "swap_growth_bytes": 0,
                    "new_throttled_pages": 0,
                    "system_memory_free_percent": 50,
                    "process_physical_footprint_bytes": 100,
                }
            ],
        }
        path.write_text(json.dumps(report))
        return path

    def test_split_policy_decision_is_mechanical(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            policy = [self._report(root / f"p{i}.json", 199, True) for i in range(2)]
            control = [self._report(root / f"c{i}.json", 41, False) for i in range(2)]
            result = analyze(policy, control)
            self.assertEqual(
                result["decision"],
                "authorize_policy_relevant_m1_native_k4_but_prohibit_expert_41_expansion",
            )
            self.assertTrue(result["gates"]["policy_expert_pass"])
            self.assertFalse(result["gates"]["control_expert_pass"])

    def test_repeat_mismatch_prevents_policy_qualification(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            policy = [self._report(root / f"p{i}.json", 199, True) for i in range(2)]
            control = [self._report(root / f"c{i}.json", 41, False) for i in range(2)]
            changed = json.loads(policy[1].read_text())
            changed["deterministic_tree"]["files"][0]["sha256"] = "c" * 64
            policy[1].write_text(json.dumps(changed))
            result = analyze(policy, control)
            self.assertEqual(result["decision"], "keep_authenticated_m4_artifacts_only")


if __name__ == "__main__":
    unittest.main()
