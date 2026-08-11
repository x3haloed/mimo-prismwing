import unittest
import subprocess
import sys
from pathlib import Path

from tools.build_native_mtp_corpus_manifest import route_counts


class NativeMtpCorpusManifestTests(unittest.TestCase):
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
        self.assertEqual(counts[(1, 7)], 9)
        self.assertEqual(counts[(1, 0)], 1)
        self.assertEqual(counts[(1, 7)], 9)
        self.assertNotIn((1, 8), counts)
        self.assertNotIn((0, 7), counts)


if __name__ == "__main__":
    unittest.main()
