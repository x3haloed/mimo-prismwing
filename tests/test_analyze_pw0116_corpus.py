import unittest
from pathlib import Path

from tools.analyze_pw0116_corpus import analyze


class Pw0116CorpusTests(unittest.TestCase):
    def test_real_corpus_closes_all_frozen_counts_and_safety_gates(self):
        result = analyze(
            Path("/Users/chad/Models/mimo-prismwing/evidence/PW-0116/corpus-001/manifest.json")
        )
        self.assertEqual(result["payload_bytes"], 132_120_576)
        self.assertEqual([row["layer"] for row in result["layers"]], [4, 24, 46])
        self.assertTrue(result["gates_passed"])
        self.assertIsNone(result["performance_claim"])


if __name__ == "__main__":
    unittest.main()
