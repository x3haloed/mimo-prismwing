import unittest
from pathlib import Path

from tools.analyze_r720_cpu_arithmetic_ceiling import analyze


class R720CpuArithmeticCeilingAnalysisTests(unittest.TestCase):
    def test_real_ceiling_validates_cpu_only_rejection(self):
        result = analyze(
            Path("/Users/chad/Models/mimo-prismwing/evidence/PW-0127/run-001.json")
        )
        self.assertTrue(result["evidence_valid"])
        self.assertFalse(result["experiment_passed"])
        self.assertLess(result["arithmetic_ceiling"]["impossible_maximum_tps"], 39)
        self.assertGreater(
            result["arithmetic_ceiling"]["targets"]["50.0"][
                "required_fraction_of_impossible_peak"
            ],
            1.28,
        )
        self.assertIsNone(result["performance_claim"])


if __name__ == "__main__":
    unittest.main()
