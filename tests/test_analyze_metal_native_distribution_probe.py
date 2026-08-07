import unittest
from pathlib import Path

from tools.analyze_metal_native_distribution_probe import analyze


class MetalNativeDistributionProbeTests(unittest.TestCase):
    def test_candidate_passes_only_the_bounded_numerical_gate(self):
        result = analyze(
            Path("/Users/chad/Models/mimo-prismwing/evidence/PW-0114/control-001/report.json"),
            Path("/Users/chad/Models/mimo-prismwing/evidence/PW-0114/candidate-001/report.json"),
            Path("/Users/chad/Models/mimo-prismwing/evidence/PW-0095/oracle-001/manifest.json"),
        )
        self.assertTrue(result["candidate_numerical_continuation_gate_passed"])
        self.assertEqual(result["repair_attribution"]["candidate_counts"], [0, 0, 0])
        self.assertEqual(result["repair_attribution"]["candidate_decoded_weight_bytes"], 0)
        self.assertEqual(
            result["layer_behavior"]["candidate_vs_control"][
                "selected_expert_mismatch_layers"
            ],
            [43, 44, 46],
        )
        self.assertIsNone(result["performance_claim"])


if __name__ == "__main__":
    unittest.main()
