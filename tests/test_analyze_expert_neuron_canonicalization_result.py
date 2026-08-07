import unittest

from tools.analyze_expert_neuron_canonicalization_result import analyze


class ExpertNeuronCanonicalizationResultTests(unittest.TestCase):
    def test_raw_result_rejects_mechanism(self):
        result = analyze(
            __import__("pathlib").Path(
                "/Users/chad/Models/mimo-prismwing/evidence/PW-0113/run-001.json"
            )
        )
        self.assertFalse(result["canonicalization_signal_gate_passed"])
        self.assertFalse(result["physical_continuation_gate_passed"])
        self.assertLess(result["aligned_fast_source_byte_reduction"], 0.01)
        self.assertTrue(result["exactness_and_accounting_passed"])


if __name__ == "__main__":
    unittest.main()
