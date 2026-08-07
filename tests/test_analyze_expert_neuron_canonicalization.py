import unittest

import numpy as np

from tools.analyze_expert_neuron_canonicalization import (
    COMPONENT_BYTES,
    EXPANDED_EXPERT_BYTES,
    EXPANDED_OVERHEAD_BYTES,
    INTERMEDIATE,
    NEURON_BYTES,
    PERMUTATION_BYTES,
    SOURCE_EXPERT_BYTES,
    assignment,
    components,
    xor_bytes,
)


class ExpertNeuronCanonicalizationTests(unittest.TestCase):
    def test_representation_byte_ledger_clears_phase_a_gate(self):
        self.assertEqual(COMPONENT_BYTES, (4096, 128, 4096, 128, 4096, 128))
        self.assertEqual(NEURON_BYTES, 12672)
        self.assertEqual(EXPANDED_EXPERT_BYTES, 25_952_256)
        self.assertEqual(EXPANDED_OVERHEAD_BYTES, 780_288)
        self.assertEqual(PERMUTATION_BYTES, 4096)
        self.assertLess(
            (EXPANDED_OVERHEAD_BYTES + PERMUTATION_BYTES) / SOURCE_EXPERT_BYTES,
            0.10,
        )

    def test_components_and_xor_fail_closed(self):
        record = bytes(index % 251 for index in range(NEURON_BYTES))
        self.assertEqual(sum(map(len, components(record))), NEURON_BYTES)
        residual = xor_bytes(record, bytes(NEURON_BYTES))
        self.assertEqual(xor_bytes(residual, bytes(NEURON_BYTES)), record)
        with self.assertRaises(ValueError):
            components(record[:-1])
        with self.assertRaises(ValueError):
            xor_bytes(b"a", b"bb")

    def test_assignment_is_bijective_and_deterministic_on_ties(self):
        reference = np.zeros((INTERMEDIATE, 384), dtype=np.float64)
        candidate = np.zeros_like(reference)
        # Give the distance range one nonzero row while retaining many exact ties.
        reference[-1, 0] = 1.0
        first, first_evidence = assignment(reference, candidate)
        second, second_evidence = assignment(reference, candidate)
        self.assertEqual(first, second)
        self.assertEqual(sorted(first), list(range(INTERMEDIATE)))
        self.assertEqual(first_evidence["combined_cost_sha256"], second_evidence["combined_cost_sha256"])


if __name__ == "__main__":
    unittest.main()
