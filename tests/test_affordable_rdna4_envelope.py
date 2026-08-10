import unittest

from tools.analyze_affordable_rdna4_envelope import (
    arithmetic_ledger,
    capacity_ledger,
    power_ledger,
)


class AffordableRdna4EnvelopeTests(unittest.TestCase):
    def test_dense_half_fails_and_sparse_diagnostic_is_inadmissible(self) -> None:
        rows = {row["mode"]: row for row in arithmetic_ledger()}
        dense = rows["dense_bf16_f32acc_source_oriented_ceiling"]
        self.assertTrue(dense["unchanged_source_weights_admissible"])
        self.assertFalse(dense["passes_1m_ttft_arithmetic_gate"])
        self.assertAlmostEqual(dense["floor_seconds"], 2064.399802048224)
        sparse = rows["structured_sparse_half_diagnostic"]
        self.assertFalse(sparse["unchanged_source_weights_admissible"])
        self.assertTrue(sparse["passes_1m_ttft_arithmetic_gate"])
        self.assertAlmostEqual(sparse["floor_seconds"], 1040.94143950886)

    def test_exact_one_million_kv_alone_exceeds_16gb(self) -> None:
        capacity = capacity_ledger()
        self.assertFalse(capacity["kv_alone_fits"])
        self.assertEqual(capacity["kv_alone_over_vram_bytes"], 7_065_559_040)
        self.assertEqual(capacity["full_source_resident_control_over_vram_bytes"], 22_221_107_536)

    def test_power_has_margin_but_is_not_installation_proof(self) -> None:
        power = power_ledger()
        self.assertEqual(power["gpu_plus_cpu_nameplate_watts"], 330)
        self.assertEqual(power["combined_12v_headroom_after_gpu_plus_cpu_watts"], 402)
        self.assertFalse(power["single_card_nameplate_is_installation_proof"])


if __name__ == "__main__":
    unittest.main()
