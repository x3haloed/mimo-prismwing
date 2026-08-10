import unittest

from tools.analyze_mi100_32gb_complete_envelope import (
    arithmetic_ledger,
    capacity_ledger,
    power_ledger,
)


class Mi100CompleteEnvelopeTests(unittest.TestCase):
    def test_bf16_fails_and_l3_fp16_passes_complete_arithmetic(self) -> None:
        rows = {row["mode"]: row for row in arithmetic_ledger()}
        self.assertFalse(rows["bf16_matrix_source_oriented_ceiling"]["passes_1m_ttft_arithmetic_gate"])
        self.assertAlmostEqual(
            rows["bf16_matrix_source_oriented_ceiling"]["floor_seconds"],
            2301.8085305624927,
        )
        self.assertTrue(rows["l3_fp16_matrix_ceiling"]["passes_1m_ttft_arithmetic_gate"])
        self.assertAlmostEqual(
            rows["l3_fp16_matrix_ceiling"]["floor_seconds"],
            1155.5142807258765,
        )

    def test_32gb_capacity_matches_prior_exact_ledger(self) -> None:
        capacity = capacity_ledger()
        self.assertFalse(capacity["full_source_resident_control_fits"])
        self.assertEqual(capacity["full_source_resident_control_over_hbm_bytes"], 6_221_107_536)
        self.assertEqual(capacity["optimistic_complete_expert_slots"], 261)

    def test_single_card_power_is_margin_not_installation_proof(self) -> None:
        power = power_ledger()
        self.assertEqual(power["gpu_plus_cpu_nameplate_watts"], 470)
        self.assertEqual(power["combined_12v_headroom_after_gpu_plus_cpu_watts"], 262)
        self.assertFalse(power["single_card_nameplate_is_installation_proof"])
        self.assertFalse(power["auxiliary_cable_requirement_authenticated"])


if __name__ == "__main__":
    unittest.main()
