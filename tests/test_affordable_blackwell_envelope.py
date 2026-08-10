import unittest

from tools.analyze_affordable_blackwell_envelope import (
    arithmetic_ledger,
    capacity_ledger,
    derived_rates,
    power_ledger,
)


class AffordableBlackwellEnvelopeTests(unittest.TestCase):
    def test_same_generation_dense_rates_are_not_ai_tops(self) -> None:
        rates = derived_rates()
        self.assertEqual(rates["direct_fp32"], 23_685_120_000_000)
        self.assertEqual(rates["bf16_tensor_fp32_accumulate"], 47_343_451_433_121)
        self.assertEqual(rates["fp16_tensor_fp16_accumulate"], 94_763_634_554_140)

    def test_even_favorable_fp16_accumulation_fails_complete_arithmetic(self) -> None:
        rows = {row["mode"]: row for row in arithmetic_ledger()}
        self.assertFalse(
            rows["bf16_tensor_fp32_accumulate_source_oriented_ceiling"]
            ["passes_1m_ttft_arithmetic_gate"]
        )
        self.assertAlmostEqual(
            rows["bf16_tensor_fp32_accumulate_source_oriented_ceiling"]["floor_seconds"],
            4453.821314194149,
        )
        self.assertFalse(
            rows["l3_fp16_tensor_fp16_accumulate_ceiling"]
            ["passes_1m_ttft_arithmetic_gate"]
        )
        self.assertAlmostEqual(
            rows["l3_fp16_tensor_fp16_accumulate_ceiling"]["floor_seconds"],
            2242.4320203829857,
        )

    def test_exact_one_million_kv_alone_exceeds_16gb(self) -> None:
        capacity = capacity_ledger()
        self.assertFalse(capacity["kv_alone_fits"])
        self.assertEqual(capacity["kv_alone_over_vram_bytes"], 7_065_559_040)
        self.assertEqual(capacity["full_source_resident_control_over_vram_bytes"], 22_221_107_536)

    def test_power_has_margin_but_is_not_installation_proof(self) -> None:
        power = power_ledger()
        self.assertEqual(power["gpu_plus_cpu_nameplate_watts"], 350)
        self.assertEqual(power["combined_12v_headroom_after_gpu_plus_cpu_watts"], 382)
        self.assertFalse(power["single_card_nameplate_is_installation_proof"])


if __name__ == "__main__":
    unittest.main()
