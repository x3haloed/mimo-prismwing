import unittest

from tools.analyze_affordable_xehpg_envelope import (
    arithmetic_ledger,
    capacity_ledger,
    derived_bf16_peak,
    power_ledger,
)


class AffordableXeHpgEnvelopeTests(unittest.TestCase):
    def test_xehpg_official_ratio_retains_ordinary_dense_one_million(self) -> None:
        self.assertEqual(derived_bf16_peak(), 131_000_000_000_000)
        arithmetic = arithmetic_ledger()
        self.assertTrue(arithmetic["passes_1m_ttft_arithmetic_gate"])
        self.assertAlmostEqual(arithmetic["floor_seconds"], 1625.6405684427161)
        self.assertAlmostEqual(arithmetic["remaining_1m_ttft_seconds"], 174.35943155728387)

    def test_exact_one_million_kv_exceeds_16gb(self) -> None:
        capacity = capacity_ledger()
        self.assertFalse(capacity["kv_alone_fits"])
        self.assertEqual(capacity["kv_alone_over_vram_bytes"], 7_065_559_040)
        self.assertEqual(capacity["full_source_resident_control_over_vram_bytes"], 22_221_107_536)

    def test_single_card_power_has_margin_but_is_not_installation_proof(self) -> None:
        power = power_ledger()
        self.assertEqual(power["gpu_plus_cpu_nameplate_watts"], 395)
        self.assertEqual(power["combined_12v_headroom_after_gpu_plus_cpu_watts"], 337)
        self.assertFalse(power["single_card_nameplate_is_installation_proof"])


if __name__ == "__main__":
    unittest.main()
