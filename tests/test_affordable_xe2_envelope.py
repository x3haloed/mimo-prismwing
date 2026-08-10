import unittest

from tools.analyze_affordable_xe2_envelope import (
    arithmetic_ledger,
    capacity_ledger,
    derived_bf16_peak,
    power_ledger,
)


class AffordableXe2EnvelopeTests(unittest.TestCase):
    def test_xe2_dpas_ratio_rejects_ordinary_dense_one_million(self) -> None:
        self.assertEqual(derived_bf16_peak(), 116_500_000_000_000)
        arithmetic = arithmetic_ledger()
        self.assertFalse(arithmetic["passes_1m_ttft_arithmetic_gate"])
        self.assertAlmostEqual(arithmetic["floor_seconds"], 1826.6923060599893)
        self.assertAlmostEqual(arithmetic["remaining_1m_ttft_seconds"], -26.692306059989278)

    def test_exact_one_million_kv_alone_exceeds_12gb(self) -> None:
        capacity = capacity_ledger()
        self.assertFalse(capacity["kv_alone_fits"])
        self.assertEqual(capacity["kv_alone_over_vram_bytes"], 11_065_559_040)
        self.assertEqual(capacity["full_source_resident_control_over_vram_bytes"], 26_221_107_536)

    def test_single_card_power_has_margin_but_is_not_installation_proof(self) -> None:
        power = power_ledger()
        self.assertEqual(power["gpu_plus_cpu_nameplate_watts"], 360)
        self.assertEqual(power["combined_12v_headroom_after_gpu_plus_cpu_watts"], 372)
        self.assertFalse(power["single_card_nameplate_is_installation_proof"])


if __name__ == "__main__":
    unittest.main()
