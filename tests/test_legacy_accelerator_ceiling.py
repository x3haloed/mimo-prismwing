import unittest

from tools.run_legacy_accelerator_ceiling import (
    configuration_ceiling,
    route_window_ceiling,
)


class LegacyAcceleratorCeilingTests(unittest.TestCase):
    def test_full_target_prefill_rejects_all_named_configurations(self):
        macs = 14_820_573_184
        one_m40 = configuration_ceiling("one_m40", 1, 7e12, macs)
        two_m40 = configuration_ceiling("two_m40", 2, 7e12, macs)
        one_p40 = configuration_ceiling("one_p40", 1, 12e12, macs)
        self.assertAlmostEqual(one_m40["mandatory_matrix_prefill_floor_seconds"], 29.088465523061824)
        self.assertAlmostEqual(two_m40["mandatory_matrix_prefill_floor_seconds"], 15.65002448152059)
        self.assertAlmostEqual(one_p40["mandatory_matrix_prefill_floor_seconds"], 18.029894384428225)
        self.assertFalse(one_m40["passes_impossible_prefill_floor"])
        self.assertFalse(two_m40["passes_impossible_prefill_floor"])
        self.assertFalse(one_p40["passes_impossible_prefill_floor"])
        self.assertGreater(one_m40["ordinary_decode_compute_tps_ceiling"], 50)

    def test_route_window_accounts_transfer_and_bounded_layer_arenas(self):
        result = route_window_ceiling(
            137,
            0,
            [31] + [19] * 45 + [17],
            25_171_968,
            1,
        )
        self.assertEqual(result["total_layer_expert_records"], 903)
        self.assertEqual(result["source_expert_transfer_bytes"], 22_730_287_104)
        self.assertAlmostEqual(result["impossible_perfect_acceptance_tps"], 94.95251820291305)
        self.assertEqual(result["maximum_single_layer_expert_bytes"], 780_331_008)
        self.assertTrue(result["three_arenas_fit_24_decimal_gb"])

    def test_invalid_inputs_fail_closed(self):
        with self.assertRaises(ValueError):
            configuration_ceiling("", 1, 7e12, 1)
        with self.assertRaises(ValueError):
            configuration_ceiling("bad", 0, 7e12, 1)
        with self.assertRaises(ValueError):
            route_window_ceiling(1, 0, [1] * 46, 1, 1)
        with self.assertRaises(ValueError):
            route_window_ceiling(1, -1, [1] * 47, 1, 1)


if __name__ == "__main__":
    unittest.main()
