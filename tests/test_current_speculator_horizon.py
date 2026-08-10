import unittest

from tools.analyze_current_speculator_horizon import (
    REQUIRED_ACCEPTED,
    adjudicate,
    published_configurations,
    validate_pw0170,
)


class CurrentSpeculatorHorizonTests(unittest.TestCase):
    def test_all_published_paths_are_structurally_below_A56(self) -> None:
        result = adjudicate(published_configurations())
        self.assertTrue(result["all_audited_direct_configurations_structurally_below_minimum"])
        self.assertEqual(result["least_demanding_required_A"], 56)
        strongest = result["strongest_published_configuration_by_granted_path"]
        self.assertEqual(strongest["name"], "BASTION")
        self.assertEqual(strongest["maximum_granted_accepted_path"], 17)
        self.assertEqual(strongest["shortfall_tokens"], 39)

    def test_cross_model_means_are_diagnostic_ratios_only(self) -> None:
        result = adjudicate(published_configurations())
        self.assertEqual(result["strongest_reported_slice_mean"]["mean_accepted_length"], 10.60)
        ratios = result["diagnostic_required_A_over_strongest_reported_slice_mean"]
        self.assertAlmostEqual(ratios["fast_storage_34_3_tps"], 56 / 10.60)
        self.assertAlmostEqual(ratios["slow_storage_50_tps"], 113 / 10.60)

    def test_scaled_depth_would_reopen_only_the_named_residual(self) -> None:
        rows = published_configurations()
        rows[-1]["configured_tree_depth"] = 55
        rows[-1]["maximum_granted_accepted_path"] = 56
        result = adjudicate(rows)
        self.assertFalse(result["all_audited_direct_configurations_structurally_below_minimum"])

    def test_pw0170_horizons_fail_closed(self) -> None:
        report = {
            "storage_scenarios": [
                {
                    "lanes": 4,
                    "granted_nameplate_bytes_per_second_per_lane": speed,
                    "targets": {
                        "34.3": {"minimum_integer_A": a34},
                        "50.0": {"minimum_integer_A": a50},
                    },
                }
                for speed, a34, a50 in (
                    (2_500_000_000.0, 77, 113),
                    (3_500_000_000.0, 56, 81),
                )
            ]
        }
        validate_pw0170(report)
        report["storage_scenarios"][0]["targets"]["50.0"]["minimum_integer_A"] = 112
        with self.assertRaisesRegex(ValueError, "acceptance horizons"):
            validate_pw0170(report)

    def test_required_horizons_preserve_both_storage_branches(self) -> None:
        self.assertEqual(REQUIRED_ACCEPTED["slow_storage_34_3_tps"], 77)
        self.assertEqual(REQUIRED_ACCEPTED["slow_storage_50_tps"], 113)
        self.assertEqual(REQUIRED_ACCEPTED["fast_storage_34_3_tps"], 56)
        self.assertEqual(REQUIRED_ACCEPTED["fast_storage_50_tps"], 81)


if __name__ == "__main__":
    unittest.main()
