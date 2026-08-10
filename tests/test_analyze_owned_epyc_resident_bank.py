import unittest

from tools.analyze_owned_epyc_resident_bank import (
    CENSUS_EVIDENCE_CLASS,
    DIMM_PRICE_USD,
    EXPECTED_TENSOR_BYTES,
    capacity_row,
    parse_memory_listing,
    resident_q137_envelope,
)


class OwnedEpycResidentBankTests(unittest.TestCase):
    def test_five_modules_are_byte_minimum_but_not_enumerated_population(self):
        four = capacity_row(4)
        five = capacity_row(5)
        self.assertFalse(four["fits_complete_tensor_payload"])
        self.assertTrue(five["fits_complete_tensor_payload"])
        self.assertEqual(
            five["population_status"],
            "byte_minimum_not_explicitly_enumerated_by_manual",
        )
        self.assertEqual(five["tensor_bytes"], EXPECTED_TENSOR_BYTES)

    def test_six_and_eight_module_status_preserves_manual_distinction(self):
        self.assertEqual(
            capacity_row(6)["population_status"],
            "manual_enumerates_unbalanced_not_recommended",
        )
        self.assertEqual(
            capacity_row(8)["population_status"],
            "manual_enumerates_balanced_recommended",
        )

    def test_resident_nameplate_reduces_but_does_not_remove_acceptance(self):
        envelope = resident_q137_envelope()
        self.assertEqual(envelope["targets"]["34.3"]["minimum_integer_A"], 32)
        self.assertEqual(envelope["targets"]["50.0"]["minimum_integer_A"], 47)
        self.assertLess(
            envelope["dual_pcie3_x16_nameplate_bytes_per_second"],
            envelope["five_channel_ddr4_2400_nameplate_bytes_per_second"],
        )

    def test_listing_parser_fails_closed_and_extracts_bound_price(self):
        html = (
            "HMAA8GL7MMR4N-UH 64GB DDR4-2400 ECC LRDIMM "
            "<strong>247</strong><sup>.19</sup> Sold  by A-Tech Add to cart "
            'data-pp-amount="247.19"'
        )
        self.assertEqual(parse_memory_listing(html)["unit_price_usd"], DIMM_PRICE_USD)
        with self.assertRaisesRegex(ValueError, "semantic mismatch"):
            parse_memory_listing(html.replace("Add to cart", "Unavailable"))

    def test_invalid_capacity_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "positive"):
            capacity_row(0)

    def test_census_evidence_class_is_not_overstated(self):
        self.assertEqual(
            CENSUS_EVIDENCE_CLASS,
            "pinned_remote_headers_not_local_payload_verification",
        )


if __name__ == "__main__":
    unittest.main()
