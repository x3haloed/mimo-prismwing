import hashlib
import math
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from tools.analyze_pw0329_k4_joint_residency import (
    ALIGNMENT_BYTES,
    ALL_IDENTITY_COUNT,
    BANDWIDTH_EXACT,
    BANDWIDTH_EXACT_FLOAT,
    CATEGORIES,
    FIXED_ALLOCATED_BYTES,
    FIXED_LOGICAL_BYTES,
    GIB,
    K4_LOGICAL_BYTES,
    K4_REPACK_STRIDE,
    K4_SCHEMA2_STRIDE,
    K4_TLUT_ALLOCATED_BYTES,
    LARGEST_FIXED_OBJECT_BYTES,
    LOGICAL_SAVING,
    Q8_EXACT_SHARED_LOGICAL_BYTES,
    REPACK_SAVING,
    ROUTED_LAYERS,
    RouteRow,
    SCHEMA2_SAVING,
    SOURCE_LOGICAL_BYTES,
    SOURCE_REPACK_STRIDE,
    SOURCE_SCHEMA2_STRIDE,
    SelectorResult,
    Window,
    aggregate_ledgers,
    disposition,
    derive_schema2_layout,
    density_survival_summary,
    fractional_miss,
    guarded_miss,
    mean_q1_unique_experts,
    nearest_rank_p10,
    normalize_pw0328_authority,
    relaxed_k4_count,
    selection_order_sha256,
    select_fixed_bank,
    select_fixed_bank_naive,
    selector_result_dict,
    whole_record_stride,
    window_storage_ledger,
    verify_zero_correction_payloads,
)


def tiny_grid():
    """Four deterministic categories with equal candidate scores and row caps."""
    windows = []
    rows = []
    for window_index, category in enumerate(CATEGORIES):
        unions = tuple(
            (layer, tuple((layer, expert) for expert in range(8)))
            for layer in ROUTED_LAYERS
        )
        windows.append(
            Window(
                window_index=window_index,
                category=category,
                transaction_index=0,
                accepted_tokens=8,
                unique_experts_per_layer=8.0,
                layer_unions=unions,
            )
        )
        for position in range(8):
            for layer in ROUTED_LAYERS:
                rows.append(
                    RouteRow(
                        category=category,
                        window_index=window_index,
                        transaction_index=0,
                        position=position,
                        layer=layer,
                        identities=tuple((layer, expert) for expert in range(8)),
                    )
                )
    return windows, rows


def varied_grid():
    windows = []
    rows = []
    for window_index, category in enumerate(CATEGORIES):
        by_layer = {}
        for layer in ROUTED_LAYERS:
            start = (window_index * (1 + layer % 3)) % 9
            by_layer[layer] = tuple(
                (layer, expert) for expert in range(start, start + 8)
            )
        windows.append(
            Window(
                window_index=window_index,
                category=category,
                transaction_index=0,
                accepted_tokens=window_index + 2,
                unique_experts_per_layer=1.0,
                layer_unions=tuple(
                    (layer, by_layer[layer]) for layer in ROUTED_LAYERS
                ),
            )
        )
        for position in range(8):
            for layer in ROUTED_LAYERS:
                rows.append(
                    RouteRow(
                        category=category,
                        window_index=window_index,
                        transaction_index=0,
                        position=position,
                        layer=layer,
                        identities=by_layer[layer],
                    )
                )
    return windows, rows


def fake_metrics(*, overall=2.0, category=2.0, p10=2.0):
    return {
        "overall": {
            "optimistic_storage_tps": overall,
            "nearest_rank_p10_window_optimistic_storage_tps": p10,
        },
        "category": {
            name: {"optimistic_storage_tps": category} for name in CATEGORIES
        },
    }


def fake_relaxed(*, strongest_overall=2.0, strongest_category=2.0, strongest_p10=2.0, d6=2.0):
    result = {}
    for density in (4, 5, 6, 8):
        metrics = fake_metrics(overall=d6 if density == 6 else 2.0)
        if density == 8:
            metrics = fake_metrics(
                overall=strongest_overall,
                category=strongest_category,
                p10=strongest_p10,
            )
        result[(density, 12 * GIB)] = {
            "metrics": {"fractional_relaxed": metrics}
        }
    return result


def fake_bank(density=3, residency_bytes=8 * GIB, target=1.25, guarded=1.25, p10=1.01):
    order = ((1, 0),)
    metrics = {
        "current_layout_guarded": fake_metrics(
            overall=guarded, category=guarded, p10=p10
        ),
        "fractional_relaxed": fake_metrics(overall=2.0, category=2.0, p10=p10),
    }
    return SelectorResult(
        density=density,
        residency_bytes=residency_bytes,
        target_tps=target,
        order=order,
        selection_order_sha256=selection_order_sha256(order),
        stop_reason="all_category_deficits_closed",
        initial_category_deficit_seconds={name: 1.0 for name in CATEGORIES},
        remaining_category_deficit_seconds={name: 0.0 for name in CATEGORIES},
        rejected_by_row_cap=(),
        independent_recomputation_pass=True,
        metrics=metrics,
        windows=(),
        coverage={},
        installed_hybrid_expert_bank_bytes=(
            K4_SCHEMA2_STRIDE + (ALL_IDENTITY_COUNT - 1) * SOURCE_SCHEMA2_STRIDE
        ),
        all_source_expert_bank_bytes=ALL_IDENTITY_COUNT * SOURCE_SCHEMA2_STRIDE,
        construction_artifact_bytes=30_000_000,
        estimated_m1_construction_seconds=500,
    )


def schema2_records():
    zero_hash = lambda count: hashlib.sha256(bytes(count)).hexdigest()

    def payload(count, offset=0):
        return {
            "alignment": ALIGNMENT_BYTES,
            "bytes": count,
            "offset": offset,
            "sha256": zero_hash(count),
        }

    panels = {
        "gate": {
            "packed": payload(4_194_304),
            "left_sign": payload(2_048),
            "right_sign": payload(4_096),
            "global_scale": payload(4),
            "row_scale": payload(4_096),
            "correction_left": payload(4_096),
            "correction_right": payload(8_192),
        },
        "up": {
            "packed": payload(4_194_304),
            "left_sign": payload(2_048),
            "right_sign": payload(4_096),
            "global_scale": payload(4),
            "row_scale": payload(4_096),
            "correction_left": payload(4_096),
            "correction_right": payload(8_192),
        },
        "down": {
            "packed": payload(4_194_304),
            "left_sign": payload(4_096),
            "right_sign": payload(2_048),
            "global_scale": payload(4),
            "row_scale": payload(8_192),
            "correction_left": payload(8_192),
            "correction_right": payload(4_096),
        },
    }
    k4 = {
        "format": "qtip_k4_ldlq",
        "projections": {
            name: {"rank": 1, "payloads": values} for name, values in panels.items()
        },
    }
    source = {
        "format": "source_fp8_e4m3_block128",
        "payloads": {
            "gate_weight": payload(8_388_608),
            "gate_scales": payload(2_048),
            "up_weight": payload(8_388_608),
            "up_scales": payload(2_048),
            "down_weight": payload(8_388_608),
            "down_scales": payload(2_048),
        },
    }
    return [k4, source]


class ByteModelTests(unittest.TestCase):
    def test_exact_bandwidth_is_raw_pw0136_derivation(self):
        self.assertEqual(float(BANDWIDTH_EXACT), BANDWIDTH_EXACT_FLOAT)
        self.assertAlmostEqual(
            201_719_808 / float(BANDWIDTH_EXACT), 0.058125375, places=12
        )

    def test_record_sizes_and_savings_close(self):
        self.assertEqual(SOURCE_LOGICAL_BYTES - K4_LOGICAL_BYTES, LOGICAL_SAVING)
        self.assertEqual(SOURCE_SCHEMA2_STRIDE - K4_SCHEMA2_STRIDE, SCHEMA2_SAVING)
        self.assertEqual(SOURCE_REPACK_STRIDE - K4_REPACK_STRIDE, REPACK_SAVING)
        self.assertEqual(whole_record_stride(K4_LOGICAL_BYTES), K4_REPACK_STRIDE)
        self.assertEqual(whole_record_stride(SOURCE_LOGICAL_BYTES), SOURCE_REPACK_STRIDE)

    def test_relaxed_k_is_per_layer_min_n_8d(self):
        self.assertEqual(relaxed_k4_count(17, 3), 17)
        self.assertEqual(relaxed_k4_count(25, 3), 24)
        self.assertEqual(relaxed_k4_count(64, 8), 64)
        with self.assertRaises(ValueError):
            relaxed_k4_count(8, 7)

    def test_U_divides_q8_union_sizes_by_eight(self):
        unions = tuple(
            (layer, tuple((layer, expert) for expert in range(16)))
            for layer in ROUTED_LAYERS
        )
        self.assertEqual(mean_q1_unique_experts(unions), 2.0)

    def test_fractional_joint_residency_cannot_double_spend_R(self):
        common = 80
        experts = 40
        residency = 100
        self.assertEqual(fractional_miss(common + experts, residency), 20)
        self.assertNotEqual(
            fractional_miss(common, residency) + fractional_miss(experts, residency),
            20,
        )

    def test_largest_object_guard_and_fit_all(self):
        self.assertEqual(guarded_miss(100, 100, 30), 0)
        self.assertEqual(guarded_miss(101, 100, 30), 31)
        self.assertEqual(guarded_miss(101, 20, 30), 101)

    def test_window_ledgers_separate_embeddings_and_allocations(self):
        windows, rows = tiny_grid()
        ledger = window_storage_ledger(
            windows[0], residency_bytes=12 * GIB, density=3, rows=rows
        )
        self.assertEqual(
            ledger["exact_logical"]["shared_bytes"]
            - ledger["fractional_relaxed"]["shared_bytes"],
            8 * 8192,
        )
        self.assertEqual(
            ledger["current_layout_guarded"]["shared_bytes"], FIXED_ALLOCATED_BYTES
        )
        self.assertEqual(
            ledger["explicit_allocation_omissions"]["k4_tlut_allocated_bytes"],
            K4_TLUT_ALLOCATED_BYTES,
        )
        self.assertFalse(
            ledger["explicit_allocation_omissions"]["charged_to_current_layout"]
        )

    def test_current_layout_uses_one_joint_total(self):
        windows, rows = tiny_grid()
        ledger = window_storage_ledger(
            windows[0], residency_bytes=8 * GIB, density=3, rows=rows
        )
        model = ledger["current_layout_guarded"]
        self.assertEqual(
            model["joint_total_bytes"], model["shared_bytes"] + model["expert_bytes"]
        )
        self.assertEqual(
            model["bytes_moved"],
            guarded_miss(
                model["joint_total_bytes"], 8 * GIB, LARGEST_FIXED_OBJECT_BYTES
            ),
        )

    def test_schema2_fixture_derives_logical_and_individual_payload_strides(self):
        derived = derive_schema2_layout(schema2_records())
        self.assertEqual(
            derived["logical_by_format"],
            {
                "qtip_k4_ldlq": K4_LOGICAL_BYTES,
                "source_fp8_e4m3_block128": SOURCE_LOGICAL_BYTES,
            },
        )
        self.assertEqual(
            derived["stride_by_format"],
            {
                "qtip_k4_ldlq": K4_SCHEMA2_STRIDE,
                "source_fp8_e4m3_block128": SOURCE_SCHEMA2_STRIDE,
            },
        )
        self.assertEqual(len(derived["correction_payloads"]), 6)

    def test_rank_one_zero_corrections_are_authenticated_and_byte_neutral(self):
        derived = derive_schema2_layout(schema2_records())
        before = (
            derived["logical_by_format"]["qtip_k4_ldlq"],
            derived["stride_by_format"]["qtip_k4_ldlq"],
        )
        verify_zero_correction_payloads(
            derived["correction_payloads"],
            lambda payload: bytes(int(payload["bytes"])),
        )
        after = (
            derived["logical_by_format"]["qtip_k4_ldlq"],
            derived["stride_by_format"]["qtip_k4_ldlq"],
        )
        self.assertEqual(before, after)
        with self.assertRaisesRegex(ValueError, "not all zero"):
            verify_zero_correction_payloads(
                derived["correction_payloads"],
                lambda payload: b"\x01" + bytes(int(payload["bytes"]) - 1),
            )

    def test_non_rank_one_schema2_fixture_rejects(self):
        records = schema2_records()
        records[0]["projections"]["gate"]["rank"] = 2
        with self.assertRaisesRegex(ValueError, "rank is not one"):
            derive_schema2_layout(records)


class AggregateTests(unittest.TestCase):
    def test_p10_is_fourth_lowest(self):
        values = [float(index) for index in range(32, 0, -1)]
        self.assertEqual(nearest_rank_p10(values), 4.0)
        with self.assertRaises(ValueError):
            nearest_rank_p10(values[:-1])

    def test_aggregate_is_token_total_over_wall_not_mean_tps(self):
        rows = []
        for index in range(32):
            wall = 1.0 if index < 31 else 9.0
            category = CATEGORIES[index // 8]
            rows.append(
                {
                    "A": 2,
                    "category": category,
                    "model": {
                        "bytes_moved": int(wall),
                        "storage_wall_seconds": wall,
                        "optimistic_storage_tps": 2.0 / wall,
                        "optimistic_storage_tps_fraction": {
                            "numerator": 2,
                            "denominator": int(wall),
                        },
                        "bandwidth_fraction": {"numerator": 1, "denominator": 1},
                    },
                }
            )
        result = aggregate_ledgers(rows, "model")
        self.assertAlmostEqual(result["overall"]["optimistic_storage_tps"], 64 / 40)
        self.assertEqual(
            result["overall"]["nearest_rank_p10_window_optimistic_storage_tps"],
            2.0,
        )

    def test_unbounded_storage_windows_sort_above_finite_tail(self):
        values = [None] * 29 + [0.8, 0.9, 1.0]
        self.assertIsNone(nearest_rank_p10(values))


class SelectorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.windows, cls.rows = tiny_grid()

    def test_optimized_selector_matches_naive_exactly(self):
        arguments = dict(
            density=3,
            residency_bytes=12 * GIB,
            target_tps=100.0,
            maximum_identities=5,
        )
        optimized = select_fixed_bank(self.windows, self.rows, **arguments)
        naive = select_fixed_bank_naive(self.windows, self.rows, **arguments)
        self.assertEqual(optimized.order, naive.order)
        self.assertEqual(
            optimized.remaining_category_deficit_seconds,
            naive.remaining_category_deficit_seconds,
        )
        self.assertEqual(optimized.stop_reason, naive.stop_reason)
        self.assertEqual(optimized.rejected_by_row_cap, naive.rejected_by_row_cap)
        self.assertTrue(optimized.independent_recomputation_pass)

    def test_optimized_matches_naive_with_unequal_occurrence_scores(self):
        windows, rows = varied_grid()
        arguments = dict(
            density=3,
            residency_bytes=12 * GIB,
            target_tps=100.0,
            maximum_identities=7,
        )
        optimized = select_fixed_bank(windows, rows, **arguments)
        naive = select_fixed_bank_naive(windows, rows, **arguments)
        self.assertEqual(optimized.order, naive.order)
        self.assertEqual(optimized.stop_reason, naive.stop_reason)
        self.assertEqual(
            optimized.remaining_category_deficit_seconds,
            naive.remaining_category_deficit_seconds,
        )
        self.assertEqual(optimized.rejected_by_row_cap, naive.rejected_by_row_cap)

    def test_exact_tie_uses_lowest_layer_then_expert(self):
        result = select_fixed_bank(
            self.windows,
            self.rows,
            density=3,
            residency_bytes=12 * GIB,
            target_tps=100.0,
            maximum_identities=2,
        )
        self.assertEqual(result.order, ((1, 0), (1, 1)))

    def test_optimized_matches_naive_at_guard_fit_discontinuity(self):
        union_occurrences = len(ROUTED_LAYERS) * 8
        all_source_total = (
            FIXED_ALLOCATED_BYTES + union_occurrences * SOURCE_SCHEMA2_STRIDE
        )
        residency = all_source_total - SCHEMA2_SAVING
        arguments = dict(
            density=3,
            residency_bytes=residency,
            target_tps=100.0,
            maximum_identities=2,
        )
        optimized = select_fixed_bank(self.windows, self.rows, **arguments)
        naive = select_fixed_bank_naive(self.windows, self.rows, **arguments)
        self.assertEqual(optimized.order, naive.order)
        self.assertEqual(optimized.order, ((1, 0),))
        self.assertEqual(
            optimized.remaining_category_deficit_seconds,
            naive.remaining_category_deficit_seconds,
        )

    def test_row_cap_rejects_d_plus_one_and_records_once(self):
        for density in (3, 4, 5, 6):
            with self.subTest(density=density):
                result = select_fixed_bank(
                    self.windows,
                    self.rows,
                    density=density,
                    residency_bytes=12 * GIB,
                    target_tps=100.0,
                    maximum_identities=density + 2,
                )
                histogram = result.coverage["row_density_histogram"]
                self.assertLessEqual(
                    max(int(hit) for hit, count in histogram.items() if count), density
                )
                rejected = [
                    (item["layer"], item["expert"])
                    for item in result.rejected_by_row_cap
                ]
                self.assertEqual(len(rejected), len(set(rejected)))
                self.assertIn((1, density), rejected)

    def test_density_eight_is_diagnostic_and_cannot_overfill_eight_row(self):
        result = select_fixed_bank(
            self.windows,
            self.rows,
            density=8,
            residency_bytes=12 * GIB,
            target_tps=100.0,
            maximum_identities=9,
        )
        value = selector_result_dict(result)
        self.assertTrue(value["diagnostic_only_density8"])
        self.assertLessEqual(
            max(
                int(hit)
                for hit, count in result.coverage["row_density_histogram"].items()
                if count
            ),
            8,
        )

    def test_fixed_identity_is_charged_in_every_window(self):
        selected = frozenset({(1, 0)})
        ledgers = [
            window_storage_ledger(
                window,
                residency_bytes=12 * GIB,
                selected=selected,
                rows=self.rows,
            )
            for window in self.windows
        ]
        self.assertEqual(
            [ledger["k4_identity_layer_occurrences"] for ledger in ledgers],
            [1, 1, 1, 1],
        )

    def test_order_hash_binds_complete_canonical_list(self):
        order = ((1, 2), (1, 3))
        expected = hashlib.sha256(
            b'[{"expert":2,"layer":1},{"expert":3,"layer":1}]\n'
        ).hexdigest()
        self.assertEqual(selection_order_sha256(order), expected)


class DispositionTests(unittest.TestCase):
    def test_density_summary_reports_earliest_crossing_for_each_strict_gate(self):
        relaxed = fake_relaxed()
        relaxed[(4, 12 * GIB)]["metrics"]["fractional_relaxed"] = fake_metrics(
            overall=1.2, category=0.9, p10=0.8
        )
        relaxed[(5, 12 * GIB)]["metrics"]["fractional_relaxed"] = fake_metrics(
            overall=1.2, category=1.2, p10=0.9
        )
        summary = density_survival_summary(relaxed)
        self.assertEqual(
            summary["earliest_density_by_gate"],
            {
                "strict_overall_above_one": 4,
                "strict_every_category_above_one": 5,
                "strict_fourth_lowest_p10_above_one": 6,
                "strict_all": 6,
            },
        )

    def test_absolute_rejection_precedes_tail_and_banks_at_equality(self):
        result = disposition(
            fake_relaxed(strongest_overall=1.0, strongest_p10=0.5),
            [fake_bank()],
            authority_complete=True,
            gate8_pass=True,
        )
        self.assertEqual(result["precedence_gate"], 1)

    def test_tail_rejection_precedes_density(self):
        result = disposition(
            fake_relaxed(strongest_p10=1.0, d6=0.5),
            [fake_bank()],
            authority_complete=True,
            gate8_pass=True,
        )
        self.assertEqual(result["precedence_gate"], 2)

    def test_d8_survival_cannot_hide_d6_failure(self):
        result = disposition(
            fake_relaxed(d6=1.0),
            [fake_bank()],
            authority_complete=True,
            gate8_pass=True,
        )
        self.assertEqual(result["precedence_gate"], 3)

    def test_no_exact_125_bank_yields_conditional_survivor(self):
        result = disposition(
            fake_relaxed(),
            [fake_bank(target=1.10)],
            authority_complete=True,
            gate8_pass=True,
        )
        self.assertEqual(result["precedence_gate"], 4)
        self.assertIsNone(result["work_order"])

    def test_density_three_prerequisite_authorizes_only_staged_validation(self):
        result = disposition(
            fake_relaxed(),
            [fake_bank(density=3)],
            authority_complete=True,
            gate8_pass=True,
        )
        self.assertEqual(result["precedence_gate"], 5)
        self.assertFalse(result["work_order"]["physical_authorization"])

    def test_density_five_keeps_four_row_and_separate_five_row(self):
        result = disposition(
            fake_relaxed(),
            [fake_bank(density=5)],
            authority_complete=True,
            gate8_pass=True,
        )
        self.assertEqual(result["precedence_gate"], 6)
        self.assertTrue(result["work_order"]["five_of_eight_row_required_after_four_row"])

    def test_density_six_requires_all_28_subsets(self):
        result = disposition(
            fake_relaxed(),
            [fake_bank(density=6)],
            authority_complete=True,
            gate8_pass=True,
        )
        self.assertEqual(result["precedence_gate"], 7)
        self.assertEqual(result["work_order"]["six_of_eight_subsets_required"], 28)

    def test_more_than_eight_gib_interposes_pressure_requalification(self):
        result = disposition(
            fake_relaxed(),
            [fake_bank(density=3, residency_bytes=12 * GIB)],
            authority_complete=True,
            gate8_pass=True,
        )
        self.assertTrue(result["work_order"]["pressure_requalification_required"])
        self.assertFalse(result["work_order"]["physical_authorization"])

    def test_authority_or_gate8_failure_blocks_work(self):
        for authority, gate8 in ((False, True), (True, False)):
            with self.subTest(authority=authority, gate8=gate8):
                result = disposition(
                    fake_relaxed(),
                    [fake_bank()],
                    authority_complete=authority,
                    gate8_pass=gate8,
                )
                self.assertEqual(result["precedence_gate"], 4)


class AuthorityAdapterTests(unittest.TestCase):
    def test_wrong_manifest_hash_rejects_before_trusting_windows(self):
        with self.assertRaisesRegex(ValueError, "PW-0328 authority"):
            normalize_pw0328_authority(
                {"manifest_sha256": "0" * 64, "builder_commit": "bad", "windows": []}
            )

    def test_observable_A_substitution_rejects_against_authorized_rows(self):
        authority = {
            "manifest_sha256":
                "36e4f10b6f807f766c87ee7078f5f18ea8fc339dd12e4dbc24f1f4ac6e824403",
            "builder_commit": "26d2ea31852c0d63bd022df6d571fd722137c39f",
            "artifact_count": 24,
            "artifacts": [{}] * 24,
            "categories": list(CATEGORIES),
            "q1_events": [{}] * 232,
            "control": {
                "windows": 32,
                "sum_A": 232,
                "sum_observable_A": 231,
                "sum_U": 142.71808510638297,
            },
            "builder_gate8": {"pass": True},
            "windows": [
                {
                    "corpus_index": 0,
                    "category": "ordinary",
                    "transaction_index": 0,
                    "A": 7,
                    "observable_A": 7,
                    "verifier_authorized_token_ids": list(range(8)),
                    "authorized_q1_rows": [{}] * 8,
                }
            ],
        }
        with self.assertRaisesRegex(ValueError, "full verifier-authorized A"):
            normalize_pw0328_authority(authority)


if __name__ == "__main__":
    unittest.main()
