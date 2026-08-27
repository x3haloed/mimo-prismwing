import copy
from fractions import Fraction
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from tools.analyze_pw0332_top7_cache_oracle import (
    ABSOLUTE_FLOOR_RATIO,
    BANDWIDTH_EXACT,
    BANDWIDTH_EXACT_FLOAT,
    BANDWIDTH_FAVORABLE,
    BANDWIDTH_FAVORABLE_FLOAT,
    BLOCK_CODES,
    CATEGORIES,
    CONTRACT_GIT_BLOB,
    CONTRACT_SHA256,
    Demand,
    EXPECTED_CATEGORY_A,
    FIXED_BF16_BYTES,
    FIXED_BF16_OBJECT_COUNT,
    FIXED_F32_BYTES,
    FIXED_F32_OBJECT_COUNT,
    FIXED_FP8_CODE_BYTES,
    FIXED_FP8_OBJECT_COUNT,
    FIXED_LOGICAL_BYTES,
    FIXED_NON_FP8_BYTES,
    OBSERVED_ESCAPE_BYTES,
    OBSERVED_MINIMUM_RATIO,
    PW0328_CAPTURE_COMMIT,
    PW0328_MANIFEST_SHA256,
    PW0328_Q1_DEMAND_STREAM_SHA256,
    RESIDENCY_BYTES,
    ROUTED_LAYERS,
    SCENARIOS,
    SOURCE_EXPERT_BYTES,
    SOURCE_EXPERT_F32_SCALE_BYTES,
    SOURCE_EXPERT_FP8_CODE_BYTES,
    TOP_K,
    ZERO_ESCAPE_BYTES,
    analyze,
    analyze_scenario,
    demand_stream_sha256,
    disposition,
    encoded_top7_bytes,
    exhaustive_joint_schedule_misses,
    exhaustive_optimal_misses,
    fixed_pinned_reference_misses,
    indexed_belady,
    nearest_rank_p10,
    normalize_corpus_authority,
    reject_nonfinite_evidence,
    replay_belady,
    scenario_layout,
    serialize_tps,
    sha256_file,
    strict_gates,
    validate_codec_floor,
    validate_layouts,
    validate_pw0212_payload,
    validate_pw0324_payload,
    validate_report_schema,
    validate_scenario_dominance,
    validate_storage_authority,
    validate_tiny_oracles,
)


def fraction_record(value):
    value = Fraction(value)
    return {"numerator": value.numerator, "denominator": value.denominator}


def gate_result(value=Fraction(2), *, p10=None, fourth=None):
    p10 = value if p10 is None else Fraction(p10)
    fourth = value if fourth is None else Fraction(fourth)
    aggregate = {
        "storage_tps_fraction": fraction_record(value),
        "nearest_rank_p10_token_storage_tps_fraction": fraction_record(p10),
        "fourth_lowest_window_storage_tps_fraction": fraction_record(fourth),
    }
    return {
        "overall": aggregate,
        "categories": {
            category: {
                "aggregate": {
                    "storage_tps_fraction": fraction_record(value),
                    "nearest_rank_p10_token_storage_tps_fraction": fraction_record(p10),
                }
            }
            for category in CATEGORIES
        },
    }


def tiny_demand(experts, *, layer=1, event=0):
    return Demand(
        category="ordinary",
        corpus_index=0,
        transaction_index=0,
        event_index=event,
        position=event,
        layer=layer,
        identities=tuple((layer, expert) for expert in experts),
    )


def frozen_shape_authority():
    """Small-content but exact-shape stand-in for the shared q1 authority."""

    a_rows = {
        "ordinary": [8, 8, 8, 8, 8, 8, 1, 1],
        "code": [8, 8, 8, 8, 8, 8, 8, 2],
        "multilingual": [8, 8, 8, 8, 8, 8, 8, 4],
        "rare_route": [8] * 8,
    }
    windows = []
    events = []
    for category in CATEGORIES:
        for transaction_index, a in enumerate(a_rows[category]):
            corpus_index = len(windows)
            tokens = [1000 + corpus_index * 8 + position for position in range(a)]
            q1_rows = []
            for position in range(a):
                layer_rows = []
                for layer in ROUTED_LAYERS:
                    experts = sorted({(corpus_index * 13 + position * 7 + layer + offset) % 256 for offset in range(TOP_K)})
                    layer_rows.append({"layer": layer, "experts": experts})
                q1_rows.append({"position": position, "layers": layer_rows})
                events.append(
                    {
                        "event_index": len(events),
                        "category": category,
                        "corpus_index": corpus_index,
                        "transaction_index": transaction_index,
                        "authorized_token_id": tokens[position],
                        "position": position,
                        "layers": layer_rows,
                    }
                )
            windows.append(
                {
                    "corpus_index": corpus_index,
                    "category": category,
                    "transaction_index": transaction_index,
                    "A": a,
                    "U": 1.0 + corpus_index / 100.0,
                    "verifier_authorized_token_ids": tokens,
                    "authorized_q1_rows": q1_rows,
                }
            )
    # The production normalizer freezes the canonical real-trace hash.  Tests
    # that exercise mutations patch only the otherwise-valid trace hash call.
    return {
        "manifest_sha256": PW0328_MANIFEST_SHA256,
        "builder_commit": PW0328_CAPTURE_COMMIT,
        "categories": list(CATEGORIES),
        "artifact_count": 24,
        "windows": windows,
        "q1_events": events,
        "control": {"sum_A": 232, "sum_observable_A": 231},
    }


class CodecAndByteLedgerTests(unittest.TestCase):
    def test_contract_blob_and_sha_are_frozen(self):
        repo = Path(__file__).resolve().parents[1]
        self.assertEqual(
            sha256_file(repo / "experiments/PW-0332-exact-top7-token-cache-oracle.md"),
            CONTRACT_SHA256,
        )
        import subprocess

        blob = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD:experiments/PW-0332-exact-top7-token-cache-oracle.md"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        self.assertEqual(blob, CONTRACT_GIT_BLOB)

    def test_escape_formula_and_exact_floor(self):
        self.assertEqual(encoded_top7_bytes(0), ZERO_ESCAPE_BYTES)
        self.assertEqual(encoded_top7_bytes(341), OBSERVED_ESCAPE_BYTES)
        self.assertEqual(Fraction(ZERO_ESCAPE_BYTES, BLOCK_CODES), ABSOLUTE_FLOOR_RATIO)
        self.assertEqual(Fraction(OBSERVED_ESCAPE_BYTES, BLOCK_CODES), OBSERVED_MINIMUM_RATIO)
        self.assertTrue(all(encoded_top7_bytes(e) >= ZERO_ESCAPE_BYTES for e in range(BLOCK_CODES + 1)))
        floor = validate_codec_floor()
        self.assertFalse(floor["ratio_below_floor_possible"])
        self.assertEqual(floor["observed_witness_escapes"], 341)

    def test_escape_formula_rejects_invalid_counts(self):
        for value in (-1, BLOCK_CODES + 1, 1.0, True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                encoded_top7_bytes(value)

    def test_scenario_ledgers_and_capacities_are_exact(self):
        layouts = [scenario_layout(scenario) for scenario in SCENARIOS]
        validate_layouts(layouts)
        self.assertEqual([row["expert_capacity"] for row in layouts], [204, 230, 250])
        self.assertEqual(layouts[2]["encoded_fixed_bytes"], 7_359_815_296)
        self.assertEqual(layouts[2]["encoded_expert_bytes_fraction"], fraction_record(Fraction(44_063_235, 2)))
        for row in layouts:
            self.assertEqual(row["fixed_bf16_logical_bytes"], FIXED_BF16_BYTES)
            self.assertEqual(row["fixed_f32_logical_bytes"], FIXED_F32_BYTES)
            self.assertEqual(row["source_expert_fp8_code_bytes"], SOURCE_EXPERT_FP8_CODE_BYTES)
            self.assertEqual(row["source_expert_f32_scale_bytes"], SOURCE_EXPERT_F32_SCALE_BYTES)
            self.assertTrue(row["expert_ratio_applies_favorably_to_complete_record_including_scales"])
            self.assertEqual(row["residency_bytes"], RESIDENCY_BYTES)

    def test_fixed_and_expert_components_close_without_embedding_rows(self):
        self.assertEqual(FIXED_FP8_CODE_BYTES + FIXED_NON_FP8_BYTES, FIXED_LOGICAL_BYTES)
        self.assertEqual(FIXED_BF16_BYTES + FIXED_F32_BYTES, FIXED_NON_FP8_BYTES)
        self.assertEqual(SOURCE_EXPERT_FP8_CODE_BYTES + SOURCE_EXPERT_F32_SCALE_BYTES, SOURCE_EXPERT_BYTES)
        self.assertEqual(
            (FIXED_FP8_OBJECT_COUNT, FIXED_BF16_OBJECT_COUNT, FIXED_F32_OBJECT_COUNT),
            (51, 185, 145),
        )

    def test_joint_residency_is_not_double_spent(self):
        layouts = [scenario_layout(scenario) for scenario in SCENARIOS]
        for row in layouts:
            used = Fraction(row["encoded_fixed_bytes"]) + row["expert_capacity"] * Fraction(
                row["encoded_expert_bytes_fraction"]["numerator"],
                row["encoded_expert_bytes_fraction"]["denominator"],
            )
            self.assertLessEqual(used, RESIDENCY_BYTES)
            self.assertLess(
                Fraction(RESIDENCY_BYTES) - used,
                Fraction(
                    row["encoded_expert_bytes_fraction"]["numerator"],
                    row["encoded_expert_bytes_fraction"]["denominator"],
                ),
            )

    def test_capacity_below_eight_rejects(self):
        with self.assertRaisesRegex(ValueError, "below one routed set"):
            scenario_layout(SCENARIOS[0], residency_bytes=FIXED_LOGICAL_BYTES + 7 * SOURCE_EXPERT_BYTES)

    def test_exact_and_rounded_bandwidth_are_distinct_and_favorable(self):
        self.assertEqual(float(BANDWIDTH_EXACT), BANDWIDTH_EXACT_FLOAT)
        self.assertEqual(float(BANDWIDTH_FAVORABLE), BANDWIDTH_FAVORABLE_FLOAT)
        self.assertGreater(BANDWIDTH_FAVORABLE, BANDWIDTH_EXACT)


class BeladyOracleTests(unittest.TestCase):
    def test_indexed_matches_exhaustive_tiny_dp(self):
        cases = [
            ([[(1, 0)], [(1, 1)], [(1, 0)]], 1),
            ([[(1, 0), (1, 1)], [(2, 0), (2, 1)], [(1, 0), (1, 2)]], 2),
            ([[(1, 0)], [(1, 1)], [(1, 2)], [(1, 0)], [(1, 2)]], 2),
            ([[(1, 0), (1, 1)], [(1, 1), (1, 2)], [(1, 0), (1, 2)]], 3),
        ]
        for trace, capacity in cases:
            with self.subTest(trace=trace, capacity=capacity):
                result = indexed_belady(trace, capacity)
                self.assertEqual(result["miss_count"], exhaustive_optimal_misses(trace, capacity))
                self.assertTrue(replay_belady(trace, capacity, result)["pass"])

    def test_batch_set_iteration_order_is_irrelevant(self):
        trace = [[(1, 2), (1, 0), (1, 1)], [(2, 2), (2, 1), (2, 0)], [(1, 0), (1, 3), (1, 2)]]
        reversed_trace = [list(reversed(row)) for row in trace]
        left = indexed_belady(trace, 3)
        right = indexed_belady(reversed_trace, 3)
        self.assertEqual(left, right)

    def test_current_set_is_protected_until_whole_demand_is_served(self):
        trace = [[(1, 0), (1, 1)], [(2, 0), (2, 1)], [(1, 0), (1, 2)]]
        result = indexed_belady(trace, 2)
        for row in result["demand_ledger"]:
            self.assertTrue(set(row["demand"]).isdisjoint(row["evictions"]))

    def test_infinity_and_reverse_canonical_eviction_tie(self):
        trace = [[(1, 0), (1, 1)], [(2, 0), (2, 1)]]
        result = indexed_belady(trace, 2)
        self.assertEqual(result["demand_ledger"][1]["evictions"], [(1, 1), (1, 0)])

    def test_free_initial_fill_uses_earliest_then_canonical(self):
        trace = [[(1, 2)], [(1, 0), (1, 1)], [(1, 3)]]
        result = indexed_belady(trace, 2)
        self.assertEqual(result["free_initial_identities"], [(1, 2), (1, 0)])

    def test_fit_all_and_zero_miss(self):
        trace = [[(1, 0), (1, 1)], [(1, 1), (1, 2)], [(1, 2), (1, 0)]]
        result = indexed_belady(trace, 3)
        self.assertEqual(result["miss_count"], 0)
        self.assertEqual(result["eviction_count"], 0)

    def test_category_reset_grants_each_first_use_for_free(self):
        trace = [[(1, 0), (1, 1)], [(2, 0), (2, 1)]]
        self.assertEqual(indexed_belady(trace, 2)["demand_ledger"][0]["misses"], [])
        self.assertEqual(indexed_belady(trace, 2)["demand_ledger"][0]["misses"], [])

    def test_capacity_below_set_and_duplicate_demand_reject(self):
        with self.assertRaisesRegex(ValueError, "below simultaneous"):
            indexed_belady([[(1, 0), (1, 1)]], 1)
        with self.assertRaisesRegex(ValueError, "distinct"):
            indexed_belady([[(1, 0), (1, 0)]], 2)

    def test_independent_replay_detects_tampering(self):
        trace = [[(1, 0), (1, 1)], [(2, 0), (2, 1)], [(1, 0), (1, 2)]]
        result = indexed_belady(trace, 2)
        result["demand_ledger"][1]["misses"] = []
        with self.assertRaisesRegex(ValueError, "replay"):
            replay_belady(trace, 2, result)

    def test_fixed_pinned_representation_matches_joint_tiny_optimum(self):
        experts = [["a"], ["b"], ["a"], ["c"]]
        pinned = fixed_pinned_reference_misses(experts, fixed_objects=1, capacity=3)
        joint = exhaustive_joint_schedule_misses(experts, fixed_objects=1, capacity=3)
        self.assertLessEqual(pinned, joint)
        self.assertEqual(pinned, joint)

    def test_frozen_tiny_implementation_validation(self):
        result = validate_tiny_oracles()
        self.assertTrue(result["pass"])
        self.assertTrue(all(row["pass"] for row in result["fixtures"]))
        self.assertFalse(
            result["fixed_residency_dominance_fixture"][
                "dynamic_fixed_eviction_improves_misses"
            ]
        )

    def test_demand_rejects_duplicate_or_wrong_layer(self):
        with self.assertRaisesRegex(ValueError, "eight distinct"):
            tiny_demand([0] * 8)
        with self.assertRaisesRegex(ValueError, "identity mismatch"):
            Demand("ordinary", 0, 0, 0, 0, 1, tuple((2, expert) for expert in range(8)))


class CorpusBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.authority = frozen_shape_authority()

    def normalize_allowing_fixture_hash(self, authority):
        # Exercise every structural guard while substituting the canonical
        # hash only for this synthetic route content.
        from unittest.mock import patch

        with patch(
            "tools.analyze_pw0332_top7_cache_oracle.demand_stream_sha256",
            return_value=PW0328_Q1_DEMAND_STREAM_SHA256,
        ):
            return normalize_corpus_authority(authority)

    def test_exact_shape_builds_32_windows_and_232_q1_events(self):
        windows, traces = self.normalize_allowing_fixture_hash(self.authority)
        self.assertEqual(len(windows), 32)
        self.assertEqual(sum(len(rows) for rows in traces.values()), 232 * 47)
        self.assertEqual(
            {category: len(traces[category]) // 47 for category in CATEGORIES},
            EXPECTED_CATEGORY_A,
        )

    def test_wrong_hash_or_category_order_rejects(self):
        authority = copy.deepcopy(self.authority)
        authority["manifest_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "manifest"):
            self.normalize_allowing_fixture_hash(authority)
        authority = copy.deepcopy(self.authority)
        authority["categories"] = list(reversed(CATEGORIES))
        with self.assertRaisesRegex(ValueError, "category order"):
            self.normalize_allowing_fixture_hash(authority)

    def test_clipped_a_and_mismatch_suffix_reject(self):
        authority = copy.deepcopy(self.authority)
        authority["windows"][0]["A"] -= 1
        with self.assertRaisesRegex(ValueError, "clipped"):
            self.normalize_allowing_fixture_hash(authority)
        authority = copy.deepcopy(self.authority)
        authority["q1_events"][1]["position"] = 2
        with self.assertRaisesRegex(ValueError, "global q1"):
            self.normalize_allowing_fixture_hash(authority)

    def test_event_reordering_and_wrong_authorized_token_reject(self):
        authority = copy.deepcopy(self.authority)
        authority["q1_events"][0], authority["q1_events"][1] = authority["q1_events"][1], authority["q1_events"][0]
        with self.assertRaisesRegex(ValueError, "global q1"):
            self.normalize_allowing_fixture_hash(authority)
        authority = copy.deepcopy(self.authority)
        authority["q1_events"][0]["authorized_token_id"] += 1
        with self.assertRaisesRegex(ValueError, "ordering"):
            self.normalize_allowing_fixture_hash(authority)

    def test_duplicate_route_and_proposal_contamination_reject(self):
        authority = copy.deepcopy(self.authority)
        authority["q1_events"][0]["layers"][0]["experts"][1] = authority["q1_events"][0]["layers"][0]["experts"][0]
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.normalize_allowing_fixture_hash(authority)
        authority = copy.deepcopy(self.authority)
        authority["q1_events"][0]["proposal_layers"] = []
        with self.assertRaisesRegex(ValueError, "proposal-route"):
            self.normalize_allowing_fixture_hash(authority)

    def test_demand_stream_hash_schema_is_stable(self):
        demand = tiny_demand(range(8))
        traces = {category: [] for category in CATEGORIES}
        traces["ordinary"] = [demand]
        expected = [
            {
                "category": "ordinary",
                "corpus_index": 0,
                "event_index": 0,
                "layer": 1,
                "experts": list(range(8)),
            }
        ]
        payload = (json.dumps(expected, sort_keys=True, separators=(",", ":")) + "\n").encode()
        import hashlib

        self.assertEqual(demand_stream_sha256(traces), hashlib.sha256(payload).hexdigest())


class AggregateAndDispositionTests(unittest.TestCase):
    def test_nearest_rank_p10_and_infinity_serialization(self):
        values = [Fraction(value) for value in range(1, 11)]
        self.assertEqual(nearest_rank_p10(values), 1)
        values32 = [Fraction(value) for value in range(1, 33)]
        self.assertEqual(nearest_rank_p10(values32), 4)
        self.assertEqual(serialize_tps(None), "infinity")
        encoded = json.dumps({"tps": serialize_tps(None)}, allow_nan=False)
        self.assertNotIn("Infinity", encoded)

    def test_strict_gate_rejects_equality_and_any_slice(self):
        self.assertFalse(strict_gates(gate_result(Fraction(1)))["all_strict_gates_pass"])
        result = gate_result(Fraction(2), p10=Fraction(1))
        self.assertFalse(strict_gates(result)["all_strict_gates_pass"])
        result = gate_result(Fraction(2))
        result["categories"]["rare_route"]["aggregate"]["storage_tps_fraction"] = fraction_record(1)
        self.assertFalse(strict_gates(result)["all_strict_gates_pass"])

    def test_disposition_precedence(self):
        passing = {scenario.name: gate_result(2) for scenario in SCENARIOS}
        self.assertEqual(disposition(passing)["decision"], "retain_analytical_survivor")
        observed = copy.deepcopy(passing)
        observed["observed_expert_only"] = gate_result(1)
        self.assertEqual(
            disposition(observed)["decision"],
            "retain_absolute_floor_survivor_reject_observed_ratio_diagnostic",
        )
        hard = copy.deepcopy(passing)
        hard["absolute_floor_all_fp8"] = gate_result(1)
        decision = disposition(hard)
        self.assertEqual(decision["decision"], "reject_exact_top7_token_cache_oracle")
        self.assertFalse(decision["decoder_authorized"])

    def test_nonfinite_and_schema_drift_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            reject_nonfinite_evidence({"value": math.inf})
        report = {
            "schema_version": 1,
            "experiment_id": "PW-0332",
            "status": "analytical_token_cache_oracle_complete",
            "decision": {
                "runtime_default_changed": False,
                "decoder_authorized": False,
            },
            "commit": "0" * 40,
            "authority": {"authority_complete": True},
            "codec_floor": {},
            "constants": {},
            "measurement_context": {
                "companion_hardware": "excluded",
                "batch_size": 1,
                "concurrency": 1,
                "accepted_tokens_experiment": 0,
            },
            "scenarios": [{"scenario": scenario.name} for scenario in SCENARIOS],
            "scenario_dominance": {},
            "safety_snapshots": [],
            "gate8_analyzer_pass": True,
            "accepted_tokens": 0,
            "A": 0,
            "U": 0,
            "performance_claim": None,
        }
        validate_report_schema(report)
        changed = copy.deepcopy(report)
        changed["unknown"] = True
        with self.assertRaisesRegex(ValueError, "schema drift"):
            validate_report_schema(changed)
        mutations = (
            ("status", "wrong"),
            ("gate8_analyzer_pass", False),
            ("authority", {"authority_complete": False}),
            ("decision", {"runtime_default_changed": True, "decoder_authorized": False}),
            ("decision", {"runtime_default_changed": False, "decoder_authorized": True}),
            ("measurement_context", {**report["measurement_context"], "companion_hardware": "allowed"}),
            ("measurement_context", {**report["measurement_context"], "batch_size": 2}),
            ("measurement_context", {**report["measurement_context"], "concurrency": 2}),
            ("measurement_context", {**report["measurement_context"], "accepted_tokens_experiment": 1}),
        )
        for key, value in mutations:
            changed = copy.deepcopy(report)
            changed[key] = value
            with self.subTest(key=key, value=value), self.assertRaisesRegex(ValueError, "semantic schema"):
                validate_report_schema(changed)


class AuthorityPayloadTests(unittest.TestCase):
    def test_canonical_pw0324_payload_semantics(self):
        path = Path("/Users/chad/Models/mimo-prismwing/evidence/PW-0324/analysis-002/analysis.json")
        result = validate_pw0324_payload(json.loads(path.read_text()))
        self.assertEqual(result["quantization_blocks"], 480)
        self.assertFalse(result["sample_is_routed_full_model_codec_census"])

    def test_canonical_pw0212_payload_semantics(self):
        path = Path("/Volumes/Elements/mimo-prismwing/evidence/PW-0212/corrected-route-prefetch-oracle-001.json")
        result = validate_pw0212_payload(json.loads(path.read_text()))
        self.assertFalse(result["imported_as_current_percentage"])

    def test_storage_dtype_census_and_bandwidth_validation(self):
        objects = []
        for dtype, count, total in (
            ("F8_E4M3", FIXED_FP8_OBJECT_COUNT, FIXED_FP8_CODE_BYTES),
            ("BF16", FIXED_BF16_OBJECT_COUNT, FIXED_BF16_BYTES),
            ("F32", FIXED_F32_OBJECT_COUNT, FIXED_F32_BYTES),
        ):
            # One object carries the total; the remaining positive fixture
            # objects carry zero solely because this pure validator sums its
            # already-authenticated logical-byte ledger.
            objects.extend(
                {"dtype": dtype, "logical_bytes": total if index == 0 else 0}
                for index in range(count)
            )
        authority = {
            "identities": {
                "revision": "63651580ca774f8504f676040460aed3e1244ac1",
                "checkpoint_verification_sha256": "9ddc8a99755f04ae2ea3c2484f6dd022d3f3a681b5a72c915ee4de833dbb0d03",
                "tensor_index_sha256": "f2e1774c9acf9a62338b68c144e6fc7a66495e59f2e64b3078c1b7ef5a196816",
                "pw0207_offline_sha256": "1dedbef7c79aa23835d194f52760a1f2c65dcca1481bd6df2d5602615c3fdad6",
                "pw0136_raw_sha256": "e6ab84cada19c6036ee7b83f318c3920631141b9ea5e882cc88eb9784d0b5a56",
                "pw0136_analysis_sha256": "7ebf2cde5c4a3f4931d2d705993f822e38af13ea66bc3efc91410296b14e2aab",
            },
            "fixed": {
                "object_count": 381,
                "logical_source_bytes": FIXED_LOGICAL_BYTES,
                "fp8_code_bytes": FIXED_FP8_CODE_BYTES,
                "non_fp8_bytes": FIXED_NON_FP8_BYTES,
                "largest_object_bytes": 1_249_902_592,
                "objects": objects,
            },
            "bandwidth": {
                "raw_exact_bytes_per_second": BANDWIDTH_EXACT_FLOAT,
                "candidate_favorable_bytes_per_second": BANDWIDTH_FAVORABLE_FLOAT,
            },
        }
        result = validate_storage_authority(authority)
        self.assertEqual(result["trace_specific_embedding_bytes_excluded"], 57_344)
        authority["fixed"]["objects"].append({"dtype": "F32", "logical_bytes": 8192})
        with self.assertRaisesRegex(ValueError, "object ledger"):
            validate_storage_authority(authority)


class ExecutionFailClosedTests(unittest.TestCase):
    def test_existing_output_rejects_before_any_authority_work(self):
        with TemporaryDirectory() as temporary:
            output = Path(temporary) / "existing"
            output.mkdir()
            with self.assertRaises(FileExistsError):
                analyze(
                    repo=Path(temporary),
                    checkpoint_root=Path(temporary),
                    output=output,
                    commit="0" * 40,
                )

    def test_dirty_git_rejection_propagates(self):
        from unittest.mock import patch

        with patch(
            "tools.analyze_pw0332_top7_cache_oracle.verify_clean_commit",
            side_effect=ValueError("dirty worktree"),
        ):
            with self.assertRaisesRegex(ValueError, "dirty worktree"):
                from tools.analyze_pw0332_top7_cache_oracle import verify_execution_commit

                verify_execution_commit(Path("/tmp"), "0" * 40)

    def test_unsafe_pressure_stops_before_output_write(self):
        from unittest.mock import patch
        from tools.host_safety import HostSafetyViolation

        class UnsafeMonitor:
            def checkpoint(self, _phase):
                raise HostSafetyViolation("unsafe pressure")

        with TemporaryDirectory() as temporary:
            output = Path(temporary) / "new-output"
            patches = (
                patch("tools.analyze_pw0332_top7_cache_oracle.verify_execution_commit"),
                patch(
                    "tools.analyze_pw0332_top7_cache_oracle.verify_target_hardware",
                    return_value={"processor": "Apple M1"},
                ),
                patch(
                    "tools.analyze_pw0332_top7_cache_oracle.HostSafetyMonitor",
                    return_value=UnsafeMonitor(),
                ),
                patch(
                    "tools.analyze_pw0332_top7_cache_oracle.authenticate_pw0328_corpus",
                    return_value={},
                ),
                patch(
                    "tools.analyze_pw0332_top7_cache_oracle.authenticate_prismwing_storage",
                    return_value={},
                ),
                patch(
                    "tools.analyze_pw0332_top7_cache_oracle.validate_storage_authority",
                    return_value={},
                ),
                patch(
                    "tools.analyze_pw0332_top7_cache_oracle.authenticate_pw0324",
                    return_value={},
                ),
                patch(
                    "tools.analyze_pw0332_top7_cache_oracle.authenticate_pw0212",
                    return_value={},
                ),
            )
            with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], patches[6], patches[7]:
                with self.assertRaisesRegex(HostSafetyViolation, "unsafe pressure"):
                    analyze(
                        repo=Path(temporary),
                        checkpoint_root=Path(temporary),
                        output=output,
                        commit="0" * 40,
                    )
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
