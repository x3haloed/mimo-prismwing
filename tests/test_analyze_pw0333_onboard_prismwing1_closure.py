import copy
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
import unittest
from unittest import mock

import tools.analyze_pw0333_onboard_prismwing1_closure as closure


def fraction_record(value):
    value = Fraction(value)
    return {"numerator": value.numerator, "denominator": value.denominator}


def strict_gate_row(value=False):
    return {
        "overall_aggregate_strictly_above_one": value,
        "required_category_aggregates_strictly_above_one": {
            category: value for category in closure.CATEGORIES
        },
        "corpus_token_p10_strictly_above_one": value,
        "category_token_p10_strictly_above_one": {
            category: value for category in closure.CATEGORIES
        },
        "fourth_lowest_window_strictly_above_one": value,
        "all_strict_gates_pass": value,
    }


def valid_conditions():
    return {name: True for name in closure.REQUIRED_CLOSURE_CONDITIONS}


def valid_branches():
    return {
        name: {
            "disposition": disposition,
            "decisive": True,
            "within_current_authenticated_portfolio": True,
        }
        for name, disposition in closure.REQUIRED_REOPENED_BRANCH_DISPOSITIONS.items()
    }


def safety_snapshot(phase, *, release=False, free=50):
    return {
        "phase": phase,
        "release_boundary": release,
        "released_resources": ["test fixture"] if release else [],
        "system_memory_free_percent": free,
        "swap_used_bytes": 0,
        "swap_growth_bytes": 0,
        "throttled_pages": 0,
        "new_throttled_pages": 0,
        "process_resident_bytes": 1,
        "process_physical_footprint_bytes": 1,
        "process_peak_resident_bytes": 1,
        "allocator_pressure_relief_bytes": 0,
        "protected_service_pids": {"ChatGPT": [1]},
        "process_disk_bytes_read": 0,
        "process_disk_bytes_written": 0,
    }


def safety_evidence():
    return [
        safety_snapshot("process_start"),
        safety_snapshot("fixture_released", release=True),
        safety_snapshot("final_service_health"),
    ]


def gate8_summary():
    return closure.validate_gate8(safety_evidence(), label="test fixture")


def codec_replay_fixture():
    return {
        "file": str(closure.PW0324_PATH),
        "sha256": closure.PW0324_SHA256,
        "local_exact_codec_replication_authenticated": True,
        "checkpoint_index_sha256": closure.CHECKPOINT_INDEX_SHA256,
        "quantization_blocks": 480,
        "observed_minimum_top7_ratio": 0.8856201171875,
        "sample_is_routed_full_model_codec_census": False,
        "limitation": "fixture is not a routed full-model codec census",
        "local_exact_codec_replay_equal_to_canonical": True,
    }


def pw0330_route_fixture():
    pairs = [(layer, expert) for layer in range(1, 48) for expert in range(256)][
        : closure.PW0330_IDENTITIES
    ]
    identities = [{"layer": layer, "expert": expert} for layer, expert in pairs]
    per_layer = []
    for layer in range(1, 48):
        experts = [expert for row_layer, expert in pairs if row_layer == layer]
        per_layer.append(
            {"layer": layer, "unique_experts": len(experts), "experts": experts}
        )
    identity_sha = hashlib.sha256(closure.canonical_json(identities)).hexdigest()
    selected = {
        "A": 4,
        "N_A": len(identities),
        "identities": identities,
        "identity_sha256": identity_sha,
        "per_layer": per_layer,
        "unique_source_expert_bytes": len(identities) * closure.SOURCE_EXPERT_BYTES,
    }
    table = [{"A": accepted} for accepted in range(1, 9)]
    table[3] = {
        "A": 4,
        "N_A": len(identities),
        "identity_sha256": identity_sha,
        "per_layer_unique_counts": [row["unique_experts"] for row in per_layer],
        "unique_source_expert_bytes": len(identities) * closure.SOURCE_EXPERT_BYTES,
    }
    return {
        "head_results": [
            {"head_index": index, "match": index < 3} for index in range(4)
        ],
        "selected_prefix_route": selected,
        "route_prefix_table": table,
    }


def synthetic_report():
    repo = Path(__file__).resolve().parents[1]
    model = json.loads((repo / "spec/throughput-model.json").read_text())
    measured = closure.measured_lower_milestones(model)
    gate = gate8_summary()
    pw0324 = {
        "portfolio_mechanisms": len(closure.PRIOR_PORTFOLIO_KEYS),
        "two_tps_closure_survivors": 0,
        "authenticated_proposer_family_states": dict(
            closure.PW0324_PROPOSER_FAMILY_STATES
        ),
        "unreconciled_proposer_family_survivors": 0,
        "portfolio": {
            name: {
                "record": f"experiments/{name}.md",
                "record_sha256": "0" * 64,
                "state": "rejected_fixture",
                "survives_two_tps_closure": False,
            }
            for name in closure.PRIOR_PORTFOLIO_KEYS
        },
        "historical_portfolio_hashes_preserved_not_rehashed_against_updated_records": True,
        "gate8": gate,
    }
    pw0328 = {
        "builder_gate8": gate,
        "generation_gate8": {"count": 4, "all_pass": True, "summaries": [gate] * 4},
        "prefill_gate8": {"count": 4, "all_pass": True, "summaries": [gate] * 4},
    }
    pw0329 = {
        "precedence_gate": 2,
        "work_order": None,
        "ceiling": {
            "strict_fourth_lowest_window_above_one": False,
            "fourth_lowest_window_storage_tps": closure.PW0329_P10_TPS,
        },
        "gate8": gate,
    }
    pw0330 = {
        "decision": "conditional_hard_storage_rejection",
        "direct_q32_first_chunk_parity": "unproven_outside_evidence_backed_survivors",
        "favorable_storage_tps_ceiling": closure.PW0330_TPS,
        "gate8": gate,
    }
    pw0331 = {
        "stage_a_pass": True,
        "stage_b_executed": False,
        "portfolio_construction_disposition": "blocked_by_pw0329_precedence_gate_two",
        "gate8": gate,
    }
    pw0332 = {
        "analytical_survivor": False,
        "decoder_authorized": False,
        "absolute_floor": {"storage_tps": closure.PW0332_ABSOLUTE_TPS},
        "scenario_capacities": [204, 230, 250],
        "q1_demand_stream_sha256": closure.PW0328_Q1_DEMAND_SHA256,
        "codec_floor": {"zero_escape_bytes": 14_340},
        "codec_replay": {
            "quantization_blocks_replayed": 480,
            "byte_equal_to_canonical": True,
        },
        "oracle_replay": {
            "capacities_recomputed": [204, 230, 250],
            "full_canonical_ledgers_equal": True,
            "scenarios": {
                name: {"misses": misses}
                for name, misses in (
                    ("uncompressed", 53_040),
                    ("observed_expert_only", 50_743),
                    ("absolute_floor_all_fp8", 49_122),
                )
            },
        },
        "strict_gates": {
            "uncompressed": strict_gate_row(False),
            "observed_expert_only": strict_gate_row(False),
            "absolute_floor_all_fp8": strict_gate_row(False),
        },
        "gate8": gate,
    }
    throughput = {
        "measured_lower_milestones": measured,
        "reconciliation_pass": True,
        "reconciled_constants": [
            "pw0329_corrected_k4_joint_residency_bound",
            "pw0330_cyclic_mtp_q32_prefix_falsifier",
            "pw0331_byte_neutral_k4_rank1_stage_a",
            "pw0332_exact_top7_token_cache_oracle",
        ],
    }
    parent_paths = {
        "PW-0324": closure.PW0324_PATH,
        "PW-0328": closure.PW0328_PATH,
        "PW-0329": closure.PW0329_PATH,
        "PW-0330": closure.PW0330_PATH,
        "PW-0331": closure.PW0331_PATH,
        "PW-0332": closure.PW0332_PATH,
    }
    return closure.synthesize_report(
        commit="0" * 40,
        hardware={
            "system": "Darwin",
            "machine": "arm64",
            "processor": "Apple M1",
            "physical_memory_bytes": 16 * 1024**3,
        },
        parent_paths=parent_paths,
        pw0324=pw0324,
        pw0328=pw0328,
        pw0329=pw0329,
        pw0330=pw0330,
        pw0331=pw0331,
        pw0332=pw0332,
        throughput=throughput,
        safety_snapshots=safety_evidence(),
    )


class FrozenAuthorityTests(unittest.TestCase):
    def test_contract_blob_sha_and_throughput_model_are_frozen(self):
        repo = Path(__file__).resolve().parents[1]
        payload = subprocess.run(
            ["git", "-C", str(repo), "cat-file", "blob", closure.CONTRACT_GIT_BLOB],
            check=True,
            capture_output=True,
        ).stdout
        self.assertEqual(hashlib.sha256(payload).hexdigest(), closure.CONTRACT_SHA256)
        self.assertEqual(
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(repo),
                    "rev-parse",
                    f"{closure.CONTRACT_FREEZE_COMMIT}:{closure.CONTRACT_PATH}",
                ],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip(),
            closure.CONTRACT_GIT_BLOB,
        )
        self.assertEqual(
            closure.sha256_file(repo / closure.CONTRACT_PATH),
            closure.CONTRACT_SHA256,
        )
        self.assertEqual(
            closure.sha256_file(repo / "spec/throughput-model.json"),
            closure.THROUGHPUT_MODEL_SHA256,
        )

    def test_wrong_hash_rejects(self):
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "evidence.json"
            path.write_text('{"schema_version":1}\n')
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                closure._read_json(path, "0" * 64, "test evidence")

    def test_stale_bonus_free_manifest_semantic_rejects(self):
        manifest = {
            "schema_version": 1,
            "experiment_id": "PW-0328",
            "status": "complete",
            "evidence_class": closure.PW0328_EVIDENCE_CLASS,
            "semantic": closure.PW0328_SEMANTIC,
            "builder_git_dirty": False,
            "accepted_tokens": 0,
            "performance_claim": None,
            "batch_size": 1,
            "concurrency": 1,
        }
        closure.validate_pw0328_manifest_header(manifest)
        manifest["semantic"] = "stale_bonus_free_routes"
        with self.assertRaisesRegex(ValueError, "semantic/header mismatch"):
            closure.validate_pw0328_manifest_header(manifest)

    @mock.patch.object(closure, "verify_clean_commit", side_effect=ValueError("dirty Git"))
    def test_dirty_git_rejects_before_authority_work(self, verify):
        with self.assertRaisesRegex(ValueError, "dirty Git"):
            closure.verify_execution_commit(Path(__file__).resolve().parents[1], "0" * 40)
        verify.assert_called_once()

    def test_existing_output_rejects_before_other_work(self):
        with TemporaryDirectory() as temporary:
            with self.assertRaises(FileExistsError):
                closure.analyze(
                    repo=Path(__file__).resolve().parents[1],
                    commit="0" * 40,
                    output=Path(temporary),
                )

    def test_validated_report_creates_new_output_directory(self):
        with TemporaryDirectory() as temporary:
            output = Path(temporary) / "new-evidence"
            report = synthetic_report()
            path = closure.write_new_report(output, report)
            self.assertEqual(path, output / "analysis.json")
            self.assertEqual(json.loads(path.read_text()), report)

    def test_failed_atomic_write_removes_only_new_empty_directory(self):
        with TemporaryDirectory() as temporary:
            output = Path(temporary) / "new-evidence"
            with mock.patch.object(closure, "atomic_write_new", side_effect=OSError("write failed")):
                with self.assertRaisesRegex(OSError, "write failed"):
                    closure.write_new_report(output, synthetic_report())
            self.assertFalse(output.exists())


class ExactArithmeticTests(unittest.TestCase):
    def test_pw0329_aggregate_and_fourth_lowest_are_exact(self):
        aggregate = closure.recompute_storage_summary(
            moved_bytes=closure.PW0329_BYTES,
            accepted_tokens=closure.PW0329_A,
        )
        self.assertEqual(
            aggregate["storage_wall_seconds_fraction"],
            {"numerator": 514538083176000000, "denominator": 3470448309677419},
        )
        self.assertEqual(
            aggregate["storage_tps_fraction"],
            {"numerator": 100643000980645151, "denominator": 64317260397000000},
        )
        self.assertEqual(aggregate["storage_tps"], closure.PW0329_AGGREGATE_TPS)
        fourth = Fraction(3470448309677419, 3931444275000000)
        values = [fourth / 4, fourth / 3, fourth / 2, fourth, *([Fraction(2)] * 28)]
        self.assertEqual(closure.nearest_rank_p10(values), fourth)
        self.assertEqual(float(fourth), closure.PW0329_P10_TPS)

    def test_pw0330_joint_residency_formula_is_exact_and_conditional(self):
        misses = max(
            0,
            closure.FIXED_LOGICAL_BYTES
            + closure.MTP_ONLY_BYTES
            + closure.PW0330_IDENTITIES * closure.SOURCE_EXPERT_BYTES
            - closure.RESIDENCY_BYTES,
        )
        self.assertEqual(misses, closure.PW0330_MISS_BYTES)
        ceiling = closure.recompute_storage_summary(
            moved_bytes=misses,
            accepted_tokens=closure.PW0330_A,
        )
        self.assertEqual(ceiling["storage_tps"], closure.PW0330_TPS)
        report = synthetic_report()
        branch = report["reopened_branch_dispositions"]["pw0330_named_cyclic_mtp_q32"]
        self.assertEqual(branch["disposition"], "conditional_below_one_for_named_schedule")
        self.assertEqual(
            report["scope_boundaries"]["direct_q32_first_chunk_parity"],
            "unproven_outside_evidence_backed_survivors",
        )

    def test_pw0330_selected_identity_hash_tampering_rejects(self):
        report = pw0330_route_fixture()
        result = closure.recompute_pw0330_selected_route(report)
        self.assertEqual(result["A"], 4)
        self.assertEqual(result["N_A"], closure.PW0330_IDENTITIES)
        tampered = copy.deepcopy(report)
        tampered["selected_prefix_route"]["identity_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "selected route recomputation"):
            closure.recompute_pw0330_selected_route(tampered)

    def test_pw0330_accepted_prefix_tampering_rejects(self):
        report = pw0330_route_fixture()
        tampered = copy.deepcopy(report)
        tampered["head_results"][2]["match"] = False
        with self.assertRaisesRegex(ValueError, "accepted-prefix recomputation"):
            closure.recompute_pw0330_selected_route(tampered)

    def test_pw0332_codec_floor_formula(self):
        codec = {
            "block_codes": 16_384,
            "observed_witness_escapes": 341,
            "zero_escape_bytes": 14_340,
            "observed_bytes": 14_510,
            "zero_escape_ratio_fraction": fraction_record(Fraction(14_340, 16_384)),
            "observed_ratio_fraction": fraction_record(Fraction(14_510, 16_384)),
            "ratio_below_floor_possible": False,
        }
        result = closure._validate_pw0332_codec_floor(codec)
        self.assertEqual(result["zero_escape_bytes"], 14_340)
        self.assertFalse(result["ratio_below_floor_possible"])
        codec["zero_escape_bytes"] -= 1
        with self.assertRaisesRegex(ValueError, "codec-floor formula"):
            closure._validate_pw0332_codec_floor(codec)

    def test_pw0332_fresh_480_block_codec_replay_tampering_rejects(self):
        local = codec_replay_fixture()
        report = {"authority": {"pw0324": copy.deepcopy(local)}}
        result = closure.validate_pw0332_codec_replay(report, local)
        self.assertEqual(result["quantization_blocks_replayed"], 480)
        report["authority"]["pw0324"]["quantization_blocks"] = 479
        with self.assertRaisesRegex(ValueError, "480-block codec replay"):
            closure.validate_pw0332_codec_replay(report, local)

    def test_pw0332_full_oracle_ledger_tampering_rejects(self):
        names = [scenario.name for scenario in closure.PW0332_SCENARIOS]
        capacities = [204, 230, 250]
        misses = [53_040, 50_743, 49_122]
        baseline = [
            {
                "scenario": name,
                "layout": {"expert_capacity": capacity},
                "overall": {
                    "misses": miss,
                    "encoded_moved_bytes_fraction": fraction_record(miss),
                    "storage_wall_seconds_fraction": fraction_record(miss),
                    "storage_tps_fraction": fraction_record(Fraction(1, miss)),
                    "nearest_rank_p10_token_storage_tps_fraction": fraction_record(
                        Fraction(1, miss)
                    ),
                    "fourth_lowest_window_storage_tps_fraction": fraction_record(
                        Fraction(1, miss)
                    ),
                },
            }
            for name, capacity, miss in zip(names, capacities, misses, strict=True)
        ]
        dominance = {"pass": True}
        decision = {"decision": "reject_exact_top7_token_cache_oracle"}
        layouts = [
            {"scenario": name, "expert_capacity": capacity}
            for name, capacity in zip(names, capacities, strict=True)
        ]
        with (
            mock.patch.object(closure, "pw0332_scenario_layout", side_effect=layouts),
            mock.patch.object(closure, "validate_pw0332_layouts"),
            mock.patch.object(
                closure,
                "analyze_pw0332_scenario",
                side_effect=[copy.deepcopy(row) for row in baseline],
            ),
            mock.patch.object(
                closure, "validate_pw0332_scenario_dominance", return_value=dominance
            ),
            mock.patch.object(closure, "pw0332_disposition", return_value=decision),
        ):
            replayed, summary = closure.replay_pw0332_cache_oracle(
                canonical_scenarios=baseline,
                canonical_decision=decision,
                canonical_dominance=dominance,
                windows=[],
                traces={},
            )
        self.assertEqual([row["overall"]["misses"] for row in replayed], misses)
        self.assertTrue(summary["full_canonical_ledgers_equal"])
        tampered = copy.deepcopy(baseline)
        tampered[2]["overall"]["misses"] -= 1
        with (
            mock.patch.object(closure, "pw0332_scenario_layout", side_effect=layouts),
            mock.patch.object(closure, "validate_pw0332_layouts"),
            mock.patch.object(
                closure,
                "analyze_pw0332_scenario",
                side_effect=[copy.deepcopy(row) for row in baseline],
            ),
        ):
            with self.assertRaisesRegex(ValueError, "independent oracle replay mismatch"):
                closure.replay_pw0332_cache_oracle(
                    canonical_scenarios=tampered,
                    canonical_decision=decision,
                    canonical_dominance=dominance,
                    windows=[],
                    traces={},
                )

    def test_one_tps_is_not_strictly_above_one(self):
        exact_one = fraction_record(Fraction(1))
        scenario = {
            "overall": {
                "storage_tps_fraction": exact_one,
                "nearest_rank_p10_token_storage_tps_fraction": exact_one,
                "fourth_lowest_window_storage_tps_fraction": exact_one,
            },
            "categories": {
                category: {
                    "aggregate": {
                        "storage_tps_fraction": exact_one,
                        "nearest_rank_p10_token_storage_tps_fraction": exact_one,
                    }
                }
                for category in closure.CATEGORIES
            },
        }
        gates = closure.pw0332_strict_gates(scenario)
        self.assertFalse(gates["all_strict_gates_pass"])
        self.assertIs(closure.pw0332_all_strict_gates_survive(gates), False)


class TimingLedgerTests(unittest.TestCase):
    def test_complete_request_and_after_prefill_ledgers_are_not_conflated(self):
        repo = Path(__file__).resolve().parents[1]
        model = json.loads((repo / "spec/throughput-model.json").read_text())
        result = closure.measured_lower_milestones(model)
        complete = result["strongest_complete_request"]
        holdout = result["strongest_after_prefill_32_token_holdout"]
        short = result["strongest_after_prefill_short_slice"]
        self.assertEqual(complete["complete_request_accepted_tps"], 0.04597815174359703)
        self.assertEqual(
            complete["complete_request_timing_scope"],
            "complete_request_including_prefill",
        )
        self.assertEqual(holdout["after_prefill_accepted_tps"], 0.0791305231426806)
        self.assertEqual(
            holdout["after_prefill_accepted_tps_fraction"],
            {"numerator": 12800000000, "denominator": 161758061133},
        )
        self.assertGreater(short["after_prefill_accepted_tps"], holdout["after_prefill_accepted_tps"])
        self.assertEqual(short["accepted_tokens"], 7)
        self.assertEqual(holdout["accepted_tokens"], 32)
        self.assertEqual(result["designated_protocol"]["authenticated_results"], 0)
        self.assertEqual(result["designated_protocol"]["sustained_duration_minutes"], 60)
        self.assertEqual(
            result["designated_protocol"]["maximum_throughput_decay_fraction"],
            0.10,
        )
        self.assertEqual(
            result["designated_protocol"]["correctness_scope"],
            "unchanged_TARGET_sections_4_through_6",
        )
        diagnostic = result["highest_raw_diagnostic"]
        self.assertEqual(diagnostic["maximum_reported_accepted_tps"], 0.21984968624124546)
        self.assertEqual(
            diagnostic["classification"],
            "dirty_warm_single_verifier_block_L3_control",
        )
        self.assertFalse(diagnostic["target_qualifying"])


class ClosureDerivationTests(unittest.TestCase):
    def test_closure_requires_every_condition(self):
        result = closure.derive_closure(
            valid_conditions(),
            valid_branches(),
            pw0332_absolute_gates=strict_gate_row(False),
        )
        self.assertEqual(result["decision"], closure.FINAL_DECISION)
        self.assertEqual(result["failed_closure_conditions"], [])
        conditions = valid_conditions()
        conditions["no_authenticated_target_spec_sustained_result_reaches_one_tps"] = False
        result = closure.derive_closure(
            conditions,
            valid_branches(),
            pw0332_absolute_gates=strict_gate_row(False),
        )
        self.assertEqual(result["decision"], closure.FRONTIER_OPEN)

    def test_every_pw0332_gate_surviving_forces_frontier_open(self):
        result = closure.derive_closure(
            valid_conditions(),
            valid_branches(),
            pw0332_absolute_gates=strict_gate_row(True),
        )
        self.assertEqual(result["decision"], closure.FRONTIER_OPEN)
        self.assertIn(
            "pw0332_absolute_floor_fails_at_least_one_strict_one_tps_gate",
            result["failed_closure_conditions"],
        )

    def test_stale_pw0332_gate_shape_forces_frontier_open(self):
        result = closure.derive_closure(
            valid_conditions(),
            valid_branches(),
            pw0332_absolute_gates={},
        )
        self.assertEqual(result["decision"], closure.FRONTIER_OPEN)

    def test_missing_or_undecided_reopened_branch_forces_frontier_open(self):
        branches = valid_branches()
        branches.pop("pw0331_byte_neutral_rank1_repair")
        result = closure.derive_closure(
            valid_conditions(),
            branches,
            pw0332_absolute_gates=strict_gate_row(False),
        )
        self.assertEqual(result["decision"], closure.FRONTIER_OPEN)
        branches = valid_branches()
        branches["new_authenticated_branch"] = {
            "within_current_authenticated_portfolio": True,
            "decisive": False,
            "disposition": "unknown",
        }
        result = closure.derive_closure(
            valid_conditions(),
            branches,
            pw0332_absolute_gates=strict_gate_row(False),
        )
        self.assertEqual(result["decision"], closure.FRONTIER_OPEN)
        self.assertIn("new_authenticated_branch", result["reopened_branch_failures"])


class FinalClaimAndSafetyTests(unittest.TestCase):
    def test_valid_report_is_zero_work_and_non_universal(self):
        report = synthetic_report()
        closure.validate_final_report_schema(report)
        self.assertEqual(report["accepted_tokens"], 0)
        self.assertEqual(report["A"], 0)
        self.assertEqual(report["U"], 0)
        self.assertIsNone(report["performance_claim"])
        self.assertFalse(report["scope_boundaries"]["unknown_future_algorithms_rejected"])
        self.assertFalse(report["pw0331_prerequisite_interaction"]["stage_b_executed"])
        self.assertTrue(
            report["pw0331_prerequisite_interaction"][
                "blocked_by_pw0329_precedence_gate_two"
            ]
        )

    def test_non_null_performance_claim_rejects(self):
        report = synthetic_report()
        report["performance_claim"] = {"accepted_tps": 1.0}
        with self.assertRaisesRegex(ValueError, "zero-work analytical contract"):
            closure.validate_final_report_schema(report)

    def test_ceiling_relabelled_as_achieved_rejects(self):
        report = synthetic_report()
        ceiling = report["analytical_ceilings"]["pw0329_k4_fractional_impossible_best"]
        ceiling["achieved"] = True
        ceiling["claim_class"] = "achieved_endpoint_tps"
        with self.assertRaisesRegex(ValueError, "ceiling claim semantics"):
            closure.validate_final_report_schema(report)

    def test_nonzero_companion_premise_rejects(self):
        report = synthetic_report()
        report["scope"]["companion_contributions"]["bandwidth_bytes_per_second"] = 1
        with self.assertRaisesRegex(ValueError, "companion exclusion"):
            closure.validate_final_report_schema(report)

    def test_universal_or_direct_q32_overclaim_rejects(self):
        report = synthetic_report()
        report["scope_boundaries"]["unknown_future_algorithms_rejected"] = True
        with self.assertRaisesRegex(ValueError, "non-universal scope boundary"):
            closure.validate_final_report_schema(report)
        report = synthetic_report()
        report["scope_boundaries"]["direct_q32_first_chunk_parity"] = "rejected"
        with self.assertRaisesRegex(ValueError, "non-universal scope boundary"):
            closure.validate_final_report_schema(report)

    def test_unsafe_gate8_rejects(self):
        evidence = safety_evidence()
        evidence[0]["system_memory_free_percent"] = 9
        with self.assertRaisesRegex(ValueError, "memory safety mismatch"):
            closure.validate_gate8(evidence, label="unsafe fixture")


CANONICAL_PATHS = (
    closure.PW0324_PATH,
    closure.PW0328_PATH,
    closure.PW0329_PATH,
    closure.PW0330_PATH,
    closure.PW0331_PATH,
    closure.PW0332_PATH,
)


@unittest.skipUnless(all(path.is_file() for path in CANONICAL_PATHS), "canonical evidence unavailable")
class CanonicalParentParityTests(unittest.TestCase):
    def test_all_six_parents_and_throughput_model_recompute(self):
        repo = Path(__file__).resolve().parents[1]
        model = json.loads((repo / "spec/throughput-model.json").read_text())
        measured = closure.measured_lower_milestones(model)
        manifest = closure._read_json(closure.PW0328_PATH, closure.PW0328_SHA256, "PW-0328")
        closure.validate_pw0328_manifest_header(manifest)
        live = closure.authenticate_pw0328_corpus(repo=repo, manifest_path=closure.PW0328_PATH)
        pw0324 = closure.validate_pw0324(
            closure._read_json(closure.PW0324_PATH, closure.PW0324_SHA256, "PW-0324"),
            measured,
        )
        pw0328 = closure.summarize_pw0328(live)
        pw0329 = closure.validate_pw0329(
            closure._read_json(closure.PW0329_PATH, closure.PW0329_SHA256, "PW-0329")
        )
        pw0330 = closure.validate_pw0330(
            closure._read_json(closure.PW0330_PATH, closure.PW0330_SHA256, "PW-0330")
        )
        pw0331 = closure.validate_pw0331(
            closure._read_json(closure.PW0331_PATH, closure.PW0331_SHA256, "PW-0331")
        )
        local_codec_replay = closure.authenticate_pw0332_pw0324(
            closure.PW0332_DEFAULT_CHECKPOINT_ROOT
        )
        pw0332 = closure.validate_pw0332(
            closure._read_json(closure.PW0332_PATH, closure.PW0332_SHA256, "PW-0332"),
            live,
            local_codec_replay,
        )
        throughput = closure.validate_throughput_model(
            model,
            pw0329=pw0329,
            pw0330=pw0330,
            pw0331=pw0331,
            pw0332=pw0332,
        )
        report = closure.synthesize_report(
            commit="0" * 40,
            hardware={
                "system": "Darwin",
                "machine": "arm64",
                "processor": "Apple M1",
                "physical_memory_bytes": 16 * 1024**3,
            },
            parent_paths={
                "PW-0324": closure.PW0324_PATH,
                "PW-0328": closure.PW0328_PATH,
                "PW-0329": closure.PW0329_PATH,
                "PW-0330": closure.PW0330_PATH,
                "PW-0331": closure.PW0331_PATH,
                "PW-0332": closure.PW0332_PATH,
            },
            pw0324=pw0324,
            pw0328=pw0328,
            pw0329=pw0329,
            pw0330=pw0330,
            pw0331=pw0331,
            pw0332=pw0332,
            throughput=throughput,
            safety_snapshots=safety_evidence(),
        )
        self.assertEqual(pw0324["unreconciled_proposer_family_survivors"], 0)
        self.assertEqual(pw0328["sum_A"], 232)
        self.assertEqual(pw0328["sum_observable_A"], 231)
        self.assertEqual(pw0329["ceiling"]["aggregate_storage_tps"], closure.PW0329_AGGREGATE_TPS)
        self.assertEqual(pw0330["favorable_storage_tps_ceiling"], closure.PW0330_TPS)
        self.assertTrue(pw0331["stage_a_pass"])
        self.assertEqual(pw0332["q1_demand_stream_sha256"], closure.PW0328_Q1_DEMAND_SHA256)
        self.assertEqual(pw0332["absolute_floor"]["storage_tps"], closure.PW0332_ABSOLUTE_TPS)
        self.assertTrue(throughput["reconciliation_pass"])
        self.assertEqual(report["decision"], closure.FINAL_DECISION)
        self.assertEqual(
            report["analytical_ceilings"]["pw0332_exact_codec_absolute_floor"]
            ["oracle_replay"]["capacities_recomputed"],
            [204, 230, 250],
        )


if __name__ == "__main__":
    unittest.main()
