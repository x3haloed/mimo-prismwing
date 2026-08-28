#!/usr/bin/env python3
"""Synthesize the corrected PW-0333 onboard one-TPS portfolio closure.

This analyzer authenticates the six frozen parent reports and independently
recomputes their decision-bearing quantities.  It does not execute a model,
construct a codec or K4 bank, accept tokens, or report endpoint throughput.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import platform
import subprocess
from typing import Any, Iterable, Mapping, Sequence

try:
    from tools.analyze_pw0332_top7_cache_oracle import (
        DEFAULT_CHECKPOINT_ROOT as PW0332_DEFAULT_CHECKPOINT_ROOT,
        SCENARIOS as PW0332_SCENARIOS,
        analyze_scenario as analyze_pw0332_scenario,
        authenticate_pw0324 as authenticate_pw0332_pw0324,
        demand_stream_sha256,
        disposition as pw0332_disposition,
        normalize_corpus_authority,
        scenario_layout as pw0332_scenario_layout,
        strict_gates as pw0332_strict_gates,
        validate_layouts as validate_pw0332_layouts,
        validate_report_schema as validate_pw0332_report_schema,
        validate_scenario_dominance as validate_pw0332_scenario_dominance,
        validate_storage_authority,
    )
    from tools.host_safety import HostSafetyMonitor, HostSafetyViolation
    from tools.openrouter_reference import atomic_write_new, canonical_json
    from tools.pw0328_corpus_authority import (
        MANIFEST_EVIDENCE_CLASS as PW0328_EVIDENCE_CLASS,
        MANIFEST_SEMANTIC as PW0328_SEMANTIC,
        authenticate_pw0328_corpus,
        validate_gate8,
    )
    from tools.reproduce_pw0311_k4_expert import verify_clean_commit
except ModuleNotFoundError:
    from analyze_pw0332_top7_cache_oracle import (
        DEFAULT_CHECKPOINT_ROOT as PW0332_DEFAULT_CHECKPOINT_ROOT,
        SCENARIOS as PW0332_SCENARIOS,
        analyze_scenario as analyze_pw0332_scenario,
        authenticate_pw0324 as authenticate_pw0332_pw0324,
        demand_stream_sha256,
        disposition as pw0332_disposition,
        normalize_corpus_authority,
        scenario_layout as pw0332_scenario_layout,
        strict_gates as pw0332_strict_gates,
        validate_layouts as validate_pw0332_layouts,
        validate_report_schema as validate_pw0332_report_schema,
        validate_scenario_dominance as validate_pw0332_scenario_dominance,
        validate_storage_authority,
    )
    from host_safety import HostSafetyMonitor, HostSafetyViolation
    from openrouter_reference import atomic_write_new, canonical_json
    from pw0328_corpus_authority import (
        MANIFEST_EVIDENCE_CLASS as PW0328_EVIDENCE_CLASS,
        MANIFEST_SEMANTIC as PW0328_SEMANTIC,
        authenticate_pw0328_corpus,
        validate_gate8,
    )
    from reproduce_pw0311_k4_expert import verify_clean_commit


EXPERIMENT_ID = "PW-0333"
CONTRACT_PATH = "experiments/PW-0333-corrected-onboard-prismwing1-closure.md"
CONTRACT_FREEZE_COMMIT = "cd9a11ca3475402a77264092fa9eb904fc1608cd"
CONTRACT_GIT_BLOB = "379d532be22e621be2832276c93f29848e806b99"
CONTRACT_SHA256 = "1edb668c9830780b7f0544568fdae8c54fbb1985dbe4c321c09923ea7db46992"
TARGET_SHA256 = "dda459684c194b03491f36e9b66521ff00c400a6cc38d23a567a5a92ef8fb17d"
RED_LINES_SHA256 = "cc261ad9bd67a865715e72cbbadf3b74c3f1f282e17a8ef86ed02c1a92fb8b36"
THROUGHPUT_MODEL_SHA256 = "8d66f3a2f269952f0b3a44fcfbe272dcdc0fd12093cc8034319d61e0bb60fe1f"

PW0324_PATH = Path(
    "/Users/chad/Models/mimo-prismwing/evidence/PW-0324/analysis-002/analysis.json"
)
PW0324_SHA256 = "97d4d20a4c709d42429973e867138495756ce9d52d417f98a7edd40b282ccff3"
PW0328_PATH = Path(
    "/Volumes/Elements/mimo-prismwing/evidence/PW-0328/corpus-001/manifest.json"
)
PW0328_SHA256 = "36e4f10b6f807f766c87ee7078f5f18ea8fc339dd12e4dbc24f1f4ac6e824403"
PW0329_PATH = Path(
    "/Volumes/Elements/mimo-prismwing/evidence/PW-0329/analysis-001/analysis.json"
)
PW0329_SHA256 = "81af4d7b9158fe170503755c38436d5266e41c57a9e67d9c98e142995fdce6f6"
PW0330_PATH = Path(
    "/Volumes/Elements/mimo-prismwing/evidence/PW-0330/run-001/report.json"
)
PW0330_SHA256 = "fbb454f6992ba8e21ade89aff416a494d14625dc126b769f420a861ed6414674"
PW0331_PATH = Path(
    "/Volumes/Elements/mimo-prismwing/evidence/PW-0331/analysis-004/analysis.json"
)
PW0331_SHA256 = "fd5ac314b7e9072f22f773496444678c91f8be0a5165fa24e8df8687906c23c7"
PW0332_PATH = Path(
    "/Volumes/Elements/mimo-prismwing/evidence/PW-0332/analysis-001/analysis.json"
)
PW0332_SHA256 = "e2452a4f2eb9b66ed89097e8e78e5158f7ea53cc00bce8a2ba52c821f61ea085"
PW0332_COMMIT = "d9691ee84bd728093305ed7fa8e403815394bb01"
PW0332_CONTRACT_GIT_BLOB = "073fcb4fd52330acb8ed8d8d645f521ae2ded3b8"
PW0332_CONTRACT_SHA256 = "e37d1586311989f2e23e1af5737774d1332077e705baaf8d53e96e63e75d90e1"
PW0328_Q1_DEMAND_SHA256 = "91fd42fe48033a1b04c1b3d9cdba30a4e6847147064db9946e71c6595bf71db6"

CHECKPOINT_REVISION = "63651580ca774f8504f676040460aed3e1244ac1"
CHECKPOINT_RECEIPT_SHA256 = "9ddc8a99755f04ae2ea3c2484f6dd022d3f3a681b5a72c915ee4de833dbb0d03"
CHECKPOINT_INDEX_SHA256 = "f2e1774c9acf9a62338b68c144e6fc7a66495e59f2e64b3078c1b7ef5a196816"

CATEGORIES = ("ordinary", "code", "multilingual", "rare_route")
EXPECTED_CATEGORY_A = {
    "ordinary": 50,
    "code": 58,
    "multilingual": 60,
    "rare_route": 64,
}
EXPECTED_CATEGORY_U = {
    "ordinary": 35.58510638297872,
    "code": 38.015957446808514,
    "multilingual": 33.11702127659574,
    "rare_route": 36.0,
}
EXPECTED_CATEGORY_OBSERVABLE_A = {
    "ordinary": 50,
    "code": 58,
    "multilingual": 60,
    "rare_route": 63,
}

SOURCE_EXPERT_BYTES = 25_171_968
FIXED_LOGICAL_BYTES = 7_743_236_992
MTP_ONLY_BYTES = 1_189_400_448
RESIDENCY_BYTES = 12 * 1024**3
FAVORABLE_BANDWIDTH = Fraction("3470448309.677419")
FAVORABLE_BANDWIDTH_FLOAT = 3_470_448_309.677419
ONE_TPS = Fraction(1, 1)

PW0329_BYTES = 514_538_083_176
PW0329_A = 232
PW0329_AGGREGATE_TPS = 1.564789923566762
PW0329_P10_TPS = 0.8827413202181071
PW0330_A = 4
PW0330_IDENTITIES = 1_035
PW0330_MISS_BYTES = 22_100_722_432
PW0330_TPS = 0.6281149080724167
PW0331_ROUTE_L2 = 0.008777164859819555
PW0331_FINAL_L2 = 0.0024352236927816023
PW0331_LEFT_SHA256 = "68bf04d76d15c4acb4fbecf37948809d3324f916f975884251ea04250eb6ba84"
PW0331_RIGHT_SHA256 = "ab7fbd8d45493906cce7ae264f26b800c099b8b50d4c30b1de64b0b20b30136d"
PW0332_ABSOLUTE_MISSES = 49_122
PW0332_ABSOLUTE_ENCODED_BYTES = 1_082_237_114_835
PW0332_ABSOLUTE_WALL = 311.8436058584013
PW0332_ABSOLUTE_TPS = 0.7439626647510745
PW0332_TOKEN_P10 = 0.5899672933278813
PW0332_WINDOW_P10 = 0.6962265958830688

PRIOR_PORTFOLIO_KEYS = {
    "source_cache_and_residency",
    "predictive_prefetch",
    "source_stream_transport",
    "native_mtp",
    "published_bounded_proposers",
    "approximate_mismatch_acceptance",
    "causal_q64",
    "exact_fp8_local_codecs",
    "six_bit_fp8_subset",
    "affine_int4",
    "global_hessian_int4",
    "affine_six_bit",
    "vector_code",
    "subvector_code",
    "microscaling_fp4",
    "activation_sparsity",
    "shared_basis",
    "routed_mixture_compiler",
    "exception_store",
    "k4_hybrid_bank",
    "fixed_subset_sparse_attention",
    "structured_sparse_attention",
}

PW0324_PROPOSER_FAMILY_STATES = {
    "predictive_prefetch": "rejected",
    "native_mtp": "rejected_cost_gate_lower_latency_retained",
    "published_bounded_proposers": "rejected_direct",
    "approximate_mismatch_acceptance": "rejected_direct_and_fidelity_unqualified",
    "causal_q64": "rejected_actual_acceptance",
}

FINAL_DECISION = "close_corrected_onboard_prismwing1_frontier_below_one_tps"
FRONTIER_OPEN = "frontier_open"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _mapping(value: Any, message: str) -> dict[str, Any]:
    _require(isinstance(value, dict), message)
    return value


def _sequence(value: Any, message: str) -> list[Any]:
    _require(isinstance(value, list), message)
    return value


def _integer(value: Any, message: str, *, minimum: int = 0) -> int:
    _require(type(value) is int and value >= minimum, message)
    return value


def _fraction(value: Any, label: str) -> Fraction:
    row = _mapping(value, f"{label}: exact fraction")
    _require(set(row) == {"numerator", "denominator"}, f"{label}: fraction schema")
    numerator = _integer(row.get("numerator"), f"{label}: numerator")
    denominator = _integer(row.get("denominator"), f"{label}: denominator", minimum=1)
    return Fraction(numerator, denominator)


def fraction_record(value: Fraction) -> dict[str, int]:
    _require(isinstance(value, Fraction), "fraction record requires an exact fraction")
    return {"numerator": value.numerator, "denominator": value.denominator}


def _same_float(actual: Any, expected: float, label: str) -> None:
    _require(
        type(actual) in (int, float)
        and not isinstance(actual, bool)
        and math.isfinite(float(actual))
        and float(actual) == expected,
        f"{label}: floating-point mismatch",
    )


def _same_display_float(actual: Any, expected: float, label: str) -> None:
    """Accept only sub-1e-15 decimal/IEEE display rounding around exact evidence."""

    _require(
        type(actual) in (int, float)
        and not isinstance(actual, bool)
        and math.isfinite(float(actual))
        and math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=1.0e-15),
        f"{label}: floating-point display mismatch",
    )


def _read_json(path: Path, expected_sha256: str, label: str) -> dict[str, Any]:
    actual = sha256_file(path)
    _require(actual == expected_sha256, f"{label}: SHA-256 mismatch")
    return _mapping(json.loads(path.read_text()), f"{label}: JSON root")


def _git(repo: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *arguments],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def verify_execution_commit(repo: Path, commit: str) -> None:
    """Require clean HEAD, exact frozen authorities, and freeze ancestry."""

    verify_clean_commit(repo, commit)
    _require(
        _git(repo, "rev-parse", f"{commit}:{CONTRACT_PATH}") == CONTRACT_GIT_BLOB,
        "PW-0333 contract Git blob mismatch",
    )
    _require(
        sha256_file(repo / CONTRACT_PATH) == CONTRACT_SHA256,
        "PW-0333 contract SHA-256 mismatch",
    )
    ancestor = subprocess.run(
        ["git", "-C", str(repo), "merge-base", "--is-ancestor", CONTRACT_FREEZE_COMMIT, commit],
        check=False,
        capture_output=True,
        text=True,
    )
    _require(ancestor.returncode == 0, "execution commit predates PW-0333 contract freeze")
    _require(sha256_file(repo / "TARGET.md") == TARGET_SHA256, "TARGET.md authority mismatch")
    _require(
        sha256_file(repo / "RED_LINES.md") == RED_LINES_SHA256,
        "RED_LINES.md authority mismatch",
    )
    _require(
        sha256_file(repo / "spec/throughput-model.json") == THROUGHPUT_MODEL_SHA256,
        "throughput model authority mismatch",
    )


def verify_target_hardware() -> dict[str, Any]:
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise ValueError("PW-0333 requires the target Apple M1 host")
    brand = subprocess.run(
        ["/usr/sbin/sysctl", "-n", "machdep.cpu.brand_string"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    memory = int(
        subprocess.run(
            ["/usr/sbin/sysctl", "-n", "hw.memsize"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    _require(brand == "Apple M1", "hardware authority is not Apple M1")
    _require(memory == 16 * 1024**3, "hardware authority is not 16 GiB")
    return {
        "system": platform.system(),
        "machine": platform.machine(),
        "processor": brand,
        "physical_memory_bytes": memory,
    }


def nearest_rank_p10(values: Sequence[Fraction]) -> Fraction:
    _require(bool(values), "p10 requires at least one value")
    _require(all(isinstance(value, Fraction) and value >= 0 for value in values), "invalid p10 value")
    return sorted(values)[math.ceil(len(values) / 10) - 1]


def recompute_storage_summary(
    *, moved_bytes: int, accepted_tokens: int, bandwidth: Fraction = FAVORABLE_BANDWIDTH
) -> dict[str, Any]:
    moved_bytes = _integer(moved_bytes, "moved bytes", minimum=1)
    accepted_tokens = _integer(accepted_tokens, "accepted tokens", minimum=1)
    _require(isinstance(bandwidth, Fraction) and bandwidth > 0, "invalid bandwidth")
    wall = Fraction(moved_bytes, 1) / bandwidth
    tps = Fraction(accepted_tokens, 1) / wall
    return {
        "bytes_moved": moved_bytes,
        "accepted_tokens_modeled": accepted_tokens,
        "bandwidth_bytes_per_second": float(bandwidth),
        "storage_wall_seconds": float(wall),
        "storage_wall_seconds_fraction": {
            "numerator": wall.numerator,
            "denominator": wall.denominator,
        },
        "storage_tps": float(tps),
        "storage_tps_fraction": {
            "numerator": tps.numerator,
            "denominator": tps.denominator,
        },
    }


def measured_lower_milestones(throughput_model: Mapping[str, Any]) -> dict[str, Any]:
    """Derive complete-request and after-prefill rates without conflating them."""

    constants = _mapping(throughput_model.get("constants"), "throughput model constants")
    rows: list[dict[str, Any]] = []
    for name, raw in constants.items():
        if not name.startswith(("pw0211_", "pw0215_", "pw0216_")):
            continue
        row = _mapping(raw, f"throughput constant {name}")
        accepted = _integer(
            row.get("accepted_tokens_per_complete_request"),
            f"{name}: accepted tokens",
            minimum=1,
        )
        _require(row.get("batch_size") == 1 and row.get("concurrency") == 1, f"{name}: batch/concurrency")
        post_ms = row.get("candidate_post_prefill_wall_median_ms")
        complete_tps = row.get("candidate_complete_accepted_tps_median")
        _require(
            type(post_ms) in (int, float)
            and not isinstance(post_ms, bool)
            and math.isfinite(float(post_ms))
            and float(post_ms) > 0,
            f"{name}: post-prefill wall",
        )
        _require(
            type(complete_tps) in (int, float)
            and not isinstance(complete_tps, bool)
            and math.isfinite(float(complete_tps))
            and float(complete_tps) > 0,
            f"{name}: complete-request TPS",
        )
        status = str(row.get("status", ""))
        _require("not_general_default" in status, f"{name}: lower-milestone scope")
        after_prefill = Fraction(accepted * 1000, 1) / Fraction(str(float(post_ms)))
        rows.append(
            {
                "constant": name,
                "accepted_tokens": accepted,
                "batch_size": 1,
                "concurrency": 1,
                "complete_request_accepted_tps": float(complete_tps),
                "after_prefill_wall_ms": float(post_ms),
                "complete_request_timing_scope": "complete_request_including_prefill",
                "after_prefill_timing_scope": "measured_post_prefill_candidate_slice",
                "after_prefill_accepted_tps": float(after_prefill),
                "after_prefill_accepted_tps_fraction": fraction_record(after_prefill),
                "scope": (
                    "32_token_untouched_holdout_lower_milestone"
                    if accepted == 32
                    else "7_token_short_slice_lower_milestone"
                ),
                "status": status,
                "provenance": row.get("provenance"),
                "meets_designated_30_by_512_protocol": False,
            }
        )
    _require(len(rows) == 6, "measured lower-milestone constant set")
    rows.sort(key=lambda row: row["constant"])
    strongest_complete = max(rows, key=lambda row: row["complete_request_accepted_tps"])
    strongest_after_prefill = max(rows, key=lambda row: row["after_prefill_accepted_tps"])
    holdouts = [row for row in rows if row["accepted_tokens"] == 32]
    _require(len(holdouts) == 2, "32-token holdout set")
    strongest_holdout = max(holdouts, key=lambda row: row["after_prefill_accepted_tps"])
    _same_float(
        strongest_complete["complete_request_accepted_tps"],
        0.04597815174359703,
        "PW-0216 strongest complete-request TPS",
    )
    _require(
        strongest_complete["constant"]
        == "pw0216_live_native_mtp_q4_ordinary_32_token_holdout",
        "strongest complete-request lower milestone identity",
    )
    _same_float(
        strongest_holdout["after_prefill_accepted_tps"],
        0.0791305231426806,
        "PW-0216 strongest 32-token after-prefill TPS",
    )
    _require(
        strongest_after_prefill["accepted_tokens"] == 7
        and strongest_after_prefill["constant"] == "pw0215_live_native_mtp_q4_multilingual",
        "strongest short after-prefill lower milestone identity",
    )
    _require(
        all(row["after_prefill_accepted_tps"] < 1.0 for row in rows)
        and all(row["complete_request_accepted_tps"] < 1.0 for row in rows),
        "measured lower milestone unexpectedly reaches one TPS",
    )
    pw0203 = _mapping(
        constants.get("pw0203_wide_source_jacobi_endpoint"),
        "throughput diagnostic PW-0203",
    )
    pw0207 = _mapping(
        constants.get("pw0207_final_forty_two_object_residency"),
        "throughput diagnostic PW-0207",
    )
    pw0206 = _mapping(
        constants.get("pw0206_corrected_jacobi_third_iteration"),
        "throughput diagnostic PW-0206",
    )
    _same_float(
        pw0203.get("warm_accepted_tps"),
        0.21984968624124546,
        "PW-0203 warm verifier diagnostic",
    )
    _same_float(
        pw0207.get("resident_median_committed_transaction_tps"),
        0.04375596599496899,
        "PW-0207 transaction diagnostic",
    )
    _same_float(
        pw0206.get("single_trace_post_prefill_accepted_tps_diagnostic"),
        0.009733450580291214,
        "PW-0206 single-trace diagnostic",
    )
    _require(
        "dirty" in str(pw0203.get("provenance"))
        and "named_l3_reduction" in str(pw0203.get("status"))
        and "two_x_endpoint_gate_failed" in str(pw0207.get("status"))
        and "no_endpoint_tps" in str(pw0206.get("status")),
        "excluded diagnostic classifications",
    )
    excluded_diagnostics = [
        {
            "constant": "pw0203_wide_source_jacobi_endpoint",
            "maximum_reported_accepted_tps": float(pw0203["warm_accepted_tps"]),
            "classification": "dirty_warm_single_verifier_block_L3_control",
            "complete_request": False,
            "sustained": False,
            "target_qualifying": False,
            "reason_excluded_from_clean_repeatable_request_slices": (
                "dirty Git; one accelerated target-verifier block; L3 reduction"
            ),
        },
        {
            "constant": "pw0207_final_forty_two_object_residency",
            "maximum_reported_accepted_tps": float(
                pw0207["resident_median_committed_transaction_tps"]
            ),
            "classification": "exact_single_transaction_lower_milestone",
            "complete_request": False,
            "sustained": False,
            "target_qualifying": False,
            "reason_excluded_from_clean_repeatable_request_slices": (
                "one verifier transaction; no endpoint promotion"
            ),
        },
        {
            "constant": "pw0206_corrected_jacobi_third_iteration",
            "maximum_reported_accepted_tps": float(
                pw0206["single_trace_post_prefill_accepted_tps_diagnostic"]
            ),
            "classification": "single_trace_post_prefill_diagnostic",
            "complete_request": False,
            "sustained": False,
            "target_qualifying": False,
            "reason_excluded_from_clean_repeatable_request_slices": "no endpoint TPS",
        },
    ]
    _require(
        max(row["maximum_reported_accepted_tps"] for row in excluded_diagnostics) < 1.0,
        "excluded diagnostic unexpectedly reaches one TPS",
    )
    return {
        "rows": rows,
        "strongest_complete_request": strongest_complete,
        "strongest_after_prefill_short_slice": strongest_after_prefill,
        "strongest_after_prefill_32_token_holdout": strongest_holdout,
        "designated_protocol": {
            "prefill_tokens": 8192,
            "generations": 30,
            "accepted_tokens_per_generation": 512,
            "decode_timing_scope": "after_prefill_complete_decode_path",
            "sustained_duration_minutes": 60,
            "maximum_throughput_decay_fraction": 0.10,
            "correctness_scope": "unchanged_TARGET_sections_4_through_6",
            "authenticated_results": 0,
            "passing_results": 0,
        },
        "ledger_scope": "clean_repeatable_native_mtp_request_slice_milestones",
        "excluded_diagnostic_classes": excluded_diagnostics,
        "highest_raw_diagnostic": excluded_diagnostics[0],
        "claim_class": "measured_lower_milestones_not_designated_sustained_protocol",
    }


def validate_pw0324(
    report: Mapping[str, Any], measured: Mapping[str, Any]
) -> dict[str, Any]:
    scope = _mapping(report.get("scope"), "PW-0324 scope")
    portfolio = _mapping(report.get("portfolio"), "PW-0324 portfolio")
    closure = _mapping(report.get("closure_conditions"), "PW-0324 closure conditions")
    path = _mapping(report.get("measured_complete_path"), "PW-0324 measured path")
    limitations = _sequence(report.get("limitations"), "PW-0324 limitations")
    _require(
        report.get("schema_version") == 1
        and report.get("experiment_id") == "PW-0324"
        and report.get("status") == "complete"
        and report.get("decision") == "close_current_onboard_prismwing2_frontier"
        and report.get("accepted_tokens") == 0
        and report.get("performance_claim") is None
        and report.get("runtime_default_changed") is False,
        "PW-0324 top-level contract",
    )
    _require(
        scope.get("hardware") == "existing 16 GiB Apple M1"
        and scope.get("companion_hardware_admissible") is False
        and scope.get("correctness_contract_changed") is False
        and scope.get("target_50_changed") is False,
        "PW-0324 scope contract",
    )
    _require(set(portfolio) == PRIOR_PORTFOLIO_KEYS, "PW-0324 portfolio mechanism set")
    for mechanism, raw in portfolio.items():
        row = _mapping(raw, f"PW-0324 portfolio {mechanism}")
        _require(
            isinstance(row.get("record"), str)
            and isinstance(row.get("record_sha256"), str)
            and len(row["record_sha256"]) == 64
            and isinstance(row.get("state"), str)
            and bool(row["state"])
            and row.get("survives_two_tps_closure") is False,
            f"PW-0324 portfolio disposition: {mechanism}",
        )
    proposer_states = {
        name: _mapping(portfolio[name], f"PW-0324 proposer {name}").get("state")
        for name in PW0324_PROPOSER_FAMILY_STATES
    }
    _require(
        proposer_states == PW0324_PROPOSER_FAMILY_STATES,
        "PW-0324 proposer-family disposition map",
    )
    _require(bool(closure) and all(value is True for value in closure.values()), "PW-0324 closure condition")
    _require(report.get("failed_closure_conditions") == [], "PW-0324 failed closure conditions")
    _require(
        any("not a theorem against unknown future algorithms" in str(item) for item in limitations),
        "PW-0324 unknown-algorithm limitation",
    )
    strongest = _mapping(measured.get("strongest_complete_request"), "measured strongest complete request")
    _require(
        path.get("constant") == strongest.get("constant")
        and path.get("accepted_tokens") == strongest.get("accepted_tokens")
        and path.get("batch_size") == 1
        and path.get("concurrency") == 1,
        "PW-0324 measured-path identity",
    )
    _same_float(
        path.get("accepted_tps"),
        float(strongest["complete_request_accepted_tps"]),
        "PW-0324 complete-request TPS",
    )
    gate8 = validate_gate8(report.get("analysis_safety"), label="PW-0324 analyzer")
    return {
        "decision": report["decision"],
        "portfolio_mechanisms": len(portfolio),
        "two_tps_closure_survivors": 0,
        "authenticated_proposer_family_states": proposer_states,
        "unreconciled_proposer_family_survivors": 0,
        "portfolio": {name: dict(row) for name, row in portfolio.items()},
        "historical_portfolio_hashes_preserved_not_rehashed_against_updated_records": True,
        "complete_request_lower_milestone": dict(path),
        "unknown_algorithm_limitation_present": True,
        "gate8": gate8,
    }


def validate_pw0328_manifest_header(manifest: Mapping[str, Any]) -> None:
    """Reject the stale bonus-free semantic before invoking the full replay."""

    _require(
        manifest.get("schema_version") == 1
        and manifest.get("experiment_id") == "PW-0328"
        and manifest.get("status") == "complete"
        and manifest.get("evidence_class") == PW0328_EVIDENCE_CLASS
        and manifest.get("semantic") == PW0328_SEMANTIC
        and manifest.get("builder_git_dirty") is False
        and manifest.get("accepted_tokens") == 0
        and manifest.get("performance_claim") is None
        and manifest.get("batch_size") == 1
        and manifest.get("concurrency") == 1,
        "PW-0328 manifest semantic/header mismatch",
    )


def summarize_pw0328(authority: Mapping[str, Any]) -> dict[str, Any]:
    windows = _sequence(authority.get("windows"), "PW-0328 windows")
    q1_events = _sequence(authority.get("q1_events"), "PW-0328 q1 events")
    sources = _sequence(authority.get("sources"), "PW-0328 generation sources")
    prefill_sources = _sequence(
        authority.get("prefill_sources"), "PW-0328 prefill sources"
    )
    categories = Counter(str(row.get("category")) for row in windows if isinstance(row, dict))
    _require(
        authority.get("manifest_sha256") == PW0328_SHA256
        and authority.get("builder_commit") == "26d2ea31852c0d63bd022df6d571fd722137c39f"
        and authority.get("artifact_count") == 24
        and len(_sequence(authority.get("artifacts"), "PW-0328 artifacts")) == 24
        and categories == Counter({category: 8 for category in CATEGORIES})
        and len(windows) == 32
        and len(q1_events) == 232,
        "PW-0328 authority cardinality",
    )
    _require(
        len(sources) == len(prefill_sources) == 4
        and [row.get("category") for row in sources] == list(CATEGORIES)
        and [row.get("category") for row in prefill_sources] == list(CATEGORIES),
        "PW-0328 source Gate 8 order",
    )
    generation_gate8 = [
        _mapping(row.get("gate8"), f"PW-0328 generation Gate 8 {index}")
        for index, row in enumerate(sources)
    ]
    prefill_gate8 = [
        _mapping(row.get("gate8"), f"PW-0328 prefill Gate 8 {index}")
        for index, row in enumerate(prefill_sources)
    ]
    _require(
        all(row.get("pass") is True for row in generation_gate8 + prefill_gate8),
        "PW-0328 generation/prefill Gate 8 closure",
    )
    sum_a = sum(_integer(row.get("A"), "PW-0328 window A", minimum=1) for row in windows)
    observable_a = sum(
        _integer(row.get("observable_A"), "PW-0328 observable A", minimum=1)
        for row in windows
    )
    sum_u = math.fsum(float(row.get("U")) for row in windows)
    by_category: dict[str, Any] = {}
    for category in CATEGORIES:
        selected = [row for row in windows if row.get("category") == category]
        category_a = sum(int(row["A"]) for row in selected)
        category_observable = sum(int(row["observable_A"]) for row in selected)
        category_u = math.fsum(float(row["U"]) for row in selected)
        _require(
            category_a == EXPECTED_CATEGORY_A[category]
            and category_observable == EXPECTED_CATEGORY_OBSERVABLE_A[category]
            and math.isclose(
                category_u,
                EXPECTED_CATEGORY_U[category],
                rel_tol=0.0,
                abs_tol=1.0e-12,
            ),
            f"PW-0328 category control: {category}",
        )
        by_category[category] = {
            "windows": 8,
            "sum_A": category_a,
            "sum_observable_A": category_observable,
            "sum_U": category_u,
        }
    _require(
        sum_a == 232
        and observable_a == 231
        and math.isclose(sum_u, 142.71808510638297, rel_tol=0.0, abs_tol=1.0e-12),
        "PW-0328 full-corpus control",
    )
    rare = _mapping(authority.get("rare_route_evidence"), "PW-0328 rare-route evidence")
    _require(
        rare.get("novel_identity_count") == 939
        and rare.get("rare_route_unique_identities") == 5167
        and rare.get("novel_routed_layers") == list(range(1, 48)),
        "PW-0328 rare-route evidence",
    )
    builder_gate = _mapping(authority.get("builder_gate8"), "PW-0328 builder Gate 8")
    _require(builder_gate.get("pass") is True, "PW-0328 builder Gate 8")
    return {
        "manifest_sha256": PW0328_SHA256,
        "artifacts_authenticated": 24,
        "windows": 32,
        "q1_events": len(q1_events),
        "sum_A": sum_a,
        "sum_observable_A": observable_a,
        "sum_U": sum_u,
        "categories": by_category,
        "rare_route_evidence": {
            "novel_identity_count": 939,
            "rare_route_unique_identities": 5167,
            "novel_routed_layers": 47,
        },
        "builder_gate8": builder_gate,
        "generation_gate8": {
            "count": 4,
            "all_pass": True,
            "summaries": generation_gate8,
        },
        "prefill_gate8": {
            "count": 4,
            "all_pass": True,
            "summaries": prefill_gate8,
        },
        "authority_complete": True,
    }


def _validate_storage_metric(
    metric: Mapping[str, Any],
    *,
    accepted: int,
    moved: int,
    tps_key: str,
    tps_fraction_key: str,
    wall_key: str = "storage_wall_seconds",
    wall_fraction_key: str = "storage_wall_fraction",
    label: str,
) -> tuple[Fraction, Fraction]:
    recomputed = recompute_storage_summary(moved_bytes=moved, accepted_tokens=accepted)
    wall = _fraction(metric.get(wall_fraction_key), f"{label}: wall")
    tps = _fraction(metric.get(tps_fraction_key), f"{label}: TPS")
    expected_wall = _fraction(recomputed["storage_wall_seconds_fraction"], f"{label}: recomputed wall")
    expected_tps = _fraction(recomputed["storage_tps_fraction"], f"{label}: recomputed TPS")
    _require(wall == expected_wall and tps == expected_tps, f"{label}: exact arithmetic")
    _same_float(metric.get(wall_key), float(wall), f"{label}: wall float")
    _same_float(metric.get(tps_key), float(tps), f"{label}: TPS float")
    return wall, tps


def recompute_pw0329_ceiling(report: Mapping[str, Any]) -> dict[str, Any]:
    scenarios = _sequence(report.get("relaxed_scenarios"), "PW-0329 relaxed scenarios")
    selected = [
        row
        for row in scenarios
        if isinstance(row, dict)
        and row.get("density") == 8
        and row.get("residency_bytes") == RESIDENCY_BYTES
    ]
    _require(len(selected) == 1, "PW-0329 strongest scenario identity")
    scenario = selected[0]
    windows = _sequence(scenario.get("windows"), "PW-0329 strongest windows")
    _require(len(windows) == 32, "PW-0329 strongest window cardinality")
    window_tps: list[Fraction] = []
    category_a: Counter[str] = Counter()
    category_bytes: Counter[str] = Counter()
    sum_a = 0
    sum_bytes = 0
    for index, raw in enumerate(windows):
        window = _mapping(raw, f"PW-0329 window {index}")
        _require(window.get("window_index") == index, f"PW-0329 window order {index}")
        category = window.get("category")
        _require(category in CATEGORIES, f"PW-0329 window category {index}")
        accepted = _integer(window.get("A"), f"PW-0329 window A {index}", minimum=1)
        metric = _mapping(window.get("fractional_relaxed"), f"PW-0329 window metric {index}")
        moved = _integer(metric.get("bytes_moved"), f"PW-0329 window bytes {index}", minimum=1)
        _require(
            metric.get("semantic") == "logical_fractional_relaxed_density_embedding_omitted"
            and metric.get("residency_budget_bytes") == RESIDENCY_BYTES
            and metric.get("unbounded_storage_only") is False
            and _fraction(metric.get("bandwidth_fraction"), f"PW-0329 bandwidth {index}")
            == FAVORABLE_BANDWIDTH,
            f"PW-0329 window semantic {index}",
        )
        _, tps = _validate_storage_metric(
            metric,
            accepted=accepted,
            moved=moved,
            tps_key="optimistic_storage_tps",
            tps_fraction_key="optimistic_storage_tps_fraction",
            label=f"PW-0329 window {index}",
        )
        window_tps.append(tps)
        category_a[str(category)] += accepted
        category_bytes[str(category)] += moved
        sum_a += accepted
        sum_bytes += moved
    _require(sum_a == PW0329_A and sum_bytes == PW0329_BYTES, "PW-0329 strongest aggregate ledger")
    _require(dict(category_a) == EXPECTED_CATEGORY_A, "PW-0329 category A ledger")
    aggregate = _mapping(
        _mapping(scenario.get("metrics"), "PW-0329 strongest metrics")
        .get("fractional_relaxed"),
        "PW-0329 fractional aggregate",
    )
    overall = _mapping(aggregate.get("overall"), "PW-0329 overall")
    _require(
        overall.get("accepted_tokens") == sum_a
        and overall.get("bytes_moved") == sum_bytes
        and overall.get("windows") == 32
        and overall.get("unbounded_storage_only") is False,
        "PW-0329 overall ledger",
    )
    wall, aggregate_tps = _validate_storage_metric(
        overall,
        accepted=sum_a,
        moved=sum_bytes,
        tps_key="optimistic_storage_tps",
        tps_fraction_key="optimistic_storage_tps_fraction",
        label="PW-0329 strongest aggregate",
    )
    p10 = nearest_rank_p10(window_tps)
    reported_p10 = _fraction(
        overall.get("nearest_rank_p10_window_optimistic_storage_tps_fraction"),
        "PW-0329 fourth-lowest window",
    )
    _require(p10 == reported_p10, "PW-0329 fourth-lowest window recomputation")
    _same_float(
        overall.get("nearest_rank_p10_window_optimistic_storage_tps"),
        float(p10),
        "PW-0329 fourth-lowest window float",
    )
    _same_float(float(aggregate_tps), PW0329_AGGREGATE_TPS, "PW-0329 canonical aggregate TPS")
    _same_float(float(p10), PW0329_P10_TPS, "PW-0329 canonical p10 TPS")
    categories = _mapping(aggregate.get("category"), "PW-0329 category metrics")
    _require(set(categories) == set(CATEGORIES), "PW-0329 category metric set")
    category_tps: dict[str, float] = {}
    category_tps_fraction: dict[str, dict[str, int]] = {}
    for category in CATEGORIES:
        metric = _mapping(categories[category], f"PW-0329 {category} aggregate")
        _require(
            metric.get("accepted_tokens") == category_a[category]
            and metric.get("bytes_moved") == category_bytes[category]
            and metric.get("windows") == 8,
            f"PW-0329 {category} aggregate ledger",
        )
        _, tps = _validate_storage_metric(
            metric,
            accepted=category_a[category],
            moved=category_bytes[category],
            tps_key="optimistic_storage_tps",
            tps_fraction_key="optimistic_storage_tps_fraction",
            label=f"PW-0329 {category} aggregate",
        )
        _require(tps > ONE_TPS, f"PW-0329 {category} aggregate not above one")
        category_tps[category] = float(tps)
        category_tps_fraction[category] = fraction_record(tps)
    all_categories_above_one = all(value > 1.0 for value in category_tps.values())
    return {
        "density": 8,
        "joint_residency_bytes": RESIDENCY_BYTES,
        "packing": "fractional_impossible_best",
        "modeled_source_A": sum_a,
        "bytes_moved": sum_bytes,
        "storage_wall_seconds": float(wall),
        "storage_wall_seconds_fraction": fraction_record(wall),
        "aggregate_storage_tps": float(aggregate_tps),
        "aggregate_storage_tps_fraction": fraction_record(aggregate_tps),
        "category_storage_tps": category_tps,
        "category_storage_tps_fraction": category_tps_fraction,
        "fourth_lowest_window_storage_tps": float(p10),
        "fourth_lowest_window_storage_tps_fraction": fraction_record(p10),
        "strict_aggregate_above_one": aggregate_tps > ONE_TPS,
        "strict_all_categories_above_one": all_categories_above_one,
        "strict_fourth_lowest_window_above_one": p10 > ONE_TPS,
        "strict_all": aggregate_tps > ONE_TPS and all_categories_above_one and p10 > ONE_TPS,
        "claim_class": "impossible_best_storage_only_ceiling_not_achieved_tps",
    }


def validate_pw0329(report: Mapping[str, Any]) -> dict[str, Any]:
    decision = _mapping(report.get("decision"), "PW-0329 decision")
    _require(
        report.get("schema_version") == 1
        and report.get("experiment_id") == "PW-0329"
        and report.get("status") == "analytical_joint_residency_bound_complete"
        and report.get("accepted_tokens") == 0
        and report.get("performance_claim") is None
        and report.get("gate8_analyzer_pass") is True
        and decision.get("decision") == "reject_k4_construction_continuation_on_tail"
        and decision.get("precedence_gate") == 2
        and decision.get("work_order") is None
        and decision.get("workflow_disposition") == "rejected",
        "PW-0329 top-level/precedence contract",
    )
    ceiling = recompute_pw0329_ceiling(report)
    _require(
        ceiling["strict_aggregate_above_one"] is True
        and ceiling["strict_all_categories_above_one"] is True
        and ceiling["strict_fourth_lowest_window_above_one"] is False
        and ceiling["strict_all"] is False,
        "PW-0329 tail disposition",
    )
    gate8 = validate_gate8(report.get("safety_snapshots"), label="PW-0329 analyzer")
    return {
        "decision": decision["decision"],
        "precedence_gate": 2,
        "work_order": None,
        "ceiling": ceiling,
        "gate8": gate8,
    }


def validate_pw0330(report: Mapping[str, Any]) -> dict[str, Any]:
    scope = _mapping(report.get("decision_scope"), "PW-0330 decision scope")
    scheduler = _mapping(report.get("scheduler"), "PW-0330 scheduler")
    ceiling = _mapping(report.get("selected_storage_ceiling"), "PW-0330 storage ceiling")
    residency = _mapping(report.get("joint_residency_authority"), "PW-0330 residency authority")
    selected_route = recompute_pw0330_selected_route(report)
    _require(
        report.get("schema_version") == 1
        and report.get("experiment_id") == "PW-0330"
        and report.get("status") == "complete"
        and report.get("decision") == "conditional_hard_storage_rejection"
        and report.get("accepted_tokens") == 0
        and report.get("performance_claim") is None
        and report.get("semantic") == "cyclic_mtp_012_v1",
        "PW-0330 top-level contract",
    )
    _require(
        scope
        == {
            "conditional_on_direct_q32_first_chunk_parity": True,
            "direct_q32_trace_required": True,
            "heads_8_through_30_authorized": False,
            "prefix_authority_rows": 8,
        },
        "PW-0330 conditional scope",
    )
    _require(
        scheduler.get("name") == "cyclic_mtp_012_v1"
        and scheduler.get("A") == selected_route["A"] == PW0330_A
        and scheduler.get("authenticated_prefix_matches") == 3
        and scheduler.get("evaluated_heads") == 4
        and scheduler.get("first_mismatch_index") == 3
        and scheduler.get("prefix_authority_exhausted") is False,
        "PW-0330 scheduler authority",
    )
    _require(
        residency.get("fixed_target_logical_source_bytes") == FIXED_LOGICAL_BYTES
        and residency.get("mtp_only_additional_logical_source_bytes") == MTP_ONLY_BYTES
        and residency.get("source_expert_logical_bytes") == SOURCE_EXPERT_BYTES
        and residency.get("joint_residency_bytes") == RESIDENCY_BYTES
        and residency.get("lm_head_already_in_fixed_target_set") is True
        and residency.get("lm_head_added_again") is False,
        "PW-0330 joint-residency authority",
    )
    miss = max(
        0,
        FIXED_LOGICAL_BYTES
        + MTP_ONLY_BYTES
        + PW0330_IDENTITIES * SOURCE_EXPERT_BYTES
        - RESIDENCY_BYTES,
    )
    _require(miss == PW0330_MISS_BYTES, "PW-0330 exact miss formula")
    summary = recompute_storage_summary(
        moved_bytes=miss,
        accepted_tokens=PW0330_A,
    )
    _require(
        ceiling.get("A") == selected_route["A"] == PW0330_A
        and ceiling.get("N_A") == selected_route["N_A"] == PW0330_IDENTITIES
        and ceiling.get("joint_residency_bytes") == RESIDENCY_BYTES
        and ceiling.get("miss_bytes") == miss
        and ceiling.get("candidate_favorable_at_or_below_one") is True
        and ceiling.get("unbounded_storage_only_ceiling") is False,
        "PW-0330 selected ceiling ledger",
    )
    _same_float(
        ceiling.get("candidate_favorable_tps_ceiling"),
        float(_fraction(summary["storage_tps_fraction"], "PW-0330 recomputed TPS")),
        "PW-0330 favorable TPS",
    )
    _same_float(
        ceiling.get("candidate_favorable_tps_ceiling"),
        PW0330_TPS,
        "PW-0330 canonical TPS",
    )
    gate8 = validate_gate8(report.get("safety_snapshots"), label="PW-0330 runner")
    _require(_mapping(report.get("safety_gate"), "PW-0330 safety gate").get("pass") is True, "PW-0330 Gate 8")
    return {
        "decision": report["decision"],
        "scope": scope,
        "scheduler": scheduler["name"],
        "conditional_A": PW0330_A,
        "unique_layer_expert_records": PW0330_IDENTITIES,
        "selected_prefix_route": selected_route,
        "unavoidable_miss_bytes": miss,
        "storage_wall_seconds": summary["storage_wall_seconds"],
        "storage_wall_seconds_fraction": summary["storage_wall_seconds_fraction"],
        "favorable_storage_tps_ceiling": PW0330_TPS,
        "favorable_storage_tps_ceiling_fraction": summary["storage_tps_fraction"],
        "direct_q32_first_chunk_parity": "unproven_outside_evidence_backed_survivors",
        "claim_class": "conditional_storage_only_ceiling_not_achieved_tps",
        "gate8": gate8,
    }


def recompute_pw0330_selected_route(report: Mapping[str, Any]) -> dict[str, Any]:
    """Recompute the named q32 prefix A, identity set, per-layer rows, and hash."""

    heads = _sequence(report.get("head_results"), "PW-0330 head results")
    _require(
        len(heads) == 4
        and all(isinstance(row, dict) and row.get("head_index") == index for index, row in enumerate(heads)),
        "PW-0330 head-result order",
    )
    matches = [row.get("match") for row in heads]
    _require(all(type(value) is bool for value in matches), "PW-0330 head match schema")
    mismatch_indices = [index for index, value in enumerate(matches) if not value]
    _require(
        mismatch_indices == [3] and all(matches[:3]) and matches[3] is False,
        "PW-0330 accepted-prefix recomputation",
    )
    accepted = mismatch_indices[0] + 1

    selected = _mapping(report.get("selected_prefix_route"), "PW-0330 selected route")
    identity_rows = _sequence(selected.get("identities"), "PW-0330 selected identities")
    pairs: list[tuple[int, int]] = []
    for index, raw in enumerate(identity_rows):
        row = _mapping(raw, f"PW-0330 identity {index}")
        _require(set(row) == {"layer", "expert"}, f"PW-0330 identity schema {index}")
        layer = _integer(row.get("layer"), f"PW-0330 identity layer {index}", minimum=1)
        expert = _integer(row.get("expert"), f"PW-0330 identity expert {index}")
        _require(layer <= 47 and expert < 256, f"PW-0330 identity bounds {index}")
        pairs.append((layer, expert))
    _require(
        len(pairs) == PW0330_IDENTITIES
        and len(set(pairs)) == len(pairs)
        and pairs == sorted(pairs),
        "PW-0330 selected identity set/order",
    )
    rebuilt_rows = [{"layer": layer, "expert": expert} for layer, expert in pairs]
    identity_sha = hashlib.sha256(canonical_json(rebuilt_rows)).hexdigest()
    per_layer = []
    for layer in range(1, 48):
        experts = [expert for row_layer, expert in pairs if row_layer == layer]
        per_layer.append(
            {"layer": layer, "unique_experts": len(experts), "experts": experts}
        )
    _require(
        selected.get("A") == accepted
        and selected.get("N_A") == len(pairs)
        and selected.get("unique_source_expert_bytes") == len(pairs) * SOURCE_EXPERT_BYTES
        and selected.get("identity_sha256") == identity_sha
        and selected.get("per_layer") == per_layer,
        "PW-0330 selected route recomputation",
    )
    table = _sequence(report.get("route_prefix_table"), "PW-0330 route prefix table")
    _require(
        len(table) == 8
        and [row.get("A") for row in table if isinstance(row, dict)] == list(range(1, 9)),
        "PW-0330 route-prefix table order",
    )
    selected_table = _mapping(table[accepted - 1], "PW-0330 selected route table row")
    _require(
        selected_table.get("A") == accepted
        and selected_table.get("N_A") == len(pairs)
        and selected_table.get("identity_sha256") == identity_sha
        and selected_table.get("per_layer_unique_counts")
        == [row["unique_experts"] for row in per_layer]
        and selected_table.get("unique_source_expert_bytes")
        == len(pairs) * SOURCE_EXPERT_BYTES,
        "PW-0330 selected route table binding",
    )
    return {
        "A": accepted,
        "N_A": len(pairs),
        "identity_sha256": identity_sha,
        "per_layer_unique_counts": [row["unique_experts"] for row in per_layer],
        "unique_source_expert_bytes": len(pairs) * SOURCE_EXPERT_BYTES,
        "recomputed_from_explicit_identities": True,
    }


def _validate_pw0331_metric_family(value: Any, *, label: str) -> None:
    rows = _mapping(value, label)
    _require(set(rows) == {"overall", "fit", "validation", "pilot"}, f"{label}: slice set")
    for slice_name, raw in rows.items():
        metric = _mapping(raw, f"{label}: {slice_name}")
        relative_l2 = metric.get("relative_l2")
        maximum_row = metric.get("maximum_row_relative_l2")
        _require(
            type(relative_l2) in (int, float)
            and not isinstance(relative_l2, bool)
            and math.isfinite(float(relative_l2))
            and 0 <= float(relative_l2) < 0.01
            and type(maximum_row) in (int, float)
            and not isinstance(maximum_row, bool)
            and math.isfinite(float(maximum_row))
            and 0 <= float(maximum_row) < 0.05,
            f"{label}: unchanged sliced gate {slice_name}",
        )


def validate_pw0331(report: Mapping[str, Any]) -> dict[str, Any]:
    repeat = _mapping(report.get("fit_repeat_authority"), "PW-0331 fit repeat")
    gates = _mapping(report.get("gates"), "PW-0331 gates")
    stage_b = _mapping(report.get("stage_b"), "PW-0331 Stage B")
    identity = _mapping(report.get("identity_local"), "PW-0331 identity metrics")
    cumulative = _mapping(report.get("cumulative_four_expert"), "PW-0331 cumulative metrics")
    diagnostic = _mapping(
        report.get("position1_error_direction_diagnostic"),
        "PW-0331 attenuation diagnostic",
    )
    _require(
        report.get("schema_version") == 1
        and report.get("experiment_id") == "PW-0331"
        and report.get("status") == "stage_a_pass"
        and report.get("decision") == "authorize_stage_b_zero_and_corrected_layout_controls"
        and report.get("exactness_class") == "L3_modified_expert_weights"
        and report.get("batch_size") == 1
        and report.get("concurrency") == 1
        and report.get("accepted_tokens") == 0
        and report.get("A") == 0
        and report.get("U") == 0
        and report.get("performance_claim") is None
        and report.get("source_replay_exact") is True,
        "PW-0331 top-level contract",
    )
    fit_hashes = repeat.get("fit_authority_sha256")
    _require(
        repeat.get("correction_left_sha256") == PW0331_LEFT_SHA256
        and repeat.get("correction_right_sha256") == PW0331_RIGHT_SHA256
        and isinstance(fit_hashes, list)
        and len(fit_hashes) == 2
        and len(set(fit_hashes)) == 1
        and repeat.get("fresh_process_repeat") is True
        and len(_sequence(repeat.get("process_receipts"), "PW-0331 process receipts")) == 2,
        "PW-0331 repeated factor authority",
    )
    expected_gate_names = {
        "attenuation_sanity",
        "cumulative_final",
        "cumulative_route",
        "identity_final",
        "identity_route",
        "pass",
        "primary_final",
        "primary_route",
        "sliced_and_primary_gates_pass",
    }
    _require(
        gates.get("maximum_relative_l2_exclusive") == 0.01
        and gates.get("maximum_row_relative_l2_exclusive") == 0.05
        and all(gates.get(name) is True for name in expected_gate_names),
        "PW-0331 unchanged gates",
    )
    for prefix, family in (("identity", identity), ("cumulative", cumulative)):
        _validate_pw0331_metric_family(
            family.get("route_candidate_vs_source"),
            label=f"PW-0331 {prefix} route",
        )
        _validate_pw0331_metric_family(
            family.get("final_candidate_vs_source"),
            label=f"PW-0331 {prefix} final",
        )
    route = _mapping(
        cumulative.get("position1_route_candidate_vs_source"),
        "PW-0331 unseen route",
    )
    final = _mapping(
        cumulative.get("position1_final_candidate_vs_source"),
        "PW-0331 unseen final",
    )
    _same_float(route.get("relative_l2"), PW0331_ROUTE_L2, "PW-0331 unseen route L2")
    _same_float(final.get("relative_l2"), PW0331_FINAL_L2, "PW-0331 unseen final L2")
    _require(
        diagnostic.get("semantic") == "position1_f64_root_with_conservative_frozen_floor_v2"
        and diagnostic.get("attenuation_requirement_pass") is True
        and diagnostic.get("frozen_floor_is_conservative") is True
        and float(diagnostic.get("observed_attenuation")) >= float(diagnostic.get("frozen_attenuation_floor"))
        and float(diagnostic.get("frozen_attenuation_floor"))
        >= float(diagnostic.get("authenticated_analytical_alpha_min")),
        "PW-0331 attenuation authority",
    )
    _require(
        stage_b.get("authorized") is True
        and stage_b.get("layout_control_constructed") is False,
        "PW-0331 Stage-B-only authorization",
    )
    claims = _sequence(report.get("claims_excluded"), "PW-0331 excluded claims")
    _require(
        "complete K4 bank" in claims
        and "endpoint TPS" in claims
        and "runtime default" in claims,
        "PW-0331 excluded claim set",
    )
    gate8 = validate_gate8(report.get("safety_snapshots"), label="PW-0331 analyzer")
    return {
        "stage_a_pass": True,
        "local_stage_b_authorized": True,
        "stage_b_executed": False,
        "byte_identical_repeated_factors": True,
        "correction_left_sha256": PW0331_LEFT_SHA256,
        "correction_right_sha256": PW0331_RIGHT_SHA256,
        "unseen_route_relative_l2": PW0331_ROUTE_L2,
        "unseen_final_relative_l2": PW0331_FINAL_L2,
        "all_unchanged_sliced_gates_pass": True,
        "portfolio_construction_disposition": "blocked_by_pw0329_precedence_gate_two",
        "claim_class": "local_modified_expert_correctness_not_k4_bank_or_endpoint_pass",
        "gate8": gate8,
    }


def _validate_pw0332_aggregate(
    summary: Mapping[str, Any],
    events: Sequence[Mapping[str, Any]],
    *,
    include_token_p10: bool,
    label: str,
) -> dict[str, Any]:
    _require(bool(events), f"{label}: empty event set")
    encoded = sum(
        (_fraction(event.get("encoded_moved_bytes_fraction"), f"{label}: event encoded") for event in events),
        Fraction(0),
    )
    logical = sum(int(event["logical_moved_bytes"]) for event in events)
    misses = sum(int(event["misses"]) for event in events)
    wall = encoded / FAVORABLE_BANDWIDTH
    tps = Fraction(len(events), 1) / wall
    _require(
        summary.get("token_events") == len(events)
        and summary.get("misses") == misses
        and summary.get("logical_moved_bytes") == logical
        and _fraction(summary.get("encoded_moved_bytes_fraction"), f"{label}: encoded") == encoded
        and _fraction(summary.get("storage_wall_seconds_fraction"), f"{label}: wall") == wall
        and _fraction(summary.get("storage_tps_fraction"), f"{label}: TPS") == tps,
        f"{label}: aggregate exact ledger",
    )
    _same_float(summary.get("encoded_moved_bytes"), float(encoded), f"{label}: encoded float")
    _same_float(summary.get("storage_wall_seconds"), float(wall), f"{label}: wall float")
    _same_float(summary.get("storage_tps"), float(tps), f"{label}: TPS float")
    result: dict[str, Any] = {
        "token_events": len(events),
        "misses": misses,
        "logical_moved_bytes": logical,
        "encoded_moved_bytes": float(encoded),
        "encoded_moved_bytes_fraction": fraction_record(encoded),
        "storage_wall_seconds": float(wall),
        "storage_wall_seconds_fraction": fraction_record(wall),
        "storage_tps": float(tps),
        "storage_tps_fraction": fraction_record(tps),
    }
    if include_token_p10:
        token_tps = [
            _fraction(event.get("storage_tps_fraction"), f"{label}: event TPS")
            for event in events
        ]
        p10 = nearest_rank_p10(token_tps)
        _require(
            summary.get("nearest_rank_p10_rank") == math.ceil(len(events) / 10)
            and _fraction(
                summary.get("nearest_rank_p10_token_storage_tps_fraction"),
                f"{label}: token p10",
            )
            == p10,
            f"{label}: token p10 exact ledger",
        )
        _same_float(
            summary.get("nearest_rank_p10_token_storage_tps"),
            float(p10),
            f"{label}: token p10 float",
        )
        result["token_p10_storage_tps"] = float(p10)
        result["token_p10_storage_tps_fraction"] = fraction_record(p10)
    return result


def recompute_pw0332_absolute_floor(scenario: Mapping[str, Any]) -> dict[str, Any]:
    _require(scenario.get("scenario") == "absolute_floor_all_fp8", "PW-0332 absolute scenario identity")
    layout = _mapping(scenario.get("layout"), "PW-0332 absolute layout")
    encoded_record = _fraction(
        layout.get("encoded_expert_bytes_fraction"),
        "PW-0332 absolute encoded expert",
    )
    _require(
        layout.get("hard_kill_authority") is True
        and layout.get("expert_capacity") == 250
        and layout.get("fixed_object_count") == 381
        and layout.get("fixed_logical_bytes") == FIXED_LOGICAL_BYTES
        and layout.get("residency_bytes") == RESIDENCY_BYTES
        and layout.get("fixed_set_pinned") is True
        and layout.get("fractional_encoded_records_granted") is True
        and encoded_record == Fraction(44_063_235, 2),
        "PW-0332 absolute layout",
    )
    categories = _mapping(scenario.get("categories"), "PW-0332 absolute categories")
    _require(set(categories) == set(CATEGORIES), "PW-0332 absolute category set")
    all_events: list[dict[str, Any]] = []
    category_summaries: dict[str, Any] = {}
    for category in CATEGORIES:
        category_row = _mapping(categories[category], f"PW-0332 {category}")
        events = _sequence(category_row.get("token_events"), f"PW-0332 {category} events")
        _require(
            category_row.get("category_reset") is True
            and category_row.get("modeled_source_A") == EXPECTED_CATEGORY_A[category]
            and len(events) == EXPECTED_CATEGORY_A[category],
            f"PW-0332 {category} event cardinality",
        )
        for index, raw in enumerate(events):
            event = _mapping(raw, f"PW-0332 {category} event {index}")
            misses = _integer(event.get("misses"), f"PW-0332 {category} misses {index}")
            encoded = misses * encoded_record
            logical = misses * SOURCE_EXPERT_BYTES
            wall = encoded / FAVORABLE_BANDWIDTH
            tps = FAVORABLE_BANDWIDTH / encoded
            _require(
                event.get("category") == category
                and event.get("layer_demands") == 47
                and event.get("logical_moved_bytes") == logical
                and _fraction(
                    event.get("encoded_moved_bytes_fraction"),
                    f"PW-0332 {category} encoded {index}",
                )
                == encoded
                and _fraction(
                    event.get("storage_wall_seconds_fraction"),
                    f"PW-0332 {category} wall {index}",
                )
                == wall
                and _fraction(
                    event.get("storage_tps_fraction"),
                    f"PW-0332 {category} TPS {index}",
                )
                == tps,
                f"PW-0332 {category} event ledger {index}",
            )
            _same_float(event.get("encoded_moved_bytes"), float(encoded), f"PW-0332 {category} event encoded {index}")
            _same_float(event.get("storage_wall_seconds"), float(wall), f"PW-0332 {category} event wall {index}")
            _same_float(event.get("storage_tps"), float(tps), f"PW-0332 {category} event TPS {index}")
        category_summaries[category] = _validate_pw0332_aggregate(
            _mapping(category_row.get("aggregate"), f"PW-0332 {category} aggregate"),
            events,
            include_token_p10=True,
            label=f"PW-0332 {category} aggregate",
        )
        all_events.extend(events)
    overall = _mapping(scenario.get("overall"), "PW-0332 absolute overall")
    overall_summary = _validate_pw0332_aggregate(
        overall,
        all_events,
        include_token_p10=True,
        label="PW-0332 absolute overall",
    )
    windows = _sequence(scenario.get("windows"), "PW-0332 absolute windows")
    _require(len(windows) == 32, "PW-0332 absolute window cardinality")
    by_corpus: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for event in all_events:
        by_corpus[int(event["corpus_index"])].append(event)
    window_tps: list[Fraction] = []
    for raw in windows:
        window = _mapping(raw, "PW-0332 absolute window")
        corpus_index = _integer(window.get("corpus_index"), "PW-0332 corpus index")
        selected = by_corpus[corpus_index]
        _require(
            window.get("category") in CATEGORIES
            and window.get("modeled_source_A") == len(selected),
            "PW-0332 window identity/A",
        )
        aggregate = _mapping(window.get("aggregate"), "PW-0332 window aggregate")
        _validate_pw0332_aggregate(
            aggregate,
            selected,
            include_token_p10=True,
            label=f"PW-0332 window {corpus_index}",
        )
        window_tps.append(_fraction(aggregate.get("storage_tps_fraction"), "PW-0332 window TPS"))
    _require(set(by_corpus) == set(range(32)), "PW-0332 window corpus indices")
    window_p10 = nearest_rank_p10(window_tps)
    _require(
        overall.get("window_nearest_rank_p10_rank") == 4
        and _fraction(
            overall.get("fourth_lowest_window_storage_tps_fraction"),
            "PW-0332 fourth-lowest window",
        )
        == window_p10,
        "PW-0332 fourth-lowest window recomputation",
    )
    _same_float(
        overall.get("fourth_lowest_window_storage_tps"),
        float(window_p10),
        "PW-0332 fourth-lowest window float",
    )
    _require(
        overall_summary["misses"] == PW0332_ABSOLUTE_MISSES
        and overall_summary["encoded_moved_bytes"] == PW0332_ABSOLUTE_ENCODED_BYTES,
        "PW-0332 canonical absolute miss/byte floor",
    )
    _same_float(overall_summary["storage_wall_seconds"], PW0332_ABSOLUTE_WALL, "PW-0332 canonical wall")
    _same_float(overall_summary["storage_tps"], PW0332_ABSOLUTE_TPS, "PW-0332 canonical aggregate TPS")
    _same_float(overall_summary["token_p10_storage_tps"], PW0332_TOKEN_P10, "PW-0332 canonical token p10")
    _same_float(float(window_p10), PW0332_WINDOW_P10, "PW-0332 canonical window p10")
    return {
        **overall_summary,
        "capacity": 250,
        "category": category_summaries,
        "fourth_lowest_window_storage_tps": float(window_p10),
        "fourth_lowest_window_storage_tps_fraction": fraction_record(window_p10),
        "claim_class": "absolute_zero_escape_storage_only_floor_not_achieved_tps",
    }


def _recompute_fixed_census(storage: Mapping[str, Any]) -> dict[str, Any]:
    fixed = _mapping(storage.get("fixed"), "PW-0332 fixed census")
    objects = _sequence(fixed.get("objects"), "PW-0332 fixed objects")
    _require(len(objects) == 381, "PW-0332 fixed object count")
    logical = sum(_integer(row.get("logical_bytes"), "fixed logical bytes", minimum=1) for row in objects)
    allocated = sum(_integer(row.get("allocated_bytes"), "fixed allocated bytes", minimum=1) for row in objects)
    by_dtype: dict[str, dict[str, int]] = {}
    for dtype in ("F8_E4M3", "BF16", "F32"):
        selected = [row for row in objects if row.get("dtype") == dtype]
        by_dtype[dtype] = {
            "objects": len(selected),
            "logical_bytes": sum(int(row["logical_bytes"]) for row in selected),
        }
    largest = max(int(row["logical_bytes"]) for row in objects)
    _require(
        logical == 7_743_236_992
        and allocated == 7_745_470_464
        and by_dtype
        == {
            "F8_E4M3": {"objects": 51, "logical_bytes": 3_073_376_256},
            "BF16": {"objects": 185, "logical_bytes": 4_471_927_680},
            "F32": {"objects": 145, "logical_bytes": 197_933_056},
        }
        and largest == 1_249_902_592,
        "PW-0332 fixed-census recomputation",
    )
    return {
        "objects": len(objects),
        "logical_bytes": logical,
        "allocated_bytes": allocated,
        "dtype_census": by_dtype,
        "largest_object_bytes": largest,
    }


def _validate_pw0332_codec_floor(codec: Mapping[str, Any]) -> dict[str, Any]:
    codes = _integer(codec.get("block_codes"), "PW-0332 codec codes", minimum=1)
    escapes = _integer(codec.get("observed_witness_escapes"), "PW-0332 codec escapes")
    zero_bytes = math.ceil((7 * codes + 4 * 0 + 28) / 8)
    observed_bytes = math.ceil((7 * codes + 4 * escapes + 28) / 8)
    _require(
        codes == 16_384
        and escapes == 341
        and zero_bytes == 14_340
        and observed_bytes == 14_510
        and codec.get("zero_escape_bytes") == zero_bytes
        and codec.get("observed_bytes") == observed_bytes
        and _fraction(codec.get("zero_escape_ratio_fraction"), "PW-0332 zero ratio")
        == Fraction(zero_bytes, codes)
        and _fraction(codec.get("observed_ratio_fraction"), "PW-0332 observed ratio")
        == Fraction(observed_bytes, codes)
        and codec.get("ratio_below_floor_possible") is False,
        "PW-0332 codec-floor formula",
    )
    return {
        "block_codes": codes,
        "zero_escape_bytes": zero_bytes,
        "observed_witness_escapes": escapes,
        "observed_bytes": observed_bytes,
        "zero_escape_ratio": float(Fraction(zero_bytes, codes)),
        "ratio_below_floor_possible": False,
    }


def validate_pw0332_codec_replay(
    report: Mapping[str, Any], local_replay: Mapping[str, Any]
) -> dict[str, Any]:
    """Bind PW-0332 to a fresh 480-block checkpoint replay of PW-0324."""

    authority = _mapping(report.get("authority"), "PW-0332 authority")
    embedded = _mapping(authority.get("pw0324"), "PW-0332 embedded codec replay")
    expected_keys = {
        "file",
        "sha256",
        "local_exact_codec_replication_authenticated",
        "checkpoint_index_sha256",
        "quantization_blocks",
        "observed_minimum_top7_ratio",
        "sample_is_routed_full_model_codec_census",
        "limitation",
        "local_exact_codec_replay_equal_to_canonical",
    }
    _require(set(local_replay) == expected_keys, "PW-0332 local codec replay schema")
    _require(
        local_replay.get("sha256") == PW0324_SHA256
        and local_replay.get("checkpoint_index_sha256") == CHECKPOINT_INDEX_SHA256
        and local_replay.get("quantization_blocks") == 480
        and local_replay.get("local_exact_codec_replication_authenticated") is True
        and local_replay.get("local_exact_codec_replay_equal_to_canonical") is True
        and local_replay.get("sample_is_routed_full_model_codec_census") is False
        and local_replay.get("observed_minimum_top7_ratio") == 0.8856201171875
        and embedded == local_replay,
        "PW-0332 independent 480-block codec replay",
    )
    return {
        "source_analysis_sha256": PW0324_SHA256,
        "checkpoint_index_sha256": CHECKPOINT_INDEX_SHA256,
        "quantization_blocks_replayed": 480,
        "observed_minimum_top7_ratio": 0.8856201171875,
        "byte_equal_to_canonical": True,
        "sample_is_routed_full_model_codec_census": False,
    }


def replay_pw0332_cache_oracle(
    *,
    canonical_scenarios: Sequence[Mapping[str, Any]],
    canonical_decision: Mapping[str, Any],
    canonical_dominance: Mapping[str, Any],
    windows: Sequence[Mapping[str, Any]],
    traces: Mapping[str, Sequence[Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Independently derive capacities and all Belady ledgers from q1 demands."""

    layouts = [pw0332_scenario_layout(scenario) for scenario in PW0332_SCENARIOS]
    validate_pw0332_layouts(layouts)
    _require(
        [layout["expert_capacity"] for layout in layouts] == [204, 230, 250],
        "PW-0332 replayed capacities",
    )
    replayed = [
        analyze_pw0332_scenario(scenario, layout, windows, traces)
        for scenario, layout in zip(PW0332_SCENARIOS, layouts, strict=True)
    ]
    _require(len(canonical_scenarios) == len(replayed), "PW-0332 scenario cardinality")
    for scenario, expected, actual in zip(
        PW0332_SCENARIOS,
        canonical_scenarios,
        replayed,
        strict=True,
    ):
        _require(actual == expected, f"PW-0332 independent oracle replay mismatch: {scenario.name}")
    by_name = {
        scenario.name: result for scenario, result in zip(PW0332_SCENARIOS, replayed, strict=True)
    }
    replayed_dominance = validate_pw0332_scenario_dominance(by_name)
    replayed_decision = pw0332_disposition(by_name)
    _require(
        replayed_dominance == canonical_dominance,
        "PW-0332 replayed scenario dominance",
    )
    _require(replayed_decision == canonical_decision, "PW-0332 replayed disposition")
    summary: dict[str, Any] = {
        "implementation": "shared_indexed_belady_plus_independent_replay",
        "q1_demand_events": 232,
        "scenario_order": [scenario.name for scenario in PW0332_SCENARIOS],
        "capacities_recomputed": [layout["expert_capacity"] for layout in layouts],
        "scenarios": {},
        "full_canonical_ledgers_equal": True,
        "dominance_recomputed": replayed_dominance,
        "decision_recomputed": replayed_decision["decision"],
    }
    for result in replayed:
        overall = _mapping(result.get("overall"), "PW-0332 replay overall")
        summary["scenarios"][result["scenario"]] = {
            "capacity": result["layout"]["expert_capacity"],
            "misses": overall["misses"],
            "encoded_moved_bytes_fraction": overall["encoded_moved_bytes_fraction"],
            "storage_wall_seconds_fraction": overall["storage_wall_seconds_fraction"],
            "storage_tps_fraction": overall["storage_tps_fraction"],
            "token_p10_storage_tps_fraction": overall[
                "nearest_rank_p10_token_storage_tps_fraction"
            ],
            "fourth_lowest_window_storage_tps_fraction": overall[
                "fourth_lowest_window_storage_tps_fraction"
            ],
        }
    _require(
        [summary["scenarios"][scenario.name]["misses"] for scenario in PW0332_SCENARIOS]
        == [53_040, 50_743, 49_122],
        "PW-0332 replayed miss counts",
    )
    return replayed, summary


def validate_pw0332(
    report: Mapping[str, Any],
    live_pw0328: Mapping[str, Any],
    local_codec_replay: Mapping[str, Any],
) -> dict[str, Any]:
    validate_pw0332_report_schema(report)
    _require(report.get("commit") == PW0332_COMMIT, "PW-0332 execution commit")
    authority = _mapping(report.get("authority"), "PW-0332 authority")
    contract = _mapping(authority.get("contract"), "PW-0332 contract authority")
    _require(
        contract.get("git_blob") == PW0332_CONTRACT_GIT_BLOB
        and contract.get("sha256") == PW0332_CONTRACT_SHA256
        and authority.get("authority_complete") is True,
        "PW-0332 contract authority",
    )
    embedded_pw0328 = _mapping(authority.get("pw0328"), "PW-0332 embedded PW-0328")
    live_windows, live_traces = normalize_corpus_authority(live_pw0328)
    embedded_windows, embedded_traces = normalize_corpus_authority(embedded_pw0328)
    live_demand_sha = demand_stream_sha256(live_traces)
    embedded_demand_sha = demand_stream_sha256(embedded_traces)
    _require(
        live_demand_sha == PW0328_Q1_DEMAND_SHA256
        and embedded_demand_sha == live_demand_sha
        and live_windows == embedded_windows,
        "PW-0332 independent q1 demand reconstruction",
    )
    codec_replay = validate_pw0332_codec_replay(report, local_codec_replay)
    storage = _mapping(authority.get("storage"), "PW-0332 storage authority")
    storage_summary = validate_storage_authority(storage)
    fixed_census = _recompute_fixed_census(storage)
    identities = _mapping(storage.get("identities"), "PW-0332 storage identities")
    _require(
        identities.get("revision") == CHECKPOINT_REVISION
        and identities.get("checkpoint_verification_sha256") == CHECKPOINT_RECEIPT_SHA256
        and identities.get("tensor_index_sha256") == CHECKPOINT_INDEX_SHA256,
        "PW-0332 checkpoint identity",
    )
    codec = _validate_pw0332_codec_floor(
        _mapping(report.get("codec_floor"), "PW-0332 codec floor")
    )
    scenarios = _sequence(report.get("scenarios"), "PW-0332 scenarios")
    _require(
        [row.get("scenario") for row in scenarios]
        == ["uncompressed", "observed_expert_only", "absolute_floor_all_fp8"]
        and [row.get("layout", {}).get("expert_capacity") for row in scenarios]
        == [204, 230, 250],
        "PW-0332 scenario capacities/order",
    )
    decision = _mapping(report.get("decision"), "PW-0332 decision")
    canonical_dominance = _mapping(
        report.get("scenario_dominance"), "PW-0332 scenario dominance"
    )
    replayed_scenarios, oracle_replay = replay_pw0332_cache_oracle(
        canonical_scenarios=scenarios,
        canonical_decision=decision,
        canonical_dominance=canonical_dominance,
        windows=live_windows,
        traces=live_traces,
    )
    absolute = recompute_pw0332_absolute_floor(replayed_scenarios[2])
    scenario_gates = _mapping(decision.get("scenario_gates"), "PW-0332 scenario gates")
    recomputed_gates = {
        row["scenario"]: pw0332_strict_gates(row) for row in scenarios
    }
    _require(scenario_gates == recomputed_gates, "PW-0332 strict-gate recomputation")
    for scenario, gates in recomputed_gates.items():
        leaves = [
            gates["overall_aggregate_strictly_above_one"],
            *gates["required_category_aggregates_strictly_above_one"].values(),
            gates["corpus_token_p10_strictly_above_one"],
            *gates["category_token_p10_strictly_above_one"].values(),
            gates["fourth_lowest_window_strictly_above_one"],
            gates["all_strict_gates_pass"],
        ]
        _require(all(value is False for value in leaves), f"PW-0332 strict gate unexpectedly survives: {scenario}")
    _require(
        decision.get("decision") == "reject_exact_top7_token_cache_oracle"
        and decision.get("analytical_survivor") is False
        and decision.get("decoder_authorized") is False
        and decision.get("runtime_default_changed") is False
        and report.get("gate8_analyzer_pass") is True
        and report.get("accepted_tokens") == 0
        and report.get("A") == 0
        and report.get("U") == 0
        and report.get("performance_claim") is None,
        "PW-0332 terminal disposition",
    )
    dominance = canonical_dominance
    encoded = [
        _fraction(row["overall"]["encoded_moved_bytes_fraction"], "PW-0332 scenario bytes")
        for row in scenarios
    ]
    _require(
        dominance.get("pass") is True
        and dominance.get("capacity_order") == [204, 230, 250]
        and encoded[2] <= encoded[1] <= encoded[0],
        "PW-0332 scenario dominance",
    )
    gate8 = validate_gate8(report.get("safety_snapshots"), label="PW-0332 analyzer")
    return {
        "decision": decision["decision"],
        "analytical_survivor": False,
        "decoder_authorized": False,
        "runtime_default_changed": False,
        "contract_git_blob": PW0332_CONTRACT_GIT_BLOB,
        "contract_sha256": PW0332_CONTRACT_SHA256,
        "q1_demand_stream_sha256": live_demand_sha,
        "q1_demand_recomputed_independently": True,
        "fixed_census": fixed_census,
        "storage_summary": storage_summary,
        "codec_floor": codec,
        "codec_replay": codec_replay,
        "scenario_capacities": [204, 230, 250],
        "oracle_replay": oracle_replay,
        "absolute_floor": absolute,
        "strict_gates": recomputed_gates,
        "gate8": gate8,
    }


def validate_throughput_model(
    model: Mapping[str, Any],
    *,
    pw0329: Mapping[str, Any],
    pw0330: Mapping[str, Any],
    pw0331: Mapping[str, Any],
    pw0332: Mapping[str, Any],
) -> dict[str, Any]:
    """Reconcile every PW-0333 decision-bearing throughput-model row."""

    _require(
        model.get("schema_version") == 1
        and model.get("source_revision") == CHECKPOINT_REVISION,
        "throughput model identity",
    )
    constants = _mapping(model.get("constants"), "throughput model constants")
    measured = measured_lower_milestones(model)

    k4 = _mapping(
        constants.get("pw0329_corrected_k4_joint_residency_bound"),
        "throughput model PW-0329",
    )
    k4_ceiling = _mapping(pw0329.get("ceiling"), "recomputed PW-0329 ceiling")
    _require(
        k4.get("target_accepted_tps") == 1.0
        and k4.get("modeled_source_A") == PW0329_A
        and math.isclose(
            float(k4.get("modeled_source_U")),
            142.71808510638297,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
        and k4.get("strongest_density") == 8
        and k4.get("strongest_joint_residency_bytes") == RESIDENCY_BYTES
        and k4.get("candidate_favorable_bytes_moved") == PW0329_BYTES
        and k4.get("decision_precedence_gate") == 2
        and k4.get("work_order_emitted") is False
        and k4.get("batch_size") == 1
        and k4.get("concurrency") == 1
        and k4.get("accepted_tokens") == 0
        and "not_achieved_tps" in str(k4.get("status")),
        "throughput model PW-0329 ledger",
    )
    _same_float(
        k4.get("candidate_favorable_storage_wall_seconds"),
        float(k4_ceiling["storage_wall_seconds"]),
        "throughput model PW-0329 wall",
    )
    _same_float(
        k4.get("candidate_favorable_aggregate_storage_only_tps"),
        float(k4_ceiling["aggregate_storage_tps"]),
        "throughput model PW-0329 aggregate",
    )
    _same_float(
        k4.get("nearest_rank_p10_window_storage_only_tps"),
        float(k4_ceiling["fourth_lowest_window_storage_tps"]),
        "throughput model PW-0329 tail",
    )
    model_k4_categories = _mapping(
        k4.get("candidate_favorable_category_storage_only_tps"),
        "throughput model PW-0329 categories",
    )
    _require(set(model_k4_categories) == set(CATEGORIES), "throughput model PW-0329 category set")
    for category in CATEGORIES:
        _same_float(
            model_k4_categories[category],
            float(k4_ceiling["category_storage_tps"][category]),
            f"throughput model PW-0329 category {category}",
        )

    cyclic = _mapping(
        constants.get("pw0330_cyclic_mtp_q32_prefix_falsifier"),
        "throughput model PW-0330",
    )
    _require(
        cyclic.get("scheduler") == "cyclic_mtp_012_v1"
        and cyclic.get("verifier_width_hypothesis") == 32
        and cyclic.get("available_target_authority_rows") == 8
        and cyclic.get("evaluated_draft_heads") == 4
        and cyclic.get("matched_draft_heads") == 3
        and cyclic.get("first_mismatch_index") == 3
        and cyclic.get("conditional_A_under_first_chunk_parity") == PW0330_A
        and cyclic.get("selected_prefix_unique_layer_expert_records") == PW0330_IDENTITIES
        and cyclic.get("fixed_target_logical_source_bytes") == FIXED_LOGICAL_BYTES
        and cyclic.get("mtp_only_additional_logical_source_bytes") == MTP_ONLY_BYTES
        and cyclic.get("source_expert_logical_bytes") == SOURCE_EXPERT_BYTES
        and cyclic.get("joint_residency_bytes") == RESIDENCY_BYTES
        and cyclic.get("unavoidable_miss_bytes") == PW0330_MISS_BYTES
        and cyclic.get("batch_size") == 1
        and cyclic.get("concurrency") == 1
        and cyclic.get("accepted_tokens") == 0
        and "conditional_hard_storage_rejection_for_named" in str(cyclic.get("status"))
        and "direct_q32_first_chunk_parity_unproven" in str(cyclic.get("status"))
        and "not_endpoint_tps" in str(cyclic.get("status")),
        "throughput model PW-0330 ledger",
    )
    _same_float(
        cyclic.get("candidate_favorable_storage_only_tps_ceiling"),
        float(pw0330["favorable_storage_tps_ceiling"]),
        "throughput model PW-0330 ceiling",
    )

    repair = _mapping(
        constants.get("pw0331_byte_neutral_k4_rank1_stage_a"),
        "throughput model PW-0331",
    )
    _require(
        repair.get("exactness_class") == "L3_modified_expert_weights"
        and repair.get("correction_left_sha256") == PW0331_LEFT_SHA256
        and repair.get("correction_right_sha256") == PW0331_RIGHT_SHA256
        and repair.get("all_stage_a_fidelity_gates_pass") is True
        and repair.get("stage_b_executed") is False
        and repair.get("batch_size") == 1
        and repair.get("concurrency") == 1
        and repair.get("accepted_tokens") == 0
        and "construction_stopped_by_failed_pw0329_higher_precedence_tail_gate"
        in str(repair.get("status"))
        and "no_endpoint_tps_or_runtime_default" in str(repair.get("status")),
        "throughput model PW-0331 ledger",
    )
    _same_float(
        repair.get("frozen_position1_corrected_route_relative_l2"),
        float(pw0331["unseen_route_relative_l2"]),
        "throughput model PW-0331 route L2",
    )
    _same_float(
        repair.get("frozen_position1_corrected_final_relative_l2"),
        float(pw0331["unseen_final_relative_l2"]),
        "throughput model PW-0331 final L2",
    )

    cache = _mapping(
        constants.get("pw0332_exact_top7_token_cache_oracle"),
        "throughput model PW-0332",
    )
    absolute = _mapping(pw0332.get("absolute_floor"), "recomputed PW-0332 floor")
    _require(
        cache.get("target_accepted_tps") == 1.0
        and cache.get("modeled_source_A") == 232
        and math.isclose(
            float(cache.get("modeled_source_U")),
            142.71808510638297,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
        and cache.get("residency_bytes") == RESIDENCY_BYTES
        and cache.get("scenario_capacities")
        == {"uncompressed": 204, "observed_expert_only": 230, "absolute_floor_all_fp8": 250}
        and cache.get("absolute_floor_misses") == int(absolute["misses"])
        and cache.get("absolute_floor_encoded_bytes_moved") == int(absolute["encoded_moved_bytes"])
        and cache.get("all_absolute_floor_strict_gates_pass") is False
        and cache.get("batch_size") == 1
        and cache.get("concurrency") == 1
        and cache.get("accepted_tokens") == 0
        and "rejected_exact_top7_token_cache_composition" in str(cache.get("status"))
        and "no_decoder_or_runtime_default" in str(cache.get("status")),
        "throughput model PW-0332 ledger",
    )
    _same_float(
        cache.get("zero_escape_ratio"),
        float(pw0332["codec_floor"]["zero_escape_ratio"]),
        "throughput model PW-0332 zero-escape ratio",
    )
    for key, result_key, label in (
        ("absolute_floor_storage_wall_seconds", "storage_wall_seconds", "wall"),
        ("absolute_floor_aggregate_storage_only_tps", "storage_tps", "aggregate"),
        ("absolute_floor_corpus_token_p10_storage_only_tps", "token_p10_storage_tps", "token p10"),
        ("absolute_floor_fourth_lowest_window_storage_only_tps", "fourth_lowest_window_storage_tps", "window p10"),
    ):
        _same_float(
            cache.get(key),
            float(absolute[result_key]),
            f"throughput model PW-0332 {label}",
        )
    cache_categories = _mapping(
        cache.get("absolute_floor_category_storage_only_tps"),
        "throughput model PW-0332 categories",
    )
    cache_misses = _mapping(
        cache.get("absolute_floor_category_misses"),
        "throughput model PW-0332 category misses",
    )
    _require(
        set(cache_categories) == set(cache_misses) == set(CATEGORIES),
        "throughput model PW-0332 category set",
    )
    for category in CATEGORIES:
        category = str(category)
        category_floor = _mapping(absolute["category"][category], f"PW-0332 {category} floor")
        _require(
            cache_misses[category] == category_floor["misses"],
            f"throughput model PW-0332 {category} misses",
        )
        _same_display_float(
            cache_categories[category],
            float(category_floor["storage_tps"]),
            f"throughput model PW-0332 {category} TPS",
        )

    return {
        "sha256": THROUGHPUT_MODEL_SHA256,
        "source_revision": CHECKPOINT_REVISION,
        "measured_lower_milestones": measured,
        "reconciled_constants": [
            "pw0329_corrected_k4_joint_residency_bound",
            "pw0330_cyclic_mtp_q32_prefix_falsifier",
            "pw0331_byte_neutral_k4_rank1_stage_a",
            "pw0332_exact_top7_token_cache_oracle",
        ],
        "reconciliation_pass": True,
    }


REQUIRED_REOPENED_BRANCH_DISPOSITIONS = {
    "corrected_k4_density8_r12": "rejected_on_required_pw0329_window_tail_gate_two",
    "pw0331_byte_neutral_rank1_repair": "blocked_on_failed_higher_precedence_pw0329_gate_two",
    "pw0330_named_cyclic_mtp_q32": "conditional_below_one_for_named_schedule",
    "pw0332_exact_top7_token_cache_oracle": "rejected_on_absolute_zero_escape_floor",
}

REQUIRED_CLOSURE_CONDITIONS = (
    "no_authenticated_target_spec_sustained_result_reaches_one_tps",
    "pw0329_current_k4_portfolio_fails_strict_window_tail_gate",
    "pw0330_named_schedule_is_conditionally_below_one_and_prior_portfolio_has_no_other_survivor",
    "pw0332_absolute_floor_fails_at_least_one_strict_one_tps_gate",
    "all_reopened_authenticated_branches_have_decisive_dispositions",
    "no_mode_has_complete_full_capability_fidelity_evidence_for_promotion",
    "all_authorities_formulas_schemas_claim_labels_and_gate8_close",
    "companion_hardware_contributes_zero_premise",
)


def pw0332_all_strict_gates_survive(gates: Any) -> bool | None:
    """Return true/false for a complete gate row, or None for stale structure."""

    if not isinstance(gates, dict):
        return None
    category_aggregate = gates.get("required_category_aggregates_strictly_above_one")
    category_p10 = gates.get("category_token_p10_strictly_above_one")
    if (
        not isinstance(category_aggregate, dict)
        or not isinstance(category_p10, dict)
        or set(category_aggregate) != set(CATEGORIES)
        or set(category_p10) != set(CATEGORIES)
    ):
        return None
    leaves = [
        gates.get("overall_aggregate_strictly_above_one"),
        *(category_aggregate[category] for category in CATEGORIES),
        gates.get("corpus_token_p10_strictly_above_one"),
        *(category_p10[category] for category in CATEGORIES),
        gates.get("fourth_lowest_window_strictly_above_one"),
    ]
    if any(type(value) is not bool for value in leaves):
        return None
    survives = all(leaves)
    if type(gates.get("all_strict_gates_pass")) is not bool:
        return None
    if gates["all_strict_gates_pass"] != survives:
        return None
    return survives


def build_reopened_branch_dispositions(
    *,
    pw0324: Mapping[str, Any],
    pw0329: Mapping[str, Any],
    pw0330: Mapping[str, Any],
    pw0331: Mapping[str, Any],
    pw0332: Mapping[str, Any],
) -> dict[str, Any]:
    _require(
        pw0324.get("unreconciled_proposer_family_survivors") == 0
        and pw0324.get("authenticated_proposer_family_states")
        == PW0324_PROPOSER_FAMILY_STATES,
        "PW-0324 proposer-family survivor ledger",
    )
    _require(
        pw0329.get("precedence_gate") == 2
        and pw0329.get("work_order") is None
        and _mapping(pw0329.get("ceiling"), "PW-0329 ceiling").get(
            "strict_fourth_lowest_window_above_one"
        )
        is False,
        "PW-0329 branch disposition",
    )
    _require(
        pw0330.get("decision") == "conditional_hard_storage_rejection"
        and pw0330.get("direct_q32_first_chunk_parity")
        == "unproven_outside_evidence_backed_survivors",
        "PW-0330 branch disposition",
    )
    _require(
        pw0331.get("stage_a_pass") is True
        and pw0331.get("stage_b_executed") is False
        and pw0331.get("portfolio_construction_disposition")
        == "blocked_by_pw0329_precedence_gate_two",
        "PW-0331 branch disposition",
    )
    _require(
        pw0332.get("analytical_survivor") is False
        and pw0332.get("decoder_authorized") is False,
        "PW-0332 branch disposition",
    )
    return {
        "corrected_k4_density8_r12": {
            "disposition": REQUIRED_REOPENED_BRANCH_DISPOSITIONS[
                "corrected_k4_density8_r12"
            ],
            "decisive": True,
            "within_current_authenticated_portfolio": True,
            "basis": "PW-0329 precedence gate two; fourth-lowest window at or below one",
        },
        "pw0331_byte_neutral_rank1_repair": {
            "disposition": REQUIRED_REOPENED_BRANCH_DISPOSITIONS[
                "pw0331_byte_neutral_rank1_repair"
            ],
            "decisive": True,
            "within_current_authenticated_portfolio": True,
            "basis": "Stage A is local correctness only; Stage B was not executed",
        },
        "pw0330_named_cyclic_mtp_q32": {
            "disposition": REQUIRED_REOPENED_BRANCH_DISPOSITIONS[
                "pw0330_named_cyclic_mtp_q32"
            ],
            "decisive": True,
            "within_current_authenticated_portfolio": True,
            "basis": "0.6281149080724167 storage-only ceiling if direct-q32 first-chunk parity holds",
        },
        "pw0332_exact_top7_token_cache_oracle": {
            "disposition": REQUIRED_REOPENED_BRANCH_DISPOSITIONS[
                "pw0332_exact_top7_token_cache_oracle"
            ],
            "decisive": True,
            "within_current_authenticated_portfolio": True,
            "basis": "absolute zero-escape aggregate and required tails are below one",
        },
    }


def derive_closure(
    conditions: Mapping[str, Any],
    reopened_branches: Mapping[str, Any],
    *,
    pw0332_absolute_gates: Any,
) -> dict[str, Any]:
    """Derive the only admissible terminal decision, failing open on ambiguity."""

    branch_failures: list[str] = []
    for name, expected_disposition in REQUIRED_REOPENED_BRANCH_DISPOSITIONS.items():
        raw = reopened_branches.get(name)
        if not isinstance(raw, dict):
            branch_failures.append(name)
            continue
        if (
            raw.get("within_current_authenticated_portfolio") is not True
            or raw.get("decisive") is not True
            or raw.get("disposition") != expected_disposition
        ):
            branch_failures.append(name)
    for name, raw in reopened_branches.items():
        if name in REQUIRED_REOPENED_BRANCH_DISPOSITIONS:
            continue
        if not isinstance(raw, dict) or (
            raw.get("within_current_authenticated_portfolio") is True
            and raw.get("decisive") is not True
        ):
            branch_failures.append(str(name))

    effective = {
        name: conditions.get(name) is True for name in REQUIRED_CLOSURE_CONDITIONS
    }
    effective["all_reopened_authenticated_branches_have_decisive_dispositions"] = (
        effective["all_reopened_authenticated_branches_have_decisive_dispositions"]
        and not branch_failures
    )
    cache_survival = pw0332_all_strict_gates_survive(pw0332_absolute_gates)
    if cache_survival is None or cache_survival is True:
        effective["pw0332_absolute_floor_fails_at_least_one_strict_one_tps_gate"] = False
    failed = [name for name in REQUIRED_CLOSURE_CONDITIONS if not effective[name]]
    return {
        "conditions": effective,
        "failed_closure_conditions": failed,
        "reopened_branch_failures": sorted(set(branch_failures)),
        "decision": FINAL_DECISION if not failed else FRONTIER_OPEN,
        "status": "complete" if not failed else "incomplete",
    }


def _ceiling_claim(
    summary: Mapping[str, Any], *, claim_class: str
) -> dict[str, Any]:
    _require("not_achieved_tps" in claim_class, "analytical ceiling claim class")
    return {
        **dict(summary),
        "claim_class": claim_class,
        "achieved": False,
        "target_qualifying": False,
        "performance_claim": None,
    }


FINAL_REPORT_KEYS = {
    "schema_version",
    "experiment_id",
    "status",
    "decision",
    "analysis_commit",
    "scope",
    "authorities",
    "prior_portfolio",
    "measured_lower_milestones",
    "analytical_ceilings",
    "pw0331_prerequisite_interaction",
    "reopened_branch_dispositions",
    "scope_boundaries",
    "closure_conditions",
    "failed_closure_conditions",
    "limitations",
    "analysis_safety",
    "gate8",
    "accepted_tokens",
    "A",
    "U",
    "performance_claim",
    "runtime_default_changed",
}


def validate_final_report_schema(report: Mapping[str, Any]) -> None:
    _require(set(report) == FINAL_REPORT_KEYS, "PW-0333 report schema drift")
    _require(
        report.get("schema_version") == 1
        and report.get("experiment_id") == EXPERIMENT_ID
        and isinstance(report.get("analysis_commit"), str)
        and len(str(report["analysis_commit"])) == 40
        and report.get("accepted_tokens") == 0
        and report.get("A") == 0
        and report.get("U") == 0
        and report.get("performance_claim") is None
        and report.get("runtime_default_changed") is False,
        "PW-0333 zero-work analytical contract",
    )
    scope = _mapping(report.get("scope"), "PW-0333 scope")
    companion = _mapping(scope.get("companion_contributions"), "PW-0333 companion ledger")
    _require(
        scope.get("hardware") == "existing 16 GiB Apple M1"
        and scope.get("batch_size") == 1
        and scope.get("concurrency") == 1
        and scope.get("decode_timing_scope") == "after_prefill"
        and scope.get("target_accepted_tps_relation") == "strictly_greater_than_one"
        and scope.get("correctness_contract_changed") is False
        and scope.get("full_capability_required") is True
        and scope.get("fidelity_required") is True
        and scope.get("local_inference_required") is True
        and scope.get("companion_hardware_admissible") is False
        and set(companion)
        == {
            "storage_bytes",
            "memory_bytes",
            "compute_operations",
            "bandwidth_bytes_per_second",
            "cost_usd",
            "performance_tps",
        }
        and all(type(value) is int and value == 0 for value in companion.values()),
        "PW-0333 fixed scope/companion exclusion",
    )
    authorities = _mapping(report.get("authorities"), "PW-0333 authorities")
    _require(
        authorities.get("all_authenticated") is True
        and _mapping(authorities.get("contract"), "PW-0333 contract authority").get("git_blob")
        == CONTRACT_GIT_BLOB
        and authorities["contract"].get("sha256") == CONTRACT_SHA256
        and _mapping(authorities.get("throughput_model"), "PW-0333 throughput authority").get("sha256")
        == THROUGHPUT_MODEL_SHA256
        and _mapping(authorities.get("target"), "PW-0333 target authority").get("sha256")
        == TARGET_SHA256
        and _mapping(authorities.get("red_lines"), "PW-0333 red-lines authority").get("sha256")
        == RED_LINES_SHA256,
        "PW-0333 frozen authority ledger",
    )
    parent_reports = _mapping(authorities.get("parent_reports"), "PW-0333 parent reports")
    expected_parent_hashes = {
        "PW-0324": PW0324_SHA256,
        "PW-0328": PW0328_SHA256,
        "PW-0329": PW0329_SHA256,
        "PW-0330": PW0330_SHA256,
        "PW-0331": PW0331_SHA256,
        "PW-0332": PW0332_SHA256,
    }
    _require(set(parent_reports) == set(expected_parent_hashes), "PW-0333 parent authority set")
    for name, digest in expected_parent_hashes.items():
        _require(
            _mapping(parent_reports[name], f"PW-0333 {name} authority").get("sha256") == digest,
            f"PW-0333 {name} authority hash",
        )

    prior = _mapping(report.get("prior_portfolio"), "PW-0333 prior portfolio")
    prior_rows = _mapping(prior.get("portfolio"), "PW-0333 prior portfolio rows")
    _require(
        prior.get("portfolio_mechanisms") == len(PRIOR_PORTFOLIO_KEYS)
        and prior.get("two_tps_closure_survivors") == 0
        and prior.get("authenticated_proposer_family_states")
        == PW0324_PROPOSER_FAMILY_STATES
        and prior.get("unreconciled_proposer_family_survivors") == 0
        and prior.get(
            "historical_portfolio_hashes_preserved_not_rehashed_against_updated_records"
        )
        is True
        and set(prior_rows) == PRIOR_PORTFOLIO_KEYS,
        "PW-0333 prior portfolio authority",
    )
    for name, raw in prior_rows.items():
        row = _mapping(raw, f"PW-0333 prior portfolio {name}")
        _require(
            isinstance(row.get("record"), str)
            and isinstance(row.get("record_sha256"), str)
            and len(row["record_sha256"]) == 64
            and isinstance(row.get("state"), str)
            and bool(row["state"])
            and row.get("survives_two_tps_closure") is False,
            f"PW-0333 prior portfolio disposition: {name}",
        )

    measured = _mapping(report.get("measured_lower_milestones"), "PW-0333 measured ledger")
    designated = _mapping(measured.get("designated_protocol"), "PW-0333 designated protocol")
    complete = _mapping(measured.get("strongest_complete_request"), "PW-0333 complete request")
    holdout = _mapping(
        measured.get("strongest_after_prefill_32_token_holdout"),
        "PW-0333 after-prefill holdout",
    )
    short = _mapping(
        measured.get("strongest_after_prefill_short_slice"),
        "PW-0333 after-prefill short slice",
    )
    _require(
        measured.get("claim_class")
        == "measured_lower_milestones_not_designated_sustained_protocol"
        and measured.get("ledger_scope")
        == "clean_repeatable_native_mtp_request_slice_milestones"
        and designated.get("authenticated_results") == 0
        and designated.get("passing_results") == 0
        and designated.get("generations") == 30
        and designated.get("accepted_tokens_per_generation") == 512
        and complete.get("complete_request_timing_scope")
        == "complete_request_including_prefill"
        and holdout.get("after_prefill_timing_scope")
        == "measured_post_prefill_candidate_slice"
        and short.get("after_prefill_timing_scope")
        == "measured_post_prefill_candidate_slice"
        and designated.get("sustained_duration_minutes") == 60
        and designated.get("maximum_throughput_decay_fraction") == 0.10
        and designated.get("correctness_scope")
        == "unchanged_TARGET_sections_4_through_6"
        and complete.get("meets_designated_30_by_512_protocol") is False
        and holdout.get("meets_designated_30_by_512_protocol") is False
        and short.get("meets_designated_30_by_512_protocol") is False,
        "PW-0333 measured timing/protocol distinction",
    )
    excluded_diagnostics = _sequence(
        measured.get("excluded_diagnostic_classes"),
        "PW-0333 excluded diagnostic classes",
    )
    _require(
        len(excluded_diagnostics) == 3
        and {row.get("constant") for row in excluded_diagnostics if isinstance(row, dict)}
        == {
            "pw0203_wide_source_jacobi_endpoint",
            "pw0207_final_forty_two_object_residency",
            "pw0206_corrected_jacobi_third_iteration",
        }
        and all(
            isinstance(row, dict)
            and row.get("complete_request") is False
            and row.get("sustained") is False
            and row.get("target_qualifying") is False
            and float(row.get("maximum_reported_accepted_tps")) < 1.0
            for row in excluded_diagnostics
        )
        and _mapping(
            measured.get("highest_raw_diagnostic"),
            "PW-0333 highest raw diagnostic",
        ).get("classification")
        == "dirty_warm_single_verifier_block_L3_control",
        "PW-0333 diagnostic exclusion ledger",
    )
    _same_float(
        measured["highest_raw_diagnostic"].get("maximum_reported_accepted_tps"),
        0.21984968624124546,
        "PW-0333 highest raw diagnostic",
    )
    _same_float(
        complete.get("complete_request_accepted_tps"),
        0.04597815174359703,
        "PW-0333 complete-request lower milestone",
    )
    _same_float(
        holdout.get("after_prefill_accepted_tps"),
        0.0791305231426806,
        "PW-0333 32-token after-prefill lower milestone",
    )
    _require(
        float(short.get("after_prefill_accepted_tps")) < 1.0
        and float(holdout.get("after_prefill_accepted_tps")) < 1.0,
        "PW-0333 lower milestone below-one ledger",
    )

    ceilings = _mapping(report.get("analytical_ceilings"), "PW-0333 analytical ceilings")
    _require(
        set(ceilings)
        == {
            "pw0329_k4_fractional_impossible_best",
            "pw0330_named_cyclic_q32_conditional",
            "pw0332_exact_codec_absolute_floor",
        },
        "PW-0333 analytical ceiling set",
    )
    for name, raw in ceilings.items():
        ceiling = _mapping(raw, f"PW-0333 ceiling {name}")
        _require(
            ceiling.get("achieved") is False
            and ceiling.get("target_qualifying") is False
            and ceiling.get("performance_claim") is None
            and "not_achieved_tps" in str(ceiling.get("claim_class")),
            f"PW-0333 ceiling claim semantics: {name}",
        )
    codec_ceiling = _mapping(
        ceilings["pw0332_exact_codec_absolute_floor"],
        "PW-0333 exact-codec ceiling",
    )
    codec_replay = _mapping(
        codec_ceiling.get("codec_replay"),
        "PW-0333 exact-codec replay",
    )
    oracle_replay = _mapping(
        codec_ceiling.get("oracle_replay"),
        "PW-0333 cache-oracle replay",
    )
    replay_scenarios = _mapping(
        oracle_replay.get("scenarios"),
        "PW-0333 replayed cache scenarios",
    )
    _require(
        codec_replay.get("quantization_blocks_replayed") == 480
        and codec_replay.get("byte_equal_to_canonical") is True
        and oracle_replay.get("capacities_recomputed") == [204, 230, 250]
        and oracle_replay.get("full_canonical_ledgers_equal") is True
        and [
            _mapping(replay_scenarios.get(name), f"PW-0333 replay {name}").get("misses")
            for name in ("uncompressed", "observed_expert_only", "absolute_floor_all_fp8")
        ]
        == [53_040, 50_743, 49_122],
        "PW-0333 fresh codec/cache replay evidence",
    )

    branches = _mapping(
        report.get("reopened_branch_dispositions"),
        "PW-0333 reopened branches",
    )
    absolute_gates = _mapping(
        _mapping(
            _mapping(ceilings["pw0332_exact_codec_absolute_floor"], "PW-0333 codec ceiling").get(
                "strict_gates"
            ),
            "PW-0333 codec strict gates",
        ).get("absolute_floor_all_fp8"),
        "PW-0333 absolute strict gates",
    )
    derived = derive_closure(
        _mapping(report.get("closure_conditions"), "PW-0333 closure conditions"),
        branches,
        pw0332_absolute_gates=absolute_gates,
    )
    _require(
        report.get("decision") == derived["decision"]
        and report.get("status") == derived["status"]
        and report.get("failed_closure_conditions") == derived["failed_closure_conditions"],
        "PW-0333 derived terminal decision",
    )
    boundary = _mapping(report.get("scope_boundaries"), "PW-0333 scope boundaries")
    _require(
        boundary.get("direct_q32_first_chunk_parity")
        == "unproven_outside_evidence_backed_survivors"
        and boundary.get("unknown_future_algorithms_rejected") is False
        and boundary.get("universal_impossibility_theorem") is False,
        "PW-0333 non-universal scope boundary",
    )
    interaction = _mapping(
        report.get("pw0331_prerequisite_interaction"),
        "PW-0333 PW-0331 interaction",
    )
    _require(
        interaction.get("stage_a_pass") is True
        and interaction.get("stage_b_executed") is False
        and interaction.get("complete_k4_bank") is False
        and interaction.get("endpoint_pass") is False
        and interaction.get("blocked_by_pw0329_precedence_gate_two") is True,
        "PW-0333 PW-0331 Stage-A-only scope",
    )
    limitations = _sequence(report.get("limitations"), "PW-0333 limitations")
    _require(
        any("not a theorem against unknown future algorithms" in str(item) for item in limitations)
        and any("No authenticated result executes the designated" in str(item) for item in limitations)
        and any("storage-only ceilings are not achieved endpoint TPS" in str(item) for item in limitations)
        and any("direct-q32 first-chunk parity remains unproven" in str(item) for item in limitations)
        and any("dirty warm single-verifier-block L3 control" in str(item) for item in limitations),
        "PW-0333 required limitations",
    )
    gate8 = _mapping(report.get("gate8"), "PW-0333 Gate 8")
    parent_gate8 = _mapping(gate8.get("parents"), "PW-0333 parent Gate 8")
    pw0328_gate8 = _mapping(parent_gate8.get("PW-0328"), "PW-0333 PW-0328 Gate 8")
    _require(
        set(parent_gate8) == {"PW-0324", "PW-0328", "PW-0329", "PW-0330", "PW-0331", "PW-0332"}
        and all(_mapping(value, "PW-0333 parent Gate 8 row").get("pass") is True for value in parent_gate8.values())
        and _mapping(pw0328_gate8.get("builder"), "PW-0333 PW-0328 builder Gate 8").get("pass")
        is True
        and _mapping(pw0328_gate8.get("generation"), "PW-0333 PW-0328 generation Gate 8").get("count")
        == 4
        and pw0328_gate8["generation"].get("all_pass") is True
        and _mapping(pw0328_gate8.get("prefill"), "PW-0333 PW-0328 prefill Gate 8").get("count")
        == 4
        and pw0328_gate8["prefill"].get("all_pass") is True
        and gate8.get("all_pass") is True,
        "PW-0333 parent Gate 8 closure",
    )
    analysis_gate = validate_gate8(report.get("analysis_safety"), label="PW-0333 analyzer")
    _require(gate8.get("analysis") == analysis_gate, "PW-0333 analyzer Gate 8 summary")


def synthesize_report(
    *,
    commit: str,
    hardware: Mapping[str, Any],
    parent_paths: Mapping[str, Path],
    pw0324: Mapping[str, Any],
    pw0328: Mapping[str, Any],
    pw0329: Mapping[str, Any],
    pw0330: Mapping[str, Any],
    pw0331: Mapping[str, Any],
    pw0332: Mapping[str, Any],
    throughput: Mapping[str, Any],
    safety_snapshots: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    measured = _mapping(
        throughput.get("measured_lower_milestones"),
        "PW-0333 measured throughput reconciliation",
    )
    branches = build_reopened_branch_dispositions(
        pw0324=pw0324,
        pw0329=pw0329,
        pw0330=pw0330,
        pw0331=pw0331,
        pw0332=pw0332,
    )
    absolute_gates = _mapping(pw0332.get("strict_gates"), "PW-0332 strict gates").get(
        "absolute_floor_all_fp8"
    )
    pw0332_survival = pw0332_all_strict_gates_survive(absolute_gates)
    _require(pw0332_survival is False, "PW-0332 absolute floor survives or is stale")
    conditions = {
        "no_authenticated_target_spec_sustained_result_reaches_one_tps": (
            measured["designated_protocol"]["authenticated_results"] == 0
            and all(
                float(row["complete_request_accepted_tps"]) < 1.0
                and float(row["after_prefill_accepted_tps"]) < 1.0
                for row in measured["rows"]
            )
        ),
        "pw0329_current_k4_portfolio_fails_strict_window_tail_gate": (
            pw0329["precedence_gate"] == 2
            and pw0329["work_order"] is None
            and pw0329["ceiling"]["strict_fourth_lowest_window_above_one"] is False
        ),
        "pw0330_named_schedule_is_conditionally_below_one_and_prior_portfolio_has_no_other_survivor": (
            pw0330["favorable_storage_tps_ceiling"] < 1.0
            and pw0324["unreconciled_proposer_family_survivors"] == 0
        ),
        "pw0332_absolute_floor_fails_at_least_one_strict_one_tps_gate": True,
        "all_reopened_authenticated_branches_have_decisive_dispositions": True,
        "no_mode_has_complete_full_capability_fidelity_evidence_for_promotion": (
            measured["designated_protocol"]["passing_results"] == 0
            and pw0331["stage_b_executed"] is False
            and pw0332["decoder_authorized"] is False
        ),
        "all_authorities_formulas_schemas_claim_labels_and_gate8_close": True,
        "companion_hardware_contributes_zero_premise": True,
    }
    closure = derive_closure(
        conditions,
        branches,
        pw0332_absolute_gates=absolute_gates,
    )
    analysis_gate = validate_gate8(safety_snapshots, label="PW-0333 analyzer")
    parent_gates = {
        "PW-0324": pw0324["gate8"],
        "PW-0328": {
            "pass": True,
            "builder": pw0328["builder_gate8"],
            "generation": pw0328["generation_gate8"],
            "prefill": pw0328["prefill_gate8"],
        },
        "PW-0329": pw0329["gate8"],
        "PW-0330": pw0330["gate8"],
        "PW-0331": pw0331["gate8"],
        "PW-0332": pw0332["gate8"],
    }
    _require(
        all(_mapping(value, "parent Gate 8").get("pass") is True for value in parent_gates.values()),
        "PW-0333 parent Gate 8 closure",
    )
    parent_hashes = {
        "PW-0324": PW0324_SHA256,
        "PW-0328": PW0328_SHA256,
        "PW-0329": PW0329_SHA256,
        "PW-0330": PW0330_SHA256,
        "PW-0331": PW0331_SHA256,
        "PW-0332": PW0332_SHA256,
    }
    result = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "status": closure["status"],
        "decision": closure["decision"],
        "analysis_commit": commit,
        "scope": {
            "hardware": "existing 16 GiB Apple M1",
            "hardware_authority": dict(hardware),
            "checkpoint_revision": CHECKPOINT_REVISION,
            "batch_size": 1,
            "concurrency": 1,
            "decode_timing_scope": "after_prefill",
            "target_accepted_tps": 1.0,
            "target_accepted_tps_relation": "strictly_greater_than_one",
            "correctness_contract_changed": False,
            "full_capability_required": True,
            "fidelity_required": True,
            "local_inference_required": True,
            "companion_hardware_admissible": False,
            "companion_contributions": {
                "storage_bytes": 0,
                "memory_bytes": 0,
                "compute_operations": 0,
                "bandwidth_bytes_per_second": 0,
                "cost_usd": 0,
                "performance_tps": 0,
            },
        },
        "authorities": {
            "contract": {
                "path": CONTRACT_PATH,
                "freeze_commit": CONTRACT_FREEZE_COMMIT,
                "git_blob": CONTRACT_GIT_BLOB,
                "sha256": CONTRACT_SHA256,
            },
            "target": {"path": "TARGET.md", "sha256": TARGET_SHA256},
            "red_lines": {"path": "RED_LINES.md", "sha256": RED_LINES_SHA256},
            "throughput_model": {
                "path": "spec/throughput-model.json",
                "sha256": THROUGHPUT_MODEL_SHA256,
                "reconciliation_pass": throughput["reconciliation_pass"],
                "reconciled_constants": throughput["reconciled_constants"],
            },
            "checkpoint": {
                "revision": CHECKPOINT_REVISION,
                "verification_sha256": CHECKPOINT_RECEIPT_SHA256,
                "tensor_index_sha256": CHECKPOINT_INDEX_SHA256,
            },
            "parent_reports": {
                name: {"path": str(parent_paths[name]), "sha256": digest}
                for name, digest in parent_hashes.items()
            },
            "pw0332_contract": {
                "execution_commit": PW0332_COMMIT,
                "git_blob": PW0332_CONTRACT_GIT_BLOB,
                "sha256": PW0332_CONTRACT_SHA256,
            },
            "pw0328_q1_demand_stream_sha256": PW0328_Q1_DEMAND_SHA256,
            "all_authenticated": True,
        },
        "prior_portfolio": pw0324,
        "measured_lower_milestones": measured,
        "analytical_ceilings": {
            "pw0329_k4_fractional_impossible_best": _ceiling_claim(
                pw0329["ceiling"],
                claim_class="impossible_best_storage_only_ceiling_not_achieved_tps",
            ),
            "pw0330_named_cyclic_q32_conditional": _ceiling_claim(
                pw0330,
                claim_class="conditional_storage_only_ceiling_not_achieved_tps",
            ),
            "pw0332_exact_codec_absolute_floor": _ceiling_claim(
                {
                    **dict(pw0332["absolute_floor"]),
                    "strict_gates": pw0332["strict_gates"],
                    "scenario_capacities": pw0332["scenario_capacities"],
                    "q1_demand_stream_sha256": pw0332["q1_demand_stream_sha256"],
                    "codec_floor": pw0332["codec_floor"],
                    "codec_replay": pw0332["codec_replay"],
                    "oracle_replay": pw0332["oracle_replay"],
                },
                claim_class="absolute_zero_escape_storage_only_floor_not_achieved_tps",
            ),
        },
        "pw0331_prerequisite_interaction": {
            **dict(pw0331),
            "complete_k4_bank": False,
            "endpoint_pass": False,
            "blocked_by_pw0329_precedence_gate_two": True,
        },
        "reopened_branch_dispositions": branches,
        "scope_boundaries": {
            "direct_q32_first_chunk_parity": "unproven_outside_evidence_backed_survivors",
            "unknown_future_algorithms_rejected": False,
            "universal_impossibility_theorem": False,
        },
        "closure_conditions": closure["conditions"],
        "failed_closure_conditions": closure["failed_closure_conditions"],
        "limitations": [
            "This closes the current authenticated evidence-backed onboard architecture portfolio, not a theorem against unknown future algorithms.",
            "No authenticated result executes the designated 30-by-512 after-prefill sustained protocol at the unchanged full-capability and fidelity scope.",
            "PW-0329, PW-0330, and PW-0332 storage-only ceilings are not achieved endpoint TPS and omit non-storage costs favorably.",
            "PW-0330 direct-q32 first-chunk parity remains unproven and its result rejects only the named cyclic schedule under the stated conditional scope.",
            "The 0.21984968624124546-TPS PW-0203 raw diagnostic is a dirty warm single-verifier-block L3 control, not a complete request, sustained run, or target-qualifying result.",
            "PW-0331 passes local Stage A only; it does not establish a complete K4 bank, endpoint parity, or a promotable runtime.",
            "Reopening requires a genuinely new representation or proposer premise with independent correctness and physical gates; threshold movement is outside this conclusion.",
        ],
        "analysis_safety": list(safety_snapshots),
        "gate8": {
            "parents": parent_gates,
            "analysis": analysis_gate,
            "all_pass": True,
        },
        "accepted_tokens": 0,
        "A": 0,
        "U": 0,
        "performance_claim": None,
        "runtime_default_changed": False,
    }
    validate_final_report_schema(result)
    return result


def analyze(
    *,
    repo: Path,
    commit: str,
    output: Path,
    pw0324_path: Path = PW0324_PATH,
    pw0328_path: Path = PW0328_PATH,
    pw0329_path: Path = PW0329_PATH,
    pw0330_path: Path = PW0330_PATH,
    pw0331_path: Path = PW0331_PATH,
    pw0332_path: Path = PW0332_PATH,
    throughput_model_path: Path | None = None,
    checkpoint_root: Path = PW0332_DEFAULT_CHECKPOINT_ROOT,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(output)
    repo = repo.resolve()
    throughput_model_path = (
        repo / "spec/throughput-model.json"
        if throughput_model_path is None
        else throughput_model_path.resolve()
    )
    checkpoint_root = checkpoint_root.resolve()
    verify_execution_commit(repo, commit)
    hardware = verify_target_hardware()
    safety = HostSafetyMonitor()

    raw_pw0324 = _read_json(pw0324_path, PW0324_SHA256, "PW-0324")
    raw_pw0328 = _read_json(pw0328_path, PW0328_SHA256, "PW-0328")
    raw_pw0329 = _read_json(pw0329_path, PW0329_SHA256, "PW-0329")
    raw_pw0330 = _read_json(pw0330_path, PW0330_SHA256, "PW-0330")
    raw_pw0331 = _read_json(pw0331_path, PW0331_SHA256, "PW-0331")
    raw_pw0332 = _read_json(pw0332_path, PW0332_SHA256, "PW-0332")
    throughput_model = _read_json(
        throughput_model_path,
        THROUGHPUT_MODEL_SHA256,
        "throughput model",
    )
    validate_pw0328_manifest_header(raw_pw0328)
    live_pw0328 = authenticate_pw0328_corpus(repo=repo, manifest_path=pw0328_path)
    local_codec_replay = authenticate_pw0332_pw0324(checkpoint_root)
    safety.checkpoint("authorities_authenticated")

    preliminary_measured = measured_lower_milestones(throughput_model)
    pw0324 = validate_pw0324(raw_pw0324, preliminary_measured)
    pw0328 = summarize_pw0328(live_pw0328)
    pw0329 = validate_pw0329(raw_pw0329)
    pw0330 = validate_pw0330(raw_pw0330)
    pw0331 = validate_pw0331(raw_pw0331)
    pw0332 = validate_pw0332(raw_pw0332, live_pw0328, local_codec_replay)
    throughput = validate_throughput_model(
        throughput_model,
        pw0329=pw0329,
        pw0330=pw0330,
        pw0331=pw0331,
        pw0332=pw0332,
    )
    safety.checkpoint("parent_recomputations_complete")

    del (
        raw_pw0324,
        raw_pw0328,
        raw_pw0329,
        raw_pw0330,
        raw_pw0331,
        raw_pw0332,
        live_pw0328,
        local_codec_replay,
        throughput_model,
        preliminary_measured,
    )
    safety.release_checkpoint(
        "parent_authorities_released",
        [
            "six frozen parent JSON objects",
            "authenticated PW-0328 artifact and q1 replay",
            "throughput-model source object",
        ],
    )
    safety.checkpoint("final_service_health")
    parent_paths = {
        "PW-0324": pw0324_path,
        "PW-0328": pw0328_path,
        "PW-0329": pw0329_path,
        "PW-0330": pw0330_path,
        "PW-0331": pw0331_path,
        "PW-0332": pw0332_path,
    }
    result = synthesize_report(
        commit=commit,
        hardware=hardware,
        parent_paths=parent_paths,
        pw0324=pw0324,
        pw0328=pw0328,
        pw0329=pw0329,
        pw0330=pw0330,
        pw0331=pw0331,
        pw0332=pw0332,
        throughput=throughput,
        safety_snapshots=safety.evidence(),
    )
    write_new_report(output, result)
    print(json.dumps({"output": str(output / "analysis.json"), "decision": result["decision"]}))
    return result


def write_new_report(output: Path, report: Mapping[str, Any]) -> Path:
    """Create a new evidence directory only after report validation succeeds."""

    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=False)
    path = output / "analysis.json"
    try:
        atomic_write_new(path, canonical_json(report))
    except BaseException:
        # atomic_write_new removes any partial file.  Remove only this analyzer's
        # newly-created directory, and only if it is still empty.
        try:
            output.rmdir()
        except OSError:
            pass
        raise
    return path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--pw0324-path", type=Path, default=PW0324_PATH)
    parser.add_argument("--pw0328-path", type=Path, default=PW0328_PATH)
    parser.add_argument("--pw0329-path", type=Path, default=PW0329_PATH)
    parser.add_argument("--pw0330-path", type=Path, default=PW0330_PATH)
    parser.add_argument("--pw0331-path", type=Path, default=PW0331_PATH)
    parser.add_argument("--pw0332-path", type=Path, default=PW0332_PATH)
    parser.add_argument("--throughput-model-path", type=Path)
    parser.add_argument("--checkpoint-root", type=Path, default=PW0332_DEFAULT_CHECKPOINT_ROOT)
    try:
        analyze(**vars(parser.parse_args()))
        return 0
    except (
        FileExistsError,
        HostSafetyViolation,
        KeyError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ) as error:
        print(json.dumps({"error": str(error)}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
