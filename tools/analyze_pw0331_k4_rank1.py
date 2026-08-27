#!/usr/bin/env python3
"""Analyze two frozen PW-0331 fits against primary and held-out evidence.

Unlike the fit runner, this process may open complete PW-0116 captures.  It
never refits or rescales the factors: their byte hashes must agree across two
fresh fit-only runs before any held-out array is loaded.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import platform
import resource
import time
from typing import Any

import numpy as np

try:
    from tools.construct_pw0313_m1_native_k4_expert import metric
    from tools.construct_pw0314_layer4_k4 import (
        load_capture,
        partition_metrics,
        reconstruct_route,
        selected_rows,
    )
    from tools.host_safety import HostSafetyMonitor, HostSafetyViolation
    from tools.openrouter_reference import atomic_write_new, canonical_json
    from tools.reproduce_pw0311_k4_expert import sha256_file
    from tools.run_pw0331_k4_rank1_fit import (
        CONTRACT_COMMIT,
        CONTRACT_GIT_BLOB,
        CONTRACT_SHA256,
        CORPUS_SHA256,
        EXPERIMENT_ID,
        EXPERT,
        EXPECTED_COUNTS,
        HIDDEN,
        IMPLEMENTATION_HASHES,
        INTERMEDIATE,
        LAYER,
        PANEL_CONTRACT_SHA256,
        PANEL_EXPORT_SHA256,
        PRIMARY_POSITION,
        PW0315_EXPERT96_REPORT_SHA256,
        PW0315_SUMMARY_SHA256,
        PW0316_REJECTION_SHA256,
        PW0318_BUNDLE_SHA256,
        PW0318_FIXTURE_SHA256,
        PW0318_MANIFEST_SHA256,
        PW0318_SUMMARY_SHA256,
        PANEL_IMPLEMENTATION_SHA256,
        RED_LINES_SHA256,
        SERIALIZED_DENSE_CONTROL_DIAGNOSTIC_SHA256,
        SERIALIZED_DENSE_CONTROL_GIT_BLOB,
        SERIALIZED_DENSE_CONTROL_SHA256,
        SERIALIZED_DENSE_CONTROL_STAGES_SHA256,
        SEMANTIC,
        STAGE_A_BASE_SEMANTIC,
        TARGET_SHA256,
        apply_serialized_rank_one,
        array_sha256,
        bf16,
        deterministic_tree,
        load_pw0318_tlut,
        load_panel_authority,
        load_zero_correction_k4,
        partition_fit_positions,
        require_legacy_framed_array_sha256,
        schema2_layout_ledger,
        sha256_bytes,
        stage_a_numerics_authority,
        stage_a_candidate_stages,
        verify_execution_authority,
    )
except ModuleNotFoundError:
    from construct_pw0313_m1_native_k4_expert import metric
    from construct_pw0314_layer4_k4 import (
        load_capture,
        partition_metrics,
        reconstruct_route,
        selected_rows,
    )
    from host_safety import HostSafetyMonitor, HostSafetyViolation
    from openrouter_reference import atomic_write_new, canonical_json
    from reproduce_pw0311_k4_expert import sha256_file
    from run_pw0331_k4_rank1_fit import (
        CONTRACT_COMMIT,
        CONTRACT_GIT_BLOB,
        CONTRACT_SHA256,
        CORPUS_SHA256,
        EXPERIMENT_ID,
        EXPERT,
        EXPECTED_COUNTS,
        HIDDEN,
        IMPLEMENTATION_HASHES,
        INTERMEDIATE,
        LAYER,
        PANEL_CONTRACT_SHA256,
        PANEL_EXPORT_SHA256,
        PRIMARY_POSITION,
        PW0315_EXPERT96_REPORT_SHA256,
        PW0315_SUMMARY_SHA256,
        PW0316_REJECTION_SHA256,
        PW0318_BUNDLE_SHA256,
        PW0318_FIXTURE_SHA256,
        PW0318_MANIFEST_SHA256,
        PW0318_SUMMARY_SHA256,
        PANEL_IMPLEMENTATION_SHA256,
        RED_LINES_SHA256,
        SERIALIZED_DENSE_CONTROL_DIAGNOSTIC_SHA256,
        SERIALIZED_DENSE_CONTROL_GIT_BLOB,
        SERIALIZED_DENSE_CONTROL_SHA256,
        SERIALIZED_DENSE_CONTROL_STAGES_SHA256,
        SEMANTIC,
        STAGE_A_BASE_SEMANTIC,
        TARGET_SHA256,
        apply_serialized_rank_one,
        array_sha256,
        bf16,
        deterministic_tree,
        load_pw0318_tlut,
        load_panel_authority,
        load_zero_correction_k4,
        partition_fit_positions,
        require_legacy_framed_array_sha256,
        schema2_layout_ledger,
        sha256_bytes,
        stage_a_numerics_authority,
        stage_a_candidate_stages,
        verify_execution_authority,
    )


MAXIMUM_RELATIVE_L2 = 0.01
MAXIMUM_ROW_RELATIVE_L2 = 0.05
EXPERTS = (96, 64, 232, 31)
ZERO_ROUTE_RELATIVE_L2 = 0.010988841869031155
ZERO_FINAL_RELATIVE_L2 = 0.0027743952049186665
EXPECTED_ALPHA_MIN = 0.1523576677
EXPECTED_ALPHA_MIN_ABSOLUTE_TOLERANCE = 5e-10


def analysis_slices() -> dict[str, np.ndarray]:
    return {
        "overall": np.arange(224, dtype=np.int64),
        "fit": np.asarray(
            [position for position in range(112) if position != PRIMARY_POSITION],
            dtype=np.int64,
        ),
        "validation": np.arange(112, 168, dtype=np.int64),
        "pilot": np.arange(168, 224, dtype=np.int64),
    }


def sliced_metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    left = np.asarray(reference, dtype=np.float32)
    right = np.asarray(candidate, dtype=np.float32)
    if left.shape != (224, HIDDEN) or right.shape != left.shape:
        raise ValueError("PW-0331 sliced metric shape mismatch")
    return {
        name: partition_metrics(left[indices], right[indices])
        for name, indices in analysis_slices().items()
    }


def sliced_gate(metrics: dict[str, Any]) -> bool:
    if set(metrics) != {"overall", "fit", "validation", "pilot"}:
        raise ValueError("PW-0331 sliced gate partition mismatch")
    return all(
        np.isfinite(row.get("relative_l2", np.nan))
        and np.isfinite(row.get("maximum_row_relative_l2", np.nan))
        and float(row["relative_l2"]) < MAXIMUM_RELATIVE_L2
        and float(row["maximum_row_relative_l2"]) < MAXIMUM_ROW_RELATIVE_L2
        for row in metrics.values()
    )


def verify_fit_array_bindings(
    *,
    fit_hashes: dict[str, str],
    selected_input: np.ndarray,
    expert_down: np.ndarray,
    route_weights: np.ndarray,
    source_offsets: np.ndarray,
    fit_local: np.ndarray,
    dynamic_hidden: np.ndarray,
    base_raw: np.ndarray,
) -> None:
    """Bind every frozen fit input, target, weight, and K4 stage to analysis data."""
    recomputed = {
        "fit_input_f32": np.asarray(selected_input[fit_local], dtype=np.float32),
        "fit_source_bf16_f32": np.asarray(
            expert_down[source_offsets[fit_local]], dtype=np.float32
        ),
        "fit_route_weights_f32": np.asarray(
            route_weights[fit_local], dtype=np.float32
        ),
        "candidate_dynamic_hidden_f32": np.asarray(
            dynamic_hidden[fit_local], dtype=np.float32
        ),
        "candidate_down_base_raw_f32": np.asarray(
            base_raw[fit_local], dtype=np.float32
        ),
        "candidate_down_base_bf16_f32": bf16(base_raw[fit_local]),
    }
    mismatches = [
        name
        for name, values in recomputed.items()
        if array_sha256(values) != fit_hashes.get(name)
    ]
    if mismatches:
        raise ValueError(
            "PW-0331 fit/analyzer Stage A row hash mismatch: "
            + ", ".join(mismatches)
        )


def stage_a_gate(
    identity_route: dict[str, Any],
    identity_final: dict[str, Any],
    cumulative_route: dict[str, Any],
    cumulative_final: dict[str, Any],
    primary_route: dict[str, float],
    primary_final: dict[str, float],
    attenuation_requirement_pass: bool,
) -> dict[str, Any]:
    gates = {
        "identity_route": sliced_gate(identity_route),
        "identity_final": sliced_gate(identity_final),
        "cumulative_route": sliced_gate(cumulative_route),
        "cumulative_final": sliced_gate(cumulative_final),
        "primary_route": bool(
            np.isfinite(primary_route.get("relative_l2", np.nan))
            and primary_route["relative_l2"] < MAXIMUM_RELATIVE_L2
        ),
        "primary_final": bool(
            np.isfinite(primary_final.get("relative_l2", np.nan))
            and primary_final["relative_l2"] < MAXIMUM_RELATIVE_L2
        ),
    }
    gates["sliced_and_primary_gates_pass"] = all(gates.values())
    gates["attenuation_sanity"] = attenuation_requirement_pass is True
    gates["pass"] = (
        gates["sliced_and_primary_gates_pass"] and gates["attenuation_sanity"]
    )
    return gates


def error_direction_diagnostic(
    source_route: np.ndarray,
    zero_route: np.ndarray,
    source_expert: np.ndarray,
    zero_expert: np.ndarray,
    corrected_expert: np.ndarray,
    route_weight: float,
) -> dict[str, Any]:
    """Compute the frozen non-authoritative PW-0331 position-1 scalar."""
    arrays = [
        np.asarray(value, dtype=np.float64)
        for value in (
            source_route,
            zero_route,
            source_expert,
            zero_expert,
            corrected_expert,
        )
    ]
    if (
        any(value.ndim != 1 or value.shape != arrays[0].shape for value in arrays)
        or arrays[0].size == 0
        or any(not np.isfinite(value).all() for value in arrays)
        or not np.isfinite(route_weight)
    ):
        raise ValueError("PW-0331 error-direction diagnostic input mismatch")
    source, zero, source_local, zero_local, corrected_local = arrays
    weight = np.float64(route_weight)
    e4 = zero - source
    d96 = weight * (zero_local - source_local)
    corrected_d96 = weight * (corrected_local - source_local)
    denominator = float(np.dot(d96, d96))
    source_norm_sq = float(np.dot(source, source))
    if denominator <= 0.0 or source_norm_sq <= 0.0:
        raise ValueError("PW-0331 error-direction diagnostic is degenerate")
    projection = float(np.dot(e4, d96))
    constant = float(np.dot(e4, e4)) - (MAXIMUM_RELATIVE_L2**2) * source_norm_sq
    discriminant = projection * projection - denominator * constant
    roundoff = np.finfo(np.float64).eps * max(
        projection * projection, abs(denominator * constant), 1.0
    )
    if discriminant < -16.0 * roundoff:
        raise ValueError("PW-0331 error-direction boundary has no real root")
    discriminant = max(discriminant, 0.0)
    root = float(np.sqrt(discriminant))
    roots = [
        (projection - root) / denominator,
        (projection + root) / denominator,
    ]
    nonnegative = sorted(value for value in roots if value >= 0.0 and np.isfinite(value))
    if not nonnegative:
        raise ValueError("PW-0331 error-direction boundary has no nonnegative root")
    alpha_min = float(nonnegative[0])
    observed_attenuation = float(
        np.dot(d96 - corrected_d96, d96) / denominator
    )
    return {
        "semantic": (
            "position1_f64_smaller_nonnegative_root_and_d96_projection_v1"
        ),
        "alpha_min": alpha_min,
        "expected_alpha_min": EXPECTED_ALPHA_MIN,
        "expected_alpha_min_absolute_tolerance": EXPECTED_ALPHA_MIN_ABSOLUTE_TOLERANCE,
        "observed_attenuation": observed_attenuation,
        "attenuation_requirement_pass": bool(observed_attenuation >= alpha_min),
        "gate_role": "subordinate_sanity_condition_after_sliced_fidelity_gates",
    }


def verify_pw0318_heldout_authorities(
    summary_path: Path, fixture_path: Path
) -> dict[str, Any]:
    """Open PW-0318's position-1 authority only in the post-freeze analyzer."""
    if sha256_file(summary_path) != PW0318_SUMMARY_SHA256:
        raise ValueError("PW-0331 PW-0318 summary mismatch")
    if sha256_file(fixture_path) != PW0318_FIXTURE_SHA256:
        raise ValueError("PW-0331 PW-0318 position-1 fixture mismatch")
    summary = json.loads(summary_path.read_text())
    fixture = json.loads(fixture_path.read_text())
    identical = summary.get("identical_artifacts", {})
    if (
        summary.get("experiment_id") != "PW-0318"
        or summary.get("status") != "layer4_decode_transaction_qualified"
        or identical.get("layer04-position001.fixture.json") != PW0318_FIXTURE_SHA256
        or identical.get("layer04-position001.k4-source.manifest.json")
        != PW0318_MANIFEST_SHA256
        or identical.get("layer04-position001.k4-source.bin") != PW0318_BUNDLE_SHA256
        or fixture.get("schema_version") != 1
        or fixture.get("layer") != LAYER
        or fixture.get("position") != PRIMARY_POSITION
    ):
        raise ValueError("PW-0331 PW-0318 held-out authority contract mismatch")
    return {
        "summary_sha256": PW0318_SUMMARY_SHA256,
        "position1_fixture_sha256": PW0318_FIXTURE_SHA256,
        "opened_after_factor_freeze": True,
    }


def _load_factor(path: Path, record: dict[str, Any], expected_shape: tuple[int, int]) -> np.ndarray:
    if (
        record.get("dtype") != "<f2"
        or record.get("shape") != list(expected_shape)
        or int(record.get("bytes", -1)) != int(np.prod(expected_shape)) * 2
    ):
        raise ValueError("PW-0331 frozen factor record mismatch")
    factor_path = (path / record["file"]).resolve()
    try:
        factor_path.relative_to(path.resolve())
    except ValueError as error:
        raise ValueError("PW-0331 frozen factor path escapes fit root") from error
    payload = factor_path.read_bytes()
    if len(payload) != record["bytes"] or sha256_bytes(payload) != record["sha256"]:
        raise ValueError("PW-0331 frozen factor payload mismatch")
    result = np.frombuffer(payload, dtype="<f2").reshape(expected_shape).copy()
    if not np.isfinite(result).all() or not np.any(result):
        raise ValueError("PW-0331 frozen factor is zero or nonfinite")
    return result


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def load_frozen_fit_run(
    root: Path, commit: str
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, dict[str, Any]]:
    report_path = root / "construction.json"
    authority_path = root / "fit-authority.json"
    report = json.loads(report_path.read_text())
    process = report.get("process_receipt", {})
    if (
        report.get("schema_version") != 1
        or report.get("experiment_id") != EXPERIMENT_ID
        or report.get("status") != "fit_factors_frozen_without_heldout_access"
        or report.get("commit") != commit
        or report.get("accepted_tokens") != 0
        or report.get("A") != 0
        or report.get("U") != 0
        or report.get("performance_claim") is not None
        or not isinstance(process.get("pid"), int)
        or process["pid"] <= 0
        or not isinstance(process.get("started_ns"), int)
        or process["started_ns"] <= 0
        or not _is_sha256(process.get("nonce"))
        or sha256_file(authority_path) != report.get("fit_authority_sha256")
    ):
        raise ValueError("PW-0331 frozen fit report mismatch")
    authority = json.loads(authority_path.read_text())
    corpus = authority.get("corpus_authority", {})
    numerics = authority.get("numerics", {})
    execution = authority.get("execution_authority", {})
    metadata = authority.get("metadata_authority", {})
    panel = authority.get("panel_authority", {})
    arrays = authority.get("array_sha256", {})
    positions = authority.get("fit_positions", [])
    offsets = authority.get("fit_source_offsets", [])
    slots = authority.get("fit_slots", [])
    tlut = authority.get("tlut_authority", {})
    k4 = authority.get("k4_authority", {})
    fit = authority.get("fit", {})
    control = authority.get("serialized_dense_control", {})
    expected_arrays = {
        "fit_input_f32",
        "fit_source_bf16_f32",
        "fit_route_weights_f32",
        "candidate_dynamic_hidden_f32",
        "candidate_down_base_raw_f32",
        "candidate_down_base_bf16_f32",
    }
    if (
        authority.get("schema_version") != 1
        or authority.get("experiment_id") != EXPERIMENT_ID
        or authority.get("semantic") != SEMANTIC
        or authority.get("exactness_class") != "L3_modified_expert_weights"
        or authority.get("commit") != commit
        or authority.get("construction_surface")
        != "fit_rows_only_no_primary_validation_or_pilot_payload_access"
        or corpus.get("whole_capture_payloads_rescanned") is not False
        or corpus.get("split_counts") != EXPECTED_COUNTS
        or corpus.get("corpus_manifest_sha256") != CORPUS_SHA256
        or corpus.get("input_read", {}).get("whole_payload_rescanned") is not False
        or corpus.get("source_read", {}).get("whole_payload_rescanned") is not False
        or execution.get("contract_commit") != CONTRACT_COMMIT
        or execution.get("contract_git_blob") != CONTRACT_GIT_BLOB
        or execution.get("contract_sha256") != CONTRACT_SHA256
        or execution.get("serialized_dense_control_git_blob")
        != SERIALIZED_DENSE_CONTROL_GIT_BLOB
        or execution.get("serialized_dense_control_sha256")
        != SERIALIZED_DENSE_CONTROL_SHA256
        or execution.get("target_sha256") != TARGET_SHA256
        or execution.get("red_lines_sha256") != RED_LINES_SHA256
        or execution.get("unchanged_implementation_sha256") != IMPLEMENTATION_HASHES
        or metadata.get("pw0315_summary_sha256") != PW0315_SUMMARY_SHA256
        or metadata.get("pw0316_rejection_sha256") != PW0316_REJECTION_SHA256
        or metadata.get("published_position1_route_relative_l2")
        != ZERO_ROUTE_RELATIVE_L2
        or metadata.get("published_position1_final_relative_l2")
        != ZERO_FINAL_RELATIVE_L2
        or metadata.get("held_out_payloads_opened") is not False
        or panel
        != {
            "contract_sha256": PANEL_CONTRACT_SHA256,
            "export_sha256": PANEL_EXPORT_SHA256,
            "implementation_sha256": PANEL_IMPLEMENTATION_SHA256,
        }
        or set(arrays) != expected_arrays
        or not all(_is_sha256(value) for value in arrays.values())
        or len(positions) != EXPECTED_COUNTS["fit"]
        or positions != sorted(set(positions))
        or any(not isinstance(value, int) or value < 0 or value >= 112 or value == 1 for value in positions)
        or len(offsets) != EXPECTED_COUNTS["fit"]
        or len(set(offsets)) != len(offsets)
        or any(not isinstance(value, int) or value < 0 or value >= 1792 for value in offsets)
        or len(slots) != EXPECTED_COUNTS["fit"]
        or any(not isinstance(value, int) or value < 0 or value >= 8 for value in slots)
        or corpus.get("input_read", {}).get("selected_rows") != positions
        or corpus.get("source_read", {}).get("selected_rows") != offsets
        or tlut.get("manifest_sha256") != PW0318_MANIFEST_SHA256
        or tlut.get("bundle_sha256") != PW0318_BUNDLE_SHA256
        or tlut.get("tlut_bytes") != 4096
        or not _is_sha256(tlut.get("tlut_sha256"))
        or set(k4) != {"gate", "up", "down", "checkpoint"}
        or any(
            not _is_sha256(k4.get(name, {}).get("manifest_sha256"))
            or not _is_sha256(k4.get(name, {}).get("candidate_array_sha256"))
            or k4.get(name, {}).get("rank") != 1
            or k4.get(name, {}).get("row_scale_identity") is not True
            or k4.get(name, {}).get("correction_zero") is not True
            for name in ("gate", "up", "down")
        )
        or k4.get("checkpoint", {}).get("revision")
        != "63651580ca774f8504f676040460aed3e1244ac1"
        or k4.get("checkpoint", {}).get("receipt_sha256")
        != "9ddc8a99755f04ae2ea3c2484f6dd022d3f3a681b5a72c915ee4de833dbb0d03"
        or k4.get("checkpoint", {}).get("index_sha256")
        != "f2e1774c9acf9a62338b68c144e6fc7a66495e59f2e64b3078c1b7ef5a196816"
        or k4.get("checkpoint", {}).get("expert96_shard_sha256_from_receipt")
        != "f8c8ab1b22da717ed0360c8248da84d0f9a58af7a89deeb6d4021a67ae98a046"
        or fit.get("fit_rows") != EXPECTED_COUNTS["fit"]
        or fit.get("input_columns") != INTERMEDIATE
        or fit.get("output_rows") != HIDDEN
        or fit.get("rcond") != 1e-12
        or set(control)
        != {
            "semantic",
            "fixture_sha256",
            "diagnostic_sha256",
            "stages_sha256",
            "independent_process_replays",
            "fit_rows",
            "held_out_payloads_opened",
            "stages",
            "pass",
        }
        or control.get("semantic")
        != "fit_only_zero_factor_serialized_vs_historical_dense_control_v1"
        or control.get("fixture_sha256") != SERIALIZED_DENSE_CONTROL_SHA256
        or control.get("diagnostic_sha256")
        != SERIALIZED_DENSE_CONTROL_DIAGNOSTIC_SHA256
        or control.get("stages_sha256") != SERIALIZED_DENSE_CONTROL_STAGES_SHA256
        or control.get("independent_process_replays") != 2
        or control.get("fit_rows") != EXPECTED_COUNTS["fit"]
        or control.get("held_out_payloads_opened") is not False
        or control.get("pass") is not True
        or sha256_bytes(canonical_json(control.get("stages")))
        != SERIALIZED_DENSE_CONTROL_STAGES_SHA256
        or authority.get("layout") != schema2_layout_ledger(4, 4)
        or set(authority.get("factors", {}))
        != {"correction_left", "correction_right"}
        or numerics != stage_a_numerics_authority()
        or authority.get("accepted_tokens") != 0
        or authority.get("A") != 0
        or authority.get("U") != 0
        or authority.get("performance_claim") is not None
    ):
        raise ValueError("PW-0331 frozen fit authority mismatch")
    left = _load_factor(root, authority["factors"]["correction_left"], (HIDDEN, 1))
    right = _load_factor(root, authority["factors"]["correction_right"], (1, INTERMEDIATE))
    if deterministic_tree(root) != report.get("deterministic_tree"):
        raise ValueError("PW-0331 frozen fit deterministic tree mismatch")
    return authority, left, right, process


def require_repeated_fits(
    roots: list[Path], commit: str
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, dict[str, Any]]:
    if len(roots) != 2 or roots[0].resolve() == roots[1].resolve():
        raise ValueError("PW-0331 analysis requires two distinct fit-run roots")
    first = load_frozen_fit_run(roots[0], commit)
    second = load_frozen_fit_run(roots[1], commit)
    if (
        first[0] != second[0]
        or not np.array_equal(first[1], second[1])
        or not np.array_equal(first[2], second[2])
    ):
        raise ValueError("PW-0331 fresh-process fit mismatch")
    if (
        first[3] == second[3]
        or first[3]["pid"] == second[3]["pid"]
        or first[3]["nonce"] == second[3]["nonce"]
    ):
        raise ValueError("PW-0331 fit runs lack distinct process receipts")
    return first[0], first[1], first[2], {
        "fit_authority_sha256": [
            sha256_file(root / "fit-authority.json") for root in roots
        ],
        "correction_left_sha256": array_sha256(first[1].astype("<f2", copy=False)),
        "correction_right_sha256": array_sha256(first[2].astype("<f2", copy=False)),
        "process_receipts": [first[3], second[3]],
        "fresh_process_repeat": True,
    }


def _replace_expert(
    outputs: np.ndarray,
    layer_row: dict[str, Any],
    expert: int,
    candidate: np.ndarray,
) -> np.ndarray:
    result = np.asarray(outputs, dtype=np.float32).copy()
    positions, _, _, offsets = selected_rows(layer_row, expert)
    replacement = np.asarray(candidate, dtype=np.float32)
    if replacement.shape != (len(positions), HIDDEN):
        raise ValueError(f"PW-0331 expert-{expert} candidate shape mismatch")
    result[offsets] = replacement
    return result


def _load_pw0315_candidate(
    evidence_root: Path,
    summary: dict[str, Any],
    layer_row: dict[str, Any],
    expert: int,
) -> np.ndarray:
    run = evidence_root / f"expert-{expert:03d}-run-001"
    report_path = run / "construction.json"
    expected_report = summary["identities"][str(expert)]["report_sha256"][0]
    if sha256_file(report_path) != expected_report:
        raise ValueError(f"PW-0331 expert-{expert} report mismatch")
    report = json.loads(report_path.read_text())
    if report.get("semantic", {}).get("gates", {}).get("pass") is not True:
        raise ValueError(f"PW-0331 expert-{expert} is not qualified")
    positions, _, _, _ = selected_rows(layer_row, expert)
    path = run / f"layer-{LAYER:02d}-expert-{expert:03d}" / "candidate-output.f32le"
    candidate = np.fromfile(path, dtype="<f4").reshape(len(positions), HIDDEN)
    require_legacy_framed_array_sha256(
        candidate,
        report["semantic"]["array_sha256"]["candidate_output_f32"],
        f"expert-{expert} candidate",
    )
    return candidate


def analyze(
    *,
    fit_runs: list[Path],
    authority_root: Path,
    corpus_manifest: Path,
    pw0315_summary: Path,
    pw0315_evidence_root: Path,
    pw0315_expert96_root: Path,
    pw0315_expert96_construction: Path,
    pw0316_rejection: Path,
    pw0318_summary: Path,
    pw0318_fixture: Path,
    pw0318_manifest: Path,
    pw0318_bundle: Path,
    repo: Path,
    commit: str,
    output: Path,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(output)
    started = time.monotonic()
    execution = verify_execution_authority(repo, commit)
    fit_authority, left, right, repeat = require_repeated_fits(fit_runs, commit)
    # Only after both frozen fit trees pass may this process open held-out arrays.
    pw0318_heldout_authority = verify_pw0318_heldout_authorities(
        pw0318_summary, pw0318_fixture
    )
    if sha256_file(corpus_manifest) != CORPUS_SHA256:
        raise ValueError("PW-0331 corpus manifest mismatch")
    if sha256_file(pw0315_summary) != PW0315_SUMMARY_SHA256:
        raise ValueError("PW-0331 PW-0315 summary mismatch")
    if sha256_file(pw0316_rejection) != PW0316_REJECTION_SHA256:
        raise ValueError("PW-0331 PW-0316 rejection mismatch")
    summary = json.loads(pw0315_summary.read_text())
    corpus = json.loads(corpus_manifest.read_text())
    layer_rows = [row for row in corpus["layers"] if int(row["layer"]) == LAYER]
    if len(layer_rows) != 1:
        raise ValueError("PW-0331 layer-4 corpus mismatch")
    layer_row = layer_rows[0]
    positions, slots96, route_weights96, source_offsets = selected_rows(layer_row, EXPERT)
    partitions = partition_fit_positions(positions)
    if (
        fit_authority.get("fit_positions") != positions[partitions["fit"]].tolist()
        or fit_authority.get("fit_source_offsets")
        != source_offsets[partitions["fit"]].tolist()
        or fit_authority.get("fit_slots") != slots96[partitions["fit"]].tolist()
    ):
        raise ValueError("PW-0331 frozen fit selection mismatch")
    safety = HostSafetyMonitor()
    modules, panel_authority = load_panel_authority(authority_root)
    if fit_authority.get("panel_authority") != panel_authority:
        raise ValueError("PW-0331 fit/analyzer panel authority mismatch")
    corpus_root = corpus_manifest.parent
    moe_input = load_capture(corpus_root, layer_row, "moe_input")
    expert_down = load_capture(corpus_root, layer_row, "expert_down")
    source_routed = load_capture(corpus_root, layer_row, "routed_output")
    post_attention = load_capture(corpus_root, layer_row, "post_attention")
    source_final = load_capture(corpus_root, layer_row, "final")
    if not np.array_equal(
        reconstruct_route(expert_down, layer_row, modules["panel"].bf16), source_routed
    ):
        raise ValueError("PW-0331 source route reconstruction mismatch")
    if not np.array_equal(modules["panel"].bf16(post_attention + source_routed), source_final):
        raise ValueError("PW-0331 source final reconstruction mismatch")
    safety.checkpoint("heldout_authorities_opened_after_factor_freeze")
    tlut, tlut_authority = load_pw0318_tlut(pw0318_manifest, pw0318_bundle)
    decoded, serialized, k4_authority = load_zero_correction_k4(
        expert_root=pw0315_expert96_root,
        construction_report=pw0315_expert96_construction,
        tlut=tlut,
        modules=modules,
    )
    if (
        fit_authority.get("tlut_authority") != tlut_authority
        or fit_authority.get("k4_authority") != k4_authority
    ):
        raise ValueError("PW-0331 fit/analyzer serialized K4 authority mismatch")
    selected_input = np.asarray(moe_input[positions], dtype=np.float32)
    stages = stage_a_candidate_stages(
        selected_input, serialized, tlut, modules["panel"].dynamic_input
    )
    dynamic_hidden = np.asarray(stages["dynamic_hidden_f32"], dtype=np.float32)
    base_raw = np.asarray(stages["down_base_raw_f32"], dtype=np.float32)
    fit_local = partitions["fit"]
    fit_hashes = fit_authority.get("array_sha256", {})
    verify_fit_array_bindings(
        fit_hashes=fit_hashes,
        selected_input=selected_input,
        expert_down=expert_down,
        route_weights=route_weights96,
        source_offsets=source_offsets,
        fit_local=fit_local,
        dynamic_hidden=dynamic_hidden,
        base_raw=base_raw,
    )
    base_bf16 = bf16(base_raw)
    if not np.array_equal(base_bf16, stages["candidate_output_bf16_f32"]):
        raise ValueError("PW-0331 zero-factor B_raw control mismatch")
    stored_zero = _load_pw0315_candidate(
        pw0315_evidence_root, summary, layer_row, EXPERT
    )
    if not np.array_equal(base_bf16, stored_zero):
        raise ValueError("PW-0331 zero factors do not reproduce PW-0315 bits")
    corrected = apply_serialized_rank_one(dynamic_hidden, base_raw, left, right)

    identity_down = _replace_expert(expert_down, layer_row, EXPERT, corrected)
    identity_route = reconstruct_route(identity_down, layer_row, modules["panel"].bf16)
    identity_final = modules["panel"].bf16(post_attention + identity_route)

    zero_cumulative_down = expert_down.copy()
    corrected_cumulative_down = expert_down.copy()
    candidates = {EXPERT: stored_zero}
    for expert in EXPERTS[1:]:
        candidates[expert] = _load_pw0315_candidate(
            pw0315_evidence_root, summary, layer_row, expert
        )
    for expert in EXPERTS:
        zero_cumulative_down = _replace_expert(
            zero_cumulative_down, layer_row, expert, candidates[expert]
        )
        corrected_candidate = corrected if expert == EXPERT else candidates[expert]
        corrected_cumulative_down = _replace_expert(
            corrected_cumulative_down, layer_row, expert, corrected_candidate
        )
    zero_route = reconstruct_route(zero_cumulative_down, layer_row, modules["panel"].bf16)
    zero_final = modules["panel"].bf16(post_attention + zero_route)
    zero_route_metric = metric(
        source_routed[PRIMARY_POSITION : PRIMARY_POSITION + 1],
        zero_route[PRIMARY_POSITION : PRIMARY_POSITION + 1],
    )
    zero_final_metric = metric(
        source_final[PRIMARY_POSITION : PRIMARY_POSITION + 1],
        zero_final[PRIMARY_POSITION : PRIMARY_POSITION + 1],
    )
    if (
        zero_route_metric["relative_l2"] != ZERO_ROUTE_RELATIVE_L2
        or zero_final_metric["relative_l2"] != ZERO_FINAL_RELATIVE_L2
    ):
        raise ValueError("PW-0331 zero-factor PW-0316 scalar mismatch")
    primary_local = int(partitions["primary"][0])
    direction = error_direction_diagnostic(
        source_routed[PRIMARY_POSITION],
        zero_route[PRIMARY_POSITION],
        expert_down[int(source_offsets[primary_local])],
        stored_zero[primary_local],
        corrected[primary_local],
        float(route_weights96[primary_local]),
    )
    if (
        abs(direction["alpha_min"] - EXPECTED_ALPHA_MIN)
        > EXPECTED_ALPHA_MIN_ABSOLUTE_TOLERANCE
    ):
        raise ValueError("PW-0331 frozen error-direction alpha_min mismatch")

    cumulative_route = reconstruct_route(
        corrected_cumulative_down, layer_row, modules["panel"].bf16
    )
    cumulative_final = modules["panel"].bf16(post_attention + cumulative_route)
    identity_route_metrics = sliced_metrics(source_routed, identity_route)
    identity_final_metrics = sliced_metrics(source_final, identity_final)
    cumulative_route_metrics = sliced_metrics(source_routed, cumulative_route)
    cumulative_final_metrics = sliced_metrics(source_final, cumulative_final)
    primary_route = metric(
        source_routed[PRIMARY_POSITION : PRIMARY_POSITION + 1],
        cumulative_route[PRIMARY_POSITION : PRIMARY_POSITION + 1],
    )
    primary_final = metric(
        source_final[PRIMARY_POSITION : PRIMARY_POSITION + 1],
        cumulative_final[PRIMARY_POSITION : PRIMARY_POSITION + 1],
    )
    gates = stage_a_gate(
        identity_route_metrics,
        identity_final_metrics,
        cumulative_route_metrics,
        cumulative_final_metrics,
        primary_route,
        primary_final,
        direction["attenuation_requirement_pass"],
    )
    output.mkdir(parents=True)
    candidate_path = output / "expert-096-corrected-output.f32le"
    atomic_write_new(candidate_path, corrected.astype("<f4", copy=False).tobytes(order="C"))
    result = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "semantic": "pw0331_stage_a_heldout_analysis_v1",
        "exactness_class": "L3_modified_expert_weights",
        "status": "stage_a_pass" if gates["pass"] else "stage_a_rejected",
        "decision": (
            "authorize_stage_b_zero_and_corrected_layout_controls"
            if gates["pass"]
            else "reject_down_only_rank_one_embodiment"
        ),
        "commit": commit,
        "execution_authority": execution,
        "fit_repeat_authority": repeat,
        "pw0318_heldout_authority": pw0318_heldout_authority,
        "tlut_authority": tlut_authority,
        "k4_authority": k4_authority,
        "panel_authority": panel_authority,
        "numerics": {
            "stage_a_base_semantic": STAGE_A_BASE_SEMANTIC,
            "stage_a_fit_and_gate_base_match": (
                fit_authority.get("numerics", {}).get("stage_a_base_semantic")
                == STAGE_A_BASE_SEMANTIC
            ),
            "stage_a_base_is_metal_answer_key": False,
            "stage_b_metal_answer_key_constructed": False,
        },
        "source_replay_exact": True,
        "zero_factor_control": {
            "position1_route": zero_route_metric,
            "position1_final": zero_final_metric,
        },
        "position1_error_direction_diagnostic": direction,
        "corrected_candidate": {
            "file": candidate_path.name,
            "bytes": candidate_path.stat().st_size,
            "sha256": sha256_file(candidate_path),
        },
        "identity_local": {
            "route_candidate_vs_source": identity_route_metrics,
            "final_candidate_vs_source": identity_final_metrics,
        },
        "cumulative_four_expert": {
            "experts": list(EXPERTS),
            "route_candidate_vs_source": cumulative_route_metrics,
            "final_candidate_vs_source": cumulative_final_metrics,
            "position1_route_candidate_vs_source": primary_route,
            "position1_final_candidate_vs_source": primary_final,
        },
        "gates": {
            **gates,
            "maximum_relative_l2_exclusive": MAXIMUM_RELATIVE_L2,
            "maximum_row_relative_l2_exclusive": MAXIMUM_ROW_RELATIVE_L2,
        },
        "stage_b": {
            "authorized": gates["pass"],
            "layout_control_constructed": False,
            "expected_layout": schema2_layout_ledger(4, 4),
        },
        "complete_seconds": time.monotonic() - started,
        "peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "host": {
            "machine": platform.machine(),
            "platform": platform.platform(),
            "total_memory_bytes": int(os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")),
        },
        "batch_size": 1,
        "concurrency": 1,
        "accepted_tokens": 0,
        "A": 0,
        "U": 0,
        "performance_claim": None,
        "claims_excluded": [
            "density-five qualification",
            "complete K4 bank",
            "endpoint TPS",
            "target-faithful weights",
            "runtime default",
        ],
    }
    del (
        decoded,
        serialized,
        stages,
        dynamic_hidden,
        base_raw,
        base_bf16,
        corrected,
        moe_input,
        expert_down,
        source_routed,
        post_attention,
        source_final,
        identity_down,
        identity_route,
        identity_final,
        zero_cumulative_down,
        corrected_cumulative_down,
        zero_route,
        zero_final,
        cumulative_route,
        cumulative_final,
        candidates,
        tlut,
    )
    gc.collect()
    safety.release_checkpoint(
        "stage_a_analysis_buffers_released",
        ["complete held-out captures", "decoded K4 weights", "candidate route staging"],
    )
    safety.checkpoint("final_service_health")
    result["safety_snapshots"] = safety.evidence()
    atomic_write_new(output / "analysis.json", canonical_json(result))
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fit-run", action="append", required=True, type=Path, dest="fit_runs")
    parser.add_argument("--authority-root", required=True, type=Path)
    parser.add_argument("--corpus-manifest", required=True, type=Path)
    parser.add_argument("--pw0315-summary", required=True, type=Path)
    parser.add_argument("--pw0315-evidence-root", required=True, type=Path)
    parser.add_argument("--pw0315-expert96-root", required=True, type=Path)
    parser.add_argument("--pw0315-expert96-construction", required=True, type=Path)
    parser.add_argument("--pw0316-rejection", required=True, type=Path)
    parser.add_argument("--pw0318-summary", required=True, type=Path)
    parser.add_argument("--pw0318-fixture", required=True, type=Path)
    parser.add_argument("--pw0318-manifest", required=True, type=Path)
    parser.add_argument("--pw0318-bundle", required=True, type=Path)
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    try:
        result = analyze(**vars(arguments))
        print(json.dumps({"output": str(arguments.output), "status": result["status"]}))
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
