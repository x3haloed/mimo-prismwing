#!/usr/bin/env python3
"""Run PW-0132's train-only rank-32 INT4 repair generalization test."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import platform
import time

import mlx.core as mx
import numpy as np

try:
    from tools.generate_real_layer1_expert_oracle import ShardedCheckpoint
    from tools.host_safety import HostSafetyMonitor
    from tools.openrouter_reference import atomic_write_new, canonical_json
    from tools.run_best_rank_real_expert_control import load_capture, sha256_file
    from tools.run_int4_low_rank_repair_oracle import (
        PW0130_SHA256,
        _factor_digest,
        apply_low_rank_repair,
        fit_low_rank_repair,
        physical_ledger,
    )
    from tools.run_int4_output_affine_repair_oracle import (
        PW0129_SHA256,
        _collect_layer,
        _parameter_digest,
        _route_rows,
        apply_output_repair,
        fit_output_repair,
    )
    from tools.run_real_activation_affine_int4_audit import (
        CORPUS_SHA256,
        LAYERS,
        REVISION,
        VERIFICATION_SHA256,
        error_metrics,
        validate_routes,
    )
except ModuleNotFoundError:
    from generate_real_layer1_expert_oracle import ShardedCheckpoint
    from host_safety import HostSafetyMonitor
    from openrouter_reference import atomic_write_new, canonical_json
    from run_best_rank_real_expert_control import load_capture, sha256_file
    from run_int4_low_rank_repair_oracle import (
        PW0130_SHA256,
        _factor_digest,
        apply_low_rank_repair,
        fit_low_rank_repair,
        physical_ledger,
    )
    from run_int4_output_affine_repair_oracle import (
        PW0129_SHA256,
        _collect_layer,
        _parameter_digest,
        _route_rows,
        apply_output_repair,
        fit_output_repair,
    )
    from run_real_activation_affine_int4_audit import (
        CORPUS_SHA256,
        LAYERS,
        REVISION,
        VERIFICATION_SHA256,
        error_metrics,
        validate_routes,
    )


PW0131_SHA256 = "e0cf60d13b3e55fd805b480bf834baa55e87f7cf5de6b49623f722c094c0d876"
RANK = 32


def split_indices(positions: list[int]) -> tuple[list[int], list[int]]:
    if any(not isinstance(position, int) or not 0 <= position < 168 for position in positions):
        raise ValueError("PW-0132 partition positions are invalid")
    train = [index for index, position in enumerate(positions) if position < 112]
    validation = [index for index, position in enumerate(positions) if position >= 112]
    if sorted(train + validation) != list(range(len(positions))):
        raise ValueError("PW-0132 partition mapping is not a bijection")
    return train, validation


def fit_train_only_repairs(
    rows: dict[int, dict],
    moe_input: np.ndarray,
) -> dict:
    affine_parameters = {}
    rank_factors = {}
    affine_rows = {}
    repaired_rows = {}
    validation_total = 0
    validation_covered = 0
    train_placements = 0
    fitted_experts = 0
    fit_started = time.perf_counter()
    for expert, record in rows.items():
        train_indices, validation_indices = split_indices(record["positions"])
        validation_total += len(validation_indices)
        candidate = record["candidate"]
        affine_all = candidate.copy()
        repaired_all = candidate.copy()
        if train_indices:
            fitted_experts += 1
            train_placements += len(train_indices)
            scale, bias = fit_output_repair(
                candidate[train_indices], record["source"][train_indices], "affine"
            )
            affine_parameters[expert] = (scale, bias)
            affine_train = apply_output_repair(candidate[train_indices], scale, bias)
            affine_all[train_indices] = affine_train
            train_positions = [record["positions"][index] for index in train_indices]
            train_input = np.asarray(moe_input[train_positions], dtype=np.float16).astype(np.float32)
            residual_train = record["source"][train_indices] - affine_train
            factors = fit_low_rank_repair(train_input, residual_train, RANK)
            rank_factors[expert] = factors
            repaired_all[train_indices] = apply_low_rank_repair(
                train_input, affine_train, factors
            )
            if validation_indices:
                validation_covered += len(validation_indices)
                validation_positions = [record["positions"][index] for index in validation_indices]
                validation_input = np.asarray(
                    moe_input[validation_positions], dtype=np.float16
                ).astype(np.float32)
                affine_validation = apply_output_repair(
                    candidate[validation_indices], scale, bias
                )
                affine_all[validation_indices] = affine_validation
                repaired_all[validation_indices] = apply_low_rank_repair(
                    validation_input, affine_validation, factors
                )
        affine_rows[expert] = {"positions": record["positions"], "repaired": affine_all}
        repaired_rows[expert] = {"positions": record["positions"], "repaired": repaired_all}
    return {
        "affine_parameters": affine_parameters,
        "rank_factors": rank_factors,
        "affine_rows": affine_rows,
        "repaired_rows": repaired_rows,
        "coverage": {
            "fitted_experts": fitted_experts,
            "train_placements": train_placements,
            "validation_placements": validation_total,
            "validation_placements_covered": validation_covered,
            "validation_coverage_fraction": validation_covered / max(validation_total, 1),
            "validation_identity_fallback_placements": validation_total - validation_covered,
        },
        "affine_parameter_sha256": _parameter_digest(affine_parameters),
        "rank_factor_sha256": _factor_digest(rank_factors),
        "fit_wall_ms": (time.perf_counter() - fit_started) * 1000.0,
    }


def _partition_metrics(
    row_map: dict[int, dict],
    authority: dict,
    routed_expected: np.ndarray,
    field: str,
) -> dict:
    started = time.perf_counter()
    routed = _route_rows(row_map, authority, field, 0, 168)
    return {
        "train": error_metrics(
            routed[:112], np.asarray(routed_expected[:112], dtype=np.float32)
        ),
        "validation": error_metrics(
            routed[112:168], np.asarray(routed_expected[112:168], dtype=np.float32)
        ),
        "application_wall_ms": (time.perf_counter() - started) * 1000.0,
    }


def _gate(layer_reports: list[dict]) -> dict:
    repaired = [row["rank32"]["validation"] for row in layer_reports]
    squared_error = sum(row["squared_error"] for row in repaired)
    expected_norm = sum(row["expected_squared_norm"] for row in repaired)
    aggregate = float(np.sqrt(squared_error / max(expected_norm, 1e-30)))
    monotonic = all(
        row["rank32"]["validation"]["relative_l2"]
        <= row["affine"]["validation"]["relative_l2"]
        <= row["baseline"]["validation"]["relative_l2"]
        for row in layer_reports
    )
    complete_coverage = all(
        row["coverage"]["validation_coverage_fraction"] == 1.0
        and row["coverage"]["validation_identity_fallback_placements"] == 0
        for row in layer_reports
    )
    physical = physical_ledger(RANK)
    strict = (
        aggregate <= 0.01
        and all(row["relative_l2"] <= 0.02 for row in repaired)
        and all(row["maximum_row_relative_l2"] <= 0.05 for row in repaired)
        and monotonic
        and complete_coverage
        and physical["combined_to_source_layer_bank_ratio"] <= 0.60
        and physical["repair_to_source_expert_mac_ratio"] <= 0.05
    )
    near_miss = (
        not strict
        and aggregate <= 0.02
        and all(row["relative_l2"] <= 0.04 for row in repaired)
        and all(row["maximum_row_relative_l2"] <= 0.08 for row in repaired)
        and complete_coverage
    )
    return {
        "aggregate_relative_l2": aggregate,
        "maximum_layer_relative_l2": max(row["relative_l2"] for row in repaired),
        "maximum_row_relative_l2": max(row["maximum_row_relative_l2"] for row in repaired),
        "nested_validation_monotonic": monotonic,
        "complete_validation_coverage": complete_coverage,
        "physical": physical,
        "strict_pass": strict,
        "near_miss": near_miss,
        "thresholds": {
            "strict": {"aggregate": 0.01, "layer": 0.02, "row": 0.05},
            "near_miss": {"aggregate": 0.02, "layer": 0.04, "row": 0.08},
        },
    }


def run(
    checkpoint_root: Path,
    verification_path: Path,
    corpus_manifest_path: Path,
    pw0129_path: Path,
    pw0131_path: Path,
    output_path: Path,
    commit: str,
) -> dict:
    if output_path.exists():
        raise ValueError(f"refusing to overwrite {output_path}")
    if len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit):
        raise ValueError("implementation commit must be lowercase 40-hex")
    for path, expected, label in (
        (verification_path, VERIFICATION_SHA256, "checkpoint verification"),
        (corpus_manifest_path, CORPUS_SHA256, "PW-0116 corpus"),
        (pw0129_path, PW0129_SHA256, "PW-0129 report"),
        (pw0131_path, PW0131_SHA256, "PW-0131 report"),
    ):
        if sha256_file(path) != expected:
            raise ValueError(f"PW-0132 {label} hash mismatch")
    started = time.perf_counter()
    safety = HostSafetyMonitor()
    corpus = json.loads(corpus_manifest_path.read_text())
    prior = json.loads(pw0129_path.read_text())
    capacity = json.loads(pw0131_path.read_text())
    if (
        corpus.get("revision") != REVISION
        or prior.get("decision") != "reject_naive_affine_int4_on_real_validation"
        or capacity.get("decision") != "authorize_train_only_int4_low_rank_repair"
        or capacity["capacity_gate"].get("smallest_passing_rank") != RANK
        or prior.get("holdout_unsealed")
        or capacity.get("holdout_unsealed")
    ):
        raise ValueError("PW-0132 authority identity mismatch")
    checkpoint = ShardedCheckpoint(checkpoint_root, verification_path)
    root = corpus_manifest_path.parent
    layer_reports = []
    for layer in LAYERS:
        authority = next(row for row in corpus["layers"] if row["layer"] == layer)
        validate_routes(authority)
        moe_input = load_capture(root, authority["captures"]["moe_input"])
        expert_down = load_capture(root, authority["captures"]["expert_down"])
        routed_expected = load_capture(root, authority["captures"]["routed_output"])
        rows = _collect_layer(
            checkpoint, authority, moe_input, expert_down, prior, layer, safety
        )
        baseline_rows = {
            expert: {"positions": record["positions"], "candidate": record["candidate"]}
            for expert, record in rows.items()
        }
        baseline = _partition_metrics(
            baseline_rows, authority, routed_expected, "candidate"
        )
        fitted = fit_train_only_repairs(rows, moe_input)
        affine = _partition_metrics(
            fitted["affine_rows"], authority, routed_expected, "repaired"
        )
        rank32 = _partition_metrics(
            fitted["repaired_rows"], authority, routed_expected, "repaired"
        )
        layer_reports.append(
            {
                "layer": layer,
                "baseline": baseline,
                "affine": affine,
                "rank32": rank32,
                "coverage": fitted["coverage"],
                "affine_parameter_sha256": fitted["affine_parameter_sha256"],
                "rank_factor_sha256": fitted["rank_factor_sha256"],
                "fit_wall_ms": fitted["fit_wall_ms"],
            }
        )
        del moe_input, expert_down, routed_expected, rows, fitted
        gc.collect()
        mx.clear_cache()
        safety.release_checkpoint(
            f"layer_{layer}_train_validation_buffers_released",
            ["real corpus", "INT4 outputs", "train-fitted affine and rank-32 repair"],
        )
    gate = _gate(layer_reports)
    decision = (
        "authorize_separate_rank32_repair_holdout"
        if gate["strict_pass"]
        else (
            "authorize_broader_training_corpus_for_rank32_repair"
            if gate["near_miss"]
            else "reject_pilot_train_only_rank32_int4_repair"
        )
    )
    safety.release_checkpoint(
        "checkpoint_and_authorities_released",
        ["checkpoint mappings", "PW-0116 corpus", "PW-0129/PW-0131 reports"],
    )
    safety.checkpoint("final_service_health")
    report = {
        "schema_version": 1,
        "evidence_class": "pw0132_train_only_int4_rank32_repair",
        "revision": REVISION,
        "commit": commit,
        "source_hashes": {
            "checkpoint_verification_sha256": VERIFICATION_SHA256,
            "corpus_manifest_sha256": CORPUS_SHA256,
            "pw0129_report_sha256": PW0129_SHA256,
            "pw0131_report_sha256": PW0131_SHA256,
        },
        "rank": RANK,
        "layer_reports": layer_reports,
        "validation_gate": gate,
        "holdout_unsealed": False,
        "decision": decision,
        "safety_snapshots": safety.evidence(),
        "complete_wall_ms": (time.perf_counter() - started) * 1000.0,
        "accepted_tokens": 0,
        "A": 0,
        "performance_claim": None,
        "limitations": (
            "train-only fit and validation on one correlated English pilot prefix; holdout "
            "sealed; no broader corpus, accumulated model, modality evaluation, measured "
            "repair kernel, endpoint, accepted tokens, or TPS claim"
        ),
        "platform": platform.platform(),
    }
    atomic_write_new(output_path, canonical_json(report))
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--verification", required=True, type=Path)
    parser.add_argument("--corpus-manifest", required=True, type=Path)
    parser.add_argument("--pw0129", required=True, type=Path)
    parser.add_argument("--pw0131", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--commit", required=True)
    arguments = parser.parse_args()
    try:
        result = run(
            arguments.checkpoint,
            arguments.verification,
            arguments.corpus_manifest,
            arguments.pw0129,
            arguments.pw0131,
            arguments.output,
            arguments.commit,
        )
        print(json.dumps({"output": str(arguments.output), "decision": result["decision"]}))
        return 0
    except (OSError, RuntimeError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
        print(json.dumps({"error": str(error)}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
