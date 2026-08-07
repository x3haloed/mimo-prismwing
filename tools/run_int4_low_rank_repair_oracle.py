#!/usr/bin/env python3
"""Run PW-0131's same-validation input-conditioned low-rank repair oracle."""

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
    from tools.run_int4_output_affine_repair_oracle import (
        FULL_AFFINE_REPAIR_BYTES_PER_LAYER,
        INT4_BYTES,
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
        SOURCE_EXPERT_BYTES,
        VERIFICATION_SHA256,
        error_metrics,
        validate_routes,
    )
except ModuleNotFoundError:
    from generate_real_layer1_expert_oracle import ShardedCheckpoint
    from host_safety import HostSafetyMonitor
    from openrouter_reference import atomic_write_new, canonical_json
    from run_best_rank_real_expert_control import load_capture, sha256_file
    from run_int4_output_affine_repair_oracle import (
        FULL_AFFINE_REPAIR_BYTES_PER_LAYER,
        INT4_BYTES,
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
        SOURCE_EXPERT_BYTES,
        VERIFICATION_SHA256,
        error_metrics,
        validate_routes,
    )


PW0130_SHA256 = "b011bd5ced8787df62f4380aeeccab9a35aef8b8ab15541207bcd99e35727994"
RANKS = (8, 16, 32, 56)
D = 4096
SOURCE_LAYER_BANK_BYTES = 256 * SOURCE_EXPERT_BYTES
INT4_LAYER_BANK_BYTES = 256 * INT4_BYTES
SOURCE_EXPERT_MACS_PER_MIXTURE = 8 * 3 * 2048 * 4096


def fit_low_rank_repair(
    inputs: np.ndarray,
    residual: np.ndarray,
    rank: int,
) -> tuple[np.ndarray, np.ndarray]:
    if (
        inputs.ndim != 2
        or residual.ndim != 2
        or inputs.shape[0] == 0
        or inputs.shape[0] != residual.shape[0]
        or inputs.shape[1] != D
        or residual.shape[1] != D
        or rank <= 0
        or not np.isfinite(inputs).all()
        or not np.isfinite(residual).all()
    ):
        raise ValueError("PW-0131 low-rank fit inputs are invalid")
    input64 = inputs.astype(np.float64, copy=False)
    residual64 = residual.astype(np.float64, copy=False)
    u, singular, vt = np.linalg.svd(residual64, full_matrices=False)
    effective = min(rank, singular.size)
    coordinates = u[:, :effective] * singular[:effective]
    gram_pinv = np.linalg.pinv(input64 @ input64.T, rcond=1e-12)
    left_effective = input64.T @ gram_pinv @ coordinates
    left = np.zeros((D, rank), dtype=np.float64)
    right = np.zeros((rank, D), dtype=np.float64)
    left[:, :effective] = left_effective
    right[:effective] = vt[:effective]
    left_f16 = left.astype(np.float16)
    right_f16 = right.astype(np.float16)
    if not np.isfinite(left_f16).all() or not np.isfinite(right_f16).all():
        raise ValueError("PW-0131 F16 low-rank factors are non-finite")
    return left_f16, right_f16


def apply_low_rank_repair(
    inputs: np.ndarray,
    base: np.ndarray,
    factors: tuple[np.ndarray, np.ndarray],
) -> np.ndarray:
    left, right = factors
    if (
        inputs.ndim != 2
        or base.shape != (inputs.shape[0], D)
        or inputs.shape[1] != D
        or left.ndim != 2
        or right.shape != (left.shape[1], D)
        or left.shape[0] != D
        or left.dtype != np.float16
        or right.dtype != np.float16
    ):
        raise ValueError("PW-0131 low-rank application shape/dtype mismatch")
    repair = (
        inputs.astype(np.float32, copy=False) @ left.astype(np.float32)
    ) @ right.astype(np.float32)
    result = base.astype(np.float32, copy=False) + repair
    if not np.isfinite(result).all():
        raise ValueError("PW-0131 repaired output is non-finite")
    return result


def _factor_digest(factors: dict[int, tuple[np.ndarray, np.ndarray]]) -> str:
    digest = hashlib.sha256()
    for expert, (left, right) in sorted(factors.items()):
        digest.update(expert.to_bytes(2, "little"))
        digest.update(np.ascontiguousarray(left.astype("<f2", copy=False)).tobytes())
        digest.update(np.ascontiguousarray(right.astype("<f2", copy=False)).tobytes())
    return digest.hexdigest()


def physical_ledger(rank: int) -> dict:
    if rank <= 0:
        raise ValueError("PW-0131 rank must be positive")
    factor_bytes = 256 * 2 * D * rank * 2
    combined = INT4_LAYER_BANK_BYTES + FULL_AFFINE_REPAIR_BYTES_PER_LAYER + factor_bytes
    repair_macs = 8 * 2 * D * rank
    return {
        "rank": rank,
        "low_rank_factor_bytes_per_layer": factor_bytes,
        "int4_layer_bank_bytes": INT4_LAYER_BANK_BYTES,
        "affine_repair_bytes_per_layer": FULL_AFFINE_REPAIR_BYTES_PER_LAYER,
        "combined_bytes_per_layer": combined,
        "combined_to_source_layer_bank_ratio": combined / SOURCE_LAYER_BANK_BYTES,
        "repair_macs_per_eight_expert_mixture": repair_macs,
        "source_expert_macs_per_eight_expert_mixture": SOURCE_EXPERT_MACS_PER_MIXTURE,
        "repair_to_source_expert_mac_ratio": repair_macs / SOURCE_EXPERT_MACS_PER_MIXTURE,
    }


def _build_affine_and_rank_repairs(
    rows: dict[int, dict],
    moe_input: np.ndarray,
) -> tuple[dict, dict[int, dict], dict[int, str]]:
    affine_parameters = {}
    affine_rows = {}
    rank_rows = {rank: {} for rank in RANKS}
    rank_factors = {rank: {} for rank in RANKS}
    for expert, record in rows.items():
        validation_indices = [
            index for index, position in enumerate(record["positions"]) if 112 <= position < 168
        ]
        candidate = record["candidate"]
        affine_all = candidate.copy()
        per_rank = {rank: candidate.copy() for rank in RANKS}
        if validation_indices:
            positions = [record["positions"][index] for index in validation_indices]
            candidate_validation = candidate[validation_indices]
            source_validation = record["source"][validation_indices]
            scale, bias = fit_output_repair(candidate_validation, source_validation, "affine")
            affine_validation = apply_output_repair(candidate_validation, scale, bias)
            affine_all[validation_indices] = affine_validation
            affine_parameters[expert] = (scale, bias)
            residual = source_validation - affine_validation
            staged_input = np.asarray(moe_input[positions], dtype=np.float16).astype(np.float32)
            for rank in RANKS:
                factors = fit_low_rank_repair(staged_input, residual, rank)
                per_rank[rank][validation_indices] = apply_low_rank_repair(
                    staged_input, affine_validation, factors
                )
                rank_factors[rank][expert] = factors
        affine_rows[expert] = {"positions": record["positions"], "repaired": affine_all}
        for rank in RANKS:
            rank_rows[rank][expert] = {
                "positions": record["positions"],
                "repaired": per_rank[rank],
            }
    return (
        {"parameters": affine_parameters, "rows": affine_rows},
        rank_rows,
        {rank: _factor_digest(factors) for rank, factors in rank_factors.items()},
    )


def _capacity_gate(rank_reports: list[dict]) -> dict:
    by_rank = {rank: [row for row in rank_reports if row["rank"] == rank] for rank in RANKS}
    if any(len(rows) != len(LAYERS) for rows in by_rank.values()):
        raise ValueError("PW-0131 rank/layer report cardinality mismatch")
    aggregate = {}
    for rank, rows in by_rank.items():
        squared_error = sum(row["metrics"]["squared_error"] for row in rows)
        expected_norm = sum(row["metrics"]["expected_squared_norm"] for row in rows)
        aggregate[rank] = float(np.sqrt(squared_error / max(expected_norm, 1e-30)))
    monotonic = all(
        all(
            next(row for row in by_rank[right] if row["layer"] == layer)["metrics"]["relative_l2"]
            <= next(row for row in by_rank[left] if row["layer"] == layer)["metrics"]["relative_l2"]
            + 1e-12
            for left, right in zip(RANKS, RANKS[1:])
        )
        for layer in LAYERS
    )
    rows_by_rank = []
    passing = []
    for rank in RANKS:
        rows = by_rank[rank]
        physical = physical_ledger(rank)
        passes = (
            aggregate[rank] <= 0.01
            and all(row["metrics"]["relative_l2"] <= 0.02 for row in rows)
            and all(row["metrics"]["maximum_row_relative_l2"] <= 0.05 for row in rows)
            and monotonic
            and physical["combined_to_source_layer_bank_ratio"] <= 0.60
            and physical["repair_to_source_expert_mac_ratio"] <= 0.05
        )
        if passes:
            passing.append(rank)
        rows_by_rank.append(
            {
                "rank": rank,
                "aggregate_relative_l2": aggregate[rank],
                "maximum_layer_relative_l2": max(row["metrics"]["relative_l2"] for row in rows),
                "maximum_row_relative_l2": max(row["metrics"]["maximum_row_relative_l2"] for row in rows),
                "physical": physical,
                "passes": passes,
            }
        )
    return {
        "ranks": rows_by_rank,
        "rank_monotonic_at_every_layer": monotonic,
        "smallest_passing_rank": min(passing) if passing else None,
        "passes": bool(passing),
        "thresholds": {
            "aggregate_relative_l2": 0.01,
            "layer_relative_l2": 0.02,
            "row_relative_l2": 0.05,
            "combined_source_byte_ratio": 0.60,
            "repair_source_expert_mac_ratio": 0.05,
        },
    }


def run(
    checkpoint_root: Path,
    verification_path: Path,
    corpus_manifest_path: Path,
    pw0129_path: Path,
    pw0130_path: Path,
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
        (pw0130_path, PW0130_SHA256, "PW-0130 report"),
    ):
        if sha256_file(path) != expected:
            raise ValueError(f"PW-0131 {label} hash mismatch")
    started = time.perf_counter()
    safety = HostSafetyMonitor()
    corpus = json.loads(corpus_manifest_path.read_text())
    prior = json.loads(pw0129_path.read_text())
    affine_authority = json.loads(pw0130_path.read_text())
    if (
        corpus.get("revision") != REVISION
        or prior.get("decision") != "reject_naive_affine_int4_on_real_validation"
        or affine_authority.get("decision") != "reject_int4_diagonal_output_affine_repair"
        or prior.get("holdout_unsealed")
        or affine_authority.get("holdout_unsealed")
    ):
        raise ValueError("PW-0131 authority identity mismatch")
    checkpoint = ShardedCheckpoint(checkpoint_root, verification_path)
    root = corpus_manifest_path.parent
    rank_reports = []
    affine_reproductions = []
    for layer in LAYERS:
        authority = next(row for row in corpus["layers"] if row["layer"] == layer)
        validate_routes(authority)
        moe_input = load_capture(root, authority["captures"]["moe_input"])
        expert_down = load_capture(root, authority["captures"]["expert_down"])
        routed_expected = load_capture(root, authority["captures"]["routed_output"])
        rows = _collect_layer(
            checkpoint, authority, moe_input, expert_down, prior, layer, safety
        )
        affine, repaired_by_rank, factor_hashes = _build_affine_and_rank_repairs(rows, moe_input)
        affine_routed = _route_rows(affine["rows"], authority, "repaired", 112, 168)
        affine_metrics = error_metrics(
            affine_routed, np.asarray(routed_expected[112:168], dtype=np.float32)
        )
        expected_affine = next(
            row["affine"] for row in affine_authority["layer_reports"] if row["layer"] == layer
        )
        if (
            affine_metrics != expected_affine["validation_metrics"]
            or _parameter_digest(affine["parameters"])
            != expected_affine["fitted_parameter_sha256"]
        ):
            raise ValueError("PW-0131 PW-0130 affine authority mismatch")
        affine_reproductions.append({"layer": layer, "metrics": affine_metrics})
        for rank in RANKS:
            routed = _route_rows(
                repaired_by_rank[rank], authority, "repaired", 112, 168
            )
            metrics = error_metrics(
                routed, np.asarray(routed_expected[112:168], dtype=np.float32)
            )
            rank_reports.append(
                {
                    "layer": layer,
                    "rank": rank,
                    "factor_sha256": factor_hashes[rank],
                    "metrics": metrics,
                }
            )
            safety.checkpoint(f"layer_{layer}_rank_{rank}_oracle_complete")
        del moe_input, expert_down, routed_expected, rows, affine, repaired_by_rank
        gc.collect()
        safety.release_checkpoint(
            f"layer_{layer}_repair_buffers_released",
            ["real corpus", "INT4 outputs", "affine repair", "low-rank factors"],
        )
    gate = _capacity_gate(rank_reports)
    decision = (
        "authorize_train_only_int4_low_rank_repair"
        if gate["passes"]
        else "reject_int4_input_conditioned_low_rank_output_repair"
    )
    safety.release_checkpoint(
        "checkpoint_and_authorities_released",
        ["checkpoint mappings", "PW-0116 corpus", "PW-0129/PW-0130 reports"],
    )
    safety.checkpoint("final_service_health")
    report = {
        "schema_version": 1,
        "evidence_class": "pw0131_int4_input_conditioned_low_rank_repair_oracle",
        "revision": REVISION,
        "commit": commit,
        "source_hashes": {
            "checkpoint_verification_sha256": VERIFICATION_SHA256,
            "corpus_manifest_sha256": CORPUS_SHA256,
            "pw0129_report_sha256": PW0129_SHA256,
            "pw0130_report_sha256": PW0130_SHA256,
        },
        "ranks": list(RANKS),
        "affine_reproductions": affine_reproductions,
        "rank_reports": rank_reports,
        "capacity_gate": gate,
        "holdout_unsealed": False,
        "decision": decision,
        "safety_snapshots": safety.evidence(),
        "complete_wall_ms": (time.perf_counter() - started) * 1000.0,
        "accepted_tokens": 0,
        "A": 0,
        "performance_claim": None,
        "limitations": (
            "same-validation noncausal F16 low-rank output-repair capacity oracle on one "
            "English pilot prefix; no train-only generalization, holdout, accumulated model, "
            "modality corpus, measured repair kernel, endpoint, or TPS claim"
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
    parser.add_argument("--pw0130", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--commit", required=True)
    arguments = parser.parse_args()
    try:
        result = run(
            arguments.checkpoint,
            arguments.verification,
            arguments.corpus_manifest,
            arguments.pw0129,
            arguments.pw0130,
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
