#!/usr/bin/env python3
"""Run PW-0137's one-expert global-Hessian fixed-grid GPTQ rescue."""

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
import torch

try:
    from tools.generate_real_layer1_expert_oracle import ShardedCheckpoint
    from tools.host_safety import HostSafetyMonitor
    from tools.openrouter_reference import atomic_write_new, canonical_json
    from tools.run_best_rank_real_expert_control import dequant_weight, load_capture, sha256_file, source_linear
    from tools.run_group_local_gptq_three_expert_control import (
        affine_grid,
        dense_expert,
        dense_projection,
        physical_ledger,
        reconstruct_fixed_grid,
        source_hidden,
        validate_grid_membership,
    )
    from tools.run_real_activation_affine_int4_audit import (
        CORPUS_SHA256,
        GROUP_SIZE,
        REVISION,
        VERIFICATION_SHA256,
        error_metrics,
        validate_routes,
    )
except ModuleNotFoundError:
    from generate_real_layer1_expert_oracle import ShardedCheckpoint
    from host_safety import HostSafetyMonitor
    from openrouter_reference import atomic_write_new, canonical_json
    from run_best_rank_real_expert_control import dequant_weight, load_capture, sha256_file, source_linear
    from run_group_local_gptq_three_expert_control import (
        affine_grid,
        dense_expert,
        dense_projection,
        physical_ledger,
        reconstruct_fixed_grid,
        source_hidden,
        validate_grid_membership,
    )
    from run_real_activation_affine_int4_audit import (
        CORPUS_SHA256,
        GROUP_SIZE,
        REVISION,
        VERIFICATION_SHA256,
        error_metrics,
        validate_routes,
    )


PW0135_SHA256 = "56b9d38c3c630359b8d5b1a911627882df06a2e2fc374751fde2fddaeb3888db"
LAYER = 46
EXPERT = 28
TRAIN_PLACEMENTS = 100
VALIDATION_PLACEMENTS = 56
DAMPING = 0.001
BLOCK_SIZE = 128
PW0135_VALIDATION_RELATIVE_L2 = 0.080659


def calibration_positions(positions: list[int]) -> tuple[list[int], list[int]]:
    if any(not isinstance(position, int) or not 0 <= position < 168 for position in positions):
        raise ValueError("PW-0137 partition positions are invalid")
    return (
        [index for index, position in enumerate(positions) if position < 112],
        [index for index, position in enumerate(positions) if 112 <= position < 168],
    )


def projected_workspace_bytes(weight: np.ndarray) -> int:
    if weight.ndim != 2:
        raise ValueError("PW-0137 projected workspace requires a matrix")
    rows, columns = weight.shape
    # Six full float64 Hessian-sized workspaces conservatively cover formation,
    # permutation, factorization, inversion, and temporary LAPACK storage. Four
    # float64 weight-sized workspaces cover the permuted matrix and block updates.
    return 6 * columns * columns * 8 + 4 * rows * columns * 8


def global_hessian_gptq_fixed_grid(
    weight: np.ndarray,
    activations: np.ndarray,
    scales: np.ndarray,
    biases: np.ndarray,
    *,
    damping: float = DAMPING,
    block_size: int = BLOCK_SIZE,
) -> tuple[np.ndarray, np.ndarray, dict]:
    if (
        weight.ndim != 2
        or activations.ndim != 2
        or activations.shape[1] != weight.shape[1]
        or activations.shape[0] == 0
        or scales.shape != biases.shape
        or scales.shape[0] != weight.shape[0]
        or weight.shape[1] % scales.shape[1]
        or damping <= 0
        or block_size <= 0
        or not np.isfinite(weight).all()
        or not np.isfinite(activations).all()
    ):
        raise ValueError("PW-0137 global-Hessian inputs are invalid")
    rows, columns = weight.shape
    group_size = columns // scales.shape[1]
    if group_size != GROUP_SIZE:
        raise ValueError("PW-0137 requires the frozen group-128 grid")

    x = activations.astype(np.float64)
    hessian = (x.T @ x) / x.shape[0]
    diagonal = np.diag(hessian).copy()
    dead = diagonal == 0
    if np.any(dead):
        dead_indices = np.flatnonzero(dead)
        hessian[dead_indices, dead_indices] = 1.0
    mean_diagonal = float(np.mean(np.diag(hessian)))
    damp = max(damping * mean_diagonal, 1e-8)
    hessian.flat[:: columns + 1] += damp

    permutation = np.argsort(-diagonal, kind="stable")
    permuted_hessian = hessian[np.ix_(permutation, permutation)]
    inverse = np.linalg.inv(permuted_hessian)
    inverse_cholesky = np.linalg.cholesky(inverse).T
    del x, hessian, permuted_hessian, inverse

    working = weight.astype(np.float64)[:, permutation].copy()
    quantized = np.empty(weight.shape, dtype=np.float16)
    codes = np.empty(weight.shape, dtype=np.uint8)
    cross_block_update_squared = 0.0
    block_count = 0
    for start in range(0, columns, block_size):
        end = min(start + block_size, columns)
        count = end - start
        block_count += 1
        block_weight = working[:, start:end].copy()
        block_error = np.empty((rows, count), dtype=np.float64)
        block_factor = inverse_cholesky[start:end, start:end]
        for local_column in range(count):
            ordered_column = start + local_column
            original_column = int(permutation[ordered_column])
            group = original_column // group_size
            scale = scales[:, group].astype(np.float64)
            bias = biases[:, group].astype(np.float64)
            safe_scale = np.where(scale == 0, 1.0, scale)
            raw_code = np.clip(
                np.rint((block_weight[:, local_column] - bias) / safe_scale), 0, 15
            )
            raw_code = np.where(scale == 0, 0, raw_code)
            value = raw_code * scale + bias
            codes[:, original_column] = raw_code.astype(np.uint8)
            quantized[:, original_column] = value.astype(np.float16)
            error = (block_weight[:, local_column] - value) / block_factor[local_column, local_column]
            block_weight[:, local_column:] -= (
                error[:, None] * block_factor[local_column, local_column:][None, :]
            )
            block_error[:, local_column] = error
        if end < columns:
            update = block_error @ inverse_cholesky[start:end, end:]
            cross_block_update_squared += float(np.sum(update * update, dtype=np.float64))
            working[:, end:] -= update
        del block_weight, block_error, block_factor

    validate_grid_membership(codes, quantized, scales, biases)
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(codes).tobytes())
    digest.update(np.ascontiguousarray(scales.astype("<f4")).tobytes())
    digest.update(np.ascontiguousarray(biases.astype("<f4")).tobytes())
    return quantized, codes, {
        "damping": damping,
        "block_size": block_size,
        "block_count": block_count,
        "dead_activation_columns": int(np.count_nonzero(dead)),
        "cross_block_update_l2": cross_block_update_squared**0.5,
        "activation_order_sha256": hashlib.sha256(
            np.ascontiguousarray(permutation.astype("<u4")).tobytes()
        ).hexdigest(),
        "grid_sha256": digest.hexdigest(),
    }


def _prior_report(prior: dict) -> dict:
    matches = [
        report
        for report in prior.get("reports", [])
        if report.get("layer") == LAYER and report.get("expert") == EXPERT
    ]
    if len(matches) != 1:
        raise ValueError("PW-0137 PW-0135 expert authority mismatch")
    return matches[0]


def _gate(report: dict) -> dict:
    baseline = report["dense_control_validation"]["relative_l2"]
    candidate = report["global_gptq_validation"]["relative_l2"]
    reduction = 1.0 - candidate / max(baseline, 1e-30)
    conditions = {
        "validation_relative_error_reduction": reduction >= 0.50,
        "validation_relative_l2": candidate <= 0.08,
        "maximum_validation_row_relative_l2": (
            report["global_gptq_validation"]["maximum_row_relative_l2"] <= 0.12
        ),
        "train_improves": (
            report["global_gptq_train"]["relative_l2"]
            < report["dense_control_train"]["relative_l2"]
        ),
        "no_worse_than_pw0135": candidate <= PW0135_VALIDATION_RELATIVE_L2,
        "physical_ledger": (
            physical_ledger()["packed_bytes_per_expert"] == 13_369_344
            and physical_ledger()["additional_runtime_macs"] == 0
        ),
    }
    return {
        "passes": all(conditions.values()),
        "conditions": conditions,
        "observed": {
            "validation_relative_error_reduction": reduction,
            "validation_relative_l2": candidate,
            "maximum_validation_row_relative_l2": report["global_gptq_validation"][
                "maximum_row_relative_l2"
            ],
            "train_relative_l2": report["global_gptq_train"]["relative_l2"],
            "pw0135_validation_relative_l2": PW0135_VALIDATION_RELATIVE_L2,
        },
        "thresholds": {
            "minimum_validation_relative_error_reduction": 0.50,
            "maximum_validation_relative_l2": 0.08,
            "maximum_validation_row_relative_l2": 0.12,
            "train_must_improve": True,
            "maximum_pw0135_validation_relative_l2": PW0135_VALIDATION_RELATIVE_L2,
        },
        "physical": physical_ledger(),
    }


def run(
    checkpoint_root: Path,
    verification_path: Path,
    corpus_manifest_path: Path,
    pw0135_path: Path,
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
        (pw0135_path, PW0135_SHA256, "PW-0135 report"),
    ):
        if sha256_file(path) != expected:
            raise ValueError(f"PW-0137 {label} hash mismatch")

    started = time.perf_counter()
    safety = HostSafetyMonitor()
    corpus = json.loads(corpus_manifest_path.read_text())
    prior = json.loads(pw0135_path.read_text())
    prior_report = _prior_report(prior)
    if (
        corpus.get("revision") != REVISION
        or prior.get("decision") != "reject_group_local_fixed_grid_gptq"
        or prior.get("holdout_unsealed")
    ):
        raise ValueError("PW-0137 authority identity mismatch")

    checkpoint = ShardedCheckpoint(checkpoint_root, verification_path)
    root = corpus_manifest_path.parent
    authority = next(row for row in corpus["layers"] if row["layer"] == LAYER)
    validate_routes(authority)
    schedule = next(row for row in authority["expert_schedule"] if row["expert"] == EXPERT)
    train_local, validation_local = calibration_positions(schedule["positions"])
    if len(train_local) != TRAIN_PLACEMENTS or len(validation_local) != VALIDATION_PLACEMENTS:
        raise ValueError("PW-0137 frozen sample coverage mismatch")
    offset = sum(
        len(row["positions"])
        for row in authority["expert_schedule"][: authority["expert_schedule"].index(schedule)]
    )
    moe_input = load_capture(root, authority["captures"]["moe_input"])
    expert_down = load_capture(root, authority["captures"]["expert_down"])
    train_ids = [schedule["positions"][index] for index in train_local]
    validation_ids = [schedule["positions"][index] for index in validation_local]
    train_expected = np.asarray(
        expert_down[[offset + index for index in train_local]], dtype=np.float32
    ).copy()
    validation_expected = np.asarray(
        expert_down[[offset + index for index in validation_local]], dtype=np.float32
    ).copy()
    prefix = f"model.layers.{LAYER}.mlp.experts.{EXPERT}"
    source_weights = {
        projection: dequant_weight(checkpoint, f"{prefix}.{projection}_proj.weight")
        for projection in ("gate", "up", "down")
    }
    hidden = source_hidden(source_weights, moe_input[train_ids])
    projection_activations = {
        "gate": moe_input[train_ids],
        "up": moe_input[train_ids],
        "down": hidden,
    }
    projection_expected = {
        "gate": source_linear(
            source_weights["gate"],
            torch.from_numpy(np.asarray(moe_input[train_ids]).copy()).to(torch.bfloat16),
        ).float().numpy(),
        "up": source_linear(
            source_weights["up"],
            torch.from_numpy(np.asarray(moe_input[train_ids]).copy()).to(torch.bfloat16),
        ).float().numpy(),
        "down": train_expected,
    }

    selected_weights: dict[str, np.ndarray] = {}
    baseline_weights: dict[str, np.ndarray] = {}
    projection_reports = {}
    for projection in ("gate", "up", "down"):
        weight = source_weights[projection]
        scales, biases, baseline, baseline_codes = affine_grid(weight)
        validate_grid_membership(baseline_codes, baseline, scales, biases)
        projected = projected_workspace_bytes(weight)
        before = safety.checkpoint(f"{projection}_global_hessian_preflight")
        if before.process_physical_footprint_bytes + projected > safety.policy.maximum_process_physical_footprint_bytes:
            raise RuntimeError(f"PW-0137 {projection} projected workspace exceeds Gate 8 headroom")
        candidate, codes, diagnostics = global_hessian_gptq_fixed_grid(
            weight,
            projection_activations[projection],
            scales,
            biases,
        )
        candidate_output = dense_projection(projection_activations[projection], candidate)
        baseline_output = dense_projection(projection_activations[projection], baseline)
        selected_weights[projection] = candidate
        baseline_weights[projection] = baseline
        projection_reports[projection] = {
            **diagnostics,
            "projected_workspace_bytes": projected,
            "candidate_train_metrics": error_metrics(
                candidate_output, projection_expected[projection]
            ),
            "baseline_train_metrics": error_metrics(
                baseline_output, projection_expected[projection]
            ),
        }
        del scales, biases, baseline_codes, codes, candidate_output, baseline_output
        gc.collect()
        safety.release_checkpoint(
            f"{projection}_global_hessian_workspace_released",
            ["full Hessian", "inverse Cholesky", "permuted working weight", "projection outputs"],
        )

    dense_control_train = dense_expert(moe_input[train_ids], baseline_weights)
    dense_control_validation = dense_expert(moe_input[validation_ids], baseline_weights)
    global_train = dense_expert(moe_input[train_ids], selected_weights)
    global_validation = dense_expert(moe_input[validation_ids], selected_weights)
    report = {
        "layer": LAYER,
        "expert": EXPERT,
        "train_placements": len(train_ids),
        "validation_placements": len(validation_ids),
        "projection_reports": projection_reports,
        "dense_control_train": error_metrics(dense_control_train, train_expected),
        "dense_control_validation": error_metrics(dense_control_validation, validation_expected),
        "global_gptq_train": error_metrics(global_train, train_expected),
        "global_gptq_validation": error_metrics(global_validation, validation_expected),
        "pw0135_group_local_train": prior_report["gptq_train"],
        "pw0135_group_local_validation": prior_report["gptq_validation"],
    }
    if report["dense_control_train"] != prior_report["dense_control_train"]:
        raise ValueError("PW-0137 dense train control differs from PW-0135")
    if report["dense_control_validation"] != prior_report["dense_control_validation"]:
        raise ValueError("PW-0137 dense validation control differs from PW-0135")
    gate = _gate(report)
    decision = (
        "authorize_three_expert_global_hessian_confirmation"
        if gate["passes"]
        else "reject_global_hessian_fixed_grid_gptq"
    )

    del source_weights, hidden, projection_activations, projection_expected
    del selected_weights, baseline_weights, dense_control_train, dense_control_validation
    del global_train, global_validation, moe_input, expert_down, train_expected, validation_expected
    gc.collect()
    mx.clear_cache()
    safety.release_checkpoint(
        "one_expert_control_released",
        ["source expert", "candidate expert", "dense controls", "captured activations"],
    )
    safety.checkpoint("final_service_health")
    result = {
        "schema_version": 1,
        "evidence_class": "pw0137_global_hessian_fixed_grid_gptq_rescue",
        "revision": REVISION,
        "commit": commit,
        "source_hashes": {
            "checkpoint_verification_sha256": VERIFICATION_SHA256,
            "corpus_manifest_sha256": CORPUS_SHA256,
            "pw0135_report_sha256": PW0135_SHA256,
        },
        "mechanism": {
            "damping": DAMPING,
            "order": "descending_full_hessian_diagonal",
            "block_size": BLOCK_SIZE,
            "static_original_group_grid": True,
            "cross_block_error_propagation": True,
        },
        "report": report,
        "continuation_gate": gate,
        "holdout_unsealed": False,
        "decision": decision,
        "safety_snapshots": safety.evidence(),
        "complete_wall_ms": (time.perf_counter() - started) * 1000.0,
        "accepted_tokens": 0,
        "A": 0,
        "performance_claim": None,
        "limitations": (
            "one previously failing expert only; fixed affine-INT4 grids; dense unpacked "
            "execution oracle; one English pilot; holdout sealed; no three-expert or full-layer "
            "audit, runtime artifact, kernel, accumulated model, modalities, endpoint, or TPS"
        ),
        "platform": platform.platform(),
    }
    atomic_write_new(output_path, canonical_json(result))
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--verification", required=True, type=Path)
    parser.add_argument("--corpus-manifest", required=True, type=Path)
    parser.add_argument("--pw0135", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--commit", required=True)
    arguments = parser.parse_args()
    try:
        result = run(
            arguments.checkpoint,
            arguments.verification,
            arguments.corpus_manifest,
            arguments.pw0135,
            arguments.output,
            arguments.commit,
        )
        print(json.dumps({"output": str(arguments.output), "decision": result["decision"]}))
        return 0
    except (OSError, RuntimeError, ValueError, KeyError, TypeError, json.JSONDecodeError, np.linalg.LinAlgError) as error:
        print(json.dumps({"error": str(error)}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
