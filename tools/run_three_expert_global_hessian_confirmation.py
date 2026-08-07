#!/usr/bin/env python3
"""Run PW-0138's three-expert global-Hessian GPTQ confirmation."""

from __future__ import annotations

import argparse
import gc
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
    from tools.run_global_hessian_gptq_rescue import (
        BLOCK_SIZE,
        DAMPING,
        global_hessian_gptq_fixed_grid,
        projected_workspace_bytes,
    )
    from tools.run_group_local_gptq_three_expert_control import (
        SAMPLES,
        affine_grid,
        dense_expert,
        dense_projection,
        physical_ledger,
        source_hidden,
        validate_grid_membership,
    )
    from tools.run_real_activation_affine_int4_audit import (
        CORPUS_SHA256,
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
    from run_global_hessian_gptq_rescue import (
        BLOCK_SIZE,
        DAMPING,
        global_hessian_gptq_fixed_grid,
        projected_workspace_bytes,
    )
    from run_group_local_gptq_three_expert_control import (
        SAMPLES,
        affine_grid,
        dense_expert,
        dense_projection,
        physical_ledger,
        source_hidden,
        validate_grid_membership,
    )
    from run_real_activation_affine_int4_audit import (
        CORPUS_SHA256,
        REVISION,
        VERIFICATION_SHA256,
        error_metrics,
        validate_routes,
    )


PW0135_SHA256 = "56b9d38c3c630359b8d5b1a911627882df06a2e2fc374751fde2fddaeb3888db"
PW0137_SHA256 = "95fee340bb676ac7c9486ea713da9c461ca6fb62441b41b32ff988e97ed1502e"
PW0137_ANALYSIS_SHA256 = "7a741514aad2f4ec783cd95b1283ae5b98afbcdad17cd64e8a7759c12f3b5d67"


def _prior_report(prior: dict, layer: int, expert: int) -> dict:
    matches = [
        report
        for report in prior.get("reports", [])
        if report.get("layer") == layer and report.get("expert") == expert
    ]
    if len(matches) != 1:
        raise ValueError("PW-0138 prior expert authority mismatch")
    return matches[0]


def _gate(reports: list[dict]) -> dict:
    expert_gates = []
    for report in reports:
        baseline = report["dense_control_validation"]["relative_l2"]
        candidate = report["global_gptq_validation"]["relative_l2"]
        prior = report["pw0135_group_local_validation"]["relative_l2"]
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
            "no_worse_than_pw0135": candidate <= prior,
        }
        expert_gates.append(
            {
                "layer": report["layer"],
                "expert": report["expert"],
                "passes": all(conditions.values()),
                "conditions": conditions,
                "validation_relative_error_reduction": reduction,
                "candidate_validation_relative_l2": candidate,
                "candidate_maximum_row_relative_l2": report["global_gptq_validation"][
                    "maximum_row_relative_l2"
                ],
            }
        )
    physical = physical_ledger()
    physical_passes = (
        physical["packed_bytes_per_expert"] == 13_369_344
        and physical["additional_runtime_macs"] == 0
    )
    return {
        "passes": all(row["passes"] for row in expert_gates) and physical_passes,
        "experts": expert_gates,
        "physical": physical,
        "physical_passes": physical_passes,
        "thresholds": {
            "minimum_validation_relative_error_reduction": 0.50,
            "maximum_validation_relative_l2": 0.08,
            "maximum_validation_row_relative_l2": 0.12,
            "train_must_improve": True,
            "must_not_regress_pw0135": True,
        },
    }


def _evaluate_expert(
    checkpoint: ShardedCheckpoint,
    root: Path,
    authority: dict,
    layer: int,
    expert: int,
    expected_train_count: int,
    expected_validation_count: int,
    prior_report: dict,
    safety: HostSafetyMonitor,
) -> dict:
    validate_routes(authority)
    schedule = next(row for row in authority["expert_schedule"] if row["expert"] == expert)
    train_local = [index for index, position in enumerate(schedule["positions"]) if position < 112]
    validation_local = [
        index for index, position in enumerate(schedule["positions"]) if 112 <= position < 168
    ]
    if len(train_local) != expected_train_count or len(validation_local) != expected_validation_count:
        raise ValueError("PW-0138 frozen sample coverage mismatch")
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
    prefix = f"model.layers.{layer}.mlp.experts.{expert}"
    source_weights = {
        projection: dequant_weight(checkpoint, f"{prefix}.{projection}_proj.weight")
        for projection in ("gate", "up", "down")
    }
    hidden = source_hidden(source_weights, moe_input[train_ids])
    activations = {"gate": moe_input[train_ids], "up": moe_input[train_ids], "down": hidden}
    expected = {
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
    selected = {}
    baseline = {}
    projection_reports = {}
    for projection in ("gate", "up", "down"):
        weight = source_weights[projection]
        scales, biases, control, control_codes = affine_grid(weight)
        validate_grid_membership(control_codes, control, scales, biases)
        projected = projected_workspace_bytes(weight)
        before = safety.checkpoint(f"layer_{layer}_expert_{expert}_{projection}_preflight")
        if before.process_physical_footprint_bytes + projected > safety.policy.maximum_process_physical_footprint_bytes:
            raise RuntimeError(f"PW-0138 layer {layer} {projection} exceeds Gate 8 headroom")
        candidate, codes, diagnostics = global_hessian_gptq_fixed_grid(
            weight, activations[projection], scales, biases
        )
        candidate_output = dense_projection(activations[projection], candidate)
        control_output = dense_projection(activations[projection], control)
        selected[projection] = candidate
        baseline[projection] = control
        projection_reports[projection] = {
            **diagnostics,
            "projected_workspace_bytes": projected,
            "candidate_train_metrics": error_metrics(candidate_output, expected[projection]),
            "baseline_train_metrics": error_metrics(control_output, expected[projection]),
        }
        del scales, biases, control_codes, codes, candidate_output, control_output
        gc.collect()
        safety.release_checkpoint(
            f"layer_{layer}_expert_{expert}_{projection}_workspace_released",
            ["full Hessian", "inverse Cholesky", "permuted working weight", "projection outputs"],
        )
    dense_train = dense_expert(moe_input[train_ids], baseline)
    dense_validation = dense_expert(moe_input[validation_ids], baseline)
    global_train = dense_expert(moe_input[train_ids], selected)
    global_validation = dense_expert(moe_input[validation_ids], selected)
    result = {
        "layer": layer,
        "expert": expert,
        "train_placements": len(train_ids),
        "validation_placements": len(validation_ids),
        "projection_reports": projection_reports,
        "dense_control_train": error_metrics(dense_train, train_expected),
        "dense_control_validation": error_metrics(dense_validation, validation_expected),
        "global_gptq_train": error_metrics(global_train, train_expected),
        "global_gptq_validation": error_metrics(global_validation, validation_expected),
        "pw0135_group_local_train": prior_report["gptq_train"],
        "pw0135_group_local_validation": prior_report["gptq_validation"],
    }
    if result["dense_control_train"] != prior_report["dense_control_train"]:
        raise ValueError("PW-0138 dense train control differs from PW-0135")
    if result["dense_control_validation"] != prior_report["dense_control_validation"]:
        raise ValueError("PW-0138 dense validation control differs from PW-0135")
    del source_weights, hidden, activations, expected, selected, baseline
    del dense_train, dense_validation, global_train, global_validation
    del moe_input, expert_down, train_expected, validation_expected
    gc.collect()
    mx.clear_cache()
    safety.release_checkpoint(
        f"layer_{layer}_expert_{expert}_complete_released",
        ["source expert", "candidate expert", "dense controls", "captured activations"],
    )
    return result


def run(
    checkpoint_root: Path,
    verification_path: Path,
    corpus_manifest_path: Path,
    pw0135_path: Path,
    pw0137_path: Path,
    pw0137_analysis_path: Path,
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
        (pw0137_path, PW0137_SHA256, "PW-0137 report"),
        (pw0137_analysis_path, PW0137_ANALYSIS_SHA256, "PW-0137 analysis"),
    ):
        if sha256_file(path) != expected:
            raise ValueError(f"PW-0138 {label} hash mismatch")
    started = time.perf_counter()
    safety = HostSafetyMonitor()
    corpus = json.loads(corpus_manifest_path.read_text())
    prior = json.loads(pw0135_path.read_text())
    rescue = json.loads(pw0137_path.read_text())
    rescue_analysis = json.loads(pw0137_analysis_path.read_text())
    if (
        corpus.get("revision") != REVISION
        or prior.get("decision") != "reject_group_local_fixed_grid_gptq"
        or rescue.get("decision") != "authorize_three_expert_global_hessian_confirmation"
        or rescue_analysis.get("decision") != rescue.get("decision")
        or rescue_analysis.get("source_report_sha256") != PW0137_SHA256
        or prior.get("holdout_unsealed")
        or rescue.get("holdout_unsealed")
        or rescue_analysis.get("holdout_unsealed")
    ):
        raise ValueError("PW-0138 authority identity mismatch")
    checkpoint = ShardedCheckpoint(checkpoint_root, verification_path)
    root = corpus_manifest_path.parent
    reports = []
    for layer, expert, train_count, validation_count in SAMPLES:
        authority = next(row for row in corpus["layers"] if row["layer"] == layer)
        reports.append(
            _evaluate_expert(
                checkpoint,
                root,
                authority,
                layer,
                expert,
                train_count,
                validation_count,
                _prior_report(prior, layer, expert),
                safety,
            )
        )
    layer46 = reports[2]
    rescue_report = rescue["report"]
    if (
        layer46["global_gptq_train"] != rescue_report["global_gptq_train"]
        or layer46["global_gptq_validation"] != rescue_report["global_gptq_validation"]
        or {
            name: (row["grid_sha256"], row["activation_order_sha256"])
            for name, row in layer46["projection_reports"].items()
        }
        != {
            name: (row["grid_sha256"], row["activation_order_sha256"])
            for name, row in rescue_report["projection_reports"].items()
        }
    ):
        raise ValueError("PW-0138 does not reproduce PW-0137")
    gate = _gate(reports)
    decision = (
        "authorize_all_validation_expert_global_hessian_audit"
        if gate["passes"]
        else "reject_three_expert_global_hessian_gptq"
    )
    safety.release_checkpoint(
        "checkpoint_and_authorities_released",
        ["checkpoint mappings", "PW-0116 corpus", "PW-0135/PW-0137 authorities"],
    )
    safety.checkpoint("final_service_health")
    result = {
        "schema_version": 1,
        "evidence_class": "pw0138_three_expert_global_hessian_gptq_confirmation",
        "revision": REVISION,
        "commit": commit,
        "source_hashes": {
            "checkpoint_verification_sha256": VERIFICATION_SHA256,
            "corpus_manifest_sha256": CORPUS_SHA256,
            "pw0135_report_sha256": PW0135_SHA256,
            "pw0137_report_sha256": PW0137_SHA256,
            "pw0137_analysis_sha256": PW0137_ANALYSIS_SHA256,
        },
        "samples": [list(row) for row in SAMPLES],
        "mechanism": {
            "damping": DAMPING,
            "order": "descending_full_hessian_diagonal",
            "block_size": BLOCK_SIZE,
            "static_original_group_grid": True,
            "cross_block_error_propagation": True,
        },
        "reports": reports,
        "pw0137_exact_reproduction": True,
        "continuation_gate": gate,
        "holdout_unsealed": False,
        "decision": decision,
        "safety_snapshots": safety.evidence(),
        "complete_wall_ms": (time.perf_counter() - started) * 1000.0,
        "accepted_tokens": 0,
        "A": 0,
        "performance_claim": None,
        "limitations": (
            "three highest-validation-coverage experts only; fixed affine-INT4 grids; dense "
            "unpacked execution oracle; one English pilot; holdout sealed; no all-expert audit, "
            "runtime artifact, kernel, accumulated model, modalities, endpoint, or TPS"
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
    parser.add_argument("--pw0137", required=True, type=Path)
    parser.add_argument("--pw0137-analysis", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--commit", required=True)
    args = parser.parse_args()
    try:
        result = run(
            args.checkpoint,
            args.verification,
            args.corpus_manifest,
            args.pw0135,
            args.pw0137,
            args.pw0137_analysis,
            args.output,
            args.commit,
        )
        print(json.dumps({"output": str(args.output), "decision": result["decision"]}))
        return 0
    except (OSError, RuntimeError, ValueError, KeyError, TypeError, json.JSONDecodeError, np.linalg.LinAlgError) as error:
        print(json.dumps({"error": str(error)}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
