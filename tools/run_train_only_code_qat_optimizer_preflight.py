#!/usr/bin/env python3
"""Run PW-0145's validation-sealed code-QAT optimizer preflight."""

from __future__ import annotations

import argparse
import gc
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
    from tools.run_best_rank_real_expert_control import dequant_weight, load_capture, sha256_file
    from tools.run_fixed_residual_hadamard_rotation_control import _prior_expert
    from tools.run_global_hessian_gptq_rescue import (
        global_hessian_gptq_fixed_grid,
        projected_workspace_bytes,
    )
    from tools.run_group_local_gptq_three_expert_control import (
        affine_grid,
        dense_expert,
        physical_ledger,
        reconstruct_fixed_grid,
        source_hidden,
    )
    from tools.run_real_activation_affine_int4_audit import (
        CORPUS_SHA256,
        REVISION,
        VERIFICATION_SHA256,
        error_metrics,
        validate_routes,
    )
    from tools.run_train_only_code_changing_int4_qat import (
        materialize_qat_weights,
        train_code_offsets,
    )
except ModuleNotFoundError:
    from generate_real_layer1_expert_oracle import ShardedCheckpoint
    from host_safety import HostSafetyMonitor
    from openrouter_reference import atomic_write_new, canonical_json
    from run_best_rank_real_expert_control import dequant_weight, load_capture, sha256_file
    from run_fixed_residual_hadamard_rotation_control import _prior_expert
    from run_global_hessian_gptq_rescue import global_hessian_gptq_fixed_grid, projected_workspace_bytes
    from run_group_local_gptq_three_expert_control import (
        affine_grid,
        dense_expert,
        physical_ledger,
        reconstruct_fixed_grid,
        source_hidden,
    )
    from run_real_activation_affine_int4_audit import (
        CORPUS_SHA256,
        REVISION,
        VERIFICATION_SHA256,
        error_metrics,
        validate_routes,
    )
    from run_train_only_code_changing_int4_qat import materialize_qat_weights, train_code_offsets


PW0139_SHA256 = "83bd204c9d5c35a684cab15a4ddacf48cf9b661563fb26223eb3655d0ef4a7b5"
PW0144_SHA256 = "8828db18f3d9471aa9abd2110b78994b86803ff393e4ec0d9fe81e10cef5d00c"
LAYER = 46
EXPERT = 249
TRAIN_PLACEMENTS = 90
LEARNING_RATES = (0.0001, 0.0005, 0.001, 0.005)
STEPS = 32
SAFETY_INTERVAL = 8


def select_trial(trials: list[dict]) -> dict:
    if [row.get("learning_rate") for row in trials] != list(LEARNING_RATES):
        raise ValueError("PW-0145 trial authority mismatch")
    return min(
        trials,
        key=lambda row: (row["train_metrics"]["relative_l2"], row["learning_rate"]),
    )


def _gate(initial_metrics: dict, selected: dict) -> dict:
    changed = sum(selected["changed_codes"].values())
    total = sum(selected["code_totals"].values())
    physical = physical_ledger()
    conditions = {
        "minimum_train_improvement": (
            selected["train_metrics"]["relative_l2"]
            <= initial_metrics["relative_l2"] * 0.75
        ),
        "finite_decreasing_loss": selected["training"]["loss_decreased"],
        "bounded_code_changes": 0 < changed <= int(total * 0.05),
        "code_domain": selected["code_domain_valid"],
        "grid_metadata_unchanged": selected["grid_metadata_unchanged"],
        "physical_ledger": (
            physical["packed_bytes_per_expert"] == 13_369_344
            and physical["additional_runtime_macs"] == 0
        ),
    }
    return {
        "passes": all(conditions.values()),
        "conditions": conditions,
        "initial_train_relative_l2": initial_metrics["relative_l2"],
        "selected_train_relative_l2": selected["train_metrics"]["relative_l2"],
        "selected_learning_rate": selected["learning_rate"],
        "changed_codes": changed,
        "total_codes": total,
        "changed_fraction": changed / total,
        "physical": physical,
    }


def run(
    checkpoint_root: Path,
    verification_path: Path,
    corpus_manifest_path: Path,
    pw0139_path: Path,
    pw0144_path: Path,
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
        (pw0139_path, PW0139_SHA256, "PW-0139 report"),
        (pw0144_path, PW0144_SHA256, "PW-0144 report"),
    ):
        if sha256_file(path) != expected:
            raise ValueError(f"PW-0145 {label} hash mismatch")
    started = time.perf_counter()
    safety = HostSafetyMonitor()
    corpus = json.loads(corpus_manifest_path.read_text())
    prior = json.loads(pw0139_path.read_text())
    failed_schedule = json.loads(pw0144_path.read_text())
    if (
        corpus.get("revision") != REVISION
        or prior.get("decision") != "reject_global_hessian_fixed_grid_on_full_validation"
        or failed_schedule.get("decision") != "reject_fixed_grid_code_changing_qat"
        or prior.get("holdout_unsealed")
        or failed_schedule.get("holdout_unsealed")
    ):
        raise ValueError("PW-0145 authority identity mismatch")
    checkpoint = ShardedCheckpoint(checkpoint_root, verification_path)
    root = corpus_manifest_path.parent
    authority = next(row for row in corpus["layers"] if row["layer"] == LAYER)
    validate_routes(authority)
    schedule = next(row for row in authority["expert_schedule"] if row["expert"] == EXPERT)
    train_local = [index for index, position in enumerate(schedule["positions"]) if position < 112]
    if len(train_local) != TRAIN_PLACEMENTS:
        raise ValueError("PW-0145 frozen train coverage mismatch")
    offset = sum(
        len(row["positions"])
        for row in authority["expert_schedule"][: authority["expert_schedule"].index(schedule)]
    )
    moe_input = load_capture(root, authority["captures"]["moe_input"])
    expert_down = load_capture(root, authority["captures"]["expert_down"])
    train_positions = [schedule["positions"][index] for index in train_local]
    train_inputs = np.asarray(moe_input[train_positions], dtype=np.float32).copy()
    train_expected = np.asarray(
        expert_down[[offset + index for index in train_local]], dtype=np.float32
    ).copy()
    prefix = f"model.layers.{LAYER}.mlp.experts.{EXPERT}"
    source_weights = {
        projection: dequant_weight(checkpoint, f"{prefix}.{projection}_proj.weight")
        for projection in ("gate", "up", "down")
    }
    hidden = source_hidden(source_weights, train_inputs)
    activations = {"gate": train_inputs, "up": train_inputs, "down": hidden}
    initial_grids = {}
    initial_weights = {}
    for projection in ("gate", "up", "down"):
        weight = source_weights[projection]
        scales, biases, _, _ = affine_grid(weight)
        projected = projected_workspace_bytes(weight)
        before = safety.checkpoint(f"{projection}_preflight")
        if before.process_physical_footprint_bytes + projected > safety.policy.maximum_process_physical_footprint_bytes:
            raise RuntimeError(f"PW-0145 {projection} exceeds Gate 8 headroom")
        candidate, codes, diagnostics = global_hessian_gptq_fixed_grid(
            weight, activations[projection], scales, biases
        )
        staged_scales = scales.astype(np.float16)
        staged_biases = biases.astype(np.float16)
        staged_candidate = reconstruct_fixed_grid(
            codes, staged_scales.astype(np.float32), staged_biases.astype(np.float32)
        )
        if not np.array_equal(candidate, staged_candidate):
            raise ValueError("PW-0145 F16 grid staging changes initialization")
        prior_projection = _prior_expert(prior, LAYER, EXPERT)["projection_reports"][projection]
        if diagnostics["grid_sha256"] != prior_projection["grid_sha256"]:
            raise ValueError("PW-0145 does not reproduce PW-0139 grid")
        initial_grids[projection] = {
            "codes": codes,
            "scales": staged_scales,
            "biases": staged_biases,
        }
        initial_weights[projection] = staged_candidate
        del candidate, codes
        gc.collect()
        safety.release_checkpoint(
            f"{projection}_workspace_released",
            ["full Hessian", "inverse Cholesky", "permuted working weight"],
        )
    initial_output = dense_expert(train_inputs, initial_weights)
    initial_metrics = error_metrics(initial_output, train_expected)
    prior_expert = _prior_expert(prior, LAYER, EXPERT)
    if initial_metrics != prior_expert["candidate_calibration_metrics"]:
        raise ValueError("PW-0145 does not reproduce PW-0139 train metric")
    trials = []
    for learning_rate in LEARNING_RATES:
        offsets, training = train_code_offsets(
            train_inputs,
            train_expected,
            initial_grids,
            safety,
            steps=STEPS,
            learning_rate=learning_rate,
            safety_interval=SAFETY_INTERVAL,
        )
        weights, final_grids, changed_codes = materialize_qat_weights(
            initial_grids, offsets
        )
        output = dense_expert(train_inputs, weights)
        trials.append(
            {
                "learning_rate": learning_rate,
                "training": training,
                "train_metrics": error_metrics(output, train_expected),
                "changed_codes": changed_codes,
                "code_totals": {
                    name: int(initial_grids[name]["codes"].size)
                    for name in ("gate", "up", "down")
                },
                "code_domain_valid": all(
                    grid["codes"].dtype == np.uint8 and np.all(grid["codes"] <= 15)
                    for grid in final_grids.values()
                ),
                "grid_metadata_unchanged": all(
                    np.array_equal(initial_grids[name]["scales"], final_grids[name]["scales"])
                    and np.array_equal(initial_grids[name]["biases"], final_grids[name]["biases"])
                    for name in ("gate", "up", "down")
                ),
                "maximum_absolute_offset": max(
                    float(np.max(np.abs(offsets[f"{name}_offset"])))
                    for name in ("gate", "up", "down")
                ),
            }
        )
        del offsets, weights, final_grids, output
        gc.collect()
        mx.clear_cache()
        safety.release_checkpoint(
            f"learning_rate_{learning_rate}_released",
            ["latent offsets", "optimizer state", "trial weights", "trial output"],
        )
    selected = select_trial(trials)
    gate = _gate(initial_metrics, selected)
    decision = (
        "authorize_frozen_code_qat_validation_confirmation"
        if gate["passes"]
        else "reject_tested_fixed_grid_code_qat_optimizer_family"
    )
    del moe_input, expert_down, train_inputs, train_expected, source_weights
    del hidden, activations, initial_grids, initial_weights, initial_output
    gc.collect()
    mx.clear_cache()
    safety.release_checkpoint(
        "checkpoint_and_train_authorities_released",
        ["checkpoint mappings", "train capture views", "PW-0139/PW-0144 authorities"],
    )
    safety.checkpoint("final_service_health")
    result = {
        "schema_version": 1,
        "evidence_class": "pw0145_train_only_code_qat_optimizer_preflight",
        "revision": REVISION,
        "commit": commit,
        "source_hashes": {
            "checkpoint_verification_sha256": VERIFICATION_SHA256,
            "corpus_manifest_sha256": CORPUS_SHA256,
            "pw0139_report_sha256": PW0139_SHA256,
            "pw0144_report_sha256": PW0144_SHA256,
        },
        "layer": LAYER,
        "expert": EXPERT,
        "train_placements": TRAIN_PLACEMENTS,
        "validation_loaded": False,
        "holdout_unsealed": False,
        "learning_rates": list(LEARNING_RATES),
        "steps": STEPS,
        "initial_train_metrics": initial_metrics,
        "trials": trials,
        "selected_learning_rate": selected["learning_rate"],
        "continuation_gate": gate,
        "decision": decision,
        "safety_snapshots": safety.evidence(),
        "complete_wall_ms": (time.perf_counter() - started) * 1000.0,
        "accepted_tokens": 0,
        "A": 0,
        "performance_claim": None,
        "limitations": (
            "one deep expert and four train-only learning rates; validation and holdout not "
            "loaded; optimizer preflight only; no fidelity, bank, kernel, model, endpoint, "
            "accepted-token, or TPS claim"
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
    parser.add_argument("--pw0139", required=True, type=Path)
    parser.add_argument("--pw0144", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--commit", required=True)
    args = parser.parse_args()
    try:
        result = run(
            args.checkpoint,
            args.verification,
            args.corpus_manifest,
            args.pw0139,
            args.pw0144,
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

