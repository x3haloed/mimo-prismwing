#!/usr/bin/env python3
"""Run PW-0144's train-only code-changing fixed-grid INT4 QAT control."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import platform
import time

import mlx.core as mx
import mlx.optimizers as optim
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
        GROUP_SIZE,
        REVISION,
        VERIFICATION_SHA256,
        error_metrics,
        validate_routes,
    )
    from tools.run_train_only_end_to_end_int4_grid_recovery import (
        _array_sha256,
        _grid_digest,
        partition_positions,
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
        GROUP_SIZE,
        REVISION,
        VERIFICATION_SHA256,
        error_metrics,
        validate_routes,
    )
    from run_train_only_end_to_end_int4_grid_recovery import (
        _array_sha256,
        _grid_digest,
        partition_positions,
    )


PW0139_SHA256 = "83bd204c9d5c35a684cab15a4ddacf48cf9b661563fb26223eb3655d0ef4a7b5"
PW0142_SHA256 = "0c2095a2068ccf347ab86beccb41e8d303444ce371b52d8475a37b26c29e9cc7"
SAMPLES = ((4, 96, 109, 56), (24, 200, 71, 37), (46, 249, 90, 56))
STEPS = 128
LEARNING_RATE = 0.05
REGULARIZATION = 1e-6
SAFETY_INTERVAL = 16


def quantized_code_ste(initial_codes: mx.array, offsets: mx.array) -> mx.array:
    continuous = initial_codes + offsets
    rounded = mx.clip(mx.round(continuous), 0.0, 15.0)
    return continuous + mx.stop_gradient(rounded - continuous)


def materialize_qat_weights(
    initial_grids: dict[str, dict[str, np.ndarray]],
    offsets: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, np.ndarray]], dict[str, int]]:
    weights = {}
    final_grids = {}
    changed = {}
    for projection in ("gate", "up", "down"):
        grid = initial_grids[projection]
        initial_codes = np.asarray(grid["codes"], dtype=np.uint8)
        latent = np.asarray(offsets[f"{projection}_offset"], dtype=np.float32)
        if latent.shape != initial_codes.shape or not np.isfinite(latent).all():
            raise ValueError("PW-0144 latent offset shape or value mismatch")
        final_codes = np.clip(
            np.rint(initial_codes.astype(np.float32) + latent), 0, 15
        ).astype(np.uint8)
        scales = np.asarray(grid["scales"], dtype=np.float16).copy()
        biases = np.asarray(grid["biases"], dtype=np.float16).copy()
        weights[projection] = reconstruct_fixed_grid(
            final_codes, scales.astype(np.float32), biases.astype(np.float32)
        )
        final_grids[projection] = {
            "codes": final_codes,
            "scales": scales,
            "biases": biases,
        }
        changed[projection] = int(np.count_nonzero(final_codes != initial_codes))
    return weights, final_grids, changed


def train_code_offsets(
    inputs: np.ndarray,
    targets: np.ndarray,
    initial_grids: dict[str, dict[str, np.ndarray]],
    safety: HostSafetyMonitor | None = None,
    *,
    steps: int = STEPS,
) -> tuple[dict[str, np.ndarray], dict]:
    if steps <= 0 or inputs.ndim != 2 or targets.ndim != 2 or inputs.shape[0] != targets.shape[0]:
        raise ValueError("PW-0144 training tensors are invalid")
    constants = {}
    offsets = {}
    for projection in ("gate", "up", "down"):
        grid = initial_grids[projection]
        codes = np.asarray(grid["codes"], dtype=np.uint8)
        scales = np.asarray(grid["scales"], dtype=np.float16).astype(np.float32)
        biases = np.asarray(grid["biases"], dtype=np.float16).astype(np.float32)
        if (
            codes.ndim != 2
            or scales.shape != biases.shape
            or scales.shape[0] != codes.shape[0]
            or codes.shape[1] != scales.shape[1] * GROUP_SIZE
            or np.any(codes > 15)
        ):
            raise ValueError("PW-0144 initial grid is invalid")
        constants[f"{projection}_codes"] = mx.array(codes.astype(np.float32))
        constants[f"{projection}_scale"] = mx.array(scales)
        constants[f"{projection}_bias"] = mx.array(biases)
        offsets[f"{projection}_offset"] = mx.zeros(codes.shape, dtype=mx.float32)
    train_inputs = mx.array(np.asarray(inputs, dtype=np.float16))
    train_targets = mx.array(np.asarray(targets, dtype=np.float32))
    target_energy = mx.maximum(mx.mean(mx.square(train_targets)), mx.array(1e-30))

    def dequantize(params: dict[str, mx.array], projection: str) -> mx.array:
        codes = quantized_code_ste(
            constants[f"{projection}_codes"], params[f"{projection}_offset"]
        )
        scale = constants[f"{projection}_scale"]
        bias = constants[f"{projection}_bias"]
        rows, columns = codes.shape
        groups = scale.shape[1]
        codes = mx.reshape(codes, (rows, groups, GROUP_SIZE))
        values = codes * mx.expand_dims(scale, 2) + mx.expand_dims(bias, 2)
        return mx.reshape(values, (rows, columns)).astype(mx.float16)

    def loss_function(params: dict[str, mx.array]) -> mx.array:
        gate_weight = dequantize(params, "gate")
        up_weight = dequantize(params, "up")
        down_weight = dequantize(params, "down")
        gate = train_inputs @ gate_weight.T
        up = train_inputs @ up_weight.T
        hidden = mx.sigmoid(gate) * gate * up
        output = (hidden @ down_weight.T).astype(mx.float32)
        normalized_mse = mx.mean(mx.square(output - train_targets)) / target_energy
        regularizer = sum(
            (mx.mean(mx.square(value)) for value in params.values()), mx.array(0.0)
        ) / len(params)
        return normalized_mse + REGULARIZATION * regularizer

    optimizer = optim.Adam(
        learning_rate=LEARNING_RATE,
        betas=[0.9, 0.999],
        eps=1e-8,
        bias_correction=True,
    )
    loss_and_grad = mx.value_and_grad(loss_function)
    history = []
    started = time.perf_counter()
    for step in range(steps + 1):
        loss, gradients = loss_and_grad(offsets)
        mx.eval(loss, gradients)
        value = float(loss.item())
        if not np.isfinite(value):
            raise RuntimeError("PW-0144 training loss is not finite")
        if step == 0 or step % SAFETY_INTERVAL == 0 or step == steps:
            history.append({"step": step, "loss": value})
            if safety is not None:
                safety.checkpoint(f"code_qat_step_{step}")
        if step == steps:
            break
        optimizer.update(offsets, gradients)
        mx.eval(offsets, optimizer.state)
    learned = {name: np.asarray(value).astype(np.float32, copy=True) for name, value in offsets.items()}
    diagnostics = {
        "steps": steps,
        "learning_rate": LEARNING_RATE,
        "betas": [0.9, 0.999],
        "epsilon": 1e-8,
        "bias_correction": True,
        "regularization": REGULARIZATION,
        "loss_history": history,
        "loss_decreased": history[-1]["loss"] < history[0]["loss"],
        "fit_wall_ms": (time.perf_counter() - started) * 1000.0,
    }
    del constants, train_inputs, train_targets, target_energy, optimizer
    del loss_and_grad, gradients, loss
    mx.clear_cache()
    return learned, diagnostics


def _gate(reports: list[dict]) -> dict:
    rows = []
    for report in reports:
        candidate = report["qat_validation_metrics"]
        prior = report["pw0139_validation_metrics"]
        total_codes = sum(report["code_totals"].values())
        changed_codes = sum(report["changed_codes"].values())
        conditions = {
            "train_improves": (
                report["qat_train_metrics"]["relative_l2"]
                < report["initial_train_metrics"]["relative_l2"]
            ),
            "finite_decreasing_loss": report["training"]["loss_decreased"],
            "some_but_not_all_codes_changed": 0 < changed_codes < total_codes,
            "code_domain": report["code_domain_valid"],
            "grid_metadata_unchanged": report["grid_metadata_unchanged"],
        }
        if report["layer"] == 4:
            conditions["early_control_no_more_than_ten_percent_regression"] = (
                candidate["relative_l2"] <= prior["relative_l2"] * 1.10
            )
        else:
            conditions.update(
                {
                    "minimum_twenty_five_percent_improvement": (
                        candidate["relative_l2"] <= prior["relative_l2"] * 0.75
                    ),
                    "validation_relative_l2": candidate["relative_l2"] <= 0.05,
                    "maximum_validation_row_relative_l2": (
                        candidate["maximum_row_relative_l2"] <= 0.08
                    ),
                }
            )
        rows.append(
            {
                "layer": report["layer"],
                "expert": report["expert"],
                "passes": all(conditions.values()),
                "conditions": conditions,
                "changed_codes": changed_codes,
                "total_codes": total_codes,
                "prior_validation_relative_l2": prior["relative_l2"],
                "qat_validation_relative_l2": candidate["relative_l2"],
                "qat_maximum_row_relative_l2": candidate["maximum_row_relative_l2"],
            }
        )
    physical = physical_ledger()
    physical_passes = (
        physical["packed_bytes_per_expert"] == 13_369_344
        and physical["additional_runtime_macs"] == 0
    )
    return {
        "passes": all(row["passes"] for row in rows) and physical_passes,
        "experts": rows,
        "physical": physical,
        "training_only_latent_bytes": 0,
        "additional_runtime_macs": 0,
        "physical_passes": physical_passes,
    }


def run(
    checkpoint_root: Path,
    verification_path: Path,
    corpus_manifest_path: Path,
    pw0139_path: Path,
    pw0142_path: Path,
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
        (pw0142_path, PW0142_SHA256, "PW-0142 report"),
    ):
        if sha256_file(path) != expected:
            raise ValueError(f"PW-0144 {label} hash mismatch")
    started = time.perf_counter()
    safety = HostSafetyMonitor()
    corpus = json.loads(corpus_manifest_path.read_text())
    prior = json.loads(pw0139_path.read_text())
    metadata_recovery = json.loads(pw0142_path.read_text())
    if (
        corpus.get("revision") != REVISION
        or prior.get("decision") != "reject_global_hessian_fixed_grid_on_full_validation"
        or metadata_recovery.get("decision") != "reject_fixed_code_group_parameter_recovery"
        or prior.get("holdout_unsealed")
        or metadata_recovery.get("holdout_unsealed")
    ):
        raise ValueError("PW-0144 authority identity mismatch")
    checkpoint = ShardedCheckpoint(checkpoint_root, verification_path)
    root = corpus_manifest_path.parent
    reports = []
    for layer, expert, expected_train, expected_validation in SAMPLES:
        authority = next(row for row in corpus["layers"] if row["layer"] == layer)
        validate_routes(authority)
        schedule = next(row for row in authority["expert_schedule"] if row["expert"] == expert)
        train_local, validation_local = partition_positions(schedule["positions"])
        if len(train_local) != expected_train or len(validation_local) != expected_validation:
            raise ValueError("PW-0144 frozen sample coverage mismatch")
        offset = sum(
            len(row["positions"])
            for row in authority["expert_schedule"][: authority["expert_schedule"].index(schedule)]
        )
        moe_input = load_capture(root, authority["captures"]["moe_input"])
        expert_down = load_capture(root, authority["captures"]["expert_down"])
        train_positions = [schedule["positions"][index] for index in train_local]
        validation_positions = [schedule["positions"][index] for index in validation_local]
        train_inputs = np.asarray(moe_input[train_positions], dtype=np.float32).copy()
        validation_inputs = np.asarray(moe_input[validation_positions], dtype=np.float32).copy()
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
        hidden = source_hidden(source_weights, train_inputs)
        projection_activations = {"gate": train_inputs, "up": train_inputs, "down": hidden}
        initial_grids = {}
        initial_weights = {}
        projection_reports = {}
        for projection in ("gate", "up", "down"):
            weight = source_weights[projection]
            scales, biases, _, _ = affine_grid(weight)
            projected = projected_workspace_bytes(weight)
            before = safety.checkpoint(f"layer_{layer}_expert_{expert}_{projection}_preflight")
            if before.process_physical_footprint_bytes + projected > safety.policy.maximum_process_physical_footprint_bytes:
                raise RuntimeError(f"PW-0144 layer {layer} expert {expert} exceeds Gate 8 headroom")
            candidate, codes, diagnostics = global_hessian_gptq_fixed_grid(
                weight, projection_activations[projection], scales, biases
            )
            staged_scales = scales.astype(np.float16)
            staged_biases = biases.astype(np.float16)
            staged_candidate = reconstruct_fixed_grid(
                codes, staged_scales.astype(np.float32), staged_biases.astype(np.float32)
            )
            if not np.array_equal(candidate, staged_candidate):
                raise ValueError("PW-0144 F16 grid staging changes PW-0139 initialization")
            initial_grids[projection] = {
                "codes": codes,
                "scales": staged_scales,
                "biases": staged_biases,
            }
            initial_weights[projection] = staged_candidate
            prior_projection = _prior_expert(prior, layer, expert)["projection_reports"][projection]
            if diagnostics["grid_sha256"] != prior_projection["grid_sha256"]:
                raise ValueError("PW-0144 does not reproduce PW-0139 codes")
            projection_reports[projection] = {
                "initial_grid_sha256": diagnostics["grid_sha256"],
                "initial_codes_sha256": _array_sha256(codes),
                "scale_sha256": _array_sha256(staged_scales),
                "bias_sha256": _array_sha256(staged_biases),
            }
            del candidate, codes
            gc.collect()
            safety.release_checkpoint(
                f"layer_{layer}_expert_{expert}_{projection}_workspace_released",
                ["full Hessian", "inverse Cholesky", "permuted working weight"],
            )
        initial_train = dense_expert(train_inputs, initial_weights)
        initial_validation = dense_expert(validation_inputs, initial_weights)
        prior_expert = _prior_expert(prior, layer, expert)
        initial_train_metrics = error_metrics(initial_train, train_expected)
        initial_validation_metrics = error_metrics(initial_validation, validation_expected)
        if (
            initial_train_metrics != prior_expert["candidate_calibration_metrics"]
            or initial_validation_metrics != prior_expert["candidate_validation_metrics"]
        ):
            raise ValueError("PW-0144 does not reproduce PW-0139 expert metrics")
        initial_grid_digest = _grid_digest(initial_grids)
        offsets, training = train_code_offsets(
            train_inputs, train_expected, initial_grids, safety
        )
        qat_weights, final_grids, changed_codes = materialize_qat_weights(
            initial_grids, offsets
        )
        qat_train = dense_expert(train_inputs, qat_weights)
        qat_validation = dense_expert(validation_inputs, qat_weights)
        grid_metadata_unchanged = all(
            np.array_equal(initial_grids[name]["scales"], final_grids[name]["scales"])
            and np.array_equal(initial_grids[name]["biases"], final_grids[name]["biases"])
            for name in ("gate", "up", "down")
        )
        code_domain_valid = all(
            grid["codes"].dtype == np.uint8 and np.all(grid["codes"] <= 15)
            for grid in final_grids.values()
        )
        reports.append(
            {
                "layer": layer,
                "expert": expert,
                "train_placements": len(train_positions),
                "validation_placements": len(validation_positions),
                "projection_reports": projection_reports,
                "initial_grid_digest": initial_grid_digest,
                "final_grid_digest": _grid_digest(final_grids),
                "changed_codes": changed_codes,
                "code_totals": {
                    name: int(initial_grids[name]["codes"].size)
                    for name in ("gate", "up", "down")
                },
                "code_domain_valid": code_domain_valid,
                "grid_metadata_unchanged": grid_metadata_unchanged,
                "maximum_absolute_offset": max(
                    float(np.max(np.abs(offsets[f"{name}_offset"])))
                    for name in ("gate", "up", "down")
                ),
                "training": training,
                "initial_train_metrics": initial_train_metrics,
                "initial_validation_metrics": initial_validation_metrics,
                "qat_train_metrics": error_metrics(qat_train, train_expected),
                "qat_validation_metrics": error_metrics(qat_validation, validation_expected),
                "pw0139_validation_metrics": prior_expert["candidate_validation_metrics"],
            }
        )
        del moe_input, expert_down, train_inputs, validation_inputs, train_expected
        del validation_expected, source_weights, hidden, projection_activations
        del initial_grids, initial_weights, initial_train, initial_validation
        del offsets, qat_weights, final_grids, qat_train, qat_validation
        gc.collect()
        mx.clear_cache()
        safety.release_checkpoint(
            f"layer_{layer}_expert_{expert}_complete_released",
            ["source expert", "QAT grids", "latent offsets", "optimizer state", "captured activations"],
        )
    gate = _gate(reports)
    decision = (
        "authorize_all_validation_expert_code_qat_audit"
        if gate["passes"]
        else "reject_fixed_grid_code_changing_qat"
    )
    safety.release_checkpoint(
        "checkpoint_and_authorities_released",
        ["checkpoint mappings", "PW-0116 corpus", "PW-0139/PW-0142 authorities"],
    )
    safety.checkpoint("final_service_health")
    result = {
        "schema_version": 1,
        "evidence_class": "pw0144_train_only_code_changing_int4_qat",
        "revision": REVISION,
        "commit": commit,
        "source_hashes": {
            "checkpoint_verification_sha256": VERIFICATION_SHA256,
            "corpus_manifest_sha256": CORPUS_SHA256,
            "pw0139_report_sha256": PW0139_SHA256,
            "pw0142_report_sha256": PW0142_SHA256,
        },
        "samples": [list(row) for row in SAMPLES],
        "training_contract": {
            "steps": STEPS,
            "learning_rate": LEARNING_RATE,
            "betas": [0.9, 0.999],
            "epsilon": 1e-8,
            "bias_correction": True,
            "regularization": REGULARIZATION,
            "full_batch": True,
            "straight_through": True,
            "fixed_f16_grid_metadata": True,
        },
        "reports": reports,
        "continuation_gate": gate,
        "holdout_unsealed": False,
        "decision": decision,
        "safety_snapshots": safety.evidence(),
        "complete_wall_ms": (time.perf_counter() - started) * 1000.0,
        "accepted_tokens": 0,
        "A": 0,
        "performance_claim": None,
        "limitations": (
            "three validation-visible experts and one frozen optimizer only; fixed F16 grids; "
            "dense unpacked execution; holdout sealed; no bank, kernel, accumulated model, "
            "modalities, endpoint, accepted tokens, or TPS"
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
    parser.add_argument("--pw0142", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--commit", required=True)
    args = parser.parse_args()
    try:
        result = run(
            args.checkpoint,
            args.verification,
            args.corpus_manifest,
            args.pw0139,
            args.pw0142,
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

