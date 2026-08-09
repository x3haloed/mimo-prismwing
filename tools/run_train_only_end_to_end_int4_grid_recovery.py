#!/usr/bin/env python3
"""Run PW-0142's train-only end-to-end fixed-code INT4 recovery control."""

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


PW0139_SHA256 = "83bd204c9d5c35a684cab15a4ddacf48cf9b661563fb26223eb3655d0ef4a7b5"
PW0141_SHA256 = "4dae2abe2a59457a77e09bd4d1328b7b6dce8f0e41e3ac115fd27645c93e56a9"
SAMPLES = ((4, 96, 109, 56), (24, 200, 71, 37), (46, 249, 90, 56))
STEPS = 128
LEARNING_RATE = 0.01
REGULARIZATION = 1e-4
SAFETY_INTERVAL = 16


def partition_positions(positions: list[int]) -> tuple[list[int], list[int]]:
    if any(not isinstance(position, int) or not 0 <= position < 224 for position in positions):
        raise ValueError("PW-0142 partition positions are invalid")
    train = [index for index, position in enumerate(positions) if position < 112]
    validation = [
        index for index, position in enumerate(positions) if 112 <= position < 168
    ]
    if set(train).intersection(validation):
        raise ValueError("PW-0142 train and validation overlap")
    return train, validation


def _array_sha256(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def _grid_digest(grids: dict[str, dict[str, np.ndarray]]) -> str:
    digest = hashlib.sha256()
    for projection in ("gate", "up", "down"):
        for field in ("codes", "scales", "biases"):
            values = np.ascontiguousarray(grids[projection][field])
            digest.update(projection.encode())
            digest.update(field.encode())
            digest.update(values.dtype.str.encode())
            digest.update(np.asarray(values.shape, dtype="<u8").tobytes())
            digest.update(values.tobytes())
    return digest.hexdigest()


def materialize_recovered_weights(
    initial_grids: dict[str, dict[str, np.ndarray]],
    parameters: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, np.ndarray]]]:
    weights = {}
    final_grids = {}
    for projection in ("gate", "up", "down"):
        grid = initial_grids[projection]
        initial_scale = np.asarray(grid["scales"], dtype=np.float32)
        initial_bias = np.asarray(grid["biases"], dtype=np.float32)
        log_scale = np.asarray(parameters[f"{projection}_log_scale"], dtype=np.float32)
        bias_delta = np.asarray(parameters[f"{projection}_bias_delta"], dtype=np.float32)
        if log_scale.shape != initial_scale.shape or bias_delta.shape != initial_bias.shape:
            raise ValueError("PW-0142 recovered parameter shape mismatch")
        bias_unit = np.maximum(np.abs(initial_scale), np.float32(1e-8))
        scales = (initial_scale * np.exp(log_scale)).astype(np.float16)
        biases = (initial_bias + bias_unit * bias_delta).astype(np.float16)
        codes = np.asarray(grid["codes"], dtype=np.uint8).copy()
        weights[projection] = reconstruct_fixed_grid(
            codes, scales.astype(np.float32), biases.astype(np.float32)
        )
        final_grids[projection] = {"codes": codes, "scales": scales, "biases": biases}
    return weights, final_grids


def train_fixed_code_grids(
    inputs: np.ndarray,
    targets: np.ndarray,
    initial_grids: dict[str, dict[str, np.ndarray]],
    safety: HostSafetyMonitor | None = None,
    *,
    steps: int = STEPS,
) -> tuple[dict[str, np.ndarray], dict]:
    if steps <= 0 or inputs.ndim != 2 or targets.ndim != 2 or inputs.shape[0] != targets.shape[0]:
        raise ValueError("PW-0142 training tensors are invalid")
    constants = {}
    parameters = {}
    for projection in ("gate", "up", "down"):
        grid = initial_grids[projection]
        scales = np.asarray(grid["scales"], dtype=np.float32)
        biases = np.asarray(grid["biases"], dtype=np.float32)
        codes = np.asarray(grid["codes"], dtype=np.uint8)
        if (
            codes.ndim != 2
            or scales.shape != biases.shape
            or scales.shape[0] != codes.shape[0]
            or codes.shape[1] != scales.shape[1] * GROUP_SIZE
            or np.any(codes > 15)
        ):
            raise ValueError("PW-0142 initial grid is invalid")
        constants[f"{projection}_codes"] = mx.array(codes.astype(np.float32))
        constants[f"{projection}_scale"] = mx.array(scales)
        constants[f"{projection}_bias"] = mx.array(biases)
        constants[f"{projection}_bias_unit"] = mx.array(
            np.maximum(np.abs(scales), np.float32(1e-8))
        )
        parameters[f"{projection}_log_scale"] = mx.zeros(scales.shape, dtype=mx.float32)
        parameters[f"{projection}_bias_delta"] = mx.zeros(biases.shape, dtype=mx.float32)
    train_inputs = mx.array(np.asarray(inputs, dtype=np.float16))
    train_targets = mx.array(np.asarray(targets, dtype=np.float32))
    target_energy = mx.maximum(mx.mean(mx.square(train_targets)), mx.array(1e-30))

    def dequantize(params: dict[str, mx.array], projection: str) -> mx.array:
        codes = constants[f"{projection}_codes"]
        initial_scale = constants[f"{projection}_scale"]
        scale = initial_scale * mx.exp(params[f"{projection}_log_scale"])
        bias = constants[f"{projection}_bias"] + (
            constants[f"{projection}_bias_unit"] * params[f"{projection}_bias_delta"]
        )
        rows, columns = codes.shape
        groups = scale.shape[1]
        values = mx.reshape(codes, (rows, groups, GROUP_SIZE))
        values = values * mx.expand_dims(scale, 2) + mx.expand_dims(bias, 2)
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
            (mx.mean(mx.square(value)) for value in params.values()),
            mx.array(0.0),
        ) / len(params)
        return normalized_mse + REGULARIZATION * regularizer

    optimizer = optim.Adam(
        learning_rate=LEARNING_RATE,
        betas=[0.9, 0.999],
        eps=1e-8,
        bias_correction=False,
    )
    loss_and_grad = mx.value_and_grad(loss_function)
    history = []
    started = time.perf_counter()
    for step in range(steps + 1):
        loss, gradients = loss_and_grad(parameters)
        mx.eval(loss, gradients)
        value = float(loss.item())
        if not np.isfinite(value):
            raise RuntimeError("PW-0142 training loss is not finite")
        if step == 0 or step % SAFETY_INTERVAL == 0 or step == steps:
            history.append({"step": step, "loss": value})
            if safety is not None:
                safety.checkpoint(f"recovery_training_step_{step}")
        if step == steps:
            break
        optimizer.update(parameters, gradients)
        mx.eval(parameters, optimizer.state)
    learned = {name: np.asarray(value).astype(np.float32, copy=True) for name, value in parameters.items()}
    diagnostics = {
        "steps": steps,
        "learning_rate": LEARNING_RATE,
        "betas": [0.9, 0.999],
        "epsilon": 1e-8,
        "bias_correction": False,
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
        candidate = report["recovered_validation_metrics"]
        prior = report["pw0139_validation_metrics"]
        conditions = {
            "train_improves": (
                report["recovered_train_metrics"]["relative_l2"]
                < report["initial_train_metrics"]["relative_l2"]
            ),
            "finite_decreasing_loss": report["training"]["loss_decreased"],
            "fixed_codes": report["codes_unchanged"],
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
                "prior_validation_relative_l2": prior["relative_l2"],
                "recovered_validation_relative_l2": candidate["relative_l2"],
                "recovered_maximum_row_relative_l2": candidate["maximum_row_relative_l2"],
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
        "training_only_parameter_bytes": 0,
        "additional_runtime_macs": 0,
        "physical_passes": physical_passes,
    }


def run(
    checkpoint_root: Path,
    verification_path: Path,
    corpus_manifest_path: Path,
    pw0139_path: Path,
    pw0141_path: Path,
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
        (pw0141_path, PW0141_SHA256, "PW-0141 report"),
    ):
        if sha256_file(path) != expected:
            raise ValueError(f"PW-0142 {label} hash mismatch")
    started = time.perf_counter()
    safety = HostSafetyMonitor()
    corpus = json.loads(corpus_manifest_path.read_text())
    prior = json.loads(pw0139_path.read_text())
    rotation = json.loads(pw0141_path.read_text())
    if (
        corpus.get("revision") != REVISION
        or prior.get("decision") != "reject_global_hessian_fixed_grid_on_full_validation"
        or rotation.get("decision") != "reject_fixed_residual_hadamard_rotation"
        or prior.get("holdout_unsealed")
        or rotation.get("holdout_unsealed")
    ):
        raise ValueError("PW-0142 authority identity mismatch")
    checkpoint = ShardedCheckpoint(checkpoint_root, verification_path)
    root = corpus_manifest_path.parent
    reports = []
    for layer, expert, expected_train, expected_validation in SAMPLES:
        authority = next(row for row in corpus["layers"] if row["layer"] == layer)
        validate_routes(authority)
        schedule = next(row for row in authority["expert_schedule"] if row["expert"] == expert)
        train_local, validation_local = partition_positions(schedule["positions"])
        if len(train_local) != expected_train or len(validation_local) != expected_validation:
            raise ValueError("PW-0142 frozen sample coverage mismatch")
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
                raise RuntimeError(f"PW-0142 layer {layer} expert {expert} exceeds Gate 8 headroom")
            candidate, codes, diagnostics = global_hessian_gptq_fixed_grid(
                weight, projection_activations[projection], scales, biases
            )
            initial_grids[projection] = {
                "codes": codes,
                "scales": scales,
                "biases": biases,
            }
            initial_weights[projection] = candidate
            prior_projection = _prior_expert(prior, layer, expert)["projection_reports"][projection]
            if diagnostics["grid_sha256"] != prior_projection["grid_sha256"]:
                raise ValueError("PW-0142 does not reproduce PW-0139 codes")
            projection_reports[projection] = {
                "initial_grid_sha256": diagnostics["grid_sha256"],
                "initial_codes_sha256": _array_sha256(codes),
                "activation_order_sha256": diagnostics["activation_order_sha256"],
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
            raise ValueError("PW-0142 does not reproduce PW-0139 expert metrics")
        initial_codes_digest = _grid_digest(initial_grids)
        parameters, training = train_fixed_code_grids(
            train_inputs, train_expected, initial_grids, safety
        )
        recovered_weights, final_grids = materialize_recovered_weights(initial_grids, parameters)
        recovered_train = dense_expert(train_inputs, recovered_weights)
        recovered_validation = dense_expert(validation_inputs, recovered_weights)
        final_codes_digest = _grid_digest(
            {
                name: {
                    "codes": final_grids[name]["codes"],
                    "scales": initial_grids[name]["scales"],
                    "biases": initial_grids[name]["biases"],
                }
                for name in ("gate", "up", "down")
            }
        )
        codes_unchanged = all(
            np.array_equal(initial_grids[name]["codes"], final_grids[name]["codes"])
            for name in ("gate", "up", "down")
        )
        reports.append(
            {
                "layer": layer,
                "expert": expert,
                "train_placements": len(train_positions),
                "validation_placements": len(validation_positions),
                "projection_reports": projection_reports,
                "initial_grid_digest": initial_codes_digest,
                "final_fixed_code_authority_digest": final_codes_digest,
                "codes_unchanged": codes_unchanged,
                "parameter_sha256": _grid_digest(final_grids),
                "maximum_absolute_log_scale": max(
                    float(np.max(np.abs(parameters[f"{name}_log_scale"])))
                    for name in ("gate", "up", "down")
                ),
                "maximum_absolute_bias_delta": max(
                    float(np.max(np.abs(parameters[f"{name}_bias_delta"])))
                    for name in ("gate", "up", "down")
                ),
                "training": training,
                "initial_train_metrics": initial_train_metrics,
                "initial_validation_metrics": initial_validation_metrics,
                "recovered_train_metrics": error_metrics(recovered_train, train_expected),
                "recovered_validation_metrics": error_metrics(
                    recovered_validation, validation_expected
                ),
                "pw0139_validation_metrics": prior_expert["candidate_validation_metrics"],
            }
        )
        del moe_input, expert_down, train_inputs, validation_inputs, train_expected
        del validation_expected, source_weights, hidden, projection_activations
        del initial_grids, initial_weights, initial_train, initial_validation
        del parameters, recovered_weights, final_grids, recovered_train, recovered_validation
        gc.collect()
        mx.clear_cache()
        safety.release_checkpoint(
            f"layer_{layer}_expert_{expert}_complete_released",
            ["source expert", "fixed-code grids", "optimizer state", "captured activations"],
        )
    gate = _gate(reports)
    decision = (
        "authorize_all_validation_expert_grid_recovery_audit"
        if gate["passes"]
        else "reject_fixed_code_group_parameter_recovery"
    )
    safety.release_checkpoint(
        "checkpoint_and_authorities_released",
        ["checkpoint mappings", "PW-0116 corpus", "PW-0139/PW-0141 authorities"],
    )
    safety.checkpoint("final_service_health")
    result = {
        "schema_version": 1,
        "evidence_class": "pw0142_train_only_end_to_end_int4_grid_recovery",
        "revision": REVISION,
        "commit": commit,
        "source_hashes": {
            "checkpoint_verification_sha256": VERIFICATION_SHA256,
            "corpus_manifest_sha256": CORPUS_SHA256,
            "pw0139_report_sha256": PW0139_SHA256,
            "pw0141_report_sha256": PW0141_SHA256,
        },
        "samples": [list(row) for row in SAMPLES],
        "training_contract": {
            "steps": STEPS,
            "learning_rate": LEARNING_RATE,
            "betas": [0.9, 0.999],
            "epsilon": 1e-8,
            "bias_correction": False,
            "regularization": REGULARIZATION,
            "full_batch": True,
            "fixed_codes": True,
            "final_grid_dtype": "float16",
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
            "three validation-visible experts and one frozen optimizer only; fixed codes; "
            "dense unpacked execution; holdout sealed; no bank, packed kernel, accumulated "
            "model, modalities, endpoint, accepted tokens, or TPS"
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
    parser.add_argument("--pw0141", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--commit", required=True)
    args = parser.parse_args()
    try:
        result = run(
            args.checkpoint,
            args.verification,
            args.corpus_manifest,
            args.pw0139,
            args.pw0141,
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
