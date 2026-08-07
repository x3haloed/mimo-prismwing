#!/usr/bin/env python3
"""Run PW-0141's fixed randomized-Hadamard residual rotation control."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
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
    from tools.run_global_hessian_gptq_rescue import global_hessian_gptq_fixed_grid, projected_workspace_bytes
    from tools.run_group_local_gptq_three_expert_control import (
        affine_grid,
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
    from run_global_hessian_gptq_rescue import global_hessian_gptq_fixed_grid, projected_workspace_bytes
    from run_group_local_gptq_three_expert_control import (
        affine_grid,
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


PW0139_SHA256 = "83bd204c9d5c35a684cab15a4ddacf48cf9b661563fb26223eb3655d0ef4a7b5"
PW0140_SHA256 = "824d66549da7833d855f430a60b761f145a98757fa191bb41db6cf6e56f78b9f"
SAMPLES = ((4, 96, 109, 56), (24, 200, 71, 37), (46, 249, 90, 56))
ROTATION_LABEL = f"PW-0141|{REVISION}"


def rotation_signs(size: int) -> np.ndarray:
    if size <= 0 or size & (size - 1):
        raise ValueError("PW-0141 rotation size must be a positive power of two")
    seed = int.from_bytes(hashlib.sha256(ROTATION_LABEL.encode()).digest()[:16], "little")
    generator = np.random.Generator(np.random.PCG64(seed))
    return (generator.integers(0, 2, size=size, dtype=np.int8) * 2 - 1).astype(np.float64)


def normalized_fwht(values: np.ndarray) -> np.ndarray:
    if values.ndim < 1 or values.shape[-1] <= 0 or values.shape[-1] & (values.shape[-1] - 1):
        raise ValueError("PW-0141 FWHT requires a power-of-two final dimension")
    result = np.asarray(values, dtype=np.float64).copy()
    width = result.shape[-1]
    step = 1
    while step < width:
        blocks = result.reshape(*result.shape[:-1], width // (2 * step), 2 * step)
        left = blocks[..., :step].copy()
        right = blocks[..., step : 2 * step].copy()
        blocks[..., :step] = left + right
        blocks[..., step : 2 * step] = left - right
        step *= 2
    result /= math.sqrt(width)
    return result


def right_rotate(values: np.ndarray, signs: np.ndarray) -> np.ndarray:
    if values.shape[-1] != signs.shape[0]:
        raise ValueError("PW-0141 right rotation shape mismatch")
    return normalized_fwht(np.asarray(values, dtype=np.float64) * signs)


def right_unrotate(values: np.ndarray, signs: np.ndarray) -> np.ndarray:
    if values.shape[-1] != signs.shape[0]:
        raise ValueError("PW-0141 right inverse rotation shape mismatch")
    return normalized_fwht(values) * signs


def left_rotate_down(weight: np.ndarray, signs: np.ndarray) -> np.ndarray:
    if weight.ndim != 2 or weight.shape[0] != signs.shape[0]:
        raise ValueError("PW-0141 down rotation shape mismatch")
    return right_rotate(weight.T, signs).T


def relative_l2(actual: np.ndarray, expected: np.ndarray) -> float:
    difference = np.asarray(actual, dtype=np.float64) - np.asarray(expected, dtype=np.float64)
    return float(np.linalg.norm(difference) / max(float(np.linalg.norm(expected)), 1e-30))


def unquantized_rotation_parity(
    inputs: np.ndarray, weights: dict[str, np.ndarray], signs: np.ndarray
) -> dict:
    x = np.asarray(inputs, dtype=np.float64)
    source_gate = x @ np.asarray(weights["gate"], dtype=np.float64).T
    source_up = x @ np.asarray(weights["up"], dtype=np.float64).T
    source_hidden = source_gate / (1.0 + np.exp(-source_gate)) * source_up
    source = source_hidden @ np.asarray(weights["down"], dtype=np.float64).T
    rotated_input = right_rotate(x, signs)
    rotated_gate_weight = right_rotate(weights["gate"], signs)
    rotated_up_weight = right_rotate(weights["up"], signs)
    rotated_down_weight = left_rotate_down(weights["down"], signs)
    gate = rotated_input @ rotated_gate_weight.T
    up = rotated_input @ rotated_up_weight.T
    hidden = gate / (1.0 + np.exp(-gate)) * up
    rotated_output = hidden @ rotated_down_weight.T
    restored = right_unrotate(rotated_output, signs)
    roundtrip = right_unrotate(right_rotate(x, signs), signs)
    return {
        "forward_relative_l2": relative_l2(restored, source),
        "roundtrip_relative_l2": relative_l2(roundtrip, x),
    }


def rotated_expert(
    rotated_inputs: np.ndarray,
    weights: dict[str, np.ndarray],
    signs: np.ndarray,
) -> np.ndarray:
    values = mx.array(np.asarray(rotated_inputs, dtype=np.float16))
    arrays = {name: mx.array(np.asarray(weight, dtype=np.float16)) for name, weight in weights.items()}
    gate = values @ arrays["gate"].T
    up = values @ arrays["up"].T
    hidden = mx.sigmoid(gate) * gate * up
    output = hidden @ arrays["down"].T
    mx.eval(output)
    rotated = np.asarray(output).astype(np.float32, copy=True)
    result = right_unrotate(rotated, signs).astype(np.float32)
    del values, arrays, gate, up, hidden, output
    mx.clear_cache()
    return result


def _prior_expert(prior: dict, layer: int, expert: int) -> dict:
    matches = [
        row
        for layer_report in prior.get("layer_reports", [])
        if layer_report.get("layer") == layer
        for row in layer_report.get("expert_reports", [])
        if row.get("expert") == expert
    ]
    if len(matches) != 1:
        raise ValueError("PW-0141 prior expert authority mismatch")
    return matches[0]


def _gate(reports: list[dict]) -> dict:
    rows = []
    for report in reports:
        candidate = report["rotated_validation_metrics"]
        prior = report["pw0139_validation_metrics"]
        if report["layer"] == 4:
            conditions = {
                "early_control_no_more_than_ten_percent_regression": (
                    candidate["relative_l2"] <= prior["relative_l2"] * 1.10
                )
            }
        else:
            conditions = {
                "minimum_twenty_five_percent_improvement": (
                    candidate["relative_l2"] <= prior["relative_l2"] * 0.75
                ),
                "validation_relative_l2": candidate["relative_l2"] <= 0.05,
                "maximum_validation_row_relative_l2": candidate["maximum_row_relative_l2"] <= 0.08,
            }
        conditions["all_projection_train_improve"] = all(
            row["candidate_train_metrics"]["relative_l2"]
            < row["baseline_train_metrics"]["relative_l2"]
            for row in report["projection_reports"].values()
        )
        conditions["unquantized_algebra"] = (
            report["unquantized_rotation_parity"]["forward_relative_l2"] <= 1e-10
            and report["unquantized_rotation_parity"]["roundtrip_relative_l2"] <= 1e-10
        )
        rows.append(
            {
                "layer": report["layer"],
                "expert": report["expert"],
                "passes": all(conditions.values()),
                "conditions": conditions,
                "prior_validation_relative_l2": prior["relative_l2"],
                "rotated_validation_relative_l2": candidate["relative_l2"],
                "rotated_maximum_row_relative_l2": candidate["maximum_row_relative_l2"],
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
        "prospective_online_residual_transform_operations": 0,
        "physical_passes": physical_passes,
    }


def run(
    checkpoint_root: Path,
    verification_path: Path,
    corpus_manifest_path: Path,
    pw0139_path: Path,
    pw0140_path: Path,
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
        (pw0140_path, PW0140_SHA256, "PW-0140 report"),
    ):
        if sha256_file(path) != expected:
            raise ValueError(f"PW-0141 {label} hash mismatch")
    started = time.perf_counter()
    safety = HostSafetyMonitor()
    corpus = json.loads(corpus_manifest_path.read_text())
    prior = json.loads(pw0139_path.read_text())
    pooled = json.loads(pw0140_path.read_text())
    if (
        corpus.get("revision") != REVISION
        or prior.get("decision") != "reject_global_hessian_fixed_grid_on_full_validation"
        or pooled.get("decision") != "reject_pooled_only_low_count_gptq"
        or prior.get("holdout_unsealed")
        or pooled.get("holdout_unsealed")
    ):
        raise ValueError("PW-0141 authority identity mismatch")
    checkpoint = ShardedCheckpoint(checkpoint_root, verification_path)
    root = corpus_manifest_path.parent
    signs = rotation_signs(4096)
    signs_sha256 = hashlib.sha256(np.ascontiguousarray(signs.astype(np.int8)).tobytes()).hexdigest()
    reports = []
    for layer, expert, expected_train, expected_validation in SAMPLES:
        authority = next(row for row in corpus["layers"] if row["layer"] == layer)
        validate_routes(authority)
        schedule = next(row for row in authority["expert_schedule"] if row["expert"] == expert)
        train_local = [index for index, position in enumerate(schedule["positions"]) if position < 112]
        validation_local = [
            index for index, position in enumerate(schedule["positions"]) if 112 <= position < 168
        ]
        if len(train_local) != expected_train or len(validation_local) != expected_validation:
            raise ValueError("PW-0141 frozen sample coverage mismatch")
        offset = sum(
            len(row["positions"])
            for row in authority["expert_schedule"][: authority["expert_schedule"].index(schedule)]
        )
        moe_input = load_capture(root, authority["captures"]["moe_input"])
        expert_down = load_capture(root, authority["captures"]["expert_down"])
        train_positions = [schedule["positions"][index] for index in train_local]
        validation_positions = [schedule["positions"][index] for index in validation_local]
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
        parity = unquantized_rotation_parity(moe_input[train_positions[: min(4, len(train_positions))]], source_weights, signs)
        if parity["forward_relative_l2"] > 1e-10 or parity["roundtrip_relative_l2"] > 1e-10:
            raise ValueError("PW-0141 unquantized rotation parity failed")
        rotated_weights = {
            "gate": right_rotate(source_weights["gate"], signs).astype(np.float32),
            "up": right_rotate(source_weights["up"], signs).astype(np.float32),
            "down": left_rotate_down(source_weights["down"], signs).astype(np.float32),
        }
        rotated_train = right_rotate(moe_input[train_positions], signs).astype(np.float32)
        rotated_validation = right_rotate(moe_input[validation_positions], signs).astype(np.float32)
        hidden = source_hidden(source_weights, moe_input[train_positions])
        activations = {"gate": rotated_train, "up": rotated_train, "down": hidden}
        expected = {
            "gate": source_linear(
                source_weights["gate"],
                torch.from_numpy(np.asarray(moe_input[train_positions]).copy()).to(torch.bfloat16),
            ).float().numpy(),
            "up": source_linear(
                source_weights["up"],
                torch.from_numpy(np.asarray(moe_input[train_positions]).copy()).to(torch.bfloat16),
            ).float().numpy(),
            "down": right_rotate(train_expected, signs).astype(np.float32),
        }
        selected = {}
        baseline = {}
        projection_reports = {}
        for projection in ("gate", "up", "down"):
            weight = rotated_weights[projection]
            scales, biases, control, control_codes = affine_grid(weight)
            validate_grid_membership(control_codes, control, scales, biases)
            projected = projected_workspace_bytes(weight)
            before = safety.checkpoint(f"layer_{layer}_expert_{expert}_{projection}_preflight")
            if before.process_physical_footprint_bytes + projected > safety.policy.maximum_process_physical_footprint_bytes:
                raise RuntimeError(f"PW-0141 layer {layer} expert {expert} exceeds Gate 8 headroom")
            candidate, codes, diagnostics = global_hessian_gptq_fixed_grid(
                weight, activations[projection], scales, biases
            )
            candidate_output = dense_projection(activations[projection], candidate)
            control_output = dense_projection(activations[projection], control)
            selected[projection] = candidate
            baseline[projection] = control
            projection_reports[projection] = {
                **diagnostics,
                "candidate_train_metrics": error_metrics(candidate_output, expected[projection]),
                "baseline_train_metrics": error_metrics(control_output, expected[projection]),
            }
            del scales, biases, control_codes, codes, candidate_output, control_output
            gc.collect()
            safety.release_checkpoint(
                f"layer_{layer}_expert_{expert}_{projection}_workspace_released",
                ["rotated weight", "full Hessian", "inverse Cholesky", "projection outputs"],
            )
        rotated_candidate_train = rotated_expert(rotated_train, selected, signs)
        rotated_candidate_validation = rotated_expert(rotated_validation, selected, signs)
        rotated_control_validation = rotated_expert(rotated_validation, baseline, signs)
        prior_expert = _prior_expert(prior, layer, expert)
        reports.append(
            {
                "layer": layer,
                "expert": expert,
                "train_placements": len(train_positions),
                "validation_placements": len(validation_positions),
                "unquantized_rotation_parity": parity,
                "projection_reports": projection_reports,
                "rotated_train_metrics": error_metrics(rotated_candidate_train, train_expected),
                "rotated_validation_metrics": error_metrics(rotated_candidate_validation, validation_expected),
                "rotated_rtn_validation_metrics": error_metrics(rotated_control_validation, validation_expected),
                "pw0139_validation_metrics": prior_expert["candidate_validation_metrics"],
            }
        )
        del moe_input, expert_down, train_expected, validation_expected, source_weights
        del rotated_weights, rotated_train, rotated_validation, hidden, activations, expected
        del selected, baseline, rotated_candidate_train, rotated_candidate_validation
        del rotated_control_validation
        gc.collect()
        mx.clear_cache()
        safety.release_checkpoint(
            f"layer_{layer}_expert_{expert}_complete_released",
            ["source expert", "rotated candidates", "captured activations", "expert outputs"],
        )
    gate = _gate(reports)
    decision = (
        "authorize_broader_fixed_residual_rotation_confirmation"
        if gate["passes"]
        else "reject_fixed_residual_hadamard_rotation"
    )
    safety.release_checkpoint(
        "checkpoint_and_authorities_released",
        ["checkpoint mappings", "PW-0116 corpus", "PW-0139/PW-0140 authorities"],
    )
    safety.checkpoint("final_service_health")
    result = {
        "schema_version": 1,
        "evidence_class": "pw0141_fixed_residual_hadamard_rotation_control",
        "revision": REVISION,
        "commit": commit,
        "source_hashes": {
            "checkpoint_verification_sha256": VERIFICATION_SHA256,
            "corpus_manifest_sha256": CORPUS_SHA256,
            "pw0139_report_sha256": PW0139_SHA256,
            "pw0140_report_sha256": PW0140_SHA256,
        },
        "samples": [list(row) for row in SAMPLES],
        "rotation": {
            "label": ROTATION_LABEL,
            "matrix": "Q = D H_4096",
            "signs_sha256": signs_sha256,
            "shared_across_layers": True,
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
            "three validation-visible experts only; local residual rotation oracle, not a "
            "whole-model rotated checkpoint; dense unpacked execution; holdout sealed; no bank, "
            "runtime, accumulated model, modalities, endpoint, accepted tokens, or TPS"
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
    parser.add_argument("--pw0140", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--commit", required=True)
    args = parser.parse_args()
    try:
        result = run(
            args.checkpoint, args.verification, args.corpus_manifest, args.pw0139,
            args.pw0140, args.output, args.commit,
        )
        print(json.dumps({"output": str(args.output), "decision": result["decision"]}))
        return 0
    except (OSError, RuntimeError, ValueError, KeyError, TypeError, json.JSONDecodeError, np.linalg.LinAlgError) as error:
        print(json.dumps({"error": str(error)}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
