#!/usr/bin/env python3
"""Run PW-0135's group-local GPTQ three-expert capacity control."""

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
    from tools.run_real_activation_affine_int4_audit import (
        CORPUS_SHA256,
        GROUP_SIZE,
        REVISION,
        SOURCE_EXPERT_BYTES,
        VERIFICATION_SHA256,
        _candidate_expert,
        _quantize_expert,
        error_metrics,
        validate_routes,
    )
except ModuleNotFoundError:
    from generate_real_layer1_expert_oracle import ShardedCheckpoint
    from host_safety import HostSafetyMonitor
    from openrouter_reference import atomic_write_new, canonical_json
    from run_best_rank_real_expert_control import dequant_weight, load_capture, sha256_file, source_linear
    from run_real_activation_affine_int4_audit import (
        CORPUS_SHA256,
        GROUP_SIZE,
        REVISION,
        SOURCE_EXPERT_BYTES,
        VERIFICATION_SHA256,
        _candidate_expert,
        _quantize_expert,
        error_metrics,
        validate_routes,
    )


PW0129_SHA256 = "1deb9dd85f0b598f31bc2d8bc1d41bf52cfabcda43de63a2ae5b3fdfad400306"
PW0134_SHA256 = "7d470bd5fa5541424c2b619afb49a2ebf493ce7a11b2498cf281b3d1c6f34490"
INT4_BYTES = 13_369_344
SAMPLES = ((4, 96, 109, 56), (24, 22, 26, 56), (46, 28, 100, 56))
DAMPINGS = (0.001, 0.01, 0.1)
ORDERS = ("natural", "activation")


def train_positions(positions: list[int]) -> list[int]:
    if any(not isinstance(position, int) or not 0 <= position < 168 for position in positions):
        raise ValueError("PW-0135 partition positions are invalid")
    return [position for position in positions if position < 112]


def unpack_int4_codes(packed: np.ndarray, columns: int) -> np.ndarray:
    if packed.ndim != 2 or packed.dtype != np.uint32 or packed.shape[1] * 8 != columns:
        raise ValueError("PW-0135 packed INT4 layout is invalid")
    codes = np.empty((packed.shape[0], columns), dtype=np.uint8)
    for nibble in range(8):
        codes[:, nibble::8] = ((packed >> (4 * nibble)) & 15).astype(np.uint8)
    return codes


def reconstruct_fixed_grid(
    codes: np.ndarray,
    scales: np.ndarray,
    biases: np.ndarray,
) -> np.ndarray:
    if codes.ndim != 2 or codes.dtype != np.uint8 or np.any(codes > 15):
        raise ValueError("PW-0135 fixed-grid codes are invalid")
    rows, columns = codes.shape
    if scales.shape != biases.shape or scales.shape[0] != rows or columns % scales.shape[1]:
        raise ValueError("PW-0135 fixed-grid reconstruction shape mismatch")
    group_size = columns // scales.shape[1]
    expanded_scale = np.repeat(scales, group_size, axis=1)
    expanded_bias = np.repeat(biases, group_size, axis=1)
    return (codes.astype(np.float32) * expanded_scale + expanded_bias).astype(np.float16)


def affine_grid(weight: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if weight.ndim != 2 or weight.shape[1] % GROUP_SIZE or not np.isfinite(weight).all():
        raise ValueError("PW-0135 affine-grid weight is invalid")
    arrays = mx.quantize(
        mx.array(np.asarray(weight, dtype=np.float16)),
        group_size=GROUP_SIZE,
        bits=4,
        mode="affine",
    )
    dequantized = mx.dequantize(
        *arrays, group_size=GROUP_SIZE, bits=4, mode="affine", dtype=mx.float16
    )
    mx.eval(*arrays, dequantized)
    packed = np.asarray(arrays[0]).astype(np.uint32, copy=True)
    scales = np.asarray(arrays[1]).astype(np.float32, copy=True)
    biases = np.asarray(arrays[2]).astype(np.float32, copy=True)
    result = np.asarray(dequantized).astype(np.float16, copy=True)
    codes = unpack_int4_codes(packed, weight.shape[1])
    if not np.array_equal(reconstruct_fixed_grid(codes, scales, biases), result):
        raise ValueError("PW-0135 decoded MLX INT4 control differs from MLX dequantization")
    del arrays, dequantized
    mx.clear_cache()
    return scales, biases, result, codes


def quantize_fixed_grid(
    values: np.ndarray,
    scales: np.ndarray,
    biases: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if values.ndim != 2 or scales.shape != biases.shape:
        raise ValueError("PW-0135 fixed-grid arguments are invalid")
    rows, columns = values.shape
    groups = scales.shape[1]
    if scales.shape[0] != rows or columns % groups:
        raise ValueError("PW-0135 fixed-grid shape mismatch")
    group_size = columns // groups
    expanded_scale = np.repeat(scales, group_size, axis=1)
    expanded_bias = np.repeat(biases, group_size, axis=1)
    zero_scale = expanded_scale == 0
    safe_scale = np.where(zero_scale, 1.0, expanded_scale)
    codes = np.clip(np.rint((values.astype(np.float32) - expanded_bias) / safe_scale), 0, 15)
    codes = np.where(zero_scale, 0, codes).astype(np.uint8)
    dequantized = reconstruct_fixed_grid(codes, scales, biases)
    return codes, dequantized


def validate_grid_membership(
    codes: np.ndarray,
    values: np.ndarray,
    scales: np.ndarray,
    biases: np.ndarray,
) -> None:
    if codes.dtype != np.uint8 or np.any(codes > 15):
        raise ValueError("PW-0135 codes leave four-bit domain")
    reconstructed = reconstruct_fixed_grid(codes, scales, biases)
    if not np.array_equal(values, reconstructed):
        raise ValueError("PW-0135 grid membership failed")


def gptq_fixed_grid(
    weight: np.ndarray,
    activations: np.ndarray,
    scales: np.ndarray,
    biases: np.ndarray,
    damping: float,
    order: str,
) -> tuple[np.ndarray, np.ndarray, dict]:
    if (
        weight.ndim != 2
        or activations.ndim != 2
        or activations.shape[1] != weight.shape[1]
        or activations.shape[0] == 0
        or scales.shape != biases.shape
        or weight.shape[0] != scales.shape[0]
        or weight.shape[1] % scales.shape[1]
        or damping not in DAMPINGS
        or order not in ORDERS
        or not np.isfinite(weight).all()
        or not np.isfinite(activations).all()
    ):
        raise ValueError("PW-0135 GPTQ inputs are invalid")
    group_size = weight.shape[1] // scales.shape[1]
    quantized = np.empty(weight.shape, dtype=np.float16)
    codes = np.empty(weight.shape, dtype=np.uint8)
    maximum_condition = 0.0
    dead_groups = 0
    for group in range(scales.shape[1]):
        start = group * group_size
        end = start + group_size
        x = activations[:, start:end].astype(np.float64)
        hessian = (x.T @ x) / max(x.shape[0], 1)
        diagonal = np.diag(hessian).copy()
        mean_diagonal = float(np.mean(diagonal))
        if mean_diagonal == 0.0:
            dead_groups += 1
        damp = max(damping * mean_diagonal, 1e-8)
        hessian.flat[:: group_size + 1] += damp
        permutation = (
            np.argsort(-diagonal, kind="stable")
            if order == "activation"
            else np.arange(group_size)
        )
        inverse_permutation = np.argsort(permutation)
        permuted_hessian = hessian[np.ix_(permutation, permutation)]
        permuted_diagonal = np.diag(permuted_hessian)
        condition = float(np.max(permuted_diagonal) / max(float(np.min(permuted_diagonal)), 1e-30))
        maximum_condition = max(maximum_condition, condition)
        inverse_cholesky = np.linalg.cholesky(np.linalg.inv(permuted_hessian)).T
        working = weight[:, start:end].astype(np.float64)[:, permutation].copy()
        group_codes = np.empty((weight.shape[0], group_size), dtype=np.uint8)
        group_values = np.empty((weight.shape[0], group_size), dtype=np.float16)
        group_scale = scales[:, group : group + 1].astype(np.float64)
        group_bias = biases[:, group : group + 1].astype(np.float64)
        safe_scale = np.where(group_scale == 0, 1.0, group_scale)
        for column in range(group_size):
            raw_code = np.clip(
                np.rint((working[:, column : column + 1] - group_bias) / safe_scale),
                0,
                15,
            )
            raw_code = np.where(group_scale == 0, 0, raw_code)
            value = raw_code * group_scale + group_bias
            group_codes[:, column : column + 1] = raw_code.astype(np.uint8)
            group_values[:, column : column + 1] = value.astype(np.float16)
            error = (working[:, column : column + 1] - value) / inverse_cholesky[column, column]
            working[:, column:] -= error * inverse_cholesky[column, column:].reshape(1, -1)
        codes[:, start:end] = group_codes[:, inverse_permutation]
        quantized[:, start:end] = group_values[:, inverse_permutation]
    validate_grid_membership(codes, quantized, scales, biases)
    return quantized, codes, {
        "maximum_hessian_condition": maximum_condition,
        "dead_activation_groups": dead_groups,
    }


def dense_projection(values: np.ndarray, weight: np.ndarray) -> np.ndarray:
    inputs = mx.array(np.asarray(values, dtype=np.float16))
    weights = mx.array(np.asarray(weight, dtype=np.float16))
    output = inputs @ weights.T
    mx.eval(output)
    result = np.asarray(output).astype(np.float32, copy=True)
    del inputs, weights, output
    mx.clear_cache()
    return result


def dense_expert(values: np.ndarray, weights: dict[str, np.ndarray]) -> np.ndarray:
    inputs = mx.array(np.asarray(values, dtype=np.float16))
    arrays = {name: mx.array(np.asarray(weight, dtype=np.float16)) for name, weight in weights.items()}
    gate = inputs @ arrays["gate"].T
    up = inputs @ arrays["up"].T
    hidden = mx.sigmoid(gate) * gate * up
    output = hidden @ arrays["down"].T
    mx.eval(output)
    result = np.asarray(output).astype(np.float32, copy=True)
    del inputs, arrays, gate, up, hidden, output
    mx.clear_cache()
    return result


def source_hidden(weights: dict[str, np.ndarray], inputs: np.ndarray) -> np.ndarray:
    values = torch.from_numpy(np.asarray(inputs).copy()).to(torch.bfloat16)
    gate = source_linear(weights["gate"], values)
    up = source_linear(weights["up"], values)
    return (torch.nn.functional.silu(gate) * up).to(torch.bfloat16).float().numpy()


def select_projection(
    weight: np.ndarray,
    activations: np.ndarray,
    expected: np.ndarray,
    safety: HostSafetyMonitor,
    phase_prefix: str,
) -> tuple[np.ndarray, dict]:
    scales, biases, baseline, baseline_codes = affine_grid(weight)
    validate_grid_membership(baseline_codes, baseline, scales, biases)
    curve = []
    best = None
    for damping in DAMPINGS:
        for order in ORDERS:
            candidate, codes, diagnostics = gptq_fixed_grid(
                weight, activations, scales, biases, damping, order
            )
            output = dense_projection(activations, candidate)
            metrics = error_metrics(output, expected)
            record = {"damping": damping, "order": order, "train_metrics": metrics, **diagnostics}
            curve.append(record)
            if best is None or (metrics["squared_error"], damping, order) < best[0]:
                best = ((metrics["squared_error"], damping, order), candidate.copy(), codes.copy(), record)
            del candidate, codes, output
            gc.collect()
            safety.release_checkpoint(
                f"{phase_prefix}_damping_{damping}_{order}_released",
                ["GPTQ candidate weight", "projection output", "Hessian workspaces"],
            )
    assert best is not None
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(best[2]).tobytes())
    digest.update(np.ascontiguousarray(scales.astype("<f4")).tobytes())
    digest.update(np.ascontiguousarray(biases.astype("<f4")).tobytes())
    return best[1], {
        "selected_damping": best[3]["damping"],
        "selected_order": best[3]["order"],
        "selected_grid_sha256": digest.hexdigest(),
        "baseline_train_metrics": error_metrics(dense_projection(activations, baseline), expected),
        "selected_train_metrics": best[3]["train_metrics"],
        "curve": curve,
    }


def physical_ledger() -> dict:
    return {
        "packed_bytes_per_expert": INT4_BYTES,
        "packed_to_source_ratio": INT4_BYTES / SOURCE_EXPERT_BYTES,
        "additional_runtime_macs": 0,
    }


def _prior_expert(prior: dict, layer: int, expert: int, partition: str) -> dict:
    matches = [
        report
        for row in prior["reports"]
        if row["layer"] == layer and row["bits"] == 4
        for report in row[partition]["expert_reports"]
        if report["expert"] == expert
    ]
    if len(matches) != 1:
        raise ValueError("PW-0135 PW-0129 expert authority mismatch")
    return matches[0]


def _gate(reports: list[dict]) -> dict:
    rows = []
    for report in reports:
        baseline = report["dense_control_validation"]["relative_l2"]
        candidate = report["gptq_validation"]["relative_l2"]
        reduction = 1.0 - candidate / max(baseline, 1e-30)
        passes = (
            reduction >= 0.50
            and candidate <= 0.08
            and report["gptq_validation"]["maximum_row_relative_l2"] <= 0.12
            and report["gptq_train"]["relative_l2"] < report["dense_control_train"]["relative_l2"]
        )
        rows.append(
            {
                "layer": report["layer"],
                "expert": report["expert"],
                "validation_relative_error_reduction": reduction,
                "candidate_validation_relative_l2": candidate,
                "candidate_maximum_row_relative_l2": report["gptq_validation"]["maximum_row_relative_l2"],
                "train_improves": report["gptq_train"]["relative_l2"] < report["dense_control_train"]["relative_l2"],
                "passes": passes,
            }
        )
    physical = physical_ledger()
    return {
        "experts": rows,
        "physical": physical,
        "passes": all(row["passes"] for row in rows) and physical["packed_to_source_ratio"] <= 0.60,
        "thresholds": {
            "minimum_validation_relative_error_reduction": 0.50,
            "maximum_validation_relative_l2": 0.08,
            "maximum_validation_row_relative_l2": 0.12,
            "train_must_improve": True,
            "maximum_source_byte_ratio": 0.60,
        },
    }


def run(
    checkpoint_root: Path,
    verification_path: Path,
    corpus_manifest_path: Path,
    pw0129_path: Path,
    pw0134_path: Path,
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
        (pw0134_path, PW0134_SHA256, "PW-0134 report"),
    ):
        if sha256_file(path) != expected:
            raise ValueError(f"PW-0135 {label} hash mismatch")
    started = time.perf_counter()
    safety = HostSafetyMonitor()
    corpus = json.loads(corpus_manifest_path.read_text())
    prior = json.loads(pw0129_path.read_text())
    awq = json.loads(pw0134_path.read_text())
    if (
        corpus.get("revision") != REVISION
        or prior.get("decision") != "reject_naive_affine_int4_on_real_validation"
        or awq.get("decision") != "reject_awq_activation_mean_scale_family"
        or prior.get("holdout_unsealed")
        or awq.get("holdout_unsealed")
    ):
        raise ValueError("PW-0135 authority identity mismatch")
    checkpoint = ShardedCheckpoint(checkpoint_root, verification_path)
    root = corpus_manifest_path.parent
    reports = []
    for layer, expert, expected_train_count, expected_validation_count in SAMPLES:
        authority = next(row for row in corpus["layers"] if row["layer"] == layer)
        validate_routes(authority)
        schedule = next(row for row in authority["expert_schedule"] if row["expert"] == expert)
        train_local = [index for index, position in enumerate(schedule["positions"]) if position < 112]
        validation_local = [index for index, position in enumerate(schedule["positions"]) if 112 <= position < 168]
        if len(train_local) != expected_train_count or len(validation_local) != expected_validation_count:
            raise ValueError("PW-0135 frozen sample coverage mismatch")
        offset = sum(
            len(row["positions"])
            for row in authority["expert_schedule"][: authority["expert_schedule"].index(schedule)]
        )
        moe_input = load_capture(root, authority["captures"]["moe_input"])
        expert_down = load_capture(root, authority["captures"]["expert_down"])
        train_ids = [schedule["positions"][index] for index in train_local]
        validation_ids = [schedule["positions"][index] for index in validation_local]
        train_expected = np.asarray(expert_down[[offset + index for index in train_local]], dtype=np.float32).copy()
        validation_expected = np.asarray(expert_down[[offset + index for index in validation_local]], dtype=np.float32).copy()
        prefix = f"model.layers.{layer}.mlp.experts.{expert}"
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
                source_weights["gate"], torch.from_numpy(np.asarray(moe_input[train_ids]).copy()).to(torch.bfloat16)
            ).float().numpy(),
            "up": source_linear(
                source_weights["up"], torch.from_numpy(np.asarray(moe_input[train_ids]).copy()).to(torch.bfloat16)
            ).float().numpy(),
            "down": train_expected,
        }
        selected_weights = {}
        projection_reports = {}
        for projection in ("gate", "up", "down"):
            selected_weights[projection], projection_reports[projection] = select_projection(
                source_weights[projection],
                projection_activations[projection],
                projection_expected[projection],
                safety,
                f"layer_{layer}_expert_{expert}_{projection}",
            )
        baseline_weights = {
            projection: affine_grid(source_weights[projection])[2]
            for projection in ("gate", "up", "down")
        }
        dense_control_train = dense_expert(moe_input[train_ids], baseline_weights)
        dense_control_validation = dense_expert(moe_input[validation_ids], baseline_weights)
        gptq_train = dense_expert(moe_input[train_ids], selected_weights)
        gptq_validation = dense_expert(moe_input[validation_ids], selected_weights)
        packed, packed_meta = _quantize_expert(checkpoint, layer, expert, 4)
        packed_train, _ = _candidate_expert(moe_input[train_ids], packed, 4)
        packed_validation, _ = _candidate_expert(moe_input[validation_ids], packed, 4)
        packed_validation_metrics = error_metrics(packed_validation, validation_expected)
        if packed_meta["packed_sha256"] != _prior_expert(prior, layer, expert, "validation")["packed_sha256"]:
            raise ValueError("PW-0135 packed control identity mismatch")
        if packed_validation_metrics != _prior_expert(prior, layer, expert, "validation")["expert_output_metrics"]:
            raise ValueError("PW-0135 packed control metrics mismatch")
        reports.append(
            {
                "layer": layer,
                "expert": expert,
                "train_placements": len(train_ids),
                "validation_placements": len(validation_ids),
                "projection_selections": projection_reports,
                "dense_control_train": error_metrics(dense_control_train, train_expected),
                "dense_control_validation": error_metrics(dense_control_validation, validation_expected),
                "packed_control_train": error_metrics(packed_train, train_expected),
                "packed_control_validation": packed_validation_metrics,
                "dense_vs_packed_control_train": error_metrics(dense_control_train, packed_train),
                "dense_vs_packed_control_validation": error_metrics(dense_control_validation, packed_validation),
                "gptq_train": error_metrics(gptq_train, train_expected),
                "gptq_validation": error_metrics(gptq_validation, validation_expected),
            }
        )
        del moe_input, expert_down, train_expected, validation_expected, source_weights
        del hidden, projection_activations, projection_expected, selected_weights, baseline_weights
        del dense_control_train, dense_control_validation, gptq_train, gptq_validation
        del packed, packed_train, packed_validation
        gc.collect()
        mx.clear_cache()
        safety.release_checkpoint(
            f"layer_{layer}_expert_{expert}_complete_control_released",
            ["source expert", "GPTQ candidates", "dense controls", "captured activations"],
        )
    gate = _gate(reports)
    decision = "authorize_full_layer_group_local_gptq_audit" if gate["passes"] else "reject_group_local_fixed_grid_gptq"
    safety.release_checkpoint(
        "checkpoint_and_authorities_released",
        ["checkpoint mappings", "PW-0116 corpus", "PW-0129/PW-0134 reports"],
    )
    safety.checkpoint("final_service_health")
    report = {
        "schema_version": 1,
        "evidence_class": "pw0135_group_local_gptq_three_expert_control",
        "revision": REVISION,
        "commit": commit,
        "source_hashes": {
            "checkpoint_verification_sha256": VERIFICATION_SHA256,
            "corpus_manifest_sha256": CORPUS_SHA256,
            "pw0129_report_sha256": PW0129_SHA256,
            "pw0134_report_sha256": PW0134_SHA256,
        },
        "samples": [list(row) for row in SAMPLES],
        "dampings": list(DAMPINGS),
        "orders": list(ORDERS),
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
            "three highest-validation-coverage experts only; group-local fixed-grid GPTQ; "
            "dense unpacked execution oracle, no packed GPTQ kernel; one English pilot; "
            "holdout sealed; no full layer, accumulated model, modalities, endpoint, or TPS"
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
    parser.add_argument("--pw0134", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--commit", required=True)
    arguments = parser.parse_args()
    try:
        result = run(
            arguments.checkpoint, arguments.verification, arguments.corpus_manifest,
            arguments.pw0129, arguments.pw0134, arguments.output, arguments.commit,
        )
        print(json.dumps({"output": str(arguments.output), "decision": result["decision"]}))
        return 0
    except (OSError, RuntimeError, ValueError, KeyError, TypeError, json.JSONDecodeError, np.linalg.LinAlgError) as error:
        print(json.dumps({"error": str(error)}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
