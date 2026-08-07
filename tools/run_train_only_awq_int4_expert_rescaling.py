#!/usr/bin/env python3
"""Run PW-0134's train-only AWQ-style INT4 expert rescaling audit."""

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
    from tools.run_int4_output_affine_repair_oracle import _route_rows
    from tools.run_real_activation_affine_int4_audit import (
        CORPUS_SHA256,
        GROUP_SIZE,
        LAYERS,
        REVISION,
        SOURCE_EXPERT_BYTES,
        VERIFICATION_SHA256,
        _candidate_expert,
        _quantize_expert,
        error_metrics,
        quantized_projection,
        validate_routes,
    )
except ModuleNotFoundError:
    from generate_real_layer1_expert_oracle import ShardedCheckpoint
    from host_safety import HostSafetyMonitor
    from openrouter_reference import atomic_write_new, canonical_json
    from run_best_rank_real_expert_control import dequant_weight, load_capture, sha256_file, source_linear
    from run_int4_output_affine_repair_oracle import _route_rows
    from run_real_activation_affine_int4_audit import (
        CORPUS_SHA256,
        GROUP_SIZE,
        LAYERS,
        REVISION,
        SOURCE_EXPERT_BYTES,
        VERIFICATION_SHA256,
        _candidate_expert,
        _quantize_expert,
        error_metrics,
        quantized_projection,
        validate_routes,
    )


PW0129_SHA256 = "1deb9dd85f0b598f31bc2d8bc1d41bf52cfabcda43de63a2ae5b3fdfad400306"
PW0133_SHA256 = "a0226e42058a04ea1009a6c00a6b44fdc85728bf36e383166a589b1d3e28b0d8"
INT4_BYTES = 13_369_344
ALPHAS = tuple(round(index / 20, 2) for index in range(20))


def activation_mean_abs(values: np.ndarray) -> np.ndarray:
    if values.ndim != 2 or values.shape[0] == 0 or not np.isfinite(values).all():
        raise ValueError("PW-0134 activation calibration matrix is invalid")
    return np.mean(np.abs(np.asarray(values, dtype=np.float32)), axis=0)


def awq_scale(mean_abs: np.ndarray, alpha: float) -> np.ndarray:
    if (
        mean_abs.ndim != 1
        or mean_abs.size == 0
        or not np.isfinite(mean_abs).all()
        or np.any(mean_abs < 0)
        or alpha not in ALPHAS
    ):
        raise ValueError("PW-0134 AWQ scale inputs are invalid")
    scale = np.maximum(mean_abs.astype(np.float64) ** alpha, 1e-4)
    normalization = np.sqrt(float(np.max(scale)) * float(np.min(scale)))
    # The runtime artifact is charged and reproduced as F16 scale vectors.
    result = (scale / normalization).astype(np.float16).astype(np.float32)
    if not np.isfinite(result).all() or np.any(result <= 0):
        raise ValueError("PW-0134 AWQ scale is non-finite")
    return result


def transform_expert_weights(
    weights: dict[str, np.ndarray],
    input_scale: np.ndarray,
    hidden_scale: np.ndarray,
) -> dict[str, np.ndarray]:
    if set(weights) != {"gate", "up", "down"}:
        raise ValueError("PW-0134 expert transform shape mismatch")
    hidden, width = weights["gate"].shape
    if (
        weights["gate"].ndim != 2
        or weights["up"].shape != (hidden, width)
        or weights["down"].shape != (width, hidden)
        or input_scale.shape != (width,)
        or hidden_scale.shape != (hidden,)
        or np.any(input_scale <= 0)
        or np.any(hidden_scale <= 0)
    ):
        raise ValueError("PW-0134 expert transform shape mismatch")
    transformed = {
        "gate": weights["gate"].astype(np.float32) * input_scale.reshape(1, -1),
        "up": (
            weights["up"].astype(np.float32)
            * input_scale.reshape(1, -1)
            / hidden_scale.reshape(-1, 1)
        ),
        "down": weights["down"].astype(np.float32) * hidden_scale.reshape(1, -1),
    }
    if any(not np.isfinite(value).all() for value in transformed.values()):
        raise ValueError("PW-0134 transformed weights are non-finite")
    return transformed


def transform_reconstruction_error(
    weights: dict[str, np.ndarray],
    transformed: dict[str, np.ndarray],
    input_scale: np.ndarray,
    hidden_scale: np.ndarray,
) -> float:
    reconstructed = (
        transformed["gate"] / input_scale.reshape(1, -1),
        transformed["up"] * hidden_scale.reshape(-1, 1) / input_scale.reshape(1, -1),
        transformed["down"] / hidden_scale.reshape(1, -1),
    )
    originals = (weights["gate"], weights["up"], weights["down"])
    numerator = sum(
        float(np.sum((actual.astype(np.float64) - expected.astype(np.float64)) ** 2))
        for actual, expected in zip(reconstructed, originals)
    )
    denominator = sum(float(np.sum(value.astype(np.float64) ** 2)) for value in originals)
    return float(np.sqrt(numerator / max(denominator, 1e-30)))


def physical_ledger() -> dict:
    scale_bytes = (4096 + 2048) * 2
    combined = INT4_BYTES + scale_bytes
    source_macs = 3 * 4096 * 2048
    return {
        "int4_bytes_per_expert": INT4_BYTES,
        "conservative_f16_scale_bytes_per_expert": scale_bytes,
        "combined_bytes_per_expert": combined,
        "combined_to_source_ratio": combined / SOURCE_EXPERT_BYTES,
        "runtime_input_divides_per_expert": 4096,
        "runtime_elementwise_to_source_expert_mac_ratio": 4096 / source_macs,
    }


def _array_digest(digest: "hashlib._Hash", name: str, array: mx.array) -> None:
    values = np.asarray(array)
    digest.update(name.encode("ascii"))
    digest.update(str(values.dtype).encode("ascii"))
    digest.update(json.dumps(list(values.shape)).encode("ascii"))
    digest.update(np.ascontiguousarray(values).tobytes())


def _quantize_projection(weight: np.ndarray) -> tuple[mx.array, mx.array, mx.array]:
    if weight.ndim != 2 or weight.shape[1] % GROUP_SIZE or not np.isfinite(weight).all():
        raise ValueError("PW-0134 projection weight is invalid")
    arrays = mx.quantize(
        mx.array(np.asarray(weight, dtype=np.float16)),
        group_size=GROUP_SIZE,
        bits=4,
        mode="affine",
    )
    mx.eval(*arrays)
    return arrays


def _quantize_transformed_expert(weights: dict[str, np.ndarray]) -> tuple[dict, dict]:
    projections = {}
    digest = hashlib.sha256()
    packed_bytes = 0
    for projection in ("gate", "up", "down"):
        arrays = _quantize_projection(weights[projection])
        projections[projection] = arrays
        for index, array in enumerate(arrays):
            _array_digest(digest, f"{projection}:{index}", array)
            packed_bytes += int(array.nbytes)
    return projections, {"packed_sha256": digest.hexdigest(), "packed_bytes": packed_bytes}


def _source_hidden(weights: dict[str, np.ndarray], inputs: np.ndarray) -> np.ndarray:
    values = torch.from_numpy(np.asarray(inputs).copy()).to(torch.bfloat16)
    gate = source_linear(weights["gate"], values)
    up = source_linear(weights["up"], values)
    return (torch.nn.functional.silu(gate) * up).to(torch.bfloat16).float().numpy()


def _input_scale_objective(
    weights: dict[str, np.ndarray],
    inputs: np.ndarray,
    expected: np.ndarray,
    scale: np.ndarray,
) -> float:
    gate_arrays = _quantize_projection(weights["gate"] * scale.reshape(1, -1))
    up_arrays = _quantize_projection(weights["up"] * scale.reshape(1, -1))
    values = mx.array(np.asarray(inputs / scale.reshape(1, -1), dtype=np.float16))
    gate = quantized_projection(values, gate_arrays, 4)
    up = quantized_projection(values, up_arrays, 4)
    hidden = mx.sigmoid(gate) * gate * up
    mx.eval(hidden)
    hidden_np = np.asarray(hidden).astype(np.float32, copy=True)
    output = source_linear(
        weights["down"], torch.from_numpy(hidden_np).to(torch.bfloat16)
    ).float().numpy()
    loss = error_metrics(output, expected)["squared_error"]
    del gate_arrays, up_arrays, values, gate, up, hidden, hidden_np, output
    mx.clear_cache()
    return loss


def _hidden_scale_objective(
    down_weight: np.ndarray,
    hidden: np.ndarray,
    expected: np.ndarray,
    scale: np.ndarray,
) -> float:
    arrays = _quantize_projection(down_weight * scale.reshape(1, -1))
    values = mx.array(np.asarray(hidden / scale.reshape(1, -1), dtype=np.float16))
    output = quantized_projection(values, arrays, 4)
    mx.eval(output)
    result = np.asarray(output).astype(np.float32, copy=True)
    loss = error_metrics(result, expected)["squared_error"]
    del arrays, values, output, result
    mx.clear_cache()
    return loss


def calibrate_expert(
    weights: dict[str, np.ndarray],
    inputs: np.ndarray,
    expected: np.ndarray,
) -> dict:
    input_mean = activation_mean_abs(inputs)
    hidden = _source_hidden(weights, inputs)
    hidden_mean = activation_mean_abs(hidden)
    input_curve = []
    hidden_curve = []
    for alpha in ALPHAS:
        input_curve.append(
            {"alpha": alpha, "squared_error": _input_scale_objective(weights, inputs, expected, awq_scale(input_mean, alpha))}
        )
    for alpha in ALPHAS:
        hidden_curve.append(
            {"alpha": alpha, "squared_error": _hidden_scale_objective(weights["down"], hidden, expected, awq_scale(hidden_mean, alpha))}
        )
    input_best = min(input_curve, key=lambda row: (row["squared_error"], row["alpha"]))
    hidden_best = min(hidden_curve, key=lambda row: (row["squared_error"], row["alpha"]))
    return {
        "input_alpha": input_best["alpha"],
        "hidden_alpha": hidden_best["alpha"],
        "input_curve": input_curve,
        "hidden_curve": hidden_curve,
    }


def _candidate(
    inputs: np.ndarray,
    projections: dict,
    input_scale: np.ndarray,
) -> np.ndarray:
    values = mx.array(np.asarray(inputs / input_scale.reshape(1, -1), dtype=np.float16))
    gate = quantized_projection(values, projections["gate"], 4)
    up = quantized_projection(values, projections["up"], 4)
    hidden = mx.sigmoid(gate) * gate * up
    output = quantized_projection(hidden, projections["down"], 4)
    mx.eval(output)
    result = np.asarray(output).astype(np.float32, copy=True)
    del values, gate, up, hidden, output
    return result


def _scale_digest(input_scale: np.ndarray, hidden_scale: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(input_scale.astype("<f2")).tobytes())
    digest.update(np.ascontiguousarray(hidden_scale.astype("<f2")).tobytes())
    return digest.hexdigest()


def _prior_validation(prior: dict, layer: int) -> dict:
    matches = [row["validation"] for row in prior["reports"] if row["layer"] == layer and row["bits"] == 4]
    if len(matches) != 1:
        raise ValueError("PW-0134 PW-0129 validation authority mismatch")
    return matches[0]


def _gate(layer_reports: list[dict]) -> dict:
    metrics = [row["candidate_validation"] for row in layer_reports]
    squared_error = sum(row["squared_error"] for row in metrics)
    expected_norm = sum(row["expected_squared_norm"] for row in metrics)
    aggregate = float(np.sqrt(squared_error / max(expected_norm, 1e-30)))
    physical = physical_ledger()
    strict = (
        aggregate <= 0.01
        and all(row["relative_l2"] <= 0.02 for row in metrics)
        and all(row["maximum_row_relative_l2"] <= 0.05 for row in metrics)
        and physical["combined_to_source_ratio"] <= 0.60
        and physical["runtime_elementwise_to_source_expert_mac_ratio"] <= 0.01
    )
    near = (
        not strict
        and aggregate <= 0.02
        and all(row["relative_l2"] <= 0.04 for row in metrics)
        and all(row["maximum_row_relative_l2"] <= 0.08 for row in metrics)
        and physical["combined_to_source_ratio"] <= 0.60
        and physical["runtime_elementwise_to_source_expert_mac_ratio"] <= 0.01
    )
    return {
        "aggregate_relative_l2": aggregate,
        "maximum_layer_relative_l2": max(row["relative_l2"] for row in metrics),
        "maximum_row_relative_l2": max(row["maximum_row_relative_l2"] for row in metrics),
        "physical": physical,
        "strict_pass": strict,
        "near_miss": near,
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
    pw0133_path: Path,
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
        (pw0133_path, PW0133_SHA256, "PW-0133 report"),
    ):
        if sha256_file(path) != expected:
            raise ValueError(f"PW-0134 {label} hash mismatch")
    started = time.perf_counter()
    safety = HostSafetyMonitor()
    corpus = json.loads(corpus_manifest_path.read_text())
    prior = json.loads(pw0129_path.read_text())
    exception_result = json.loads(pw0133_path.read_text())
    if (
        corpus.get("revision") != REVISION
        or prior.get("decision") != "reject_naive_affine_int4_on_real_validation"
        or exception_result.get("decision") != "reject_diagonal_sensitivity_source_fp8_exception_store"
        or prior.get("holdout_unsealed")
        or exception_result.get("holdout_unsealed")
    ):
        raise ValueError("PW-0134 authority identity mismatch")
    checkpoint = ShardedCheckpoint(checkpoint_root, verification_path)
    root = corpus_manifest_path.parent
    layer_reports = []
    for layer in LAYERS:
        authority = next(row for row in corpus["layers"] if row["layer"] == layer)
        validate_routes(authority)
        moe_input = load_capture(root, authority["captures"]["moe_input"])
        expert_down = load_capture(root, authority["captures"]["expert_down"])
        routed_expected = load_capture(root, authority["captures"]["routed_output"])
        entries = []
        offset = 0
        for schedule in authority["expert_schedule"]:
            validation_local = [index for index, position in enumerate(schedule["positions"]) if 112 <= position < 168]
            if validation_local:
                entries.append(
                    {
                        "expert": schedule["expert"],
                        "train_positions": [position for position in schedule["positions"] if position < 112],
                        "validation_positions": [schedule["positions"][index] for index in validation_local],
                        "validation_local": validation_local,
                        "offset": offset,
                    }
                )
            offset += len(schedule["positions"])

        calibrations = {}
        for entry in entries:
            if not entry["train_positions"]:
                continue
            expert = entry["expert"]
            prefix = f"model.layers.{layer}.mlp.experts.{expert}"
            weights = {
                projection: dequant_weight(checkpoint, f"{prefix}.{projection}_proj.weight")
                for projection in ("gate", "up", "down")
            }
            schedule = next(row for row in authority["expert_schedule"] if row["expert"] == expert)
            train_local = [index for index, position in enumerate(schedule["positions"]) if position < 112]
            expected = np.asarray(
                expert_down[[entry["offset"] + index for index in train_local]], dtype=np.float32
            ).copy()
            calibrations[expert] = calibrate_expert(weights, moe_input[entry["train_positions"]], expected)
            del weights, expected
            gc.collect()
            mx.clear_cache()
            safety.release_checkpoint(
                f"layer_{layer}_expert_{expert}_awq_calibration_released",
                ["source expert", "AWQ search quantizations", "training activations"],
            )
        if not calibrations:
            raise ValueError("PW-0134 layer has no train-seen validation expert")
        median_input_alpha = float(np.median([row["input_alpha"] for row in calibrations.values()]))
        median_hidden_alpha = float(np.median([row["hidden_alpha"] for row in calibrations.values()]))
        # Median of grid values may lie halfway between points; select the lower nearest declared grid value.
        median_input_alpha = min(ALPHAS, key=lambda value: (abs(value - median_input_alpha), value))
        median_hidden_alpha = min(ALPHAS, key=lambda value: (abs(value - median_hidden_alpha), value))

        baseline_rows = {}
        candidate_rows = {}
        expert_reports = []
        prior_hashes = {
            row["expert"]: row["packed_sha256"]
            for row in _prior_validation(prior, layer)["expert_reports"]
        }
        for entry in entries:
            expert = entry["expert"]
            prefix = f"model.layers.{layer}.mlp.experts.{expert}"
            weights = {
                projection: dequant_weight(checkpoint, f"{prefix}.{projection}_proj.weight")
                for projection in ("gate", "up", "down")
            }
            if entry["train_positions"]:
                inputs = moe_input[entry["train_positions"]]
                calibration = calibrations[expert]
                input_alpha = calibration["input_alpha"]
                hidden_alpha = calibration["hidden_alpha"]
                fallback = "none"
            else:
                inputs = moe_input[:112]
                calibration = None
                input_alpha = median_input_alpha
                hidden_alpha = median_hidden_alpha
                fallback = "layer_pooled_activations_and_median_alphas"
            source_hidden = _source_hidden(weights, inputs)
            input_scale = awq_scale(activation_mean_abs(inputs), input_alpha)
            hidden_scale = awq_scale(activation_mean_abs(source_hidden), hidden_alpha)
            transformed = transform_expert_weights(weights, input_scale, hidden_scale)
            algebra_error = transform_reconstruction_error(weights, transformed, input_scale, hidden_scale)
            if algebra_error > 1e-6:
                raise ValueError("PW-0134 exact transform algebra failed")
            projections, packed = _quantize_transformed_expert(transformed)
            baseline_projections, baseline_packed = _quantize_expert(checkpoint, layer, expert, 4)
            if baseline_packed["packed_sha256"] != prior_hashes.get(expert):
                raise ValueError("PW-0134 PW-0129 packed baseline mismatch")
            validation_inputs = moe_input[entry["validation_positions"]]
            baseline, _ = _candidate_expert(validation_inputs, baseline_projections, 4)
            candidate = _candidate(validation_inputs, projections, input_scale)
            expected = np.asarray(
                expert_down[[entry["offset"] + index for index in entry["validation_local"]]], dtype=np.float32
            ).copy()
            baseline_rows[expert] = {"positions": entry["validation_positions"], "candidate": baseline}
            candidate_rows[expert] = {"positions": entry["validation_positions"], "candidate": candidate}
            expert_reports.append(
                {
                    "expert": expert,
                    "train_placements": len(entry["train_positions"]),
                    "validation_placements": len(entry["validation_positions"]),
                    "fallback": fallback,
                    "input_alpha": input_alpha,
                    "hidden_alpha": hidden_alpha,
                    "scale_sha256": _scale_digest(input_scale, hidden_scale),
                    "packed_sha256": packed["packed_sha256"],
                    "packed_bytes": packed["packed_bytes"],
                    "transform_reconstruction_relative_l2": algebra_error,
                    "baseline_expert_output_metrics": error_metrics(baseline, expected),
                    "candidate_expert_output_metrics": error_metrics(candidate, expected),
                    "calibration": calibration,
                }
            )
            del weights, inputs, source_hidden, input_scale, hidden_scale, transformed
            del projections, baseline_projections, baseline, candidate, expected
            gc.collect()
            mx.clear_cache()
            if mx.get_active_memory() != 0:
                raise ValueError("PW-0134 MLX expert buffers did not release")
            safety.release_checkpoint(
                f"layer_{layer}_expert_{expert}_awq_candidate_released",
                ["source expert", "baseline INT4", "AWQ INT4", "expert activations"],
            )

        baseline_routed = _route_rows(baseline_rows, authority, "candidate", 112, 168)
        candidate_routed = _route_rows(candidate_rows, authority, "candidate", 112, 168)
        expected_routed = np.asarray(routed_expected[112:168], dtype=np.float32)
        baseline_metrics = error_metrics(baseline_routed, expected_routed)
        if baseline_metrics != _prior_validation(prior, layer)["routed_output_metrics"]:
            raise ValueError("PW-0134 PW-0129 routed baseline mismatch")
        layer_reports.append(
            {
                "layer": layer,
                "median_input_alpha": median_input_alpha,
                "median_hidden_alpha": median_hidden_alpha,
                "baseline_validation": baseline_metrics,
                "candidate_validation": error_metrics(candidate_routed, expected_routed),
                "fallback_experts": sum(row["fallback"] != "none" for row in expert_reports),
                "fallback_validation_placements": sum(
                    row["validation_placements"] for row in expert_reports if row["fallback"] != "none"
                ),
                "experts": expert_reports,
            }
        )
        del moe_input, expert_down, routed_expected, entries, calibrations
        del baseline_rows, candidate_rows, baseline_routed, candidate_routed, expected_routed
        gc.collect()
        mx.clear_cache()
        safety.release_checkpoint(
            f"layer_{layer}_awq_corpus_released",
            ["corpus captures", "calibration curves", "routed candidates"],
        )

    gate = _gate(layer_reports)
    decision = (
        "authorize_awq_int4_holdout_and_kernel_probe"
        if gate["strict_pass"]
        else (
            "authorize_awq_exception_composition"
            if gate["near_miss"]
            else "reject_awq_activation_mean_scale_family"
        )
    )
    safety.release_checkpoint(
        "checkpoint_and_authorities_released",
        ["checkpoint mappings", "PW-0116 corpus", "PW-0129/PW-0133 reports"],
    )
    safety.checkpoint("final_service_health")
    report = {
        "schema_version": 1,
        "evidence_class": "pw0134_train_only_awq_int4_expert_rescaling",
        "revision": REVISION,
        "commit": commit,
        "source_hashes": {
            "checkpoint_verification_sha256": VERIFICATION_SHA256,
            "corpus_manifest_sha256": CORPUS_SHA256,
            "pw0129_report_sha256": PW0129_SHA256,
            "pw0133_report_sha256": PW0133_SHA256,
        },
        "alphas": list(ALPHAS),
        "selection_partition": {"start": 0, "end_exclusive": 112},
        "validation_partition": {"start": 112, "end_exclusive": 168},
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
            "official AWQ activation-mean scale family adapted to independently routed MiMo "
            "experts with sequential gate/up and down searches; one correlated English pilot; "
            "holdout sealed; no accumulated model, modalities, endpoint, accepted tokens, or TPS"
        ),
        "platform": platform.platform(),
        "mlx_version": "0.31.2",
    }
    atomic_write_new(output_path, canonical_json(report))
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--verification", required=True, type=Path)
    parser.add_argument("--corpus-manifest", required=True, type=Path)
    parser.add_argument("--pw0129", required=True, type=Path)
    parser.add_argument("--pw0133", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--commit", required=True)
    arguments = parser.parse_args()
    try:
        result = run(
            arguments.checkpoint, arguments.verification, arguments.corpus_manifest,
            arguments.pw0129, arguments.pw0133, arguments.output, arguments.commit,
        )
        print(json.dumps({"output": str(arguments.output), "decision": result["decision"]}))
        return 0
    except (OSError, RuntimeError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
        print(json.dumps({"error": str(error)}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
