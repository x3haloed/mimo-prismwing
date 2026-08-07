#!/usr/bin/env python3
"""Run PW-0130's same-validation INT4 expert-output repair oracle."""

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
    from tools.run_best_rank_real_expert_control import load_capture, sha256_file
    from tools.run_real_activation_affine_int4_audit import (
        CORPUS_SHA256,
        LAYERS,
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
    from run_best_rank_real_expert_control import load_capture, sha256_file
    from run_real_activation_affine_int4_audit import (
        CORPUS_SHA256,
        LAYERS,
        REVISION,
        SOURCE_EXPERT_BYTES,
        VERIFICATION_SHA256,
        _candidate_expert,
        _quantize_expert,
        error_metrics,
        validate_routes,
    )


PW0129_SHA256 = "1deb9dd85f0b598f31bc2d8bc1d41bf52cfabcda43de63a2ae5b3fdfad400306"
INT4_BYTES = 13_369_344
FULL_AFFINE_REPAIR_BYTES_PER_LAYER = 256 * 4096 * 2 * 2
BIAS_REPAIR_BYTES_PER_LAYER = 256 * 4096 * 2


def fit_output_repair(candidate: np.ndarray, source: np.ndarray, mode: str) -> tuple[np.ndarray, np.ndarray]:
    if (
        candidate.shape != source.shape
        or candidate.ndim != 2
        or candidate.shape[0] == 0
        or mode not in {"bias", "affine"}
        or not np.isfinite(candidate).all()
        or not np.isfinite(source).all()
    ):
        raise ValueError("PW-0130 output-repair fit inputs are invalid")
    candidate64 = candidate.astype(np.float64, copy=False)
    source64 = source.astype(np.float64, copy=False)
    if mode == "bias":
        scale = np.ones(candidate.shape[1], dtype=np.float64)
        bias = np.mean(source64 - candidate64, axis=0)
    else:
        candidate_mean = np.mean(candidate64, axis=0)
        source_mean = np.mean(source64, axis=0)
        centered_candidate = candidate64 - candidate_mean
        denominator = np.sum(centered_candidate * centered_candidate, axis=0)
        numerator = np.sum(centered_candidate * (source64 - source_mean), axis=0)
        scale = np.ones(candidate.shape[1], dtype=np.float64)
        nonconstant = denominator > 1e-30
        scale[nonconstant] = numerator[nonconstant] / denominator[nonconstant]
        bias = source_mean - scale * candidate_mean
    scale_f16 = scale.astype(np.float16)
    bias_f16 = bias.astype(np.float16)
    if not np.isfinite(scale_f16).all() or not np.isfinite(bias_f16).all():
        raise ValueError("PW-0130 F16 repair parameters are non-finite")
    return scale_f16, bias_f16


def apply_output_repair(
    candidate: np.ndarray,
    scale: np.ndarray,
    bias: np.ndarray,
) -> np.ndarray:
    if (
        candidate.ndim != 2
        or scale.shape != (candidate.shape[1],)
        or bias.shape != scale.shape
        or scale.dtype != np.float16
        or bias.dtype != np.float16
    ):
        raise ValueError("PW-0130 repair application shape/dtype mismatch")
    result = candidate.astype(np.float32, copy=False) * scale.astype(np.float32) + bias.astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError("PW-0130 repaired output is non-finite")
    return result


def _parameter_digest(parameters: dict[int, tuple[np.ndarray, np.ndarray]]) -> str:
    digest = hashlib.sha256()
    for expert, (scale, bias) in sorted(parameters.items()):
        digest.update(expert.to_bytes(2, "little"))
        digest.update(np.ascontiguousarray(scale.astype("<f2", copy=False)).tobytes())
        digest.update(np.ascontiguousarray(bias.astype("<f2", copy=False)).tobytes())
    return digest.hexdigest()


def _route_rows(
    rows_by_expert: dict[int, dict],
    authority: dict,
    field: str,
    start: int,
    end: int,
) -> np.ndarray:
    result = np.zeros((end - start, 4096), dtype=np.float32)
    selected = authority["selected_experts_by_position"]
    weights = authority["route_weights_by_position"]
    for expert, record in sorted(rows_by_expert.items()):
        values = record[field]
        positions = record["positions"]
        if values.shape != (len(positions), 4096):
            raise ValueError("PW-0130 expert-row routing shape mismatch")
        for local, position in enumerate(positions):
            if start <= position < end:
                slot = selected[position].index(expert)
                result[position - start] += values[local] * np.float32(weights[position][slot])
    return torch.from_numpy(result).to(torch.bfloat16).float().numpy()


def _prior_report(source: dict, layer: int, partition: str) -> dict:
    matches = [
        row[partition]
        for row in source["reports"]
        if row["layer"] == layer and row["bits"] == 4
    ]
    if len(matches) != 1:
        raise ValueError("PW-0130 PW-0129 layer authority mismatch")
    return matches[0]


def _collect_layer(
    checkpoint: ShardedCheckpoint,
    authority: dict,
    moe_input: np.ndarray,
    expert_down: np.ndarray,
    prior: dict,
    layer: int,
    safety: HostSafetyMonitor,
) -> dict[int, dict]:
    prior_hashes = {}
    for partition in ("train", "validation"):
        for report in _prior_report(prior, layer, partition)["expert_reports"]:
            existing = prior_hashes.setdefault(report["expert"], report["packed_sha256"])
            if existing != report["packed_sha256"]:
                raise ValueError("PW-0130 prior packed hash differs by partition")
    rows_by_expert = {}
    offset = 0
    for schedule in authority["expert_schedule"]:
        expert = schedule["expert"]
        local_indices = [
            index for index, position in enumerate(schedule["positions"]) if position < 168
        ]
        if local_indices:
            positions = [schedule["positions"][index] for index in local_indices]
            projections, packed = _quantize_expert(checkpoint, layer, expert, 4)
            if (
                packed["packed_bytes"] != INT4_BYTES
                or packed["packed_sha256"] != prior_hashes.get(expert)
            ):
                raise ValueError("PW-0130 recomputed packed artifact mismatch")
            candidate, _ = _candidate_expert(moe_input[positions], projections, 4)
            source_rows = np.asarray(
                expert_down[[offset + index for index in local_indices]], dtype=np.float32
            ).copy()
            rows_by_expert[expert] = {
                "positions": positions,
                "candidate": candidate,
                "source": source_rows,
            }
            del projections
            gc.collect()
            mx.clear_cache()
            if mx.get_active_memory() != 0:
                raise ValueError("PW-0130 MLX expert buffers did not release")
            safety.release_checkpoint(
                f"layer_{layer}_expert_{expert}_recomputed_and_released",
                ["source expert", "INT4 expert", "expert activations"],
            )
        offset += len(schedule["positions"])
    if set(rows_by_expert) != set(prior_hashes):
        raise ValueError("PW-0130 recomputed expert set mismatch")
    return rows_by_expert


def _baseline_metrics(
    rows_by_expert: dict[int, dict], authority: dict, routed_expected: np.ndarray
) -> dict:
    candidate = _route_rows(rows_by_expert, authority, "candidate", 0, 168)
    return {
        "train": error_metrics(candidate[:112], np.asarray(routed_expected[:112], dtype=np.float32)),
        "validation": error_metrics(candidate[112:168], np.asarray(routed_expected[112:168], dtype=np.float32)),
    }


def _oracle(
    rows_by_expert: dict[int, dict],
    authority: dict,
    routed_expected: np.ndarray,
    mode: str,
) -> dict:
    parameters = {}
    repaired = {}
    touched = 0
    for expert, record in rows_by_expert.items():
        validation_indices = [
            index for index, position in enumerate(record["positions"]) if 112 <= position < 168
        ]
        candidate = record["candidate"]
        result = candidate.copy()
        if validation_indices:
            scale, bias = fit_output_repair(
                candidate[validation_indices], record["source"][validation_indices], mode
            )
            result[validation_indices] = apply_output_repair(
                candidate[validation_indices], scale, bias
            )
            parameters[expert] = (scale, bias)
            touched += 1
        repaired[expert] = {"positions": record["positions"], "repaired": result}
    routed = _route_rows(repaired, authority, "repaired", 112, 168)
    metrics = error_metrics(routed, np.asarray(routed_expected[112:168], dtype=np.float32))
    parameter_bytes = BIAS_REPAIR_BYTES_PER_LAYER if mode == "bias" else FULL_AFFINE_REPAIR_BYTES_PER_LAYER
    return {
        "mode": mode,
        "validation_touched_experts": touched,
        "fitted_parameter_sha256": _parameter_digest(parameters),
        "full_layer_parameter_bytes": parameter_bytes,
        "parameter_to_source_layer_bank_ratio": parameter_bytes / (256 * SOURCE_EXPERT_BYTES),
        "validation_metrics": metrics,
    }


def _gate(layer_reports: list[dict]) -> dict:
    affine = [row["affine"] for row in layer_reports]
    squared_error = sum(row["validation_metrics"]["squared_error"] for row in affine)
    expected_norm = sum(row["validation_metrics"]["expected_squared_norm"] for row in affine)
    aggregate = float(np.sqrt(squared_error / max(expected_norm, 1e-30)))
    monotonic = all(
        row["affine"]["validation_metrics"]["relative_l2"]
        <= row["bias"]["validation_metrics"]["relative_l2"]
        <= row["baseline_validation"]["relative_l2"]
        for row in layer_reports
    )
    return {
        "aggregate_relative_l2": aggregate,
        "aggregate_maximum": 0.01,
        "maximum_layer_relative_l2": max(
            row["validation_metrics"]["relative_l2"] for row in affine
        ),
        "layer_maximum": 0.02,
        "maximum_row_relative_l2": max(
            row["validation_metrics"]["maximum_row_relative_l2"] for row in affine
        ),
        "row_maximum": 0.05,
        "nested_oracles_monotonic": monotonic,
        "maximum_parameter_to_source_layer_bank_ratio": max(
            row["parameter_to_source_layer_bank_ratio"] for row in affine
        ),
        "parameter_ratio_maximum": 0.002,
        "passes": (
            aggregate <= 0.01
            and all(row["validation_metrics"]["relative_l2"] <= 0.02 for row in affine)
            and all(row["validation_metrics"]["maximum_row_relative_l2"] <= 0.05 for row in affine)
            and monotonic
            and all(row["parameter_to_source_layer_bank_ratio"] <= 0.002 for row in affine)
        ),
    }


def run(
    checkpoint_root: Path,
    verification_path: Path,
    corpus_manifest_path: Path,
    pw0129_path: Path,
    output_path: Path,
    commit: str,
) -> dict:
    if output_path.exists():
        raise ValueError(f"refusing to overwrite {output_path}")
    if len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit):
        raise ValueError("implementation commit must be lowercase 40-hex")
    if sha256_file(verification_path) != VERIFICATION_SHA256:
        raise ValueError("PW-0130 checkpoint verification hash mismatch")
    if sha256_file(corpus_manifest_path) != CORPUS_SHA256:
        raise ValueError("PW-0130 corpus hash mismatch")
    if sha256_file(pw0129_path) != PW0129_SHA256:
        raise ValueError("PW-0130 PW-0129 report hash mismatch")
    started = time.perf_counter()
    safety = HostSafetyMonitor()
    corpus = json.loads(corpus_manifest_path.read_text())
    prior = json.loads(pw0129_path.read_text())
    if (
        corpus.get("revision") != REVISION
        or corpus.get("target_layers") != list(LAYERS)
        or prior.get("evidence_class") != "pw0129_real_activation_affine_int4_layer_audit"
        or prior.get("holdout_unsealed")
        or prior.get("decision") != "reject_naive_affine_int4_on_real_validation"
    ):
        raise ValueError("PW-0130 source authority mismatch")
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
        baseline = _baseline_metrics(rows, authority, routed_expected)
        prior_train = _prior_report(prior, layer, "train")["routed_output_metrics"]
        prior_validation = _prior_report(prior, layer, "validation")["routed_output_metrics"]
        if baseline["train"] != prior_train or baseline["validation"] != prior_validation:
            raise ValueError("PW-0130 recomputed PW-0129 metrics mismatch")
        bias = _oracle(rows, authority, routed_expected, "bias")
        affine = _oracle(rows, authority, routed_expected, "affine")
        layer_reports.append(
            {
                "layer": layer,
                "baseline_train": baseline["train"],
                "baseline_validation": baseline["validation"],
                "bias": bias,
                "affine": affine,
            }
        )
        del moe_input, expert_down, routed_expected, rows
        gc.collect()
        safety.release_checkpoint(
            f"layer_{layer}_oracle_buffers_released",
            ["real corpus", "INT4 outputs", "repair parameters"],
        )
    gate = _gate(layer_reports)
    decision = (
        "authorize_train_only_int4_output_affine_calibration"
        if gate["passes"]
        else "reject_int4_diagonal_output_affine_repair"
    )
    safety.release_checkpoint(
        "checkpoint_and_authorities_released",
        ["checkpoint mappings", "PW-0116 corpus", "PW-0129 report"],
    )
    safety.checkpoint("final_service_health")
    report = {
        "schema_version": 1,
        "evidence_class": "pw0130_int4_output_affine_repair_oracle",
        "revision": REVISION,
        "commit": commit,
        "checkpoint_verification_sha256": VERIFICATION_SHA256,
        "corpus_manifest_sha256": CORPUS_SHA256,
        "pw0129_report_sha256": PW0129_SHA256,
        "layer_reports": layer_reports,
        "capacity_gate": gate,
        "holdout_unsealed": False,
        "decision": decision,
        "safety_snapshots": safety.evidence(),
        "complete_wall_ms": (time.perf_counter() - started) * 1000.0,
        "accepted_tokens": 0,
        "A": 0,
        "performance_claim": None,
        "limitations": (
            "same-validation noncausal per-expert diagonal output-repair oracle on one "
            "English pilot prefix; no train-only generalization, holdout, accumulated model, "
            "modality corpus, endpoint, or TPS claim"
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
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--commit", required=True)
    arguments = parser.parse_args()
    try:
        result = run(
            arguments.checkpoint,
            arguments.verification,
            arguments.corpus_manifest,
            arguments.pw0129,
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
