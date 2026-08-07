#!/usr/bin/env python3
"""Run PW-0129's streamed real-activation affine-INT4 layer audit."""

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
    from tools.run_best_rank_real_expert_control import (
        dequant_weight,
        load_capture,
        parity,
        sha256_file,
        source_linear,
    )
except ModuleNotFoundError:
    from generate_real_layer1_expert_oracle import ShardedCheckpoint
    from host_safety import HostSafetyMonitor
    from openrouter_reference import atomic_write_new, canonical_json
    from run_best_rank_real_expert_control import (
        dequant_weight,
        load_capture,
        parity,
        sha256_file,
        source_linear,
    )


REVISION = "63651580ca774f8504f676040460aed3e1244ac1"
VERIFICATION_SHA256 = "9ddc8a99755f04ae2ea3c2484f6dd022d3f3a681b5a72c915ee4de833dbb0d03"
CORPUS_SHA256 = "b9df976876d63c1ffbbe0c70507aea8b939a749ce5b1db27cbca0b5d82cf802e"
LAYERS = (4, 24, 46)
BITS = (4, 8)
GROUP_SIZE = 128
SOURCE_EXPERT_BYTES = 25_171_968
VALIDATION_AGGREGATE_MAX = 0.01
VALIDATION_LAYER_MAX = 0.02
VALIDATION_ROW_MAX = 0.05
PACKED_RATIO_MAX = 0.60


def error_metrics(actual: np.ndarray, expected: np.ndarray) -> dict:
    if actual.shape != expected.shape or actual.ndim != 2 or actual.shape[0] == 0:
        raise ValueError("error metrics require equal non-empty matrices")
    if not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise ValueError("error metrics reject non-finite values")
    actual64 = actual.astype(np.float64, copy=False)
    expected64 = expected.astype(np.float64, copy=False)
    difference = actual64 - expected64
    expected_norm = np.linalg.norm(expected64)
    row_denominator = np.linalg.norm(expected64, axis=1)
    row_error = np.linalg.norm(difference, axis=1) / np.maximum(row_denominator, 1e-30)
    denominator = max(float(expected_norm), 1e-30)
    return {
        "relative_l2": float(np.linalg.norm(difference) / denominator),
        "maximum_absolute_error": float(np.max(np.abs(difference))),
        "maximum_row_relative_l2": float(np.max(row_error)),
        "mean_row_relative_l2": float(np.mean(row_error)),
        "cosine_similarity": float(
            np.vdot(actual64.ravel(), expected64.ravel())
            / max(float(np.linalg.norm(actual64) * expected_norm), 1e-30)
        ),
        "squared_error": float(np.vdot(difference.ravel(), difference.ravel())),
        "expected_squared_norm": float(np.vdot(expected64.ravel(), expected64.ravel())),
        "rows": int(actual.shape[0]),
    }


def validate_routes(authority: dict) -> None:
    selected = authority.get("selected_experts_by_position")
    weights = authority.get("route_weights_by_position")
    schedule = authority.get("expert_schedule")
    if (
        not isinstance(selected, list)
        or not isinstance(weights, list)
        or len(selected) != 224
        or len(weights) != 224
        or any(len(row) != 8 or len(set(row)) != 8 for row in selected)
        or any(len(row) != 8 or not np.isfinite(row).all() for row in map(np.asarray, weights))
        or not isinstance(schedule, list)
        or sum(len(row.get("positions", [])) for row in schedule) != 1792
    ):
        raise ValueError("PW-0129 route authority mismatch")
    placements = []
    for row in schedule:
        expert = row.get("expert")
        positions = row.get("positions")
        if (
            not isinstance(expert, int)
            or not 0 <= expert < 256
            or not isinstance(positions, list)
            or any(not isinstance(position, int) or not 0 <= position < 224 for position in positions)
        ):
            raise ValueError("PW-0129 expert schedule mismatch")
        placements.extend((position, expert) for position in positions)
    expected = sorted(
        (position, expert)
        for position, experts in enumerate(selected)
        for expert in experts
    )
    if sorted(placements) != expected:
        raise ValueError("PW-0129 schedule is not a route bijection")


def reconstruct_routed(
    expert_down: np.ndarray,
    authority: dict,
    start: int,
    end: int,
) -> np.ndarray:
    if not 0 <= start < end <= 224 or expert_down.shape != (1792, 4096):
        raise ValueError("PW-0129 reconstruction scope mismatch")
    validate_routes(authority)
    routed = torch.zeros((end - start, 4096), dtype=torch.float32)
    offset = 0
    selected = authority["selected_experts_by_position"]
    weights = authority["route_weights_by_position"]
    for schedule in authority["expert_schedule"]:
        expert = schedule["expert"]
        positions = schedule["positions"]
        rows = torch.from_numpy(np.asarray(expert_down[offset : offset + len(positions)]).copy())
        for local, position in enumerate(positions):
            if start <= position < end:
                slot = selected[position].index(expert)
                routed[position - start] += rows[local] * float(weights[position][slot])
        offset += len(positions)
    return routed.to(torch.bfloat16).float().numpy()


def quantized_projection(
    values: mx.array,
    arrays: tuple[mx.array, mx.array, mx.array],
    bits: int,
    group_size: int = GROUP_SIZE,
) -> mx.array:
    if bits not in BITS or group_size <= 0:
        raise ValueError("unsupported affine quantized projection")
    return mx.quantized_matmul(
        values,
        *arrays,
        group_size=group_size,
        bits=bits,
        mode="affine",
    )


def _array_digest(digest: "hashlib._Hash", name: str, array: mx.array) -> None:
    values = np.asarray(array)
    digest.update(name.encode("utf-8"))
    digest.update(str(values.dtype).encode("ascii"))
    digest.update(json.dumps(list(values.shape)).encode("ascii"))
    digest.update(np.ascontiguousarray(values).tobytes())


def _quantize_expert(
    checkpoint: ShardedCheckpoint,
    layer: int,
    expert: int,
    bits: int,
) -> tuple[dict[str, tuple[mx.array, mx.array, mx.array]], dict]:
    if bits not in BITS:
        raise ValueError("PW-0129 supports only frozen INT4 and INT8 modes")
    arrays = {}
    digest = hashlib.sha256()
    packed_bytes = 0
    started = time.perf_counter()
    for projection in ("gate", "up", "down"):
        name = f"model.layers.{layer}.mlp.experts.{expert}.{projection}_proj.weight"
        weight = dequant_weight(checkpoint, name)
        if weight.shape not in ((2048, 4096), (4096, 2048)):
            raise ValueError(f"{name}: production expert shape mismatch")
        source = mx.array(np.asarray(weight, dtype=np.float16))
        quantized = mx.quantize(
            source,
            group_size=GROUP_SIZE,
            bits=bits,
            mode="affine",
        )
        mx.eval(*quantized)
        for index, array in enumerate(quantized):
            _array_digest(digest, f"{projection}:{index}", array)
            packed_bytes += int(array.nbytes)
        arrays[projection] = quantized
        del source, weight
        gc.collect()
    return arrays, {
        "packed_sha256": digest.hexdigest(),
        "packed_bytes": packed_bytes,
        "setup_wall_ms": (time.perf_counter() - started) * 1000.0,
    }


def _candidate_expert(
    inputs: np.ndarray,
    projections: dict[str, tuple[mx.array, mx.array, mx.array]],
    bits: int,
) -> tuple[np.ndarray, float]:
    values = mx.array(np.asarray(inputs, dtype=np.float16))
    started = time.perf_counter()
    gate = quantized_projection(values, projections["gate"], bits)
    up = quantized_projection(values, projections["up"], bits)
    hidden = mx.sigmoid(gate) * gate * up
    output = quantized_projection(hidden, projections["down"], bits)
    mx.eval(output)
    mx.synchronize()
    wall_ms = (time.perf_counter() - started) * 1000.0
    result = np.asarray(output).astype(np.float32, copy=True)
    del values, gate, up, hidden, output
    return result, wall_ms


def _source_replay(
    checkpoint: ShardedCheckpoint,
    layer: int,
    authority: dict,
    moe_input: np.ndarray,
    expert_down: np.ndarray,
    end: int,
) -> dict:
    eligible = [
        row for row in authority["expert_schedule"] if sum(position < end for position in row["positions"]) >= 2
    ]
    if not eligible:
        raise ValueError("PW-0129 has no multi-placement source replay expert")
    schedule = max(eligible, key=lambda row: (sum(position < end for position in row["positions"]), -row["expert"]))
    expert = schedule["expert"]
    local_indices = [index for index, position in enumerate(schedule["positions"]) if position < end]
    positions = [schedule["positions"][index] for index in local_indices]
    schedule_index = authority["expert_schedule"].index(schedule)
    offset = sum(
        len(row["positions"])
        for row in authority["expert_schedule"][:schedule_index]
    )
    expected = torch.from_numpy(
        np.asarray(expert_down[[offset + index for index in local_indices]]).copy()
    ).to(torch.bfloat16)
    values = torch.from_numpy(np.asarray(moe_input[positions]).copy()).to(torch.bfloat16)
    prefix = f"model.layers.{layer}.mlp.experts.{expert}"
    gate = source_linear(dequant_weight(checkpoint, f"{prefix}.gate_proj.weight"), values)
    up = source_linear(dequant_weight(checkpoint, f"{prefix}.up_proj.weight"), values)
    hidden = (torch.nn.functional.silu(gate) * up).to(torch.bfloat16)
    output = source_linear(dequant_weight(checkpoint, f"{prefix}.down_proj.weight"), hidden)
    result = parity(output, expected)
    if result["relative_l2"] > 1e-3 or result["maximum_absolute_error"] > 0.02:
        raise ValueError(f"PW-0129 layer {layer} source replay failed")
    return {"expert": expert, "positions": positions, "metrics": result}


def _evaluate_layer_bit(
    checkpoint: ShardedCheckpoint,
    layer: int,
    authority: dict,
    moe_input: np.ndarray,
    expert_down: np.ndarray,
    routed_expected: np.ndarray,
    bits: int,
    start: int,
    end: int,
    safety: HostSafetyMonitor,
) -> dict:
    candidate = np.zeros((end - start, 4096), dtype=np.float32)
    selected = authority["selected_experts_by_position"]
    route_weights = authority["route_weights_by_position"]
    expert_reports = []
    offset = 0
    mx.reset_peak_memory()
    for schedule in authority["expert_schedule"]:
        expert = schedule["expert"]
        full_positions = schedule["positions"]
        local_indices = [
            index for index, position in enumerate(full_positions) if start <= position < end
        ]
        if local_indices:
            positions = [full_positions[index] for index in local_indices]
            projections, packed = _quantize_expert(checkpoint, layer, expert, bits)
            output, execution_wall_ms = _candidate_expert(moe_input[positions], projections, bits)
            expected_rows = np.asarray(
                expert_down[[offset + index for index in local_indices]], dtype=np.float32
            )
            for local, position in enumerate(positions):
                slot = selected[position].index(expert)
                candidate[position - start] += output[local] * np.float32(
                    route_weights[position][slot]
                )
            expert_reports.append(
                {
                    "expert": expert,
                    "positions": len(positions),
                    **packed,
                    "execution_wall_ms": execution_wall_ms,
                    "expert_output_metrics": error_metrics(output, expected_rows),
                }
            )
            del projections, output, expected_rows
            gc.collect()
            mx.clear_cache()
            if mx.get_active_memory() != 0:
                raise ValueError(f"PW-0129 layer {layer} expert {expert} MLX buffers did not release")
            safety.release_checkpoint(
                f"layer_{layer}_int{bits}_expert_{expert}_released",
                ["dequantized source expert", "MLX affine expert", "expert activations"],
            )
        offset += len(full_positions)
    candidate = (
        torch.from_numpy(candidate).to(torch.bfloat16).float().numpy()
    )
    expected = np.asarray(routed_expected[start:end], dtype=np.float32)
    metrics = error_metrics(candidate, expected)
    packed_sizes = {row["packed_bytes"] for row in expert_reports}
    if len(packed_sizes) != 1:
        raise ValueError(f"PW-0129 INT{bits} expert packed sizes differ")
    packed_bytes = packed_sizes.pop()
    return {
        "bits": bits,
        "start_position": start,
        "end_position_exclusive": end,
        "positions": end - start,
        "unique_experts": len(expert_reports),
        "source_bytes_per_expert": SOURCE_EXPERT_BYTES,
        "packed_bytes_per_expert": packed_bytes,
        "packed_to_source_ratio": packed_bytes / SOURCE_EXPERT_BYTES,
        "routed_output_metrics": metrics,
        "expert_reports": expert_reports,
        "setup_wall_ms": sum(row["setup_wall_ms"] for row in expert_reports),
        "execution_wall_ms": sum(row["execution_wall_ms"] for row in expert_reports),
        "peak_mlx_memory_bytes": int(mx.get_peak_memory()),
    }


def _validation_gate(layer_reports: list[dict]) -> dict:
    int4 = [row for row in layer_reports if row["bits"] == 4]
    if len(int4) != len(LAYERS):
        raise ValueError("PW-0129 validation gate lacks all INT4 layers")
    squared_error = sum(row["routed_output_metrics"]["squared_error"] for row in int4)
    expected_norm = sum(row["routed_output_metrics"]["expected_squared_norm"] for row in int4)
    aggregate = float(np.sqrt(squared_error / max(expected_norm, 1e-30)))
    return {
        "aggregate_relative_l2": aggregate,
        "aggregate_maximum": VALIDATION_AGGREGATE_MAX,
        "maximum_layer_relative_l2": max(
            row["routed_output_metrics"]["relative_l2"] for row in int4
        ),
        "layer_maximum": VALIDATION_LAYER_MAX,
        "maximum_row_relative_l2": max(
            row["routed_output_metrics"]["maximum_row_relative_l2"] for row in int4
        ),
        "row_maximum": VALIDATION_ROW_MAX,
        "maximum_packed_to_source_ratio": max(row["packed_to_source_ratio"] for row in int4),
        "packed_ratio_maximum": PACKED_RATIO_MAX,
        "passes": (
            aggregate <= VALIDATION_AGGREGATE_MAX
            and all(row["routed_output_metrics"]["relative_l2"] <= VALIDATION_LAYER_MAX for row in int4)
            and all(row["routed_output_metrics"]["maximum_row_relative_l2"] <= VALIDATION_ROW_MAX for row in int4)
            and all(row["packed_to_source_ratio"] <= PACKED_RATIO_MAX for row in int4)
        ),
    }


def run(
    checkpoint_root: Path,
    verification_path: Path,
    corpus_manifest_path: Path,
    output_path: Path,
    commit: str,
) -> dict:
    if output_path.exists():
        raise ValueError(f"refusing to overwrite {output_path}")
    if len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit):
        raise ValueError("implementation commit must be lowercase 40-hex")
    if sha256_file(verification_path) != VERIFICATION_SHA256:
        raise ValueError("PW-0129 checkpoint verification hash mismatch")
    if sha256_file(corpus_manifest_path) != CORPUS_SHA256:
        raise ValueError("PW-0129 corpus manifest hash mismatch")
    started = time.perf_counter()
    safety = HostSafetyMonitor()
    corpus = json.loads(corpus_manifest_path.read_text())
    if corpus.get("revision") != REVISION or corpus.get("target_layers") != list(LAYERS):
        raise ValueError("PW-0129 corpus identity mismatch")
    checkpoint = ShardedCheckpoint(checkpoint_root, verification_path)
    root = corpus_manifest_path.parent
    reports = []
    source_replays = []
    authority_reports = []
    for layer in LAYERS:
        authority = next(row for row in corpus["layers"] if row["layer"] == layer)
        validate_routes(authority)
        moe_input = load_capture(root, authority["captures"]["moe_input"])
        expert_down = load_capture(root, authority["captures"]["expert_down"])
        routed_expected = load_capture(root, authority["captures"]["routed_output"])
        reconstructed = reconstruct_routed(expert_down, authority, 0, 168)
        expected_prefix = np.asarray(routed_expected[:168], dtype=np.float32)
        if not np.array_equal(reconstructed, expected_prefix):
            raise ValueError(f"PW-0129 layer {layer} source prefix reconstruction mismatch")
        source_replays.append(
            {"layer": layer, **_source_replay(checkpoint, layer, authority, moe_input, expert_down, 168)}
        )
        safety.checkpoint(f"layer_{layer}_source_authority_passed")
        for bits in BITS:
            train = _evaluate_layer_bit(
                checkpoint, layer, authority, moe_input, expert_down, routed_expected,
                bits, 0, 112, safety,
            )
            validation = _evaluate_layer_bit(
                checkpoint, layer, authority, moe_input, expert_down, routed_expected,
                bits, 112, 168, safety,
            )
            reports.append({"layer": layer, "bits": bits, "train": train, "validation": validation})
        authority_reports.append(
            {
                "layer": layer,
                "source_prefix_reconstruction_exact": True,
                "full_corpus_reconstruction_sha256": authority["routed_reconstruction_sha256"],
                "moe_input_sha256": authority["captures"]["moe_input"]["sha256"],
                "expert_down_sha256": authority["captures"]["expert_down"]["sha256"],
                "routed_output_sha256": authority["captures"]["routed_output"]["sha256"],
            }
        )
        del moe_input, expert_down, routed_expected, reconstructed, expected_prefix
        gc.collect()
        safety.release_checkpoint(
            f"layer_{layer}_corpus_buffers_released",
            ["moe input", "expert outputs", "routed output"],
        )

    validation_rows = [
        {"layer": row["layer"], **row["validation"]}
        for row in reports
    ]
    validation_gate = _validation_gate(validation_rows)
    holdout_unsealed = validation_gate["passes"]
    holdout_reports = []
    holdout_gate = None
    if holdout_unsealed:
        for layer in LAYERS:
            authority = next(row for row in corpus["layers"] if row["layer"] == layer)
            moe_input = load_capture(root, authority["captures"]["moe_input"])
            expert_down = load_capture(root, authority["captures"]["expert_down"])
            routed_expected = load_capture(root, authority["captures"]["routed_output"])
            reconstructed = reconstruct_routed(expert_down, authority, 168, 224)
            if not np.array_equal(reconstructed, np.asarray(routed_expected[168:224], dtype=np.float32)):
                raise ValueError(f"PW-0129 layer {layer} source holdout reconstruction mismatch")
            result = _evaluate_layer_bit(
                checkpoint, layer, authority, moe_input, expert_down, routed_expected,
                4, 168, 224, safety,
            )
            holdout_reports.append({"layer": layer, **result})
            del moe_input, expert_down, routed_expected, reconstructed
            gc.collect()
            safety.release_checkpoint(
                f"layer_{layer}_holdout_buffers_released",
                ["holdout moe input", "holdout expert outputs", "holdout routed output"],
            )
        holdout_gate = _validation_gate(holdout_reports)
    candidate_passed = validation_gate["passes"] and bool(
        holdout_gate is not None and holdout_gate["passes"]
    )
    decision = (
        "authorize_accumulated_affine_int4_route_logit_probe"
        if candidate_passed
        else (
            "reject_naive_affine_int4_on_real_holdout"
            if holdout_unsealed
            else "reject_naive_affine_int4_on_real_validation"
        )
    )
    safety.release_checkpoint(
        "checkpoint_and_corpus_released",
        ["checkpoint mappings", "PW-0116 capture mappings"],
    )
    safety.checkpoint("final_service_health")
    report = {
        "schema_version": 1,
        "evidence_class": "pw0129_real_activation_affine_int4_layer_audit",
        "revision": REVISION,
        "commit": commit,
        "checkpoint_verification_sha256": VERIFICATION_SHA256,
        "corpus_manifest_sha256": CORPUS_SHA256,
        "layers": list(LAYERS),
        "bits": list(BITS),
        "group_size": GROUP_SIZE,
        "authority": authority_reports,
        "source_replays": source_replays,
        "reports": reports,
        "validation_gate": validation_gate,
        "holdout_unsealed": holdout_unsealed,
        "holdout_reports": holdout_reports,
        "holdout_gate": holdout_gate,
        "candidate_passed": candidate_passed,
        "decision": decision,
        "safety_snapshots": safety.evidence(),
        "complete_wall_ms": (time.perf_counter() - started) * 1000.0,
        "accepted_tokens": 0,
        "A": 0,
        "performance_claim": None,
        "limitations": (
            "layer-local source routes on one correlated English pilot prefix; fixed MLX "
            "affine INT4/INT8 only; no accumulated model, modality corpus, endpoint, accepted "
            "tokens, recovery training, calibration, or performance claim"
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
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--commit", required=True)
    arguments = parser.parse_args()
    try:
        result = run(
            arguments.checkpoint,
            arguments.verification,
            arguments.corpus_manifest,
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
