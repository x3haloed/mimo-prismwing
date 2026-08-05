#!/usr/bin/env python3
"""Benchmark the exact PW-0039 MoE fixture with resident F32 MLX matrices."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
from safetensors import safe_open
import torch

try:
    from tools.openrouter_reference import atomic_write_new, canonical_json
except ModuleNotFoundError:
    from openrouter_reference import atomic_write_new, canonical_json


REVISION = "63651580ca774f8504f676040460aed3e1244ac1"
ROUTER_SHA256 = "12c1579d28b78dd69ec9342eb9d1f378efc5aa3c2f2a28b5ec73578e6a8bbcdd"
BATCH = 8
HIDDEN = 4096
TOP_K = 8
WARMUPS = 10
MEASUREMENTS = 30


def sha256_file(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * fraction)]


def dequantize(values: dict[str, torch.Tensor], name: str) -> np.ndarray:
    weight = values[name].float()
    scales = values[name + "_scale_inv"].float()
    if weight.ndim != 2 or tuple(scales.shape) != (
        weight.shape[0] // 128,
        weight.shape[1] // 128,
    ):
        raise ValueError(f"{name}: source FP8 layout mismatch")
    expanded = weight * scales.repeat_interleave(128, 0).repeat_interleave(128, 1)
    result = expanded.numpy().astype(np.float32, copy=False)
    if not np.isfinite(result).all():
        raise ValueError(f"{name}: non-finite expanded weight")
    return result


def error_metrics(actual: np.ndarray, reference: np.ndarray) -> tuple[float, float]:
    difference = actual.astype(np.float64) - reference.astype(np.float64)
    denominator = np.linalg.norm(reference.astype(np.float64).ravel())
    return float(np.linalg.norm(difference.ravel()) / denominator), float(
        np.max(np.abs(difference))
    )


def benchmark(
    router_path: Path,
    manifest_path: Path,
    artifact_root: Path,
    input_path: Path,
    reference_path: Path,
) -> dict:
    mlx_version = importlib.metadata.version("mlx")
    if str(mx.default_device()) != "Device(gpu, 0)" or mlx_version != "0.31.2":
        raise ValueError("unknown MLX device or version")
    install_start = time.perf_counter_ns()
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    if (
        manifest.get("schema_version") != 1
        or manifest.get("revision") != REVISION
        or manifest.get("layer") != 43
        or manifest.get("batch_size") != BATCH
        or manifest.get("top_k") != TOP_K
        or len(manifest.get("experts", [])) != 9
    ):
        raise ValueError("unknown PW-0039 manifest identity")
    values: dict[str, torch.Tensor] = {}
    source_bytes = 0
    for name, expected in sorted(manifest["artifact_sha256"].items()):
        path = artifact_root / name
        if sha256_file(path) != expected:
            raise ValueError(f"artifact hash mismatch: {name}")
        with safe_open(path, framework="pt", device="cpu") as tensors:
            for tensor_name in tensors.keys():
                if tensor_name in values:
                    raise ValueError(f"duplicate selected tensor: {tensor_name}")
                tensor = tensors.get_tensor(tensor_name)
                values[tensor_name] = tensor
                source_bytes += tensor.numel() * tensor.element_size()
    if sha256_file(router_path) != ROUTER_SHA256:
        raise ValueError("router hash mismatch")
    with safe_open(router_path, framework="pt", device="cpu") as tensors:
        router_weight = tensors.get_tensor("model.layers.43.mlp.gate.weight").float()
        correction = tensors.get_tensor(
            "model.layers.43.mlp.gate.e_score_correction_bias"
        ).float()
    input_bytes = input_path.read_bytes()
    reference_bytes = reference_path.read_bytes()
    if (
        hashlib.sha256(input_bytes).hexdigest() != manifest["input_sha256"]
        or hashlib.sha256(reference_bytes).hexdigest() != manifest["reference_sha256"]
    ):
        raise ValueError("input or reference hash mismatch")
    inputs_np = np.frombuffer(input_bytes, dtype="<f4").reshape(BATCH, HIDDEN).copy()
    reference = np.frombuffer(reference_bytes, dtype="<f4").reshape(BATCH, HIDDEN)
    expanded: dict[int, dict[str, mx.array]] = {}
    expanded_bytes = 0
    for entry in manifest["experts"]:
        expert = int(entry["expert"])
        prefix = entry["prefix"]
        projections = {}
        for kind in ("gate", "up", "down"):
            name = f"{prefix}.{kind}_proj.weight"
            array = mx.array(dequantize(values, name))
            projections[kind] = array
            expanded_bytes += array.nbytes
        expanded[expert] = projections
    del values
    mx_router = mx.array(router_weight.numpy())
    mx_correction = mx.array(correction.numpy())
    mx_inputs = mx.array(inputs_np)
    mx.eval(mx_router, mx_correction, mx_inputs, *[
        value for projections in expanded.values() for value in projections.values()
    ])
    mx.synchronize()
    install_ms = (time.perf_counter_ns() - install_start) / 1_000_000
    expected_sets = [set(row) for row in manifest["selected_experts_by_position"]]

    def run() -> tuple[float, mx.array, list[list[int]], float]:
        started = time.perf_counter_ns()
        scores = mx.sigmoid(mx_inputs @ mx_router.T)
        choices = scores + mx_correction
        selected = mx.argpartition(choices, kth=-TOP_K, axis=-1)[:, -TOP_K:]
        mx.eval(scores, selected)
        mx.synchronize()
        scores_np = np.array(scores).astype(np.float32)
        selected_np = np.array(selected).astype(np.int64)
        selected_sets = [set(row.tolist()) for row in selected_np]
        if selected_sets != expected_sets:
            raise ValueError("MLX selected expert set mismatch")
        schedule: dict[int, list[tuple[int, float]]] = {}
        maximum_route_error = 0.0
        for position in range(BATCH):
            ids = selected_np[position]
            weights = scores_np[position, ids]
            weights = weights / (weights.sum(dtype=np.float32) + np.float32(1e-20))
            reference_ids = manifest["selected_experts_by_position"][position]
            reference_weights = manifest["route_weights_by_position"][position]
            for expert, weight in zip(ids.tolist(), weights.tolist()):
                slot = reference_ids.index(expert)
                maximum_route_error = max(
                    maximum_route_error, abs(weight - reference_weights[slot])
                )
                schedule.setdefault(expert, []).append((position, weight))
        if maximum_route_error > 2e-6 or set(schedule) != set(expanded):
            raise ValueError("MLX dynamic route parity failed")
        result = mx.zeros((BATCH, HIDDEN), dtype=mx.float32)
        for expert, placements in sorted(schedule.items()):
            positions = [position for position, _ in placements]
            route_weights = mx.array([weight for _, weight in placements])
            weights = expanded[expert]
            expert_inputs = mx_inputs[positions]
            gate = expert_inputs @ weights["gate"].T
            up = expert_inputs @ weights["up"].T
            output = (mx.sigmoid(gate) * gate * up) @ weights["down"].T
            placement = np.zeros((BATCH, len(positions)), dtype=np.float32)
            for column, position in enumerate(positions):
                placement[position, column] = 1.0
            result = result + mx.array(placement) @ (output * route_weights[:, None])
        mx.eval(result)
        mx.synchronize()
        return (
            (time.perf_counter_ns() - started) / 1_000_000,
            result,
            selected_np.tolist(),
            maximum_route_error,
        )

    for _ in range(WARMUPS):
        run()
    measurements = []
    output = None
    selected = None
    route_error = None
    for _ in range(MEASUREMENTS):
        elapsed, output, selected, route_error = run()
        measurements.append(elapsed)
    assert output is not None and selected is not None and route_error is not None
    actual = np.array(output).astype(np.float32)
    relative_l2, maximum_absolute_error = error_metrics(actual, reference)
    if relative_l2 > 4e-5 or maximum_absolute_error > 3e-8:
        raise ValueError(
            f"complete parity failed: rel={relative_l2} max={maximum_absolute_error}"
        )
    return {
        "schema_version": 1,
        "semantic": "mimo_layer43_hot_exact_f32_mlx_dynamic_moe",
        "revision": REVISION,
        "mlx_version": mlx_version,
        "device": str(mx.default_device()),
        "machine": platform.machine(),
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "router_sha256": ROUTER_SHA256,
        "input_sha256": hashlib.sha256(input_bytes).hexdigest(),
        "reference_sha256": hashlib.sha256(reference_bytes).hexdigest(),
        "selected_experts_by_position": selected,
        "maximum_route_weight_absolute_error": route_error,
        "output_sha256": hashlib.sha256(actual.astype("<f4").tobytes()).hexdigest(),
        "output_first8": actual[0, :8].tolist(),
        "relative_l2": relative_l2,
        "maximum_absolute_error": maximum_absolute_error,
        "install_ms": install_ms,
        "warmups": WARMUPS,
        "measurements": MEASUREMENTS,
        "wall_ms": measurements,
        "wall_p10_ms": percentile(measurements, 0.1),
        "wall_median_ms": percentile(measurements, 0.5),
        "wall_p90_ms": percentile(measurements, 0.9),
        "source_tensor_bytes": source_bytes,
        "expanded_expert_bytes": expanded_bytes,
        "executable_bytes": expanded_bytes + mx_router.nbytes + mx_correction.nbytes,
        "peak_memory_bytes": mx.get_peak_memory(),
        "batch_size": BATCH,
        "concurrency": 1,
        "accepted_tokens": 0,
        "A": 8,
        "U": 1.125,
        "cache_state": "expanded exact F32 expert union and router resident; source validation and installation excluded from warm timings",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--router", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    result = benchmark(
        arguments.router,
        arguments.manifest,
        arguments.artifact_root,
        arguments.input,
        arguments.reference,
    )
    payload = canonical_json(result)
    atomic_write_new(arguments.output, payload)
    print(payload.decode("utf-8"))


if __name__ == "__main__":
    main()
