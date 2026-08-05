#!/usr/bin/env python3
"""Benchmark one complete real MiMo routed expert with MLX affine INT4."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import statistics
import time

import mlx.core as mx
import numpy as np
from safetensors import safe_open
import torch

try:
    from tools.openrouter_reference import atomic_write_new, canonical_json
except ModuleNotFoundError:
    from openrouter_reference import atomic_write_new, canonical_json


GROUP_SIZE = 128
BITS = 4
MODE = "affine"


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * fraction)]


def dequantize_source(path: Path, weight_name: str, scale_name: str) -> torch.Tensor:
    with safe_open(path, framework="pt", device="cpu") as tensors:
        weight = tensors.get_tensor(weight_name).float()
        scales = tensors.get_tensor(scale_name).float()
    if weight.ndim != 2 or scales.ndim != 2:
        raise ValueError(f"{weight_name}: expected two-dimensional weight and scale")
    if weight.shape[0] % scales.shape[0] or weight.shape[1] % scales.shape[1]:
        raise ValueError(f"{weight_name}: scale grid does not divide weight shape")
    return weight * scales.repeat_interleave(
        weight.shape[0] // scales.shape[0], 0
    ).repeat_interleave(weight.shape[1] // scales.shape[1], 1)


def quantize(weight: torch.Tensor) -> tuple[mx.array, mx.array, mx.array]:
    arrays = mx.quantize(
        mx.array(weight.to(torch.float16).numpy()),
        group_size=GROUP_SIZE,
        bits=BITS,
        mode=MODE,
    )
    mx.eval(*arrays)
    return arrays


def quantized_linear(
    inputs: mx.array, arrays: tuple[mx.array, mx.array, mx.array]
) -> mx.array:
    packed, scales, biases = arrays
    return mx.quantized_matmul(
        inputs,
        packed,
        scales,
        biases,
        group_size=GROUP_SIZE,
        bits=BITS,
        mode=MODE,
    )


def full_expert(
    inputs: mx.array,
    gate: tuple[mx.array, mx.array, mx.array],
    up: tuple[mx.array, mx.array, mx.array],
    down: tuple[mx.array, mx.array, mx.array],
) -> mx.array:
    gate_values = quantized_linear(inputs, gate)
    up_values = quantized_linear(inputs, up)
    hidden = mx.sigmoid(gate_values) * gate_values * up_values
    return quantized_linear(hidden, down)


def tensor_names(layer: int, expert: int) -> dict[str, str]:
    prefix = f"model.layers.{layer}.mlp.experts.{expert}"
    return {
        projection: f"{prefix}.{projection}_proj.weight"
        for projection in ("gate", "up", "down")
    }


def error_metrics(actual: np.ndarray, expected: np.ndarray) -> dict[str, float]:
    difference = actual.astype(np.float64) - expected.astype(np.float64)
    actual_flat = actual.astype(np.float64).ravel()
    expected_flat = expected.astype(np.float64).ravel()
    expected_norm = np.linalg.norm(expected_flat)
    denominator = np.linalg.norm(actual_flat) * expected_norm
    return {
        "relative_l2": float(np.linalg.norm(difference) / expected_norm),
        "cosine_similarity": float(np.dot(actual_flat, expected_flat) / denominator),
        "max_abs": float(np.max(np.abs(difference))),
    }


def benchmark(shard_zero: Path, shard_one: Path, layer: int, expert: int) -> dict:
    names = tensor_names(layer, expert)
    source = {
        "gate": dequantize_source(shard_zero, names["gate"], names["gate"] + "_scale_inv"),
        "up": dequantize_source(shard_zero, names["up"], names["up"] + "_scale_inv"),
        "down": dequantize_source(shard_one, names["down"], names["down"] + "_scale_inv"),
    }
    if source["gate"].shape != source["up"].shape:
        raise ValueError("gate and up projection shapes differ")
    if source["down"].shape != (source["gate"].shape[1], source["gate"].shape[0]):
        raise ValueError("down projection is not the transpose-compatible expert shape")

    input_width = source["gate"].shape[1]
    inputs = {
        batch_size: np.array(
            [
                [np.sin((column + 19 * row) / 17.0) * 0.01 for column in range(input_width)]
                for row in range(batch_size)
            ],
            dtype=np.float16,
        )
        for batch_size in (1, 8)
    }
    reference = {}
    for batch_size, values in inputs.items():
        torch_input = torch.from_numpy(values).float()
        gate_values = torch_input @ source["gate"].T
        up_values = torch_input @ source["up"].T
        reference[batch_size] = (
            (torch.sigmoid(gate_values) * gate_values * up_values) @ source["down"].T
        ).numpy()

    mx.reset_peak_memory()
    quantized = {projection: quantize(weight) for projection, weight in source.items()}
    executable_bytes = {
        projection: sum(array.nbytes for array in arrays)
        for projection, arrays in quantized.items()
    }
    del source

    batches = []
    for batch_size, values in inputs.items():
        mx_input = mx.array(values)

        def run() -> tuple[float, mx.array]:
            started = time.perf_counter_ns()
            output = full_expert(
                mx_input, quantized["gate"], quantized["up"], quantized["down"]
            )
            mx.eval(output)
            mx.synchronize()
            return (time.perf_counter_ns() - started) / 1_000_000, output

        for _ in range(10):
            run()
        measurements = []
        output = None
        for _ in range(30):
            elapsed, output = run()
            measurements.append(elapsed)
        assert output is not None
        actual = np.array(output).astype(np.float32)
        median_ms = statistics.median(measurements)
        batches.append(
            {
                "batch_size": batch_size,
                "warmup_runs": 10,
                "measured_runs": 30,
                "wall_ms": measurements,
                "wall_median_ms": median_ms,
                "wall_p10_ms": percentile(measurements, 0.1),
                "wall_p90_ms": percentile(measurements, 0.9),
                "expert_positions_per_second": batch_size / (median_ms / 1_000),
                "error_vs_source_fp8": error_metrics(actual, reference[batch_size]),
                "first_output_values": actual[0, :8].tolist(),
                "first_source_fp8_values": reference[batch_size][0, :8].tolist(),
            }
        )

    return {
        "schema_version": 1,
        "semantic": "complete_routed_expert_gate_up_swiglu_down",
        "exactness": "L3_affine_INT4_candidate_vs_source_FP8",
        "device": str(mx.default_device()),
        "machine": platform.machine(),
        "mlx_version": "0.31.2",
        "layer": layer,
        "expert": expert,
        "source_files": [shard_zero.name, shard_one.name],
        "tensors": names,
        "group_size": GROUP_SIZE,
        "bits": BITS,
        "mode": MODE,
        "executable_bytes_by_projection": executable_bytes,
        "total_executable_bytes": sum(executable_bytes.values()),
        "cache_state": "application buffers warm; source load and installation quantization excluded",
        "concurrency": 1,
        "peak_mlx_memory_bytes_during_setup_and_benchmark": mx.get_peak_memory(),
        "batches": batches,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("shard_zero", type=Path)
    parser.add_argument("shard_one", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--layer", type=int, default=43)
    parser.add_argument("--expert", type=int, default=32)
    arguments = parser.parse_args()
    try:
        result = benchmark(
            arguments.shard_zero,
            arguments.shard_one,
            arguments.layer,
            arguments.expert,
        )
        atomic_write_new(arguments.output, canonical_json(result))
        print(arguments.output)
        return 0
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"error: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
