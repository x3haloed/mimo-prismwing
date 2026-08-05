#!/usr/bin/env python3
"""Interleave affine precision candidates on one complete real expert."""

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
    from tools.mlx_full_expert_benchmark import dequantize_source, error_metrics, percentile, tensor_names
    from tools.openrouter_reference import atomic_write_new, canonical_json
except ModuleNotFoundError:
    from mlx_full_expert_benchmark import dequantize_source, error_metrics, percentile, tensor_names
    from openrouter_reference import atomic_write_new, canonical_json


BITS = (4, 5, 6, 8)
GROUP_SIZE = 128


def qlinear(inputs: mx.array, arrays: tuple, bits: int) -> mx.array:
    return mx.quantized_matmul(
        inputs,
        *arrays,
        group_size=GROUP_SIZE,
        bits=bits,
        mode="affine",
    )


def expert(inputs: mx.array, projections: dict, bits: int) -> mx.array:
    gate = qlinear(inputs, projections["gate"], bits)
    up = qlinear(inputs, projections["up"], bits)
    return qlinear(inputs=mx.sigmoid(gate) * gate * up, arrays=projections["down"], bits=bits)


def benchmark(shard_zero: Path, shard_one: Path, layer: int, expert_id: int) -> dict:
    names = tensor_names(layer, expert_id)
    source = {
        "gate": dequantize_source(shard_zero, names["gate"], names["gate"] + "_scale_inv"),
        "up": dequantize_source(shard_zero, names["up"], names["up"] + "_scale_inv"),
        "down": dequantize_source(shard_one, names["down"], names["down"] + "_scale_inv"),
    }
    input_values = np.array(
        [
            [np.sin((column + 19 * row) / 17.0) * 0.01 for column in range(4096)]
            for row in range(8)
        ],
        dtype=np.float16,
    )
    torch_input = torch.from_numpy(input_values).float()
    gate = torch_input @ source["gate"].T
    up = torch_input @ source["up"].T
    reference = ((torch.sigmoid(gate) * gate * up) @ source["down"].T).numpy()

    mx.reset_peak_memory()
    candidates = {}
    for bits in BITS:
        projections = {}
        for name, weight in source.items():
            arrays = mx.quantize(
                mx.array(weight.to(torch.float16).numpy()),
                group_size=GROUP_SIZE,
                bits=bits,
                mode="affine",
            )
            mx.eval(*arrays)
            projections[name] = arrays
        candidates[bits] = projections
    del source
    inputs = mx.array(input_values)

    def run(bits: int) -> tuple[float, mx.array]:
        started = time.perf_counter_ns()
        output = expert(inputs, candidates[bits], bits)
        mx.eval(output)
        mx.synchronize()
        return (time.perf_counter_ns() - started) / 1_000_000, output

    for _ in range(10):
        for bits in BITS:
            run(bits)
    measurements = {bits: [] for bits in BITS}
    outputs = {}
    for iteration in range(30):
        order = BITS[iteration % len(BITS) :] + BITS[: iteration % len(BITS)]
        for bits in order:
            elapsed, outputs[bits] = run(bits)
            measurements[bits].append(elapsed)

    records = []
    for bits in BITS:
        values = measurements[bits]
        median_ms = statistics.median(values)
        executable_bytes = sum(
            array.nbytes for arrays in candidates[bits].values() for array in arrays
        )
        records.append(
            {
                "bits": bits,
                "group_size": GROUP_SIZE,
                "mode": "affine",
                "executable_bytes": executable_bytes,
                "wall_ms": values,
                "wall_median_ms": median_ms,
                "wall_p10_ms": percentile(values, 0.1),
                "wall_p90_ms": percentile(values, 0.9),
                "expert_positions_per_second": 8 / (median_ms / 1_000),
                "error_vs_source_fp8": error_metrics(
                    np.array(outputs[bits]).astype(np.float32), reference
                ),
            }
        )
    return {
        "schema_version": 1,
        "semantic": "complete_real_expert_affine_precision_sweep",
        "exactness": "L3_candidates_vs_source_FP8",
        "device": str(mx.default_device()),
        "machine": platform.machine(),
        "mlx_version": "0.31.2",
        "layer": layer,
        "expert": expert_id,
        "batch_size": 8,
        "warmup_runs_per_candidate": 10,
        "measured_runs_per_candidate": 30,
        "run_order": "rotating_interleaved_4_5_6_8_bits",
        "cache_state": "all candidate buffers warm; source load and quantization excluded",
        "concurrency": 1,
        "peak_mlx_memory_bytes_during_setup_and_benchmark": mx.get_peak_memory(),
        "candidates": records,
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
            arguments.shard_zero, arguments.shard_one, arguments.layer, arguments.expert
        )
        atomic_write_new(arguments.output, canonical_json(result))
        print(arguments.output)
        return 0
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"error: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
