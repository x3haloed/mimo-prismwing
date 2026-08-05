#!/usr/bin/env python3
"""Benchmark pinned MLX affine-INT4 qmatmul on a real MiMo projection."""

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
    from tools.generate_real_fp8_fixture import SCALE, WEIGHT
    from tools.openrouter_reference import atomic_write_new, canonical_json
except ModuleNotFoundError:
    from generate_real_fp8_fixture import SCALE, WEIGHT
    from openrouter_reference import atomic_write_new, canonical_json


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * fraction)]


def benchmark(checkpoint: Path, fixture_path: Path) -> dict:
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    if fixture.get("semantic") != "mlx_affine_int4_group128_gemv_slice":
        raise ValueError("unexpected correctness fixture semantic")
    with safe_open(checkpoint, framework="pt", device="cpu") as tensors:
        source = tensors.get_tensor(WEIGHT).float()
        source_scales = tensors.get_tensor(SCALE).float()
    source = source * source_scales.repeat_interleave(128, 0).repeat_interleave(128, 1)
    weight = mx.array(source.to(torch.float16).numpy())
    del source, source_scales
    mx.reset_peak_memory()
    packed, scales, biases = mx.quantize(
        weight, group_size=128, bits=4, mode="affine"
    )
    mx.eval(packed, scales, biases)
    del weight

    base_input = np.array(fixture["input_f16"], dtype=np.float16)
    expected = np.array(fixture["expected_mlx_f16"], dtype=np.float32)
    batches = []
    for batch_size in (1, 8):
        input_array = mx.array(np.repeat(base_input[None, :], batch_size, axis=0))

        def run():
            started = time.perf_counter_ns()
            output = mx.quantized_matmul(
                input_array,
                packed,
                scales,
                biases,
                group_size=128,
                bits=4,
                mode="affine",
            )
            mx.eval(output)
            mx.synchronize()
            elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000
            return elapsed_ms, output

        for _ in range(10):
            run()
        measurements = []
        output = None
        for _ in range(30):
            elapsed, output = run()
            measurements.append(elapsed)
        assert output is not None
        first_four = np.array(output)[:, :4].astype(np.float32)
        max_error = float(np.max(np.abs(first_four - expected[None, :])))
        if max_error > 5e-5:
            raise ValueError(f"MLX output disagrees with fixture: {max_error}")
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
                "arithmetic_gflop_per_second": (
                    2 * packed.shape[0] * base_input.size * batch_size
                )
                / (median_ms / 1_000)
                / 1_000_000_000,
                "effective_accepted_projection_tps": batch_size / (median_ms / 1_000),
                "first_four_max_abs_error_vs_mlx_fixture": max_error,
            }
        )
    return {
        "schema_version": 1,
        "device": str(mx.default_device()),
        "machine": platform.machine(),
        "mlx_version": "0.31.2",
        "tensor": WEIGHT,
        "rows": packed.shape[0],
        "columns": base_input.size,
        "group_size": 128,
        "bits": 4,
        "mode": "affine",
        "packed_weight_bytes": packed.nbytes,
        "scale_bytes": scales.nbytes,
        "bias_bytes": biases.nbytes,
        "executable_bytes": packed.nbytes + scales.nbytes + biases.nbytes,
        "cache_state": "application buffers warm; source load and quantization excluded",
        "concurrency": 1,
        "peak_mlx_memory_bytes_during_setup_and_benchmark": mx.get_peak_memory(),
        "batches": batches,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("fixture", type=Path)
    parser.add_argument("output", type=Path)
    arguments = parser.parse_args()
    try:
        result = benchmark(arguments.checkpoint, arguments.fixture)
        atomic_write_new(arguments.output, canonical_json(result))
        print(arguments.output)
        return 0
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"error: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
