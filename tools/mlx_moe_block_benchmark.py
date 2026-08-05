#!/usr/bin/env python3
"""Benchmark one real routed MiMo MoE block with heterogeneous experts."""

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
    from tools.mlx_full_expert_benchmark import (
        error_metrics,
        percentile,
        tensor_names,
    )
    from tools.openrouter_reference import atomic_write_new, canonical_json
except ModuleNotFoundError:
    from mlx_full_expert_benchmark import error_metrics, percentile, tensor_names
    from openrouter_reference import atomic_write_new, canonical_json


TOP_K = 8
GROUP_SIZE = 128


def quantize_candidate(weight: torch.Tensor, bits: int) -> tuple:
    arrays = mx.quantize(
        mx.array(weight.to(torch.float16).numpy()),
        group_size=GROUP_SIZE,
        bits=bits,
        mode="affine",
    )
    mx.eval(*arrays)
    return arrays


def candidate_linear(inputs: mx.array, arrays: tuple, bits: int) -> mx.array:
    return mx.quantized_matmul(
        inputs,
        *arrays,
        group_size=GROUP_SIZE,
        bits=bits,
        mode="affine",
    )


def candidate_expert(inputs: mx.array, projections: dict, bits: int) -> mx.array:
    gate = candidate_linear(inputs, projections["gate"], bits)
    up = candidate_linear(inputs, projections["up"], bits)
    return candidate_linear(mx.sigmoid(gate) * gate * up, projections["down"], bits)


def load_all(paths: list[Path]) -> dict[str, torch.Tensor]:
    values = {}
    for path in paths:
        with safe_open(path, framework="pt", device="cpu") as tensors:
            for name in tensors.keys():
                if name in values:
                    raise ValueError(f"duplicate tensor across expert slices: {name}")
                values[name] = tensors.get_tensor(name)
    return values


def dequantize(values: dict[str, torch.Tensor], name: str) -> torch.Tensor:
    weight = values[name].float()
    scales = values[name + "_scale_inv"].float()
    if weight.ndim != 2 or scales.ndim != 2:
        raise ValueError(f"{name}: expected two-dimensional weight and scales")
    return weight * scales.repeat_interleave(
        weight.shape[0] // scales.shape[0], 0
    ).repeat_interleave(weight.shape[1] // scales.shape[1], 1)


def source_routes(
    inputs: torch.Tensor, router_weight: torch.Tensor, correction_bias: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    scores = torch.sigmoid(inputs.float() @ router_weight.float().T)
    indices = torch.topk(scores + correction_bias.float(), TOP_K, dim=-1, sorted=False).indices
    weights = scores.gather(1, indices)
    return indices, weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)


def route_schedule(indices: np.ndarray) -> dict[int, tuple[list[int], list[int]]]:
    schedule: dict[int, tuple[list[int], list[int]]] = {}
    for token in range(indices.shape[0]):
        for slot in range(indices.shape[1]):
            expert = int(indices[token, slot])
            positions, slots = schedule.setdefault(expert, ([], []))
            positions.append(token)
            slots.append(slot)
    return schedule


def verify_fixture(result: dict, fixture: dict) -> None:
    if (
        fixture.get("schema_version") != 1
        or fixture.get("semantic") != result["semantic"]
        or fixture.get("layer") != result["layer"]
        or ("bits" in fixture and fixture["bits"] != result["bits"])
        or fixture.get("selected_experts_by_position")
        != result["selected_experts_by_position"]
    ):
        raise ValueError("MoE-block fixture identity or routing mismatch")
    tolerance = fixture["first_output_abs_tolerance"]
    for name in ("first_output_values", "first_source_fp8_values"):
        if max(
            abs(left - right)
            for left, right in zip(result[name], fixture[name])
        ) > tolerance:
            raise ValueError(f"MoE-block fixture output mismatch: {name}")


def benchmark(
    router_path: Path,
    expert_paths: list[Path],
    layer: int,
    bits: int = 4,
    fixture: dict | None = None,
) -> dict:
    with safe_open(router_path, framework="pt", device="cpu") as tensors:
        router_weight = tensors.get_tensor(f"model.layers.{layer}.mlp.gate.weight").float()
        correction_bias = tensors.get_tensor(
            f"model.layers.{layer}.mlp.gate.e_score_correction_bias"
        ).float()
    batch_size = 8
    input_values = np.array(
        [
            [np.sin((column + 19 * row) / 17.0) * 0.01 for column in range(4096)]
            for row in range(batch_size)
        ],
        dtype=np.float16,
    )
    torch_inputs = torch.from_numpy(input_values).float()
    indices, source_route_weights = source_routes(
        torch_inputs, router_weight, correction_bias
    )
    indices_np = indices.numpy()
    schedule = route_schedule(indices_np)

    raw_experts = load_all(expert_paths)
    source_experts = {}
    for expert in sorted(schedule):
        names = tensor_names(layer, expert)
        source_experts[expert] = {
            projection: dequantize(raw_experts, name)
            for projection, name in names.items()
        }
    del raw_experts

    reference = torch.zeros_like(torch_inputs)
    for expert, (positions, slots) in schedule.items():
        values = torch_inputs[positions]
        weights = source_experts[expert]
        gate = values @ weights["gate"].T
        up = values @ weights["up"].T
        output = (torch.sigmoid(gate) * gate * up) @ weights["down"].T
        route_weight = source_route_weights[positions, slots].unsqueeze(-1)
        reference.index_add_(0, torch.tensor(positions), output * route_weight)
    reference_np = reference.numpy()

    mx.reset_peak_memory()
    quantized = {
        expert: {
            projection: quantize_candidate(weight, bits)
            for projection, weight in projections.items()
        }
        for expert, projections in source_experts.items()
    }
    del source_experts, reference
    mx_router_weight = mx.array(router_weight.numpy())
    mx_correction_bias = mx.array(correction_bias.numpy())
    mx_inputs = mx.array(input_values)
    static_indices = mx.array(indices_np)
    placement = {}
    for expert, (positions, slots) in schedule.items():
        matrix = np.zeros((batch_size, len(positions)), dtype=np.float32)
        for column, token in enumerate(positions):
            matrix[token, column] = 1.0
        placement[expert] = (
            mx.array(positions),
            mx.array(slots),
            mx.array(matrix),
        )

    def run() -> tuple[float, mx.array, mx.array]:
        started = time.perf_counter_ns()
        scores = mx.sigmoid(mx_inputs.astype(mx.float32) @ mx_router_weight.T)
        choices = scores + mx_correction_bias
        chosen = mx.argpartition(choices, kth=-TOP_K, axis=-1)[:, -TOP_K:]
        selected_scores = mx.take_along_axis(scores, static_indices, axis=-1)
        route_weights = selected_scores / (mx.sum(selected_scores, axis=-1, keepdims=True) + 1e-20)
        result = mx.zeros(mx_inputs.shape, dtype=mx.float32)
        for expert, (positions, slots, matrix) in placement.items():
            expert_output = candidate_expert(
                mx_inputs[positions],
                quantized[expert],
                bits,
            )
            weights = route_weights[positions, slots]
            result = result + matrix @ (expert_output * weights[:, None])
        mx.eval(result, chosen)
        mx.synchronize()
        return (time.perf_counter_ns() - started) / 1_000_000, result, chosen

    for _ in range(10):
        run()
    measurements = []
    output = None
    chosen = None
    for _ in range(30):
        elapsed, output, chosen = run()
        measurements.append(elapsed)
    assert output is not None and chosen is not None
    chosen_sets = [sorted(row) for row in np.array(chosen).tolist()]
    expected_sets = [sorted(row) for row in indices_np.tolist()]
    if chosen_sets != expected_sets:
        raise ValueError("MLX router selection differs from source-derived route")
    actual = np.array(output).astype(np.float32)
    median_ms = statistics.median(measurements)
    executable_bytes = sum(
        array.nbytes
        for projections in quantized.values()
        for arrays in projections.values()
        for array in arrays
    ) + mx_router_weight.nbytes + mx_correction_bias.nbytes
    result = {
        "schema_version": 1,
        "semantic": "real_noaux_tc_router_heterogeneous_expert_weighted_sum",
        "exactness": f"L3_affine_INT{bits}_experts_with_source_router",
        "device": str(mx.default_device()),
        "machine": platform.machine(),
        "mlx_version": "0.31.2",
        "layer": layer,
        "batch_size": batch_size,
        "top_k": TOP_K,
        "bits": bits,
        "group_size": GROUP_SIZE,
        "selected_experts_by_position": indices_np.tolist(),
        "route_weights_by_position": source_route_weights.numpy().tolist(),
        "unique_experts": sorted(schedule),
        "unique_expert_count": len(schedule),
        "expert_union_factor": len(schedule) / TOP_K,
        "expert_position_counts": {
            str(expert): len(positions) for expert, (positions, _) in schedule.items()
        },
        "warmup_runs": 10,
        "measured_runs": 30,
        "wall_ms": measurements,
        "wall_median_ms": median_ms,
        "wall_p10_ms": percentile(measurements, 0.1),
        "wall_p90_ms": percentile(measurements, 0.9),
        "accepted_positions_per_second": batch_size / (median_ms / 1_000),
        "executable_bytes": executable_bytes,
        "peak_mlx_memory_bytes_during_setup_and_benchmark": mx.get_peak_memory(),
        "cache_state": "all selected expert buffers warm; source load and quantization excluded",
        "concurrency": 1,
        "router_selection_matches_source": True,
        "error_vs_source_fp8_moe_output": error_metrics(actual, reference_np),
        "first_output_values": actual[0, :8].tolist(),
        "first_source_fp8_values": reference_np[0, :8].tolist(),
        "limitation": "Static source-derived dispatch schedule is specialized to this fixture; router scores and weights are recomputed in every timed run.",
    }
    if fixture is not None:
        verify_fixture(result, fixture)
        result["correctness_fixture_verified"] = True
    else:
        result["correctness_fixture_verified"] = False
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--router", required=True, type=Path)
    parser.add_argument("--expert-dir", required=True, type=Path)
    parser.add_argument("--layer", type=int, default=43)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--fixture", type=Path)
    parser.add_argument("--bits", type=int, default=4, choices=(2, 3, 4, 5, 6, 8))
    arguments = parser.parse_args()
    try:
        paths = sorted(arguments.expert_dir.glob("*.safetensors"))
        if not paths:
            raise ValueError("expert directory has no safetensors slices")
        fixture = (
            json.loads(arguments.fixture.read_text(encoding="utf-8"))
            if arguments.fixture
            else None
        )
        result = benchmark(arguments.router, paths, arguments.layer, arguments.bits, fixture)
        atomic_write_new(arguments.output, canonical_json(result))
        print(arguments.output)
        return 0
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"error: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
