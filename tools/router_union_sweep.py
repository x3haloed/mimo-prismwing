#!/usr/bin/env python3
"""Measure real-router expert union sensitivity to input correlation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform

import numpy as np
from safetensors import safe_open
import torch

try:
    from tools.mlx_moe_block_benchmark import source_routes
    from tools.openrouter_reference import atomic_write_new, canonical_json
except ModuleNotFoundError:
    from mlx_moe_block_benchmark import source_routes
    from openrouter_reference import atomic_write_new, canonical_json


CORRELATIONS = (0.0, 0.5, 0.9, 0.99, 0.999, 1.0)


def summarize(values: list[int]) -> dict:
    array = np.array(values)
    return {
        "minimum_unique_experts": int(np.min(array)),
        "median_unique_experts": float(np.median(array)),
        "p90_unique_experts": float(np.percentile(array, 90)),
        "maximum_unique_experts": int(np.max(array)),
        "mean_expert_union_factor": float(np.mean(array) / 8),
        "fraction_exactly_eight_unique_experts": float(np.mean(array == 8)),
        "histogram": {
            str(value): int(count)
            for value, count in zip(*np.unique(array, return_counts=True))
        },
    }


def sweep(router_path: Path, layer: int, trials: int, seed: int) -> dict:
    with safe_open(router_path, framework="pt", device="cpu") as tensors:
        weight = tensors.get_tensor(f"model.layers.{layer}.mlp.gate.weight").float()
        bias = tensors.get_tensor(
            f"model.layers.{layer}.mlp.gate.e_score_correction_bias"
        ).float()
    generator = np.random.default_rng(seed)
    base = generator.standard_normal((trials, 1, weight.shape[1]), dtype=np.float32)
    noise = generator.standard_normal((trials, 8, weight.shape[1]), dtype=np.float32)
    records = []
    for correlation in CORRELATIONS:
        values = correlation * base + np.sqrt(1 - correlation**2) * noise
        values /= np.sqrt(np.mean(values * values, axis=-1, keepdims=True))
        indices, _ = source_routes(
            torch.from_numpy(values.reshape(-1, weight.shape[1])), weight, bias
        )
        indices = indices.reshape(trials, 8, 8).numpy()
        union_counts = [len(set(trial.flatten().tolist())) for trial in indices]
        records.append(
            {
                "inter_position_input_correlation": correlation,
                **summarize(union_counts),
            }
        )
    return {
        "schema_version": 1,
        "semantic": "source_noaux_tc_router_union_sensitivity",
        "evidence_class": "synthetic_RMS1_sensitivity_not_real_activation_distribution",
        "machine": platform.machine(),
        "layer": layer,
        "batch_positions": 8,
        "active_experts_per_position": 8,
        "trials_per_correlation": trials,
        "seed": seed,
        "input_distribution": "correlated standard normal, independently RMS-normalized per position",
        "records": records,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--router", required=True, type=Path)
    parser.add_argument("--layer", type=int, default=43)
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--seed", type=int, default=160043)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    try:
        result = sweep(arguments.router, arguments.layer, arguments.trials, arguments.seed)
        atomic_write_new(arguments.output, canonical_json(result))
        print(arguments.output)
        return 0
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"error: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
