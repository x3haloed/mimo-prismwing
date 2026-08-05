#!/usr/bin/env python3
"""Generate the seeded tiny MiMo noaux_tc + SwiGLU answer key."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def dot(row, vector):
    return sum(left * right for left, right in zip(row, vector, strict=True))


def linear(matrix, vector):
    return [dot(row, vector) for row in matrix]


def silu(value):
    return value / (1.0 + math.exp(-value))


def expert(expert_weights, vector):
    gate = linear(expert_weights["gate"], vector)
    up = linear(expert_weights["up"], vector)
    activated = [silu(gate_value) * up_value for gate_value, up_value in zip(gate, up, strict=True)]
    return linear(expert_weights["down"], activated)


def route(fixture, vector):
    scores = [1.0 / (1.0 + math.exp(-dot(row, vector))) for row in fixture["router_weight"]]
    choice = [score + bias for score, bias in zip(scores, fixture["correction_bias"], strict=True)]
    group_size = len(scores) // fixture["groups"]
    group_scores = []
    for group in range(fixture["groups"]):
        values = sorted(choice[group * group_size : (group + 1) * group_size], reverse=True)
        group_scores.append(sum(values[:2]))
    selected_groups = sorted(range(fixture["groups"]), key=lambda group: (-group_scores[group], group))[
        : fixture["topk_group"]
    ]
    candidates = [index for index in range(len(scores)) if index // group_size in selected_groups]
    indices = sorted(candidates, key=lambda index: (-choice[index], index))[: fixture["top_k"]]
    denominator = sum(scores[index] for index in indices) + 1e-20
    weights = [scores[index] / denominator * fixture["routed_scaling_factor"] for index in indices]
    return indices, weights


def generate():
    fixture = {
        "schema_version": 1,
        "semantic": "mimo_v2_noaux_tc_swiglu_moe",
        "source_revision": "63651580ca774f8504f676040460aed3e1244ac1",
        "hidden_size": 3,
        "intermediate_size": 2,
        "groups": 2,
        "topk_group": 1,
        "top_k": 2,
        "normalize_topk": True,
        "routed_scaling_factor": 1.0,
        "inputs": [[1.0, -0.5, 0.25], [-0.4, 0.8, 1.2]],
        "router_weight": [
            [0.8, -0.1, 0.2],
            [0.4, 0.3, -0.7],
            [-0.6, 0.9, 0.5],
            [0.2, -0.8, 0.6],
        ],
        "correction_bias": [0.02, -0.01, 0.03, -0.02],
        "experts": [],
    }
    for expert_index in range(4):
        base = expert_index + 1
        fixture["experts"].append(
            {
                "gate": [
                    [0.10 * base, -0.07, 0.03],
                    [-0.02, 0.08 * base, 0.05],
                ],
                "up": [
                    [0.06, 0.04 * base, -0.09],
                    [0.07 * base, -0.03, 0.02],
                ],
                "down": [
                    [0.11, -0.05 * base],
                    [0.02 * base, 0.09],
                    [-0.04, 0.07 * base],
                ],
            }
        )
    expected = []
    for vector in fixture["inputs"]:
        indices, weights = route(fixture, vector)
        outputs = [expert(fixture["experts"][index], vector) for index in indices]
        combined = [
            sum(weights[position] * outputs[position][dimension] for position in range(len(indices)))
            for dimension in range(fixture["hidden_size"])
        ]
        expected.append({"topk_indices": indices, "topk_weights": weights, "output": combined})
    fixture["expected"] = expected
    return fixture


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    arguments = parser.parse_args()
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(generate(), indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
