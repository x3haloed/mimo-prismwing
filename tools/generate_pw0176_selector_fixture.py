#!/usr/bin/env python3
"""Generate the independently computed tiny PW-0176 selector fixture."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import struct

import torch


SOURCE = Path(
    "/Users/chad/Models/mimo-prismwing/evidence/PW-0175/sources-001/minference-forward.py"
)
SOURCE_SHA256 = "b368e765fcc2021591f7cdf970e4e1a71731ed66f19e97023a13b70a02e6edc2"
CONTEXT = 140
LAST_QUERIES = 64
VERTICAL_SIZE = 31
SLASH_SIZE = 101


def score(row: int, key: int) -> float:
    value = ((row * 37 + key * 17) % 29 - 14) / 8.0
    spikes = {(0, 0): 7.0, (17, 31): 8.0, (63, 80): 9.0, (63, 139): 10.0}
    return spikes.get((row, key), value)


def descending(values: list[float], count: int) -> list[int]:
    return sorted(range(len(values)), key=lambda index: (-values[index], index))[:count]


def build() -> dict:
    if hashlib.sha256(SOURCE.read_bytes()).hexdigest() != SOURCE_SHA256:
        raise ValueError("MInference source hash mismatch")
    query_start = CONTEXT - LAST_QUERIES
    scores = torch.empty((LAST_QUERIES, CONTEXT), dtype=torch.float32)
    for row in range(LAST_QUERIES):
        query_position = query_start + row
        for key in range(CONTEXT):
            scores[row, key] = score(row, key) if key <= query_position else -torch.inf
    probabilities = torch.softmax(scores, dim=-1, dtype=torch.float32)
    vertical = probabilities.sum(dim=0).tolist()
    for index in range(30):
        vertical[index] = math.inf
    slash = [0.0] * CONTEXT
    for row in range(LAST_QUERIES):
        query_position = query_start + row
        for key in range(query_position + 1):
            slash[query_position - key] += float(probabilities[row, key])
    for index in range(100):
        slash[index] = math.inf
    vertical_positions = sorted(descending(vertical, VERTICAL_SIZE))
    slash_distances = sorted(descending(slash, SLASH_SIZE))

    def selected(query_position: int) -> list[int]:
        result = [position for position in vertical_positions if position <= query_position]
        result.extend(
            query_position - distance
            for distance in slash_distances
            if distance <= query_position
        )
        return sorted(set(result))

    probability_bytes = b"".join(
        struct.pack("<f", float(value)) for value in probabilities.flatten()
    )
    return {
        "schema_version": 1,
        "semantic": "minference_last64_vertical_slash_tiny_fixture",
        "source_sha256": SOURCE_SHA256,
        "context": CONTEXT,
        "last_queries": LAST_QUERIES,
        "query_start": query_start,
        "score_formula": "((row*37 + key*17) % 29 - 14) / 8 with four frozen spikes; future=-inf",
        "spikes": [[0, 0, 7.0], [17, 31, 8.0], [63, 80, 9.0], [63, 139, 10.0]],
        "softmax_dtype": "torch.float32",
        "probability_f32le_sha256": hashlib.sha256(probability_bytes).hexdigest(),
        "vertical_size": VERTICAL_SIZE,
        "slash_size": SLASH_SIZE,
        "forced_vertical_positions": 30,
        "forced_recent_slash_distances": 100,
        "tie_break": "descending value then lower original index",
        "vertical_positions": vertical_positions,
        "slash_distances": slash_distances,
        "selected_positions": {
            str(position): selected(position) for position in (63, 100, 139)
        },
        "full_selection_at_139": list(range(140)),
    }


if __name__ == "__main__":
    print(json.dumps(build(), sort_keys=True, separators=(",", ":")))
