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
    # Reproduce the captured source's `sum_all_diagonal_matrix` construction,
    # including its source-index space and F32 reduction, before translating
    # selected indices into causal distances.
    matrix = probabilities[None, None]
    zero = torch.zeros((1, 1, LAST_QUERIES, LAST_QUERIES), dtype=torch.float32)
    padded = torch.cat((zero, matrix, zero), dim=-1)
    strided = padded.as_strided(
        (1, 1, LAST_QUERIES, LAST_QUERIES + CONTEXT),
        (1, LAST_QUERIES * (2 * LAST_QUERIES + CONTEXT), 2 * LAST_QUERIES + CONTEXT + 1, 1),
    )
    slash_source = torch.sum(strided, dim=2)[:, :, 1:][
        ..., : -LAST_QUERIES + 1
    ].flatten().tolist()
    for index in range(CONTEXT - 30, CONTEXT):
        slash_source[index] = math.inf
    vertical_positions = sorted(descending(vertical, VERTICAL_SIZE))
    slash_source_indices = descending(slash_source, SLASH_SIZE)
    slash_distances = sorted(CONTEXT - 1 - index for index in slash_source_indices)

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
        "forced_recent_slash_distances": 30,
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
