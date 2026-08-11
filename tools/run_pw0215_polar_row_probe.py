#!/usr/bin/env python3
"""Run PW-0215's bounded recursive-polar row-query probe."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

try:
    from tools.openrouter_reference import atomic_write_new, canonical_json
    from tools.run_pw0213_projection_probe import (
        CORPUS_SHA256, decode_weight, metrics, sha256_file,
    )
    from tools.run_pw0214_joint_swiglu_balance import affine6
except ModuleNotFoundError:
    from openrouter_reference import atomic_write_new, canonical_json
    from run_pw0213_projection_probe import CORPUS_SHA256, decode_weight, metrics, sha256_file
    from run_pw0214_joint_swiglu_balance import affine6


SEED = 0x215
LEVEL_BITS = 6
CENTROIDS = 1 << LEVEL_BITS


def signed_hadamard(values: np.ndarray, signs: np.ndarray) -> np.ndarray:
    result = np.asarray(values, dtype=np.float32) * signs
    width = result.shape[1]
    if width & (width - 1):
        raise ValueError("Hadamard width must be a power of two")
    result = result.copy()
    step = 1
    while step < width:
        view = result.reshape(result.shape[0], -1, step * 2)
        left = view[..., :step].copy()
        right = view[..., step:].copy()
        view[..., :step] = left + right
        view[..., step:] = left - right
        step *= 2
    return result / np.float32(np.sqrt(width))


def factor_rows(rows: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    current = np.asarray(rows, dtype=np.float32)
    levels = []
    while current.shape[1] > 1:
        paired = current.reshape(current.shape[0], -1, 2)
        levels.append(np.arctan2(paired[..., 1], paired[..., 0]).astype(np.float32))
        current = np.hypot(paired[..., 0], paired[..., 1]).astype(np.float32)
    return current[:, 0], levels


def fit_codebook(values: np.ndarray) -> np.ndarray:
    flat = values.reshape(-1).astype(np.float64)
    centers = np.quantile(flat, (np.arange(CENTROIDS) + 0.5) / CENTROIDS)
    for _ in range(12):
        boundaries = (centers[:-1] + centers[1:]) * 0.5
        codes = np.searchsorted(boundaries, flat)
        counts = np.bincount(codes, minlength=CENTROIDS)
        sums = np.bincount(codes, weights=flat, minlength=CENTROIDS)
        updated = centers.copy()
        present = counts > 0
        updated[present] = sums[present] / counts[present]
        if np.array_equal(updated, centers):
            break
        centers = updated
    return centers.astype(np.float16).astype(np.float32)


def encode_levels(levels: list[np.ndarray]) -> tuple[list[np.ndarray], list[np.ndarray]]:
    codebooks, codes = [], []
    for values in levels:
        centers = fit_codebook(values)
        boundaries = (centers[:-1] + centers[1:]) * 0.5
        codebooks.append(centers)
        codes.append(np.searchsorted(boundaries, values).astype(np.uint8))
    return codebooks, codes


def reconstruct(roots: np.ndarray, codebooks: list[np.ndarray], codes: list[np.ndarray]) -> np.ndarray:
    current = np.asarray(roots, dtype=np.float16).astype(np.float32)[:, None]
    for centers, level_codes in reversed(list(zip(codebooks, codes))):
        angles = centers[level_codes]
        expanded = np.empty((current.shape[0], current.shape[1] * 2), dtype=np.float32)
        expanded[:, 0::2] = current * np.cos(angles)
        expanded[:, 1::2] = current * np.sin(angles)
        current = expanded
    return current


def run(source: Path, corpus_path: Path) -> dict:
    if sha256_file(corpus_path) != CORPUS_SHA256:
        raise ValueError("PW-0116 corpus authority mismatch")
    corpus = json.loads(corpus_path.read_text())
    authority = next(x for x in corpus["layers"] if x["layer"] == 4)
    schedule = next(x for x in authority["expert_schedule"] if x["expert"] == 96)
    positions = [p for p in schedule["positions"] if 112 <= p < 168]
    capture = authority["captures"]["moe_input"]
    root = corpus_path.parent
    if sha256_file(root / capture["file"]) != capture["sha256"]:
        raise ValueError("PW-0116 capture mismatch")
    inputs = np.memmap(root / capture["file"], dtype="<f4", mode="r", shape=tuple(capture["shape"]))
    queries = np.asarray(inputs[positions], dtype=np.float32)
    arrays = np.load(source, allow_pickle=False)
    key = "layers__4__mlp__experts__96__gate_proj__weight"
    weight = decode_weight(arrays[key + "__codes"], arrays[key + "__scales"])
    rng = np.random.default_rng(SEED)
    signs = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=weight.shape[1])
    transformed_weight = signed_hadamard(weight, signs)
    transformed_queries = signed_hadamard(queries, signs)
    roots, levels = factor_rows(transformed_weight)
    codebooks, codes = encode_levels(levels)
    reconstructed = reconstruct(roots, codebooks, codes)
    expected = queries @ weight.T
    polar = transformed_queries @ reconstructed.T
    affine = queries @ affine6(weight).T
    row_payload = ((weight.shape[1] - 1) * LEVEL_BITS + 7) // 8 + 2
    physical_bytes = row_payload * weight.shape[0] + sum(x.nbytes // 2 for x in codebooks)
    source_bytes = weight.size
    weight_error = np.linalg.norm(reconstructed.astype(np.float64) - transformed_weight) / np.linalg.norm(transformed_weight)
    return {
        "schema_version": 1,
        "evidence_class": "pw0215_recursive_polar_row_query_probe",
        "layer": 4,
        "expert": 96,
        "projection": "gate",
        "positions": positions,
        "seed": SEED,
        "source_bytes": source_bytes,
        "physical_bytes": physical_bytes,
        "physical_ratio": physical_bytes / source_bytes,
        "row_payload_bytes": row_payload,
        "codebook_bytes": sum(x.nbytes // 2 for x in codebooks),
        "level_count": len(levels),
        "weight_relative_l2": float(weight_error),
        "affine6_metrics": metrics(affine, expected),
        "polar6_metrics": metrics(polar, expected),
        "extra_work": {
            "query_hadamard_add_subtracts": int(weight.shape[1] * np.log2(weight.shape[1])),
            "row_angle_lookups": weight.shape[0] * (weight.shape[1] - 1),
            "row_recursive_multiply_adds": weight.shape[0] * 2 * (weight.shape[1] - 1),
            "note": "Diagnostic dense reconstruction was used; counts describe a direct tree decoder before GEMV fusion.",
        },
        "codebook_sha256": hashlib.sha256(b"".join(x.astype("<f2").tobytes() for x in codebooks)).hexdigest(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    atomic_write_new(args.output, canonical_json(run(args.source, args.corpus)))
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
