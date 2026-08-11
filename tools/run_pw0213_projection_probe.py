#!/usr/bin/env python3
"""Run one bounded PW-0213 real-query projection probe."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

try:
    from tools.openrouter_reference import atomic_write_new, canonical_json
    from tools.remote_fp8_symbol_census import decode_e4m3fn
    from tools.row_query_quantizers import (
        estimate_mse_inner_products,
        estimate_covariance_inner_products,
        estimate_turbo_prod_inner_products,
        fit_block_covariance_transform,
        quantize_covariance_rows,
        quantize_covariance_shared_block_codebooks,
        quantize_rotated_rows,
        seeded_signs,
        turbo_prod_quantize_rows,
    )
except ModuleNotFoundError:
    from openrouter_reference import atomic_write_new, canonical_json
    from remote_fp8_symbol_census import decode_e4m3fn
    from row_query_quantizers import (
        estimate_mse_inner_products,
        estimate_covariance_inner_products,
        estimate_turbo_prod_inner_products,
        fit_block_covariance_transform,
        quantize_covariance_rows,
        quantize_covariance_shared_block_codebooks,
        quantize_rotated_rows,
        seeded_signs,
        turbo_prod_quantize_rows,
    )


CORPUS_SHA256 = "b9df976876d63c1ffbbe0c70507aea8b939a749ce5b1db27cbca0b5d82cf802e"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bf16(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    bits = values.view(np.uint32)
    rounded = bits + np.uint32(0x7FFF) + ((bits >> 16) & 1)
    return (rounded & np.uint32(0xFFFF0000)).view(np.float32)


def fp8_nearest(values: np.ndarray) -> np.ndarray:
    finite = sorted({decode_e4m3fn(code) for code in range(256) if code not in (0x7F, 0xFF)})
    grid = np.asarray(finite, dtype=np.float32)
    boundaries = (grid[:-1] + grid[1:]) * 0.5
    return grid[np.searchsorted(boundaries, values)]


def dynamic_input(values: np.ndarray) -> np.ndarray:
    values = bf16(values)
    rows, columns = values.shape
    grouped = values.reshape(rows, columns // 128, 128)
    scales = np.maximum(np.max(np.abs(grouped), axis=-1), 1e-10) / 448.0
    encoded = fp8_nearest(np.clip(grouped / scales[..., None], -448.0, 448.0))
    return (encoded * scales[..., None]).reshape(rows, columns).astype(np.float32)


def decode_weight(codes: np.ndarray, scales: np.ndarray) -> np.ndarray:
    lut = np.asarray([decode_e4m3fn(code) for code in range(256)], dtype=np.float32)
    if not np.all(np.isfinite(lut[codes])):
        raise ValueError("source weight contains NaN code")
    return lut[codes] * np.repeat(np.repeat(scales, 128, axis=0), 128, axis=1)


def affine6(weight: np.ndarray) -> np.ndarray:
    rows, columns = weight.shape
    grouped = weight.reshape(rows, columns // 128, 128)
    minimum = grouped.min(axis=-1)
    maximum = grouped.max(axis=-1)
    step = (maximum - minimum) / 63.0
    codes = np.rint((grouped - minimum[..., None]) / np.where(step[..., None], step[..., None], 1.0))
    return (minimum[..., None] + np.clip(codes, 0, 63) * step[..., None]).reshape(rows, columns)


def metrics(actual: np.ndarray, expected: np.ndarray) -> dict:
    error = np.asarray(actual, dtype=np.float64) - np.asarray(expected, dtype=np.float64)
    squared_error = float(np.sum(error * error))
    reference = float(np.sum(np.asarray(expected, dtype=np.float64) ** 2))
    row_reference = np.linalg.norm(expected, axis=1)
    row_error = np.linalg.norm(error, axis=1) / np.maximum(row_reference, 1e-30)
    return {
        "relative_l2": (squared_error / reference) ** 0.5,
        "mean_signed_error": float(np.mean(error)),
        "normalized_bias": float(np.mean(error)) / max(float(np.sqrt(np.mean(expected ** 2))), 1e-30),
        "error_variance": float(np.var(error)),
        "median_row_relative_l2": float(np.median(row_error)),
        "p95_row_relative_l2": float(np.quantile(row_error, 0.95)),
        "maximum_row_relative_l2": float(np.max(row_error)),
    }


def run(source: Path, corpus: Path, layer: int, expert: int, projection: str, seed: int) -> dict:
    if sha256_file(corpus) != CORPUS_SHA256:
        raise ValueError("PW-0116 corpus authority mismatch")
    manifest = json.loads(corpus.read_text(encoding="utf-8"))
    authority = next(record for record in manifest["layers"] if record["layer"] == layer)
    schedule = next(record for record in authority["expert_schedule"] if record["expert"] == expert)
    train_positions = [position for position in schedule["positions"] if position < 112]
    positions = [position for position in schedule["positions"] if 112 <= position < 168]
    if not positions:
        raise ValueError("expert has no validation placements")
    root = corpus.parent
    capture = authority["captures"]["moe_input"]
    capture_path = root / capture["file"]
    if sha256_file(capture_path) != capture["sha256"]:
        raise ValueError("moe_input payload mismatch")
    moe_input = np.memmap(capture_path, dtype="<f4", mode="r", shape=tuple(capture["shape"]))
    arrays = np.load(source, allow_pickle=False)
    key = f"layers__{layer}__mlp__experts__{expert}__{projection}_proj__weight"
    weight = decode_weight(arrays[f"{key}__codes"], arrays[f"{key}__scales"])
    if projection == "down":
        raise ValueError("bounded first probe supports gate/up; down requires source SwiGLU construction")
    queries = dynamic_input(np.asarray(moe_input[positions], dtype=np.float32))
    train_queries = dynamic_input(np.asarray(moe_input[train_positions], dtype=np.float32))
    expected = bf16(queries @ weight.T)
    signs = seeded_signs(weight.shape[1], seed)
    qjl_signs = seeded_signs(weight.shape[1], seed ^ 0x9E3779B9)
    mse6 = quantize_rotated_rows(weight, 6, signs)
    prod6 = turbo_prod_quantize_rows(weight, signs, qjl_signs)
    covariance = fit_block_covariance_transform(train_queries)
    covariance6 = quantize_covariance_rows(weight, covariance)
    covariance_shared6 = quantize_covariance_shared_block_codebooks(weight, covariance)
    prod_total, prod_base, prod_correction = estimate_turbo_prod_inner_products(
        queries, prod6, signs, qjl_signs
    )
    source_bytes = int(weight.size)
    rows, columns = weight.shape
    return {
        "schema_version": 1,
        "evidence_class": "pw0213_bounded_real_query_projection_probe",
        "layer": layer,
        "expert": expert,
        "projection": projection,
        "seed": seed,
        "validation_positions": positions,
        "source_weight_bytes": source_bytes,
        "candidates": {
            "affine6_rtn": {
                "bytes": columns * rows * 6 // 8 + rows * (columns // 128) * 4,
                "metrics": metrics(queries @ affine6(weight).T, expected),
            },
            "turboquant_mse6": {
                "bytes": columns * rows * 6 // 8 + rows * 4,
                "metrics": metrics(estimate_mse_inner_products(queries, mse6, signs), expected),
            },
            "turboquant_prod6_structured_qjl": {
                "bytes": columns * rows * 6 // 8 + rows * 4,
                "metrics": metrics(prod_total, expected),
                "base_metrics": metrics(prod_base, expected),
                "correction_rms": float(np.sqrt(np.mean(prod_correction ** 2))),
            },
            "block_covariance6": {
                "bytes_without_shared_basis": columns * rows * 6 // 8 + rows * 4,
                "shared_basis_f32_bytes": (columns // 128) * 128 * 128 * 4,
                "query_transform_multiplications": (columns // 128) * 128 * 128,
                "metrics": metrics(
                    estimate_covariance_inner_products(queries, covariance6, covariance),
                    expected,
                ),
            },
            "block_covariance_shared_grid6": {
                "code_bytes": columns * rows * 6 // 8,
                "shared_codebook_f16_bytes": (columns // 128) * 64 * 2,
                "shared_basis_f32_bytes": (columns // 128) * 128 * 128 * 4,
                "query_transform_multiplications": (columns // 128) * 128 * 128,
                "metrics": metrics(
                    estimate_covariance_inner_products(
                        queries, covariance_shared6, covariance
                    ),
                    expected,
                ),
            },
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--layer", required=True, type=int)
    parser.add_argument("--expert", required=True, type=int)
    parser.add_argument("--projection", required=True, choices=("gate", "up", "down"))
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    result = run(arguments.source, arguments.corpus, arguments.layer, arguments.expert, arguments.projection, arguments.seed)
    atomic_write_new(arguments.output, canonical_json(result))
    print(arguments.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
