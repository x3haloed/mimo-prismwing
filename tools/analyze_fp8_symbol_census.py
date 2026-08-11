#!/usr/bin/env python3
"""Analyze a remote FP8 symbol census and simulate a sampled global 6-bit codebook."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import statistics
import re
from typing import Any

try:
    from tools.remote_fp8_symbol_census import decode_e4m3fn
except ModuleNotFoundError:
    from remote_fp8_symbol_census import decode_e4m3fn

try:
    from tools.openrouter_reference import atomic_write_new, canonical_json
except ModuleNotFoundError:
    from openrouter_reference import atomic_write_new, canonical_json


SCHEMA_VERSION = 1


def greedy_codebook(counts: list[int], size: int) -> list[int]:
    if len(counts) != 256 or size <= 0 or size > 254:
        raise ValueError("invalid symbol counts or codebook size")
    values = [decode_e4m3fn(code) for code in range(256)]
    candidates = [code for code, value in enumerate(values) if math.isfinite(value)]
    selected: list[int] = [max(candidates, key=lambda code: (counts[code], -code))]
    nearest = [math.inf] * 256
    first_value = values[selected[0]]
    for code in range(256):
        nearest[code] = (values[code] - first_value) ** 2
    for _ in range(size - 1):
        best_code = None
        best_error = math.inf
        for candidate in candidates:
            if candidate in selected:
                continue
            candidate_error = 0.0
            value = values[candidate]
            for code, count in enumerate(counts):
                if not count:
                    continue
                distance = (values[code] - value) ** 2
                candidate_error += count * min(nearest[code], distance)
            if candidate_error < best_error:
                best_error = candidate_error
                best_code = candidate
        if best_code is None:
            raise ValueError("could not extend codebook")
        selected.append(best_code)
        value = values[best_code]
        for code in range(256):
            nearest[code] = min(nearest[code], (values[code] - value) ** 2)
    return sorted(selected, key=lambda code: (values[code], code))


def squared_error_sums(counts: list[int], codebook: list[int]) -> tuple[float, float]:
    values = [decode_e4m3fn(code) for code in range(256)]
    error = 0.0
    reference = 0.0
    for code, count in enumerate(counts):
        if not count:
            continue
        value = values[code]
        reconstructed = min(
            (values[candidate] for candidate in codebook),
            key=lambda candidate: abs(candidate - value),
        )
        error += count * (value - reconstructed) ** 2
        reference += count * value * value
    return error, reference


def relative_l2(counts: list[int], codebook: list[int]) -> float:
    error, reference = squared_error_sums(counts, codebook)
    return math.sqrt(error / reference) if reference else 0.0


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        raise ValueError("percentile requires values")
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * fraction)]


def error_summary(values: list[float]) -> dict[str, float]:
    return {
        "minimum": min(values),
        "median": statistics.median(values),
        "p95": percentile(values, 0.95),
        "maximum": max(values),
    }


def aggregate_counts(blocks: list[dict[str, Any]]) -> list[int]:
    return [sum(block["symbol_counts"][code] for block in blocks) for code in range(256)]


def holdout_analysis(train: dict[str, Any], holdout: dict[str, Any]) -> dict[str, Any]:
    train_samples = {sample["tensor"]: sample for sample in train["samples"]}
    holdout_samples = {sample["tensor"]: sample for sample in holdout["samples"]}
    if set(train_samples) != set(holdout_samples):
        raise ValueError("train/holdout tensor sets differ")
    tensor_errors = []
    tile_errors = []
    records = []
    for name in sorted(train_samples):
        training = train_samples[name]
        testing = holdout_samples[name]
        if training["row_block"] == testing["row_block"]:
            raise ValueError(f"train/holdout row blocks overlap: {name}")
        tensor_codebook = greedy_codebook(aggregate_counts(training["blocks"]), 64)
        tile_codebook = greedy_codebook(aggregate_counts(testing["blocks"]), 64)
        tensor_values = [relative_l2(block["symbol_counts"], tensor_codebook) for block in testing["blocks"]]
        tile_values = [relative_l2(block["symbol_counts"], tile_codebook) for block in testing["blocks"]]
        tensor_errors.extend(tensor_values)
        tile_errors.extend(tile_values)
        tile_source_bytes = testing["fetched_bytes"]
        records.append({
            "tensor": name,
            "train_row_block": training["row_block"],
            "holdout_row_block": testing["row_block"],
            "holdout_block_count": len(testing["blocks"]),
            "per_tensor_codebook_holdout_error": error_summary(tensor_values),
            "per_tile_codebook_error": error_summary(tile_values),
            "per_tile_physical_ratio": ((tile_source_bytes * 6 + 7) // 8 + 64) / tile_source_bytes,
        })
    return {
        "per_tensor_codebook_holdout_error": error_summary(tensor_errors),
        "per_tile_codebook_error": error_summary(tile_errors),
        "records": records,
        "limitation": (
            "Weight-only reconstruction errors. Per-tile codebooks are representation-local; "
            "per-tensor codebooks are trained on one row tile and scored on a disjoint row tile."
        ),
    }


EXPERT_TENSOR = re.compile(
    r"^model\.layers\.(?P<layer>\d+)\.mlp\.experts\.(?P<expert>\d+)\."
    r"(?P<projection>gate_proj|up_proj|down_proj)\.weight$"
)


def full_expert_analysis(census: dict[str, Any]) -> dict[str, Any]:
    """Score complete experts with scale-aware subset and affine-RTN errors."""
    grouped: dict[str, list[dict[str, Any]]] = {}
    identities: dict[str, tuple[int, int, str]] = {}
    for sample in census["samples"]:
        match = EXPERT_TENSOR.fullmatch(sample["tensor"])
        if not match:
            raise ValueError(f"not an expert projection: {sample['tensor']}")
        grouped.setdefault(sample["tensor"], []).append(sample)
        identities[sample["tensor"]] = (
            int(match["layer"]), int(match["expert"]), match["projection"]
        )

    projection_records = []
    expert_accumulators: dict[tuple[int, int], dict[str, float]] = {}
    for tensor in sorted(grouped):
        samples = sorted(grouped[tensor], key=lambda sample: sample["row_block"])
        shape = samples[0]["shape"]
        expected_rows = shape[0] // 128
        if shape[0] % 128 or [sample["row_block"] for sample in samples] != list(range(expected_rows)):
            raise ValueError(f"incomplete row-block coverage: {tensor}")
        if any(sample["shape"] != shape for sample in samples):
            raise ValueError(f"shape changed within tensor: {tensor}")

        subset_error = subset_reference = 0.0
        affine_error = affine_reference = 0.0
        block_errors = []
        source_bytes = candidate_bytes = affine_bytes = 0
        for sample in samples:
            if len(sample["scale_values"]) != len(sample["blocks"]):
                raise ValueError(f"scale/block mismatch: {tensor}")
            codebook = greedy_codebook(aggregate_counts(sample["blocks"]), 64)
            source_bytes += sample["fetched_bytes"]
            candidate_bytes += (sample["fetched_bytes"] * 6 + 7) // 8 + 64
            affine_bytes += (sample["fetched_bytes"] * 6 + 7) // 8
            affine_bytes += (sample["fetched_bytes"] // 128) * 4
            for block, scale in zip(sample["blocks"], sample["scale_values"]):
                scale_squared = scale * scale
                raw_error, raw_reference = squared_error_sums(block["symbol_counts"], codebook)
                subset_error += raw_error * scale_squared
                subset_reference += raw_reference * scale_squared
                affine_error += block["affine6_rtn_squared_error"] * scale_squared
                affine_reference += block["reference_squared_sum"] * scale_squared
                block_errors.append(math.sqrt(raw_error / raw_reference) if raw_reference else 0.0)

        layer, expert, projection = identities[tensor]
        record = {
            "tensor": tensor,
            "layer": layer,
            "expert": expert,
            "projection": projection,
            "shape": shape,
            "row_tile_count": len(samples),
            "quantization_block_count": sum(len(sample["blocks"]) for sample in samples),
            "source_weight_bytes": source_bytes,
            "fp8_subset_6bit_bytes": candidate_bytes,
            "fp8_subset_6bit_ratio": candidate_bytes / source_bytes,
            "affine6_rtn_bytes": affine_bytes,
            "affine6_rtn_ratio": affine_bytes / source_bytes,
            "fp8_subset_6bit_relative_l2": math.sqrt(subset_error / subset_reference),
            "affine6_rtn_relative_l2": math.sqrt(affine_error / affine_reference),
            "unscaled_block_fp8_subset_relative_l2": error_summary(block_errors),
        }
        projection_records.append(record)
        accumulator = expert_accumulators.setdefault(
            (layer, expert),
            {"subset_error": 0.0, "subset_reference": 0.0, "affine_error": 0.0,
             "affine_reference": 0.0, "source_bytes": 0.0, "candidate_bytes": 0.0,
             "affine_bytes": 0.0},
        )
        accumulator["subset_error"] += subset_error
        accumulator["subset_reference"] += subset_reference
        accumulator["affine_error"] += affine_error
        accumulator["affine_reference"] += affine_reference
        accumulator["source_bytes"] += source_bytes
        accumulator["candidate_bytes"] += candidate_bytes
        accumulator["affine_bytes"] += affine_bytes

    expert_records = []
    for (layer, expert), values in sorted(expert_accumulators.items()):
        expert_records.append({
            "layer": layer,
            "expert": expert,
            "source_weight_bytes": int(values["source_bytes"]),
            "fp8_subset_6bit_bytes": int(values["candidate_bytes"]),
            "fp8_subset_6bit_ratio": values["candidate_bytes"] / values["source_bytes"],
            "affine6_rtn_bytes": int(values["affine_bytes"]),
            "affine6_rtn_ratio": values["affine_bytes"] / values["source_bytes"],
            "fp8_subset_6bit_relative_l2": math.sqrt(
                values["subset_error"] / values["subset_reference"]
            ),
            "affine6_rtn_relative_l2": math.sqrt(
                values["affine_error"] / values["affine_reference"]
            ),
        })
    return {
        "projection_records": projection_records,
        "expert_records": expert_records,
        "limitation": (
            "Scale-aware complete-expert weight reconstruction only; no routed-output, "
            "route-stability, accumulated-logit, decoder-cost, or endpoint evidence."
        ),
    }


def analyze(
    census: dict[str, Any],
    holdout: dict[str, Any] | None = None,
    include_full_expert: bool = False,
) -> dict[str, Any]:
    if (
        census.get("schema_version") != 1
        or census.get("evidence_class") != "pinned_remote_deterministic_fp8_row_tile_samples"
    ):
        raise ValueError("unknown census identity")
    blocks = [block for sample in census["samples"] for block in sample["blocks"]]
    if not blocks:
        raise ValueError("census contains no blocks")
    aggregate = [0] * 256
    for block in blocks:
        counts = block.get("symbol_counts")
        if not isinstance(counts, list) or len(counts) != 256 or sum(counts) != 128 * 128:
            raise ValueError("malformed block symbol counts")
        aggregate = [left + right for left, right in zip(aggregate, counts)]
    if aggregate[0x7F] or aggregate[0xFF]:
        raise ValueError("sample contains E4M3FN NaN codes")
    codebook = greedy_codebook(aggregate, 64)
    errors = [relative_l2(block["symbol_counts"], codebook) for block in blocks]
    entropies = [block["entropy_bits_per_weight"] for block in blocks]
    escape_ratios = [block["exponent_top7_escape_bytes"] / (128 * 128) for block in blocks]
    result = {
        "schema_version": SCHEMA_VERSION,
        "evidence_class": "analysis_of_pinned_remote_fp8_symbol_samples",
        "repository": census["repository"],
        "revision": census["revision"],
        "sample_count": census["sample_count"],
        "quantization_block_count": len(blocks),
        "network_bytes": census["network_bytes"],
        "exact_palette_6bit_blocks": sum(
            block["exact_palette_bytes"]["6"] is not None for block in blocks
        ),
        "exact_palette_7bit_blocks": sum(
            block["exact_palette_bytes"]["7"] is not None for block in blocks
        ),
        "entropy_bits_per_weight": {
            "minimum": min(entropies),
            "median": statistics.median(entropies),
            "maximum": max(entropies),
        },
        "idealized_top7_exponent_escape_ratio": {
            "minimum": min(escape_ratios),
            "median": statistics.median(escape_ratios),
            "maximum": max(escape_ratios),
        },
        "greedy_global_fp8_subset_6bit": {
            "codebook_bytes": codebook,
            "codebook_values": [decode_e4m3fn(code) for code in codebook],
            "sampled_weight_relative_l2": error_summary(errors),
            "limitation": (
                "Weight-only sampled reconstruction error; it is not routed-output, "
                "route-stability, accumulated-logit, or endpoint fidelity evidence."
            ),
        },
    }
    if holdout is not None:
        if (
            holdout.get("repository") != census["repository"]
            or holdout.get("revision") != census["revision"]
        ):
            raise ValueError("holdout checkpoint identity mismatch")
        result["holdout"] = holdout_analysis(census, holdout)
    if include_full_expert:
        result["full_expert"] = full_expert_analysis(census)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--census", required=True, type=Path)
    parser.add_argument("--holdout-census", type=Path)
    parser.add_argument("--full-expert", action="store_true")
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    try:
        holdout = (
            json.loads(arguments.holdout_census.read_text(encoding="utf-8"))
            if arguments.holdout_census
            else None
        )
        result = analyze(
            json.loads(arguments.census.read_text(encoding="utf-8")),
            holdout,
            include_full_expert=arguments.full_expert,
        )
        atomic_write_new(arguments.output, canonical_json(result))
        print(arguments.output)
        return 0
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"error: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
