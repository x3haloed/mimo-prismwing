#!/usr/bin/env python3
"""Validate and summarize PW-0302's bounded early-gate probes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics

try:
    from tools.openrouter_reference import atomic_write_new, canonical_json
except ModuleNotFoundError:
    from openrouter_reference import atomic_write_new, canonical_json


SEEDS = (16001, 16002, 16003, 16004, 16005)


def analyze(records: list[dict], covariance: dict) -> dict:
    if [record.get("seed") for record in records] != list(SEEDS):
        raise ValueError("seed panel mismatch")
    for record in records:
        if (
            record.get("evidence_class") != "pw0302_bounded_real_query_projection_probe"
            or (record.get("layer"), record.get("expert"), record.get("projection"))
            != (4, 96, "gate")
        ):
            raise ValueError("probe authority mismatch")
    affine = records[0]["candidates"]["affine6_rtn"]["metrics"]["relative_l2"]
    if any(record["candidates"]["affine6_rtn"]["metrics"]["relative_l2"] != affine for record in records):
        raise ValueError("affine control changed across seeds")
    candidates = {}
    for name in ("turboquant_mse6", "turboquant_prod6_structured_qjl"):
        errors = [record["candidates"][name]["metrics"]["relative_l2"] for record in records]
        biases = [record["candidates"][name]["metrics"]["normalized_bias"] for record in records]
        candidates[name] = {
            "minimum_relative_l2": min(errors),
            "median_relative_l2": statistics.median(errors),
            "maximum_relative_l2": max(errors),
            "median_regression_versus_affine": statistics.median(errors) / affine - 1.0,
            "minimum_normalized_bias": min(biases),
            "maximum_normalized_bias": max(biases),
            "passes_early_gate": max(errors) < affine,
        }
    shared = covariance["candidates"]["block_covariance_shared_grid6"]
    candidates["block_covariance_shared_grid6"] = {
        "relative_l2": shared["metrics"]["relative_l2"],
        "regression_versus_affine": shared["metrics"]["relative_l2"] / affine - 1.0,
        "normalized_bias": shared["metrics"]["normalized_bias"],
        "passes_early_gate": shared["metrics"]["relative_l2"] < affine,
    }
    return {
        "schema_version": 1,
        "evidence_class": "pw0302_validated_bounded_real_query_row_representation_control",
        "layer": 4,
        "expert": 96,
        "projection": "gate",
        "affine6_relative_l2": affine,
        "candidates": candidates,
        "decision": "reject_tested_row_representations_before_nine_projection_expansion",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", action="append", required=True, type=Path)
    parser.add_argument("--covariance", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    records = [json.loads(path.read_text(encoding="utf-8")) for path in arguments.probe]
    covariance = json.loads(arguments.covariance.read_text(encoding="utf-8"))
    atomic_write_new(arguments.output, canonical_json(analyze(records, covariance)))
    print(arguments.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
