#!/usr/bin/env python3
"""Measure a conservative BF16 ambiguity bound for PW-0202."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from safetensors import safe_open
import torch


INDEX_SHA256 = "f2e1774c9acf9a62338b68c144e6fc7a66495e59f2e64b3078c1b7ef5a196816"
REFERENCE_MANIFEST_SHA256 = "9c96d85e45832abdccd3be2325db993749579a904469d1862c8f3437cafab86d"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dynamic_input(values: torch.Tensor) -> np.ndarray:
    rows, columns = values.shape
    grouped = values.float().reshape(rows, columns // 128, 128)
    scales = grouped.abs().amax(-1).clamp(min=1e-10) / 448.0
    encoded = torch.clamp(grouped / scales.unsqueeze(-1), -448.0, 448.0).to(
        torch.float8_e4m3fn
    )
    return (encoded.float() * scales.unsqueeze(-1)).reshape(rows, columns).numpy()[0]


def finite_bf16_values() -> np.ndarray:
    bits = np.arange(65536, dtype=np.uint32) << 16
    values = bits.view(np.float32)
    return np.unique(np.sort(values[np.isfinite(values)].astype(np.float64)))


def midpoint_distances(values: np.ndarray, bf16: np.ndarray) -> np.ndarray:
    indices = np.searchsorted(bf16, values.astype(np.float64), side="right")
    indices = np.clip(indices, 1, len(bf16) - 1)
    lower_midpoint = (bf16[indices - 1] + bf16[indices]) * 0.5
    upper_indices = np.clip(indices + 1, 1, len(bf16) - 1)
    upper_midpoint = (bf16[upper_indices - 1] + bf16[upper_indices]) * 0.5
    return np.minimum(
        np.abs(values.astype(np.float64) - lower_midpoint),
        np.abs(values.astype(np.float64) - upper_midpoint),
    )


def write_new(path: Path, payload: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as output:
        output.write(payload)
        output.flush()
        os.fsync(output.fileno())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--index", required=True, type=Path)
    parser.add_argument("--reference-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    if sha256(args.index) != INDEX_SHA256:
        raise ValueError("checkpoint index SHA-256 mismatch")
    if sha256(args.reference_root / "manifest.json") != REFERENCE_MANIFEST_SHA256:
        raise ValueError("PW-0101 reference manifest SHA-256 mismatch")
    index = json.loads(args.index.read_bytes())["weight_map"]
    audit = json.loads((args.audit / "manifest.json").read_bytes())
    bf16 = finite_bf16_values()
    unit_roundoff = 2.0 ** -24
    projection_records = []
    total_flagged = total_outputs = total_mismatches = captured_mismatches = 0

    for summary in audit["projection_reports"]:
        expert = summary["expert"]
        projection = summary["projection"]
        stem = f"expert-{expert}-{projection}"
        report = json.loads((args.audit / f"{stem}.json").read_bytes())
        raw = np.asarray(report["pre_round_output"], dtype=np.float32)
        candidate = np.fromfile(args.audit / f"{stem}.f32", dtype="<f4")
        reference = np.fromfile(
            args.reference_root / f"expert_{expert}_{projection}.f32", dtype="<f4"
        )
        input_path = (
            args.reference_root / "layer4_moe_input.f32"
            if projection != "down"
            else args.reference_root / f"expert_{expert}_swiglu.f32"
        )
        input_values = np.fromfile(input_path, dtype="<f4").copy()
        quantized = dynamic_input(torch.from_numpy(input_values).reshape(1, -1))
        name = f"model.layers.4.mlp.experts.{expert}.{projection}_proj.weight"
        with safe_open(args.checkpoint / index[name], framework="pt", device="cpu") as tensors:
            weight = tensors.get_tensor(name).float()
            scale = tensors.get_tensor(name + "_scale_inv").float()
        expanded = (
            weight * scale.repeat_interleave(128, 0).repeat_interleave(128, 1)
        ).numpy()
        sum_abs = np.abs(expanded.astype(np.float64)) @ np.abs(quantized.astype(np.float64))
        columns = expanded.shape[1]
        gamma = columns * unit_roundoff / (1.0 - columns * unit_roundoff)
        cross_backend_bound = 2.0 * gamma * sum_abs
        distance = midpoint_distances(raw, bf16)
        flagged = cross_backend_bound >= distance
        mismatch = candidate.view(np.uint32) != reference.view(np.uint32)
        captured = flagged & mismatch
        total_flagged += int(np.count_nonzero(flagged))
        total_outputs += len(raw)
        total_mismatches += int(np.count_nonzero(mismatch))
        captured_mismatches += int(np.count_nonzero(captured))
        projection_records.append({
            "expert": expert,
            "projection": projection,
            "columns": columns,
            "gamma_n": gamma,
            "outputs": len(raw),
            "flagged_rows": int(np.count_nonzero(flagged)),
            "flagged_fraction": float(np.mean(flagged)),
            "mismatch_rows": int(np.count_nonzero(mismatch)),
            "captured_mismatch_rows": int(np.count_nonzero(captured)),
            "maximum_bound": float(np.max(cross_backend_bound)),
            "minimum_midpoint_distance": float(np.min(distance)),
        })
        del weight, scale, expanded, sum_abs

    result = {
        "schema_version": 1,
        "semantic": "pw0202_conservative_cross_backend_bf16_ambiguity_bound",
        "checkpoint_index_sha256": INDEX_SHA256,
        "reference_manifest_sha256": REFERENCE_MANIFEST_SHA256,
        "bound": "2*gamma_n*sum_abs_products",
        "unit_roundoff": unit_roundoff,
        "projection_records": projection_records,
        "total_outputs": total_outputs,
        "total_flagged_rows": total_flagged,
        "aggregate_flagged_fraction": total_flagged / total_outputs,
        "total_mismatch_rows": total_mismatches,
        "captured_mismatch_rows": captured_mismatches,
        "all_mismatches_captured": captured_mismatches == total_mismatches,
        "accepted_tokens": 0,
        "A": 0,
        "U": 0,
        "performance_claim": None,
    }
    payload = (json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n").encode()
    write_new(args.output, payload)


if __name__ == "__main__":
    main()
