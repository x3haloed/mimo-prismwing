#!/usr/bin/env python3
"""Test exact-dot repair at PW-0200's sparse BF16 mismatch sites."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from safetensors import safe_open
import torch


PW0200_SHA256 = "7ecc133ed4fa6319b6fda0ef3c74bc4c93e0428b29a583e4df6ab62b0a7748a5"
INDEX_SHA256 = "f2e1774c9acf9a62338b68c144e6fc7a66495e59f2e64b3078c1b7ef5a196816"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dynamic_input(values: torch.Tensor) -> torch.Tensor:
    rows, columns = values.shape
    grouped = values.float().reshape(rows, columns // 128, 128)
    scales = grouped.abs().amax(-1).clamp(min=1e-10) / 448.0
    encoded = torch.clamp(grouped / scales.unsqueeze(-1), -448.0, 448.0).to(
        torch.float8_e4m3fn
    )
    return (encoded.float() * scales.unsqueeze(-1)).reshape(rows, columns)


def write_new(path: Path, payload: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as output:
        output.write(payload)
        output.flush()
        os.fsync(output.fileno())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pw0200", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--index", required=True, type=Path)
    parser.add_argument("--reference-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    manifest_path = args.pw0200 / "manifest.json"
    if sha256(manifest_path) != PW0200_SHA256 or sha256(args.index) != INDEX_SHA256:
        raise ValueError("PW-0200 or checkpoint-index authority mismatch")
    manifest = json.loads(manifest_path.read_bytes())
    index = json.loads(args.index.read_bytes())["weight_map"]
    records = []

    for report in manifest["projection_reports"]:
        if report["bf16_mismatch_count"] == 0:
            continue
        expert = report["expert"]
        projection = report["projection"]
        stem = f"expert-{expert}-{projection}"
        candidate = np.fromfile(args.pw0200 / f"{stem}.f32", dtype="<f4")
        reference = np.fromfile(
            args.reference_root / f"expert_{expert}_{projection}.f32", dtype="<f4"
        )
        mismatch_rows = np.flatnonzero(candidate.view(np.uint32) != reference.view(np.uint32))
        input_path = (
            args.reference_root / "layer4_moe_input.f32"
            if projection != "down"
            else args.reference_root / f"expert_{expert}_swiglu.f32"
        )
        raw_input = np.fromfile(input_path, dtype="<f4").copy()
        quantized = dynamic_input(torch.from_numpy(raw_input).reshape(1, -1))[0].numpy()
        name = f"model.layers.4.mlp.experts.{expert}.{projection}_proj.weight"
        shard = args.checkpoint / index[name]
        with safe_open(shard, framework="pt", device="cpu") as tensors:
            weight = tensors.get_tensor(name).float()
            scale = tensors.get_tensor(name + "_scale_inv").float()
        for row in mismatch_rows.tolist():
            weight_row = (
                weight[row]
                * scale[row // 128].repeat_interleave(128)
            ).numpy()
            exact_f64 = float(np.sum(
                weight_row.astype(np.float64) * quantized.astype(np.float64),
                dtype=np.float64,
            ))
            exact_bf16 = float(torch.tensor(exact_f64).to(torch.bfloat16).float())
            exact_matches_reference = bool(
                np.float32(exact_bf16).view(np.uint32)
                == reference[row].view(np.uint32)
            )
            records.append({
                "expert": expert,
                "projection": projection,
                "row": row,
                "candidate_bits": int(candidate[row].view(np.uint32)),
                "reference_bits": int(reference[row].view(np.uint32)),
                "exact_f64": exact_f64,
                "exact_bf16_bits": int(np.float32(exact_bf16).view(np.uint32)),
                "exact_matches_reference": exact_matches_reference,
            })
        del weight, scale

    result = {
        "schema_version": 1,
        "semantic": "pw0201_exact_f64_dot_bf16_sparse_repair_oracle",
        "pw0200_manifest_sha256": PW0200_SHA256,
        "mismatch_sites": len(records),
        "exact_matches_reference": int(sum(record["exact_matches_reference"] for record in records)),
        "all_exact_match_reference": all(record["exact_matches_reference"] for record in records),
        "records": records,
        "accepted_tokens": 0,
        "A": 0,
        "U": 0,
        "performance_claim": None,
    }
    payload = (json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n").encode()
    write_new(args.output, payload)


if __name__ == "__main__":
    main()
