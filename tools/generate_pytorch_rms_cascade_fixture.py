#!/usr/bin/env python3
"""Extract PW-0078's hash-bound PyTorch F32 RMS cascade fixture."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import struct

import numpy as np
import torch

try:
    from tools.openrouter_reference import atomic_write_new, canonical_json
except ModuleNotFoundError:
    from openrouter_reference import atomic_write_new, canonical_json


MANIFEST_SHA256 = "53e6b5db1d63128fddc2d3d6a8445424021f89f6f20131c98a42ab857f819e1f"
ROW = 1


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def f32_bits(value: torch.Tensor | np.float32) -> int:
    return struct.unpack("<I", struct.pack("<f", float(value)))[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oracle", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if sha256(args.oracle) != MANIFEST_SHA256:
        raise ValueError("RMS fixture oracle manifest hash mismatch")
    manifest = json.loads(args.oracle.read_text())
    if manifest.get("semantic") != "mimo_real_layer14_complete_oracle":
        raise ValueError("RMS fixture oracle semantic mismatch")
    record = manifest["captures"]["post_attention"]
    path = args.oracle.parent / record["file"]
    if (record.get("shape") != [27, 4096]
            or record.get("dtype") != "BF16_widened_F32"
            or sha256(path) != record.get("sha256")):
        raise ValueError("RMS fixture capture mismatch")
    row = np.fromfile(path, dtype="<f4").reshape(27, 4096)[ROW].copy()
    tensor = torch.from_numpy(row).to(torch.bfloat16)
    variance = tensor.float().pow(2).mean()
    inverse = torch.rsqrt(variance + torch.tensor(1.0e-5, dtype=torch.float32))
    high_precision_variance = np.float32(np.sum(row.astype(np.float64) ** 2) / row.size)
    if f32_bits(variance) == f32_bits(high_precision_variance):
        raise ValueError("high-precision reduction no longer discriminates RMS boundary")
    output = {
        "schema_version": 1,
        "semantic": "pytorch_aarch64_f32_rms_cascade_order",
        "torch_version": torch.__version__,
        "torch_commit": "cf30153c4c131c8164ee7798e5022d810682e2cb",
        "source_manifest_sha256": MANIFEST_SHA256,
        "post_attention_capture_sha256": record["sha256"],
        "row": ROW,
        "width": 4096,
        "epsilon_f32_u32": f32_bits(np.float32(1.0e-5)),
        "input_bf16_u16": tensor.view(torch.uint16).tolist(),
        "variance_f32_u32": f32_bits(variance),
        "inverse_f32_u32": f32_bits(inverse),
        "high_precision_variance_f32_u32": f32_bits(high_precision_variance),
    }
    atomic_write_new(args.output, canonical_json(output))


if __name__ == "__main__":
    main()
