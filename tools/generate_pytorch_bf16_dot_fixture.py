#!/usr/bin/env python3
"""Extract PW-0070's hash-bound PyTorch BF16 dot-product fixture."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import struct

import numpy as np
import torch

try:
    from tools.openrouter_reference import atomic_write_new, canonical_json
except ModuleNotFoundError:
    from openrouter_reference import atomic_write_new, canonical_json


ORACLE_MANIFEST_SHA256 = "632b19962663bee4c603cba96ff5f3f65c3f6f72747d0a22e1df0481acd79d55"
POSITION = 22
HEAD = 12
KV_HEAD = 1
SOURCE_TOKEN = 17
WIDTH = 192


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def f32_bits(value: torch.Tensor | float) -> int:
    return struct.unpack("<I", struct.pack("<f", float(value)))[0]


def source_four_lane_dot(left: np.ndarray, right: np.ndarray) -> np.float32:
    partials = [np.float32(0.0)] * 4
    for index, (a, b) in enumerate(zip(left, right, strict=True)):
        partials[index % 4] = np.float32(partials[index % 4] + np.float32(a * b))
    for lane in range(1, 4):
        partials[0] = np.float32(partials[0] + partials[lane])
    return partials[0]


def forward_dot(left: np.ndarray, right: np.ndarray) -> np.float32:
    total = np.float32(0.0)
    for a, b in zip(left, right, strict=True):
        total = np.float32(total + np.float32(a * b))
    return total


def checked_capture(root: Path, manifest: dict, name: str, shape: list[int]) -> np.ndarray:
    record = manifest["captures"][name]
    path = root / record["file"]
    if record["shape"] != shape or record["dtype"] != "BF16_widened_F32":
        raise ValueError(f"unexpected {name} capture metadata")
    if sha256(path) != record["sha256"]:
        raise ValueError(f"{name} capture hash mismatch")
    return np.fromfile(path, dtype="<f4").reshape(shape)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oracle", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    if sha256(args.oracle) != ORACLE_MANIFEST_SHA256:
        raise ValueError("PW-0069 oracle manifest hash mismatch")
    manifest = json.loads(args.oracle.read_text())
    if manifest.get("semantic") != "mimo_real_layer7_complete_oracle":
        raise ValueError("PW-0069 oracle semantic mismatch")
    root = args.oracle.parent
    query = checked_capture(root, manifest, "query", [27, 64, WIDTH])
    key = checked_capture(root, manifest, "key", [27, 8, WIDTH])

    query_tensor = torch.from_numpy(query[POSITION, HEAD].copy()).to(torch.bfloat16)
    key_tensor = torch.from_numpy(key[SOURCE_TOKEN, KV_HEAD].copy()).to(torch.bfloat16)
    dot = query_tensor @ key_tensor
    dot_bf16 = dot.to(torch.bfloat16)
    scale = torch.tensor(1.0 / math.sqrt(WIDTH), dtype=torch.float32)
    scaled = (dot_bf16 * scale).to(torch.bfloat16)

    row = []
    for token in range(POSITION + 1):
        row.append((query_tensor @ torch.from_numpy(key[token, KV_HEAD].copy()).to(torch.bfloat16)).to(torch.bfloat16) * scale)
    sink = checked_capture(root, manifest, "sinks", [64])[HEAD]
    row_tensor = torch.cat((torch.stack(row).to(torch.bfloat16), torch.tensor([sink], dtype=torch.bfloat16)))
    maximum = row_tensor.max()
    centered = (scaled - maximum).to(torch.bfloat16)

    output = {
        "schema_version": 1,
        "semantic": "pytorch_aarch64_bf16_dot_four_lane_order",
        "torch_version": torch.__version__,
        "torch_commit": "cf30153c4c131c8164ee7798e5022d810682e2cb",
        "source_manifest_sha256": ORACLE_MANIFEST_SHA256,
        "query_capture_sha256": manifest["captures"]["query"]["sha256"],
        "key_capture_sha256": manifest["captures"]["key"]["sha256"],
        "position": POSITION,
        "head": HEAD,
        "kv_head": KV_HEAD,
        "source_token": SOURCE_TOKEN,
        "width": WIDTH,
        "query_bf16_u16": query_tensor.view(torch.uint16).tolist(),
        "key_bf16_u16": key_tensor.view(torch.uint16).tolist(),
        "source_four_lane_dot_f32_u32": f32_bits(source_four_lane_dot(
            query[POSITION, HEAD], key[SOURCE_TOKEN, KV_HEAD])),
        "forward_dot_f32_u32": f32_bits(forward_dot(
            query[POSITION, HEAD], key[SOURCE_TOKEN, KV_HEAD])),
        "dot_bf16_u16": int(dot_bf16.view(torch.uint16)),
        "scale_f32_u32": f32_bits(scale),
        "scaled_score_bf16_u16": int(scaled.view(torch.uint16)),
        "row_maximum_bf16_u16": int(maximum.view(torch.uint16)),
        "centered_score_bf16_u16": int(centered.view(torch.uint16)),
    }
    atomic_write_new(args.output, canonical_json(output))


if __name__ == "__main__":
    main()
