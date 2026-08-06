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


WIDTH = 192
CASES = {
    "four_lane": {
        "manifest_sha256": "632b19962663bee4c603cba96ff5f3f65c3f6f72747d0a22e1df0481acd79d55",
        "oracle_semantic": "mimo_real_layer7_complete_oracle",
        "fixture_semantic": "pytorch_aarch64_bf16_dot_four_lane_order",
        "position": 22, "head": 12, "kv_head": 1, "source_token": 17, "kv_heads": 8,
    },
    "vector": {
        "manifest_sha256": "639730fb729855f94eecb5716abdbf68d6d98849c0cfbfbf1d87d86dc9d462dd",
        "oracle_semantic": "mimo_real_layer11_complete_oracle",
        "fixture_semantic": "pytorch_aarch64_bf16_specialized_vector_dot_order",
        "position": 22, "head": 3, "kv_head": 0, "source_token": 16, "kv_heads": 4,
    },
    "swa_vector": {
        "manifest_sha256": "5bf6ed69aa01293e8020e3d4b2dc3a34dd087672901f59e321258f8ab1c0313b",
        "oracle_semantic": "mimo_real_layer19_complete_oracle",
        "fixture_semantic": "pytorch_aarch64_bf16_specialized_swa_score_dot_order",
        "position": 12, "head": 25, "kv_head": 3, "source_token": 2, "kv_heads": 8,
    },
}


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


def source_specialized_vector_dot(left: np.ndarray, right: np.ndarray) -> np.float32:
    if left.size != right.size:
        raise ValueError("specialized vector-dot operands disagree")
    accumulators = np.zeros((8, 4), dtype=np.float32)
    complete_blocks = left.size // 32 * 32
    for block in range(0, complete_blocks, 32):
        for register in range(8):
            for lane in range(4):
                index = block + register * 4 + lane
                accumulators[register, lane] = np.float32(
                    accumulators[register, lane] + np.float32(left[index] * right[index]))
    for offset in (4, 2, 1):
        for register in range(offset):
            accumulators[register] = np.array([
                np.float32(accumulators[register, lane] + accumulators[offset + register, lane])
                for lane in range(4)], dtype=np.float32)
    reduced = np.float32(
        np.float32(accumulators[0, 0] + accumulators[0, 1])
        + np.float32(accumulators[0, 2] + accumulators[0, 3]))
    complete_vectors = left.size // 8 * 8
    tail = np.zeros(4, dtype=np.float32)
    for block in range(complete_blocks, complete_vectors, 8):
        for lane in range(4):
            tail[lane] = np.float32(tail[lane] + np.float32(left[block + lane] * right[block + lane]))
            tail[lane] = np.float32(tail[lane] + np.float32(left[block + 4 + lane] * right[block + 4 + lane]))
    reduced = np.float32(reduced + np.float32(
        np.float32(tail[0] + tail[1]) + np.float32(tail[2] + tail[3])))
    for index in range(complete_vectors, left.size):
        reduced = np.float32(reduced + np.float32(left[index] * right[index]))
    return reduced


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
    parser.add_argument("--case", choices=tuple(CASES), default="four_lane")
    args = parser.parse_args()
    case = CASES[args.case]

    if sha256(args.oracle) != case["manifest_sha256"]:
        raise ValueError("dot fixture oracle manifest hash mismatch")
    manifest = json.loads(args.oracle.read_text())
    if manifest.get("semantic") != case["oracle_semantic"]:
        raise ValueError("dot fixture oracle semantic mismatch")
    root = args.oracle.parent
    query = checked_capture(root, manifest, "query", [27, 64, WIDTH])
    key = checked_capture(root, manifest, "key", [27, case["kv_heads"], WIDTH])

    position = case["position"]; head = case["head"]
    kv_head = case["kv_head"]; source_token = case["source_token"]
    query_values = query[position, head]
    key_values = key[source_token, kv_head]
    query_tensor = torch.from_numpy(query_values.copy()).to(torch.bfloat16)
    key_tensor = torch.from_numpy(key_values.copy()).to(torch.bfloat16)
    dot = query_tensor @ key_tensor
    dot_bf16 = dot.to(torch.bfloat16)
    scale = torch.tensor(1.0 / math.sqrt(WIDTH), dtype=torch.float32)
    scaled = (dot_bf16 * scale).to(torch.bfloat16)

    row = []
    for token in range(position + 1):
        row.append((query_tensor @ torch.from_numpy(key[token, kv_head].copy()).to(torch.bfloat16)).to(torch.bfloat16) * scale)
    sinks_shape = manifest["captures"]["sinks"]["shape"]
    sinks = checked_capture(root, manifest, "sinks", sinks_shape)
    row_tensor = torch.stack(row).to(torch.bfloat16)
    if sinks.size:
        row_tensor = torch.cat((row_tensor, torch.tensor([sinks[head]], dtype=torch.bfloat16)))
    maximum = row_tensor.max()
    centered = (scaled - maximum).to(torch.bfloat16)

    output = {
        "schema_version": 1,
        "semantic": case["fixture_semantic"],
        "torch_version": torch.__version__,
        "torch_commit": "cf30153c4c131c8164ee7798e5022d810682e2cb",
        "source_manifest_sha256": case["manifest_sha256"],
        "query_capture_sha256": manifest["captures"]["query"]["sha256"],
        "key_capture_sha256": manifest["captures"]["key"]["sha256"],
        "position": position,
        "head": head,
        "kv_head": kv_head,
        "source_token": source_token,
        "width": WIDTH,
        "query_bf16_u16": query_tensor.view(torch.uint16).tolist(),
        "key_bf16_u16": key_tensor.view(torch.uint16).tolist(),
        "dot_bf16_u16": int(dot_bf16.view(torch.uint16)),
        "scale_f32_u32": f32_bits(scale),
        "scaled_score_bf16_u16": int(scaled.view(torch.uint16)),
        "row_maximum_bf16_u16": int(maximum.view(torch.uint16)),
        "centered_score_bf16_u16": int(centered.view(torch.uint16)),
    }
    if args.case == "four_lane":
        output["source_four_lane_dot_f32_u32"] = f32_bits(source_four_lane_dot(query_values, key_values))
        output["forward_dot_f32_u32"] = f32_bits(forward_dot(query_values, key_values))
    else:
        output["source_specialized_vector_dot_f32_u32"] = f32_bits(
            source_specialized_vector_dot(query_values, key_values))
        output["four_lane_dot_f32_u32"] = f32_bits(source_four_lane_dot(query_values, key_values))
    atomic_write_new(args.output, canonical_json(output))


if __name__ == "__main__":
    main()
