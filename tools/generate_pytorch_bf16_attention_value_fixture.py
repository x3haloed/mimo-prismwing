#!/usr/bin/env python3
"""Extract PW-0075's hash-bound BF16 attention-value dot fixture."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import struct

import numpy as np
import torch

try:
    from tools.generate_pytorch_bf16_dot_fixture import forward_dot, source_specialized_vector_dot
    from tools.openrouter_reference import atomic_write_new, canonical_json
except ModuleNotFoundError:
    from generate_pytorch_bf16_dot_fixture import forward_dot, source_specialized_vector_dot
    from openrouter_reference import atomic_write_new, canonical_json


MANIFEST_SHA256 = "294e25355d4cb6ca3dcdcb060e131e7599b6603987eaaf0664a39f95ff0ddf74"
POSITION = 24
HEAD = 4
KV_HEAD = 0
VALUE_DIMENSION = 52


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def f32_bits(value: torch.Tensor | np.float32) -> int:
    return struct.unpack("<I", struct.pack("<f", float(value)))[0]


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

    if sha256(args.oracle) != MANIFEST_SHA256:
        raise ValueError("attention-value fixture oracle manifest hash mismatch")
    manifest = json.loads(args.oracle.read_text())
    if manifest.get("semantic") != "mimo_real_layer13_complete_oracle":
        raise ValueError("attention-value fixture oracle semantic mismatch")
    root = args.oracle.parent
    probabilities = checked_capture(root, manifest, "attention_probabilities", [25920])
    values = checked_capture(root, manifest, "value", [27, 8, 128])

    offset = sum((position + 2) * 64 for position in range(POSITION))
    offset += HEAD * (POSITION + 2)
    probability = probabilities[offset:offset + POSITION + 2][:-1].copy()
    value = values[:POSITION + 1, KV_HEAD, VALUE_DIMENSION].copy()
    if probability.size != 25 or value.size != 25:
        raise ValueError("attention-value operand width mismatch")
    probability_tensor = torch.from_numpy(probability).to(torch.bfloat16)
    value_tensor = torch.from_numpy(value).to(torch.bfloat16)
    dot = probability_tensor @ value_tensor
    source = source_specialized_vector_dot(probability, value)
    forward = forward_dot(probability, value)
    if torch.tensor(float(source)).to(torch.bfloat16).view(torch.uint16) != dot.view(torch.uint16):
        raise ValueError("source topology does not reproduce PyTorch BF16 dot")
    if torch.tensor(float(forward)).to(torch.bfloat16).view(torch.uint16) == dot.view(torch.uint16):
        raise ValueError("forward accumulation no longer discriminates the boundary")

    output = {
        "schema_version": 1,
        "semantic": "pytorch_aarch64_bf16_attention_value_dot_order",
        "torch_version": torch.__version__,
        "torch_commit": "cf30153c4c131c8164ee7798e5022d810682e2cb",
        "source_manifest_sha256": MANIFEST_SHA256,
        "probability_capture_sha256": manifest["captures"]["attention_probabilities"]["sha256"],
        "value_capture_sha256": manifest["captures"]["value"]["sha256"],
        "position": POSITION,
        "head": HEAD,
        "kv_head": KV_HEAD,
        "value_dimension": VALUE_DIMENSION,
        "width": 25,
        "probability_bf16_u16": probability_tensor.view(torch.uint16).tolist(),
        "value_bf16_u16": value_tensor.view(torch.uint16).tolist(),
        "source_specialized_vector_dot_f32_u32": f32_bits(source),
        "forward_dot_f32_u32": f32_bits(forward),
        "dot_bf16_u16": int(dot.view(torch.uint16)),
    }
    atomic_write_new(args.output, canonical_json(output))


if __name__ == "__main__":
    main()
