#!/usr/bin/env python3
"""Extract PW-0087's discriminating BF16 attention value GEMM fixture."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import struct

import numpy as np
import torch

try:
    from tools.generate_pytorch_bf16_dot_fixture import (
        source_four_lane_dot, source_specialized_vector_dot,
    )
    from tools.openrouter_reference import atomic_write_new, canonical_json
except ModuleNotFoundError:
    from generate_pytorch_bf16_dot_fixture import source_four_lane_dot, source_specialized_vector_dot
    from openrouter_reference import atomic_write_new, canonical_json


MANIFEST_SHA256 = "dcf92f0c37e825766984f524b2338701adf28dd528ffafd374d59e6f20673fc1"
POSITION = 24
HEAD = 49
KV_HEAD = 6
VALUE_DIMENSION = 2


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def f32_bits(value: np.float32) -> int:
    return struct.unpack("<I", struct.pack("<f", float(value)))[0]


def checked_capture(root: Path, manifest: dict, name: str, shape: list[int]) -> np.ndarray:
    record = manifest["captures"][name]
    path = root / record["file"]
    if (record.get("shape") != shape or record.get("dtype") != "BF16_widened_F32"
            or sha256(path) != record.get("sha256")):
        raise ValueError(f"{name} capture authority mismatch")
    return np.fromfile(path, dtype="<f4").reshape(shape)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oracle", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if sha256(args.oracle) != MANIFEST_SHA256:
        raise ValueError("attention-value GEMM oracle manifest hash mismatch")
    manifest = json.loads(args.oracle.read_text())
    if manifest.get("semantic") != "mimo_real_layer34_complete_oracle":
        raise ValueError("attention-value GEMM oracle semantic mismatch")
    root = args.oracle.parent
    probabilities = checked_capture(root, manifest, "attention_probabilities", [25920])
    values = checked_capture(root, manifest, "value", [27, 8, 128])
    offset = sum((position + 2) * 64 for position in range(POSITION))
    offset += HEAD * (POSITION + 2)
    probability = probabilities[offset:offset + POSITION + 2][:-1].copy()
    value_matrix = values[:POSITION + 1, KV_HEAD].copy()
    value = value_matrix[:, VALUE_DIMENSION].copy()
    probability_tensor = torch.from_numpy(probability).to(torch.bfloat16)
    matrix_tensor = torch.from_numpy(value_matrix).to(torch.bfloat16)
    matrix_result = probability_tensor @ matrix_tensor
    generic = source_four_lane_dot(probability, value)
    specialized = source_specialized_vector_dot(probability, value)
    generic_bf16 = torch.tensor(float(generic)).to(torch.bfloat16)
    specialized_bf16 = torch.tensor(float(specialized)).to(torch.bfloat16)
    if generic_bf16.view(torch.uint16) != matrix_result[VALUE_DIMENSION].view(torch.uint16):
        raise ValueError("generic four-lane reduction does not reproduce matrix result")
    if specialized_bf16.view(torch.uint16) == matrix_result[VALUE_DIMENSION].view(torch.uint16):
        raise ValueError("specialized reduction no longer discriminates boundary")
    output = {
        "schema_version": 1,
        "semantic": "pytorch_aarch64_bf16_attention_value_gemm_order",
        "torch_version": torch.__version__,
        "torch_commit": "cf30153c4c131c8164ee7798e5022d810682e2cb",
        "source_manifest_sha256": MANIFEST_SHA256,
        "probability_capture_sha256": manifest["captures"]["attention_probabilities"]["sha256"],
        "value_capture_sha256": manifest["captures"]["value"]["sha256"],
        "position": POSITION,
        "head": HEAD,
        "kv_head": KV_HEAD,
        "value_dimension": VALUE_DIMENSION,
        "width": POSITION + 1,
        "probability_bf16_u16": probability_tensor.view(torch.uint16).tolist(),
        "value_bf16_u16": torch.from_numpy(value).to(torch.bfloat16).view(torch.uint16).tolist(),
        "source_generic_four_lane_f32_u32": f32_bits(generic),
        "source_specialized_vector_f32_u32": f32_bits(specialized),
        "matrix_result_bf16_u16": int(matrix_result[VALUE_DIMENSION].view(torch.uint16)),
    }
    atomic_write_new(args.output, canonical_json(output))


if __name__ == "__main__":
    main()
