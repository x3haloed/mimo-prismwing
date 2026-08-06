#!/usr/bin/env python3
"""Extract PW-0084's hash-bound ARM horizontal softmax fixture."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

try:
    from tools.openrouter_reference import atomic_write_new, canonical_json
except ModuleNotFoundError:
    from openrouter_reference import atomic_write_new, canonical_json


MANIFEST_SHA256 = "94c7411a5879f4ade7a700a4309d3a2b48354cc67409701e39003391cadde736"
POSITION = 22
HEAD = 15
SOURCE = 20
WIDTH = POSITION + 1


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bits(values: np.ndarray | torch.Tensor) -> list[int]:
    if isinstance(values, torch.Tensor):
        return values.contiguous().view(torch.uint32).tolist()
    return values.astype("<f4", copy=False).view("<u4").tolist()


def horizontal_denominator(exponentials: np.ndarray, cross: bool) -> np.float32:
    lanes = exponentials[:4].copy()
    full = exponentials.size - exponentials.size % 4
    for offset in range(4, full, 4):
        lanes = np.float32(lanes + exponentials[offset:offset + 4])
    for lane, value in enumerate(exponentials[full:]):
        lanes[lane] = np.float32(lanes[lane] + value)
    if cross:
        return np.float32(np.float32(lanes[0] + lanes[2]) + np.float32(lanes[1] + lanes[3]))
    return np.float32(np.float32(lanes[0] + lanes[1]) + np.float32(lanes[2] + lanes[3]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oracle", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if sha256(args.oracle) != MANIFEST_SHA256:
        raise ValueError("softmax fixture oracle manifest hash mismatch")
    manifest = json.loads(args.oracle.read_text())
    if manifest.get("semantic") != "mimo_real_layer29_complete_oracle":
        raise ValueError("softmax fixture oracle semantic mismatch")
    record = manifest["captures"]["attention_scores"]
    path = args.oracle.parent / record["file"]
    if (record.get("shape") != [24192]
            or record.get("dtype") != "BF16_widened_F32"
            or sha256(path) != record.get("sha256")):
        raise ValueError("softmax fixture capture mismatch")
    offset = sum((position + 1) * 64 for position in range(POSITION)) + HEAD * WIDTH
    scores = np.fromfile(path, dtype="<f4")[offset:offset + WIDTH].copy()
    tensor = torch.from_numpy(scores)
    exponentials = torch.exp(tensor)
    probabilities = torch.softmax(tensor, dim=-1, dtype=torch.float32)
    exponential_values = exponentials.numpy()
    cross = horizontal_denominator(exponential_values, True)
    adjacent = horizontal_denominator(exponential_values, False)
    modeled = np.float32(exponential_values * np.float32(np.float32(1.0) / cross))
    if bits(modeled) != bits(probabilities):
        raise ValueError("ARM cross-lane reduction does not reproduce PyTorch F32 softmax")
    adjacent_modeled = np.float32(exponential_values * np.float32(np.float32(1.0) / adjacent))
    expected_bf16 = probabilities.to(torch.bfloat16).view(torch.uint16).tolist()
    adjacent_bf16 = torch.from_numpy(adjacent_modeled).to(torch.bfloat16).view(torch.uint16).tolist()
    if adjacent_bf16[SOURCE] == expected_bf16[SOURCE]:
        raise ValueError("adjacent horizontal reduction no longer discriminates boundary")
    output = {
        "schema_version": 1,
        "semantic": "pytorch_aarch64_f32_softmax_horizontal_order",
        "torch_version": torch.__version__,
        "torch_commit": "cf30153c4c131c8164ee7798e5022d810682e2cb",
        "source_manifest_sha256": MANIFEST_SHA256,
        "attention_scores_capture_sha256": record["sha256"],
        "position": POSITION,
        "head": HEAD,
        "discriminating_source": SOURCE,
        "width": WIDTH,
        "score_bf16_u16": tensor.to(torch.bfloat16).view(torch.uint16).tolist(),
        "exponential_f32_u32": bits(exponentials),
        "probability_f32_u32": bits(probabilities),
        "probability_bf16_u16": expected_bf16,
        "cross_denominator_f32_u32": bits(np.asarray([cross]))[0],
        "adjacent_denominator_f32_u32": bits(np.asarray([adjacent]))[0],
    }
    atomic_write_new(args.output, canonical_json(output))


if __name__ == "__main__":
    main()
