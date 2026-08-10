#!/usr/bin/env python3
"""Generate PW-0157's exact pinned-PyTorch tied top-k fixture."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import struct

import torch

try:
    from tools.openrouter_reference import atomic_write_new, canonical_json
except ModuleNotFoundError:
    from openrouter_reference import atomic_write_new, canonical_json


TORCH_VERSION = "2.13.0"
TORCH_COMMIT = "cf30153c4c131c8164ee7798e5022d810682e2cb"
TOPK_IMPL_SHA256 = "1ff24ba878ccb3816511ba34609d7247225342c6aa61740b51917c8ca79407ab"
WIDTH = 256
TOP_K = 8


def f32_bits(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", value))[0]


def fixture_rows() -> dict[str, list[float]]:
    boundary_pair = [-1000.0 - index for index in range(WIDTH)]
    for index in range(7):
        boundary_pair[index] = 20.0 - index
    boundary_pair[7] = 7.0
    boundary_pair[200] = 7.0

    multiway_boundary = [-1000.0 - index for index in range(WIDTH)]
    for index in range(5):
        multiway_boundary[index] = 20.0 - index
    for index in (5, 77, 122, 200):
        multiway_boundary[index] = 7.0

    all_equal = [1.25] * WIDTH

    repeated_plateaus = [-float(index // 12) for index in range(WIDTH)]

    signed_zero = [-1000.0 - index for index in range(WIDTH)]
    for index in range(6):
        signed_zero[index] = 20.0 - index
    for index in range(6, WIDTH):
        signed_zero[index] = -0.0 if index % 2 else 0.0

    return {
        "boundary_pair": boundary_pair,
        "multiway_boundary": multiway_boundary,
        "all_equal": all_equal,
        "repeated_plateaus": repeated_plateaus,
        "signed_zero": signed_zero,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--topk-header", required=True, type=Path)
    arguments = parser.parse_args()

    if torch.__version__ != TORCH_VERSION or torch.version.git_version != TORCH_COMMIT:
        raise ValueError("PW-0157 requires the pinned PyTorch build")
    header_sha256 = hashlib.sha256(arguments.topk_header.read_bytes()).hexdigest()
    if header_sha256 != TOPK_IMPL_SHA256:
        raise ValueError("pinned TopKImpl.h hash mismatch")

    cases = []
    for name, values in fixture_rows().items():
        tensor = torch.tensor(values, dtype=torch.float32)
        selected = torch.topk(tensor, TOP_K, sorted=False).indices.tolist()
        chosen = tensor[selected]
        rejected = tensor[[index for index in range(WIDTH) if index not in selected]]
        if float(chosen.min()) != float(rejected.max()):
            raise ValueError(f"{name} does not contain a boundary tie")
        cases.append({
            "name": name,
            "corrected_f32_u32": [f32_bits(value) for value in values],
            "selected_experts": selected,
            "boundary_f32_u32": f32_bits(float(chosen.min())),
        })

    output = {
        "schema_version": 1,
        "semantic": "pinned_pytorch_cpu_unsorted_topk_tied_rows",
        "torch_version": TORCH_VERSION,
        "torch_commit": TORCH_COMMIT,
        "topk_impl_sha256": header_sha256,
        "cpu_capability": "DEFAULT",
        "standard_library": "libc++",
        "width": WIDTH,
        "top_k": TOP_K,
        "cases": cases,
    }
    atomic_write_new(arguments.output, canonical_json(output))


if __name__ == "__main__":
    main()
