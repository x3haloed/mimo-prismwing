#!/usr/bin/env python3
"""Freeze PW-0015 input and an independent source-FP8 complete-expert oracle."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path

import numpy as np
from safetensors import safe_open
import torch

try:
    from tools.openrouter_reference import atomic_write_new, canonical_json
except ModuleNotFoundError:
    from openrouter_reference import atomic_write_new, canonical_json


LAYER = 43
EXPERT = 32
PREFIX = f"model.layers.{LAYER}.mlp.experts.{EXPERT}"
NAMES = {kind: f"{PREFIX}.{kind}_proj.weight" for kind in ("gate", "up", "down")}


def write_new(path: Path, payload: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def dequantize(path: Path, name: str) -> torch.Tensor:
    with safe_open(path, framework="pt", device="cpu") as tensors:
        weight = tensors.get_tensor(name).float()
        scale = tensors.get_tensor(name + "_scale_inv").float()
    if weight.ndim != 2 or scale.ndim != 2:
        raise ValueError(f"{name}: rank mismatch")
    if weight.shape[0] % 128 or weight.shape[1] % 128:
        raise ValueError(f"{name}: dimensions are not 128-block aligned")
    expected_scale = (weight.shape[0] // 128, weight.shape[1] // 128)
    if tuple(scale.shape) != expected_scale:
        raise ValueError(f"{name}: scale-grid mismatch")
    return weight * scale.repeat_interleave(128, 0).repeat_interleave(128, 1)


def generate(
    gate_up: Path,
    down_shard: Path,
    input_path: Path,
    expected_path: Path,
    batch_size: int,
) -> dict:
    gate = dequantize(gate_up, NAMES["gate"])
    up = dequantize(gate_up, NAMES["up"])
    down = dequantize(down_shard, NAMES["down"])
    if tuple(gate.shape) != (2048, 4096) or up.shape != gate.shape:
        raise ValueError("gate/up production shape mismatch")
    if tuple(down.shape) != (4096, 2048):
        raise ValueError("down production shape mismatch")

    # Exact PW-0015 batch-one construction: values are rounded to FP16, then
    # promoted to F32 for the source-FP8 reference computation.
    input_f16 = np.array(
        [
            [np.sin((column + 19 * row) / 17.0) * 0.01 for column in range(4096)]
            for row in range(batch_size)
        ],
        dtype=np.float16,
    )
    inputs = torch.from_numpy(input_f16).float()
    gate_values = inputs @ gate.T
    up_values = inputs @ up.T
    hidden = torch.sigmoid(gate_values) * gate_values * up_values
    expected = (hidden @ down.T).numpy().astype("<f4").reshape(-1)
    input_f32 = input_f16.astype("<f4")
    if not np.isfinite(input_f32).all() or not np.isfinite(expected).all():
        raise ValueError("complete-expert fixture is non-finite")

    input_bytes = input_f32.tobytes()
    expected_bytes = expected.tobytes()
    write_new(input_path, input_bytes)
    write_new(expected_path, expected_bytes)
    return {
        "schema_version": 1,
        "semantic": "mimo_layer43_expert32_source_fp8_complete_expert",
        "layer": LAYER,
        "expert": EXPERT,
        "batch_size": batch_size,
        "source_files": [gate_up.name, down_shard.name],
        "tensors": NAMES,
        "input_f32_count": int(input_f32.size),
        "expected_f32_count": int(expected.size),
        "input_sha256": digest(input_bytes),
        "expected_sha256": digest(expected_bytes),
        "expected_first8": expected[:8].tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate-up", required=True, type=Path)
    parser.add_argument("--down-shard", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--expected", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, choices=(1, 8), default=1)
    args = parser.parse_args()
    report = generate(
        args.gate_up,
        args.down_shard,
        args.input,
        args.expected,
        args.batch_size,
    )
    atomic_write_new(args.report, canonical_json(report))


if __name__ == "__main__":
    main()
