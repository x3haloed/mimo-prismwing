#!/usr/bin/env python3
"""Freeze the real normalized input and MLX QKV oracle for native Rust GEMV."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import struct

import mlx.core as mx
import numpy as np
from safetensors import safe_open

try:
    from tools.generate_mtp_attention_fixture import (
        EXPECTED,
        NORM,
        QKV,
        SCALE,
        SOURCE_SHA256,
        sha256_file,
    )
    from tools.generate_mtp_decoder_block_fixture import install_scales, rms_norm
    from tools.openrouter_reference import atomic_write_new, canonical_json
except ModuleNotFoundError:
    from generate_mtp_attention_fixture import (
        EXPECTED,
        NORM,
        QKV,
        SCALE,
        SOURCE_SHA256,
        sha256_file,
    )
    from generate_mtp_decoder_block_fixture import install_scales, rms_norm
    from openrouter_reference import atomic_write_new, canonical_json


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


def generate(checkpoint: Path, input_path: Path, expected_path: Path) -> dict:
    if sha256_file(checkpoint) != SOURCE_SHA256:
        raise ValueError("MTP source SHA-256 mismatch")
    with checkpoint.open("rb") as source:
        header_size = struct.unpack("<Q", source.read(8))[0]
        header = json.loads(source.read(header_size))
    for name in (NORM, QKV, SCALE):
        if header.get(name) != EXPECTED[name]:
            raise ValueError(f"MTP metadata mismatch: {name}")
    with safe_open(checkpoint, framework="pt", device="cpu") as tensors:
        norm = tensors.get_tensor(NORM).float().numpy()
        raw_qkv = tensors.get_tensor(QKV).float().numpy()
        scales = tensors.get_tensor(SCALE).float().numpy()
    start, end = EXPECTED[NORM]["data_offsets"]
    with checkpoint.open("rb") as source:
        source.seek(8 + header_size + start)
        raw_norm = np.frombuffer(source.read(end - start), dtype="<u2")
    manual_norm = (raw_norm.astype(np.uint32) << 16).view(np.float32)
    if not np.array_equal(manual_norm, norm):
        raise ValueError("BF16 norm raw decode mismatch")
    qkv_weight = install_scales(raw_qkv, scales)
    rng = np.random.default_rng(260026)
    hidden = rng.standard_normal((17, 4096), dtype=np.float32)
    normalized = rms_norm(hidden, norm)[0].astype("<f4")
    projected_mx = mx.matmul(mx.array(normalized.reshape(1, -1)), mx.array(qkv_weight).T)
    mx.eval(projected_mx)
    projected = np.array(projected_mx, copy=False).astype("<f4").reshape(-1)
    if normalized.shape != (4096,) or projected.shape != (14848,):
        raise ValueError("native FP8 fixture shape mismatch")
    if not np.isfinite(normalized).all() or not np.isfinite(projected).all():
        raise ValueError("native FP8 fixture is non-finite")
    input_bytes = normalized.tobytes()
    expected_bytes = projected.tobytes()
    write_new(input_path, input_bytes)
    write_new(expected_path, expected_bytes)
    rows = [0, 1, 12288, 13824]
    scalar = {
        str(row): float(np.dot(normalized.astype(np.float64), qkv_weight[row].astype(np.float64)))
        for row in rows
    }
    return {
        "schema_version": 1,
        "semantic": "mimo_mtp_layer0_normalized_hidden0_fused_qkv",
        "source_sha256": SOURCE_SHA256,
        "input_f32_count": 4096,
        "expected_f32_count": 14848,
        "input_sha256": digest(input_bytes),
        "expected_mlx_sha256": digest(expected_bytes),
        "sample_scalar_f64": scalar,
        "expected_mlx_first8": projected[:8].tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--expected", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    result = generate(args.checkpoint, args.input, args.expected)
    atomic_write_new(args.report, canonical_json(result))


if __name__ == "__main__":
    main()
