#!/usr/bin/env python3
"""Generate an independently decodable MLX affine-INT4 real-row fixture."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlx.core as mx
import numpy as np
from safetensors import safe_open
import torch

try:
    from tools.generate_real_fp8_fixture import REVISION, SCALE, SOURCE_SHA256, WEIGHT
except ModuleNotFoundError:
    from generate_real_fp8_fixture import REVISION, SCALE, SOURCE_SHA256, WEIGHT


GROUP_SIZE = 128
BITS = 4


def generate(
    path: Path,
    weight_name: str = WEIGHT,
    scale_name: str = SCALE,
    source_sha256: str = SOURCE_SHA256,
) -> dict:
    rows = 4
    with safe_open(path, framework="pt", device="cpu") as tensors:
        source = tensors.get_slice(weight_name)[:rows, :].float()
        source_scales = tensors.get_slice(scale_name)[0, :].float().repeat_interleave(GROUP_SIZE)
    source = source * source_scales
    weight = mx.array(source.to(torch.float16).numpy())
    packed, scales, biases = mx.quantize(weight, group_size=GROUP_SIZE, bits=BITS, mode="affine")
    mx.eval(packed, scales, biases)
    packed_np = np.array(packed)
    scales_np = np.array(scales)
    biases_np = np.array(biases)
    input_np = np.array(
        [[np.sin(index / 17.0) * 0.01 for index in range(source.shape[1])]],
        dtype=np.float16,
    )
    output = mx.quantized_matmul(
        mx.array(input_np),
        packed,
        scales,
        biases,
        group_size=GROUP_SIZE,
        bits=BITS,
        mode="affine",
    )
    mx.eval(output)

    quantized = np.empty(source.shape, dtype=np.float32)
    words_per_group = GROUP_SIZE * BITS // 32
    for row in range(rows):
        for group in range(source.shape[1] // GROUP_SIZE):
            for within in range(GROUP_SIZE):
                word = int(packed_np[row, group * words_per_group + within // 8])
                code = (word >> ((within % 8) * BITS)) & 0x0F
                quantized[row, group * GROUP_SIZE + within] = (
                    code * float(scales_np[row, group]) + float(biases_np[row, group])
                )
    manual = quantized @ input_np[0].astype(np.float32)
    source_output = source.numpy() @ input_np[0].astype(np.float32)
    return {
        "schema_version": 1,
        "semantic": "mlx_affine_int4_group128_gemv_slice",
        "exactness": "L3_bounded_approximation",
        "source_revision": REVISION,
        "source_file": path.name,
        "source_sha256": source_sha256,
        "tensor": weight_name,
        "source_scale_tensor": scale_name,
        "mlx_version": "0.31.2",
        "rows": rows,
        "columns": source.shape[1],
        "group_size": GROUP_SIZE,
        "bits": BITS,
        "mode": "affine",
        "packed_u32": packed_np.tolist(),
        "scale_f16": scales_np.astype(np.float32).tolist(),
        "bias_f16": biases_np.astype(np.float32).tolist(),
        "input_f16": input_np[0].astype(np.float32).tolist(),
        "expected_manual_f32": manual.tolist(),
        "expected_mlx_f16": np.array(output)[0].astype(np.float32).tolist(),
        "source_fp8_expected_f32": source_output.tolist(),
        "manual_minus_source_output": (manual - source_output).tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--source-file", default="model_mtp.safetensors")
    parser.add_argument("--weight", default=WEIGHT)
    parser.add_argument("--scale", default=SCALE)
    parser.add_argument("--source-sha256", default=SOURCE_SHA256)
    arguments = parser.parse_args()
    value = generate(
        arguments.checkpoint / arguments.source_file,
        arguments.weight,
        arguments.scale,
        arguments.source_sha256,
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(value, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
