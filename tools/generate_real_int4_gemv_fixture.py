#!/usr/bin/env python3
"""Quantize four real MiMo rows into a deterministic signed-INT4 fixture."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from safetensors import safe_open
import torch

try:
    from tools.generate_real_fp8_fixture import REVISION, SCALE, SOURCE_SHA256, WEIGHT
except ModuleNotFoundError:
    from generate_real_fp8_fixture import REVISION, SCALE, SOURCE_SHA256, WEIGHT


BLOCK_COLUMNS = 128


def quantize_signed_int4(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if values.ndim != 2 or values.shape[1] % BLOCK_COLUMNS != 0:
        raise ValueError("matrix width must be divisible by the INT4 block width")
    blocks = values.reshape(values.shape[0], -1, BLOCK_COLUMNS)
    scales = blocks.abs().amax(dim=2) / 7.0
    if torch.any(scales == 0):
        raise ValueError("zero-valued block needs an explicit zero-scale convention")
    quantized = torch.round(blocks / scales.unsqueeze(2)).clamp(-7, 7).to(torch.int8)
    flat = quantized.reshape(values.shape)
    low = torch.bitwise_and(flat[:, 0::2], 0x0F).to(torch.uint8)
    high = torch.bitwise_left_shift(
        torch.bitwise_and(flat[:, 1::2], 0x0F).to(torch.uint8), 4
    )
    return torch.bitwise_or(low, high), scales.to(torch.float32)


def generate(path: Path) -> dict:
    rows = 4
    with safe_open(path, framework="pt", device="cpu") as tensors:
        source = tensors.get_slice(WEIGHT)[:rows, :].float()
        source_scales = tensors.get_slice(SCALE)[0, :].float().repeat_interleave(BLOCK_COLUMNS)
    dequantized_source = source * source_scales
    packed, scales = quantize_signed_int4(dequantized_source)
    signed = packed.unsqueeze(2).expand(-1, -1, 2).clone()
    signed[:, :, 0] = packed & 0x0F
    signed[:, :, 1] = packed >> 4
    signed = signed.to(torch.int8)
    signed[signed >= 8] -= 16
    restored = (signed.reshape(rows, -1).reshape(rows, -1, BLOCK_COLUMNS) * scales.unsqueeze(2)).reshape(rows, -1)
    columns = restored.shape[1]
    input_values = torch.tensor(
        [float(torch.sin(torch.tensor(index / 17.0)) * 0.01) for index in range(columns)],
        dtype=torch.float32,
    )
    expected = (restored @ input_values).tolist()
    source_expected = (dequantized_source @ input_values).tolist()
    output_error = [actual - reference for actual, reference in zip(expected, source_expected)]
    return {
        "schema_version": 1,
        "semantic": "mimo_signed_int4_group128_gemv_slice",
        "exactness": "L3_bounded_approximation",
        "source_revision": REVISION,
        "source_file": "model_mtp.safetensors",
        "source_sha256": SOURCE_SHA256,
        "tensor": WEIGHT,
        "source_scale_tensor": SCALE,
        "rows": rows,
        "columns": columns,
        "block_columns": BLOCK_COLUMNS,
        "quantized_range": [-7, 7],
        "nibble_order": "low_then_high_twos_complement",
        "rounding": "nearest_even",
        "packed_u8": packed.tolist(),
        "scale": scales.tolist(),
        "input": input_values.tolist(),
        "expected_f32": expected,
        "source_fp8_expected_f32": source_expected,
        "int4_minus_source_output": output_error,
        "max_abs_weight_error": float((restored - dequantized_source).abs().max()),
        "oracle": f"safetensors 0.7.0 / torch {torch.__version__} f32",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    arguments = parser.parse_args()
    value = generate(arguments.checkpoint / "model_mtp.safetensors")
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(value, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
