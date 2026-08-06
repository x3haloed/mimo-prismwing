#!/usr/bin/env python3
"""Generate the PW-0053 independent dynamic E4M3FN activation fixture."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

try:
    from tools.openrouter_reference import atomic_write_new, canonical_json
except ModuleNotFoundError:
    from openrouter_reference import atomic_write_new, canonical_json


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    arguments = parser.parse_args()

    generator = torch.Generator().manual_seed(260053)
    values = torch.randn((2, 256), generator=generator, dtype=torch.float32) * 3.25
    values[0, :12] = torch.tensor(
        [0.0, -0.0, 1e-12, -1e-12, 0.5, -0.5, 1.0, -1.0, 447.0, -447.0, 500.0, -500.0],
        dtype=torch.float32,
    )
    grouped = values.reshape(2, 2, 128)
    scales = grouped.abs().amax(dim=-1).clamp(min=1e-10) / 448.0
    quantized = torch.clamp(grouped / scales.unsqueeze(-1), -448.0, 448.0).to(
        torch.float8_e4m3fn
    )
    dequantized = quantized.float() * scales.unsqueeze(-1)
    payload = {
        "schema_version": 1,
        "semantic": "dynamic_fp8_e4m3fn_per_token_group_128",
        "torch_version": torch.__version__,
        "seed": 260053,
        "rows": 2,
        "columns": 256,
        "group_size": 128,
        "epsilon": 1e-10,
        "fp8_max": 448.0,
        "input_f32_bits": values.view(torch.int32).tolist(),
        "scale_f32_bits": scales.contiguous().view(torch.int32).tolist(),
        "encoded_u8": quantized.contiguous().view(torch.uint8).tolist(),
        "dequantized_f32_bits": dequantized.contiguous().view(torch.int32).tolist(),
    }
    atomic_write_new(arguments.output, canonical_json(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
