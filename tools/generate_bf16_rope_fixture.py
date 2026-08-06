#!/usr/bin/env python3
"""Generate the PW-0055 independent BF16 text-RoPE fixture."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

try:
    from tools.openrouter_reference import atomic_write_new, canonical_json
except ModuleNotFoundError:
    from openrouter_reference import atomic_write_new, canonical_json


def rotate_half(values: torch.Tensor) -> torch.Tensor:
    half = values.shape[-1] // 2
    return torch.cat((-values[..., half:], values[..., :half]), dim=-1)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    arguments = parser.parse_args()

    generator = torch.Generator().manual_seed(260055)
    cases = []
    for theta in (10_000.0, 10_000_000.0):
        inv_freq = 1.0 / (
            theta ** (torch.arange(0, 64, 2, dtype=torch.float32) / 64)
        )
        for position in (0, 1, 7, 27):
            values = torch.randn((2, 192), generator=generator).to(torch.bfloat16)
            freqs = inv_freq * float(position)
            cos = torch.cat((freqs, freqs)).cos().to(torch.bfloat16)
            sin = torch.cat((freqs, freqs)).sin().to(torch.bfloat16)
            rotated = (values[:, :64] * cos) + (rotate_half(values[:, :64]) * sin)
            output = torch.cat((rotated, values[:, 64:]), dim=-1)
            cases.append(
                {
                    "theta": theta,
                    "position": position,
                    "heads": 2,
                    "head_dim": 192,
                    "rope_dim": 64,
                    "input_bf16_u16": values.contiguous().view(torch.uint16).tolist(),
                    "output_bf16_u16": output.contiguous().view(torch.uint16).tolist(),
                }
            )
    payload = {
        "schema_version": 1,
        "semantic": "mimo_text_rope_bf16_operation_staging",
        "torch_version": torch.__version__,
        "seed": 260055,
        "cases": cases,
    }
    atomic_write_new(arguments.output, canonical_json(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
