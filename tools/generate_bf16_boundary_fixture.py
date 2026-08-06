#!/usr/bin/env python3
"""Generate the PW-0054 independent BF16 conversion fixture."""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import torch

try:
    from tools.openrouter_reference import atomic_write_new, canonical_json
except ModuleNotFoundError:
    from openrouter_reference import atomic_write_new, canonical_json


def f32_from_bits(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", bits))[0]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    arguments = parser.parse_args()

    generator = torch.Generator().manual_seed(260054)
    random_bits = torch.randint(0, 2**32, (512,), generator=generator, dtype=torch.int64)
    special_bits = torch.tensor(
        [
            0x00000000, 0x80000000, 0x00000001, 0x007FFFFF,
            0x00800000, 0x3F800000, 0xBF800000, 0x7F7FFFFF,
            0xFF7FFFFF, 0x7F800000, 0xFF800000, 0x7FC00001,
            0xFFC00001, 0x3F7F7FFF, 0x3F7F8000, 0x3F7F8001,
            0x3F807FFF, 0x3F808000, 0x3F808001, 0xBF7F8000,
        ],
        dtype=torch.int64,
    )
    bits = torch.cat([special_bits, random_bits])
    values = torch.tensor([f32_from_bits(int(value)) for value in bits], dtype=torch.float32)
    bf16 = values.to(torch.bfloat16)
    widened = bf16.to(torch.float32)

    attention_cases = []
    for with_sink in (False, True):
        query = torch.tensor([0.5, -1.25, 2.0, 0.125], dtype=torch.bfloat16)
        keys = torch.tensor(
            [[1.0, -0.5, 0.25, 2.0], [-0.75, 1.5, 0.5, -1.0]],
            dtype=torch.bfloat16,
        )
        value_rows = torch.tensor(
            [[0.25, -2.0, 1.5], [1.25, 0.5, -0.75]], dtype=torch.bfloat16
        )
        scale = 0.5
        scores = (query @ keys.T) * scale
        sink = torch.tensor(-0.375, dtype=torch.bfloat16) if with_sink else None
        if sink is not None:
            scores = torch.cat([scores, sink.reshape(1)])
        scores = scores - scores.max()
        probabilities = torch.softmax(scores, dim=-1, dtype=torch.float32).to(
            torch.bfloat16
        )
        if sink is not None:
            probabilities = probabilities[:-1]
        output = probabilities @ value_rows
        attention_cases.append(
            {
                "query": query.float().tolist(),
                "keys": keys.float().tolist(),
                "values": value_rows.float().tolist(),
                "scale": scale,
                "sink": None if sink is None else sink.float().item(),
                "expected_bf16_u16": output.view(torch.uint16).tolist(),
            }
        )
    payload = {
        "schema_version": 1,
        "semantic": "f32_to_bf16_rne",
        "torch_version": torch.__version__,
        "seed": 260054,
        "input_f32_bits": values.view(torch.int32).tolist(),
        "bf16_u16": bf16.view(torch.uint16).tolist(),
        "widened_f32_bits": widened.view(torch.int32).tolist(),
        "attention_cases": attention_cases,
    }
    atomic_write_new(arguments.output, canonical_json(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
