#!/usr/bin/env python3
"""Generate PW-0057 PyTorch F32-softmax-to-BF16 cases."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

try:
    from tools.openrouter_reference import atomic_write_new, canonical_json
except ModuleNotFoundError:
    from openrouter_reference import atomic_write_new, canonical_json


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    generator = torch.Generator().manual_seed(260057)
    cases = []
    for length in (2, 7, 27):
        scores = (torch.randn(length, generator=generator) * 4.0).to(torch.bfloat16)
        if length == 27:
            scores[:5] = torch.tensor([20.0, 19.875, -20.0, 0.0, 0.0078125], dtype=torch.bfloat16)
        probabilities = torch.softmax(scores, dim=-1, dtype=torch.float32).to(torch.bfloat16)
        cases.append({"length": length, "score_bf16_u16": scores.view(torch.uint16).tolist(),
                      "probability_bf16_u16": probabilities.view(torch.uint16).tolist()})
    atomic_write_new(args.output, canonical_json({"schema_version": 1,
        "semantic": "pytorch_f32_softmax_to_bf16", "torch_version": torch.__version__,
        "seed": 260057, "cases": cases}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
