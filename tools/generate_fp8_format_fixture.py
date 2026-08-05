#!/usr/bin/env python3
"""Capture PyTorch's f8_e4m3fn interpretation of all 256 byte patterns."""

import argparse
import json
from pathlib import Path
import struct

import torch


def generate():
    raw = torch.arange(256, dtype=torch.uint8)
    decoded = raw.view(torch.float8_e4m3fn).float().tolist()
    return {
        "schema_version": 1,
        "semantic": "f8_e4m3fn_exhaustive_f32_bits",
        "oracle": f"torch {torch.__version__}",
        "expected_f32_bits": [struct.unpack("<I", struct.pack("<f", value))[0] for value in decoded],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    arguments = parser.parse_args()
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(generate(), indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
