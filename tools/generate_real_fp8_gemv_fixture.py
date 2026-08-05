#!/usr/bin/env python3
"""Capture four production-width rows from a real MiMo block-FP8 matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import struct

from safetensors import safe_open
import torch

try:
    from tools.generate_real_fp8_fixture import (
        REVISION,
        SCALE,
        SOURCE_SHA256,
        WEIGHT,
        load_header,
    )
except ModuleNotFoundError:
    from generate_real_fp8_fixture import REVISION, SCALE, SOURCE_SHA256, WEIGHT, load_header


def generate(path: Path):
    header_size, header = load_header(path)
    weight = header[WEIGHT]
    scale = header[SCALE]
    rows = 4
    columns = weight["shape"][1]
    payload = 8 + header_size
    raw_rows = []
    with path.open("rb") as handle:
        for row in range(rows):
            handle.seek(payload + weight["data_offsets"][0] + row * columns)
            data = handle.read(columns)
            if len(data) != columns:
                raise ValueError("short real weight read")
            raw_rows.append(list(data))
        handle.seek(payload + scale["data_offsets"][0])
        scale_values = list(struct.unpack("<32f", handle.read(32 * 4)))
    input_values = [float(torch.sin(torch.tensor(index / 17.0)) * 0.01) for index in range(columns)]
    with safe_open(path, framework="pt", device="cpu") as tensors:
        decoded = tensors.get_slice(WEIGHT)[:rows, :].float()
        library_scales = tensors.get_slice(SCALE)[0, :].float()
    if library_scales.tolist() != scale_values:
        raise ValueError("raw and library scale vectors disagree")
    expanded_scales = library_scales.repeat_interleave(128)
    input_tensor = torch.tensor(input_values, dtype=torch.float32)
    expected = ((decoded * expanded_scales) @ input_tensor).tolist()
    return {
        "schema_version": 1,
        "semantic": "mimo_block_fp8_gemv_slice",
        "source_revision": REVISION,
        "source_file": "model_mtp.safetensors",
        "source_sha256": SOURCE_SHA256,
        "tensor": WEIGHT,
        "scale_tensor": SCALE,
        "rows": rows,
        "columns": columns,
        "block_columns": 128,
        "raw_u8": raw_rows,
        "scale_inv": scale_values,
        "input": input_values,
        "expected_f32": expected,
        "oracle": f"safetensors 0.7.0 / torch {torch.__version__} f32 matmul",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    arguments = parser.parse_args()
    value = generate(arguments.checkpoint / "model_mtp.safetensors")
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(value, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
