#!/usr/bin/env python3
"""Capture a small real MiMo FP8 block with independent decoded values."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import struct

from safetensors import safe_open


WEIGHT = "model.mtp.layers.0.mlp.gate_proj.weight"
SCALE = "model.mtp.layers.0.mlp.gate_proj.weight_scale_inv"
REVISION = "63651580ca774f8504f676040460aed3e1244ac1"
SOURCE_SHA256 = "a0e41a193b2762b0c83e577f83206d0777028de6916408c8c368730c0c9e2143"


def load_header(path):
    with path.open("rb") as handle:
        size = struct.unpack("<Q", handle.read(8))[0]
        return size, json.loads(handle.read(size))


def generate(path):
    header_size, header = load_header(path)
    weight = header[WEIGHT]
    scale = header[SCALE]
    if weight["dtype"] != "F8_E4M3" or weight["shape"] != [16384, 4096]:
        raise ValueError("unexpected real FP8 tensor layout")
    if scale["dtype"] != "F32" or scale["shape"] != [128, 32]:
        raise ValueError("unexpected real FP8 scale layout")
    payload = 8 + header_size
    raw = []
    with path.open("rb") as handle:
        for row in range(4):
            handle.seek(payload + weight["data_offsets"][0] + row * weight["shape"][1])
            raw.append(list(handle.read(8)))
        handle.seek(payload + scale["data_offsets"][0])
        scale_value = struct.unpack("<f", handle.read(4))[0]
    with safe_open(path, framework="pt", device="cpu") as tensors:
        decoded = tensors.get_slice(WEIGHT)[:4, :8].float().tolist()
        library_scale = float(tensors.get_slice(SCALE)[0, 0])
    if scale_value != library_scale:
        raise ValueError("raw and library scale reads disagree")
    return {
        "schema_version": 1,
        "semantic": "safetensors_f8_e4m3fn_block_dequant",
        "source_revision": REVISION,
        "source_file": "model_mtp.safetensors",
        "source_sha256": SOURCE_SHA256,
        "tensor": WEIGHT,
        "scale_tensor": SCALE,
        "weight_block_size": [128, 128],
        "row_start": 0,
        "column_start": 0,
        "raw_u8": raw,
        "decoded_fp8": decoded,
        "scale_inv": scale_value,
        "dequantized": [[value * scale_value for value in row] for row in decoded],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    arguments = parser.parse_args()
    value = generate(arguments.checkpoint / "model_mtp.safetensors")
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
