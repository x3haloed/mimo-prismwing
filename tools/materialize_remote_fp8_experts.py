#!/usr/bin/env python3
"""Materialize selected pinned FP8 expert tensors with sequential range reads."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import struct

import numpy as np

try:
    from tools.remote_fp8_symbol_census import load_header, retry_fetch_range
    from tools.remote_safetensors_audit import load_lock
except ModuleNotFoundError:
    from remote_fp8_symbol_census import load_header, retry_fetch_range
    from remote_safetensors_audit import load_lock


def materialize(lock: dict, index: dict, names: list[str]) -> tuple[dict[str, np.ndarray], int]:
    weight_map = index.get("weight_map", {})
    arrays: dict[str, np.ndarray] = {}
    headers = {}
    network_bytes = 0
    for name in names:
        shard = weight_map.get(name)
        if not isinstance(shard, str):
            raise ValueError(f"tensor absent from index: {name}")
        if shard not in headers:
            headers[shard] = load_header(lock, shard, retry_fetch_range)
            network_bytes += 8 + headers[shard][0]
        header_bytes, header = headers[shard]
        metadata = header.get(name)
        scale_name = f"{name}_scale_inv"
        scale_metadata = header.get(scale_name)
        if not isinstance(metadata, dict) or metadata.get("dtype") != "F8_E4M3":
            raise ValueError(f"unsupported weight tensor: {name}")
        rows, columns = metadata["shape"]
        if rows % 128 or columns % 128:
            raise ValueError(f"unaligned weight tensor: {name}")
        if (
            not isinstance(scale_metadata, dict)
            or scale_metadata.get("dtype") != "F32"
            or scale_metadata.get("shape") != [rows // 128, columns // 128]
        ):
            raise ValueError(f"invalid paired scale tensor: {scale_name}")
        start, end = metadata["data_offsets"]
        absolute = 8 + header_bytes + start
        payload = retry_fetch_range(lock["repository"], lock["revision"], shard, absolute, 8 + header_bytes + end - 1)
        network_bytes += len(payload)
        scale_start, scale_end = scale_metadata["data_offsets"]
        scale_absolute = 8 + header_bytes + scale_start
        scale_payload = retry_fetch_range(
            lock["repository"], lock["revision"], shard,
            scale_absolute, 8 + header_bytes + scale_end - 1,
        )
        network_bytes += len(scale_payload)
        codes = np.frombuffer(payload, dtype=np.uint8).reshape(rows, columns).copy()
        scales = np.frombuffer(scale_payload, dtype="<f4").reshape(rows // 128, columns // 128).copy()
        if np.any(~np.isfinite(scales)) or np.any(scales <= 0):
            raise ValueError(f"invalid source scales: {scale_name}")
        key = name.removeprefix("model.").replace(".", "__")
        arrays[f"{key}__codes"] = codes
        arrays[f"{key}__scales"] = scales
    return arrays, network_bytes


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", required=True, type=Path)
    parser.add_argument("--index", required=True, type=Path)
    parser.add_argument("--tensor", action="append", required=True)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    if arguments.output.exists():
        raise SystemExit(f"error: output exists: {arguments.output}")
    lock = load_lock(arguments.lock)
    index = json.loads(arguments.index.read_text(encoding="utf-8"))
    arrays, network_bytes = materialize(lock, index, arguments.tensor)
    arrays["network_bytes"] = np.asarray(network_bytes, dtype=np.int64)
    arrays["repository"] = np.asarray(lock["repository"])
    arrays["revision"] = np.asarray(lock["revision"])
    np.savez(arguments.output, **arrays)
    print(arguments.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
