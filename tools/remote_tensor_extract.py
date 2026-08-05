#!/usr/bin/env python3
"""Losslessly materialize selected tensors from a pinned remote safetensors file."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
from typing import Any, Callable

try:
    from tools.openrouter_reference import atomic_write_new, canonical_json
    from tools.remote_safetensors_audit import fetch_range, load_lock
except ModuleNotFoundError:
    from openrouter_reference import atomic_write_new, canonical_json
    from remote_safetensors_audit import fetch_range, load_lock


FetchRange = Callable[[str, str, str, int, int], bytes]


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def materialize(
    lock: dict[str, Any],
    remote_path: str,
    output: Path,
    tensor_names: list[str],
    fetch: FetchRange = fetch_range,
) -> dict[str, Any]:
    if output.exists():
        raise ValueError(f"refusing to overwrite {output}")
    names = sorted(tensor_names)
    if not names or len(set(names)) != len(names):
        raise ValueError("tensor names must be non-empty and unique")
    files = {item["path"]: item for item in lock["files"]}
    if remote_path not in files:
        raise ValueError(f"remote path is absent from lock: {remote_path}")
    source_record = files[remote_path]
    repository = lock["repository"]
    revision = lock["revision"]

    raw_prefix = fetch(repository, revision, remote_path, 0, 7)
    source_header_bytes = struct.unpack("<Q", raw_prefix)[0]
    if (
        source_header_bytes <= 0
        or source_header_bytes > 256 * 1024 * 1024
        or source_header_bytes + 8 > source_record["bytes"]
    ):
        raise ValueError("invalid remote safetensors header length")
    raw_header = fetch(
        repository, revision, remote_path, 8, 8 + source_header_bytes - 1
    )
    source_header = json.loads(raw_header)
    if not isinstance(source_header, dict):
        raise ValueError("remote safetensors header is not an object")

    payload = bytearray()
    output_header: dict[str, Any] = {}
    records = []
    source_payload_start = 8 + source_header_bytes
    for name in names:
        metadata = source_header.get(name)
        if not isinstance(metadata, dict):
            raise ValueError(f"source tensor is absent or malformed: {name}")
        dtype = metadata.get("dtype")
        shape = metadata.get("shape")
        offsets = metadata.get("data_offsets")
        if (
            not isinstance(dtype, str)
            or not isinstance(shape, list)
            or not all(isinstance(value, int) and value >= 0 for value in shape)
            or not isinstance(offsets, list)
            or len(offsets) != 2
            or not all(isinstance(value, int) and value >= 0 for value in offsets)
            or offsets[1] <= offsets[0]
        ):
            raise ValueError(f"malformed tensor metadata: {name}")
        absolute_start = source_payload_start + offsets[0]
        absolute_end = source_payload_start + offsets[1] - 1
        if absolute_end >= source_record["bytes"]:
            raise ValueError(f"tensor exceeds locked source file: {name}")
        data = fetch(repository, revision, remote_path, absolute_start, absolute_end)
        output_start = len(payload)
        payload.extend(data)
        output_end = len(payload)
        output_header[name] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [output_start, output_end],
        }
        records.append(
            {
                "name": name,
                "dtype": dtype,
                "shape": shape,
                "source_data_offsets": offsets,
                "data_bytes": len(data),
                "sha256": sha256_bytes(data),
            }
        )

    encoded_header = json.dumps(
        output_header, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    encoded_header += b" " * ((8 - len(encoded_header) % 8) % 8)
    artifact = struct.pack("<Q", len(encoded_header)) + encoded_header + payload
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("xb") as stream:
            stream.write(artifact)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)

    return {
        "schema_version": 1,
        "evidence_class": "pinned_remote_lossless_tensor_ranges",
        "repository": repository,
        "revision": revision,
        "source_file": remote_path,
        "source_file_bytes": source_record["bytes"],
        "source_file_sha256_from_lfs_lock": source_record["sha256"],
        "source_header_bytes": source_header_bytes,
        "output_file": output.name,
        "output_bytes": len(artifact),
        "output_sha256": sha256_bytes(artifact),
        "tensors": records,
        "limitation": (
            "Selected payload bytes are hashed exactly, but the complete remote source-file "
            "hash remains the Hugging Face LFS object identity until local checkpoint closure."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", required=True, type=Path)
    parser.add_argument("--remote-path", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("tensor", nargs="+")
    arguments = parser.parse_args()
    try:
        result = materialize(
            load_lock(arguments.lock),
            arguments.remote_path,
            arguments.output,
            arguments.tensor,
        )
        atomic_write_new(arguments.manifest, canonical_json(result))
        print(arguments.output)
        print(arguments.manifest)
        return 0
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"error: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
