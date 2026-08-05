#!/usr/bin/env python3
"""Prove remote safetensors payload inequality with deterministic byte samples."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import struct
from typing import Any
from urllib.parse import quote
from urllib.request import Request, urlopen

try:
    from tools.openrouter_reference import atomic_write_new, canonical_json
except ModuleNotFoundError:  # Direct execution places tools/ first on sys.path.
    from openrouter_reference import atomic_write_new, canonical_json


SCHEMA_VERSION = 1


def fetch_range(repository: str, revision: str, path: str, start: int, end: int) -> bytes:
    repository_part = quote(repository, safe="/")
    revision_part = quote(revision, safe="")
    path_part = quote(path, safe="/")
    url = (
        f"https://huggingface.co/{repository_part}/resolve/{revision_part}/"
        f"{path_part}?download=true"
    )
    request = Request(url, headers={"Range": f"bytes={start}-{end}"})
    with urlopen(request, timeout=60) as response:
        payload = response.read()
        expected = end - start + 1
        if response.status != 206 or len(payload) != expected:
            raise ValueError(
                f"range request for {path} returned status={response.status}, "
                f"bytes={len(payload)}, expected={expected}"
            )
        return payload


def sample_offsets(payload_bytes: int, sample_bytes: int) -> list[int]:
    if payload_bytes <= 0 or sample_bytes <= 0 or sample_bytes > payload_bytes:
        raise ValueError("sample must fit within a non-empty payload")
    return sorted({0, (payload_bytes - sample_bytes) // 2, payload_bytes - sample_bytes})


def safetensors_header_bytes(repository: str, revision: str, path: str) -> int:
    return struct.unpack("<Q", fetch_range(repository, revision, path, 0, 7))[0]


def load_lock(path: Path) -> dict[str, Any]:
    lock = json.loads(path.read_text(encoding="utf-8"))
    if lock.get("schema_version") != 1 or not isinstance(lock.get("files"), list):
        raise ValueError(f"unknown or malformed checkpoint lock: {path}")
    return lock


def audit(left: dict[str, Any], right: dict[str, Any], prefix: str, sample_bytes: int) -> dict[str, Any]:
    left_files = {item["path"]: item for item in left["files"]}
    right_files = {item["path"]: item for item in right["files"]}
    paths = sorted(
        path
        for path in set(left_files) & set(right_files)
        if path.startswith(prefix) and path.endswith(".safetensors")
    )
    if not paths:
        raise ValueError("no common safetensors paths matched the prefix")

    def audit_path(path: str) -> dict[str, Any]:
        left_header = safetensors_header_bytes(left["repository"], left["revision"], path)
        right_header = safetensors_header_bytes(right["repository"], right["revision"], path)
        left_payload = left_files[path]["bytes"] - 8 - left_header
        right_payload = right_files[path]["bytes"] - 8 - right_header
        offsets = sample_offsets(min(left_payload, right_payload), sample_bytes)
        samples = []
        for offset in offsets:
            left_bytes = fetch_range(
                left["repository"],
                left["revision"],
                path,
                8 + left_header + offset,
                8 + left_header + offset + sample_bytes - 1,
            )
            right_bytes = fetch_range(
                right["repository"],
                right["revision"],
                path,
                8 + right_header + offset,
                8 + right_header + offset + sample_bytes - 1,
            )
            left_hash = hashlib.sha256(left_bytes).hexdigest()
            right_hash = hashlib.sha256(right_bytes).hexdigest()
            samples.append(
                {
                    "payload_offset": offset,
                    "bytes": sample_bytes,
                    "left_sha256": left_hash,
                    "right_sha256": right_hash,
                    "equal": left_hash == right_hash,
                }
            )
        return {
            "path": path,
            "left_file_sha256": left_files[path]["sha256"],
            "right_file_sha256": right_files[path]["sha256"],
            "left_header_bytes": left_header,
            "right_header_bytes": right_header,
            "left_payload_bytes": left_payload,
            "right_payload_bytes": right_payload,
            "payload_sizes_equal": left_payload == right_payload,
            "samples": samples,
            "sampled_payload_difference_proven": any(not item["equal"] for item in samples),
        }

    with ThreadPoolExecutor(max_workers=min(8, len(paths))) as executor:
        records = list(executor.map(audit_path, paths))

    return {
        "schema_version": SCHEMA_VERSION,
        "left": {"repository": left["repository"], "revision": left["revision"]},
        "right": {"repository": right["repository"], "revision": right["revision"]},
        "path_prefix": prefix,
        "sample_bytes": sample_bytes,
        "files": records,
        "file_count": len(records),
        "all_files_have_sampled_payload_difference": all(
            item["sampled_payload_difference_proven"] for item in records
        ),
        "limitation": (
            "Unequal samples prove payload inequality. Equal samples would not prove complete "
            "payload identity because this audit does not download every payload byte."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--left-lock", required=True, type=Path)
    parser.add_argument("--right-lock", required=True, type=Path)
    parser.add_argument("--path-prefix", default="model_pp")
    parser.add_argument("--sample-bytes", type=int, default=65536)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()

    try:
        result = audit(
            load_lock(arguments.left_lock),
            load_lock(arguments.right_lock),
            arguments.path_prefix,
            arguments.sample_bytes,
        )
        atomic_write_new(arguments.output, canonical_json(result))
        print(arguments.output)
        return 0
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"error: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
