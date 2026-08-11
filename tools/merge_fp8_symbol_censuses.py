#!/usr/bin/env python3
"""Merge compatible deterministic FP8 census records without rewriting samples."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from tools.openrouter_reference import atomic_write_new, canonical_json
except ModuleNotFoundError:
    from openrouter_reference import atomic_write_new, canonical_json


IDENTITY_KEYS = ("schema_version", "evidence_class", "repository", "revision")


def merge(records: list[dict]) -> dict:
    if not records:
        raise ValueError("at least one census is required")
    identity = {key: records[0].get(key) for key in IDENTITY_KEYS}
    if identity["schema_version"] != 1:
        raise ValueError("unknown census schema")
    samples = []
    seen = set()
    network_bytes = 0
    for record in records:
        if any(record.get(key) != value for key, value in identity.items()):
            raise ValueError("census identity mismatch")
        if record.get("sample_count") != len(record.get("samples", [])):
            raise ValueError("sample count mismatch")
        network_bytes += record["network_bytes"]
        for sample in record["samples"]:
            key = (sample["tensor"], sample["row_block"])
            if key in seen:
                raise ValueError(f"duplicate sample: {key}")
            seen.add(key)
            samples.append(sample)
    samples.sort(key=lambda sample: (sample["tensor"], sample["row_block"]))
    return {
        **identity,
        "network_bytes": network_bytes,
        "sample_count": len(samples),
        "quantization_block_count": sum(len(sample["blocks"]) for sample in samples),
        "samples": samples,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--census", action="append", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    try:
        records = [json.loads(path.read_text(encoding="utf-8")) for path in arguments.census]
        atomic_write_new(arguments.output, canonical_json(merge(records)))
        print(arguments.output)
        return 0
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"error: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
