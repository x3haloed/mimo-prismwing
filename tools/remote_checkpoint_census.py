#!/usr/bin/env python3
"""Census pinned safetensors headers without downloading tensor payloads."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import struct
from typing import Any

try:
    from tools.openrouter_reference import atomic_write_new, canonical_json
    from tools.remote_safetensors_audit import fetch_range, load_lock
except ModuleNotFoundError:
    from openrouter_reference import atomic_write_new, canonical_json
    from remote_safetensors_audit import fetch_range, load_lock


SCHEMA_VERSION = 1


def classify_tensor(name: str, shard: str) -> str:
    if shard == "audio_tokenizer/model.safetensors":
        return "audio_tokenizer"
    if name.startswith("visual."):
        return "vision_encoder"
    if name.startswith("audio_encoder.") or name.startswith("speech_embeddings."):
        return "audio_path"
    if ".mtp." in name:
        return "mtp"
    if ".mlp.experts." in name:
        return "routed_experts"
    if ".mlp.gate.weight" in name:
        return "routers"
    if name == "model.embed_tokens.weight":
        return "token_embeddings"
    if name == "lm_head.weight":
        return "lm_head"
    if name.startswith("model.layers.0.mlp."):
        return "dense_layer_zero"
    if "self_attn" in name or "layernorm" in name or name == "model.norm.weight":
        return "attention_and_norms"
    return "other_language_or_projector"


def fetch_header(repository: str, revision: str, path: str, file_bytes: int) -> dict[str, Any]:
    prefix = fetch_range(repository, revision, path, 0, 7)
    header_bytes = struct.unpack("<Q", prefix)[0]
    if header_bytes <= 0 or header_bytes > 256 * 1024 * 1024 or header_bytes + 8 > file_bytes:
        raise ValueError(f"invalid safetensors header length for {path}")
    raw = fetch_range(repository, revision, path, 8, 8 + header_bytes - 1)
    header = json.loads(raw)
    if not isinstance(header, dict):
        raise ValueError(f"safetensors header is not an object: {path}")
    return {"path": path, "file_bytes": file_bytes, "header_bytes": header_bytes, "header": header}


def build_census(lock: dict[str, Any], index: dict[str, Any]) -> dict[str, Any]:
    files = {
        item["path"]: item
        for item in lock["files"]
        if item["path"].endswith(".safetensors")
    }
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError("source index lacks a non-empty weight_map")
    indexed_shards = set(weight_map.values())
    standalone = {"model_mtp.safetensors", "audio_tokenizer/model.safetensors"}
    expected_shards = indexed_shards | standalone
    if expected_shards != set(files):
        missing = sorted(expected_shards - set(files))
        extra = sorted(set(files) - expected_shards)
        raise ValueError(f"lock/index safetensors mismatch: missing={missing}, extra={extra}")

    def one(path: str) -> dict[str, Any]:
        return fetch_header(
            lock["repository"], lock["revision"], path, files[path]["bytes"]
        )

    with ThreadPoolExecutor(max_workers=8) as executor:
        shards = list(executor.map(one, sorted(files)))

    records = []
    seen_indexed = set()
    categories: dict[str, dict[str, int]] = {}
    tensor_data_bytes = 0
    header_and_padding_bytes = 0
    for shard in shards:
        maximum_end = 0
        for name, metadata in shard["header"].items():
            if name == "__metadata__":
                continue
            if not isinstance(metadata, dict):
                raise ValueError(f"malformed metadata: {name}")
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
                or offsets[1] < offsets[0]
            ):
                raise ValueError(f"malformed tensor metadata: {name}")
            path = shard["path"]
            if name in weight_map:
                if weight_map[name] != path or name in seen_indexed:
                    raise ValueError(f"index assignment mismatch or duplicate: {name}")
                seen_indexed.add(name)
            elif path not in standalone:
                raise ValueError(f"unindexed tensor in main shard: {name}")
            data_bytes = offsets[1] - offsets[0]
            maximum_end = max(maximum_end, offsets[1])
            category = classify_tensor(name, path)
            total = categories.setdefault(category, {"tensors": 0, "data_bytes": 0})
            total["tensors"] += 1
            total["data_bytes"] += data_bytes
            tensor_data_bytes += data_bytes
            records.append(
                {
                    "name": name,
                    "shard": path,
                    "dtype": dtype,
                    "shape": shape,
                    "data_bytes": data_bytes,
                    "category": category,
                }
            )
        payload_bytes = shard["file_bytes"] - 8 - shard["header_bytes"]
        if maximum_end > payload_bytes:
            raise ValueError(f"tensor offsets exceed payload: {shard['path']}")
        header_and_padding_bytes += shard["file_bytes"] - maximum_end
    if len(seen_indexed) != len(weight_map):
        raise ValueError(
            f"index assigns {len(weight_map)} tensors but headers contain {len(seen_indexed)}"
        )
    records.sort(key=lambda item: (item["name"], item["shard"]))
    return {
        "schema_version": SCHEMA_VERSION,
        "repository": lock["repository"],
        "revision": lock["revision"],
        "evidence_class": "pinned_remote_headers_not_local_payload_verification",
        "tensor_count": len(records),
        "indexed_tensor_count": len(seen_indexed),
        "tensor_data_bytes": tensor_data_bytes,
        "safetensors_file_bytes": sum(item["bytes"] for item in files.values()),
        "header_padding_and_non_tensor_bytes": header_and_padding_bytes,
        "categories": dict(sorted(categories.items())),
        "tensors": records,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", required=True, type=Path)
    parser.add_argument("--index", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    try:
        lock = load_lock(arguments.lock)
        index = json.loads(arguments.index.read_text(encoding="utf-8"))
        census = build_census(lock, index)
        atomic_write_new(arguments.output, canonical_json(census))
        print(arguments.output)
        return 0
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"error: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
