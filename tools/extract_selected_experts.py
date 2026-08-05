#!/usr/bin/env python3
"""Extract every tensor for selected experts from a pinned sharded checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from tools.openrouter_reference import atomic_write_new, canonical_json
    from tools.remote_safetensors_audit import load_lock
    from tools.remote_tensor_extract import materialize
except ModuleNotFoundError:
    from openrouter_reference import atomic_write_new, canonical_json
    from remote_safetensors_audit import load_lock
    from remote_tensor_extract import materialize


def selected_tensor_names(layer: int, experts: list[int]) -> list[str]:
    names = []
    for expert in sorted(experts):
        prefix = f"model.layers.{layer}.mlp.experts.{expert}"
        for projection in ("gate", "up", "down"):
            weight = f"{prefix}.{projection}_proj.weight"
            names.extend((weight, weight + "_scale_inv"))
    return names


def group_by_shard(index: dict, names: list[str]) -> dict[str, list[str]]:
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        raise ValueError("checkpoint index lacks weight_map")
    grouped: dict[str, list[str]] = {}
    for name in names:
        shard = weight_map.get(name)
        if not isinstance(shard, str):
            raise ValueError(f"checkpoint index lacks selected tensor: {name}")
        grouped.setdefault(shard, []).append(name)
    return dict(sorted(grouped.items()))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", required=True, type=Path)
    parser.add_argument("--index", required=True, type=Path)
    parser.add_argument("--layer", required=True, type=int)
    parser.add_argument("--experts", required=True, type=int, nargs="+")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    arguments = parser.parse_args()
    try:
        if len(set(arguments.experts)) != len(arguments.experts):
            raise ValueError("expert IDs must be unique")
        lock = load_lock(arguments.lock)
        index = json.loads(arguments.index.read_text(encoding="utf-8"))
        grouped = group_by_shard(
            index, selected_tensor_names(arguments.layer, arguments.experts)
        )
        arguments.output_dir.mkdir(parents=True, exist_ok=True)
        sources = []
        for shard, names in grouped.items():
            output = arguments.output_dir / f"{Path(shard).stem}.selected.safetensors"
            sources.append(materialize(lock, shard, output, names))
        result = {
            "schema_version": 1,
            "evidence_class": "pinned_remote_lossless_selected_experts",
            "repository": lock["repository"],
            "revision": lock["revision"],
            "layer": arguments.layer,
            "experts": sorted(arguments.experts),
            "source_slices": sources,
        }
        atomic_write_new(arguments.manifest, canonical_json(result))
        print(arguments.output_dir)
        print(arguments.manifest)
        return 0
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"error: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
