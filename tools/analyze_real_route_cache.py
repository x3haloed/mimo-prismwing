#!/usr/bin/env python3
"""Replay PW-0091 routes through exact cache policies, including Belady."""

from __future__ import annotations

import argparse
from collections import Counter, OrderedDict, defaultdict, deque
import hashlib
import json
import math
from pathlib import Path
import subprocess
from typing import Hashable

try:
    from tools.generate_real_layer0_bf16_oracle import PROMPT_IDS, REVISION
    from tools.openrouter_reference import atomic_write_new, canonical_json
except ModuleNotFoundError:
    from generate_real_layer0_bf16_oracle import PROMPT_IDS, REVISION
    from openrouter_reference import atomic_write_new, canonical_json


PW0091_SHA256 = "87466b59480a5a5b4256c490f1dfe670fe09f28d21d169085ab13bb1b4b7ab59"
EXPERT_BYTES = 25_171_968
ROUTED_LAYERS = 47
TOP_K = 8


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_accesses(path: Path) -> tuple[list[tuple[int, int]], dict]:
    if sha256_file(path) != PW0091_SHA256:
        raise ValueError("PW-0091 manifest hash mismatch")
    manifest = json.loads(path.read_text())
    traces = manifest.get("layer_traces")
    if (
        manifest.get("revision") != REVISION
        or manifest.get("prompt_token_ids") != PROMPT_IDS
        or not isinstance(traces, list)
        or len(traces) != 48
    ):
        raise ValueError("PW-0091 route authority mismatch")
    accesses: list[tuple[int, int]] = []
    for position in range(len(PROMPT_IDS)):
        for layer in range(1, 48):
            rows = traces[layer].get("selected_experts_by_position")
            if not isinstance(rows, list) or len(rows) != len(PROMPT_IDS):
                raise ValueError(f"PW-0091 route row count mismatch at layer {layer}")
            experts = rows[position]
            if (
                not isinstance(experts, list)
                or len(experts) != TOP_K
                or len(set(experts)) != TOP_K
                or any(not isinstance(expert, int) or expert < 0 or expert >= 256 for expert in experts)
            ):
                raise ValueError(f"PW-0091 route identity mismatch at layer {layer}, position {position}")
            accesses.extend((layer, expert) for expert in experts)
    expected = len(PROMPT_IDS) * ROUTED_LAYERS * TOP_K
    if len(accesses) != expected:
        raise ValueError("route access count mismatch")
    return accesses, manifest


def lru_hits(accesses: list[Hashable], capacity: int) -> int:
    cache: OrderedDict[Hashable, None] = OrderedDict()
    hits = 0
    for item in accesses:
        if item in cache:
            hits += 1
            cache.move_to_end(item)
        else:
            if len(cache) >= capacity:
                cache.popitem(last=False)
            cache[item] = None
    return hits


def lfu_hits(accesses: list[Hashable], capacity: int) -> int:
    cache: set[Hashable] = set()
    frequency: Counter = Counter()
    last_used: dict[Hashable, int] = {}
    hits = 0
    for position, item in enumerate(accesses):
        frequency[item] += 1
        last_used[item] = position
        if item in cache:
            hits += 1
            continue
        if len(cache) >= capacity:
            victim = min(cache, key=lambda value: (frequency[value], last_used[value]))
            cache.remove(victim)
        cache.add(item)
    return hits


def belady_hits(accesses: list[Hashable], capacity: int) -> int:
    future: dict[Hashable, deque[int]] = defaultdict(deque)
    for position, item in enumerate(accesses):
        future[item].append(position)
    cache: set[Hashable] = set()
    hits = 0
    for position, item in enumerate(accesses):
        if not future[item] or future[item].popleft() != position:
            raise ValueError("Belady future-index corruption")
        if item in cache:
            hits += 1
            continue
        if len(cache) >= capacity:
            victim = max(
                cache,
                key=lambda value: future[value][0] if future[value] else math.inf,
            )
            cache.remove(victim)
        cache.add(item)
    return hits


def analyze(manifest_path: Path) -> dict:
    accesses, _manifest = load_accesses(manifest_path)
    access_bytes = canonical_json([[layer, expert] for layer, expert in accesses])
    unique = len(set(accesses))
    curves = []
    for gib in range(1, 11):
        capacity_bytes = gib * 1024**3
        capacity = capacity_bytes // EXPERT_BYTES
        policies = {}
        for name, function in (("lru", lru_hits), ("lfu", lfu_hits), ("belady", belady_hits)):
            hits = function(accesses, capacity)
            misses = len(accesses) - hits
            policies[name] = {
                "hits": hits,
                "misses": misses,
                "hit_ratio": hits / len(accesses),
                "logical_miss_bytes": misses * EXPERT_BYTES,
                "logical_miss_bytes_per_token": misses * EXPERT_BYTES / len(PROMPT_IDS),
            }
        if not (
            policies["belady"]["hits"] >= policies["lru"]["hits"]
            and policies["belady"]["hits"] >= policies["lfu"]["hits"]
        ):
            raise ValueError("Belady failed to upper-bound causal policies")
        curves.append({"capacity_gib": gib, "capacity_experts": capacity, "policies": policies})
    eight = curves[7]["policies"]["belady"]
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, text=True, capture_output=True
    ).stdout.strip()
    dirty = bool(subprocess.run(
        ["git", "status", "--porcelain"], check=True, text=True, capture_output=True
    ).stdout.strip())
    return {
        "schema_version": 1,
        "evidence_class": "pw0104_real_route_exact_cache_policy_bound",
        "revision": REVISION,
        "pw0091_manifest_sha256": PW0091_SHA256,
        "git_commit": commit,
        "git_dirty": dirty,
        "order": "token_major_then_layer_1_through_47_then_native_top8_order",
        "positions": len(PROMPT_IDS),
        "routed_layers": ROUTED_LAYERS,
        "top_k": TOP_K,
        "accesses": len(accesses),
        "unique_layer_experts": unique,
        "expert_bytes": EXPERT_BYTES,
        "access_list_sha256": hashlib.sha256(access_bytes).hexdigest(),
        "capacity_curves": curves,
        "required_hit_ratio_range": [0.93, 0.98],
        "eight_gib_belady_hit_ratio": eight["hit_ratio"],
        "eight_gib_belady_gap_to_minimum_required": 0.93 - eight["hit_ratio"],
        "decision": "reject_exact_6_to_8_gib_cache_as_primary_throughput_mechanism_for_this_trace",
        "limitations": "single 27-position causal text trace; logical equal-size expert payloads; no physical timing claim",
        "performance_claim": None,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    try:
        result = analyze(arguments.manifest)
        atomic_write_new(arguments.output, canonical_json(result))
        print(json.dumps({"output": str(arguments.output),
                          "eight_gib_belady_hit_ratio": result["eight_gib_belady_hit_ratio"]}))
        return 0
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as error:
        print(json.dumps({"error": str(error)}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
