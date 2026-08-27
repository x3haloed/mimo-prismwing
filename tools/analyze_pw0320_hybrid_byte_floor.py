#!/usr/bin/env python3
"""Compute an optimistic corrected-width8 hybrid expert storage bound."""

from __future__ import annotations

import argparse, hashlib, json, math
from collections import defaultdict
from pathlib import Path
from typing import Any

from tools.analyze_pw0319_corrected_route_bank import (
    CORPUS_SHA256, load_rows, greedy_order, sha256_file,
)
from tools.host_safety import HostSafetyMonitor, HostSafetyViolation
from tools.openrouter_reference import atomic_write_new, canonical_json
from tools.reproduce_pw0311_k4_expert import verify_clean_commit

PW0319_SHA256 = "1dd69cfe879cc9783aac7281396d16ab35b1c9cd05dcf0a55eef7137509d1406"
PW0318_MANIFEST_SHA256 = "ca2cd8005c3c8f712fabd0b2fc88183d740bd6613efa065cdd4b25738c4924c3"
K4_BYTES = 12_654_604
SOURCE_BYTES = 25_171_968
STORAGE_BYTES_PER_SECOND = 201_719_808 / (58.125 / 1000.0)
BUDGETS = (512, 1024, 2048)
CACHES = (0, 2 * 1024**3, 4 * 1024**3)


def oracle_cached_bytes(record_sizes: list[int], capacity: int) -> int:
    counts = {size: record_sizes.count(size) for size in set(record_sizes)}
    if any(size not in (K4_BYTES, SOURCE_BYTES) for size in counts):
        # Tiny fixtures may use arbitrary sizes; exact subset-sum keeps this
        # helper honest without entering the production-sized analysis path.
        reachable = {0}
        for size in record_sizes:
            reachable |= {value + size for value in tuple(reachable) if value + size <= capacity}
        return sum(record_sizes) - max(reachable)
    best = 0
    for source_count in range(counts.get(SOURCE_BYTES, 0) + 1):
        source_total = source_count * SOURCE_BYTES
        if source_total > capacity:
            break
        k4_count = min(counts.get(K4_BYTES, 0), (capacity - source_total) // K4_BYTES)
        best = max(best, source_total + k4_count * K4_BYTES)
    return sum(record_sizes) - best


def window_metrics(identities: set[tuple[int, int]], selected: set[tuple[int, int]], accepted: int, cache: int) -> dict[str, Any]:
    sizes = [K4_BYTES if identity in selected else SOURCE_BYTES for identity in identities]
    bytes_after = oracle_cached_bytes(sizes, cache)
    wall = bytes_after / STORAGE_BYTES_PER_SECOND
    return {
        "unique_identities": len(identities),
        "unique_k4_identities": sum(identity in selected for identity in identities),
        "unique_source_identities": sum(identity not in selected for identity in identities),
        "uncached_bytes": sum(sizes),
        "oracle_cache_bytes": cache,
        "bytes_after_oracle_cache": bytes_after,
        "accepted_tokens": accepted,
        "bytes_per_accepted_token": bytes_after / accepted,
        "storage_wall_seconds": wall,
        "optimistic_accepted_tps": accepted / wall if wall else math.inf,
        "required_bytes_per_second_for_two_tps": 2.0 * bytes_after / accepted,
    }


def analyze(*, corpus_manifest: Path, pw0319_analysis: Path, pw0318_manifest: Path, output: Path, repo: Path, commit: str) -> dict[str, Any]:
    if output.exists(): raise FileExistsError(output)
    verify_clean_commit(repo.resolve(), commit)
    if sha256_file(pw0319_analysis) != PW0319_SHA256: raise ValueError("PW-0319 analysis mismatch")
    if sha256_file(pw0318_manifest) != PW0318_MANIFEST_SHA256: raise ValueError("PW-0318 manifest mismatch")
    manifest = json.loads(corpus_manifest.read_text())
    if sha256_file(corpus_manifest) != CORPUS_SHA256: raise ValueError("PW-0208 manifest mismatch")
    accepted = {int(w["corpus_index"]): int(w["A"]) for w in manifest["primary_windows"]}
    categories = {int(w["corpus_index"]): w["category"] for w in manifest["primary_windows"]}
    if set(accepted) != set(range(32)) or any(not 1 <= a <= 8 for a in accepted.values()): raise ValueError("window A authority mismatch")
    rows, route_sha, _ = load_rows(corpus_manifest)
    by_window: dict[int, set[tuple[int, int]]] = defaultdict(set)
    for row in rows: by_window[row.corpus_index].update(row.identities)
    order = greedy_order(rows, maximum_budget=max(BUDGETS))
    safety = HostSafetyMonitor(); safety.checkpoint("authorities_loaded")
    curves=[]
    for budget in BUDGETS:
        chosen=set(order[:budget])
        for cache in CACHES:
            windows=[]
            for index in range(32):
                row=window_metrics(by_window[index], chosen, accepted[index], cache)
                row.update(corpus_index=index, category=categories[index]); windows.append(row)
            category_pass={name: any(w["optimistic_accepted_tps"] >= 2 for w in windows if w["category"] == name) for name in sorted(set(categories.values()))}
            passing=sum(w["optimistic_accepted_tps"] >= 2 for w in windows)
            curves.append({"budget":budget,"oracle_cache_bytes":cache,"passing_windows":passing,"passing_fraction":passing/32,"category_has_pass":category_pass,"continuation_gate_pass":passing>=16 and all(category_pass.values()),"minimum_optimistic_tps":min(w["optimistic_accepted_tps"] for w in windows),"median_optimistic_tps":sorted(w["optimistic_accepted_tps"] for w in windows)[16],"maximum_required_bandwidth_for_two_tps":max(w["required_bytes_per_second_for_two_tps"] for w in windows),"windows":windows})
    safety.release_checkpoint("analysis_released", ["route rows", "coverage order"]); safety.checkpoint("final_service_health")
    strongest=next(c for c in curves if c["budget"]==2048 and c["oracle_cache_bytes"]==4*1024**3)
    report={"schema_version":1,"experiment_id":"PW-0320","status":"complete","decision":"authorize_integrated_streaming_runner" if strongest["continuation_gate_pass"] else "reject_width8_hybrid_on_current_m1_storage","commit":commit,"authority":{"corpus_manifest_sha256":CORPUS_SHA256,"corrected_route_sha256":route_sha,"pw0319_analysis_sha256":PW0319_SHA256,"pw0318_manifest_sha256":PW0318_MANIFEST_SHA256},"constants":{"k4_executable_bytes":K4_BYTES,"source_executable_bytes":SOURCE_BYTES,"cold_storage_bytes_per_second":STORAGE_BYTES_PER_SECOND,"storage_source":"PW-0136 two-worker cold median"},"curves":curves,"safety_snapshots":safety.evidence(),"accepted_tokens":0,"performance_claim":None}
    output.mkdir(parents=True); path=output/'analysis.json'; atomic_write_new(path, canonical_json(report)); print(json.dumps({"output":str(path),"decision":report["decision"]})); return report


def main() -> int:
    p=argparse.ArgumentParser()
    for name in ("corpus_manifest","pw0319_analysis","pw0318_manifest","output","repo"): p.add_argument('--'+name.replace('_','-'), required=True, type=Path)
    p.add_argument('--commit',required=True)
    try: analyze(**vars(p.parse_args())); return 0
    except (FileExistsError,HostSafetyViolation,KeyError,OSError,RuntimeError,TypeError,ValueError,json.JSONDecodeError) as e: print(json.dumps({"error":str(e)})); return 1
if __name__ == '__main__': raise SystemExit(main())
