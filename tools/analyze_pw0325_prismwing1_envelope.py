#!/usr/bin/env python3
"""Find a deterministic category-balanced K4 storage envelope for Prismwing-1."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np

try:
    from tools.analyze_pw0319_corrected_route_bank import (
        ARTIFACT_BYTES_PER_IDENTITY,
        CORPUS_SHA256,
        EXPERTS_PER_LAYER,
        M1_SECONDS_PER_IDENTITY,
        M4_SECONDS_PER_IDENTITY,
        ROUTED_LAYERS,
        load_rows,
        sha256_file,
    )
    from tools.analyze_pw0320_hybrid_byte_floor import (
        K4_BYTES,
        SOURCE_BYTES,
        STORAGE_BYTES_PER_SECOND,
        window_metrics,
    )
    from tools.host_safety import HostSafetyMonitor, HostSafetyViolation
    from tools.openrouter_reference import atomic_write_new, canonical_json
    from tools.reproduce_pw0311_k4_expert import verify_clean_commit
except ModuleNotFoundError:
    from analyze_pw0319_corrected_route_bank import (
        ARTIFACT_BYTES_PER_IDENTITY,
        CORPUS_SHA256,
        EXPERTS_PER_LAYER,
        M1_SECONDS_PER_IDENTITY,
        M4_SECONDS_PER_IDENTITY,
        ROUTED_LAYERS,
        load_rows,
        sha256_file,
    )
    from analyze_pw0320_hybrid_byte_floor import (
        K4_BYTES,
        SOURCE_BYTES,
        STORAGE_BYTES_PER_SECOND,
        window_metrics,
    )
    from host_safety import HostSafetyMonitor, HostSafetyViolation
    from openrouter_reference import atomic_write_new, canonical_json
    from reproduce_pw0311_k4_expert import verify_clean_commit


EXPERIMENT_ID = "PW-0325"
PW0318_SUMMARY_SHA256 = "a91af31bdea45749c9ae9d5d679260bcbcd8284c238479938206a7e7e0b5eb2f"
PW0319_ANALYSIS_SHA256 = "1dd69cfe879cc9783aac7281396d16ab35b1c9cd05dcf0a55eef7137509d1406"
PW0320_ANALYSIS_SHA256 = "de6424aa68d0c65f8f9206a53f61475286bde501873cd4f6ee06299c9b37d7a9"
CACHE_CAPACITIES = (4 * 1024**3, 6 * 1024**3, 8 * 1024**3)
CATEGORY_TARGETS = (1.10, 1.25, 1.50)
PRISMWING1_CANDIDATE_CACHE = 8 * 1024**3
PRISMWING1_CANDIDATE_TARGET = 1.25
MAXIMUM_CANDIDATE_IDENTITIES = 4096


def canonical_universe(
    layers: Iterable[int] = ROUTED_LAYERS,
    experts_per_layer: int = EXPERTS_PER_LAYER,
) -> tuple[tuple[int, int], ...]:
    return tuple((layer, expert) for layer in layers for expert in range(experts_per_layer))


def conservative_cached_bytes(total_uncached_bytes: int, cache_bytes: int) -> int:
    """Apply only the guaranteed capacity-minus-one-source-record cache credit."""
    if total_uncached_bytes < 0 or cache_bytes < 0:
        raise ValueError("byte counts must be non-negative")
    credit = max(0, cache_bytes - SOURCE_BYTES)
    return max(0, total_uncached_bytes - credit)


def balanced_order(
    occurrence_counts: np.ndarray,
    initial_deficits: np.ndarray,
    *,
    byte_reduction: int = SOURCE_BYTES - K4_BYTES,
) -> tuple[list[int], np.ndarray]:
    """Greedily reduce normalized category deficits with canonical-index ties.

    ``occurrence_counts[c, i]`` is the number of windows in category ``c``
    containing identity ``i``. Scores use the deficit at the start of the
    scenario as the normalization denominator. ``numpy.argmax`` returns the
    first maximum, which is the canonical layer/expert tie break.
    """
    counts = np.asarray(occurrence_counts, dtype=np.int64)
    deficits = np.asarray(initial_deficits, dtype=np.float64)
    if counts.ndim != 2 or deficits.shape != (counts.shape[0],):
        raise ValueError("category occurrence/deficit shape mismatch")
    if np.any(counts < 0) or np.any(~np.isfinite(deficits)) or np.any(deficits < 0):
        raise ValueError("invalid category occurrence/deficit value")
    if byte_reduction <= 0:
        raise ValueError("byte reduction must be positive")
    if not np.any(deficits > 0):
        return [], deficits.copy()

    remaining = deficits.copy()
    denominators = np.where(deficits > 0, deficits, 1.0)
    chosen = np.zeros(counts.shape[1], dtype=bool)
    order: list[int] = []
    reductions = counts.astype(np.float64) * float(byte_reduction)
    while np.any(remaining > 1.0e-6):
        active = remaining > 1.0e-6
        scores = np.minimum(reductions[active], remaining[active, None])
        scores = np.sum(scores / denominators[active, None], axis=0)
        scores[chosen] = -1.0
        identity_index = int(np.argmax(scores))
        if scores[identity_index] <= 0.0:
            break
        chosen[identity_index] = True
        order.append(identity_index)
        remaining = np.maximum(0.0, remaining - reductions[:, identity_index])
    return order, remaining


def nearest_rank(values: list[float], percentile: float) -> float:
    if not values or not 0.0 < percentile <= 1.0:
        raise ValueError("nearest-rank input mismatch")
    return sorted(values)[math.ceil(percentile * len(values)) - 1]


def category_aggregate(
    windows: list[dict[str, Any]],
    bytes_key: str,
) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for window in windows:
        groups[str(window["category"])].append(window)
    result: dict[str, dict[str, Any]] = {}
    for category, rows in sorted(groups.items()):
        accepted = sum(int(row["accepted_tokens"]) for row in rows)
        moved = sum(int(row[bytes_key]) for row in rows)
        wall = moved / STORAGE_BYTES_PER_SECOND
        result[category] = {
            "windows": len(rows),
            "accepted_tokens": accepted,
            "bytes_after_cache": moved,
            "storage_wall_seconds": wall,
            "optimistic_accepted_tps": accepted / wall if wall else math.inf,
        }
    return result


def selection_sha256(universe: tuple[tuple[int, int], ...], order: list[int]) -> str:
    payload = [
        {"layer": universe[index][0], "expert": universe[index][1]}
        for index in order
    ]
    return hashlib.sha256(canonical_json(payload)).hexdigest()


def validate_upstream(
    *,
    pw0318_summary: Path,
    pw0319_analysis: Path,
    pw0320_analysis: Path,
) -> None:
    expected = (
        (pw0318_summary, PW0318_SUMMARY_SHA256, "PW-0318 summary"),
        (pw0319_analysis, PW0319_ANALYSIS_SHA256, "PW-0319 analysis"),
        (pw0320_analysis, PW0320_ANALYSIS_SHA256, "PW-0320 analysis"),
    )
    for path, digest, label in expected:
        if sha256_file(path) != digest:
            raise ValueError(f"{label} mismatch")
    pw0318 = json.loads(pw0318_summary.read_text())
    pw0319 = json.loads(pw0319_analysis.read_text())
    pw0320 = json.loads(pw0320_analysis.read_text())
    if (
        pw0318.get("experiment_id") != "PW-0318"
        or pw0318.get("status") != "layer4_decode_transaction_qualified"
        or pw0319.get("experiment_id") != "PW-0319"
        or pw0319.get("authority", {}).get("corpus_manifest_sha256") != CORPUS_SHA256
        or pw0320.get("experiment_id") != "PW-0320"
        or pw0320.get("authority", {}).get("pw0319_analysis_sha256")
        != PW0319_ANALYSIS_SHA256
        or int(pw0320.get("constants", {}).get("k4_executable_bytes", -1)) != K4_BYTES
        or int(pw0320.get("constants", {}).get("source_executable_bytes", -1))
        != SOURCE_BYTES
        or not math.isclose(
            float(pw0320.get("constants", {}).get("cold_storage_bytes_per_second", -1)),
            STORAGE_BYTES_PER_SECOND,
            rel_tol=1.0e-12,
        )
    ):
        raise ValueError("upstream semantic authority mismatch")


def scenario(
    *,
    universe: tuple[tuple[int, int], ...],
    by_window: dict[int, set[tuple[int, int]]],
    categories: dict[int, str],
    accepted: dict[int, int],
    occurrence_counts: np.ndarray,
    category_names: tuple[str, ...],
    cache_bytes: int,
    target_tps: float,
) -> dict[str, Any]:
    base_by_category = np.zeros(len(category_names), dtype=np.float64)
    accepted_by_category = np.zeros(len(category_names), dtype=np.float64)
    category_indices = {name: index for index, name in enumerate(category_names)}
    for window_index in range(32):
        category_index = category_indices[categories[window_index]]
        base_by_category[category_index] += conservative_cached_bytes(
            len(by_window[window_index]) * SOURCE_BYTES,
            cache_bytes,
        )
        accepted_by_category[category_index] += accepted[window_index]
    byte_limits = accepted_by_category * STORAGE_BYTES_PER_SECOND / target_tps
    initial_deficits = np.maximum(0.0, base_by_category - byte_limits)
    order_indices, remaining = balanced_order(occurrence_counts, initial_deficits)
    selected = {universe[index] for index in order_indices}

    windows: list[dict[str, Any]] = []
    for window_index in range(32):
        exact = window_metrics(
            by_window[window_index], selected, accepted[window_index], cache_bytes
        )
        conservative_bytes = conservative_cached_bytes(
            int(exact["uncached_bytes"]), cache_bytes
        )
        exact.update(
            corpus_index=window_index,
            category=categories[window_index],
            conservative_bytes_after_cache=conservative_bytes,
            conservative_storage_wall_seconds=(
                conservative_bytes / STORAGE_BYTES_PER_SECOND
            ),
            conservative_optimistic_accepted_tps=(
                accepted[window_index]
                * STORAGE_BYTES_PER_SECOND
                / conservative_bytes
                if conservative_bytes
                else math.inf
            ),
        )
        windows.append(exact)

    conservative_categories = category_aggregate(
        windows, "conservative_bytes_after_cache"
    )
    exact_categories = category_aggregate(windows, "bytes_after_oracle_cache")
    exact_values = [float(window["optimistic_accepted_tps"]) for window in windows]
    exact_bytes = sum(int(window["bytes_after_oracle_cache"]) for window in windows)
    total_accepted = sum(accepted.values())
    exact_aggregate = total_accepted * STORAGE_BYTES_PER_SECOND / exact_bytes
    selected_occurrences = int(
        sum(
            identity in selected
            for window_identities in by_window.values()
            for identity in window_identities
        )
    )
    total_occurrences = sum(len(identities) for identities in by_window.values())
    layer_counts: dict[str, int] = defaultdict(int)
    for layer, _ in selected:
        layer_counts[str(layer)] += 1
    category_coverage = {
        name: {
            "selected_identity_window_occurrences": int(
                sum(
                    occurrence_counts[category_indices[name], index]
                    for index in order_indices
                )
            ),
            "total_identity_window_occurrences": int(
                np.sum(occurrence_counts[category_indices[name]])
            ),
        }
        for name in category_names
    }
    for value in category_coverage.values():
        value["selected_occurrence_fraction"] = (
            value["selected_identity_window_occurrences"]
            / value["total_identity_window_occurrences"]
        )

    order = [
        {"layer": universe[index][0], "expert": universe[index][1]}
        for index in order_indices
    ]
    selected_count = len(order)
    return {
        "oracle_cache_bytes": cache_bytes,
        "category_target_tps": target_tps,
        "selector_complete": bool(np.all(remaining <= 1.0e-6)),
        "remaining_category_deficit_bytes": {
            name: float(remaining[index])
            for index, name in enumerate(category_names)
        },
        "selected_identities": selected_count,
        "selection_order_sha256": selection_sha256(universe, order_indices),
        "selection_order": order,
        "coverage": {
            "selected_identity_window_occurrences": selected_occurrences,
            "total_identity_window_occurrences": total_occurrences,
            "selected_occurrence_fraction": selected_occurrences / total_occurrences,
            "selected_identities_by_layer": dict(sorted(layer_counts.items(), key=lambda x: int(x[0]))),
            "category": category_coverage,
        },
        "installed_hybrid_expert_bank_bytes": (
            selected_count * K4_BYTES
            + (len(universe) - selected_count) * SOURCE_BYTES
        ),
        "all_source_expert_bank_bytes": len(universe) * SOURCE_BYTES,
        "construction_artifact_bytes": selected_count * ARTIFACT_BYTES_PER_IDENTITY,
        "estimated_m1_construction_seconds": selected_count * M1_SECONDS_PER_IDENTITY,
        "estimated_m4_construction_seconds": selected_count * M4_SECONDS_PER_IDENTITY,
        "conservative_category": conservative_categories,
        "exact_oracle_category": exact_categories,
        "exact_oracle_summary": {
            "accepted_tokens": total_accepted,
            "bytes_after_cache": exact_bytes,
            "storage_wall_seconds": exact_bytes / STORAGE_BYTES_PER_SECOND,
            "aggregate_optimistic_accepted_tps": exact_aggregate,
            "minimum_window_optimistic_accepted_tps": min(exact_values),
            "nearest_rank_p10_window_optimistic_accepted_tps": nearest_rank(exact_values, 0.10),
            "nearest_rank_median_window_optimistic_accepted_tps": nearest_rank(exact_values, 0.50),
            "maximum_window_optimistic_accepted_tps": max(exact_values),
        },
        "windows": windows,
    }


def analyze(
    *,
    corpus_manifest: Path,
    pw0318_summary: Path,
    pw0319_analysis: Path,
    pw0320_analysis: Path,
    output: Path,
    repo: Path,
    commit: str,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(output)
    verify_clean_commit(repo.resolve(), commit)
    validate_upstream(
        pw0318_summary=pw0318_summary,
        pw0319_analysis=pw0319_analysis,
        pw0320_analysis=pw0320_analysis,
    )
    manifest = json.loads(corpus_manifest.read_text())
    if sha256_file(corpus_manifest) != CORPUS_SHA256:
        raise ValueError("PW-0208 manifest mismatch")
    accepted = {int(row["corpus_index"]): int(row["A"]) for row in manifest["primary_windows"]}
    categories = {int(row["corpus_index"]): str(row["category"]) for row in manifest["primary_windows"]}
    if set(accepted) != set(range(32)) or any(not 1 <= value <= 8 for value in accepted.values()):
        raise ValueError("window A authority mismatch")

    safety = HostSafetyMonitor()
    rows, route_sha256, source_hashes = load_rows(corpus_manifest)
    safety.checkpoint("authorities_loaded")
    by_window: dict[int, set[tuple[int, int]]] = defaultdict(set)
    for row in rows:
        by_window[row.corpus_index].update(row.identities)
    universe = canonical_universe()
    universe_index = {identity: index for index, identity in enumerate(universe)}
    category_names = tuple(sorted(set(categories.values())))
    category_indices = {name: index for index, name in enumerate(category_names)}
    occurrence_counts = np.zeros((len(category_names), len(universe)), dtype=np.int16)
    for window_index, identities in by_window.items():
        category_index = category_indices[categories[window_index]]
        for identity in identities:
            occurrence_counts[category_index, universe_index[identity]] += 1

    scenarios = []
    for cache_bytes in CACHE_CAPACITIES:
        for target_tps in CATEGORY_TARGETS:
            scenarios.append(
                scenario(
                    universe=universe,
                    by_window=by_window,
                    categories=categories,
                    accepted=accepted,
                    occurrence_counts=occurrence_counts,
                    category_names=category_names,
                    cache_bytes=cache_bytes,
                    target_tps=target_tps,
                )
            )
    safety.checkpoint("balanced_envelopes_complete")
    candidate = next(
        item
        for item in scenarios
        if item["oracle_cache_bytes"] == PRISMWING1_CANDIDATE_CACHE
        and item["category_target_tps"] == PRISMWING1_CANDIDATE_TARGET
    )
    candidate_gate = {
        "selected_identities_at_most_4096": candidate["selected_identities"] <= MAXIMUM_CANDIDATE_IDENTITIES,
        "every_exact_category_at_least_1_25": all(
            row["optimistic_accepted_tps"] >= PRISMWING1_CANDIDATE_TARGET
            for row in candidate["exact_oracle_category"].values()
        ),
        "nearest_rank_p10_window_at_least_1": (
            candidate["exact_oracle_summary"]["nearest_rank_p10_window_optimistic_accepted_tps"] >= 1.0
        ),
        "installed_hybrid_smaller_than_all_source": (
            candidate["installed_hybrid_expert_bank_bytes"]
            < candidate["all_source_expert_bank_bytes"]
        ),
        "selector_complete": candidate["selector_complete"],
    }
    candidate_gate["pass"] = all(candidate_gate.values())
    safety.release_checkpoint("analysis_released", ["corrected route rows", "selection matrices"])
    safety.checkpoint("final_service_health")
    report = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "status": "complete",
        "decision": (
            "authorize_bounded_six_of_eight_density_falsifier"
            if candidate_gate["pass"]
            else "reject_prismwing1_balanced_k4_envelope"
        ),
        "commit": commit,
        "authority": {
            "corpus_manifest_sha256": CORPUS_SHA256,
            "corrected_route_sha256": route_sha256,
            "source_report_sha256": source_hashes,
            "pw0318_summary_sha256": PW0318_SUMMARY_SHA256,
            "pw0319_analysis_sha256": PW0319_ANALYSIS_SHA256,
            "pw0320_analysis_sha256": PW0320_ANALYSIS_SHA256,
        },
        "constants": {
            "k4_executable_bytes": K4_BYTES,
            "source_executable_bytes": SOURCE_BYTES,
            "cold_storage_bytes_per_second": STORAGE_BYTES_PER_SECOND,
            "storage_source": "PW-0136 two-worker cold median",
            "artifact_bytes_per_identity": ARTIFACT_BYTES_PER_IDENTITY,
            "m1_seconds_per_identity": M1_SECONDS_PER_IDENTITY,
            "m4_seconds_per_identity": M4_SECONDS_PER_IDENTITY,
            "selector_normalization": "initial category byte deficit",
            "selector_tie_break": "canonical layer then expert",
            "conservative_cache_credit": "max(0, capacity - one source record)",
        },
        "candidate_gate": candidate_gate,
        "candidate_selection_order_sha256": candidate["selection_order_sha256"],
        "current_internal_installation_note": (
            "The installed hybrid expert bank is smaller than the all-source expert bank, "
            "but current internal free space cannot hold source and constructed bank "
            "concurrently; this analysis authorizes no relocation or construction."
        ),
        "scenarios": scenarios,
        "safety_snapshots": safety.evidence(),
        "accepted_tokens": 0,
        "performance_claim": None,
    }
    output.mkdir(parents=True)
    path = output / "analysis.json"
    atomic_write_new(path, canonical_json(report))
    print(json.dumps({"output": str(path), "decision": report["decision"]}))
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    for name in (
        "corpus_manifest",
        "pw0318_summary",
        "pw0319_analysis",
        "pw0320_analysis",
        "output",
        "repo",
    ):
        parser.add_argument("--" + name.replace("_", "-"), required=True, type=Path)
    parser.add_argument("--commit", required=True)
    try:
        analyze(**vars(parser.parse_args()))
        return 0
    except (
        FileExistsError,
        HostSafetyViolation,
        KeyError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ) as error:
        print(json.dumps({"error": str(error)}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
