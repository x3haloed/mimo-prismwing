#!/usr/bin/env python3
"""Measure corrected-route K4 bank coverage and emit a bounded work order."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

try:
    from tools.host_safety import HostSafetyMonitor, HostSafetyViolation
    from tools.openrouter_reference import atomic_write_new, canonical_json
    from tools.reproduce_pw0311_k4_expert import verify_clean_commit
except ModuleNotFoundError:
    from host_safety import HostSafetyMonitor, HostSafetyViolation
    from openrouter_reference import atomic_write_new, canonical_json
    from reproduce_pw0311_k4_expert import verify_clean_commit


EXPERIMENT_ID = "PW-0319"
CORPUS_SHA256 = "a9bb6bd26bf048a2144133cc0a96023a8af112eae58122b666915149f2993a7b"
PW0318_SUMMARY_SHA256 = "a91af31bdea45749c9ae9d5d679260bcbcd8284c238479938206a7e7e0b5eb2f"
CHECKPOINT_REVISION = "63651580ca774f8504f676040460aed3e1244ac1"
BUDGETS = (141, 256, 512, 1024, 2048, 4096, 12032)
ROUTED_LAYERS = tuple(range(1, 48))
EXPERTS_PER_LAYER = 256
K4_HITS_REQUIRED = 3
ARTIFACT_BYTES_PER_IDENTITY = 30_000_000
M4_SECONDS_PER_IDENTITY = 183.0
M1_SECONDS_PER_IDENTITY = 500.0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class RouteRow:
    category: str
    corpus_index: int
    position: int
    layer: int
    identities: tuple[tuple[int, int], ...]
    weights: tuple[float, ...]


def load_rows(corpus_manifest: Path) -> tuple[list[RouteRow], str, dict[str, str]]:
    if sha256_file(corpus_manifest) != CORPUS_SHA256:
        raise ValueError("PW-0208 corrected corpus manifest mismatch")
    manifest = json.loads(corpus_manifest.read_text())
    if (
        manifest.get("schema_version") != 1
        or manifest.get("evidence_class")
        != "pw0208_balanced_corrected_native_mtp_window_corpus"
        or manifest.get("builder_git_dirty") is not False
        or len(manifest.get("primary_windows", [])) != 32
    ):
        raise ValueError("PW-0208 corrected corpus contract mismatch")
    sources = {source["category"]: source for source in manifest["sources"]}
    if set(sources) != {"ordinary", "code", "multilingual", "rare_route"}:
        raise ValueError("PW-0208 category authority mismatch")
    reports: dict[str, dict[str, Any]] = {}
    report_hashes: dict[str, str] = {}
    for category, source in sources.items():
        path = Path(source["report_file"])
        observed = sha256_file(path)
        if observed != source["report_sha256"]:
            raise ValueError(f"PW-0208 source report mismatch: {category}")
        reports[category] = json.loads(path.read_text())
        report_hashes[category] = observed
    rows: list[RouteRow] = []
    canonical_rows: list[dict[str, Any]] = []
    category_counts = Counter()
    corpus_indices = set()
    for window in manifest["primary_windows"]:
        category = window["category"]
        corpus_index = int(window["corpus_index"])
        transaction_index = int(window["transaction_index"])
        if category not in reports or corpus_index in corpus_indices:
            raise ValueError("PW-0208 primary-window identity mismatch")
        corpus_indices.add(corpus_index)
        category_counts[category] += 1
        matches = [
            transaction
            for transaction in reports[category]["transactions"]
            if int(transaction["index"]) == transaction_index
        ]
        if len(matches) != 1 or len(matches[0]["verification_layer_traces"]) != 48:
            raise ValueError("PW-0208 verification trace identity mismatch")
        traces = matches[0]["verification_layer_traces"]
        if traces[0]["layer"] != 0 or traces[0]["selected_experts_by_position"] != []:
            raise ValueError("PW-0208 dense layer-0 trace mismatch")
        for layer in ROUTED_LAYERS:
            trace = traces[layer]
            selected = trace["selected_experts_by_position"]
            weights = trace["route_weights_by_position"]
            if int(trace["layer"]) != layer or len(selected) != 8 or len(weights) != 8:
                raise ValueError("PW-0208 corrected routed trace shape mismatch")
            for position, (expert_ids, route_weights) in enumerate(zip(selected, weights)):
                ids = tuple(map(int, expert_ids))
                values = tuple(map(float, route_weights))
                if (
                    len(ids) != 8
                    or len(set(ids)) != 8
                    or any(expert < 0 or expert >= EXPERTS_PER_LAYER for expert in ids)
                    or len(values) != 8
                    or any(not math.isfinite(value) or value <= 0.0 for value in values)
                    or abs(math.fsum(values) - 1.0) > 2.0e-5
                ):
                    raise ValueError("PW-0208 corrected route value mismatch")
                row = RouteRow(
                    category=category,
                    corpus_index=corpus_index,
                    position=position,
                    layer=layer,
                    identities=tuple((layer, expert) for expert in ids),
                    weights=values,
                )
                rows.append(row)
                canonical_rows.append(
                    {
                        "category": category,
                        "corpus_index": corpus_index,
                        "position": position,
                        "layer": layer,
                        "expert_ids": list(ids),
                        "route_weights": list(values),
                    }
                )
    if (
        corpus_indices != set(range(32))
        or any(category_counts[category] != 8 for category in sources)
        or len(rows) != 12_032
        or sum(len(row.identities) for row in rows) != 96_256
    ):
        raise ValueError("PW-0208 corrected route cardinality mismatch")
    authority_sha256 = hashlib.sha256(canonical_json(canonical_rows)).hexdigest()
    return rows, authority_sha256, report_hashes


def coverage_snapshot(
    rows: list[RouteRow],
    selected: set[tuple[int, int]],
    budget: int,
) -> dict[str, Any]:
    hits = [sum(identity in selected for identity in row.identities) for row in rows]
    selected_mass = math.fsum(
        weight
        for row in rows
        for identity, weight in zip(row.identities, row.weights)
        if identity in selected
    )
    total_mass = math.fsum(weight for row in rows for weight in row.weights)
    capped_mass = math.fsum(
        math.fsum(
            sorted(
                (
                    weight
                    for identity, weight in zip(row.identities, row.weights)
                    if identity in selected
                ),
                reverse=True,
            )[:K4_HITS_REQUIRED]
        )
        for row in rows
    )
    category: dict[str, Any] = {}
    for name in sorted({row.category for row in rows}):
        indices = [index for index, row in enumerate(rows) if row.category == name]
        covered = sum(hits[index] >= K4_HITS_REQUIRED for index in indices)
        category[name] = {
            "rows": len(indices),
            "rows_with_at_least_three": covered,
            "coverage_fraction": covered / len(indices),
        }
    layer: dict[str, Any] = {}
    for value in sorted({row.layer for row in rows}):
        indices = [index for index, row in enumerate(rows) if row.layer == value]
        covered = sum(hits[index] >= K4_HITS_REQUIRED for index in indices)
        layer[str(value)] = {
            "rows": len(indices),
            "rows_with_at_least_three": covered,
            "coverage_fraction": covered / len(indices),
        }
    fallback_distribution = Counter(8 - value for value in hits)
    return {
        "budget": budget,
        "selected_identities": len(selected),
        "rows": len(rows),
        "rows_with_at_least_one": sum(value >= 1 for value in hits),
        "rows_with_at_least_two": sum(value >= 2 for value in hits),
        "rows_with_at_least_three": sum(value >= 3 for value in hits),
        "coverage_fraction": sum(value >= 3 for value in hits) / len(rows),
        "selected_route_weight_fraction": selected_mass / total_mass,
        "three_hit_capped_route_weight_fraction": capped_mass / total_mass,
        "source_fallback_count_distribution": {
            str(key): fallback_distribution.get(key, 0) for key in range(9)
        },
        "category": category,
        "layer": layer,
        "minimum_category_coverage_fraction": min(
            value["coverage_fraction"] for value in category.values()
        ),
        "minimum_layer_coverage_fraction": min(
            value["coverage_fraction"] for value in layer.values()
        ),
        "artifact_bytes": budget * ARTIFACT_BYTES_PER_IDENTITY,
        "estimated_m4_construction_seconds": budget * M4_SECONDS_PER_IDENTITY,
        "estimated_m1_construction_seconds": budget * M1_SECONDS_PER_IDENTITY,
    }


def greedy_order(
    rows: list[RouteRow],
    layers: Iterable[int] = ROUTED_LAYERS,
    experts_per_layer: int = EXPERTS_PER_LAYER,
    maximum_budget: int | None = None,
) -> list[tuple[int, int]]:
    layer_values = tuple(layers)
    universe = tuple(
        (layer, expert) for layer in layer_values for expert in range(experts_per_layer)
    )
    if maximum_budget is None:
        maximum_budget = len(universe)
    if maximum_budget < K4_HITS_REQUIRED * len(layer_values) or maximum_budget > len(universe):
        raise ValueError("K4 bank budget is outside the layer-complete bounds")
    occurrences: dict[tuple[int, int], list[tuple[int, float]]] = defaultdict(list)
    for row_index, row in enumerate(rows):
        for identity, weight in zip(row.identities, row.weights):
            occurrences[identity].append((row_index, weight))
    scores = {
        identity: math.fsum(weight for _, weight in occurrences.get(identity, []))
        for identity in universe
    }
    capped_hits = [0] * len(rows)
    selected: set[tuple[int, int]] = set()
    order: list[tuple[int, int]] = []

    def choose(candidates: Iterable[tuple[int, int]]) -> tuple[int, int]:
        available = [identity for identity in candidates if identity not in selected]
        if not available:
            raise ValueError("K4 coverage planner exhausted candidates")
        return max(
            available,
            key=lambda identity: (
                scores[identity],
                sum(
                    capped_hits[row_index] == K4_HITS_REQUIRED - 1
                    for row_index, _ in occurrences.get(identity, [])
                ),
                len(occurrences.get(identity, [])),
                -identity[0],
                -identity[1],
            ),
        )

    def apply(identity: tuple[int, int]) -> None:
        selected.add(identity)
        order.append(identity)
        for row_index, _ in occurrences.get(identity, []):
            if capped_hits[row_index] >= K4_HITS_REQUIRED:
                continue
            capped_hits[row_index] += 1
            if capped_hits[row_index] == K4_HITS_REQUIRED:
                row = rows[row_index]
                for affected, weight in zip(row.identities, row.weights):
                    scores[affected] -= weight
                    if abs(scores[affected]) < 1.0e-12:
                        scores[affected] = 0.0

    for layer in layer_values:
        candidates = ((layer, expert) for expert in range(experts_per_layer))
        for _ in range(K4_HITS_REQUIRED):
            apply(choose(candidates))
            candidates = ((layer, expert) for expert in range(experts_per_layer))
    while len(order) < maximum_budget:
        if max(scores[identity] for identity in universe if identity not in selected) <= 0.0:
            for identity in universe:
                if identity not in selected and len(order) < maximum_budget:
                    apply(identity)
            break
        apply(choose(universe))
    return order


def analyze(
    *,
    corpus_manifest: Path,
    pw0318_summary: Path,
    output: Path,
    repo: Path,
    commit: str,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(output)
    verify_clean_commit(repo.resolve(), commit)
    if sha256_file(pw0318_summary) != PW0318_SUMMARY_SHA256:
        raise ValueError("PW-0318 boundary summary mismatch")
    safety = HostSafetyMonitor()
    rows, route_authority_sha256, report_hashes = load_rows(corpus_manifest)
    safety.checkpoint("corrected_route_authority_loaded")
    order = greedy_order(rows, maximum_budget=max(BUDGETS))
    curves = []
    for budget in BUDGETS:
        curves.append(coverage_snapshot(rows, set(order[:budget]), budget))
    safety.checkpoint("coverage_curve_complete")
    qualifying = [
        row
        for row in curves
        if row["coverage_fraction"] >= 0.5
        and row["minimum_category_coverage_fraction"] >= 0.4
        and row["minimum_layer_coverage_fraction"] >= 0.25
    ]
    smallest_qualifying_budget = qualifying[0]["budget"] if qualifying else None
    bounded_budget = (
        smallest_qualifying_budget
        if smallest_qualifying_budget is not None and smallest_qualifying_budget <= 512
        else None
    )
    output.mkdir(parents=True)
    work_order_record = None
    if bounded_budget is not None:
        reserve_count = math.ceil(bounded_budget * 0.2)
        identities = order[: bounded_budget + reserve_count]
        work_order = {
            "schema_version": 1,
            "experiment_id": EXPERIMENT_ID,
            "semantic": "resumable_corrected_route_weighted_k4_construction_work_order",
            "commit": commit,
            "checkpoint_revision": CHECKPOINT_REVISION,
            "representation_revision": "m1-native-k4-v1",
            "primary_count": bounded_budget,
            "reserve_count": reserve_count,
            "route_authority_sha256": route_authority_sha256,
            "items": [
                {
                    "sequence": index,
                    "cohort": "primary" if index <= bounded_budget else "reserve",
                    "layer": layer,
                    "expert": expert,
                    "status": "pending",
                    "output_directory": (
                        f"/Volumes/Elements/mimo-prismwing/evidence/PW-0319/"
                        f"construction/layer-{layer:02d}-expert-{expert:03d}"
                    ),
                    "authorities": {
                        "checkpoint_revision": CHECKPOINT_REVISION,
                        "corrected_route_authority_sha256": route_authority_sha256,
                        "pw0318_summary_sha256": PW0318_SUMMARY_SHA256,
                    },
                    "required_gates": [
                        "target_native_repeatability",
                        "identity_local_source_distance",
                        "cumulative_route_distance",
                        "gate8_release",
                    ],
                    "result": None,
                }
                for index, (layer, expert) in enumerate(identities, start=1)
            ],
        }
        path = output / "work-order.json"
        atomic_write_new(path, canonical_json(work_order))
        work_order_record = {
            "file": path.name,
            "sha256": sha256_file(path),
            "primary_count": bounded_budget,
            "reserve_count": reserve_count,
        }
    safety.release_checkpoint("analysis_buffers_released", ["corrected route rows", "coverage planner"])
    safety.checkpoint("final_service_health")
    decision = (
        "emit_bounded_m4_construction_tranche"
        if bounded_budget is not None
        else "reject_bounded_512_identity_tranche"
    )
    report = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "status": "corrected_route_k4_coverage_curve_complete",
        "decision": decision,
        "commit": commit,
        "authority": {
            "corpus_manifest_sha256": CORPUS_SHA256,
            "source_report_sha256": report_hashes,
            "route_authority_sha256": route_authority_sha256,
            "pw0318_summary_sha256": PW0318_SUMMARY_SHA256,
            "rows": len(rows),
            "placements": sum(len(row.identities) for row in rows),
        },
        "planner": {
            "objective": "route_weighted_marginal_until_three_hits_per_row",
            "minimum_identities_per_layer": K4_HITS_REQUIRED,
            "tie_break": ["newly_completed_rows", "placement_count", "lower_layer", "lower_expert"],
            "budgets": list(BUDGETS),
        },
        "coverage_curve": curves,
        "smallest_measured_qualifying_budget": smallest_qualifying_budget,
        "bounded_qualifying_budget": bounded_budget,
        "work_order": work_order_record,
        "cost_constants": {
            "artifact_bytes_per_identity": ARTIFACT_BYTES_PER_IDENTITY,
            "m4_seconds_per_identity": M4_SECONDS_PER_IDENTITY,
            "m1_seconds_per_identity": M1_SECONDS_PER_IDENTITY,
            "diagnostic_only": True,
        },
        "safety_snapshots": safety.evidence(),
        "batch_size": 1,
        "concurrency": 1,
        "accepted_tokens": 0,
        "performance_claim": None,
        "claims_excluded": [
            "new identity semantic qualification",
            "complete bank",
            "modality coverage",
            "endpoint execution",
            "accepted-token TPS",
            "Prismwing completion",
        ],
    }
    analysis_path = output / "analysis.json"
    atomic_write_new(analysis_path, canonical_json(report))
    print(json.dumps({"output": str(analysis_path), "decision": decision}))
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus-manifest", required=True, type=Path)
    parser.add_argument("--pw0318-summary", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--commit", required=True)
    args = parser.parse_args()
    try:
        analyze(**vars(args))
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
