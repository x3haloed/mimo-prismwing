#!/usr/bin/env python3
"""Analyze PW-0212 bounded prefetch controls on corrected verifier routes."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import subprocess
from typing import Any

try:
    from tools.openrouter_reference import atomic_write_new, canonical_json
except ModuleNotFoundError:
    from openrouter_reference import atomic_write_new, canonical_json


CORPUS_SHA256 = "a9bb6bd26bf048a2144133cc0a96023a8af112eae58122b666915149f2993a7b"
EXPERT_SOURCE_BYTES = 25_171_968
COLD_EIGHT_EXPERT_ACQUISITION_MS = 58.033833
EXPERT_ACQUISITION_MS = COLD_EIGHT_EXPERT_ACQUISITION_MS / 8.0
MAXIMUM_PREFETCH_TAX = 0.25
REQUIRED_COMPLETE_WALL_HIDDEN = 0.10
CATEGORIES = ("ordinary", "code", "multilingual", "rare_route")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_route(route: Any) -> tuple[int, ...]:
    if (
        not isinstance(route, list)
        or len(route) != 8
        or len(set(route)) != 8
        or any(not isinstance(expert, int) or not 0 <= expert < 256 for expert in route)
    ):
        raise ValueError("corrected route must contain eight unique expert IDs")
    return tuple(route)


def authenticate_windows(manifest_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if sha256_file(manifest_path) != CORPUS_SHA256:
        raise ValueError("PW-0208 corrected corpus hash mismatch")
    manifest = json.loads(manifest_path.read_text())
    if (
        manifest.get("schema_version") != 1
        or manifest.get("evidence_class")
        != "pw0208_balanced_corrected_native_mtp_window_corpus"
        or manifest.get("verifier_window_shape") != [8, 4096]
        or len(manifest.get("primary_windows", [])) != 32
    ):
        raise ValueError("PW-0208 corrected corpus identity mismatch")

    reports: dict[str, dict[str, Any]] = {}
    for source in [*manifest["sources"], *manifest["prefill_sources"]]:
        if source.get("category") not in CATEGORIES:
            raise ValueError("unknown corrected corpus category")
        for kind in ("report", "progress", "hidden"):
            file_key, hash_key = f"{kind}_file", f"{kind}_sha256"
            if file_key in source and sha256_file(Path(source[file_key])) != source[hash_key]:
                raise ValueError(f"PW-0208 {source['category']} {kind} hash mismatch")
        if source in manifest["sources"]:
            reports[source["category"]] = json.loads(Path(source["report_file"]).read_text())

    windows = []
    seen: dict[str, list[int]] = defaultdict(list)
    for authority in manifest["primary_windows"]:
        category = authority.get("category")
        transaction_index = authority.get("transaction_index")
        report = reports.get(category)
        if (
            report is None
            or not isinstance(transaction_index, int)
            or transaction_index < 0
            or transaction_index >= len(report.get("transactions", []))
        ):
            raise ValueError("window transaction authority mismatch")
        transaction = report["transactions"][transaction_index]
        if (
            transaction.get("index") != transaction_index
            or transaction.get("proposal_token_ids") != authority.get("proposal_token_ids")
            or transaction.get("posterior_token_ids") != authority.get("posterior_token_ids")
            or transaction.get("verifier_authorized_token_ids")
            != authority.get("verifier_authorized_token_ids")
            or not math.isclose(transaction.get("proposal_wall_ms", -1), authority["proposal_wall_ms"])
            or not math.isclose(
                transaction.get("verification_wall_ms", -1), authority["verification_wall_ms"]
            )
        ):
            raise ValueError("window/report transaction mismatch")
        traces = transaction.get("verification_layer_traces")
        if not isinstance(traces, list) or len(traces) != 48:
            raise ValueError("verifier route trace layer count mismatch")
        routes: dict[int, tuple[tuple[int, ...], ...]] = {}
        walls: dict[int, float] = {}
        for layer, trace in enumerate(traces):
            if trace.get("layer") != layer or not isinstance(trace.get("wall_ms"), (int, float)):
                raise ValueError("verifier route trace identity mismatch")
            if layer == 0:
                if trace.get("selected_experts_by_position") != []:
                    raise ValueError("dense layer zero unexpectedly routed")
                continue
            layer_routes = trace.get("selected_experts_by_position")
            if not isinstance(layer_routes, list) or len(layer_routes) != 8:
                raise ValueError("verifier route trace position count mismatch")
            routes[layer] = tuple(_validate_route(route) for route in layer_routes)
            walls[layer] = float(trace["wall_ms"])
        seen[category].append(transaction_index)
        windows.append({
            "corpus_index": authority["corpus_index"],
            "category": category,
            "transaction_index": transaction_index,
            "proposal_wall_ms": float(authority["proposal_wall_ms"]),
            "verification_wall_ms": float(authority["verification_wall_ms"]),
            "routes": routes,
            "layer_wall_ms": walls,
        })
    if any(seen[category] != list(range(1, 9)) for category in CATEGORIES):
        raise ValueError("corrected corpus chronology mismatch")
    return manifest, windows


def split_windows(windows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    calibration = [window for window in windows if window["transaction_index"] <= 4]
    holdout = [window for window in windows if window["transaction_index"] >= 5]
    if len(calibration) != 16 or len(holdout) != 16:
        raise ValueError("calibration/holdout split mismatch")
    return calibration, holdout


def route_frequencies(windows: list[dict[str, Any]]) -> dict[int, Counter[int]]:
    result: dict[int, Counter[int]] = {layer: Counter() for layer in range(1, 48)}
    for window in windows:
        for layer in range(1, 48):
            for route in window["routes"][layer]:
                result[layer].update(route)
    return result


def category_route_frequencies(
    windows: list[dict[str, Any]],
) -> dict[str, dict[int, Counter[int]]]:
    return {
        category: route_frequencies([window for window in windows if window["category"] == category])
        for category in CATEGORIES
    }


def top_experts(counts: Counter[int], count: int = 8) -> tuple[int, ...]:
    return tuple(expert for expert, _ in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:count])


def predictor_routes(
    name: str,
    window: dict[str, Any],
    layer: int,
    prior_window: dict[str, Any] | None,
    global_frequency: dict[int, Counter[int]],
    category_frequency: dict[str, dict[int, Counter[int]]],
) -> tuple[tuple[int, ...], ...]:
    actual = window["routes"][layer]
    if name == "same_layer_previous_position":
        return ((), *actual[:-1])
    if name == "previous_layer_same_position":
        return ((),) * 8 if layer == 1 else window["routes"][layer - 1]
    if name == "previous_transaction_same_position":
        return ((),) * 8 if prior_window is None else prior_window["routes"][layer]
    if name == "last_route":
        seed = () if prior_window is None else prior_window["routes"][layer][-1]
        return (seed, *actual[:-1])
    if name == "global_frequency":
        route = top_experts(global_frequency[layer])
        return (route,) * 8
    if name == "category_frequency":
        route = top_experts(category_frequency[window["category"]][layer])
        return (route,) * 8
    if name == "offline_future_oracle":
        return actual
    raise ValueError(f"unknown predictor {name}")


def _empty_metrics() -> dict[str, int]:
    return {
        "events": 0,
        "events_with_prediction": 0,
        "demand_selections": 0,
        "predicted_selections": 0,
        "matching_selections": 0,
        "demand_unique_records": 0,
        "predicted_unique_records": 0,
        "useful_unique_records": 0,
        "wasted_unique_records": 0,
        "duplicated_prediction_records": 0,
    }


def _finalize_metrics(metrics: dict[str, int]) -> dict[str, Any]:
    demand = metrics["demand_selections"]
    predicted = metrics["predicted_selections"]
    matching = metrics["matching_selections"]
    result: dict[str, Any] = dict(metrics)
    result.update({
        "recall_at_8": matching / demand if demand else 0.0,
        "precision": matching / predicted if predicted else 0.0,
        "bytes_prefetched": metrics["predicted_unique_records"] * EXPERT_SOURCE_BYTES,
        "useful_bytes": metrics["useful_unique_records"] * EXPERT_SOURCE_BYTES,
        "late_bytes": (
            metrics["demand_unique_records"] - metrics["useful_unique_records"]
        ) * EXPERT_SOURCE_BYTES,
        "wasted_bytes": metrics["wasted_unique_records"] * EXPERT_SOURCE_BYTES,
        "duplicated_bytes": metrics["duplicated_prediction_records"] * EXPERT_SOURCE_BYTES,
    })
    return result


def evaluate_predictor(
    name: str,
    calibration: list[dict[str, Any]],
    holdout: list[dict[str, Any]],
) -> dict[str, Any]:
    global_frequency = route_frequencies(calibration)
    category_frequency = category_route_frequencies(calibration)
    prior = {
        category: max(
            (window for window in calibration if window["category"] == category),
            key=lambda window: window["transaction_index"],
        )
        for category in CATEGORIES
    }
    aggregate = _empty_metrics()
    by_category = {category: _empty_metrics() for category in CATEGORIES}
    for window in holdout:
        for layer in range(1, 48):
            predictions = predictor_routes(
                name, window, layer, prior.get(window["category"]), global_frequency,
                category_frequency,
            )
            actual = window["routes"][layer]
            demand_union = set().union(*map(set, actual))
            predicted_union = set().union(*map(set, predictions))
            useful_union = demand_union & predicted_union
            targets = (aggregate, by_category[window["category"]])
            for target in targets:
                target["demand_unique_records"] += len(demand_union)
                target["predicted_unique_records"] += len(predicted_union)
                target["useful_unique_records"] += len(useful_union)
                target["wasted_unique_records"] += len(predicted_union - demand_union)
                target["duplicated_prediction_records"] += (
                    sum(len(route) for route in predictions) - len(predicted_union)
                )
            for predicted, demanded in zip(predictions, actual):
                matches = len(set(predicted) & set(demanded))
                for target in targets:
                    target["events"] += 1
                    target["events_with_prediction"] += bool(predicted)
                    target["demand_selections"] += 8
                    target["predicted_selections"] += len(predicted)
                    target["matching_selections"] += matches
        prior[window["category"]] = window
    return {
        "aggregate": _finalize_metrics(aggregate),
        "by_category": {
            category: _finalize_metrics(metrics) for category, metrics in by_category.items()
        },
    }


def disagreement_count(
    calibration: list[dict[str, Any]], holdout: list[dict[str, Any]]
) -> dict[str, int]:
    names = (
        "last_route",
        "same_layer_previous_position",
        "previous_layer_same_position",
        "previous_transaction_same_position",
        "global_frequency",
    )
    global_frequency = route_frequencies(calibration)
    category_frequency = category_route_frequencies(calibration)
    prior = {
        category: max(
            (window for window in calibration if window["category"] == category),
            key=lambda window: window["transaction_index"],
        )
        for category in CATEGORIES
    }
    total = 0
    differing = 0
    for window in holdout:
        for layer in range(1, 48):
            controls = [
                predictor_routes(
                    name, window, layer, prior[window["category"]], global_frequency,
                    category_frequency,
                )
                for name in names
            ]
            for position in range(8):
                predictions = {controls[index][position] for index in range(len(names))}
                total += 1
                differing += len(predictions) > 1
        prior[window["category"]] = window
    return {"holdout_events": total, "events_where_controls_disagree": differing}


def bounded_oracle_model(windows: list[dict[str, Any]]) -> dict[str, Any]:
    records = []
    for window in windows:
        demand_records = 0
        prefetched_records = 0
        hidden_ms = 0.0
        for layer in range(1, 48):
            demand = len(set().union(*map(set, window["routes"][layer])))
            budget = math.floor(demand * MAXIMUM_PREFETCH_TAX)
            lead_ms = (
                window["proposal_wall_ms"] if layer == 1 else window["layer_wall_ms"][layer - 1]
            )
            demand_records += demand
            prefetched_records += budget
            hidden_ms += min(lead_ms, budget * EXPERT_ACQUISITION_MS)
        complete_ms = window["proposal_wall_ms"] + window["verification_wall_ms"]
        records.append({
            "corpus_index": window["corpus_index"],
            "category": window["category"],
            "transaction_index": window["transaction_index"],
            "demand_records": demand_records,
            "prefetched_records": prefetched_records,
            "prefetch_bandwidth_tax": prefetched_records / demand_records,
            "optimistic_hidden_acquisition_ms": hidden_ms,
            "verification_wall_ms": window["verification_wall_ms"],
            "complete_transaction_wall_ms": complete_ms,
            "verification_wall_hidden_fraction": hidden_ms / window["verification_wall_ms"],
            "complete_wall_hidden_fraction": hidden_ms / complete_ms,
        })
    demand = sum(record["demand_records"] for record in records)
    prefetched = sum(record["prefetched_records"] for record in records)
    hidden = sum(record["optimistic_hidden_acquisition_ms"] for record in records)
    verification = sum(record["verification_wall_ms"] for record in records)
    complete = sum(record["complete_transaction_wall_ms"] for record in records)
    by_category = {}
    for category in CATEGORIES:
        subset = [record for record in records if record["category"] == category]
        subset_hidden = sum(record["optimistic_hidden_acquisition_ms"] for record in subset)
        subset_complete = sum(record["complete_transaction_wall_ms"] for record in subset)
        by_category[category] = {
            "windows": len(subset),
            "complete_wall_hidden_fraction": subset_hidden / subset_complete,
            "maximum_window_complete_wall_hidden_fraction": max(
                record["complete_wall_hidden_fraction"] for record in subset
            ),
        }
    return {
        "records": records,
        "aggregate": {
            "windows": len(records),
            "demand_records": demand,
            "prefetched_records": prefetched,
            "prefetch_bandwidth_tax": prefetched / demand,
            "optimistic_hidden_acquisition_ms": hidden,
            "verification_wall_hidden_fraction": hidden / verification,
            "complete_wall_hidden_fraction": hidden / complete,
            "maximum_window_complete_wall_hidden_fraction": max(
                record["complete_wall_hidden_fraction"] for record in records
            ),
        },
        "by_category": by_category,
    }


def source_control(expected_commit: str) -> None:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, text=True, capture_output=True
    ).stdout.strip()
    dirty = bool(subprocess.run(
        ["git", "status", "--porcelain"], check=True, text=True, capture_output=True
    ).stdout.strip())
    if dirty or commit != expected_commit:
        raise ValueError("PW-0212 requires the declared clean commit")


def analyze(manifest_path: Path, commit: str) -> dict[str, Any]:
    source_control(commit)
    manifest, windows = authenticate_windows(manifest_path)
    calibration, holdout = split_windows(windows)
    predictor_names = (
        "last_route",
        "same_layer_previous_position",
        "previous_layer_same_position",
        "previous_transaction_same_position",
        "global_frequency",
        "category_frequency",
        "offline_future_oracle",
    )
    controls = {
        name: evaluate_predictor(name, calibration, holdout) for name in predictor_names
    }
    oracle = bounded_oracle_model(holdout)
    oracle_passes = (
        oracle["aggregate"]["complete_wall_hidden_fraction"]
        >= REQUIRED_COMPLETE_WALL_HIDDEN
        and oracle["aggregate"]["prefetch_bandwidth_tax"] <= MAXIMUM_PREFETCH_TAX
    )
    return {
        "schema_version": 1,
        "evidence_class": "pw0212_corrected_route_predictive_prefetch_oracle",
        "status": "complete",
        "implementation": {"commit": commit, "dirty": False},
        "identities": {
            "pw0208_corpus_sha256": CORPUS_SHA256,
            "source_report_sha256": {
                source["category"]: source["report_sha256"] for source in manifest["sources"]
            },
        },
        "constants": {
            "expert_source_bytes": EXPERT_SOURCE_BYTES,
            "cold_eight_expert_acquisition_ms": COLD_EIGHT_EXPERT_ACQUISITION_MS,
            "cold_single_expert_acquisition_ms": EXPERT_ACQUISITION_MS,
            "maximum_prefetch_bandwidth_tax": MAXIMUM_PREFETCH_TAX,
            "required_complete_wall_hidden_fraction": REQUIRED_COMPLETE_WALL_HIDDEN,
        },
        "split": {
            "semantic": "first four chronological windows per category calibrate; final four hold out",
            "calibration_corpus_indices": [window["corpus_index"] for window in calibration],
            "holdout_corpus_indices": [window["corpus_index"] for window in holdout],
        },
        "discrimination": disagreement_count(calibration, holdout),
        "logical_controls": controls,
        "bandwidth_bounded_offline_future_oracle": oracle,
        "gates": {
            "controls_discriminate": disagreement_count(calibration, holdout)[
                "events_where_controls_disagree"
            ] > 0,
            "offline_oracle_implementation_gate_passed": oracle_passes,
        },
        "decision": (
            "authorize_bounded_runtime_prefetch_pilot" if oracle_passes
            else "reject_runtime_prefetch_under_frozen_tax_and_complete_wall_gate"
        ),
        "accepted_tokens": 0,
        "performance_claim": None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    report = analyze(arguments.manifest, arguments.commit)
    atomic_write_new(arguments.output, canonical_json(report))
    print(canonical_json(report).decode(), end="")


if __name__ == "__main__":
    main()
