#!/usr/bin/env python3
"""Authenticate repeated PW-0315 identities and gate their cumulative bank."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

try:
    from tools.construct_pw0314_layer4_k4 import (
        CHECKPOINT_RECEIPT_SHA256,
        CORPUS_SHA256,
        HIDDEN,
        LAYER,
        PARTITIONS,
        load_capture,
        metrics_pass,
        reconstruct_route,
        selected_rows,
        sliced_metrics,
    )
    from tools.construct_pw0315_layer4_expert import (
        EXPERIMENT_ID,
        EXPERT64_CONTROL,
        EXPECTED_PLACEMENTS,
    )
    from tools.openrouter_reference import atomic_write_new, canonical_json
    from tools.reproduce_pw0311_k4_expert import (
        _load_authority_modules,
        array_sha256,
        authority_paths,
        sha256_file,
        verify_clean_commit,
    )
except ModuleNotFoundError:
    from construct_pw0314_layer4_k4 import (
        CHECKPOINT_RECEIPT_SHA256,
        CORPUS_SHA256,
        HIDDEN,
        LAYER,
        PARTITIONS,
        load_capture,
        metrics_pass,
        reconstruct_route,
        selected_rows,
        sliced_metrics,
    )
    from construct_pw0315_layer4_expert import (
        EXPERIMENT_ID,
        EXPERT64_CONTROL,
        EXPECTED_PLACEMENTS,
    )
    from openrouter_reference import atomic_write_new, canonical_json
    from reproduce_pw0311_k4_expert import (
        _load_authority_modules,
        array_sha256,
        authority_paths,
        sha256_file,
        verify_clean_commit,
    )


EXPERTS = (64, 96, 31, 232)


def stable_projections(projections: dict[str, Any]) -> dict[str, Any]:
    return {
        name: {key: value for key, value in row.items() if key != "quantization_seconds"}
        for name, row in projections.items()
    }


def replace_expert_outputs(
    outputs: np.ndarray,
    layer_row: dict[str, Any],
    expert: int,
    candidate: np.ndarray,
) -> np.ndarray:
    result = np.asarray(outputs, dtype=np.float32).copy()
    _, _, _, offsets = selected_rows(layer_row, expert)
    replacement = np.asarray(candidate, dtype=np.float32)
    if replacement.shape != (len(offsets), result.shape[1]):
        raise ValueError(f"expert-{expert} candidate output shape mismatch")
    result[offsets] = replacement
    return result


def load_report(path: Path, expert: int, commit: str) -> tuple[dict[str, Any], str]:
    report_path = path / "construction.json"
    digest = sha256_file(report_path)
    report = json.loads(report_path.read_text())
    if (
        report.get("schema_version") != 1
        or report.get("experiment_id") != EXPERIMENT_ID
        or report.get("layer") != LAYER
        or report.get("expert") != expert
        or report.get("commit") != commit
        or report.get("failure") is not None
        or report.get("accepted_tokens") != 0
        or report.get("performance_claim") is not None
        or report.get("semantic", {}).get("gates", {}).get("pass") is not True
        or report.get("semantic", {}).get("placements", {}).get("total")
        != EXPECTED_PLACEMENTS[expert]
    ):
        raise ValueError(f"expert-{expert} construction report contract mismatch")
    return report, digest


def analyze(
    *,
    evidence_root: Path,
    corpus_manifest: Path,
    authority_root: Path,
    repo: Path,
    commit: str,
    output: Path,
) -> dict[str, Any]:
    verify_clean_commit(repo.resolve(), commit)
    if sha256_file(corpus_manifest) != CORPUS_SHA256:
        raise ValueError("PW-0116 corpus manifest mismatch")
    corpus = json.loads(corpus_manifest.read_text())
    layer_rows = [row for row in corpus["layers"] if int(row["layer"]) == LAYER]
    if len(layer_rows) != 1:
        raise ValueError("PW-0116 layer-4 authority mismatch")
    layer_row = layer_rows[0]
    modules = _load_authority_modules(authority_paths(authority_root.resolve()))
    corpus_root = corpus_manifest.parent
    expert_down = load_capture(corpus_root, layer_row, "expert_down")
    source_routed = load_capture(corpus_root, layer_row, "routed_output")
    post_attention = load_capture(corpus_root, layer_row, "post_attention")
    source_final = load_capture(corpus_root, layer_row, "final")
    if not np.array_equal(
        reconstruct_route(expert_down, layer_row, modules["panel"].bf16),
        source_routed,
    ):
        raise ValueError("source route reconstruction mismatch")
    if not np.array_equal(
        modules["panel"].bf16(post_attention + source_routed), source_final
    ):
        raise ValueError("source final reconstruction mismatch")

    candidate_down = expert_down.copy()
    identity_reports: dict[str, Any] = {}
    all_snapshots: list[dict[str, Any]] = []
    for expert in EXPERTS:
        runs = []
        reports = []
        digests = []
        for run in (1, 2):
            run_path = evidence_root / f"expert-{expert:03d}-run-{run:03d}"
            report, digest = load_report(run_path, expert, commit)
            candidate_path = (
                run_path
                / f"layer-{LAYER:02d}-expert-{expert:03d}"
                / "candidate-output.f32le"
            )
            positions, _, _, _ = selected_rows(layer_row, expert)
            candidate = np.fromfile(candidate_path, dtype="<f4").reshape(
                len(positions), HIDDEN
            )
            if array_sha256(candidate) != report["semantic"]["array_sha256"]["candidate_output_f32"]:
                raise ValueError(f"expert-{expert} candidate output hash mismatch")
            runs.append((report, candidate))
            reports.append(report)
            digests.append(digest)
            all_snapshots.extend(report.get("safety_snapshots", []))
        first, second = runs
        if (
            first[0]["deterministic_tree"] != second[0]["deterministic_tree"]
            or stable_projections(first[0]["projections"])
            != stable_projections(second[0]["projections"])
            or first[0]["semantic"] != second[0]["semantic"]
            or not np.array_equal(first[1], second[1])
        ):
            raise ValueError(f"expert-{expert} fresh-process repeat mismatch")
        if expert == 64:
            for name, expected in EXPERT64_CONTROL.items():
                for field, digest in expected.items():
                    if first[0]["projections"][name].get(field) != digest:
                        raise ValueError(f"expert-64 immutable control mismatch: {name}.{field}")
        candidate_down = replace_expert_outputs(
            candidate_down, layer_row, expert, first[1]
        )
        identity_reports[str(expert)] = {
            "report_sha256": digests,
            "deterministic_tree": first[0]["deterministic_tree"],
            "projections": stable_projections(first[0]["projections"]),
            "semantic": first[0]["semantic"],
            "complete_seconds": [row["complete_seconds"] for row in reports],
            "peak_rss_bytes": [row["peak_rss_bytes"] for row in reports],
        }

    candidate_route = reconstruct_route(
        candidate_down, layer_row, modules["panel"].bf16
    )
    candidate_final = modules["panel"].bf16(post_attention + candidate_route)
    route_metrics = sliced_metrics(source_routed, candidate_route)
    final_metrics = sliced_metrics(source_final, candidate_final)
    cumulative_pass = metrics_pass(route_metrics) and metrics_pass(final_metrics)
    release_snapshots = [
        row
        for row in all_snapshots
        if row.get("phase") == "construction_buffers_released"
    ]
    result = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "status": "qualified" if cumulative_pass else "cumulative_semantic_gate_failed",
        "decision": "authorize_bounded_bank_expansion" if cumulative_pass else "stop_geometric_bank_expansion",
        "commit": commit,
        "authority": {
            "checkpoint_receipt_sha256": CHECKPOINT_RECEIPT_SHA256,
            "corpus_sha256": CORPUS_SHA256,
            "experts": list(EXPERTS),
        },
        "identities": identity_reports,
        "cumulative": {
            "route_candidate_vs_source": route_metrics,
            "final_candidate_vs_source": final_metrics,
            "array_sha256": {
                "candidate_route_f32": array_sha256(candidate_route),
                "candidate_final_f32": array_sha256(candidate_final),
            },
            "gates": {
                "route_pass": metrics_pass(route_metrics),
                "final_pass": metrics_pass(final_metrics),
                "pass": cumulative_pass,
            },
        },
        "safety_summary": {
            "minimum_memory_free_percent": min(
                float(row["system_memory_free_percent"]) for row in all_snapshots
            ),
            "maximum_process_footprint_bytes": max(
                int(row["process_physical_footprint_bytes"])
                for row in all_snapshots
            ),
            "maximum_peak_rss_bytes": max(
                value
                for row in identity_reports.values()
                for value in row["peak_rss_bytes"]
            ),
            "maximum_release_footprint_bytes": max(
                int(row["process_physical_footprint_bytes"])
                for row in release_snapshots
            ),
            "maximum_swap_growth_bytes": max(
                int(row["swap_growth_bytes"]) for row in all_snapshots
            ),
            "maximum_new_throttled_pages": max(
                int(row["new_throttled_pages"]) for row in all_snapshots
            ),
        },
        "batch_size": 1,
        "concurrency": 1,
        "accepted_tokens": 0,
        "A": 0,
        "U": 0,
        "performance_claim": None,
    }
    atomic_write_new(output, canonical_json(result))
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence-root", required=True, type=Path)
    parser.add_argument("--corpus-manifest", required=True, type=Path)
    parser.add_argument("--authority-root", required=True, type=Path)
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    try:
        result = analyze(**vars(arguments))
        print(json.dumps({"output": str(arguments.output), "status": result["status"]}))
        return 0
    except (FileExistsError, KeyError, OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
        print(json.dumps({"error": str(error)}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
