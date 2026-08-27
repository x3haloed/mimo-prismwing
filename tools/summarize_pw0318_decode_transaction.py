#!/usr/bin/env python3
"""Validate and summarize the two PW-0318 decode-authority runs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

try:
    from tools.openrouter_reference import atomic_write_new, canonical_json
except ModuleNotFoundError:
    from openrouter_reference import atomic_write_new, canonical_json


EXPERIMENT_ID = "PW-0318"
BUNDLE_CONTROL_SHA256 = "e87a0af2aba57f46b6a2f394d70e530533d04c18aa61650afbc8528a4b8bdc35"
IDENTICAL_FILES = (
    "build-spec.json",
    "layer04-position001.k4-source.bin",
    "layer04-position001.k4-source.manifest.json",
    "layer04-position001.fixture.json",
    "loader-verification.json",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safety_extrema(reports: list[dict[str, Any]]) -> dict[str, int]:
    snapshots = [snapshot for report in reports for snapshot in report["safety_snapshots"]]
    if not snapshots or not any(snapshot["release_boundary"] for snapshot in snapshots):
        raise ValueError("PW-0318 safety evidence lacks a release boundary")
    return {
        "minimum_system_memory_free_percent": min(
            int(snapshot["system_memory_free_percent"]) for snapshot in snapshots
        ),
        "maximum_process_peak_resident_bytes": max(
            int(snapshot["process_peak_resident_bytes"]) for snapshot in snapshots
        ),
        "maximum_process_physical_footprint_bytes": max(
            int(snapshot["process_physical_footprint_bytes"]) for snapshot in snapshots
        ),
        "maximum_swap_growth_bytes": max(
            int(snapshot["swap_growth_bytes"]) for snapshot in snapshots
        ),
        "maximum_new_throttled_pages": max(
            int(snapshot["new_throttled_pages"]) for snapshot in snapshots
        ),
    }


def summarize(
    run_roots: tuple[Path, Path],
    output: Path,
    implementation_commit: str,
    analysis_commit: str,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(output)
    if len(implementation_commit) != 40 or len(analysis_commit) != 40:
        raise ValueError("PW-0318 commits must be full Git object IDs")
    runs = []
    identical: dict[str, str] = {}
    for index, root in enumerate(run_roots, start=1):
        build_path = root / "build.json"
        loader_path = root / "loader-verification.json"
        metal_path = root / "metal-run.json"
        build = json.loads(build_path.read_text())
        loader = json.loads(loader_path.read_text())
        metal = json.loads(metal_path.read_text())
        if (
            build.get("experiment_id") != EXPERIMENT_ID
            or build.get("status") != "layer4_three_five_decode_bundle_ready"
            or build.get("commit") != implementation_commit
            or loader.get("commit") != implementation_commit
            or loader.get("status") != "modified_k4_source_candidate_bundle_verified"
            or metal.get("commit") != implementation_commit
            or metal.get("status") != "modified_k4_source_layer_candidate_fixture_bitexact"
            or metal.get("device") != "Apple M1"
            or metal.get("warmups") != 20
            or metal.get("samples") != 100
            or metal.get("unequal_count") != 0
            or metal.get("relative_l2") != 0.0
            or metal.get("maximum_absolute_error") != 0.0
            or metal.get("candidate_route_gate_pass") is not True
        ):
            raise ValueError(f"PW-0318 run-{index:03d} contract mismatch")
        semantic = build["semantic"]
        if (
            semantic.get("metal_answer_key") != "decode_one_row"
            or semantic["route_candidate_vs_source"]["relative_l2"] >= 0.01
            or semantic["final_candidate_vs_source"]["relative_l2"] >= 0.01
            or semantic["decode_route_candidate_vs_source"]["relative_l2"] >= 0.01
            or semantic["decode_final_candidate_vs_source"]["relative_l2"] >= 0.01
        ):
            raise ValueError(f"PW-0318 run-{index:03d} source-distance gate mismatch")
        file_hashes = {
            name: sha256_file(root / name)
            for name in (*IDENTICAL_FILES, "build.json", "metal-run.json")
        }
        if file_hashes["layer04-position001.k4-source.bin"] != BUNDLE_CONTROL_SHA256:
            raise ValueError("PW-0318 bundle lost the PW-0317 payload control")
        if not identical:
            identical = {name: file_hashes[name] for name in IDENTICAL_FILES}
        elif any(file_hashes[name] != identical[name] for name in IDENTICAL_FILES):
            raise ValueError("PW-0318 repeated deterministic artifact mismatch")
        runs.append(
            {
                "run": index,
                "root": str(root),
                "build_report_sha256": file_hashes["build.json"],
                "metal_report_sha256": file_hashes["metal-run.json"],
                "build_complete_seconds": build["complete_seconds"],
                "build_peak_rss_bytes": build["peak_rss_bytes"],
                "complete_call_wall": metal["complete_call_wall"],
                "gpu_time": metal["gpu_time"],
                "safety_snapshots": metal["safety_snapshots"],
            }
        )
    first_build = json.loads((run_roots[0] / "build.json").read_text())
    result = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "status": "layer4_decode_transaction_qualified",
        "decision": "promote_partial_bank_decode_integration_boundary",
        "implementation_commit": implementation_commit,
        "analysis_commit": analysis_commit,
        "exactness_class": "L3_modified_weights",
        "metal_answer_key": "decode_one_row",
        "source_fidelity_comparator": "PW-0116_expert_major_batch",
        "identical_artifacts": identical,
        "semantic": first_build["semantic"],
        "runs": runs,
        "safety": safety_extrema(runs),
        "batch_size": 1,
        "concurrency": 1,
        "accepted_tokens": 0,
        "performance_claim": None,
        "claims_excluded": [
            "prefill/decode intermediate bit identity",
            "arbitrary routes",
            "complete bank",
            "complete endpoint",
            "accepted-token TPS",
            "Prismwing completion",
        ],
    }
    atomic_write_new(output, canonical_json(result))
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-001", required=True, type=Path)
    parser.add_argument("--run-002", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--implementation-commit", required=True)
    parser.add_argument("--analysis-commit", required=True)
    args = parser.parse_args()
    try:
        result = summarize(
            (args.run_001, args.run_002),
            args.output,
            args.implementation_commit,
            args.analysis_commit,
        )
        print(json.dumps({"status": result["status"], "output": str(args.output)}))
        return 0
    except (FileExistsError, KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        print(json.dumps({"error": str(error)}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
