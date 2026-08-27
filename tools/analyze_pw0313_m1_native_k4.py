#!/usr/bin/env python3
"""Bind and decide the repeated PW-0313 target-native K4 constructions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    from tools.openrouter_reference import atomic_write_new, canonical_json
    from tools.reproduce_pw0311_k4_expert import sha256_file
except ModuleNotFoundError:
    from openrouter_reference import atomic_write_new, canonical_json
    from reproduce_pw0311_k4_expert import sha256_file


EXPERIMENT_ID = "PW-0313"
REVISION = "m1-native-k4-v1"
TARGETS = (199, 41)


def _safe(report: dict[str, Any]) -> bool:
    snapshots = report.get("safety_snapshots", [])
    releases = [row for row in snapshots if row.get("release_boundary")]
    return bool(snapshots) and bool(releases) and all(
        int(row.get("swap_growth_bytes", 1)) == 0
        and int(row.get("new_throttled_pages", 1)) == 0
        and int(row["system_memory_free_percent"]) >= 10
        and int(row["process_physical_footprint_bytes"]) <= 13 * 1024**3
        for row in snapshots
    ) and all(
        int(row["process_physical_footprint_bytes"]) <= 12 * 1024**3
        for row in releases
    )


def _load_pair(paths: list[Path], expert: int) -> dict[str, Any]:
    if len(paths) != 2:
        raise ValueError(f"expert {expert} requires exactly two runs")
    reports = []
    records = []
    for path in paths:
        report = json.loads(path.read_text())
        if (
            report.get("schema_version") != 1
            or report.get("experiment_id") != EXPERIMENT_ID
            or report.get("revision") != REVISION
            or int(report.get("expert", -1)) != expert
            or report.get("failure") is not None
        ):
            raise ValueError(f"invalid PW-0313 report: {path}")
        reports.append(report)
        records.append(
            {
                "path": str(path.resolve()),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "status": report["status"],
                "complete_seconds": report["complete_seconds"],
                "peak_rss_bytes": report["peak_rss_bytes"],
            }
        )
    if reports[0]["commit"] != reports[1]["commit"]:
        raise ValueError(f"expert {expert} repeat commit mismatch")
    tree_equal = reports[0]["deterministic_tree"] == reports[1]["deterministic_tree"]
    semantic_equal = reports[0]["semantic"] == reports[1]["semantic"]
    return {
        "expert": expert,
        "runs": records,
        "commit": reports[0]["commit"],
        "tree_equal": tree_equal,
        "semantic_equal": semantic_equal,
        "tree_files": len(reports[0]["deterministic_tree"]["files"]),
        "tree_bytes": reports[0]["deterministic_tree"]["total_bytes"],
        "classifications": {
            name: row["classification"] for name, row in reports[0]["projections"].items()
        },
        "semantic": reports[0]["semantic"],
        "safety_pass": all(_safe(report) for report in reports),
    }


def analyze(expert_199: list[Path], expert_41: list[Path]) -> dict[str, Any]:
    policy = _load_pair(expert_199, 199)
    control = _load_pair(expert_41, 41)
    if policy["commit"] != control["commit"]:
        raise ValueError("PW-0313 target commit mismatch")
    policy_pass = (
        policy["tree_equal"]
        and policy["semantic_equal"]
        and policy["safety_pass"]
        and policy["semantic"]["gates"]["pass"]
    )
    control_pass = (
        control["tree_equal"]
        and control["semantic_equal"]
        and control["safety_pass"]
        and control["semantic"]["gates"]["pass"]
    )
    if policy_pass and not control_pass:
        status = "policy_expert_qualified_control_expert_rejected"
        decision = "authorize_policy_relevant_m1_native_k4_but_prohibit_expert_41_expansion"
    elif policy_pass and control_pass:
        status = "both_target_native_experts_qualified"
        decision = "authorize_new_layer_m1_native_k4_construction"
    else:
        status = "policy_relevant_target_native_expert_unqualified"
        decision = "keep_authenticated_m4_artifacts_only"
    return {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "revision": REVISION,
        "status": status,
        "decision": decision,
        "commit": policy["commit"],
        "experts": {"199": policy, "41": control},
        "gates": {
            "policy_expert_pass": policy_pass,
            "control_expert_pass": control_pass,
            "all_repeats_byte_identical": policy["tree_equal"] and control["tree_equal"],
            "all_repeats_semantically_identical": policy["semantic_equal"] and control["semantic_equal"],
            "all_safety_pass": policy["safety_pass"] and control["safety_pass"],
        },
        "accepted_tokens": 0,
        "A": 0,
        "U": 0,
        "performance_claim": None,
        "claims_excluded": [
            "arbitrary expert construction",
            "expert 41 promotion",
            "other layers or complete bank",
            "hosted or multimodal equivalence",
            "ordinary endpoint execution",
            "accepted-token TPS",
            "Prismwing-2 or Prismwing 50 completion",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expert-199", action="append", required=True, type=Path)
    parser.add_argument("--expert-41", action="append", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    try:
        result = analyze(arguments.expert_199, arguments.expert_41)
        atomic_write_new(arguments.output, canonical_json(result))
        print(json.dumps({"output": str(arguments.output), "status": result["status"]}))
        return 0
    except (FileExistsError, KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        print(json.dumps({"error": str(error)}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
