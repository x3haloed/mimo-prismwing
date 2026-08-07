#!/usr/bin/env python3
"""Validate and summarize PW-0137's immutable global-Hessian GPTQ result."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from tools.analyze_group_local_gptq_three_expert_control import _safety_summary
    from tools.analyze_pw0116_corpus import sha256_file
    from tools.openrouter_reference import atomic_write_new, canonical_json
    from tools.run_global_hessian_gptq_rescue import (
        BLOCK_SIZE,
        DAMPING,
        EXPERT,
        LAYER,
        TRAIN_PLACEMENTS,
        VALIDATION_PLACEMENTS,
        _gate,
        physical_ledger,
    )
except ModuleNotFoundError:
    from analyze_group_local_gptq_three_expert_control import _safety_summary
    from analyze_pw0116_corpus import sha256_file
    from openrouter_reference import atomic_write_new, canonical_json
    from run_global_hessian_gptq_rescue import (
        BLOCK_SIZE,
        DAMPING,
        EXPERT,
        LAYER,
        TRAIN_PLACEMENTS,
        VALIDATION_PLACEMENTS,
        _gate,
        physical_ledger,
    )


SOURCE_SHA256 = "95fee340bb676ac7c9486ea713da9c461ca6fb62441b41b32ff988e97ed1502e"


def analyze(source_path: Path) -> dict:
    if sha256_file(source_path) != SOURCE_SHA256:
        raise ValueError("PW-0137 source report hash mismatch")
    source = json.loads(source_path.read_text())
    mechanism = source.get("mechanism", {})
    if (
        source.get("schema_version") != 1
        or source.get("evidence_class") != "pw0137_global_hessian_fixed_grid_gptq_rescue"
        or source.get("holdout_unsealed")
        or source.get("accepted_tokens") != 0
        or source.get("A") != 0
        or source.get("performance_claim") is not None
        or source.get("decision") != "authorize_three_expert_global_hessian_confirmation"
        or mechanism.get("damping") != DAMPING
        or mechanism.get("block_size") != BLOCK_SIZE
        or mechanism.get("order") != "descending_full_hessian_diagonal"
        or not mechanism.get("static_original_group_grid")
        or not mechanism.get("cross_block_error_propagation")
    ):
        raise ValueError("PW-0137 source authority mismatch")

    report = source["report"]
    if (
        report.get("layer") != LAYER
        or report.get("expert") != EXPERT
        or report.get("train_placements") != TRAIN_PLACEMENTS
        or report.get("validation_placements") != VALIDATION_PLACEMENTS
        or set(report.get("projection_reports", {})) != {"gate", "up", "down"}
    ):
        raise ValueError("PW-0137 expert authority mismatch")
    for projection in report["projection_reports"].values():
        if (
            projection["damping"] != DAMPING
            or projection["block_size"] != BLOCK_SIZE
            or projection["block_count"] <= 1
            or projection["cross_block_update_l2"] <= 0
            or projection["dead_activation_columns"] != 0
            or len(projection["activation_order_sha256"]) != 64
            or len(projection["grid_sha256"]) != 64
            or projection["candidate_train_metrics"]["relative_l2"]
            >= projection["baseline_train_metrics"]["relative_l2"]
            or projection["projected_workspace_bytes"] > 8 * 1024**3
        ):
            raise ValueError("PW-0137 projection evidence mismatch")

    recomputed = _gate(report)
    if source["continuation_gate"] != recomputed or recomputed["physical"] != physical_ledger():
        raise ValueError("PW-0137 gate mismatch")
    if not recomputed["passes"] or not all(recomputed["conditions"].values()):
        raise ValueError("PW-0137 authorization conflicts with gate")

    candidate = report["global_gptq_validation"]
    prior = report["pw0135_group_local_validation"]
    return {
        "schema_version": 1,
        "evidence_class": "pw0137_validated_global_hessian_gptq_rescue",
        "source_report_sha256": SOURCE_SHA256,
        "continuation_gate": recomputed,
        "result": {
            "layer": LAYER,
            "expert": EXPERT,
            "baseline_validation_relative_l2": report["dense_control_validation"]["relative_l2"],
            "group_local_validation_relative_l2": prior["relative_l2"],
            "global_hessian_validation_relative_l2": candidate["relative_l2"],
            "global_hessian_maximum_row_relative_l2": candidate["maximum_row_relative_l2"],
            "relative_improvement_over_group_local": 1.0
            - candidate["relative_l2"] / prior["relative_l2"],
        },
        "holdout_unsealed": False,
        "safety": _safety_summary(source["safety_snapshots"]),
        "evidence_valid": True,
        "experiment_passed": True,
        "decision": source["decision"],
        "next_branch": "three_expert_global_hessian_confirmation",
        "limitations": source["limitations"],
        "performance_claim": None,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    try:
        result = analyze(arguments.source)
        atomic_write_new(arguments.output, canonical_json(result))
        print(json.dumps({"output": str(arguments.output), "decision": result["decision"]}))
        return 0
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
        print(json.dumps({"error": str(error)}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
