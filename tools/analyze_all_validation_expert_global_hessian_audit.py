#!/usr/bin/env python3
"""Validate and summarize PW-0139's immutable all-expert audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from tools.analyze_group_local_gptq_three_expert_control import _safety_summary
    from tools.analyze_pw0116_corpus import sha256_file
    from tools.openrouter_reference import atomic_write_new, canonical_json
    from tools.run_all_validation_expert_global_hessian_audit import (
        LAYERS,
        _validation_gate,
    )
except ModuleNotFoundError:
    from analyze_group_local_gptq_three_expert_control import _safety_summary
    from analyze_pw0116_corpus import sha256_file
    from openrouter_reference import atomic_write_new, canonical_json
    from run_all_validation_expert_global_hessian_audit import LAYERS, _validation_gate


SOURCE_SHA256 = "83bd204c9d5c35a684cab15a4ddacf48cf9b661563fb26223eb3655d0ef4a7b5"
EXPECTED_EXPERTS = {4: 10, 24: 16, 46: 15}
EXPECTED_FALLBACKS = {4: [], 24: [25, 251], 46: []}


def analyze(source_path: Path) -> dict:
    if sha256_file(source_path) != SOURCE_SHA256:
        raise ValueError("PW-0139 source report hash mismatch")
    source = json.loads(source_path.read_text())
    if (
        source.get("schema_version") != 1
        or source.get("evidence_class") != "pw0139_all_validation_expert_global_hessian_audit"
        or source.get("layers") != list(LAYERS)
        or source.get("validation_range") != [112, 168]
        or source.get("holdout_unsealed")
        or source.get("accepted_tokens") != 0
        or source.get("A") != 0
        or source.get("performance_claim") is not None
        or source.get("decision") != "reject_global_hessian_fixed_grid_on_full_validation"
    ):
        raise ValueError("PW-0139 source authority mismatch")
    reports = source["layer_reports"]
    for report in reports:
        layer = report["layer"]
        if (
            report["unique_validation_experts"] != EXPECTED_EXPERTS[layer]
            or report["validation_placements"] != 448
            or report["train_absent_fallback_experts"] != EXPECTED_FALLBACKS[layer]
            or not report["pw0138_control_reproduced"]
            or sum(row["validation_placements"] for row in report["expert_reports"]) != 448
        ):
            raise ValueError("PW-0139 coverage authority mismatch")
        for expert in report["expert_reports"]:
            if set(expert["projection_reports"]) != {"gate", "up", "down"}:
                raise ValueError("PW-0139 projection identity mismatch")
            for projection in expert["projection_reports"].values():
                if (
                    projection["cross_block_update_l2"] <= 0
                    or projection["candidate_calibration_metrics"]["relative_l2"]
                    >= projection["baseline_calibration_metrics"]["relative_l2"]
                ):
                    raise ValueError("PW-0139 projection evidence mismatch")
    recomputed = _validation_gate(reports)
    if source["validation_gate"] != recomputed:
        raise ValueError("PW-0139 gate mismatch")
    if recomputed["passes"]:
        raise ValueError("PW-0139 rejection conflicts with gate")
    return {
        "schema_version": 1,
        "evidence_class": "pw0139_validated_all_expert_global_hessian_rejection",
        "source_report_sha256": SOURCE_SHA256,
        "validation_gate": recomputed,
        "layer_results": [
            {
                "layer": row["layer"],
                "unique_validation_experts": row["unique_validation_experts"],
                "train_absent_fallback_experts": row["train_absent_fallback_experts"],
                "routed_output_metrics": row["routed_output_metrics"],
            }
            for row in reports
        ],
        "holdout_unsealed": False,
        "safety": _safety_summary(source["safety_snapshots"]),
        "evidence_valid": True,
        "experiment_passed": False,
        "decision": source["decision"],
        "next_branches": [
            "pooled_hessian_shrinkage",
            "function_preserving_rotation",
            "recovery_training",
        ],
        "limitations": source["limitations"],
        "performance_claim": None,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        result = analyze(args.source)
        atomic_write_new(args.output, canonical_json(result))
        print(json.dumps({"output": str(args.output), "decision": result["decision"]}))
        return 0
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
        print(json.dumps({"error": str(error)}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
