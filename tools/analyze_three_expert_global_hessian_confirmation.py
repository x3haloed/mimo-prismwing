#!/usr/bin/env python3
"""Validate PW-0138's immutable three-expert confirmation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from tools.analyze_group_local_gptq_three_expert_control import _safety_summary
    from tools.analyze_pw0116_corpus import sha256_file
    from tools.openrouter_reference import atomic_write_new, canonical_json
    from tools.run_group_local_gptq_three_expert_control import SAMPLES, physical_ledger
    from tools.run_three_expert_global_hessian_confirmation import (
        BLOCK_SIZE,
        DAMPING,
        _gate,
    )
except ModuleNotFoundError:
    from analyze_group_local_gptq_three_expert_control import _safety_summary
    from analyze_pw0116_corpus import sha256_file
    from openrouter_reference import atomic_write_new, canonical_json
    from run_group_local_gptq_three_expert_control import SAMPLES, physical_ledger
    from run_three_expert_global_hessian_confirmation import BLOCK_SIZE, DAMPING, _gate


SOURCE_SHA256 = "37fa27ce90d0dc46b4b9308ed708c99405eb7ad3d924b859489716b9771bde49"


def analyze(source_path: Path) -> dict:
    if sha256_file(source_path) != SOURCE_SHA256:
        raise ValueError("PW-0138 source report hash mismatch")
    source = json.loads(source_path.read_text())
    mechanism = source.get("mechanism", {})
    if (
        source.get("schema_version") != 1
        or source.get("evidence_class") != "pw0138_three_expert_global_hessian_gptq_confirmation"
        or source.get("samples") != [list(row) for row in SAMPLES]
        or not source.get("pw0137_exact_reproduction")
        or source.get("holdout_unsealed")
        or source.get("accepted_tokens") != 0
        or source.get("A") != 0
        or source.get("performance_claim") is not None
        or source.get("decision") != "authorize_all_validation_expert_global_hessian_audit"
        or mechanism.get("damping") != DAMPING
        or mechanism.get("block_size") != BLOCK_SIZE
        or not mechanism.get("static_original_group_grid")
        or not mechanism.get("cross_block_error_propagation")
    ):
        raise ValueError("PW-0138 source authority mismatch")
    reports = source["reports"]
    if [(row["layer"], row["expert"]) for row in reports] != [
        (layer, expert) for layer, expert, _, _ in SAMPLES
    ]:
        raise ValueError("PW-0138 expert identity mismatch")
    for report, (_, _, train_count, validation_count) in zip(reports, SAMPLES):
        if (
            report["train_placements"] != train_count
            or report["validation_placements"] != validation_count
            or set(report["projection_reports"]) != {"gate", "up", "down"}
        ):
            raise ValueError("PW-0138 expert coverage mismatch")
        for projection in report["projection_reports"].values():
            if (
                projection["cross_block_update_l2"] <= 0
                or projection["candidate_train_metrics"]["relative_l2"]
                >= projection["baseline_train_metrics"]["relative_l2"]
            ):
                raise ValueError("PW-0138 projection evidence mismatch")
    recomputed = _gate(reports)
    if source["continuation_gate"] != recomputed or recomputed["physical"] != physical_ledger():
        raise ValueError("PW-0138 gate mismatch")
    if not recomputed["passes"] or not all(row["passes"] for row in recomputed["experts"]):
        raise ValueError("PW-0138 authorization conflicts with gate")
    return {
        "schema_version": 1,
        "evidence_class": "pw0138_validated_three_expert_global_hessian_confirmation",
        "source_report_sha256": SOURCE_SHA256,
        "continuation_gate": recomputed,
        "expert_results": [
            {
                "layer": report["layer"],
                "expert": report["expert"],
                "baseline_validation_relative_l2": report["dense_control_validation"]["relative_l2"],
                "group_local_validation_relative_l2": report["pw0135_group_local_validation"]["relative_l2"],
                "global_hessian_validation_relative_l2": report["global_gptq_validation"]["relative_l2"],
                "global_hessian_maximum_row_relative_l2": report["global_gptq_validation"]["maximum_row_relative_l2"],
            }
            for report in reports
        ],
        "holdout_unsealed": False,
        "safety": _safety_summary(source["safety_snapshots"]),
        "evidence_valid": True,
        "experiment_passed": True,
        "decision": source["decision"],
        "next_branch": "all_validation_expert_global_hessian_audit",
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
