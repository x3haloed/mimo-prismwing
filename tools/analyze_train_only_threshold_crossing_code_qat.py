#!/usr/bin/env python3
"""Validate PW-0146's immutable threshold-crossing QAT rejection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from tools.analyze_group_local_gptq_three_expert_control import _safety_summary
    from tools.analyze_pw0116_corpus import sha256_file
    from tools.openrouter_reference import atomic_write_new, canonical_json
    from tools.run_train_only_code_qat_optimizer_preflight import _gate, select_trial
    from tools.run_train_only_threshold_crossing_code_qat import LEARNING_RATE, PW0145_SHA256
except ModuleNotFoundError:
    from analyze_group_local_gptq_three_expert_control import _safety_summary
    from analyze_pw0116_corpus import sha256_file
    from openrouter_reference import atomic_write_new, canonical_json
    from run_train_only_code_qat_optimizer_preflight import _gate, select_trial
    from run_train_only_threshold_crossing_code_qat import LEARNING_RATE, PW0145_SHA256


SOURCE_SHA256 = "7bb795455927295c673bfe65d06ae6311dbdd97b9d3517caa357307d189bdcf3"


def analyze(source_path: Path) -> dict:
    if sha256_file(source_path) != SOURCE_SHA256:
        raise ValueError("PW-0146 source report hash mismatch")
    source = json.loads(source_path.read_text())
    if (
        source.get("schema_version") != 1
        or source.get("evidence_class") != "pw0146_train_only_threshold_crossing_code_qat"
        or source.get("learning_rates") != [LEARNING_RATE]
        or source.get("steps") != 32
        or source.get("validation_loaded") is not False
        or source.get("holdout_unsealed")
        or source.get("source_hashes", {}).get("pw0145_report_sha256") != PW0145_SHA256
        or source.get("accepted_tokens") != 0
        or source.get("A") != 0
        or source.get("performance_claim") is not None
        or source.get("decision") != "reject_threshold_crossing_fixed_grid_code_qat"
    ):
        raise ValueError("PW-0146 source authority mismatch")
    trials = source.get("trials", [])
    selected = select_trial(trials, (LEARNING_RATE,))
    changed = sum(selected["changed_codes"].values())
    total = sum(selected["code_totals"].values())
    if (
        not 0 < changed <= int(total * 0.05)
        or selected["training"]["loss_decreased"]
        or selected["train_metrics"]["relative_l2"] <= source["initial_train_metrics"]["relative_l2"]
        or not selected.get("code_domain_valid")
        or not selected.get("grid_metadata_unchanged")
    ):
        raise ValueError("PW-0146 threshold-crossing disposition mismatch")
    recomputed = _gate(source["initial_train_metrics"], selected)
    if source.get("continuation_gate") != recomputed or recomputed["passes"]:
        raise ValueError("PW-0146 gate mismatch")
    return {
        "schema_version": 1,
        "evidence_class": "pw0146_validated_threshold_crossing_qat_rejection",
        "source_report_sha256": SOURCE_SHA256,
        "continuation_gate": recomputed,
        "validation_loaded": False,
        "holdout_unsealed": False,
        "safety": _safety_summary(source["safety_snapshots"]),
        "evidence_valid": True,
        "experiment_passed": False,
        "decision": source["decision"],
        "next_branch": "grid_changing_training_or_different_executable_representation",
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

