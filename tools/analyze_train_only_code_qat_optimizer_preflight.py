#!/usr/bin/env python3
"""Validate PW-0145's immutable train-only QAT optimizer preflight."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from tools.analyze_group_local_gptq_three_expert_control import _safety_summary
    from tools.analyze_pw0116_corpus import sha256_file
    from tools.openrouter_reference import atomic_write_new, canonical_json
    from tools.run_train_only_code_qat_optimizer_preflight import (
        EXPERT,
        LAYER,
        LEARNING_RATES,
        STEPS,
        TRAIN_PLACEMENTS,
        _gate,
        select_trial,
    )
except ModuleNotFoundError:
    from analyze_group_local_gptq_three_expert_control import _safety_summary
    from analyze_pw0116_corpus import sha256_file
    from openrouter_reference import atomic_write_new, canonical_json
    from run_train_only_code_qat_optimizer_preflight import (
        EXPERT,
        LAYER,
        LEARNING_RATES,
        STEPS,
        TRAIN_PLACEMENTS,
        _gate,
        select_trial,
    )


SOURCE_SHA256 = "1d5f4f4bf9dacc39114d483f90e3e61590f847aa24a31a1c6d48dbb077deafa4"


def analyze(source_path: Path) -> dict:
    if sha256_file(source_path) != SOURCE_SHA256:
        raise ValueError("PW-0145 source report hash mismatch")
    source = json.loads(source_path.read_text())
    if (
        source.get("schema_version") != 1
        or source.get("evidence_class") != "pw0145_train_only_code_qat_optimizer_preflight"
        or source.get("layer") != LAYER
        or source.get("expert") != EXPERT
        or source.get("train_placements") != TRAIN_PLACEMENTS
        or source.get("learning_rates") != list(LEARNING_RATES)
        or source.get("steps") != STEPS
        or source.get("validation_loaded") is not False
        or source.get("holdout_unsealed")
        or source.get("accepted_tokens") != 0
        or source.get("A") != 0
        or source.get("performance_claim") is not None
        or source.get("decision") != "reject_tested_fixed_grid_code_qat_optimizer_family"
    ):
        raise ValueError("PW-0145 source authority mismatch")
    trials = source.get("trials", [])
    selected = select_trial(trials)
    if source.get("selected_learning_rate") != selected["learning_rate"]:
        raise ValueError("PW-0145 selection mismatch")
    for trial in trials:
        if sum(trial["changed_codes"].values()) != 0:
            raise ValueError("PW-0145 observed code change conflicts with sealed result")
        if not trial.get("code_domain_valid") or not trial.get("grid_metadata_unchanged"):
            raise ValueError("PW-0145 artifact authority mismatch")
        if trial["training"]["loss_decreased"]:
            raise ValueError("PW-0145 loss disposition mismatch")
    recomputed = _gate(source["initial_train_metrics"], selected)
    if source.get("continuation_gate") != recomputed or recomputed["passes"]:
        raise ValueError("PW-0145 gate mismatch")
    return {
        "schema_version": 1,
        "evidence_class": "pw0145_validated_train_only_optimizer_rejection",
        "source_report_sha256": SOURCE_SHA256,
        "continuation_gate": recomputed,
        "validation_loaded": False,
        "holdout_unsealed": False,
        "safety": _safety_summary(source["safety_snapshots"]),
        "evidence_valid": True,
        "experiment_passed": False,
        "decision": source["decision"],
        "next_branch": "threshold_crossing_train_only_schedule_or_different_representation",
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

