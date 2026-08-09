#!/usr/bin/env python3
"""Validate PW-0142's immutable fixed-code recovery-training result."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from tools.analyze_group_local_gptq_three_expert_control import _safety_summary
    from tools.analyze_pw0116_corpus import sha256_file
    from tools.openrouter_reference import atomic_write_new, canonical_json
    from tools.run_train_only_end_to_end_int4_grid_recovery import (
        LEARNING_RATE,
        REGULARIZATION,
        SAFETY_INTERVAL,
        SAMPLES,
        STEPS,
        _gate,
    )
except ModuleNotFoundError:
    from analyze_group_local_gptq_three_expert_control import _safety_summary
    from analyze_pw0116_corpus import sha256_file
    from openrouter_reference import atomic_write_new, canonical_json
    from run_train_only_end_to_end_int4_grid_recovery import (
        LEARNING_RATE,
        REGULARIZATION,
        SAFETY_INTERVAL,
        SAMPLES,
        STEPS,
        _gate,
    )


SOURCE_SHA256 = "0c2095a2068ccf347ab86beccb41e8d303444ce371b52d8475a37b26c29e9cc7"


def analyze(source_path: Path) -> dict:
    if sha256_file(source_path) != SOURCE_SHA256:
        raise ValueError("PW-0142 source report hash mismatch")
    source = json.loads(source_path.read_text())
    expected_steps = [0, *range(SAFETY_INTERVAL, STEPS + 1, SAFETY_INTERVAL)]
    if (
        source.get("schema_version") != 1
        or source.get("evidence_class") != "pw0142_train_only_end_to_end_int4_grid_recovery"
        or source.get("samples") != [list(row) for row in SAMPLES]
        or source.get("holdout_unsealed")
        or source.get("accepted_tokens") != 0
        or source.get("A") != 0
        or source.get("performance_claim") is not None
        or source.get("decision") != "reject_fixed_code_group_parameter_recovery"
    ):
        raise ValueError("PW-0142 source authority mismatch")
    contract = source.get("training_contract", {})
    if (
        contract.get("steps") != STEPS
        or contract.get("learning_rate") != LEARNING_RATE
        or contract.get("regularization") != REGULARIZATION
        or contract.get("betas") != [0.9, 0.999]
        or contract.get("epsilon") != 1e-8
        or contract.get("bias_correction") is not False
        or contract.get("full_batch") is not True
        or contract.get("fixed_codes") is not True
        or contract.get("final_grid_dtype") != "float16"
    ):
        raise ValueError("PW-0142 training contract mismatch")
    reports = source.get("reports", [])
    if [(row.get("layer"), row.get("expert")) for row in reports] != [
        (layer, expert) for layer, expert, _, _ in SAMPLES
    ]:
        raise ValueError("PW-0142 expert identity mismatch")
    for report in reports:
        history = report.get("training", {}).get("loss_history", [])
        if [row.get("step") for row in history] != expected_steps:
            raise ValueError("PW-0142 loss-history schedule mismatch")
        if not all(isinstance(row.get("loss"), (int, float)) for row in history):
            raise ValueError("PW-0142 loss-history value mismatch")
        if not report.get("codes_unchanged"):
            raise ValueError("PW-0142 fixed-code authority mismatch")
        if report.get("initial_validation_metrics") != report.get("pw0139_validation_metrics"):
            raise ValueError("PW-0142 does not preserve PW-0139 validation authority")
        if len(report.get("parameter_sha256", "")) != 64:
            raise ValueError("PW-0142 parameter hash mismatch")
    recomputed = _gate(reports)
    if source.get("continuation_gate") != recomputed:
        raise ValueError("PW-0142 continuation gate mismatch")
    if recomputed["passes"] or any(row["passes"] for row in recomputed["experts"]):
        raise ValueError("PW-0142 rejection conflicts with gate")
    return {
        "schema_version": 1,
        "evidence_class": "pw0142_validated_fixed_code_recovery_rejection",
        "source_report_sha256": SOURCE_SHA256,
        "continuation_gate": recomputed,
        "training_loss_decreased_by_expert": [
            {
                "layer": row["layer"],
                "expert": row["expert"],
                "loss_decreased": row["training"]["loss_decreased"],
            }
            for row in reports
        ],
        "holdout_unsealed": False,
        "safety": _safety_summary(source["safety_snapshots"]),
        "evidence_valid": True,
        "experiment_passed": False,
        "decision": source["decision"],
        "next_branch": "code_changing_qat_or_different_executable_representation",
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

