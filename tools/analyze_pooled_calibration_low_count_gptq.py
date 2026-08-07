#!/usr/bin/env python3
"""Validate PW-0140's immutable pooled-calibration result."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from tools.analyze_group_local_gptq_three_expert_control import _safety_summary
    from tools.analyze_pw0116_corpus import sha256_file
    from tools.openrouter_reference import atomic_write_new, canonical_json
    from tools.run_pooled_calibration_low_count_gptq import SAMPLES, _gate
except ModuleNotFoundError:
    from analyze_group_local_gptq_three_expert_control import _safety_summary
    from analyze_pw0116_corpus import sha256_file
    from openrouter_reference import atomic_write_new, canonical_json
    from run_pooled_calibration_low_count_gptq import SAMPLES, _gate


SOURCE_SHA256 = "824d66549da7833d855f430a60b761f145a98757fa191bb41db6cf6e56f78b9f"


def analyze(source_path: Path) -> dict:
    if sha256_file(source_path) != SOURCE_SHA256:
        raise ValueError("PW-0140 source report hash mismatch")
    source = json.loads(source_path.read_text())
    if (
        source.get("schema_version") != 1
        or source.get("evidence_class") != "pw0140_pooled_calibration_low_count_gptq_falsification"
        or source.get("samples") != [list(row) for row in SAMPLES]
        or source.get("holdout_unsealed")
        or source.get("accepted_tokens") != 0
        or source.get("A") != 0
        or source.get("performance_claim") is not None
        or source.get("decision") != "reject_pooled_only_low_count_gptq"
    ):
        raise ValueError("PW-0140 source authority mismatch")
    reports = source["reports"]
    if [(row["layer"], row["expert"]) for row in reports] != [
        (layer, expert) for layer, expert, _, _ in SAMPLES
    ]:
        raise ValueError("PW-0140 expert identity mismatch")
    recomputed = _gate(reports)
    if source["continuation_gate"] != recomputed:
        raise ValueError("PW-0140 gate mismatch")
    if recomputed["passes"] or [row["passes"] for row in recomputed["experts"]] != [True, True, False]:
        raise ValueError("PW-0140 rejection conflicts with gate")
    return {
        "schema_version": 1,
        "evidence_class": "pw0140_validated_pooled_calibration_rejection",
        "source_report_sha256": SOURCE_SHA256,
        "continuation_gate": recomputed,
        "holdout_unsealed": False,
        "safety": _safety_summary(source["safety_snapshots"]),
        "evidence_valid": True,
        "experiment_passed": False,
        "decision": source["decision"],
        "next_branches": ["function_preserving_rotation", "recovery_training"],
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
