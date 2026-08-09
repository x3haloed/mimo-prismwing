#!/usr/bin/env python3
"""Validate PW-0149's immutable nonuniform INT4 result."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from tools.analyze_group_local_gptq_three_expert_control import _safety_summary
    from tools.analyze_pw0116_corpus import sha256_file
    from tools.openrouter_reference import atomic_write_new, canonical_json
    from tools.run_five_bit_global_hessian_three_expert_control import SAMPLES
    from tools.run_nonuniform_int4_global_hessian_control import PACKED_BYTES, _gate
except ModuleNotFoundError:
    from analyze_group_local_gptq_three_expert_control import _safety_summary
    from analyze_pw0116_corpus import sha256_file
    from openrouter_reference import atomic_write_new, canonical_json
    from run_five_bit_global_hessian_three_expert_control import SAMPLES
    from run_nonuniform_int4_global_hessian_control import PACKED_BYTES, _gate


SOURCE_SHA256 = "f8860f648cc6596d5c6a35eca7b2236270676aa421d0890c1d2b02236dffd54a"


def analyze(source_path: Path) -> dict:
    if sha256_file(source_path) != SOURCE_SHA256:
        raise ValueError("PW-0149 source report hash mismatch")
    source = json.loads(source_path.read_text())
    if (
        source.get("schema_version") != 1
        or source.get("evidence_class") != "pw0149_nonuniform_int4_global_hessian_control"
        or source.get("samples") != [list(row) for row in SAMPLES]
        or source.get("holdout_unsealed")
        or source.get("accepted_tokens") != 0
        or source.get("A") != 0
        or source.get("performance_claim") is not None
        or source.get("decision")
        != "reject_nonuniform_int4_global_hessian_three_expert_control"
    ):
        raise ValueError("PW-0149 source authority mismatch")
    reports = source.get("reports", [])
    if [(row.get("layer"), row.get("expert")) for row in reports] != [
        (layer, expert) for layer, expert, _, _ in SAMPLES
    ]:
        raise ValueError("PW-0149 expert identity mismatch")
    if not all(
        row.get("four_bit_control_reproduced") and row.get("code_domain_valid")
        for row in reports
    ):
        raise ValueError("PW-0149 control authority mismatch")
    recomputed = _gate(reports)
    if source.get("continuation_gate") != recomputed:
        raise ValueError("PW-0149 gate mismatch")
    if (
        recomputed["passes"]
        or [row["passes"] for row in recomputed["experts"]] != [False, False, False]
        or recomputed["physical"]["packed_bytes_per_expert"] != PACKED_BYTES
    ):
        raise ValueError("PW-0149 rejection conflicts with gate")
    return {
        "schema_version": 1,
        "evidence_class": "pw0149_validated_nonuniform_int4_rejection",
        "source_report_sha256": SOURCE_SHA256,
        "continuation_gate": recomputed,
        "holdout_unsealed": False,
        "safety": _safety_summary(source["safety_snapshots"]),
        "evidence_valid": True,
        "experiment_passed": False,
        "decision": source["decision"],
        "next_branch": "vector_or_program_quantization_or_companion_envelope",
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
