#!/usr/bin/env python3
"""Run PW-0146's single threshold-crossing train-only QAT schedule."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

try:
    from tools.run_best_rank_real_expert_control import sha256_file
    from tools.run_train_only_code_qat_optimizer_preflight import run
except (ImportError, ModuleNotFoundError):
    from run_best_rank_real_expert_control import sha256_file
    from run_train_only_code_qat_optimizer_preflight import run


PW0145_SHA256 = "1d5f4f4bf9dacc39114d483f90e3e61590f847aa24a31a1c6d48dbb077deafa4"
LEARNING_RATE = 0.02


def run_threshold(
    checkpoint_root: Path,
    verification_path: Path,
    corpus_manifest_path: Path,
    pw0139_path: Path,
    pw0144_path: Path,
    pw0145_path: Path,
    output_path: Path,
    commit: str,
) -> dict:
    if sha256_file(pw0145_path) != PW0145_SHA256:
        raise ValueError("PW-0146 PW-0145 report hash mismatch")
    prior = json.loads(pw0145_path.read_text())
    if (
        prior.get("decision") != "reject_tested_fixed_grid_code_qat_optimizer_family"
        or prior.get("validation_loaded") is not False
        or prior.get("holdout_unsealed")
    ):
        raise ValueError("PW-0146 prior authority mismatch")
    return run(
        checkpoint_root,
        verification_path,
        corpus_manifest_path,
        pw0139_path,
        pw0144_path,
        output_path,
        commit,
        learning_rates=(LEARNING_RATE,),
        evidence_class="pw0146_train_only_threshold_crossing_code_qat",
        pass_decision="authorize_threshold_code_qat_validation_confirmation",
        reject_decision="reject_threshold_crossing_fixed_grid_code_qat",
        extra_source_hashes={"pw0145_report_sha256": PW0145_SHA256},
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--verification", required=True, type=Path)
    parser.add_argument("--corpus-manifest", required=True, type=Path)
    parser.add_argument("--pw0139", required=True, type=Path)
    parser.add_argument("--pw0144", required=True, type=Path)
    parser.add_argument("--pw0145", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--commit", required=True)
    args = parser.parse_args()
    try:
        result = run_threshold(
            args.checkpoint,
            args.verification,
            args.corpus_manifest,
            args.pw0139,
            args.pw0144,
            args.pw0145,
            args.output,
            args.commit,
        )
        print(json.dumps({"output": str(args.output), "decision": result["decision"]}))
        return 0
    except (OSError, RuntimeError, ValueError, KeyError, TypeError, json.JSONDecodeError, np.linalg.LinAlgError) as error:
        print(json.dumps({"error": str(error)}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
