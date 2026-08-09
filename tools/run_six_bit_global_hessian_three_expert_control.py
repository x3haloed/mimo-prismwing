#!/usr/bin/env python3
"""Run PW-0148's six-bit global-Hessian three-expert control."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

try:
    from tools.run_five_bit_global_hessian_three_expert_control import (
        NBitControlConfig,
        _gate as _nbit_gate,
        affine_nbit_grid,
        physical_ledger as _physical_ledger,
        reconstruct_nbit_grid,
        run as _run_nbit,
    )
except ModuleNotFoundError:
    from run_five_bit_global_hessian_three_expert_control import (
        NBitControlConfig,
        _gate as _nbit_gate,
        affine_nbit_grid,
        physical_ledger as _physical_ledger,
        reconstruct_nbit_grid,
        run as _run_nbit,
    )


BITS = 6
MAXIMUM_CODE = 63
PACKED_CODE_BYTES = 18_874_368
METADATA_BYTES = 786_432
PACKED_BYTES = 19_660_800
PACKED_RATIO = 0.7810593116914816
PW0147_SHA256 = "a7706fce33dc716930d080988e197089bcf1ebb6fb5729adcdb3203a8cccd62e"

SIX_BIT_CONFIG = NBitControlConfig(
    experiment="PW-0148",
    bits=BITS,
    packed_code_bytes=PACKED_CODE_BYTES,
    maximum_packed_ratio=0.80,
    candidate_label="six_bit",
    evidence_class="pw0148_six_bit_global_hessian_three_expert_control",
    pass_decision="authorize_all_validation_expert_six_bit_audit",
    reject_decision="reject_six_bit_global_hessian_three_expert_control",
    prerequisite_sha256=PW0147_SHA256,
    prerequisite_label="PW-0147 report",
    prerequisite_decision="reject_five_bit_global_hessian_three_expert_control",
    prerequisite_source_key="pw0147_report_sha256",
    prior_candidate_label="five_bit",
)


def affine_six_bit_grid(weight: np.ndarray):
    return affine_nbit_grid(weight, BITS)


def reconstruct_six_bit_grid(codes: np.ndarray, scales: np.ndarray, biases: np.ndarray):
    return reconstruct_nbit_grid(codes, scales, biases, BITS)


def physical_ledger() -> dict:
    return _physical_ledger(SIX_BIT_CONFIG)


def _gate(reports: list[dict]) -> dict:
    return _nbit_gate(reports, SIX_BIT_CONFIG)


def run(
    checkpoint_root: Path,
    verification_path: Path,
    corpus_manifest_path: Path,
    pw0138_path: Path,
    pw0147_path: Path,
    output_path: Path,
    commit: str,
) -> dict:
    return _run_nbit(
        checkpoint_root,
        verification_path,
        corpus_manifest_path,
        pw0138_path,
        pw0147_path,
        output_path,
        commit,
        SIX_BIT_CONFIG,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--verification", required=True, type=Path)
    parser.add_argument("--corpus-manifest", required=True, type=Path)
    parser.add_argument("--pw0138", required=True, type=Path)
    parser.add_argument("--pw0147", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--commit", required=True)
    args = parser.parse_args()
    try:
        result = run(
            args.checkpoint,
            args.verification,
            args.corpus_manifest,
            args.pw0138,
            args.pw0147,
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
