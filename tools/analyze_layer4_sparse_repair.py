#!/usr/bin/env python3
"""Prove the causal fan-out of PW-0101's expert-245 gate repair failure."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

try:
    from tools.generate_full_prefix_oracle import checked_fp8
    from tools.generate_real_layer0_bf16_oracle import Safety
    from tools.generate_real_layer1_expert_oracle import ShardedCheckpoint
    from tools.openrouter_reference import atomic_write_new, canonical_json
except ModuleNotFoundError:
    from generate_full_prefix_oracle import checked_fp8
    from generate_real_layer0_bf16_oracle import Safety
    from generate_real_layer1_expert_oracle import ShardedCheckpoint
    from openrouter_reference import atomic_write_new, canonical_json


ORACLE_SHA256 = "9c96d85e45832abdccd3be2325db993749579a904469d1862c8f3437cafab86d"
CANDIDATE_SHA256 = "b2021bb4d37383a62693565da7f39a0e313a721419a49fb3b64881bfd91893bf"
EXPERT = 245
GATE_INDEX = 1798
METAL_GATE_BITS = 0x40800000
ORACLE_GATE_BITS = 0x40810000


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def capture(oracle_root: Path, record: dict) -> torch.Tensor:
    path = oracle_root / record["file"]
    if sha256(path) != record["sha256"]:
        raise ValueError(f"capture hash mismatch: {path.name}")
    raw = np.fromfile(path, dtype="<f4")
    if raw.size != int(np.prod(record["shape"])) or not np.isfinite(raw).all():
        raise ValueError(f"capture shape/value mismatch: {path.name}")
    return torch.from_numpy(raw.copy()).reshape(1, -1).to(torch.bfloat16)


def parity(actual: torch.Tensor, expected: torch.Tensor) -> dict:
    a = actual.float().reshape(-1)
    e = expected.float().reshape(-1)
    difference = a - e
    return {
        "equal_values": int(torch.eq(a, e).sum()),
        "total_values": a.numel(),
        "equality_fraction": float(torch.eq(a, e).float().mean()),
        "maximum_absolute_error": float(difference.abs().max()),
        "relative_l2": float(torch.linalg.vector_norm(difference) / torch.linalg.vector_norm(e)),
        "first_mismatch_indices": torch.nonzero(a != e).reshape(-1)[:16].tolist(),
    }


def analyze(checkpoint_root: Path, verification: Path, oracle_manifest: Path,
            candidate_report: Path, output: Path) -> None:
    if sha256(oracle_manifest) != ORACLE_SHA256 or sha256(candidate_report) != CANDIDATE_SHA256:
        raise ValueError("PW-0101 evidence hash mismatch")
    oracle = json.loads(oracle_manifest.read_text())
    candidate = json.loads(candidate_report.read_text())
    diagnostic = next(
        item for item in candidate["expert_diagnostics"] if item["expert"] == EXPERT
    )
    gate_diagnostic = next(item for item in diagnostic["stages"] if item["stage"] == "gate")
    if gate_diagnostic["sparse_repairs"] != 1 or not gate_diagnostic["first_mismatches"]:
        raise ValueError("expert-245 gate repair evidence mismatch")
    expected_fragment = (
        f"index={GATE_INDEX},actual=0x{METAL_GATE_BITS:08x},"
        f"expected=0x{ORACLE_GATE_BITS:08x},pre=0x40808000,midpoint_distance=0"
    )
    if gate_diagnostic["first_mismatches"][0] != expected_fragment:
        raise ValueError("expert-245 decisive gate mismatch changed")
    records = oracle["layer4_expert_captures"][str(EXPERT)]
    gate_oracle = capture(oracle_manifest.parent, records["gate"])
    up_oracle = capture(oracle_manifest.parent, records["up"])
    swiglu_oracle = capture(oracle_manifest.parent, records["swiglu"])
    down_oracle = capture(oracle_manifest.parent, records["down"])
    if gate_oracle.float().reshape(-1)[GATE_INDEX].item() != np.array(
        [ORACLE_GATE_BITS], dtype="<u4"
    ).view("<f4")[0]:
        raise ValueError("oracle gate value mismatch")
    safety = Safety()
    checkpoint = ShardedCheckpoint(checkpoint_root, verification)
    safety.check("checkpoint_open")
    gate_wrong = gate_oracle.clone()
    gate_wrong.reshape(-1)[GATE_INDEX] = torch.tensor(
        np.array([METAL_GATE_BITS], dtype="<u4").view("<f4")[0], dtype=torch.bfloat16
    )
    swiglu_wrong = (torch.nn.functional.silu(gate_wrong) * up_oracle).to(torch.bfloat16)
    prefix = f"model.layers.4.mlp.experts.{EXPERT}.down_proj.weight"
    down_wrong = checked_fp8(checkpoint, prefix, swiglu_wrong)
    down_restored = checked_fp8(checkpoint, prefix, swiglu_oracle)
    safety.check("counterfactual_complete")
    result = {
        "schema_version": 1,
        "semantic": "pw0101_expert245_gate_counterfactual",
        "oracle_manifest_sha256": ORACLE_SHA256,
        "candidate_report_sha256": CANDIDATE_SHA256,
        "expert": EXPERT,
        "gate_index": GATE_INDEX,
        "metal_gate_bits": f"0x{METAL_GATE_BITS:08x}",
        "oracle_gate_bits": f"0x{ORACLE_GATE_BITS:08x}",
        "pre_round_bits": "0x40808000",
        "midpoint_distance": 0,
        "selected_sparse_repairs": gate_diagnostic["sparse_repairs"],
        "wrong_gate_vs_oracle": parity(gate_wrong, gate_oracle),
        "wrong_swiglu_vs_oracle": parity(swiglu_wrong, swiglu_oracle),
        "wrong_down_vs_oracle": parity(down_wrong, down_oracle),
        "restored_down_vs_oracle": parity(down_restored, down_oracle),
        "causal_statement": "replacing only gate[1798] with the wrong BF16 neighbor causes the downstream fan-out; restoring the oracle neighbor removes it",
        "safety_snapshots": safety.snapshots,
        "performance_claim": None,
    }
    atomic_write_new(output, canonical_json(result))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--verification", required=True, type=Path)
    parser.add_argument("--oracle-manifest", required=True, type=Path)
    parser.add_argument("--candidate-report", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    analyze(
        args.checkpoint,
        args.verification,
        args.oracle_manifest,
        args.candidate_report,
        args.output,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
