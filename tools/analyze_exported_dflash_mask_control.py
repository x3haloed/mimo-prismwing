#!/usr/bin/env python3
"""Authenticate and analyze PW-0150's exported-mask DFlash proposal."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import time

import numpy as np

try:
    from tools.analyze_pw0116_corpus import sha256_file
    from tools.host_safety import HostSafetyMonitor
    from tools.openrouter_reference import atomic_write_new, canonical_json
except ModuleNotFoundError:
    from analyze_pw0116_corpus import sha256_file
    from host_safety import HostSafetyMonitor
    from openrouter_reference import atomic_write_new, canonical_json


REVISION = "63651580ca774f8504f676040460aed3e1244ac1"
RAW_SHA256 = "0582f905d8d6531e0c7d4e9a50def819a6d337a62c5e9b0cac351caa9435f882"
TARGET_SHA256 = "cb30738d5a79d7d85587a68b53f876a59101d5ca09bbc7c895daaf501954f4d3"
EXPECTED_PROPOSAL = [264, 11, 11, 11, 11, 11, 11, 11]
EXPECTED_OLD_PROPOSAL = [264, 1773, 102092, 102092, 102092, 1773, 1773, 1773]
EXPECTED_TARGET_POSTERIOR = [13, 15, 18, 481, 15, 481, 15, 15]
EXPECTED_LOGITS_SHA256 = "9a97853dee932e215c801075bbb2bd0abc95ebb276aeea7fd14c226c176492f7"
EXPECTED_HIDDEN_SHA256 = "8cabe5649d6aecaf65b66f86cd9852f0aa0779c6c4609c261b7da1c507c3b8f4"


def token_rank(logits: np.ndarray, token_id: int) -> dict:
    if logits.shape != (152576,) or not np.isfinite(logits).all():
        raise ValueError("PW-0150 logits row identity mismatch")
    if token_id < 0 or token_id >= logits.size:
        raise ValueError("PW-0150 token ID outside vocabulary")
    greedy = int(np.argmax(logits))
    return {
        "token_id": token_id,
        "logit": float(logits[token_id]),
        "rank": int(np.count_nonzero(logits > logits[token_id]) + 1),
        "gap_from_greedy": float(logits[greedy] - logits[token_id]),
        "greedy_token_id": greedy,
        "greedy_logit": float(logits[greedy]),
    }


def _authenticate(raw_path: Path, target_path: Path) -> tuple[dict, dict]:
    if sha256_file(raw_path) != RAW_SHA256:
        raise ValueError("PW-0150 raw manifest hash mismatch")
    if sha256_file(target_path) != TARGET_SHA256:
        raise ValueError("PW-0102 target manifest hash mismatch")
    raw = json.loads(raw_path.read_text())
    target = json.loads(target_path.read_text())
    if (
        raw.get("schema_version") != 1
        or raw.get("evidence_class") != "pw0150_exported_mask_dflash_proposal"
        or raw.get("status") != "passed"
        or raw.get("base_revision") != REVISION
        or raw.get("proposed_block_token_ids") != EXPECTED_PROPOSAL
        or raw.get("exported_mask_embedding_used") is not True
        or raw.get("performance_claim") is not None
        or target.get("evidence_class")
        != "pw0102_source_target_dflash_block_verification"
        or target.get("revision") != REVISION
        or target.get("proposed_block_token_ids") != EXPECTED_OLD_PROPOSAL
        or target.get("target_posterior_token_ids") != EXPECTED_TARGET_POSTERIOR
        or target.get("greedy_verification", {}).get("correction_token_id") != 13
    ):
        raise ValueError("PW-0150 source authority mismatch")
    return raw, target


def run(raw_path: Path, target_path: Path, output_path: Path, commit: str) -> dict:
    if output_path.exists():
        raise ValueError(f"refusing to overwrite {output_path}")
    if len(commit) != 40 or any(value not in "0123456789abcdef" for value in commit):
        raise ValueError("implementation commit must be lowercase 40-hex")
    started = time.perf_counter()
    safety = HostSafetyMonitor()
    raw, target = _authenticate(raw_path, target_path)
    safety.checkpoint("source_manifests_authenticated")

    raw_root = raw_path.parent
    logits_record = raw.get("captures", {}).get("draft_logits")
    hidden_record = raw.get("captures", {}).get("draft_final_hidden")
    if (
        logits_record.get("shape") != [1, 7, 152576]
        or logits_record.get("sha256") != EXPECTED_LOGITS_SHA256
        or hidden_record.get("shape") != [1, 8, 4096]
        or hidden_record.get("sha256") != EXPECTED_HIDDEN_SHA256
        or sha256_file(raw_root / logits_record["file"]) != EXPECTED_LOGITS_SHA256
        or sha256_file(raw_root / hidden_record["file"]) != EXPECTED_HIDDEN_SHA256
    ):
        raise ValueError("PW-0150 capture identity mismatch")
    logits = np.fromfile(raw_root / logits_record["file"], dtype="<f4")
    if logits.size != 7 * 152576:
        raise ValueError("PW-0150 logits payload size mismatch")
    logits = logits.reshape(7, 152576)
    first_target = int(target["target_posterior_token_ids"][0])
    first_proposal = int(raw["proposed_block_token_ids"][1])
    if first_proposal == first_target:
        raise ValueError("PW-0150 expected first-token rejection absent")
    rank = token_rank(logits[0], first_target)
    if rank["greedy_token_id"] != first_proposal:
        raise ValueError("PW-0150 proposal/logit mismatch")
    safety.checkpoint("capture_analysis_complete")

    raw_safety = raw.get("safety")
    if not isinstance(raw_safety, list) or not raw_safety:
        raise ValueError("PW-0150 raw safety evidence missing")
    safety_summary = {
        "snapshots": len(raw_safety),
        "minimum_system_memory_free_percent": min(
            row["system_memory_free_percent"] for row in raw_safety
        ),
        "maximum_process_peak_resident_bytes": max(
            row["process_peak_resident_bytes"] for row in raw_safety
        ),
        "maximum_process_physical_footprint_bytes": max(
            row["process_physical_footprint_bytes"] for row in raw_safety
        ),
        "maximum_swap_growth_bytes": max(row["swap_growth_bytes"] for row in raw_safety),
        "maximum_new_throttled_pages": max(
            row["new_throttled_pages"] for row in raw_safety
        ),
        "final_process_physical_footprint_bytes": raw_safety[-1][
            "process_physical_footprint_bytes"
        ],
    }
    if (
        safety_summary["minimum_system_memory_free_percent"] < 20
        or safety_summary["maximum_process_peak_resident_bytes"] > 8 * 1024**3
        or safety_summary["maximum_swap_growth_bytes"] > 512 * 1024**2
        or safety_summary["maximum_new_throttled_pages"] > 0
        or safety_summary["final_process_physical_footprint_bytes"] > 4 * 1024**3
    ):
        raise ValueError("PW-0150 raw Gate-8 evidence failed")

    del logits, raw, target
    safety.release_checkpoint(
        "source_evidence_released", ["PW-0150 draft logits", "authenticated manifests"]
    )
    safety.checkpoint("final_service_health")
    report = {
        "schema_version": 1,
        "evidence_class": "pw0150_exported_mask_dflash_control_analysis",
        "revision": REVISION,
        "commit": commit,
        "source_hashes": {
            "pw0150_raw_manifest_sha256": RAW_SHA256,
            "pw0102_target_manifest_sha256": TARGET_SHA256,
            "draft_logits_sha256": EXPECTED_LOGITS_SHA256,
            "draft_final_hidden_sha256": EXPECTED_HIDDEN_SHA256,
        },
        "old_proposed_block_token_ids": EXPECTED_OLD_PROPOSAL,
        "exported_mask_proposed_block_token_ids": EXPECTED_PROPOSAL,
        "first_required_target_token_id": first_target,
        "first_proposed_token_id": first_proposal,
        "first_target_token_draft_rank": rank,
        "matching_draft_suffix_tokens": 0,
        "accepted_anchor_tokens": 1,
        "A": 1,
        "minimum_possible_normalized_route_union_u": 1.0,
        "maximum_possible_a_over_u": 1.0,
        "passes_strict_routed_byte_leverage_gate": False,
        "new_target_walk_authorized": False,
        "decision": "reject_exported_mask_dflash8_on_frozen_pinned_base_trace",
        "raw_safety": safety_summary,
        "analysis_safety": safety.evidence(),
        "complete_wall_ms": (time.perf_counter() - started) * 1000.0,
        "accepted_tokens": 0,
        "performance_claim": None,
        "limitations": (
            "one frozen text prefix and one width-eight proposal; rejects this supplied "
            "draft/mask/base combination, not a base-trained proposer or wider lattice"
        ),
        "platform": platform.platform(),
    }
    atomic_write_new(output_path, canonical_json(report))
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-manifest", required=True, type=Path)
    parser.add_argument("--target-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--commit", required=True)
    arguments = parser.parse_args()
    try:
        result = run(
            arguments.raw_manifest,
            arguments.target_manifest,
            arguments.output,
            arguments.commit,
        )
        print(json.dumps({"output": str(arguments.output), "decision": result["decision"]}))
        return 0
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
        print(json.dumps({"error": str(error)}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
