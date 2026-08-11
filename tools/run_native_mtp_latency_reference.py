#!/usr/bin/env python3
"""Run PW-0211's authenticated native-MTP correctness/latency reference."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any

import numpy as np
import torch

try:
    from tools.dflash_semantics import verify_greedy_block
    from tools.generate_native_mtp_first_proposal import (
        PW0206_DECODE_SHA256,
        PW0206_PREFIX_SHA256,
        authenticate,
        load_target_hidden,
        sha256_file,
    )
    from tools.host_safety import HostSafetyMonitor
    from tools.native_mtp_reference import (
        generate_layer_proposal,
        q4_proposal_block,
        rotate_mtp_input_ids,
    )
    from tools.openrouter_reference import atomic_write_new, canonical_json
except ModuleNotFoundError:
    from dflash_semantics import verify_greedy_block
    from generate_native_mtp_first_proposal import (
        PW0206_DECODE_SHA256,
        PW0206_PREFIX_SHA256,
        authenticate,
        load_target_hidden,
        sha256_file,
    )
    from host_safety import HostSafetyMonitor
    from native_mtp_reference import (
        generate_layer_proposal,
        q4_proposal_block,
        rotate_mtp_input_ids,
    )
    from openrouter_reference import atomic_write_new, canonical_json


CORPUS_SHA256 = "a9bb6bd26bf048a2144133cc0a96023a8af112eae58122b666915149f2993a7b"
KNOWN_MTP_MANIFEST_SHA256 = "07233ee71f194c887d96aac2cb341239df1728a7e05fc36f692a1188c65b3379"


def endpoint_committed_tokens(verification: Any) -> int:
    """Convert source prefix convergence to newly committed endpoint output."""
    if (
        verification.q < 2
        or verification.accepted_length_a != verification.matching_draft_tokens + 1
        or not 1 <= verification.accepted_length_a <= verification.q
    ):
        raise ValueError("inconsistent greedy verification accounting")
    return min(verification.accepted_length_a, verification.q - 1)


def authenticate_corpus(path: Path) -> dict[str, Any]:
    if sha256_file(path) != CORPUS_SHA256:
        raise ValueError("PW-0208 complete-history corpus hash mismatch")
    manifest = json.loads(path.read_text())
    if (
        manifest.get("evidence_class") != "pw0208_balanced_corrected_native_mtp_window_corpus"
        or manifest.get("verifier_window_shape") != [8, 4096]
        or manifest.get("hidden_dtype") != "float32_little_endian"
        or len(manifest.get("primary_windows", [])) != 32
    ):
        raise ValueError("PW-0208 complete-history corpus identity mismatch")
    for record in [*manifest["sources"], *manifest["prefill_sources"]]:
        for kind in ("report", "progress", "hidden"):
            file_key, hash_key = f"{kind}_file", f"{kind}_sha256"
            if file_key in record and sha256_file(Path(record[file_key])) != record[hash_key]:
                raise ValueError(f"PW-0208 {record['category']} {kind} hash mismatch")
    return manifest


def load_segmented_history(window: dict[str, Any]) -> torch.Tensor:
    rows: list[np.ndarray] = []
    for segment in window["target_hidden_segments"]:
        if segment["byte_offset"] % 4 or segment["byte_length"] != segment["rows"] * 4096 * 4:
            raise ValueError("target hidden segment layout mismatch")
        values = np.fromfile(
            Path(segment["file"]),
            dtype="<f4",
            count=segment["rows"] * 4096,
            offset=segment["byte_offset"],
        )
        if values.size != segment["rows"] * 4096 or not np.isfinite(values).all():
            raise ValueError("target hidden segment payload mismatch")
        rows.append(values.reshape(segment["rows"], 4096))
    combined = np.concatenate(rows, axis=0)
    if combined.shape != (window["target_hidden_rows"], 4096):
        raise ValueError("complete target hidden history row mismatch")
    return torch.from_numpy(combined.copy()).to(torch.bfloat16)


def logits_identity(logits: torch.Tensor) -> str:
    payload = logits.detach().cpu().contiguous().numpy().astype("<f4", copy=False).tobytes()
    return hashlib.sha256(payload).hexdigest()


def source_control() -> tuple[str, bool]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, text=True, capture_output=True
    ).stdout.strip()
    dirty = bool(subprocess.run(
        ["git", "status", "--porcelain"], check=True, text=True, capture_output=True
    ).stdout.strip())
    return commit, dirty


def base_authorities(arguments: argparse.Namespace):
    return authenticate(
        arguments.checkpoint,
        arguments.verification,
        arguments.known_prefix,
        None,
        arguments.corrected_decode,
        arguments.source_lock,
        arguments.source_root,
    )


def run_known(arguments: argparse.Namespace, safety: HostSafetyMonitor) -> dict[str, Any]:
    started = time.monotonic()
    checkpoint, mtp_path, prefix_manifest, identities, anchor, expected, _ = base_authorities(arguments)
    safety.checkpoint("known_authorities_authenticated")
    target_hidden = load_target_hidden(arguments.known_prefix, prefix_manifest)
    input_ids = [*prefix_manifest["prompt_token_ids"][1:], anchor]
    result = generate_layer_proposal(
        checkpoint, mtp_path, 0, target_hidden, input_ids, safety.checkpoint
    )
    known_manifest_path = arguments.known_mtp_manifest
    if sha256_file(known_manifest_path) != KNOWN_MTP_MANIFEST_SHA256:
        raise ValueError("PW-0206 full-row MTP manifest hash mismatch")
    known_manifest = json.loads(known_manifest_path.read_text())
    known_logits = np.fromfile(
        known_manifest_path.parent / known_manifest["captures"]["logits"]["file"], dtype="<f4"
    )
    if known_logits.shape != (152_576,):
        raise ValueError("PW-0206 full-row MTP logits layout mismatch")
    exact = bool(torch.equal(result.logits, torch.from_numpy(known_logits.copy())))
    commit, dirty = source_control()
    return {
        "schema_version": 1,
        "evidence_class": "pw0211_last_row_native_mtp_known_proposal_validation",
        "status": "passed" if exact and result.proposal_token_id == expected else "rejected",
        "identities": {
            **identities,
            "known_prefix_manifest_sha256": PW0206_PREFIX_SHA256,
            "corrected_decode_sha256": PW0206_DECODE_SHA256,
            "known_full_row_mtp_manifest_sha256": KNOWN_MTP_MANIFEST_SHA256,
            "known_full_row_logits_sha256": known_manifest["captures"]["logits"]["sha256"],
        },
        "implementation": {"commit": commit, "dirty": dirty},
        "target_proposal_token_id": expected,
        "last_row_proposal_token_id": result.proposal_token_id,
        "full_row_logits_bit_identical": exact,
        "last_row_logits_sha256": logits_identity(result.logits),
        "top20": [{"token_id": token, "logit": value} for token, value in result.top20],
        "timings_ms": {**result.timings_ms, "complete": (time.monotonic() - started) * 1000},
        "hardware": {"machine": platform.machine(), "platform": platform.platform()},
        "runtime": {"python": sys.version, "torch": torch.__version__, "numpy": np.__version__},
        "safety": safety.evidence(),
        "accepted_tokens": 0,
        "performance_claim": None,
    }


def run_corpus(arguments: argparse.Namespace, safety: HostSafetyMonitor) -> dict[str, Any]:
    started = time.monotonic()
    checkpoint, mtp_path, _, identities, _, _, _ = base_authorities(arguments)
    corpus = authenticate_corpus(arguments.corpus_manifest)
    if not 0 <= arguments.corpus_index < len(corpus["primary_windows"]):
        raise ValueError("corpus index is out of range")
    window = corpus["primary_windows"][arguments.corpus_index]
    target_hidden = load_segmented_history(window)
    input_ids = window["mtp_layer0_input_token_ids"]
    if len(input_ids) != target_hidden.shape[0] or input_ids[-1] != window["anchor_token_id"]:
        raise ValueError("MTP input/history pairing mismatch")
    safety.checkpoint("corpus_authority_and_history_authenticated")

    layer_records = []
    proposals = []
    for layer in range(3):
        layer_started = time.monotonic()
        layer_result = generate_layer_proposal(
            checkpoint, mtp_path, layer, target_hidden, input_ids, safety.checkpoint
        )
        proposals.append(layer_result.proposal_token_id)
        layer_records.append({
            "layer": layer,
            "input_token_ids_sha256": hashlib.sha256(
                np.asarray(input_ids, dtype="<i8").tobytes()
            ).hexdigest(),
            "proposal_token_id": layer_result.proposal_token_id,
            "logits_sha256": logits_identity(layer_result.logits),
            "top20": [
                {"token_id": token, "logit": value} for token, value in layer_result.top20
            ],
            "timings_ms": {
                **layer_result.timings_ms,
                "complete": (time.monotonic() - layer_started) * 1000,
            },
        })
        del layer_result
        if layer < 2:
            input_ids = rotate_mtp_input_ids(input_ids, proposals[-1])

    block = q4_proposal_block(window["anchor_token_id"], proposals)
    posterior = window["posterior_token_ids"][:4]
    verification = verify_greedy_block(
        torch.tensor([block], dtype=torch.long), torch.tensor([posterior], dtype=torch.long)
    )
    # verify_greedy_block reports the converged prefix including the already
    # committed anchor. Endpoint A counts only newly committed output tokens.
    endpoint_committed_length = endpoint_committed_tokens(verification)
    commit, dirty = source_control()
    return {
        "schema_version": 1,
        "evidence_class": "pw0211_complete_history_native_mtp_q4_cpu_reference",
        "status": "passed",
        "identities": {**identities, "pw0208_corpus_manifest_sha256": CORPUS_SHA256},
        "implementation": {"commit": commit, "dirty": dirty},
        "corpus_index": arguments.corpus_index,
        "category": window["category"],
        "transaction_index": window["transaction_index"],
        "target_hidden_rows": window["target_hidden_rows"],
        "target_anchor_token_id": window["anchor_token_id"],
        "native_mtp_q4_block_token_ids": block,
        "target_posterior_token_ids": posterior,
        "greedy_verification": verification.to_dict(),
        "endpoint_committed_length_A": endpoint_committed_length,
        "layer_results": layer_records,
        "control": {
            "proposal_block_token_ids": window["proposal_token_ids"],
            "A": window["A"],
            "U": window["U"],
            "proposal_wall_ms": window["proposal_wall_ms"],
            "verification_wall_ms": window["verification_wall_ms"],
        },
        "timings_ms": {"complete": (time.monotonic() - started) * 1000},
        "hardware": {"machine": platform.machine(), "platform": platform.platform()},
        "runtime": {"python": sys.version, "torch": torch.__version__, "numpy": np.__version__},
        "safety": safety.evidence(),
        "accepted_tokens": 0,
        "performance_claim": None,
    }


def run_live_request(arguments: argparse.Namespace, safety: HostSafetyMonitor) -> dict[str, Any]:
    started = time.monotonic()
    checkpoint, mtp_path, _, identities, _, _, _ = base_authorities(arguments)
    request_bytes = arguments.live_request.read_bytes()
    request = json.loads(request_bytes)
    hidden_path = Path(request.get("target_hidden_file", ""))
    rows = request.get("target_hidden_rows")
    input_ids = request.get("mtp_layer0_input_token_ids")
    anchor = request.get("anchor_token_id")
    if (
        request.get("schema_version") != 1
        or request.get("semantic") != "pw0211_live_target_cache_native_mtp_request"
        or not isinstance(rows, int)
        or rows < 1
        or not isinstance(input_ids, list)
        or len(input_ids) != rows
        or input_ids[-1] != anchor
        or sha256_file(hidden_path) != request.get("target_hidden_sha256")
    ):
        raise ValueError("live native MTP request authority mismatch")
    values = np.fromfile(hidden_path, dtype="<f4")
    if values.size != rows * 4096 or not np.isfinite(values).all():
        raise ValueError("live native MTP target hidden payload mismatch")
    target_hidden = torch.from_numpy(values.reshape(rows, 4096).copy()).to(torch.bfloat16)
    safety.checkpoint("live_request_authority_authenticated")
    layer_records = []
    proposals = []
    for layer in range(3):
        layer_started = time.monotonic()
        layer_result = generate_layer_proposal(
            checkpoint, mtp_path, layer, target_hidden, input_ids, safety.checkpoint
        )
        proposals.append(layer_result.proposal_token_id)
        layer_records.append({
            "layer": layer,
            "proposal_token_id": layer_result.proposal_token_id,
            "logits_sha256": logits_identity(layer_result.logits),
            "timings_ms": {
                **layer_result.timings_ms,
                "complete": (time.monotonic() - layer_started) * 1000,
            },
        })
        del layer_result
        if layer < 2:
            input_ids = rotate_mtp_input_ids(input_ids, proposals[-1])
    block = q4_proposal_block(anchor, proposals)
    commit, dirty = source_control()
    snapshots = safety.evidence()
    process_disk_bytes_read = (
        snapshots[-1]["process_disk_bytes_read"] - snapshots[0]["process_disk_bytes_read"]
    )
    logical_source_bytes = (
        3 * 396_466_816
        + 3 * rows * 4096 * 2
        + 3 * 152_576 * 4096 * 2
    )
    return {
        "schema_version": 1,
        "evidence_class": "pw0211_live_target_cache_native_mtp_q4_proposal",
        "status": "passed",
        "identities": {
            **identities,
            "request_sha256": hashlib.sha256(request_bytes).hexdigest(),
            "target_hidden_sha256": request["target_hidden_sha256"],
        },
        "implementation": {"commit": commit, "dirty": dirty},
        "target_hidden_rows": rows,
        "anchor_token_id": anchor,
        "native_mtp_q4_block_token_ids": block,
        "layer_results": layer_records,
        "logical_source_bytes": logical_source_bytes,
        "process_disk_bytes_read": process_disk_bytes_read,
        "timings_ms": {"complete": (time.monotonic() - started) * 1000},
        "hardware": {"machine": platform.machine(), "platform": platform.platform()},
        "runtime": {"python": sys.version, "torch": torch.__version__, "numpy": np.__version__},
        "safety": snapshots,
        "accepted_tokens": 0,
        "performance_claim": None,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--verification", required=True, type=Path)
    parser.add_argument("--known-prefix", required=True, type=Path)
    parser.add_argument("--corrected-decode", required=True, type=Path)
    parser.add_argument("--known-mtp-manifest", required=True, type=Path)
    parser.add_argument("--source-lock", required=True, type=Path)
    parser.add_argument("--source-root", required=True, type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--validate-known", action="store_true")
    mode.add_argument("--corpus-manifest", type=Path)
    mode.add_argument("--live-request", type=Path)
    parser.add_argument("--corpus-index", type=int, default=0)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    safety = HostSafetyMonitor()
    try:
        torch.set_num_threads(1)
        arguments.output.mkdir(parents=True, exist_ok=False)
        if arguments.validate_known:
            report = run_known(arguments, safety)
        elif arguments.live_request is not None:
            report = run_live_request(arguments, safety)
        else:
            report = run_corpus(arguments, safety)
        atomic_write_new(arguments.output / "report.json", canonical_json(report))
        print(json.dumps({
            "output": str(arguments.output),
            "status": report["status"],
            "evidence_class": report["evidence_class"],
        }))
        return 0 if report["status"] == "passed" else 1
    except Exception as error:
        if arguments.output.is_dir() and not (arguments.output / "failure.json").exists():
            atomic_write_new(arguments.output / "failure.json", canonical_json({
                "schema_version": 1,
                "evidence_class": "pw0211_native_mtp_reference_failure",
                "status": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
                "safety": safety.evidence(),
            }))
        print(json.dumps({"error": str(error), "output": str(arguments.output)}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
