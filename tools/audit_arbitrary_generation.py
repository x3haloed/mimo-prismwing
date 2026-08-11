#!/usr/bin/env python3
"""Fail-closed audit for a Prismwing arbitrary-text generation report."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"audit failed: {message}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    parser.add_argument("progress", type=Path)
    parser.add_argument("--commit")
    parser.add_argument("--prompt", type=Path)
    args = parser.parse_args()

    report = json.loads(args.report.read_text())
    progress_lines = [
        json.loads(line) for line in args.progress.read_text().splitlines() if line
    ]

    require(report["schema_version"] == 2, "unexpected report schema")
    require(
        report["semantic"]
        == "mimo_v2_5_sglang_directed_blockscaled_qkv_deinterleaved_text_generation",
        "unexpected or target-faithful semantic label",
    )
    require(report["revision"] == "63651580ca774f8504f676040460aed3e1244ac1", "model revision")
    require(not report["git_dirty"], "run was not from a clean worktree")
    if args.commit:
        require(report["commit"] == args.commit, "commit does not match")
    if args.prompt:
        require(report["user_prompt_utf8"] == args.prompt.read_text(), "prompt does not match")
    require(sha256(args.progress) == report["progress_sha256"], "progress hash")
    require(report["metal_device"] == "Apple M1", "hardware is not Apple M1")
    require(report["batch_size"] == 1, "batch size is not one")
    require(report["concurrency"] == 1, "concurrency is not one")
    require(report["verifier_width"] == 8, "verifier width is not eight")
    require(report["minimum_output_tokens"] == 32, "minimum is not 32")
    require(32 <= report["accepted_tokens"] <= 64, "accepted tokens outside 32--64")
    require(report["accepted_tokens"] == len(report["generated_token_ids"]), "accepted-token count")
    require(report["stop_reason"] == "completed_second_sentence", "stop reason")
    require(sum(report["generated_text"].count(mark) for mark in ".!?") >= 2, "sentence boundary")
    require(len(progress_lines) == len(report["transactions"]) + 1, "progress lines")
    require(progress_lines[0]["phase"] == "prefill_complete", "missing prefill record")

    reconstructed = report["generated_token_ids"][:1]
    acceptance = []
    for index, transaction in enumerate(report["transactions"]):
        emitted = transaction["emitted_token_ids"]
        authorized = transaction["verifier_authorized_token_ids"]
        accepted = len(emitted)
        union = transaction["U"]
        require(transaction["index"] == index, "non-contiguous transaction index")
        require(len(transaction["proposal_token_ids"]) == 8, "proposal width")
        require(len(transaction["posterior_token_ids"]) == 8, "posterior width")
        require(authorized[:accepted] == emitted, "emitted tokens lack verifier authority")
        require(transaction["retained_proposal_rows"] == accepted, "cache/output mismatch")
        require(accepted > 0 and union >= 1.0, "invalid A or U")
        progress = progress_lines[index + 1]
        require(progress["phase"] == "transaction_complete", "progress transaction phase")
        require(progress["transaction"] == index, "progress transaction index")
        require(progress["emitted_tokens"] == accepted, "progress A mismatch")
        require(progress["retained_proposal_rows"] == accepted, "progress cache mismatch")
        reconstructed.extend(emitted)
        acceptance.append({"transaction": index, "A": accepted, "U": union, "A/U": accepted / union})
    require(reconstructed == report["generated_token_ids"], "transaction reconstruction")

    proposal_ms = sum(item["proposal_wall_ms"] for item in report["transactions"])
    verification_ms = sum(item["verification_wall_ms"] for item in report["transactions"])
    require(abs(proposal_ms - report["proposal_wall_ms"]) < 0.01, "proposal timing sum")
    require(abs(verification_ms - report["verification_wall_ms"]) < 0.01, "verification timing sum")
    component_ms = sum(report[key] for key in (
        "preprocessing_wall_ms", "prefill_wall_ms", "proposal_wall_ms", "verification_wall_ms"
    ))
    require(report["complete_wall_ms"] >= component_ms, "complete wall excludes work")

    snapshots = report["safety_snapshots"]
    require(snapshots, "missing safety snapshots")
    require(all(item["swap_growth_bytes"] == 0 for item in snapshots), "swap growth")
    require(all(item["new_throttled_pages"] == 0 for item in snapshots), "new throttling")
    protected = snapshots[0]["protected_service_pids"]
    require(all(item["protected_service_pids"] == protected for item in snapshots), "protected service changed")

    output = {
        "status": "passed",
        "report_sha256": sha256(args.report),
        "progress_sha256": sha256(args.progress),
        "commit": report["commit"],
        "accepted_tokens": report["accepted_tokens"],
        "complete_wall_ms": report["complete_wall_ms"],
        "complete_path_tokens_per_second": report["accepted_tokens"] / (report["complete_wall_ms"] / 1000.0),
        "A_U_by_transaction": acceptance,
        "logical_source_bytes": report["logical_source_bytes"],
        "process_disk_bytes_read": report["process_disk_bytes_read"],
        "peak_resident_bytes": report["peak_resident_bytes"],
        "generated_text": report["generated_text"],
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
