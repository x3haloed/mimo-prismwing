#!/usr/bin/env python3
"""Fail-closed audit for one PW-0208 corrected verifier-window corpus run."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Callable

REVISION = "63651580ca774f8504f676040460aed3e1244ac1"
SEMANTIC = "mimo_v2_5_pw0208_native_mtp_corrected_verifier_window_capture"
TARGET_BONUS_SEMANTIC = (
    "mimo_v2_5_pw0208_native_mtp_corrected_verifier_window_capture_"
    "target_bonus_full_match_v1"
)
EVIDENCE_CLASS = "pw0208_native_mtp_corrected_window_capture"
CATEGORIES = {"ordinary", "code", "multilingual", "rare_route"}
WIDTH = 8
HIDDEN = 4096
LAYERS = 48
ROUTED_LAYERS = 47
TOP_K = 8
PRIMARY_WINDOWS = 8


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def verification_union(trace: list[dict[str, Any]]) -> float:
    require(len(trace) == LAYERS, "verification trace must contain 48 layers")
    normalized = []
    for layer, item in enumerate(trace):
        require(item["layer"] == layer, "verification layer indices are not contiguous")
        selected = item["selected_experts_by_position"]
        weights = item["route_weights_by_position"]
        if layer == 0:
            require(selected == [] and weights == [], "dense layer zero reported expert routes")
            continue
        require(len(selected) == WIDTH and len(weights) == WIDTH, "route row count")
        for experts, route_weights in zip(selected, weights, strict=True):
            require(len(experts) == TOP_K, "expert top-k width")
            require(len(set(experts)) == TOP_K, "duplicate expert in one route row")
            require(all(type(expert) is int and 0 <= expert < 256 for expert in experts), "expert id")
            require(len(route_weights) == TOP_K, "route-weight top-k width")
            require(all(math.isfinite(weight) and weight >= 0.0 for weight in route_weights), "route weight")
        derived = len({expert for row in selected for expert in row}) / WIDTH
        require(math.isclose(item["U"], derived, abs_tol=1e-12), "layer U")
        normalized.append(derived)
    return sum(normalized) / ROUTED_LAYERS


def audit_transaction(
    transaction: dict[str, Any], progress: dict[str, Any], index: int
) -> dict[str, float | int]:
    emitted = transaction["emitted_token_ids"]
    authorized = transaction["verifier_authorized_token_ids"]
    accepted = len(emitted)
    require(transaction["index"] == index, "non-contiguous transaction index")
    require(len(transaction["proposal_token_ids"]) == WIDTH, "proposal width")
    require(len(transaction["posterior_token_ids"]) == WIDTH, "posterior width")
    require(authorized[:accepted] == emitted, "emitted tokens lack verifier authority")
    require(transaction["retained_proposal_rows"] == accepted, "cache/output mismatch")
    require(1 <= accepted <= WIDTH - 1, "invalid accepted-token count")
    proposal_traces = transaction["proposal_layer_traces"]
    require(len(proposal_traces) == WIDTH - 1, "missing proposal route traces")
    require(all(len(step) == LAYERS for step in proposal_traces), "proposal layer trace width")
    derived_union = verification_union(transaction["verification_layer_traces"])
    require(math.isclose(transaction["U"], derived_union, abs_tol=1e-12), "transaction U")
    require(progress["phase"] == "transaction_complete", "progress transaction phase")
    require(progress["transaction"] == index, "progress transaction index")
    require(progress["emitted_tokens"] == accepted, "progress A mismatch")
    require(progress["retained_proposal_rows"] == accepted, "progress cache mismatch")
    require(math.isclose(progress["U"], derived_union, abs_tol=1e-12), "progress U mismatch")
    return {"transaction": index, "A": accepted, "U": derived_union, "A/U": accepted / derived_union}


def audit_target_bonus_transaction(
    transaction: dict[str, Any], progress: dict[str, Any], index: int
) -> dict[str, float | int]:
    emitted = transaction["emitted_token_ids"]
    authorized = transaction["verifier_authorized_token_ids"]
    proposal = transaction["proposal_token_ids"]
    posterior = transaction["posterior_token_ids"]
    accepted = len(emitted)
    verifier_retained = transaction["verifier_retained_proposal_rows"]
    retained = transaction["retained_proposal_rows"]
    require(transaction["index"] == index, "non-contiguous transaction index")
    require(len(proposal) == WIDTH, "proposal width")
    require(len(posterior) == WIDTH, "posterior width")
    require(authorized[:accepted] == emitted, "emitted tokens lack verifier authority")
    require(retained == accepted, "cache/output mismatch")
    require(1 <= accepted <= WIDTH, "invalid accepted-token count")
    require(1 <= verifier_retained <= WIDTH, "invalid verifier-retained row count")
    if transaction["proposal_converged"]:
        require(
            posterior[:-1] == proposal[1:],
            "converged proposal/posterior suffix mismatch",
        )
        require(
            authorized == [*proposal[1:], posterior[-1]],
            "converged target bonus authority mismatch",
        )
        require(verifier_retained == WIDTH, "converged verifier retention mismatch")
    else:
        require(verifier_retained == len(authorized), "mismatch verifier retention mismatch")
    require(retained <= verifier_retained, "output clipping retained rejected rows")
    proposal_traces = transaction["proposal_layer_traces"]
    require(len(proposal_traces) == WIDTH - 1, "missing proposal route traces")
    require(all(len(step) == LAYERS for step in proposal_traces), "proposal layer trace width")
    derived_union = verification_union(transaction["verification_layer_traces"])
    require(math.isclose(transaction["U"], derived_union, abs_tol=1e-12), "transaction U")
    require(progress["phase"] == "transaction_complete", "progress transaction phase")
    require(progress["transaction"] == index, "progress transaction index")
    require(progress["emitted_tokens"] == accepted, "progress A mismatch")
    require(progress["retained_proposal_rows"] == retained, "progress cache mismatch")
    require(math.isclose(progress["U"], derived_union, abs_tol=1e-12), "progress U mismatch")
    return {"transaction": index, "A": accepted, "U": derived_union, "A/U": accepted / derived_union}


def protected_baseline_survived(snapshots: list[dict[str, Any]]) -> bool:
    if not snapshots:
        return False
    baseline = {
        service: set(pids)
        for service, pids in snapshots[0]["protected_service_pids"].items()
    }
    return all(
        all(
            service in snapshot["protected_service_pids"]
            and pids.issubset(set(snapshot["protected_service_pids"][service]))
            for service, pids in baseline.items()
        )
        for snapshot in snapshots
    )


def _audit(
    report_path: Path,
    progress_path: Path,
    hidden_path: Path,
    *,
    category: str,
    commit: str | None = None,
    prompt_path: Path | None = None,
    expected_semantic: str,
    transaction_auditor: Callable[
        [dict[str, Any], dict[str, Any], int], dict[str, float | int]
    ],
) -> dict[str, Any]:
    require(category in CATEGORIES, "unknown category")
    report = json.loads(report_path.read_text())
    progress_lines = [
        json.loads(line) for line in progress_path.read_text().splitlines() if line
    ]
    require(report["schema_version"] == 6, "unexpected report schema")
    require(report["semantic"] == expected_semantic, "unexpected semantic")
    require(report["evidence_class"] == EVIDENCE_CLASS, "unexpected evidence class")
    require(report["revision"] == REVISION, "model revision")
    require(not report["git_dirty"], "run was not from a clean worktree")
    if commit is not None:
        require(report["commit"] == commit, "commit does not match")
    if prompt_path is not None:
        require(report["user_prompt_utf8"] == prompt_path.read_text(), "prompt does not match")
    require(sha256(progress_path) == report["progress_sha256"], "progress hash")
    require(report["metal_device"] == "Apple M1", "hardware is not Apple M1")
    require(report["batch_size"] == 1 and report["concurrency"] == 1, "batch or concurrency")
    require(report["verifier_width"] == WIDTH, "verifier width")
    require(report["requested_output_tokens"] == 64, "requested output token count")
    require(report["accepted_tokens"] == 64, "run did not complete 64 tokens")
    require(report["stop_reason"] == "requested_maximum", "capture stopped early")
    require(report["accepted_tokens"] == len(report["generated_token_ids"]), "accepted-token count")
    require(report["route_trace_captured"] is True, "routes were not captured")
    transactions = report["transactions"]
    require(len(transactions) >= PRIMARY_WINDOWS, "fewer than eight windows")
    require(len(progress_lines) == len(transactions) + 1, "progress lines")
    require(progress_lines[0]["phase"] == "prefill_complete", "missing prefill record")

    reconstructed = report["generated_token_ids"][:1]
    metrics = []
    for index, transaction in enumerate(transactions):
        metrics.append(transaction_auditor(transaction, progress_lines[index + 1], index))
        reconstructed.extend(transaction["emitted_token_ids"])
    require(reconstructed == report["generated_token_ids"], "transaction reconstruction")

    capture = report["native_mtp_window"]
    require(capture["category"] == category, "capture category")
    require(capture["artifact_file"] == hidden_path.name, "hidden artifact filename")
    require(capture["artifact_sha256"] == sha256(hidden_path), "hidden artifact hash")
    require(capture["windows"] == len(transactions), "hidden window count")
    require(capture["shape"] == [len(transactions), WIDTH, HIDDEN], "hidden shape")
    require(capture["dtype"] == "float32", "hidden dtype")
    require(capture["byte_order"] == "little_endian", "hidden byte order")
    require(hidden_path.stat().st_size == len(transactions) * WIDTH * HIDDEN * 4, "hidden byte size")

    proposal_ms = sum(item["proposal_wall_ms"] for item in transactions)
    verification_ms = sum(item["verification_wall_ms"] for item in transactions)
    require(math.isclose(proposal_ms, report["proposal_wall_ms"], abs_tol=0.01), "proposal timing sum")
    require(math.isclose(verification_ms, report["verification_wall_ms"], abs_tol=0.01), "verification timing sum")
    component_ms = sum(
        report[key]
        for key in ("preprocessing_wall_ms", "prefill_wall_ms", "proposal_wall_ms", "verification_wall_ms")
    )
    require(report["complete_wall_ms"] >= component_ms, "complete wall excludes work")
    require(report["peak_resident_bytes"] <= 8 * 1024**3, "peak residency exceeds release gate")
    snapshots = report["safety_snapshots"]
    require(snapshots, "missing safety snapshots")
    require(all(item["swap_growth_bytes"] == 0 for item in snapshots), "swap growth")
    require(all(item["new_throttled_pages"] == 0 for item in snapshots), "new throttling")
    require(protected_baseline_survived(snapshots), "baseline protected service PID disappeared")

    primary = metrics[:PRIMARY_WINDOWS]
    return {
        "status": "passed",
        "category": category,
        "report_sha256": sha256(report_path),
        "progress_sha256": sha256(progress_path),
        "hidden_sha256": sha256(hidden_path),
        "commit": report["commit"],
        "windows": len(transactions),
        "primary_windows": PRIMARY_WINDOWS,
        "primary_A": [item["A"] for item in primary],
        "primary_U": [item["U"] for item in primary],
        "primary_A_per_U": sum(item["A"] for item in primary) / sum(item["U"] for item in primary),
        "complete_wall_ms": report["complete_wall_ms"],
        "peak_resident_bytes": report["peak_resident_bytes"],
    }


def audit(
    report_path: Path,
    progress_path: Path,
    hidden_path: Path,
    *,
    category: str,
    commit: str | None = None,
    prompt_path: Path | None = None,
) -> dict[str, Any]:
    return _audit(
        report_path,
        progress_path,
        hidden_path,
        category=category,
        commit=commit,
        prompt_path=prompt_path,
        expected_semantic=SEMANTIC,
        transaction_auditor=audit_transaction,
    )


def audit_target_bonus(
    report_path: Path,
    progress_path: Path,
    hidden_path: Path,
    *,
    category: str,
    commit: str | None = None,
    prompt_path: Path | None = None,
) -> dict[str, Any]:
    return _audit(
        report_path,
        progress_path,
        hidden_path,
        category=category,
        commit=commit,
        prompt_path=prompt_path,
        expected_semantic=TARGET_BONUS_SEMANTIC,
        transaction_auditor=audit_target_bonus_transaction,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    parser.add_argument("progress", type=Path)
    parser.add_argument("hidden", type=Path)
    parser.add_argument("--category", required=True, choices=sorted(CATEGORIES))
    parser.add_argument("--commit")
    parser.add_argument("--prompt", type=Path)
    args = parser.parse_args()
    try:
        result = audit(
            args.report,
            args.progress,
            args.hidden,
            category=args.category,
            commit=args.commit,
            prompt_path=args.prompt,
        )
    except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError) as error:
        raise SystemExit(f"audit failed: {error}") from error
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
