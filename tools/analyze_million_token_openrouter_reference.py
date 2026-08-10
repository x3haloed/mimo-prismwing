#!/usr/bin/env python3
"""Consolidate PW-0160's bounded hosted-reference attempts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

try:
    from tools.host_safety import HostSafetyMonitor
    from tools.million_token_openrouter_reference import (
        MAX_OUTPUT_TOKENS,
        TARGET_PROMPT_TOKENS,
        canonical_json,
        read_json,
        sha256_bytes,
        summarize_http_error_body,
        verify_prepared,
    )
except ModuleNotFoundError:
    from host_safety import HostSafetyMonitor
    from million_token_openrouter_reference import (
        MAX_OUTPUT_TOKENS,
        TARGET_PROMPT_TOKENS,
        canonical_json,
        read_json,
        sha256_bytes,
        summarize_http_error_body,
        verify_prepared,
    )


SCHEMA_VERSION = 1
TRANSIENT_CODES = {429, 502, 503, 504}
MAXIMUM_ATTEMPTS = 3


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def git_value(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments], check=True, capture_output=True, text=True
    ).stdout.strip()


def require_clean_commit() -> str:
    commit = git_value("rev-parse", "HEAD")
    if len(commit) != 40 or git_value("status", "--porcelain"):
        raise ValueError("PW-0160 analysis requires a clean committed worktree")
    return commit


def validate_safety(snapshots: list[dict[str, Any]]) -> dict[str, Any]:
    if not snapshots or not snapshots[-1].get("release_boundary"):
        raise ValueError("PW-0160 safety evidence lacks a final release boundary")
    baseline = {
        name for name, pids in snapshots[0].get("protected_service_pids", {}).items() if pids
    }
    for row in snapshots:
        if row.get("system_memory_free_percent", 0) < 20:
            raise ValueError("PW-0160 safety memory-free gate failed")
        if row.get("process_physical_footprint_bytes", 0) > 8 * 1024**3:
            raise ValueError("PW-0160 safety physical-footprint gate failed")
        if row.get("process_peak_resident_bytes", 0) > 8 * 1024**3:
            raise ValueError("PW-0160 safety peak-RSS gate failed")
        if row.get("swap_growth_bytes", 0) > 512 * 1024**2:
            raise ValueError("PW-0160 safety swap gate failed")
        if row.get("new_throttled_pages", 0) != 0:
            raise ValueError("PW-0160 safety throttling gate failed")
        if any(not row.get("protected_service_pids", {}).get(name) for name in baseline):
            raise ValueError("PW-0160 protected service disappeared")
    final = snapshots[-1]
    if final.get("process_physical_footprint_bytes", 0) > 4 * 1024**3:
        raise ValueError("PW-0160 post-release footprint gate failed")
    return {
        "snapshot_count": len(snapshots),
        "minimum_system_memory_free_percent": min(
            row["system_memory_free_percent"] for row in snapshots
        ),
        "maximum_process_peak_resident_bytes": max(
            row["process_peak_resident_bytes"] for row in snapshots
        ),
        "maximum_process_physical_footprint_bytes": max(
            row["process_physical_footprint_bytes"] for row in snapshots
        ),
        "maximum_swap_growth_bytes": max(row["swap_growth_bytes"] for row in snapshots),
        "maximum_new_throttled_pages": max(
            row["new_throttled_pages"] for row in snapshots
        ),
        "final_release_physical_footprint_bytes": final[
            "process_physical_footprint_bytes"
        ],
        "protected_services_stable": True,
    }


def extract_attempt(root: Path) -> dict[str, Any]:
    prepared = verify_prepared(root)
    attempt_dirs = sorted(root.glob("attempt-*"))
    if len(attempt_dirs) != 1:
        raise ValueError(f"PW-0160 expected one attempt under {root}")
    attempt_dir = attempt_dirs[0]
    manifest_path = attempt_dir / "manifest.json"
    manifest = read_json(manifest_path)
    if manifest.get("schema_version") != SCHEMA_VERSION or manifest.get("status") != "failed":
        raise ValueError("PW-0160 bounded evidence unexpectedly contains a passing attempt")
    if manifest.get("commit") != prepared.get("commit"):
        raise ValueError("PW-0160 attempt/preparation commit mismatch")
    if manifest.get("request_sha256") != prepared.get("request_sha256"):
        raise ValueError("PW-0160 attempt/preparation request mismatch")

    body_path: Path | None = None
    body_kind: str
    declared_hash: str | None
    if manifest.get("response_file"):
        body_path = attempt_dir / manifest["response_file"]
        body_kind = "http_200_json_error"
        declared_hash = manifest.get("response_sha256")
    elif manifest.get("error_body_file"):
        body_path = attempt_dir / manifest["error_body_file"]
        body_kind = "http_error_body"
        declared_hash = manifest.get("error_body_sha256")
    elif (attempt_dir / "http-error-body.bin").is_file():
        body_path = attempt_dir / "http-error-body.bin"
        body_kind = "legacy_unbound_http_error_body"
        declared_hash = None
    else:
        raise ValueError("PW-0160 failed attempt lacks a preserved provider body")
    body = body_path.read_bytes()
    body_hash = sha256_bytes(body)
    if declared_hash is not None and body_hash != declared_hash:
        raise ValueError("PW-0160 provider body hash mismatch")
    parsed = json.loads(body)
    summary = summarize_http_error_body(body)
    if summary is None:
        raise ValueError("PW-0160 provider body lacks structured error metadata")
    code = summary.get("code")
    if code not in TRANSIENT_CODES:
        raise ValueError(f"PW-0160 provider failure is not classified transient: {code}")
    usage = parsed.get("usage") if isinstance(parsed, dict) else None
    if usage is not None:
        raise ValueError("PW-0160 failure unexpectedly reports billable usage")
    return {
        "root": str(root),
        "prepared_manifest_sha256": sha256_file(root / "prepared-manifest.json"),
        "attempt_manifest_sha256": sha256_file(manifest_path),
        "prepared_commit": prepared["commit"],
        "request_sha256": prepared["request_sha256"],
        "prompt_tokens": prepared["generation"]["prompt_tokens"],
        "needle_code": prepared["generation"]["needle_code"],
        "body_kind": body_kind,
        "provider_body_file": body_path.name,
        "provider_body_sha256": body_hash,
        "declared_provider_body_hash": declared_hash is not None,
        "error": summary,
        "reported_usage": None,
        "reported_cost_usd": None,
        "safety": validate_safety(prepared["safety_snapshots"] + manifest["safety_snapshots"]),
    }


def analyze(roots: list[Path], output: Path) -> Path:
    if len(roots) != MAXIMUM_ATTEMPTS:
        raise ValueError("PW-0160 analysis requires exactly three preserved attempts")
    commit = require_clean_commit()
    safety = HostSafetyMonitor()
    safety.checkpoint("analysis-start")
    attempts = [extract_attempt(root) for root in roots]
    safety.checkpoint("attempts-authenticated")
    request_hashes = {row["request_sha256"] for row in attempts}
    needle_codes = {row["needle_code"] for row in attempts}
    if len(request_hashes) != 1 or len(needle_codes) != 1:
        raise ValueError("PW-0160 attempts did not use one identical frozen request")
    codes = [row["error"]["code"] for row in attempts]
    if codes != [502, 429, 429]:
        raise ValueError(f"PW-0160 authoritative failure sequence drifted: {codes}")

    prompt_rate = float(
        verify_prepared(roots[-1])["endpoint_summary"]["provider_pricing"]["prompt"]
    )
    completion_rate = float(
        verify_prepared(roots[-1])["endpoint_summary"]["provider_pricing"]["completion"]
    )
    worst_case_spend = MAXIMUM_ATTEMPTS * (
        TARGET_PROMPT_TOKENS * prompt_rate + MAX_OUTPUT_TOKENS * completion_rate
    )
    if worst_case_spend > 0.50:
        raise ValueError("PW-0160 worst-case attempt spend exceeds contract")
    safety.release_checkpoint("analysis-released", ["attempt manifests and provider bodies"])
    report = {
        "schema_version": SCHEMA_VERSION,
        "experiment": "PW-0160",
        "analysis_commit": commit,
        "attempts": attempts,
        "attempt_count": len(attempts),
        "request_sha256": next(iter(request_hashes)),
        "needle_code": next(iter(needle_codes)),
        "provider_error_sequence": codes,
        "all_failures_transient_upstream_classes": True,
        "provider_capability_failure_observed": False,
        "passing_response_observed": False,
        "reported_cost_usd": 0.0,
        "reported_cost_authority": "all three error bodies omit usage and cost",
        "worst_case_if_every_attempt_had_been_fully_billed_usd": worst_case_spend,
        "contract_spend_ceiling_usd": 0.50,
        "attempt_budget_exhausted": True,
        "outcome": "inconclusive_transient_provider_unavailability",
        "hosted_one_million_reference_proven": False,
        "hosted_one_million_reference_killed": False,
        "changed_attention_authorized_by_pw0160": False,
        "accepted_tokens": 0,
        "performance_claim": False,
        "endpoint_tps": None,
        "safety_snapshots": safety.evidence(),
        "created_at_unix_ns": time.time_ns(),
    }
    output.mkdir(parents=True, exist_ok=False)
    path = output / "manifest.json"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(path, flags, 0o644)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(canonical_json(report))
        handle.flush()
        os.fsync(handle.fileno())
    return path


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser()
    root.add_argument("--attempt-root", action="append", required=True, type=Path)
    root.add_argument("--output", required=True, type=Path)
    return root


def main() -> int:
    arguments = parser().parse_args()
    try:
        print(analyze(arguments.attempt_root, arguments.output))
        return 0
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError, subprocess.CalledProcessError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
