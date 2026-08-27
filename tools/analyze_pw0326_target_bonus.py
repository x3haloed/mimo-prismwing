#!/usr/bin/env python3
"""Verify the PW-0326 full-match target-bonus transaction repair."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
from typing import Any

try:
    from tools.host_safety import HostSafetyMonitor, HostSafetyViolation
    from tools.openrouter_reference import atomic_write_new, canonical_json
    from tools.reproduce_pw0311_k4_expert import verify_clean_commit
except ModuleNotFoundError:
    from host_safety import HostSafetyMonitor, HostSafetyViolation
    from openrouter_reference import atomic_write_new, canonical_json
    from reproduce_pw0311_k4_expert import verify_clean_commit


EXPERIMENT_ID = "PW-0326"
PW0204_SHA256 = "b2119f645bc02b8ed8ea2c3c2f4f13dc48ddfa8a990b7867b83d873317401a3c"
SOURCES_SHA256 = "2eaef140bb7164efdde31718254cf5f217b81fb4272920bc61849d7df1e9820a"
TRANSACTION_SEMANTIC = "target_bonus_full_match_v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def commit_fixture(proposal: list[int], posterior: list[int]) -> dict[str, Any]:
    if len(proposal) < 2 or len(proposal) != len(posterior):
        raise ValueError("Jacobi acceptance requires equal widths of at least two")
    mismatch = next(
        (
            index
            for index in range(len(proposal) - 1)
            if posterior[index] != proposal[index + 1]
        ),
        None,
    )
    if mismatch is not None:
        correction = posterior[mismatch]
        return {
            "proposal": proposal,
            "posterior": posterior,
            "emitted": [*proposal[1 : mismatch + 1], correction],
            "retained_proposal_rows": mismatch + 1,
            "next_anchor": correction,
            "proposal_converged": False,
        }
    target_bonus = posterior[-1]
    return {
        "proposal": proposal,
        "posterior": posterior,
        "emitted": [*proposal[1:], target_bonus],
        "retained_proposal_rows": len(proposal),
        "next_anchor": target_bonus,
        "proposal_converged": True,
    }


def parse_rust_test_summary(output: str) -> dict[str, int]:
    matches = re.findall(
        r"test result: ok\. (\d+) passed; (\d+) failed; (\d+) ignored; (\d+) measured; (\d+) filtered out",
        output,
    )
    if not matches:
        raise ValueError("Rust test output lacks a passing library summary")
    passed, failed, ignored, measured, filtered = map(int, matches[-1])
    if passed < 1 or failed != 0 or filtered != 0:
        raise ValueError("complete Rust library suite did not pass")
    return {
        "passed": passed,
        "failed": failed,
        "ignored": ignored,
        "measured": measured,
        "filtered_out": filtered,
    }


def analyze(*, repo: Path, output: Path, commit: str) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(output)
    repo = repo.resolve()
    verify_clean_commit(repo, commit)
    pw0204 = repo / "experiments" / "PW-0204-arbitrary-prompt-generation-transaction.md"
    sources = repo / "docs" / "SOURCES.md"
    source = repo / "src" / "text_endpoint.rs"
    if sha256_file(pw0204) != PW0204_SHA256:
        raise ValueError("PW-0204 transaction authority mismatch")
    if sha256_file(sources) != SOURCES_SHA256:
        raise ValueError("target-bonus source authority mismatch")

    mismatch = commit_fixture(
        [264, 13, 15, 13, 15, 15, 15, 15],
        [13, 15, 13, 15, 481, 13, 15, 15],
    )
    converged = commit_fixture([41, 42, 43, 44], [42, 43, 44, 45])
    q2 = commit_fixture([1, 2], [2, 3])
    second = commit_fixture([45, 46, 47, 48], [46, 47, 48, 49])
    two_transaction_tokens = [*converged["emitted"], *second["emitted"]]
    if (
        mismatch["emitted"] != [13, 15, 13, 15, 481]
        or mismatch["retained_proposal_rows"] != 5
        or mismatch["next_anchor"] != 481
        or converged["emitted"] != [42, 43, 44, 45]
        or converged["retained_proposal_rows"] != 4
        or converged["next_anchor"] != 45
        or q2["emitted"] != [2, 3]
        or q2["retained_proposal_rows"] != 2
        or two_transaction_tokens != [42, 43, 44, 45, 46, 47, 48, 49]
    ):
        raise ValueError("target-bonus deterministic fixture failed")

    safety = HostSafetyMonitor()
    safety.checkpoint("authorities_and_fixtures_loaded")
    command = ["cargo", "test", "--lib", "--", "--nocapture"]
    completed = subprocess.run(
        command,
        cwd=repo,
        check=False,
        text=True,
        capture_output=True,
    )
    combined = completed.stdout + completed.stderr
    if completed.returncode != 0:
        raise RuntimeError("complete Rust library suite failed")
    rust_summary = parse_rust_test_summary(combined)
    safety.release_checkpoint("test_process_released", ["cargo test process"])
    safety.checkpoint("final_service_health")

    report = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "status": "complete",
        "decision": "promote_target_bonus_transaction_repair",
        "commit": commit,
        "transaction_semantic": TRANSACTION_SEMANTIC,
        "authority": {
            "pw0204_transaction_contract_sha256": PW0204_SHA256,
            "published_target_bonus_source_ledger_sha256": SOURCES_SHA256,
            "text_endpoint_source_sha256": sha256_file(source),
        },
        "fixtures": {
            "mismatch_control": mismatch,
            "full_match_target_bonus": converged,
            "minimum_width_full_match": q2,
            "second_full_match": second,
            "two_transaction_emitted_tokens": two_transaction_tokens,
            "target_bonus_emitted_once_across_boundary": True,
        },
        "rust_library_tests": {
            "command": command,
            "summary": rust_summary,
            "combined_output_sha256": hashlib.sha256(combined.encode()).hexdigest(),
        },
        "measurement_context": {
            "hardware": "Apple M1 16 GiB",
            "batch_size": 1,
            "concurrency": 1,
            "accepted_tokens": 0,
            "A": 0,
            "U": None,
            "bytes_moved": 0,
            "cache_state": "deterministic transaction fixtures; no model cache populated",
            "prefill_state": "not applicable",
        },
        "legacy_evidence_policy": (
            "PW-0204 through PW-0325 reports remain immutable bonus-free authorities; "
            "no A+1 projection or stale-route reuse is admissible"
        ),
        "safety_snapshots": safety.evidence(),
        "accepted_tokens": 0,
        "performance_claim": None,
    }
    output.mkdir(parents=True)
    path = output / "analysis.json"
    atomic_write_new(path, canonical_json(report))
    print(json.dumps({"output": str(path), "decision": report["decision"]}))
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--commit", required=True)
    try:
        analyze(**vars(parser.parse_args()))
        return 0
    except (
        FileExistsError,
        HostSafetyViolation,
        KeyError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ) as error:
        print(json.dumps({"error": str(error)}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
