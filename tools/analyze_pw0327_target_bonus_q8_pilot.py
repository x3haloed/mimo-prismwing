#!/usr/bin/env python3
"""Authenticate the four-category PW-0327 target-bonus q8 pilot."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

try:
    from tools.analyze_pw0319_corrected_route_bank import sha256_file
    from tools.analyze_pw0326_target_bonus import commit_fixture
    from tools.host_safety import HostSafetyMonitor, HostSafetyPolicy, HostSafetyViolation
    from tools.openrouter_reference import atomic_write_new, canonical_json
    from tools.reproduce_pw0311_k4_expert import verify_clean_commit
except ModuleNotFoundError:
    from analyze_pw0319_corrected_route_bank import sha256_file
    from analyze_pw0326_target_bonus import commit_fixture
    from host_safety import HostSafetyMonitor, HostSafetyPolicy, HostSafetyViolation
    from openrouter_reference import atomic_write_new, canonical_json
    from reproduce_pw0311_k4_expert import verify_clean_commit


EXPERIMENT_ID = "PW-0327"
REVISION = "63651580ca774f8504f676040460aed3e1244ac1"
MODEL_LOCK_SHA256 = "df8c74e6f9e1cef154aae5881b9042777653206aaff72855f7b1a1340e0d1050"
CHECKPOINT_RECEIPT_SHA256 = "9ddc8a99755f04ae2ea3c2484f6dd022d3f3a681b5a72c915ee4de833dbb0d03"
KERNEL_SHA256 = "9bc149eee32ebf28af35929d5fa160edfe9e1767cdcde59a54ec61b7016882ee"
SEMANTIC = (
    "mimo_v2_5_sglang_directed_blockscaled_qkv_deinterleaved_generation_probe_"
    "target_bonus_full_match_v1"
)
EVIDENCE_CLASS = "pw0205_arbitrary_prompt_bounded_generation_probe"
SOURCE_EXPERT_BYTES = 25_171_968
VERIFIER_WIDTH = 8
LAYERS = 48
TARGET_SELF_PROPOSER = (
    "greedy source-checkpoint proposer using the same retained K/V, deinterleaved "
    "checkpoint-TP QKV layout, and SGLang-directed block-scaled Metal arithmetic"
)
PROMPTS = {
    "ordinary": (
        "evals/fixtures/requests/pw0208-ordinary.txt",
        "d15e7fad81828b710303ce5e9dc5fd9c2104450108eb627167e6bc2080b9ee5d",
    ),
    "code": (
        "evals/fixtures/requests/pw0208-code.txt",
        "ad2940784d5028baa1dfab4585cb3a5a7fbffa22ca224f455fabc851549daefa",
    ),
    "multilingual": (
        "evals/fixtures/requests/pw0208-multilingual.txt",
        "6ece2dd3189d6b482f3356d344db6228e428db60a7530283eedc39be77d1beca",
    ),
    "rare_route": (
        "evals/fixtures/requests/pw0208-rare-route.txt",
        "5a71638364fff89af264dd3acea1ce31ef92128c3922cc8fb64826e793643373",
    ),
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def positive_int(value: Any, message: str) -> int:
    require(type(value) is int and value > 0, message)
    return value


def validate_byte_ledgers(
    transaction: dict[str, Any],
    report: dict[str, Any],
    progress_transaction: dict[str, Any],
    *,
    category: str,
) -> dict[str, int]:
    values = {
        "transaction_logical_source_bytes": positive_int(
            transaction["logical_source_bytes"],
            f"{category}: transaction logical byte ledger",
        ),
        "transaction_process_disk_bytes_read": positive_int(
            transaction["process_disk_bytes_read"],
            f"{category}: transaction physical byte ledger",
        ),
        "report_logical_source_bytes": positive_int(
            report["logical_source_bytes"], f"{category}: report logical byte ledger"
        ),
        "report_process_disk_bytes_read": positive_int(
            report["process_disk_bytes_read"], f"{category}: report physical byte ledger"
        ),
    }
    require(
        values["transaction_logical_source_bytes"] <= values["report_logical_source_bytes"]
        and values["transaction_process_disk_bytes_read"]
        <= values["report_process_disk_bytes_read"],
        f"{category}: transaction/report byte-ledger order",
    )
    require(
        type(progress_transaction["process_disk_bytes_read"]) is int
        and progress_transaction["process_disk_bytes_read"]
        == values["transaction_process_disk_bytes_read"],
        f"{category}: progress physical byte ledger",
    )
    return values


def route_metrics(traces: list[dict[str, Any]]) -> dict[str, Any]:
    require(len(traces) == LAYERS, "verification trace must contain 48 layers")
    identities: set[tuple[int, int]] = set()
    layer_u = []
    for layer, trace in enumerate(traces):
        require(int(trace["layer"]) == layer, "verification layer indices are not contiguous")
        selected = trace["selected_experts_by_position"]
        weights = trace["route_weights_by_position"]
        if layer == 0:
            require(selected == [] and weights == [], "dense layer zero contains routes")
            continue
        require(len(selected) == 8 and len(weights) == 8, "q8 route row count")
        union: set[int] = set()
        for expert_row, weight_row in zip(selected, weights, strict=True):
            require(
                len(expert_row) == 8
                and len(set(expert_row)) == 8
                and all(type(expert) is int and 0 <= expert < 256 for expert in expert_row),
                "expert route row mismatch",
            )
            require(
                len(weight_row) == 8
                and all(math.isfinite(float(weight)) and float(weight) > 0.0 for weight in weight_row)
                and abs(math.fsum(map(float, weight_row)) - 1.0) <= 2.0e-5,
                "route weight row mismatch",
            )
            union.update(expert_row)
            identities.update((layer, expert) for expert in expert_row)
        derived_layer_u = len(union) / 8.0
        require(math.isclose(float(trace["U"]), derived_layer_u, abs_tol=1.0e-12), "layer U mismatch")
        layer_u.append(derived_layer_u)
    mean_u = math.fsum(layer_u) / 47.0
    return {
        "U": mean_u,
        "unique_identities": len(identities),
        "unique_source_expert_bytes": len(identities) * SOURCE_EXPERT_BYTES,
        "identities": [
            {"layer": layer, "expert": expert}
            for layer, expert in sorted(identities)
        ],
    }


def validate_proposal_traces(traces: list[list[dict[str, Any]]]) -> None:
    require(len(traces) == VERIFIER_WIDTH - 1, "proposal trace must contain seven steps")
    for step, layer_traces in enumerate(traces):
        require(len(layer_traces) == LAYERS, f"proposal step {step} must contain 48 layers")
        for layer, trace in enumerate(layer_traces):
            require(int(trace["layer"]) == layer, "proposal layer indices are not contiguous")
            selected = trace["selected_experts_by_position"]
            weights = trace["route_weights_by_position"]
            if layer == 0:
                require(selected == [] and weights == [], "proposal dense layer zero contains routes")
                require(math.isclose(float(trace["U"]), 0.0, abs_tol=1.0e-12), "proposal layer U mismatch")
                continue
            require(len(selected) == 1 and len(weights) == 1, "proposal route row count")
            expert_row = selected[0]
            weight_row = weights[0]
            require(
                len(expert_row) == 8
                and len(set(expert_row)) == 8
                and all(type(expert) is int and 0 <= expert < 256 for expert in expert_row),
                "proposal expert route row mismatch",
            )
            require(
                len(weight_row) == 8
                and all(math.isfinite(float(weight)) and float(weight) > 0.0 for weight in weight_row)
                and abs(math.fsum(map(float, weight_row)) - 1.0) <= 2.0e-5,
                "proposal route weight row mismatch",
            )
            require(math.isclose(float(trace["U"]), 8.0, abs_tol=1.0e-12), "proposal layer U mismatch")


def safety_gate(snapshots: list[dict[str, Any]]) -> dict[str, Any]:
    require(bool(snapshots), "missing safety snapshots")
    policy = HostSafetyPolicy()
    baseline_names = {
        name for name, pids in snapshots[0]["protected_service_pids"].items() if pids
    }
    require(bool(baseline_names), "missing protected service baseline")
    for row in snapshots:
        free_percent = positive_int(row["system_memory_free_percent"], "Gate 8 free-memory field")
        footprint = positive_int(
            row["process_physical_footprint_bytes"], "Gate 8 physical-footprint field"
        )
        peak = positive_int(row["process_peak_resident_bytes"], "Gate 8 peak-resident field")
        require(
            free_percent >= policy.minimum_system_memory_free_percent
            and footprint <= policy.maximum_process_physical_footprint_bytes
            and peak <= policy.maximum_process_physical_footprint_bytes
            and (
                not bool(row["release_boundary"])
                or footprint <= policy.maximum_post_release_physical_footprint_bytes
            )
            and type(row["swap_growth_bytes"]) is int
            and row["swap_growth_bytes"] == policy.maximum_swap_growth_bytes
            and type(row["new_throttled_pages"]) is int
            and row["new_throttled_pages"] == policy.maximum_new_throttled_pages
            and all(row["protected_service_pids"].get(name) for name in baseline_names),
            "Gate 8 safety mismatch",
        )
    require(any(bool(row["release_boundary"]) for row in snapshots), "missing release boundary")
    return {
        "pass": True,
        "minimum_system_memory_free_percent": min(
            int(row["system_memory_free_percent"]) for row in snapshots
        ),
        "maximum_process_physical_footprint_bytes": max(
            int(row["process_physical_footprint_bytes"]) for row in snapshots
        ),
        "maximum_process_peak_resident_bytes": max(
            int(row["process_peak_resident_bytes"]) for row in snapshots
        ),
        "maximum_swap_growth_bytes": max(int(row["swap_growth_bytes"]) for row in snapshots),
        "maximum_new_throttled_pages": max(int(row["new_throttled_pages"]) for row in snapshots),
        "release_boundary_present": True,
        "protected_service_names": sorted(baseline_names),
    }


def analyze_report(
    *,
    category: str,
    report_path: Path,
    prompt_path: Path,
    prompt_sha256: str,
    capture_commit: str,
) -> dict[str, Any]:
    require(sha256_file(prompt_path) == prompt_sha256, f"{category}: prompt hash mismatch")
    report = json.loads(report_path.read_text())
    require(report["schema_version"] == 3, f"{category}: report schema")
    require(report["evidence_class"] == EVIDENCE_CLASS, f"{category}: evidence class")
    require(report["semantic"] == SEMANTIC, f"{category}: semantic")
    require(report["revision"] == REVISION, f"{category}: model revision")
    require(
        report["commit"] == capture_commit and report["git_dirty"] is False,
        f"{category}: Git identity",
    )
    require(report["model_lock_sha256"] == MODEL_LOCK_SHA256, f"{category}: model lock")
    require(
        report["checkpoint_verification_sha256"] == CHECKPOINT_RECEIPT_SHA256,
        f"{category}: checkpoint receipt",
    )
    require(report["kernel_sha256"] == KERNEL_SHA256, f"{category}: kernel")
    require(report["metal_device"] == "Apple M1", f"{category}: hardware")
    require(report["verifier_width"] == VERIFIER_WIDTH, f"{category}: verifier width")
    require(report["proposer"] == TARGET_SELF_PROPOSER, f"{category}: proposer identity")
    require(report["user_prompt_utf8"] == prompt_path.read_text(), f"{category}: prompt text")
    require(
        report["requested_output_tokens"] == 2
        and report["minimum_output_tokens"] == 2
        and report["accepted_tokens"] == 2
        and len(report["generated_token_ids"]) == 2,
        f"{category}: diagnostic output bound",
    )
    require(report["batch_size"] == 1 and report["concurrency"] == 1, f"{category}: batch/concurrency")
    require(report.get("route_trace_captured") is True, f"{category}: route trace flag")
    require("cold process start" in report["cache_state"], f"{category}: cold state")
    require(len(report["transactions"]) == 1, f"{category}: transaction count")
    progress_path = report_path.with_suffix(".progress.jsonl")
    require(
        sha256_file(progress_path) == report["progress_sha256"],
        f"{category}: progress hash",
    )
    progress = [
        json.loads(line)
        for line in progress_path.read_text().splitlines()
        if line.strip()
    ]
    require(
        len(progress) == 2
        and progress[0]["phase"] == "prefill_complete"
        and progress[1]["phase"] == "transaction_complete"
        and progress[1]["transaction"] == 0,
        f"{category}: progress transaction structure",
    )
    transaction = report["transactions"][0]
    require(transaction["index"] == 0, f"{category}: transaction index")
    proposal = list(map(int, transaction["proposal_token_ids"]))
    posterior = list(map(int, transaction["posterior_token_ids"]))
    require(
        len(proposal) == VERIFIER_WIDTH and len(posterior) == VERIFIER_WIDTH,
        f"{category}: q8 transaction width",
    )
    validate_proposal_traces(transaction["proposal_layer_traces"])
    commit_result = commit_fixture(proposal, posterior)
    authorized = list(map(int, transaction["verifier_authorized_token_ids"]))
    emitted = list(map(int, transaction["emitted_token_ids"]))
    require(authorized == commit_result["emitted"], f"{category}: verifier commit mismatch")
    require(
        int(transaction["verifier_retained_proposal_rows"])
        == commit_result["retained_proposal_rows"],
        f"{category}: verifier retention mismatch",
    )
    require(
        bool(transaction["proposal_converged"]) == commit_result["proposal_converged"],
        f"{category}: convergence mismatch",
    )
    require(emitted == authorized[:1], f"{category}: output clipping mismatch")
    require(int(transaction["retained_proposal_rows"]) == 1, f"{category}: clipped retention")
    require(
        progress[1]["emitted_tokens"] == 1
        and progress[1]["retained_proposal_rows"] == 1
        and progress[1]["proposal_converged"] == transaction["proposal_converged"],
        f"{category}: progress commit mismatch",
    )
    require(
        report["generated_token_ids"][0] == proposal[0]
        and report["generated_token_ids"][1] == emitted[0],
        f"{category}: generated token authority",
    )
    metrics = route_metrics(transaction["verification_layer_traces"])
    require(math.isclose(float(transaction["U"]), metrics["U"], abs_tol=1.0e-12), f"{category}: mean U")
    require(math.isclose(float(progress[1]["U"]), metrics["U"], abs_tol=1.0e-12), f"{category}: progress U")
    byte_ledgers = validate_byte_ledgers(
        transaction, report, progress[1], category=category
    )
    gate8 = safety_gate(report["safety_snapshots"])
    require(
        positive_int(report["peak_resident_bytes"], f"{category}: report peak resident")
        == gate8["maximum_process_peak_resident_bytes"],
        f"{category}: report/snapshot peak resident",
    )
    components = sum(
        float(report[name])
        for name in (
            "preprocessing_wall_ms",
            "prefill_wall_ms",
            "proposal_wall_ms",
            "verification_wall_ms",
        )
    )
    require(float(report["complete_wall_ms"]) >= components, f"{category}: complete wall")
    full_a = len(authorized)
    return {
        "category": category,
        "report_file": str(report_path),
        "report_sha256": sha256_file(report_path),
        "progress_file": str(progress_path),
        "progress_sha256": sha256_file(progress_path),
        "prompt_file": str(prompt_path),
        "prompt_sha256": prompt_sha256,
        "proposal_token_ids": proposal,
        "posterior_token_ids": posterior,
        "verifier_authorized_token_ids": authorized,
        "observable_emitted_token_ids": emitted,
        "proposal_converged": commit_result["proposal_converged"],
        "target_bonus_token_id": posterior[-1] if commit_result["proposal_converged"] else None,
        "A": full_a,
        "observable_transaction_tokens": len(emitted),
        "verifier_retained_proposal_rows": transaction["verifier_retained_proposal_rows"],
        "retained_proposal_rows_after_output_clip": transaction["retained_proposal_rows"],
        "U": metrics["U"],
        "A_per_U": full_a / metrics["U"],
        "route": metrics,
        **byte_ledgers,
        "preprocessing_wall_ms": float(report["preprocessing_wall_ms"]),
        "prefill_wall_ms": float(report["prefill_wall_ms"]),
        "proposal_wall_ms": float(report["proposal_wall_ms"]),
        "verification_wall_ms": float(report["verification_wall_ms"]),
        "complete_wall_ms": float(report["complete_wall_ms"]),
        "peak_resident_bytes": int(report["peak_resident_bytes"]),
        "gate8": gate8,
    }


def analyze(
    *,
    pilot_root: Path,
    output: Path,
    repo: Path,
    commit: str,
    capture_commit: str,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(output)
    repo = repo.resolve()
    verify_clean_commit(repo, commit)
    require(
        len(capture_commit) == 40
        and all(character in "0123456789abcdef" for character in capture_commit),
        "capture commit must be lowercase 40-hex",
    )
    safety = HostSafetyMonitor()
    categories = []
    for category, (relative_prompt, prompt_sha256) in PROMPTS.items():
        categories.append(
            analyze_report(
                category=category,
                report_path=pilot_root / category / "report.json",
                prompt_path=repo / relative_prompt,
                prompt_sha256=prompt_sha256,
                capture_commit=capture_commit,
            )
        )
    safety.checkpoint("pilot_reports_authenticated")
    full_match_categories = [row["category"] for row in categories if row["proposal_converged"]]
    gate = {
        "all_four_category_reports_pass": len(categories) == 4,
        "at_least_three_full_match_bonus_branches": len(full_match_categories) >= 3,
        "all_report_gate8_pass": all(row["gate8"]["pass"] for row in categories),
    }
    gate["pass"] = all(gate.values())
    safety.release_checkpoint("analysis_released", ["four pilot reports and route rows"])
    safety.checkpoint("final_service_health")
    report = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "status": "complete",
        "decision": (
            "authorize_complete_target_bonus_q8_causal_recapture"
            if gate["pass"]
            else "reject_complete_target_bonus_q8_causal_recapture"
        ),
        "analysis_commit": commit,
        "capture_commit": capture_commit,
        "authority": {
            "revision": REVISION,
            "model_lock_sha256": MODEL_LOCK_SHA256,
            "checkpoint_receipt_sha256": CHECKPOINT_RECEIPT_SHA256,
            "kernel_sha256": KERNEL_SHA256,
            "transaction_semantic": SEMANTIC,
            "evidence_class": EVIDENCE_CLASS,
        },
        "continuation_gate": gate,
        "full_match_categories": full_match_categories,
        "aggregate": {
            "categories": 4,
            "full_verifier_A": sum(row["A"] for row in categories),
            "sum_U": math.fsum(row["U"] for row in categories),
            "sum_A_per_sum_U": (
                sum(row["A"] for row in categories)
                / math.fsum(row["U"] for row in categories)
            ),
            "observable_generated_tokens_including_prefill_anchors": 8,
            "transaction_logical_source_bytes": sum(
                row["transaction_logical_source_bytes"] for row in categories
            ),
            "transaction_process_disk_bytes_read": sum(
                row["transaction_process_disk_bytes_read"] for row in categories
            ),
            "complete_wall_ms": sum(row["complete_wall_ms"] for row in categories),
        },
        "categories": categories,
        "analysis_safety_snapshots": safety.evidence(),
        "batch_size": 1,
        "concurrency": 1,
        "accepted_tokens": 8,
        "performance_claim": None,
        "claims_excluded": [
            "sustained or complete endpoint TPS",
            "native-MTP proposal quality or latency",
            "stale-route A+1 or bank/cache construction",
            "multimodal or full-capability promotion",
        ],
    }
    output.mkdir(parents=True)
    path = output / "analysis.json"
    atomic_write_new(path, canonical_json(report))
    print(json.dumps({"output": str(path), "decision": report["decision"]}))
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--capture-commit", required=True)
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
