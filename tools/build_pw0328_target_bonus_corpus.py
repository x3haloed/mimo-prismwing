#!/usr/bin/env python3
"""Build the repaired-semantic PW-0328 balanced q8 causal corpus."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import struct
from typing import Any

try:
    from tools.analyze_pw0319_corrected_route_bank import sha256_file
    from tools.analyze_pw0326_target_bonus import commit_fixture
    from tools.analyze_pw0327_target_bonus_q8_pilot import (
        PROMPTS,
        TARGET_SELF_PROPOSER,
        route_metrics,
        safety_gate,
        validate_byte_ledgers,
        validate_proposal_traces,
    )
    from tools.host_safety import HostSafetyMonitor
    from tools.openrouter_reference import atomic_write_new, canonical_json
    from tools.reproduce_pw0311_k4_expert import verify_clean_commit
except ModuleNotFoundError:
    from analyze_pw0319_corrected_route_bank import sha256_file
    from analyze_pw0326_target_bonus import commit_fixture
    from analyze_pw0327_target_bonus_q8_pilot import (
        PROMPTS,
        TARGET_SELF_PROPOSER,
        route_metrics,
        safety_gate,
        validate_byte_ledgers,
        validate_proposal_traces,
    )
    from host_safety import HostSafetyMonitor
    from openrouter_reference import atomic_write_new, canonical_json
    from reproduce_pw0311_k4_expert import verify_clean_commit


EXPERIMENT_ID = "PW-0328"
REVISION = "63651580ca774f8504f676040460aed3e1244ac1"
MODEL_LOCK_SHA256 = "df8c74e6f9e1cef154aae5881b9042777653206aaff72855f7b1a1340e0d1050"
CHECKPOINT_RECEIPT_SHA256 = "9ddc8a99755f04ae2ea3c2484f6dd022d3f3a681b5a72c915ee4de833dbb0d03"
TOKENIZER_SHA256 = "633518aad78f9f61bae2ae420d621215754a4424c918b052cd8c22a3b59e99d2"
TOKENIZER_CONFIG_SHA256 = "fd34b805f75a890a5c123d79a2982bbe240b3b6efb156d22401bd619484d9bd2"
CAPTURE_KERNEL_SHA256 = "9bc149eee32ebf28af35929d5fa160edfe9e1767cdcde59a54ec61b7016882ee"
CAPTURE_SEMANTIC = (
    "mimo_v2_5_pw0208_native_mtp_corrected_verifier_window_capture_"
    "target_bonus_full_match_v1"
)
CAPTURE_EVIDENCE_CLASS = "pw0208_native_mtp_corrected_window_capture"
PREFILL_SEMANTIC = "mimo_v2_5_pw0208_corrected_target_layer47_prefill_hidden_capture"
PREFILL_EVIDENCE_CLASS = "pw0208_native_mtp_corrected_prefill_hidden_capture"
VERIFIER_HIDDEN_SEMANTIC = (
    "consecutive_target_layer_47_final_hidden_before_model_final_norm_for_each_"
    "width_eight_verifier_window_and_row"
)
PREFILL_HIDDEN_SEMANTIC = (
    "target_layer_47_final_hidden_before_model_final_norm_for_each_serialized_prompt_token"
)
CATEGORIES = tuple(PROMPTS)
PRIMARY_WINDOWS = 8
WIDTH = 8
HIDDEN = 4096
VOCAB = 152_576
HIDDEN_ROW_BYTES = HIDDEN * 4
WINDOW_BYTES = WIDTH * HIDDEN_ROW_BYTES
SOURCE_EXPERT_BYTES = 25_171_968


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def strict_tokens(value: Any, message: str) -> list[int]:
    require(
        isinstance(value, list)
        and bool(value)
        and all(type(token) is int and 0 <= token < VOCAB for token in value),
        message,
    )
    return value


def strict_int(value: Any, message: str, *, minimum: int = 0) -> int:
    require(type(value) is int and value >= minimum, message)
    return value


def finite_f32_payload(path: Path, expected_bytes: int, message: str) -> None:
    require(path.stat().st_size == expected_bytes, f"{message}: byte size")
    with path.open("rb") as source:
        while payload := source.read(1024 * 1024):
            require(len(payload) % 4 == 0, f"{message}: alignment")
            require(
                all(math.isfinite(value[0]) for value in struct.iter_unpack("<f", payload)),
                f"{message}: non-finite value",
            )


def validate_aggregate_byte_ledgers(
    transactions: list[dict[str, Any]], report: dict[str, Any], *, category: str
) -> None:
    transaction_logical = sum(
        strict_int(
            transaction["logical_source_bytes"],
            f"{category}: aggregate transaction logical byte ledger",
            minimum=1,
        )
        for transaction in transactions
    )
    transaction_physical = sum(
        strict_int(
            transaction["process_disk_bytes_read"],
            f"{category}: aggregate transaction physical byte ledger",
            minimum=1,
        )
        for transaction in transactions
    )
    report_logical = strict_int(
        report["logical_source_bytes"], f"{category}: report logical byte ledger", minimum=1
    )
    report_physical = strict_int(
        report["process_disk_bytes_read"],
        f"{category}: report physical byte ledger",
        minimum=1,
    )
    require(
        transaction_logical <= report_logical and transaction_physical <= report_physical,
        f"{category}: aggregate transaction/report byte-ledger order",
    )


def selected_primary_transactions(report: dict[str, Any]) -> list[dict[str, Any]]:
    transactions = report["transactions"]
    require(len(transactions) >= PRIMARY_WINDOWS, "capture contains fewer than eight transactions")
    selected = transactions[:PRIMARY_WINDOWS]
    require(
        [transaction["index"] for transaction in selected] == list(range(PRIMARY_WINDOWS)),
        "primary transaction indices are not zero through seven",
    )
    return selected


def transaction_semantics(
    transaction: dict[str, Any], *, index: int, terminal: bool
) -> dict[str, Any]:
    require(
        strict_int(transaction["index"], "invalid transaction index") == index,
        "non-contiguous transaction index",
    )
    proposal = strict_tokens(transaction["proposal_token_ids"], "invalid proposal tokens")
    posterior = strict_tokens(transaction["posterior_token_ids"], "invalid posterior tokens")
    require(len(proposal) == WIDTH and len(posterior) == WIDTH, "transaction is not q8")
    expected = commit_fixture(proposal, posterior)
    authorized = strict_tokens(
        transaction["verifier_authorized_token_ids"], "invalid verifier-authorized tokens"
    )
    emitted = strict_tokens(transaction["emitted_token_ids"], "invalid emitted tokens")
    require(authorized == expected["emitted"], "verifier-authorized commit mismatch")
    require(emitted == authorized[: len(emitted)], "observable output is not an authority prefix")
    verifier_retained = strict_int(
        transaction["verifier_retained_proposal_rows"],
        "invalid verifier retention",
        minimum=1,
    )
    retained = strict_int(
        transaction["retained_proposal_rows"], "invalid observable retention", minimum=1
    )
    require(
        verifier_retained == expected["retained_proposal_rows"],
        "verifier retention mismatch",
    )
    require(
        transaction["proposal_converged"] is expected["proposal_converged"],
        "proposal convergence mismatch",
    )
    require(retained == len(emitted), "observable retention mismatch")
    if not terminal:
        require(emitted == authorized, "nonterminal transaction was output-clipped")
        require(
            retained == verifier_retained,
            "nonterminal transaction dropped verifier-retained rows",
        )
    return {
        "proposal_token_ids": proposal,
        "posterior_token_ids": posterior,
        "verifier_authorized_token_ids": authorized,
        "observable_emitted_token_ids": emitted,
        "proposal_converged": expected["proposal_converged"],
        "next_anchor_token_id": expected["next_anchor"],
        "A": len(authorized),
        "observable_A": len(emitted),
        "verifier_retained_proposal_rows": verifier_retained,
        "retained_proposal_rows": retained,
    }


def target_hidden_binding(
    report: dict[str, Any],
    transaction_index: int,
    prefill_hidden_file: str,
    verifier_hidden_file: str,
) -> dict[str, Any]:
    require(
        type(transaction_index) is int
        and 0 <= transaction_index < len(report["transactions"]),
        "transaction index out of range",
    )
    if transaction_index == 0:
        row = len(report["prompt_token_ids"]) - 1
        require(row >= 0, "transaction zero has no prefill target-hidden row")
        return {
            "target_hidden_source": "prefill",
            "target_hidden_source_transaction_index": None,
            "target_hidden_source_row": row,
            "target_hidden_file": prefill_hidden_file,
            "target_hidden_byte_offset": row * HIDDEN_ROW_BYTES,
            "target_hidden_byte_length": HIDDEN_ROW_BYTES,
        }
    previous = report["transactions"][transaction_index - 1]
    require(previous["index"] == transaction_index - 1, "preceding transaction index mismatch")
    row = strict_int(
        previous["retained_proposal_rows"], "invalid preceding retained rows", minimum=1
    ) - 1
    require(0 <= row < WIDTH, "preceding retained row cannot supply target hidden")
    return {
        "target_hidden_source": "verifier_transaction",
        "target_hidden_source_transaction_index": transaction_index - 1,
        "target_hidden_source_row": row,
        "target_hidden_file": verifier_hidden_file,
        "target_hidden_byte_offset": (transaction_index - 1) * WINDOW_BYTES
        + row * HIDDEN_ROW_BYTES,
        "target_hidden_byte_length": HIDDEN_ROW_BYTES,
    }


def mtp_history_binding(
    report: dict[str, Any],
    transaction_index: int,
    prefill_hidden_file: str,
    verifier_hidden_file: str,
) -> dict[str, Any]:
    require(
        type(transaction_index) is int
        and 0 <= transaction_index < len(report["transactions"]),
        "transaction index out of range",
    )
    prompt_ids = strict_tokens(report["prompt_token_ids"], "invalid prompt tokens")
    generated = [strict_tokens(report["generated_token_ids"], "invalid generated tokens")[0]]
    segments = [
        {
            "source": "prefill",
            "file": prefill_hidden_file,
            "byte_offset": 0,
            "byte_length": len(prompt_ids) * HIDDEN_ROW_BYTES,
            "rows": len(prompt_ids),
        }
    ]
    for previous in report["transactions"][:transaction_index]:
        retained = strict_int(
            previous["retained_proposal_rows"], "invalid retained history rows", minimum=1
        )
        require(retained <= WIDTH, "invalid retained history rows")
        segments.append(
            {
                "source": "verifier_transaction",
                "transaction_index": previous["index"],
                "file": verifier_hidden_file,
                "byte_offset": previous["index"] * WINDOW_BYTES,
                "byte_length": retained * HIDDEN_ROW_BYTES,
                "rows": retained,
            }
        )
        generated.extend(strict_tokens(previous["emitted_token_ids"], "invalid prior emitted tokens"))
    transaction = report["transactions"][transaction_index]
    anchor = strict_tokens(transaction["proposal_token_ids"], "invalid proposal tokens")[0]
    require(generated[-1] == anchor, "reconstructed history anchor mismatch")
    target_input_ids = [*prompt_ids, *generated[:-1]]
    require(
        sum(segment["rows"] for segment in segments) == len(target_input_ids),
        "target hidden and token history lengths differ",
    )
    require(len(target_input_ids) >= 1, "target input history is empty")
    return {
        "target_input_token_ids": target_input_ids,
        "target_hidden_rows": len(target_input_ids),
        "target_hidden_segments": segments,
        "mtp_layer0_input_token_ids": [*target_input_ids[1:], anchor],
    }


def validate_prefill_source(
    *,
    category: str,
    prefill_root: Path,
    generation_report: dict[str, Any],
    prompt_path: Path,
    capture_commit: str,
) -> dict[str, Any]:
    root = prefill_root / category
    report_path = root / "report.json"
    hidden_path = root / "target-layer47-hidden.f32"
    report_sha256 = sha256_file(report_path)
    hidden_sha256 = sha256_file(hidden_path)
    report = json.loads(report_path.read_text())
    prompt_ids = strict_tokens(report["prompt_token_ids"], f"{category}: prefill prompt tokens")
    require(
        report["schema_version"] == 1
        and report["evidence_class"] == PREFILL_EVIDENCE_CLASS
        and report["semantic"] == PREFILL_SEMANTIC
        and report["revision"] == REVISION
        and report["commit"] == capture_commit
        and report["git_dirty"] is False,
        f"{category}: prefill identity",
    )
    require(
        report["model_lock_sha256"] == MODEL_LOCK_SHA256
        and report["checkpoint_verification_sha256"] == CHECKPOINT_RECEIPT_SHA256
        and report["tokenizer_sha256"] == TOKENIZER_SHA256
        and report["tokenizer_config_sha256"] == TOKENIZER_CONFIG_SHA256
        and report["kernel_sha256"] == CAPTURE_KERNEL_SHA256,
        f"{category}: prefill model/kernel authority",
    )
    require(
        report["user_prompt_utf8"] == prompt_path.read_text()
        and report["serialized_prompt_utf8"] == generation_report["serialized_prompt_utf8"]
        and prompt_ids == generation_report["prompt_token_ids"]
        and report["first_anchor_token_id"] == generation_report["generated_token_ids"][0]
        and report["first_anchor_token_id"]
        == generation_report["transactions"][0]["proposal_token_ids"][0]
        and report["prefill_chunks"] == generation_report["prefill_chunks"],
        f"{category}: prefill/generation agreement",
    )
    capture = report["target_hidden"]
    require(
        capture["category"] == category
        and capture["artifact_file"] == hidden_path.name
        and capture["artifact_sha256"] == hidden_sha256
        and capture["shape"] == [len(prompt_ids), HIDDEN]
        and capture["dtype"] == "float32"
        and capture["byte_order"] == "little_endian",
        f"{category}: prefill hidden metadata",
    )
    require(
        capture["semantic"] == PREFILL_HIDDEN_SEMANTIC
        and report["metal_device"] == "Apple M1"
        and report["batch_size"] == 1
        and report["concurrency"] == 1
        and "cold process start" in report["cache_state"],
        f"{category}: prefill execution context",
    )
    require(
        len(report["chunk_layer_traces"]) == report["prefill_chunks"]
        and all(len(chunk) == 48 for chunk in report["chunk_layer_traces"]),
        f"{category}: prefill trace shape",
    )
    finite_f32_payload(hidden_path, len(prompt_ids) * HIDDEN_ROW_BYTES, f"{category}: prefill hidden")
    gate8 = safety_gate(report["safety_snapshots"])
    require(
        report["peak_resident_bytes"] == gate8["maximum_process_peak_resident_bytes"]
        and type(report["process_disk_bytes_read"]) is int
        and report["process_disk_bytes_read"] > 0
        and type(report["ledger"]["logical_source_bytes"]) is int
        and report["ledger"]["logical_source_bytes"] > 0
        and report["complete_wall_ms"] >= report["prefill_wall_ms"] > 0,
        f"{category}: prefill ledger/timing",
    )
    return {
        "category": category,
        "capture_commit": report["commit"],
        "report_file": str(report_path),
        "report_sha256": report_sha256,
        "hidden_file": str(hidden_path),
        "hidden_sha256": hidden_sha256,
        "hidden_rows": len(prompt_ids),
        "first_anchor_token_id": report["first_anchor_token_id"],
        "logical_source_bytes": report["ledger"]["logical_source_bytes"],
        "process_disk_bytes_read": report["process_disk_bytes_read"],
        "complete_wall_ms": report["complete_wall_ms"],
        "peak_resident_bytes": report["peak_resident_bytes"],
        "gate8": gate8,
    }


def validate_generation_source(
    *,
    category: str,
    evidence_root: Path,
    repo: Path,
    capture_commit: str,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    root = evidence_root / category
    report_path = root / "report.json"
    progress_path = root / "report.progress.jsonl"
    hidden_path = root / "verification-layer47-hidden.f32"
    prompt_relative, prompt_sha256 = PROMPTS[category]
    prompt_path = repo / prompt_relative
    require(sha256_file(prompt_path) == prompt_sha256, f"{category}: prompt hash")
    report = json.loads(report_path.read_text())
    require(
        report["schema_version"] == 6
        and report["evidence_class"] == CAPTURE_EVIDENCE_CLASS
        and report["semantic"] == CAPTURE_SEMANTIC
        and report["revision"] == REVISION
        and report["commit"] == capture_commit
        and report["git_dirty"] is False,
        f"{category}: capture identity",
    )
    require(
        report["model_lock_sha256"] == MODEL_LOCK_SHA256
        and report["checkpoint_verification_sha256"] == CHECKPOINT_RECEIPT_SHA256
        and report["tokenizer_sha256"] == TOKENIZER_SHA256
        and report["tokenizer_config_sha256"] == TOKENIZER_CONFIG_SHA256
        and report["kernel_sha256"] == CAPTURE_KERNEL_SHA256,
        f"{category}: capture model/kernel authority",
    )
    require(
        report["metal_device"] == "Apple M1"
        and report["batch_size"] == 1
        and report["concurrency"] == 1
        and report["verifier_width"] == WIDTH
        and report["proposer"] == TARGET_SELF_PROPOSER
        and "cold process start" in report["cache_state"],
        f"{category}: execution context",
    )
    require(
        report["user_prompt_utf8"] == prompt_path.read_text()
        and report["requested_output_tokens"] == 64
        and report["minimum_output_tokens"] == 32
        and report["accepted_tokens"] == 64
        and len(report["generated_token_ids"]) == 64
        and report["stop_reason"] == "requested_maximum"
        and report.get("route_trace_captured") is True,
        f"{category}: fixed-64 capture contract",
    )
    generated_token_ids = strict_tokens(
        report["generated_token_ids"], f"{category}: generated tokens"
    )
    require(sha256_file(progress_path) == report["progress_sha256"], f"{category}: progress hash")
    progress = [json.loads(line) for line in progress_path.read_text().splitlines() if line.strip()]
    transactions = report["transactions"]
    require(
        len(transactions) >= PRIMARY_WINDOWS
        and len(progress) == len(transactions) + 1
        and progress[0]["phase"] == "prefill_complete"
        and progress[0]["prompt_tokens"] == len(report["prompt_token_ids"])
        and progress[0]["prefill_chunks"] == report["prefill_chunks"]
        and math.isclose(
            progress[0]["prefill_wall_ms"], report["prefill_wall_ms"], abs_tol=1.0e-9
        )
        and progress[0]["first_anchor_token_id"] == generated_token_ids[0],
        f"{category}: progress structure",
    )
    capture = report["native_mtp_window"]
    require(
        capture["category"] == category
        and capture["artifact_file"] == hidden_path.name
        and capture["artifact_sha256"] == sha256_file(hidden_path)
        and capture["windows"] == len(transactions)
        and capture["shape"] == [len(transactions), WIDTH, HIDDEN]
        and capture["dtype"] == "float32"
        and capture["byte_order"] == "little_endian"
        and capture["semantic"] == VERIFIER_HIDDEN_SEMANTIC,
        f"{category}: verifier hidden metadata",
    )
    finite_f32_payload(
        hidden_path,
        len(transactions) * WINDOW_BYTES,
        f"{category}: verifier hidden",
    )

    reconstructed = [generated_token_ids[0]]
    windows = []
    for index, transaction in enumerate(transactions):
        progress_row = progress[index + 1]
        semantic = transaction_semantics(
            transaction, index=index, terminal=index + 1 == len(transactions)
        )
        require(semantic["proposal_token_ids"][0] == reconstructed[-1], f"{category}: anchor chain")
        reconstructed.extend(semantic["observable_emitted_token_ids"])
        require(
            progress_row["phase"] == "transaction_complete"
            and progress_row["transaction"] == index
            and progress_row["generated_tokens"] == len(reconstructed)
            and progress_row["emitted_tokens"] == semantic["observable_A"]
            and progress_row["retained_proposal_rows"] == semantic["retained_proposal_rows"]
            and progress_row["proposal_converged"] is semantic["proposal_converged"],
            f"{category}: transaction progress closure",
        )
        validate_proposal_traces(transaction["proposal_layer_traces"])
        route = route_metrics(transaction["verification_layer_traces"])
        require(
            math.isclose(transaction["U"], route["U"], abs_tol=1.0e-12)
            and math.isclose(progress_row["U"], route["U"], abs_tol=1.0e-12),
            f"{category}: route U closure",
        )
        ledgers = validate_byte_ledgers(transaction, report, progress_row, category=category)
        require(
            transaction["proposal_wall_ms"] > 0
            and transaction["verification_wall_ms"] > 0
            and math.isclose(
                transaction["proposal_wall_ms"], progress_row["proposal_wall_ms"], abs_tol=1.0e-9
            )
            and math.isclose(
                transaction["verification_wall_ms"],
                progress_row["verification_wall_ms"],
                abs_tol=1.0e-9,
            ),
            f"{category}: transaction timing closure",
        )
        if index < PRIMARY_WINDOWS:
            windows.append(
                {
                    "transaction_index": index,
                    **semantic,
                    "U": route["U"],
                    "A_per_U": semantic["A"] / route["U"],
                    "route": route,
                    **ledgers,
                    "proposal_wall_ms": transaction["proposal_wall_ms"],
                    "verification_wall_ms": transaction["verification_wall_ms"],
                }
            )
    require(reconstructed == generated_token_ids, f"{category}: generated reconstruction")
    selected_primary_transactions(report)
    validate_aggregate_byte_ledgers(transactions, report, category=category)
    require(
        math.isclose(
            sum(transaction["proposal_wall_ms"] for transaction in transactions),
            report["proposal_wall_ms"],
            abs_tol=0.01,
        )
        and math.isclose(
            sum(transaction["verification_wall_ms"] for transaction in transactions),
            report["verification_wall_ms"],
            abs_tol=0.01,
        )
        and report["complete_wall_ms"]
        >= sum(
            report[name]
            for name in (
                "preprocessing_wall_ms",
                "prefill_wall_ms",
                "proposal_wall_ms",
                "verification_wall_ms",
            )
        ),
        f"{category}: report timing closure",
    )
    gate8 = safety_gate(report["safety_snapshots"])
    require(
        report["peak_resident_bytes"] == gate8["maximum_process_peak_resident_bytes"],
        f"{category}: peak resident closure",
    )
    source = {
        "category": category,
        "capture_commit": capture_commit,
        "prompt_file": prompt_relative,
        "prompt_sha256": prompt_sha256,
        "report_file": str(report_path),
        "report_sha256": sha256_file(report_path),
        "progress_file": str(progress_path),
        "progress_sha256": sha256_file(progress_path),
        "hidden_file": str(hidden_path),
        "hidden_sha256": sha256_file(hidden_path),
        "captured_windows": len(transactions),
        "complete_wall_ms": report["complete_wall_ms"],
        "logical_source_bytes": report["logical_source_bytes"],
        "process_disk_bytes_read": report["process_disk_bytes_read"],
        "peak_resident_bytes": report["peak_resident_bytes"],
        "gate8": gate8,
    }
    return report, source, windows


def build_manifest(
    *,
    evidence_root: Path,
    prefill_root: Path,
    repo: Path,
    capture_commit: str,
) -> dict[str, Any]:
    verify_clean_commit(repo.resolve(), capture_commit)
    safety = HostSafetyMonitor()
    sources = []
    prefill_sources = []
    primary_windows = []
    for category in CATEGORIES:
        report, source, windows = validate_generation_source(
            category=category,
            evidence_root=evidence_root,
            repo=repo,
            capture_commit=capture_commit,
        )
        prompt_path = repo / PROMPTS[category][0]
        prefill = validate_prefill_source(
            category=category,
            prefill_root=prefill_root,
            generation_report=report,
            prompt_path=prompt_path,
            capture_commit=capture_commit,
        )
        sources.append(source)
        prefill_sources.append(prefill)
        verifier_hidden = source["hidden_file"]
        for window in windows:
            index = window["transaction_index"]
            primary_windows.append(
                {
                    "corpus_index": len(primary_windows),
                    "category": category,
                    **target_hidden_binding(
                        report, index, prefill["hidden_file"], verifier_hidden
                    ),
                    **mtp_history_binding(
                        report, index, prefill["hidden_file"], verifier_hidden
                    ),
                    **window,
                }
            )
    require(len(primary_windows) == len(CATEGORIES) * PRIMARY_WINDOWS, "primary corpus cardinality")
    safety.checkpoint("corpus_sources_authenticated")
    category_metrics = {}
    for category in CATEGORIES:
        windows = [row for row in primary_windows if row["category"] == category]
        sum_a = sum(row["A"] for row in windows)
        sum_u = math.fsum(row["U"] for row in windows)
        category_metrics[category] = {
            "windows": len(windows),
            "sum_A": sum_a,
            "sum_observable_A": sum(row["observable_A"] for row in windows),
            "sum_U": sum_u,
            "sum_A_per_sum_U": sum_a / sum_u,
            "unique_identities": len(
                {
                    (identity["layer"], identity["expert"])
                    for row in windows
                    for identity in row["route"]["identities"]
                }
            ),
        }
    total_a = sum(row["A"] for row in primary_windows)
    total_u = math.fsum(row["U"] for row in primary_windows)
    control_identity_counts: Counter[tuple[int, int]] = Counter()
    rare_identity_counts: Counter[tuple[int, int]] = Counter()
    for row in primary_windows:
        destination = (
            rare_identity_counts if row["category"] == "rare_route" else control_identity_counts
        )
        destination.update(
            (identity["layer"], identity["expert"])
            for identity in row["route"]["identities"]
        )
    rare_novel = sorted(set(rare_identity_counts) - set(control_identity_counts))
    safety.release_checkpoint(
        "manifest_inputs_released",
        ["four corrected q8 reports", "four verifier hidden payloads", "four prefill payloads"],
    )
    safety.checkpoint("final_service_health")
    return {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "evidence_class": "pw0328_target_bonus_balanced_q8_causal_corpus",
        "semantic": (
            "first_eight_chronological_target_bonus_q8_windows_per_category_"
            "with_transaction_zero_prefill_and_complete_segmented_target_history"
        ),
        "status": "complete",
        "builder_commit": capture_commit,
        "builder_git_dirty": False,
        "verifier_window_shape": [WIDTH, HIDDEN],
        "hidden_dtype": "float32_little_endian",
        "source_expert_bytes": SOURCE_EXPERT_BYTES,
        "sources": sources,
        "prefill_sources": prefill_sources,
        "primary_windows": primary_windows,
        "control": {
            "windows": len(primary_windows),
            "sum_A": total_a,
            "sum_observable_A": sum(row["observable_A"] for row in primary_windows),
            "sum_U": total_u,
            "sum_A_per_sum_U": total_a / total_u,
            "category_metrics": category_metrics,
        },
        "rare_route_evidence": {
            "control_categories": [category for category in CATEGORIES if category != "rare_route"],
            "control_unique_identities": len(control_identity_counts),
            "rare_route_unique_identities": len(rare_identity_counts),
            "novel_identities": [
                {"layer": layer, "expert": expert} for layer, expert in rare_novel
            ],
            "novel_identity_count": len(rare_novel),
            "novel_routed_layers": sorted({layer for layer, _expert in rare_novel}),
        },
        "builder_safety_snapshots": safety.evidence(),
        "batch_size": 1,
        "concurrency": 1,
        "accepted_tokens": 0,
        "performance_claim": None,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence-root", required=True, type=Path)
    parser.add_argument("--prefill-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--capture-commit", required=True)
    args = parser.parse_args()
    try:
        if args.output.exists():
            raise FileExistsError(args.output)
        manifest = build_manifest(
            evidence_root=args.evidence_root,
            prefill_root=args.prefill_root,
            repo=args.repo,
            capture_commit=args.capture_commit,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_new(args.output, canonical_json(manifest))
        print(
            json.dumps(
                {
                    "status": "complete",
                    "output": str(args.output),
                    "sha256": sha256_file(args.output),
                    "windows": manifest["control"]["windows"],
                    "sum_A_per_sum_U": manifest["control"]["sum_A_per_sum_U"],
                }
            )
        )
        return 0
    except (
        FileExistsError,
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
