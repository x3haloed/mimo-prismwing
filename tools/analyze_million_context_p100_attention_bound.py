#!/usr/bin/env python3
"""Run PW-0158's authenticated one-million-context two-P100 ceiling."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import time

try:
    from tools.analyze_owned_epyc_companion_envelope import (
        authenticate_implementation_commit,
    )
    from tools.analyze_pw0116_corpus import sha256_file
    from tools.host_safety import HostSafetyMonitor
    from tools.openrouter_reference import atomic_write_new, canonical_json
except ModuleNotFoundError:
    from analyze_owned_epyc_companion_envelope import authenticate_implementation_commit
    from analyze_pw0116_corpus import sha256_file
    from host_safety import HostSafetyMonitor
    from openrouter_reference import atomic_write_new, canonical_json


REVISION = "63651580ca774f8504f676040460aed3e1244ac1"
TARGET_SHA256 = "91fe6e0441bb0a0e1ab0852db60fb575d131b61ff069002c9c333f9b776e4950"
CONFIG_SHA256 = "292a60e74ae9a6d53422b31b21468ce2111c0ab3f7f7a4f4e9c7cd5133b96587"
PW0151_SHA256 = "d6919e47f0f4495ccac2ad56ebcfe6662b3309aebd3296c6b546a50836829cb1"
P100_BRIEF_SHA256 = "bda27f98b088ab9ff54e374048e18093374c510c781efcad1e9325b301df4662"
POSITIONS = 1_000_000
TTFT_LIMIT_SECONDS = 1_800
P100_COUNT = 2
P100_FP16_FLOPS = 18.7e12
P100_FP32_FLOPS = 9.3e12
AGGREGATE_HBM_BYTES = 32_000_000_000
NON_ROUTED_SOURCE_BYTES = 12_814_555_472
THREE_ARENA_BYTES = 2_340_993_024
EXPERT_BYTES = 25_171_968


def attention_ledger(config: dict, positions: int = POSITIONS) -> dict:
    if positions <= 0:
        raise ValueError("positions must be positive")
    pattern = config.get("hybrid_layer_pattern")
    if not isinstance(pattern, list) or len(pattern) != 48 or set(pattern) != {0, 1}:
        raise ValueError("hybrid attention pattern mismatch")
    global_layers = pattern.count(0)
    sliding_layers = pattern.count(1)
    expected_global_indices = [0, 5, 11, 17, 23, 29, 35, 41, 47]
    if [index for index, kind in enumerate(pattern) if kind == 0] != expected_global_indices:
        raise ValueError("global attention schedule mismatch")
    query_heads = config.get("num_attention_heads")
    qk_dimension = config.get("head_dim")
    value_dimension = config.get("v_head_dim")
    window = config.get("sliding_window_size")
    if (query_heads, qk_dimension, value_dimension, window) != (64, 192, 128, 128):
        raise ValueError("attention geometry mismatch")
    flops_per_pair_per_query_head = 2 * (qk_dimension + value_dimension)
    causal_pairs_per_global_layer = positions * (positions + 1) // 2
    if positions < window:
        causal_pairs_per_sliding_layer = positions * (positions + 1) // 2
    else:
        causal_pairs_per_sliding_layer = (
            (window - 1) * window // 2 + window * (positions - window + 1)
        )
    global_flops = (
        global_layers
        * query_heads
        * causal_pairs_per_global_layer
        * flops_per_pair_per_query_head
    )
    sliding_flops = (
        sliding_layers
        * query_heads
        * causal_pairs_per_sliding_layer
        * flops_per_pair_per_query_head
    )
    return {
        "positions": positions,
        "global_attention_layers": global_layers,
        "sliding_window_layers": sliding_layers,
        "query_heads": query_heads,
        "qk_dimension": qk_dimension,
        "value_dimension": value_dimension,
        "sliding_window": window,
        "flops_per_pair_per_query_head": flops_per_pair_per_query_head,
        "causal_pairs_per_global_layer": causal_pairs_per_global_layer,
        "causal_pairs_per_sliding_layer": causal_pairs_per_sliding_layer,
        "global_attention_flops": global_flops,
        "sliding_window_attention_flops": sliding_flops,
        "mandatory_attention_flops": global_flops + sliding_flops,
    }


def kv_ledger(config: dict, positions: int = POSITIONS) -> dict:
    pattern = config.get("hybrid_layer_pattern")
    if not isinstance(pattern, list) or len(pattern) != 48:
        raise ValueError("hybrid attention pattern mismatch")
    if (
        config.get("num_key_value_heads"),
        config.get("head_dim"),
        config.get("v_head_dim"),
        config.get("swa_num_key_value_heads"),
        config.get("swa_head_dim"),
        config.get("swa_v_head_dim"),
        config.get("sliding_window_size"),
    ) != (4, 192, 128, 8, 192, 128, 128):
        raise ValueError("KV geometry mismatch")
    global_bytes = pattern.count(0) * positions * 4 * (192 + 128) * 2
    sliding_bytes = pattern.count(1) * min(positions, 128) * 8 * (192 + 128) * 2
    total = global_bytes + sliding_bytes
    full_reservation = total + THREE_ARENA_BYTES + NON_ROUTED_SOURCE_BYTES
    streamed_available = AGGREGATE_HBM_BYTES - total - THREE_ARENA_BYTES
    slots = max(0, streamed_available // EXPERT_BYTES)
    return {
        "dtype": "BF16",
        "positions": positions,
        "global_attention_kv_bytes": global_bytes,
        "sliding_window_kv_bytes": sliding_bytes,
        "total_kv_bytes": total,
        "aggregate_two_p100_hbm_decimal_bytes": AGGREGATE_HBM_BYTES,
        "three_arena_bytes": THREE_ARENA_BYTES,
        "all_non_routed_source_tensor_bytes": NON_ROUTED_SOURCE_BYTES,
        "full_reservation_bytes": full_reservation,
        "full_reservation_over_hbm_bytes": max(0, full_reservation - AGGREGATE_HBM_BYTES),
        "free_streaming_non_routed_available_expert_bytes": streamed_available,
        "free_streaming_non_routed_complete_expert_slots": slots,
        "free_streaming_non_routed_expert_cache_bytes": slots * EXPERT_BYTES,
        "free_streaming_non_routed_unallocated_tail_bytes": streamed_available - slots * EXPERT_BYTES,
    }


def peak_ceiling(mandatory_flops: int, flop_rate: float, name: str) -> dict:
    if mandatory_flops <= 0 or flop_rate <= 0:
        raise ValueError("peak ceiling inputs must be positive")
    seconds = mandatory_flops / flop_rate
    return {
        "name": name,
        "granted_combined_flops_per_second": flop_rate,
        "attention_only_floor_seconds": seconds,
        "attention_only_floor_minutes": seconds / 60,
        "passes_1800_second_complete_prefill_gate": seconds <= TTFT_LIMIT_SECONDS,
    }


def _authenticate(target: Path, config: Path, pw0151: Path, p100_brief: Path) -> tuple[dict, dict]:
    for path, expected in {
        target: TARGET_SHA256,
        config: CONFIG_SHA256,
        pw0151: PW0151_SHA256,
        p100_brief: P100_BRIEF_SHA256,
    }.items():
        if sha256_file(path) != expected:
            raise ValueError(f"PW-0158 source hash mismatch: {path.name}")
    target_text = target.read_text(errors="strict")
    required_target_fragments = (
        "one 1M-token smoke case",
        "begin generation within 30 minutes",
        "at least 1 accepted TPS",
        "USD $500 total",
    )
    if any(fragment not in target_text for fragment in required_target_fragments):
        raise ValueError("TARGET capability authority mismatch")
    if p100_brief.read_bytes()[:5] != b"%PDF-":
        raise ValueError("P100 product brief is not a PDF")
    parsed_config = json.loads(config.read_text())
    pw0151_report = json.loads(pw0151.read_text())
    if (
        parsed_config.get("model_type") != "mimo_v2"
        or parsed_config.get("dtype") != "bfloat16"
        or parsed_config.get("num_hidden_layers") != 48
        or parsed_config.get("num_experts_per_tok") != 8
        or parsed_config.get("n_routed_experts") != 256
    ):
        raise ValueError("pinned config authority mismatch")
    direct = {
        row.get("name"): row
        for row in pw0151_report.get("direct_fp32_accelerators", [])
        if isinstance(row, dict)
    }
    l3 = {
        row.get("name"): row
        for row in pw0151_report.get("l3_diagnostics", [])
        if isinstance(row, dict)
    }
    if (
        pw0151_report.get("evidence_class") != "pw0151_owned_epyc_companion_envelope"
        or pw0151_report.get("revision") != REVISION
        or direct.get("two_tesla_p100_direct_fp32", {}).get("accelerator_count") != 2
        or direct.get("two_tesla_p100_direct_fp32", {}).get("advertised_fp32_each") != P100_FP32_FLOPS
        or l3.get("one_tesla_p100_fp16_l3", {}).get("advertised_fp32_each") != P100_FP16_FLOPS
    ):
        raise ValueError("PW-0151 P100 authority mismatch")
    return parsed_config, pw0151_report


def run(target: Path, config: Path, pw0151: Path, p100_brief: Path, output: Path, commit: str) -> dict:
    if output.exists():
        raise ValueError(f"refusing to overwrite {output}")
    authenticate_implementation_commit(commit)
    started = time.perf_counter()
    safety = HostSafetyMonitor()
    parsed_config, _ = _authenticate(target, config, pw0151, p100_brief)
    safety.checkpoint("source_evidence_authenticated")
    attention = attention_ledger(parsed_config)
    kv = kv_ledger(parsed_config)
    fp16 = peak_ceiling(attention["mandatory_attention_flops"], P100_COUNT * P100_FP16_FLOPS, "two_p100_advertised_fp16")
    fp32 = peak_ceiling(attention["mandatory_attention_flops"], P100_COUNT * P100_FP32_FLOPS, "two_p100_advertised_fp32_diagnostic")
    required_rate = attention["mandatory_attention_flops"] / TTFT_LIMIT_SECONDS
    safety.checkpoint("attention_and_hbm_bounds_complete")
    safety.release_checkpoint("source_reports_released", ["TARGET", "config", "PW-0151 report", "P100 product brief"])
    safety.checkpoint("final_service_health")
    decision = "reject_two_p100_ordinary_dense_attention_for_required_1m_capability_slice"
    if fp16["passes_1800_second_complete_prefill_gate"]:
        raise ValueError("PW-0158 predeclared rejection did not reproduce")
    report = {
        "schema_version": 1,
        "evidence_class": "pw0158_million_context_two_p100_attention_ceiling",
        "revision": REVISION,
        "commit": commit,
        "source_hashes": {
            "target_sha256": TARGET_SHA256,
            "config_sha256": CONFIG_SHA256,
            "pw0151_analysis_sha256": PW0151_SHA256,
            "p100_product_brief_sha256": P100_BRIEF_SHA256,
        },
        "attention_work_ledger": attention,
        "peak_ceiling": {
            "p100_count": P100_COUNT,
            "advertised_fp16_flops_each": P100_FP16_FLOPS,
            "advertised_fp32_flops_each": P100_FP32_FLOPS,
            "fp16_primary_favorable_bound": fp16,
            "fp32_diagnostic": fp32,
            "required_effective_flops_per_second_for_1800_seconds": required_rate,
            "required_effective_tflops_for_1800_seconds": required_rate / 1e12,
            "factor_over_combined_advertised_fp16_peak": required_rate / (P100_COUNT * P100_FP16_FLOPS),
            "exceeded_wall_budget_seconds": fp16["attention_only_floor_seconds"] - TTFT_LIMIT_SECONDS,
        },
        "exact_bf16_kv_hbm_ledger": kv,
        "decision": decision,
        "reopening_condition": "separately_named_changed_attention_L3_or_L4_branch_or_different_complete_hardware_candidate",
        "limitations": [
            "analytical roofline bound, not measured endpoint performance",
            "advertised FP16 peak is a favorable arithmetic grant and not a fidelity promotion",
            "does not reject sparse, linear, retrieval, recurrent, summarized, or learned changed-attention branches",
            "does not by itself prove the complete Prismwing target impossible",
        ],
        "A": 0,
        "accepted_tokens": 0,
        "performance_claim": None,
        "gates_passed": False,
        "safety": safety.evidence(),
        "complete_wall_ms": (time.perf_counter() - started) * 1000.0,
        "platform": platform.platform(),
    }
    atomic_write_new(output, canonical_json(report))
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--pw0151", type=Path, required=True)
    parser.add_argument("--p100-brief", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--commit", required=True)
    args = parser.parse_args()
    try:
        result = run(args.target, args.config, args.pw0151, args.p100_brief, args.output, args.commit)
        print(json.dumps({"output": str(args.output), "decision": result["decision"]}))
        return 0
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
        print(json.dumps({"error": str(error)}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
