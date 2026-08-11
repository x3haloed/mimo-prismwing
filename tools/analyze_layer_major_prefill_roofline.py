#!/usr/bin/env python3
"""Build PW-0209's M1 layer-major prefill memory and roofline bound."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

try:
    from tools.openrouter_reference import atomic_write_new, canonical_json
except ModuleNotFoundError:
    from openrouter_reference import atomic_write_new, canonical_json


GIB = 1024**3
F32_BYTES = 4
HIDDEN = 4096
Q_ROWS = 64 * 192
FULL_QKV_ROWS = Q_ROWS + 4 * 192 + 4 * 128
LAYERS = 48
FULL_ATTENTION_LAYERS = 9
SWA_LAYERS = 39
FULL_KV_HEADS = 4
SWA_KV_HEADS = 8
QK_HEAD_DIM = 192
V_HEAD_DIM = 128
SOURCE_EXPERT_BYTES = 25_171_968
SHARED_SCANNED_BYTES = 7_743_236_992
PROJECTION_GIB_PER_SECOND = 37.76477121733443
UNCACHED_LAYER_WIDENED_BYTES = 202_162_176
UNCACHED_LAYER_WALL_MS = 72.191416
BASELINE_PROMPT_TOKENS = 87
BASELINE_PREFILL_MS = 205_294.176
ARENA_LIMIT_BYTES = 12 * GIB
REQUIRED_SPEEDUP = 4.0
TTFT_TARGET_MS = 15_000.0

# PW-0157 exact observations. The 128-token case conservatively uses the
# observed 512-prefix count because a causal prefix cannot lose distinct
# (layer, expert) records.
PREFIX_EXPERT_BOUNDS = {
    128: {"records": 2_980, "kind": "upper_bound_from_exact_prefix_512"},
    1_024: {"records": 3_572, "kind": "exact_prefix_1024"},
    8_000: {"records": 4_903, "kind": "exact_prefix_8000"},
}


def kv_cache_bytes(tokens: int) -> int:
    per_head = (QK_HEAD_DIM + V_HEAD_DIM) * F32_BYTES
    return tokens * per_head * (
        FULL_ATTENTION_LAYERS * FULL_KV_HEADS + SWA_LAYERS * SWA_KV_HEADS
    )


def transient_arena_bytes(tokens: int) -> dict[str, int]:
    """Conservative mutually-live layer-major F32 arenas.

    The expert scratch assumes one routed expert is consumed at a time. It
    intentionally keeps input, gate, up, hidden, codes/scales, expert output,
    and scatter output live together instead of relying on allocator reuse.
    """
    hidden_row = tokens * HIDDEN * F32_BYTES
    intermediate_row = tokens * 2048 * F32_BYTES
    expert_scratch = (
        2 * hidden_row
        + 3 * intermediate_row
        + tokens * HIDDEN
        + tokens * 2048
        + tokens * (HIDDEN // 128 + 2048 // 128) * F32_BYTES
    )
    return {
        "hidden_ping_pong": 2 * hidden_row,
        "normalized_hidden": hidden_row,
        "maximum_qkv": tokens * FULL_QKV_ROWS * F32_BYTES,
        "rope_queries": tokens * Q_ROWS * F32_BYTES,
        "attention_output": tokens * 64 * V_HEAD_DIM * F32_BYTES,
        "post_attention_normalized": hidden_row,
        "router_scores_weights_and_indices": tokens * 256 * 12,
        "routed_output": hidden_row,
        "single_expert_scratch_conservative": expert_scratch,
    }


def model_length(tokens: int, records: int, bound_kind: str) -> dict:
    transient = transient_arena_bytes(tokens)
    kv_bytes = kv_cache_bytes(tokens)
    total_arena = kv_bytes + sum(transient.values())
    source_bytes = records * SOURCE_EXPERT_BYTES + SHARED_SCANNED_BYTES
    uncached_bytes_per_second = UNCACHED_LAYER_WIDENED_BYTES / (UNCACHED_LAYER_WALL_MS / 1000.0)
    acquisition_ms = source_bytes / uncached_bytes_per_second * 1000.0
    projection_ms = source_bytes / (PROJECTION_GIB_PER_SECOND * GIB) * 1000.0
    optimistic_roofline_ms = max(acquisition_ms, projection_ms)
    baseline_scaled_ms = BASELINE_PREFILL_MS * tokens / BASELINE_PROMPT_TOKENS
    roofline_headroom_speedup = baseline_scaled_ms / optimistic_roofline_ms
    return {
        "tokens": tokens,
        "distinct_layer_expert_records": records,
        "record_count_authority": bound_kind,
        "kv_cache_bytes": kv_bytes,
        "transient_arenas": transient,
        "total_declared_arena_bytes": total_arena,
        "total_declared_arena_gib": total_arena / GIB,
        "arena_below_12_gib": total_arena < ARENA_LIMIT_BYTES,
        "unique_source_bytes": source_bytes,
        "uncached_acquisition_floor_ms": acquisition_ms,
        "warm_projection_floor_ms": projection_ms,
        "optimistic_overlapped_roofline_ms": optimistic_roofline_ms,
        "scaled_current_width8_prefill_ms": baseline_scaled_ms,
        "roofline_headroom_speedup_vs_scaled_control": roofline_headroom_speedup,
        "roofline_supports_four_x": roofline_headroom_speedup >= REQUIRED_SPEEDUP,
        "source_acquisition_alone_meets_15s": acquisition_ms <= TTFT_TARGET_MS,
    }


def analyze() -> dict:
    lengths = {
        str(tokens): model_length(tokens, authority["records"], authority["kind"])
        for tokens, authority in PREFIX_EXPERT_BOUNDS.items()
    }
    eight_k = lengths["8000"]
    arena_gate = all(row["arena_below_12_gib"] for row in lengths.values())
    roofline_gate = eight_k["roofline_supports_four_x"]
    ttft_storage_gate = eight_k["source_acquisition_alone_meets_15s"]
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, text=True, capture_output=True
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"], check=True, text=True, capture_output=True
        ).stdout
    )
    return {
        "schema_version": 1,
        "evidence_class": "pw0209_layer_major_prefill_memory_and_roofline_bound",
        "implementation_commit": head,
        "git_dirty": dirty,
        "inputs": {
            "baseline_prompt_tokens": BASELINE_PROMPT_TOKENS,
            "baseline_prefill_ms": BASELINE_PREFILL_MS,
            "projection_gib_per_second": PROJECTION_GIB_PER_SECOND,
            "uncached_stream_bytes_per_second": UNCACHED_LAYER_WIDENED_BYTES
            / (UNCACHED_LAYER_WALL_MS / 1000.0),
            "source_expert_bytes": SOURCE_EXPERT_BYTES,
            "shared_scanned_bytes": SHARED_SCANNED_BYTES,
            "arena_limit_bytes": ARENA_LIMIT_BYTES,
            "required_speedup": REQUIRED_SPEEDUP,
            "ttft_target_ms": TTFT_TARGET_MS,
        },
        "lengths": lengths,
        "gates": {
            "arena_gate_passed": arena_gate,
            "four_x_roofline_gate_passed": roofline_gate,
            "fifteen_second_source_acquisition_gate_passed": ttft_storage_gate,
            "authorize_width128_layer_slice": arena_gate and roofline_gate,
        },
        "decision": (
            "authorize_width128_layer_slice; preserve_layer_major_speedup_path; "
            "source_fp8_internal_ssd_cannot_meet_15s_8k_ttft_without_a_changed_storage_premise"
        ),
        "limitations": (
            "necessary optimistic roofline, not candidate wall or endpoint TTFT; "
            "8K attention compute is not timed; 128-token expert count is a conservative "
            "upper bound from the exact 512-prefix observation"
        ),
        "accepted_tokens": 0,
        "performance_claim": None,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    try:
        result = analyze()
        if result["git_dirty"]:
            raise ValueError("PW-0209 evidence requires a clean Git worktree")
        atomic_write_new(arguments.output, canonical_json(result))
        print(json.dumps({"output": str(arguments.output), "decision": result["decision"]}))
        return 0
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        print(json.dumps({"error": str(error)}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
