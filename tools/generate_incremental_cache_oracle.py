#!/usr/bin/env python3
"""Generate an independent PyTorch 27-token prefill plus one cached decode."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import time

import torch

try:
    from tools.generate_full_prefix_oracle import (
        NUMERICS, PATTERN, checked_bf16, checked_fp8, embedding, routed_mlp,
    )
    from tools.generate_real_layer0_bf16_oracle import (
        PROMPT_IDS, REVISION, VERIFICATION_SHA256, Safety, apply_rope, rms_norm, write_capture,
    )
    from tools.generate_real_layer1_expert_oracle import ShardedCheckpoint
    from tools.openrouter_reference import atomic_write_new, canonical_json
except ModuleNotFoundError:
    from generate_full_prefix_oracle import (
        NUMERICS, PATTERN, checked_bf16, checked_fp8, embedding, routed_mlp,
    )
    from generate_real_layer0_bf16_oracle import (
        PROMPT_IDS, REVISION, VERIFICATION_SHA256, Safety, apply_rope, rms_norm, write_capture,
    )
    from generate_real_layer1_expert_oracle import ShardedCheckpoint
    from openrouter_reference import atomic_write_new, canonical_json


APPENDED_TOKEN = 264


@dataclass
class LayerCache:
    keys: torch.Tensor
    values: torch.Tensor


def apply_rope_at(values: torch.Tensor, theta: float, position: int) -> torch.Tensor:
    if values.ndim != 3 or values.shape[0] != 1 or values.shape[2] != 192 or position < 0:
        raise ValueError("incremental RoPE shape or position mismatch")
    inv = 1.0 / (theta ** (torch.arange(0, 64, 2, dtype=torch.float32) / 64))
    frequencies = inv * float(position)
    cosine = torch.cat((frequencies, frequencies)).cos().to(torch.bfloat16)
    sine = torch.cat((frequencies, frequencies)).sin().to(torch.bfloat16)
    result = values.clone()
    rotating = values[0, :, :64]
    half = torch.cat((-rotating[:, 32:], rotating[:, :32]), dim=-1)
    result[0, :, :64] = (rotating * cosine) + (half * sine)
    return result


def visible_start(is_swa: bool, end: int, window: int = 128) -> int:
    if end <= 0 or window <= 0:
        raise ValueError("attention visibility requires positive bounds")
    return max(0, end - window) if is_swa else 0


def validate_cache(cache: LayerCache, positions: int, kv_heads: int) -> None:
    if (cache.keys.dtype != torch.bfloat16 or cache.values.dtype != torch.bfloat16
            or tuple(cache.keys.shape) != (positions, kv_heads, 192)
            or tuple(cache.values.shape) != (positions, kv_heads, 128)
            or not torch.isfinite(cache.keys.float()).all()
            or not torch.isfinite(cache.values.float()).all()):
        raise ValueError("incremental K/V cache authority mismatch")


def attention_prefill(checkpoint: ShardedCheckpoint, layer: int,
                      normalized: torch.Tensor) -> tuple[torch.Tensor, LayerCache]:
    rows = normalized.shape[0]
    if rows != len(PROMPT_IDS):
        raise ValueError("prefill row authority mismatch")
    prefix = f"model.layers.{layer}.self_attn"
    is_swa = PATTERN[layer] == 1
    kv_heads = 8 if is_swa else 4
    qkv = checked_fp8(checkpoint, f"{prefix}.qkv_proj.weight", normalized,
                      full_qkv=not is_swa)
    q_size, k_size = 64 * 192, kv_heads * 192
    q = apply_rope(qkv[:, :q_size].reshape(rows, 64, 192),
                   10_000.0 if is_swa else 10_000_000.0)
    k = apply_rope(qkv[:, q_size:q_size + k_size].reshape(rows, kv_heads, 192),
                   10_000.0 if is_swa else 10_000_000.0)
    v = (qkv[:, q_size + k_size:].reshape(rows, kv_heads, 128) * 0.707).to(torch.bfloat16)
    cache = LayerCache(k, v)
    validate_cache(cache, rows, kv_heads)
    sinks = checkpoint.tensor(f"{prefix}.attention_sink_bias") if is_swa else None
    core = torch.empty((rows, 64, 128), dtype=torch.bfloat16)
    scale = 1.0 / math.sqrt(192)
    groups = 64 // kv_heads
    for position in range(rows):
        end = position + 1
        start = visible_start(is_swa, end)
        for head in range(64):
            kv_head = head // groups
            scores = (q[position, head] @ k[start:end, kv_head].T) * scale
            if sinks is not None:
                scores = torch.cat((scores, sinks[head:head + 1]))
            scores = scores - scores.max()
            probabilities = torch.softmax(scores, dim=-1, dtype=torch.float32).to(torch.bfloat16)
            if sinks is not None:
                probabilities = probabilities[:-1]
            core[position, head] = probabilities @ v[start:end, kv_head]
    return checked_bf16(checkpoint, f"{prefix}.o_proj.weight", core.reshape(rows, 8192)), cache


def attention_incremental(checkpoint: ShardedCheckpoint, layer: int,
                          normalized: torch.Tensor, cache: LayerCache) -> torch.Tensor:
    if tuple(normalized.shape) != (1, 4096):
        raise ValueError("incremental attention requires one hidden row")
    prefix = f"model.layers.{layer}.self_attn"
    is_swa = PATTERN[layer] == 1
    kv_heads = 8 if is_swa else 4
    validate_cache(cache, len(PROMPT_IDS), kv_heads)
    qkv = checked_fp8(checkpoint, f"{prefix}.qkv_proj.weight", normalized,
                      full_qkv=not is_swa)
    q_size, k_size = 64 * 192, kv_heads * 192
    theta = 10_000.0 if is_swa else 10_000_000.0
    q = apply_rope_at(qkv[:, :q_size].reshape(1, 64, 192), theta, len(PROMPT_IDS))
    new_k = apply_rope_at(qkv[:, q_size:q_size + k_size].reshape(1, kv_heads, 192),
                          theta, len(PROMPT_IDS))
    new_v = (qkv[:, q_size + k_size:].reshape(1, kv_heads, 128) * 0.707).to(torch.bfloat16)
    cache.keys = torch.cat((cache.keys, new_k))
    cache.values = torch.cat((cache.values, new_v))
    validate_cache(cache, len(PROMPT_IDS) + 1, kv_heads)
    end = len(PROMPT_IDS) + 1
    start = visible_start(is_swa, end)
    sinks = checkpoint.tensor(f"{prefix}.attention_sink_bias") if is_swa else None
    core = torch.empty((1, 64, 128), dtype=torch.bfloat16)
    scale = 1.0 / math.sqrt(192)
    groups = 64 // kv_heads
    for head in range(64):
        kv_head = head // groups
        scores = (q[0, head] @ cache.keys[start:end, kv_head].T) * scale
        if sinks is not None:
            scores = torch.cat((scores, sinks[head:head + 1]))
        scores = scores - scores.max()
        probabilities = torch.softmax(scores, dim=-1, dtype=torch.float32).to(torch.bfloat16)
        if sinks is not None:
            probabilities = probabilities[:-1]
        core[0, head] = probabilities @ cache.values[start:end, kv_head]
    return checked_bf16(checkpoint, f"{prefix}.o_proj.weight", core.reshape(1, 8192))


def decoder_layer(checkpoint: ShardedCheckpoint, layer: int, hidden: torch.Tensor,
                  attention_output: torch.Tensor) -> tuple[torch.Tensor, list[list[int]],
                                                            list[list[float]], int]:
    prefix = f"model.layers.{layer}"
    post_attention = (hidden + attention_output).to(torch.bfloat16)
    moe_input = rms_norm(post_attention,
                         checkpoint.tensor(f"{prefix}.post_attention_layernorm.weight"))
    if layer == 0:
        gate = checked_fp8(checkpoint, f"{prefix}.mlp.gate_proj.weight", moe_input)
        up = checked_fp8(checkpoint, f"{prefix}.mlp.up_proj.weight", moe_input)
        activated = (torch.nn.functional.silu(gate) * up).to(torch.bfloat16)
        mlp = checked_fp8(checkpoint, f"{prefix}.mlp.down_proj.weight", activated)
        selected, weights, union = [], [], 0
    else:
        mlp, selected, weights, union = routed_mlp(checkpoint, layer, moe_input)
    return (post_attention + mlp).to(torch.bfloat16), selected, weights, union


def generate(checkpoint_root: Path, verification: Path, output: Path) -> None:
    started = time.monotonic()
    torch.set_num_threads(1)
    output.mkdir(parents=True, exist_ok=False)
    safety = Safety()
    checkpoint = ShardedCheckpoint(checkpoint_root, verification)
    safety.check("checkpoint_open")
    caches: list[LayerCache] = []
    prefill_routes = []
    hidden = embedding(checkpoint, PROMPT_IDS)
    for layer in range(48):
        prefix = f"model.layers.{layer}"
        normalized = rms_norm(hidden, checkpoint.tensor(f"{prefix}.input_layernorm.weight"))
        projected, cache = attention_prefill(checkpoint, layer, normalized)
        hidden, selected, weights, union = decoder_layer(checkpoint, layer, hidden, projected)
        caches.append(cache)
        prefill_routes.append({"layer": layer, "selected_experts_by_position": selected,
                               "route_weights_by_position": weights, "expert_union": union})
        safety.check(f"prefill_layer_{layer}_complete")

    hidden = embedding(checkpoint, [APPENDED_TOKEN])
    captures = {}
    incremental_routes = []
    for layer in range(48):
        prefix = f"model.layers.{layer}"
        normalized = rms_norm(hidden, checkpoint.tensor(f"{prefix}.input_layernorm.weight"))
        projected = attention_incremental(checkpoint, layer, normalized, caches[layer])
        hidden, selected, weights, union = decoder_layer(checkpoint, layer, hidden, projected)
        name = f"layer_{layer:02}_incremental_final"
        captures[name] = write_capture(output, name, hidden, safety)
        incremental_routes.append({"layer": layer,
            "selected_experts_by_position": selected,
            "route_weights_by_position": weights, "expert_union": union,
            "cache_positions": caches[layer].keys.shape[0],
            "kv_heads": caches[layer].keys.shape[1]})

    final_norm = rms_norm(hidden, checkpoint.tensor("model.norm.weight"))
    captures["incremental_final_norm"] = write_capture(
        output, "incremental_final_norm", final_norm, safety)
    logits = checked_bf16(checkpoint, "lm_head.weight", final_norm).float().reshape(-1)
    captures["incremental_last_logits"] = write_capture(
        output, "incremental_last_logits", logits, safety, "F32")
    token = int(torch.argmax(logits))
    manifest = {"schema_version": 1, "semantic": "mimo_pytorch_incremental_cache_oracle",
        "revision": REVISION, "checkpoint_verification_sha256": VERIFICATION_SHA256,
        "prefill_token_ids": PROMPT_IDS, "incremental_input_token_id": APPENDED_TOKEN,
        "output_token_id": token, "numerics": NUMERICS, "captures": captures,
        "prefill_layer_traces": prefill_routes, "incremental_layer_traces": incremental_routes,
        "torch_version": torch.__version__, "safety_snapshots": safety.snapshots,
        "wall_ms": (time.monotonic() - started) * 1000.0, "batch_size": 1,
        "concurrency": 1, "accepted_tokens": 0, "performance_claim": None}
    atomic_write_new(output / "manifest.json", canonical_json(manifest))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--verification", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    generate(args.checkpoint, args.verification, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
