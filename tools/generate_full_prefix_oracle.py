#!/usr/bin/env python3
"""Generate the independent PW-0060 48-layer prefix oracle."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import torch

try:
    from tools.generate_real_layer0_bf16_oracle import (
        PROMPT_IDS, REVISION, VERIFICATION_SHA256, Safety, apply_rope, bf16_linear,
        fp8_linear, rms_norm, write_capture,
    )
    from tools.generate_real_layer1_expert_oracle import (
        NUMERICS, ShardedCheckpoint, expert_linear,
    )
    from tools.openrouter_reference import atomic_write_new, canonical_json
except ModuleNotFoundError:
    from generate_real_layer0_bf16_oracle import (
        PROMPT_IDS, REVISION, VERIFICATION_SHA256, Safety, apply_rope, bf16_linear,
        fp8_linear, rms_norm, write_capture,
    )
    from generate_real_layer1_expert_oracle import NUMERICS, ShardedCheckpoint, expert_linear
    from openrouter_reference import atomic_write_new, canonical_json


PATTERN = [0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1,
           1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1,
           1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0]
PW0092_RUN_001_SHA256 = "18c3ccde4a8645d9ea46d0091f877eebe256ca2c7d82c34e771f5f4114bb5f25"


def load_token_fixture(path: Path) -> tuple[list[int], str]:
    payload = path.read_bytes()
    fixture = json.loads(payload)
    base = fixture.get("base_prompt_token_ids")
    appended = fixture.get("appended_token_id")
    whole = fixture.get("whole_sequence_token_ids")
    if (fixture.get("schema_version") != 1
            or fixture.get("semantic") != "mimo_pw0093_whole_sequence_identity"
            or base != PROMPT_IDS
            or not isinstance(appended, int)
            or appended < 0 or appended >= 152_576
            or whole != [*PROMPT_IDS, appended]
            or fixture.get("source_endpoint_report_sha256") != PW0092_RUN_001_SHA256):
        raise ValueError("whole-sequence token fixture identity mismatch")
    return whole, hashlib.sha256(payload).hexdigest()


def checked_fp8(checkpoint: ShardedCheckpoint, name: str, values: torch.Tensor,
                full_qkv: bool = False) -> torch.Tensor:
    shard = checkpoint.shard(name)
    if checkpoint.shard(name + "_scale_inv") != shard:
        raise ValueError(f"{name}: weight and scale shard disagreement")
    return fp8_linear(shard, name, values, full_qkv)


def checked_bf16(checkpoint: ShardedCheckpoint, name: str,
                 values: torch.Tensor) -> torch.Tensor:
    return bf16_linear(checkpoint.shard(name), name, values)


def embedding(checkpoint: ShardedCheckpoint, token_ids: list[int]) -> torch.Tensor:
    name = "model.embed_tokens.weight"
    path = checkpoint.shard(name)
    from safetensors import safe_open
    with safe_open(path, framework="pt", device="cpu") as source:
        view = source.get_slice(name)
        hidden = torch.cat([view[token:token + 1] for token in token_ids])
    if hidden.dtype != torch.bfloat16 or tuple(hidden.shape) != (len(token_ids), 4096):
        raise ValueError("embedding shape mismatch")
    return hidden


def attention(checkpoint: ShardedCheckpoint, layer: int,
              normalized: torch.Tensor) -> torch.Tensor:
    rows = normalized.shape[0]
    prefix = f"model.layers.{layer}.self_attn"
    swa = PATTERN[layer] == 1
    kv_heads = 8 if swa else 4
    qkv = checked_fp8(checkpoint, f"{prefix}.qkv_proj.weight", normalized,
                      full_qkv=not swa)
    q_size, k_size = 64 * 192, kv_heads * 192
    q = apply_rope(qkv[:, :q_size].reshape(rows, 64, 192), 10_000.0 if swa else 10_000_000.0)
    k = apply_rope(qkv[:, q_size:q_size + k_size].reshape(rows, kv_heads, 192),
                   10_000.0 if swa else 10_000_000.0)
    v = (qkv[:, q_size + k_size:].reshape(rows, kv_heads, 128) * 0.707).to(torch.bfloat16)
    sinks = checkpoint.tensor(f"{prefix}.attention_sink_bias") if swa else None
    core = torch.empty((rows, 64, 128), dtype=torch.bfloat16)
    scale = 1.0 / math.sqrt(192)
    groups = 64 // kv_heads
    for position in range(rows):
        for head in range(64):
            kv_head = head // groups
            scores = (q[position, head] @ k[:position + 1, kv_head].T) * scale
            if sinks is not None:
                scores = torch.cat((scores, sinks[head:head + 1]))
            scores = scores - scores.max()
            probabilities = torch.softmax(scores, dim=-1, dtype=torch.float32).to(torch.bfloat16)
            if sinks is not None:
                probabilities = probabilities[:-1]
            core[position, head] = probabilities @ v[:position + 1, kv_head]
    return checked_bf16(checkpoint, f"{prefix}.o_proj.weight", core.reshape(rows, 8192))


def routed_mlp(checkpoint: ShardedCheckpoint, layer: int,
               values: torch.Tensor) -> tuple[torch.Tensor, list[list[int]], list[list[float]], int]:
    rows = values.shape[0]
    prefix = f"model.layers.{layer}.mlp"
    gate = checkpoint.tensor(f"{prefix}.gate.weight").float()
    if tuple(gate.shape) != (256, 4096):
        raise ValueError(f"layer {layer}: router shape mismatch")
    logits = values.float() @ gate.T
    scores = torch.sigmoid(logits)
    correction = checkpoint.tensor(f"{prefix}.gate.e_score_correction_bias").float()
    selected = torch.topk(scores + correction, 8, dim=-1, sorted=False).indices
    chosen = scores.gather(1, selected)
    weights = chosen / (chosen.sum(dim=-1, keepdim=True) + 1e-20)
    schedule: dict[int, list[tuple[int, float]]] = {}
    for position in range(rows):
        for slot in range(8):
            expert = int(selected[position, slot])
            schedule.setdefault(expert, []).append((position, float(weights[position, slot])))
    output = torch.zeros((rows, 4096), dtype=torch.float32)
    for expert in sorted(schedule):
        placements = schedule[expert]
        gathered = values[[position for position, _ in placements]]
        expert_prefix = f"{prefix}.experts.{expert}"
        expert_gate = expert_linear(checkpoint, f"{expert_prefix}.gate_proj.weight", gathered)
        up = expert_linear(checkpoint, f"{expert_prefix}.up_proj.weight", gathered)
        activated = (torch.nn.functional.silu(expert_gate) * up).to(torch.bfloat16)
        down = expert_linear(checkpoint, f"{expert_prefix}.down_proj.weight", activated)
        for local, (position, weight) in enumerate(placements):
            output[position] += down[local].float() * weight
    return (output.to(torch.bfloat16), selected.tolist(), weights.tolist(), len(schedule))


def generate(checkpoint_root: Path, verification: Path, output: Path,
             token_ids: list[int] | None = None,
             token_fixture_sha256: str | None = None) -> None:
    started = time.monotonic(); torch.set_num_threads(1)
    checkpoint = ShardedCheckpoint(checkpoint_root, verification)
    output.mkdir(parents=True, exist_ok=False); safety = Safety(); captures = {}; traces = []
    tokens = PROMPT_IDS if token_ids is None else token_ids
    if not tokens or any(not isinstance(token, int) or token < 0 or token >= 152_576
                         for token in tokens):
        raise ValueError("full-prefix token identity is invalid")
    hidden = embedding(checkpoint, tokens)
    captures["embedding"] = write_capture(output, "embedding", hidden, safety)
    for layer in range(48):
        prefix = f"model.layers.{layer}"
        normalized = rms_norm(hidden, checkpoint.tensor(f"{prefix}.input_layernorm.weight"))
        projected = attention(checkpoint, layer, normalized)
        post_attention = (hidden + projected).to(torch.bfloat16)
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
        hidden = (post_attention + mlp).to(torch.bfloat16)
        name = f"layer_{layer:02}_final"
        captures[name] = write_capture(output, name, hidden, safety)
        traces.append({"layer": layer, "attention": "sliding_window_128" if PATTERN[layer] else "full",
                       "selected_experts_by_position": selected,
                       "route_weights_by_position": weights,
                       "expert_union": union})
    final_norm = rms_norm(hidden, checkpoint.tensor("model.norm.weight"))
    captures["final_norm"] = write_capture(output, "final_norm", final_norm, safety)
    last_logits = checked_bf16(checkpoint, "lm_head.weight", final_norm[-1:]).float().reshape(-1)
    captures["last_logits"] = write_capture(output, "last_logits", last_logits, safety, "F32")
    manifest = {"schema_version": 1, "semantic": "mimo_full_prefix_layer_final_oracle",
        "revision": REVISION, "checkpoint_verification_sha256": VERIFICATION_SHA256,
        "prompt_token_ids": tokens, "numerics": NUMERICS, "captures": captures,
        "layer_traces": traces, "torch_version": torch.__version__,
        "safety_snapshots": safety.snapshots,
        "wall_ms": (time.monotonic() - started) * 1000.0, "performance_claim": None}
    if token_fixture_sha256 is not None:
        manifest["token_fixture_sha256"] = token_fixture_sha256
    atomic_write_new(output / "manifest.json", canonical_json(manifest))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--verification", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--token-fixture", type=Path)
    args = parser.parse_args()
    tokens, fixture_hash = (load_token_fixture(args.token_fixture)
                            if args.token_fixture else (PROMPT_IDS, None))
    generate(args.checkpoint, args.verification, args.output, tokens, fixture_hash)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
