#!/usr/bin/env python3
"""Freeze a bounded five-layer cached oracle for PW-0101."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import torch

try:
    from tools.generate_full_prefix_oracle import NUMERICS, checked_fp8, embedding
    from tools.generate_incremental_cache_oracle import (
        APPENDED_TOKEN,
        PROMPT_IDS,
        attention_incremental,
        attention_prefill,
        decoder_layer,
    )
    from tools.generate_real_layer0_bf16_oracle import (
        REVISION,
        VERIFICATION_SHA256,
        Safety,
        rms_norm,
        write_capture,
    )
    from tools.generate_real_layer1_expert_oracle import ShardedCheckpoint
    from tools.openrouter_reference import atomic_write_new, canonical_json
except ModuleNotFoundError:
    from generate_full_prefix_oracle import NUMERICS, checked_fp8, embedding
    from generate_incremental_cache_oracle import (
        APPENDED_TOKEN,
        PROMPT_IDS,
        attention_incremental,
        attention_prefill,
        decoder_layer,
    )
    from generate_real_layer0_bf16_oracle import (
        REVISION,
        VERIFICATION_SHA256,
        Safety,
        rms_norm,
        write_capture,
    )
    from generate_real_layer1_expert_oracle import ShardedCheckpoint
    from openrouter_reference import atomic_write_new, canonical_json


LAST_LAYER = 4


def routed_layer4(checkpoint: ShardedCheckpoint, hidden: torch.Tensor,
                  attention_output: torch.Tensor, output: Path,
                  safety: Safety) -> tuple[torch.Tensor, dict, dict]:
    prefix = "model.layers.4"
    post_attention = (hidden + attention_output).to(torch.bfloat16)
    moe_input = rms_norm(
        post_attention, checkpoint.tensor(f"{prefix}.post_attention_layernorm.weight")
    )
    captures = {
        "post_attention": write_capture(output, "layer4_post_attention", post_attention, safety),
        "moe_input": write_capture(output, "layer4_moe_input", moe_input, safety),
    }
    gate_prefix = f"{prefix}.mlp.gate"
    logits = moe_input.float() @ checkpoint.tensor(f"{gate_prefix}.weight").float().T
    scores = torch.sigmoid(logits)
    correction = checkpoint.tensor(f"{gate_prefix}.e_score_correction_bias").float()
    _, selected = torch.topk(scores + correction, k=8, dim=-1, sorted=False)
    route_weights = scores.gather(1, selected)
    route_weights = route_weights / (route_weights.sum(-1, keepdim=True) + 1e-20)
    routed = torch.zeros((1, 4096), dtype=torch.float32)
    expert_captures: dict[str, dict] = {}
    for slot, expert_value in enumerate(selected[0].tolist()):
        expert = int(expert_value)
        expert_prefix = f"{prefix}.mlp.experts.{expert}"
        gate = checked_fp8(checkpoint, f"{expert_prefix}.gate_proj.weight", moe_input)
        up = checked_fp8(checkpoint, f"{expert_prefix}.up_proj.weight", moe_input)
        swiglu = (torch.nn.functional.silu(gate) * up).to(torch.bfloat16)
        down = checked_fp8(checkpoint, f"{expert_prefix}.down_proj.weight", swiglu)
        routed += down.float() * float(route_weights[0, slot])
        expert_captures[str(expert)] = {
            stage: write_capture(output, f"expert_{expert}_{stage}", value, safety)
            for stage, value in (
                ("gate", gate),
                ("up", up),
                ("swiglu", swiglu),
                ("down", down),
            )
        }
        safety.check(f"layer4_expert_{expert}_complete")
    routed_bf16 = routed.to(torch.bfloat16)
    final = (post_attention + routed_bf16).to(torch.bfloat16)
    captures["routed"] = write_capture(output, "layer4_routed", routed_bf16, safety)
    captures["final"] = write_capture(output, "layer4_final", final, safety)
    routes = {
        "selected_experts": selected[0].tolist(),
        "route_weights": route_weights[0].tolist(),
    }
    return final, captures, {"routes": routes, "experts": expert_captures}


def generate(checkpoint_root: Path, verification: Path, output: Path) -> None:
    started = time.monotonic()
    torch.set_num_threads(1)
    output.mkdir(parents=True, exist_ok=False)
    safety = Safety()
    checkpoint = ShardedCheckpoint(checkpoint_root, verification)
    safety.check("checkpoint_open")
    caches = []
    hidden = embedding(checkpoint, PROMPT_IDS)
    for layer in range(LAST_LAYER + 1):
        prefix = f"model.layers.{layer}"
        normalized = rms_norm(
            hidden, checkpoint.tensor(f"{prefix}.input_layernorm.weight")
        )
        projected, cache = attention_prefill(checkpoint, layer, normalized)
        hidden, _, _, _ = decoder_layer(checkpoint, layer, hidden, projected)
        caches.append(cache)
        safety.check(f"prefill_layer_{layer}_complete")

    cache_captures = {}
    for layer, cache in enumerate(caches):
        cache_captures[str(layer)] = {
            "keys": write_capture(output, f"layer_{layer:02}_prefill_keys", cache.keys, safety),
            "values": write_capture(
                output, f"layer_{layer:02}_prefill_values", cache.values, safety
            ),
        }
    hidden = embedding(checkpoint, [APPENDED_TOKEN])
    incremental_finals = {}
    layer4_captures = None
    layer4_authority = None
    for layer in range(LAST_LAYER + 1):
        prefix = f"model.layers.{layer}"
        normalized = rms_norm(
            hidden, checkpoint.tensor(f"{prefix}.input_layernorm.weight")
        )
        projected = attention_incremental(checkpoint, layer, normalized, caches[layer])
        if layer == LAST_LAYER:
            hidden, layer4_captures, layer4_authority = routed_layer4(
                checkpoint, hidden, projected, output, safety
            )
        else:
            hidden, _, _, _ = decoder_layer(checkpoint, layer, hidden, projected)
        incremental_finals[str(layer)] = write_capture(
            output, f"layer_{layer:02}_incremental_final", hidden, safety
        )
        safety.check(f"incremental_layer_{layer}_complete")
    if layer4_captures is None or layer4_authority is None:
        raise RuntimeError("layer-4 authority was not produced")
    manifest = {
        "schema_version": 1,
        "semantic": "mimo_pytorch_layer4_partial_cached_oracle",
        "revision": REVISION,
        "checkpoint_verification_sha256": VERIFICATION_SHA256,
        "prefill_token_ids": PROMPT_IDS,
        "incremental_input_token_id": APPENDED_TOKEN,
        "last_layer": LAST_LAYER,
        "numerics": NUMERICS,
        "cache_captures": cache_captures,
        "incremental_finals": incremental_finals,
        "layer4_captures": layer4_captures,
        "layer4_routes": layer4_authority["routes"],
        "layer4_expert_captures": layer4_authority["experts"],
        "torch_version": torch.__version__,
        "safety_snapshots": safety.snapshots,
        "wall_ms": (time.monotonic() - started) * 1000.0,
        "batch_size": 1,
        "concurrency": 1,
        "accepted_tokens": 0,
        "performance_claim": None,
    }
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
