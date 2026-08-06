#!/usr/bin/env python3
"""Generate an independent complete routed-layer trace."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

try:
    from tools.generate_full_prefix_oracle import checked_bf16, checked_fp8
    from tools.generate_real_layer0_bf16_oracle import (
        PROMPT_IDS, REVISION, VERIFICATION_SHA256, Safety, apply_rope, rms_norm, write_capture,
    )
    from tools.generate_real_layer1_expert_oracle import NUMERICS, ShardedCheckpoint, expert_linear
    from tools.openrouter_reference import atomic_write_new, canonical_json
except ModuleNotFoundError:
    from generate_full_prefix_oracle import checked_bf16, checked_fp8
    from generate_real_layer0_bf16_oracle import (
        PROMPT_IDS, REVISION, VERIFICATION_SHA256, Safety, apply_rope, rms_norm, write_capture,
    )
    from generate_real_layer1_expert_oracle import NUMERICS, ShardedCheckpoint, expert_linear
    from openrouter_reference import atomic_write_new, canonical_json


def sha256(path: Path) -> str:
    import hashlib
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_input(manifest_path: Path, target_layer: int) -> tuple[torch.Tensor, str]:
    manifest = json.loads(manifest_path.read_text())
    if (manifest.get("semantic") != "mimo_full_prefix_layer_final_oracle"
            or manifest.get("revision") != REVISION
            or manifest.get("checkpoint_verification_sha256") != VERIFICATION_SHA256
            or manifest.get("prompt_token_ids") != PROMPT_IDS or manifest.get("numerics") != NUMERICS):
        raise ValueError("PW-0060 oracle authority mismatch")
    record = manifest["captures"][f"layer_{target_layer - 1:02}_final"]
    path = manifest_path.parent / record["file"]
    if record.get("shape") != [27, 4096] or record.get("dtype") != "BF16_widened_F32" or sha256(path) != record.get("sha256"):
        raise ValueError(f"PW-0060 layer-{target_layer - 1} capture mismatch")
    values = np.fromfile(path, dtype="<f4").reshape(27, 4096)
    return torch.from_numpy(values.copy()).to(torch.bfloat16), record["sha256"]


def generate(checkpoint_root: Path, verification: Path, source_manifest: Path,
             output: Path, target_layer: int = 2) -> None:
    if target_layer not in (2, 4, 7, 11):
        raise ValueError(f"unsupported routed trace layer {target_layer}")
    started = time.monotonic(); torch.set_num_threads(1)
    hidden, source_hash = load_input(source_manifest, target_layer)
    checkpoint = ShardedCheckpoint(checkpoint_root, verification)
    output.mkdir(parents=True, exist_ok=False); safety = Safety(); captures = {}
    captures["incoming"] = write_capture(output, "incoming", hidden, safety)
    prefix = f"model.layers.{target_layer}"; attention_prefix = f"{prefix}.self_attn"
    normalized = rms_norm(hidden, checkpoint.tensor(f"{prefix}.input_layernorm.weight"))
    captures["input_norm"] = write_capture(output, "input_norm", normalized, safety)
    qkv = checked_fp8(checkpoint, f"{attention_prefix}.qkv_proj.weight", normalized)
    captures["qkv"] = write_capture(output, "qkv", qkv, safety)
    q = apply_rope(qkv[:, :12288].reshape(27, 64, 192), 10_000.0)
    k = apply_rope(qkv[:, 12288:13824].reshape(27, 8, 192), 10_000.0)
    v = (qkv[:, 13824:].reshape(27, 8, 128) * 0.707).to(torch.bfloat16)
    sinks = checkpoint.tensor(f"{attention_prefix}.attention_sink_bias")
    for name, value in (("query", q), ("key", k), ("value", v), ("sinks", sinks)):
        captures[name] = write_capture(output, name, value, safety)
    core = torch.empty((27, 64, 128), dtype=torch.bfloat16); score_rows = []; probability_rows = []
    for position in range(27):
        for head in range(64):
            kv_head = head // 8
            scores = (q[position, head] @ k[:position + 1, kv_head].T) / math.sqrt(192)
            scores = torch.cat((scores, sinks[head:head + 1])); scores = scores - scores.max()
            probabilities = torch.softmax(scores, dim=-1, dtype=torch.float32).to(torch.bfloat16)
            score_rows.append(scores); probability_rows.append(probabilities)
            core[position, head] = probabilities[:-1] @ v[:position + 1, kv_head]
    captures["attention_scores"] = write_capture(output, "attention_scores", torch.cat(score_rows), safety)
    captures["attention_probabilities"] = write_capture(output, "attention_probabilities", torch.cat(probability_rows), safety)
    captures["attention"] = write_capture(output, "attention", core, safety)
    projected = checked_bf16(checkpoint, f"{attention_prefix}.o_proj.weight", core.reshape(27, 8192))
    captures["attention_projection"] = write_capture(output, "attention_projection", projected, safety)
    post_attention = (hidden + projected).to(torch.bfloat16)
    captures["post_attention"] = write_capture(output, "post_attention", post_attention, safety)
    moe_input = rms_norm(post_attention, checkpoint.tensor(f"{prefix}.post_attention_layernorm.weight"))
    captures["moe_input"] = write_capture(output, "moe_input", moe_input, safety)
    router = checkpoint.tensor(f"{prefix}.mlp.gate.weight").float()
    logits = moe_input.float() @ router.T; scores = torch.sigmoid(logits)
    correction = checkpoint.tensor(f"{prefix}.mlp.gate.e_score_correction_bias").float()
    selected = torch.topk(scores + correction, 8, dim=-1, sorted=False).indices
    chosen = scores.gather(1, selected); weights = chosen / (chosen.sum(dim=-1, keepdim=True) + 1e-20)
    captures["router_logits"] = write_capture(output, "router_logits", logits, safety, "F32")
    captures["router_scores"] = write_capture(output, "router_scores", scores, safety, "F32")
    schedule: dict[int, list[tuple[int, float]]] = {}
    for position in range(27):
        for slot in range(8):
            schedule.setdefault(int(selected[position, slot]), []).append((position, float(weights[position, slot])))
    gates = []; ups = []; swiglus = []; downs = []; routed = torch.zeros((27, 4096), dtype=torch.float32)
    schedule_manifest = []
    for expert in sorted(schedule):
        placements = schedule[expert]; positions = [position for position, _ in placements]
        gathered = moe_input[positions]; expert_prefix = f"{prefix}.mlp.experts.{expert}"
        gate = expert_linear(checkpoint, f"{expert_prefix}.gate_proj.weight", gathered)
        up = expert_linear(checkpoint, f"{expert_prefix}.up_proj.weight", gathered)
        swiglu = (torch.nn.functional.silu(gate) * up).to(torch.bfloat16)
        down = expert_linear(checkpoint, f"{expert_prefix}.down_proj.weight", swiglu)
        gates.append(gate); ups.append(up); swiglus.append(swiglu); downs.append(down)
        for local, (position, weight) in enumerate(placements): routed[position] += down[local].float() * weight
        schedule_manifest.append({"expert": expert, "positions": positions}); safety.check(f"layer_{target_layer}_expert_{expert}_complete")
    routed = routed.to(torch.bfloat16); final = (post_attention + routed).to(torch.bfloat16)
    for name, value in (("expert_gate", torch.cat(gates)), ("expert_up", torch.cat(ups)),
                        ("expert_swiglu", torch.cat(swiglus)), ("expert_down", torch.cat(downs)),
                        ("routed_output", routed), ("final", final)):
        captures[name] = write_capture(output, name, value, safety)
    manifest = {"schema_version": 1, "semantic": f"mimo_real_layer{target_layer}_complete_oracle",
        "revision": REVISION, "checkpoint_verification_sha256": VERIFICATION_SHA256,
        "prompt_token_ids": PROMPT_IDS, "source_input_sha256": source_hash, "numerics": NUMERICS,
        "captures": captures, "selected_experts_by_position": selected.tolist(),
        "route_weights_by_position": weights.tolist(), "expert_schedule": schedule_manifest,
        "safety_snapshots": safety.snapshots, "wall_ms": (time.monotonic() - started) * 1000.0,
        "performance_claim": None}
    atomic_write_new(output / "manifest.json", canonical_json(manifest))


def main() -> int:
    p = argparse.ArgumentParser(); p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--verification", required=True, type=Path); p.add_argument("--source-manifest", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--layer", choices=(2, 4, 7, 11), default=2, type=int); a = p.parse_args()
    generate(a.checkpoint, a.verification, a.source_manifest, a.output, a.layer); return 0


if __name__ == "__main__": raise SystemExit(main())
