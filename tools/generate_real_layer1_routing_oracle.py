#!/usr/bin/env python3
"""Generate the independent PW-0058 real layer-1 attention/routing trace."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

try:
    from tools.generate_real_layer0_bf16_oracle import (
        PROMPT_IDS, REVISION, VERIFICATION_SHA256, Safety, apply_rope, bf16_linear,
        fp8_linear, load_verified_shard, rms_norm, tensor, write_capture,
    )
    from tools.openrouter_reference import atomic_write_new, canonical_json
except ModuleNotFoundError:
    from generate_real_layer0_bf16_oracle import (
        PROMPT_IDS, REVISION, VERIFICATION_SHA256, Safety, apply_rope, bf16_linear,
        fp8_linear, load_verified_shard, rms_norm, tensor, write_capture,
    )
    from openrouter_reference import atomic_write_new, canonical_json


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_layer0_final(manifest_path: Path) -> tuple[torch.Tensor, str]:
    manifest = json.loads(manifest_path.read_text())
    if (manifest.get("semantic") != "mimo_real_layer0_bf16_dynamic_fp8_oracle"
            or manifest.get("revision") != REVISION
            or manifest.get("prompt_token_ids") != PROMPT_IDS
            or manifest.get("checkpoint_verification_sha256") != VERIFICATION_SHA256):
        raise ValueError("layer-0 oracle authority mismatch")
    record = manifest["captures"]["final"]
    path = manifest_path.parent / record["file"]
    if record.get("shape") != [27, 4096] or record.get("dtype") != "BF16_widened_F32" or sha256(path) != record.get("sha256"):
        raise ValueError("layer-0 final capture mismatch")
    values = np.fromfile(path, dtype="<f4").reshape(27, 4096)
    hidden = torch.from_numpy(values.copy()).to(torch.bfloat16)
    return hidden, record["sha256"]


def generate(checkpoint: Path, verification: Path, layer0_manifest: Path, output: Path) -> None:
    started = time.monotonic()
    torch.set_num_threads(1)
    shard = load_verified_shard(checkpoint, verification)
    incoming, source_hash = load_layer0_final(layer0_manifest)
    output.mkdir(parents=True, exist_ok=False)
    safety = Safety()
    captures = {"incoming": write_capture(output, "incoming", incoming, safety)}
    prefix = "model.layers.1"
    normalized = rms_norm(incoming, tensor(shard, f"{prefix}.input_layernorm.weight"))
    captures["input_norm"] = write_capture(output, "input_norm", normalized, safety)
    qkv = fp8_linear(shard, f"{prefix}.self_attn.qkv_proj.weight", normalized)
    captures["qkv"] = write_capture(output, "qkv", qkv, safety)
    q = apply_rope(qkv[:, :12288].reshape(27, 64, 192), 10_000.0)
    k = apply_rope(qkv[:, 12288:13824].reshape(27, 8, 192), 10_000.0)
    v = (qkv[:, 13824:].reshape(27, 8, 128) * 0.707).to(torch.bfloat16)
    sinks = tensor(shard, f"{prefix}.self_attn.attention_sink_bias")
    captures["query"] = write_capture(output, "query", q, safety)
    captures["key"] = write_capture(output, "key", k, safety)
    captures["value"] = write_capture(output, "value", v, safety)
    captures["sinks"] = write_capture(output, "sinks", sinks, safety)
    core = torch.empty((27, 64, 128), dtype=torch.bfloat16)
    score_rows, probability_rows = [], []
    scale = 1.0 / math.sqrt(192)
    for position in range(27):
        for head in range(64):
            kv_head = head // 8
            scores = (q[position, head] @ k[:position + 1, kv_head].T) * scale
            scores = torch.cat((scores, sinks[head:head + 1]))
            scores = scores - scores.max()
            probabilities = torch.softmax(scores, dim=-1, dtype=torch.float32).to(torch.bfloat16)
            score_rows.append(scores); probability_rows.append(probabilities)
            core[position, head] = probabilities[:-1] @ v[:position + 1, kv_head]
    captures["attention_scores"] = write_capture(output, "attention_scores", torch.cat(score_rows), safety)
    captures["attention_probabilities"] = write_capture(output, "attention_probabilities", torch.cat(probability_rows), safety)
    captures["attention"] = write_capture(output, "attention", core, safety)
    projected = bf16_linear(shard, f"{prefix}.self_attn.o_proj.weight", core.reshape(27, 8192))
    captures["attention_projection"] = write_capture(output, "attention_projection", projected, safety)
    post_attention = (incoming + projected).to(torch.bfloat16)
    captures["post_attention"] = write_capture(output, "post_attention", post_attention, safety)
    post_norm = rms_norm(post_attention, tensor(shard, f"{prefix}.post_attention_layernorm.weight"))
    captures["post_attention_norm"] = write_capture(output, "post_attention_norm", post_norm, safety)
    router_weight = tensor(shard, f"{prefix}.mlp.gate.weight").float()
    logits = post_norm.float() @ router_weight.T
    scores = torch.sigmoid(logits)
    correction = tensor(shard, f"{prefix}.mlp.gate.e_score_correction_bias").float()
    selected = torch.topk(scores + correction, 8, dim=-1, sorted=False).indices
    chosen = scores.gather(1, selected)
    weights = chosen / (chosen.sum(dim=-1, keepdim=True) + 1e-20)
    captures["router_logits"] = write_capture(output, "router_logits", logits, safety, "F32")
    captures["router_scores"] = write_capture(output, "router_scores", scores, safety, "F32")
    manifest = {"schema_version": 1,
        "semantic": "mimo_real_layer1_attention_to_routing_oracle", "revision": REVISION,
        "checkpoint_verification_sha256": VERIFICATION_SHA256,
        "prompt_token_ids": PROMPT_IDS, "source_input_sha256": source_hash,
        "numerics": "dynamic_fp8_e4m3fn_per_token_group_128_bf16_boundaries",
        "captures": captures, "selected_experts_by_position": selected.tolist(),
        "route_weights_by_position": weights.tolist(), "torch_version": torch.__version__,
        "safety_snapshots": safety.snapshots, "wall_ms": (time.monotonic() - started) * 1000.0,
        "performance_claim": None}
    atomic_write_new(output / "manifest.json", canonical_json(manifest))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--verification", required=True, type=Path)
    parser.add_argument("--layer0-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    generate(args.checkpoint, args.verification, args.layer0_manifest, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
