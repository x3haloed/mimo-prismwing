#!/usr/bin/env python3
"""Generate PW-0103's first native-MTP proposal from frozen base hidden states."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import numpy as np
from safetensors import safe_open
import torch

try:
    from tools.generate_full_prefix_oracle import checked_bf16, checked_fp8
    from tools.generate_real_layer0_bf16_oracle import (
        PROMPT_IDS,
        REVISION,
        VERIFICATION_SHA256,
        apply_rope,
        rms_norm,
    )
    from tools.generate_real_layer1_expert_oracle import ShardedCheckpoint, validate_file
    from tools.host_safety import HostSafetyMonitor
    from tools.openrouter_reference import atomic_write_new, canonical_json
except ModuleNotFoundError:
    from generate_full_prefix_oracle import checked_bf16, checked_fp8
    from generate_real_layer0_bf16_oracle import (
        PROMPT_IDS,
        REVISION,
        VERIFICATION_SHA256,
        apply_rope,
        rms_norm,
    )
    from generate_real_layer1_expert_oracle import ShardedCheckpoint, validate_file
    from host_safety import HostSafetyMonitor
    from openrouter_reference import atomic_write_new, canonical_json


PW0091_SHA256 = "87466b59480a5a5b4256c490f1dfe670fe09f28d21d169085ab13bb1b4b7ab59"
PW0095_SHA256 = "75b4a5799bcc7dc898643c266d42a00b52c75be0f1fe1682ef253ce8fe4287a8"
MTP_SHA256 = "a0e41a193b2762b0c83e577f83206d0777028de6916408c8c368730c0c9e2143"
SGLANG_REVISION = "2fc557254b3aaf539e80266e52a6d1e1f8da9980"
LAYER47_SHA256 = "3809f2fb5cc8ff3f543cd2d0362dccd136f2822e8aa844f6080d2effb7e6e300"
FIRST_TARGET_TOKEN = 264
EXPECTED_PROPOSAL = 13
MTP_FILE = "model_mtp.safetensors"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expected_mtp_inventory() -> dict[str, tuple[str, tuple[int, ...]]]:
    result: dict[str, tuple[str, tuple[int, ...]]] = {}
    per_layer = {
        "eh_proj.weight": ("BF16", (4096, 8192)),
        "enorm.weight": ("BF16", (4096,)),
        "final_layernorm.weight": ("BF16", (4096,)),
        "hnorm.weight": ("BF16", (4096,)),
        "input_layernorm.weight": ("BF16", (4096,)),
        "mlp.down_proj.weight": ("F8_E4M3", (4096, 16384)),
        "mlp.down_proj.weight_scale_inv": ("F32", (32, 128)),
        "mlp.gate_proj.weight": ("F8_E4M3", (16384, 4096)),
        "mlp.gate_proj.weight_scale_inv": ("F32", (128, 32)),
        "mlp.up_proj.weight": ("F8_E4M3", (16384, 4096)),
        "mlp.up_proj.weight_scale_inv": ("F32", (128, 32)),
        "pre_mlp_layernorm.weight": ("BF16", (4096,)),
        "self_attn.attention_sink_bias": ("BF16", (64,)),
        "self_attn.o_proj.weight": ("BF16", (4096, 8192)),
        "self_attn.qkv_proj.weight": ("F8_E4M3", (14848, 4096)),
        "self_attn.qkv_proj.weight_scale_inv": ("F32", (116, 32)),
    }
    for layer in range(3):
        prefix = f"model.mtp.layers.{layer}."
        result.update({prefix + suffix: identity for suffix, identity in per_layer.items()})
    return result


def validate_mtp_inventory(path: Path) -> None:
    expected = expected_mtp_inventory()
    with safe_open(path, framework="pt", device="cpu") as source:
        actual = {
            name: (source.get_slice(name).get_dtype(), tuple(source.get_slice(name).get_shape()))
            for name in source.keys()
        }
    if actual.keys() != expected.keys():
        raise ValueError(
            f"MTP tensor names mismatch: missing={sorted(expected.keys() - actual.keys())}, "
            f"extra={sorted(actual.keys() - expected.keys())}"
        )
    for name, identity in actual.items():
        if identity != expected[name]:
            raise ValueError(f"MTP tensor layout mismatch: {name}: {identity}")


def authenticate(
    checkpoint_root: Path,
    verification_path: Path,
    prefix_manifest_path: Path,
    cached_manifest_path: Path,
    source_lock_path: Path,
    source_root: Path,
) -> tuple[ShardedCheckpoint, Path, dict[str, Any], dict[str, str]]:
    checkpoint = ShardedCheckpoint(checkpoint_root, verification_path)
    verification = json.loads(verification_path.read_text())
    mtp_record = next(
        (item for item in verification.get("files", []) if item.get("path") == MTP_FILE), None
    )
    if mtp_record is None:
        raise ValueError("checkpoint verification lacks MTP payload")
    mtp_path = checkpoint_root / MTP_FILE
    validate_file(mtp_path, mtp_record)
    if mtp_record.get("sha256") != MTP_SHA256 or sha256_file(mtp_path) != MTP_SHA256:
        raise ValueError("MTP complete-file SHA-256 mismatch")
    validate_mtp_inventory(mtp_path)

    if sha256_file(prefix_manifest_path) != PW0091_SHA256:
        raise ValueError("PW-0091 manifest hash mismatch")
    prefix_manifest = json.loads(prefix_manifest_path.read_text())
    layer_record = prefix_manifest.get("captures", {}).get("layer_47_final")
    if (
        prefix_manifest.get("revision") != REVISION
        or prefix_manifest.get("prompt_token_ids") != PROMPT_IDS
        or not isinstance(layer_record, dict)
        or layer_record.get("shape") != [27, 4096]
        or layer_record.get("dtype") != "BF16_widened_F32"
        or layer_record.get("sha256") != LAYER47_SHA256
    ):
        raise ValueError("PW-0091 layer-47 authority mismatch")
    if sha256_file(prefix_manifest_path.parent / layer_record["file"]) != LAYER47_SHA256:
        raise ValueError("PW-0091 layer-47 payload mismatch")

    if sha256_file(cached_manifest_path) != PW0095_SHA256:
        raise ValueError("PW-0095 manifest hash mismatch")
    cached = json.loads(cached_manifest_path.read_text())
    if cached.get("revision") != REVISION or cached.get("output_token_id") != EXPECTED_PROPOSAL:
        raise ValueError("PW-0095 target token mismatch")

    lock_bytes = source_lock_path.read_bytes()
    lock = json.loads(lock_bytes)
    if lock.get("revision") != SGLANG_REVISION:
        raise ValueError("SGLang MTP source revision mismatch")
    for relative, expected in lock.get("files", {}).items():
        if sha256_file(source_root / relative) != expected:
            raise ValueError(f"SGLang MTP source mismatch: {relative}")
    return checkpoint, mtp_path, prefix_manifest, {
        "checkpoint_verification_sha256": VERIFICATION_SHA256,
        "mtp_sha256": MTP_SHA256,
        "pw0091_manifest_sha256": PW0091_SHA256,
        "pw0095_manifest_sha256": PW0095_SHA256,
        "sglang_mtp_lock_sha256": hashlib.sha256(lock_bytes).hexdigest(),
    }


def load_target_hidden(prefix_manifest_path: Path, manifest: dict[str, Any]) -> torch.Tensor:
    record = manifest["captures"]["layer_47_final"]
    values = np.fromfile(prefix_manifest_path.parent / record["file"], dtype="<f4")
    if values.size != 27 * 4096 or not np.isfinite(values).all():
        raise ValueError("PW-0091 layer-47 values mismatch")
    return torch.from_numpy(values.reshape(27, 4096).copy()).to(torch.bfloat16)


def tensor_capture(output: Path, name: str, value: torch.Tensor, dtype: str) -> dict[str, Any]:
    widened = value.detach().float().cpu().contiguous().numpy().astype("<f4", copy=False)
    payload = widened.tobytes()
    atomic_write_new(output / f"{name}.f32", payload)
    return {"file": f"{name}.f32", "shape": list(value.shape), "dtype": dtype,
            "sha256": hashlib.sha256(payload).hexdigest()}


def mtp_attention(path: Path, prefix: str, values: torch.Tensor) -> torch.Tensor:
    normalized = rms_norm(values, tensor(path, f"{prefix}.input_layernorm.weight"))
    qkv = checked_fp8_path(path, f"{prefix}.self_attn.qkv_proj.weight", normalized)
    q = apply_rope(qkv[:, :12288].reshape(27, 64, 192), 10_000.0)
    k = apply_rope(qkv[:, 12288:13824].reshape(27, 8, 192), 10_000.0)
    v = (qkv[:, 13824:].reshape(27, 8, 128) * 0.707).to(torch.bfloat16)
    sinks = tensor(path, f"{prefix}.self_attn.attention_sink_bias")
    core = torch.empty((27, 64, 128), dtype=torch.bfloat16)
    scale = 1.0 / math.sqrt(192)
    for position in range(27):
        start = max(0, position + 1 - 128)
        for head in range(64):
            kv_head = head // 8
            scores = (q[position, head] @ k[start:position + 1, kv_head].T) * scale
            scores = torch.cat((scores, sinks[head:head + 1]))
            scores = scores - scores.max()
            probabilities = torch.softmax(scores, dim=-1, dtype=torch.float32).to(torch.bfloat16)
            core[position, head] = probabilities[:-1] @ v[start:position + 1, kv_head]
    return bf16_linear_path(path, f"{prefix}.self_attn.o_proj.weight", core.reshape(27, 8192))


def tensor(path: Path, name: str) -> torch.Tensor:
    with safe_open(path, framework="pt", device="cpu") as source:
        return source.get_tensor(name)


def bf16_linear_path(path: Path, name: str, values: torch.Tensor) -> torch.Tensor:
    weight = tensor(path, name)
    if weight.dtype != torch.bfloat16:
        raise ValueError(f"{name}: BF16 layout mismatch")
    return (values.float() @ weight.float().T).to(torch.bfloat16)


def checked_fp8_path(path: Path, name: str, values: torch.Tensor) -> torch.Tensor:
    from tools.generate_real_layer0_bf16_oracle import fp8_linear
    return fp8_linear(path, name, values)


def run(arguments: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    torch.set_num_threads(1)
    arguments.output.mkdir(parents=True, exist_ok=False)
    safety = HostSafetyMonitor()
    arguments._safety = safety
    checkpoint, mtp_path, prefix_manifest, identities = authenticate(
        arguments.checkpoint, arguments.verification, arguments.prefix_manifest,
        arguments.cached_manifest, arguments.source_lock, arguments.source_root,
    )
    safety.checkpoint("authorities_and_mtp_inventory_authenticated")
    first_disk_read = safety.evidence()[0]["process_disk_bytes_read"]
    captures: dict[str, Any] = {}

    target_hidden = load_target_hidden(arguments.prefix_manifest, prefix_manifest)
    shifted_ids = [*PROMPT_IDS[1:], FIRST_TARGET_TOKEN]
    embed_path = checkpoint.shard("model.embed_tokens.weight")
    with safe_open(embed_path, framework="pt", device="cpu") as source:
        view = source.get_slice("model.embed_tokens.weight")
        token_embeddings = torch.cat([view[token:token + 1] for token in shifted_ids])
    if target_hidden.shape != token_embeddings.shape or target_hidden.dtype != torch.bfloat16:
        raise ValueError("MTP paired input layout mismatch")
    safety.checkpoint("shifted_embedding_and_target_hidden_loaded")

    prefix = "model.mtp.layers.0"
    fusion_started = time.monotonic()
    normalized_embedding = rms_norm(token_embeddings, tensor(mtp_path, f"{prefix}.enorm.weight"))
    normalized_hidden = rms_norm(target_hidden, tensor(mtp_path, f"{prefix}.hnorm.weight"))
    fused = bf16_linear_path(
        mtp_path, f"{prefix}.eh_proj.weight",
        torch.cat((normalized_embedding, normalized_hidden), dim=-1),
    )
    fusion_ms = (time.monotonic() - fusion_started) * 1000
    captures["fused_hidden"] = tensor_capture(arguments.output, "fused_hidden", fused, "BF16_widened_F32")
    safety.checkpoint("mtp_fusion_complete")

    attention_started = time.monotonic()
    attention_output = mtp_attention(mtp_path, prefix, fused)
    attention_ms = (time.monotonic() - attention_started) * 1000
    captures["attention_output"] = tensor_capture(
        arguments.output, "attention_output", attention_output, "BF16_widened_F32"
    )
    post_attention = (fused + attention_output).to(torch.bfloat16)
    captures["post_attention"] = tensor_capture(
        arguments.output, "post_attention", post_attention, "BF16_widened_F32"
    )
    safety.checkpoint("mtp_attention_complete")

    mlp_started = time.monotonic()
    mlp_input = rms_norm(
        post_attention, tensor(mtp_path, f"{prefix}.pre_mlp_layernorm.weight")
    )
    gate = checked_fp8_path(mtp_path, f"{prefix}.mlp.gate_proj.weight", mlp_input)
    safety.checkpoint("mtp_gate_projection_complete")
    up = checked_fp8_path(mtp_path, f"{prefix}.mlp.up_proj.weight", mlp_input)
    safety.checkpoint("mtp_up_projection_complete")
    activated = (torch.nn.functional.silu(gate) * up).to(torch.bfloat16)
    mlp_output = checked_fp8_path(mtp_path, f"{prefix}.mlp.down_proj.weight", activated)
    mlp_ms = (time.monotonic() - mlp_started) * 1000
    captures["mlp_output"] = tensor_capture(
        arguments.output, "mlp_output", mlp_output, "BF16_widened_F32"
    )
    block_hidden = (post_attention + mlp_output).to(torch.bfloat16)
    final_norm = rms_norm(
        block_hidden, tensor(mtp_path, f"{prefix}.final_layernorm.weight")
    )
    captures["final_norm"] = tensor_capture(
        arguments.output, "final_norm", final_norm, "BF16_widened_F32"
    )
    safety.checkpoint("mtp_mlp_and_final_norm_complete")

    lm_started = time.monotonic()
    logits = checked_bf16(checkpoint, "lm_head.weight", final_norm[-1:]).float().reshape(-1)
    lm_head_ms = (time.monotonic() - lm_started) * 1000
    if logits.shape != (152576,) or not torch.isfinite(logits).all():
        raise ValueError("MTP logits layout mismatch")
    captures["logits"] = tensor_capture(arguments.output, "logits", logits, "F32")
    proposal = int(logits.argmax())
    target_logit = float(logits[EXPECTED_PROPOSAL])
    rank = int((logits > target_logit).sum().item()) + 1
    top_values, top_ids = torch.topk(logits, 20)
    top20 = [
        {"token_id": int(token), "logit": float(value)}
        for token, value in zip(top_ids.tolist(), top_values.tolist(), strict=True)
    ]
    safety.checkpoint("mtp_logits_and_proposal_complete")

    del target_hidden, token_embeddings, normalized_embedding, normalized_hidden
    del fused, attention_output, post_attention, mlp_input, gate, up, activated
    del mlp_output, block_hidden, final_norm, logits, checkpoint
    safety.release_checkpoint(
        "mtp_and_lm_buffers_released",
        ["MTP paired inputs", "MTP decoder intermediates", "base LM head", "full logits"],
    )
    safety.checkpoint("pw0103_final_health")
    snapshots = safety.evidence()
    git_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, text=True, capture_output=True
    ).stdout.strip()
    git_dirty = bool(subprocess.run(
        ["git", "status", "--porcelain"], check=True, text=True, capture_output=True
    ).stdout.strip())
    result = {
        "schema_version": 1,
        "evidence_class": "pw0103_native_mtp_first_causal_proposal",
        "status": "passed" if proposal == EXPECTED_PROPOSAL else "rejected",
        "revision": REVISION,
        "identities": identities,
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "mtp_layer": 0,
        "target_hidden_layer": 47,
        "input_pairing": "rotate_prompt_left_and_append_target_anchor",
        "shifted_input_token_ids": shifted_ids,
        "target_anchor_token_id": FIRST_TARGET_TOKEN,
        "target_next_token_id": EXPECTED_PROPOSAL,
        "mtp_proposal_token_id": proposal,
        "target_token_rank_in_mtp_logits": rank,
        "target_token_logit": target_logit,
        "top20": top20,
        "captures": captures,
        "logical_source_bytes": {
            "selected_mtp_layer_tensor_bytes": 396466816,
            "complete_mtp_file_tensor_bytes": 1189400448,
            "selected_base_embedding_rows": len(shifted_ids) * 4096 * 2,
            "base_lm_head": 152576 * 4096 * 2,
        },
        "physical_io": {
            "process_disk_bytes_read_delta": snapshots[-1]["process_disk_bytes_read"] - first_disk_read,
            "process_disk_bytes_written_delta": snapshots[-1]["process_disk_bytes_written"] - snapshots[0]["process_disk_bytes_written"],
        },
        "timings_ms": {
            "fusion": fusion_ms,
            "attention": attention_ms,
            "mlp": mlp_ms,
            "lm_head": lm_head_ms,
            "complete": (time.monotonic() - started) * 1000,
        },
        "runtime": {"python": sys.version, "torch": torch.__version__, "numpy": np.__version__},
        "safety": snapshots,
        "performance_claim": None,
    }
    atomic_write_new(arguments.output / "manifest.json", canonical_json(result))
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--verification", required=True, type=Path)
    parser.add_argument("--prefix-manifest", required=True, type=Path)
    parser.add_argument("--cached-manifest", required=True, type=Path)
    parser.add_argument("--source-lock", required=True, type=Path)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    try:
        result = run(arguments)
        print(json.dumps({"output": str(arguments.output), "status": result["status"],
                          "proposal": result["mtp_proposal_token_id"],
                          "target_rank": result["target_token_rank_in_mtp_logits"]}))
        return 0
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as error:
        safety = getattr(arguments, "_safety", None)
        failure_path = arguments.output / "failure.json"
        if safety is not None and arguments.output.is_dir() and not failure_path.exists():
            failure = {"schema_version": 1,
                "evidence_class": "pw0103_native_mtp_first_causal_proposal_failure",
                "status": "failed", "error_type": type(error).__name__, "error": str(error),
                "safety": safety.evidence()}
            try:
                atomic_write_new(failure_path, canonical_json(failure))
            except OSError:
                pass
        print(json.dumps({"error": str(error), "output": str(arguments.output)}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
