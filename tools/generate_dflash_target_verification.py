#!/usr/bin/env python3
"""Run the one PW-0102 Phase-C source-target width-eight verification walk."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time
from typing import Any

import numpy as np
import torch

try:
    from tools.dflash_semantics import verify_greedy_block
    from tools.generate_full_prefix_oracle import (
        NUMERICS,
        PATTERN,
        checked_bf16,
        checked_fp8,
        embedding,
        routed_mlp,
    )
    from tools.generate_incremental_cache_oracle import (
        LayerCache,
        attention_prefill,
        decoder_layer,
        validate_cache,
        visible_start,
    )
    from tools.generate_real_layer0_bf16_oracle import (
        PROMPT_IDS,
        REVISION,
        VERIFICATION_SHA256,
        rms_norm,
    )
    from tools.generate_real_layer1_expert_oracle import ShardedCheckpoint
    from tools.host_safety import HostSafetyMonitor
    from tools.openrouter_reference import atomic_write_new, canonical_json
except ModuleNotFoundError:
    from dflash_semantics import verify_greedy_block
    from generate_full_prefix_oracle import (
        NUMERICS,
        PATTERN,
        checked_bf16,
        checked_fp8,
        embedding,
        routed_mlp,
    )
    from generate_incremental_cache_oracle import (
        LayerCache,
        attention_prefill,
        decoder_layer,
        validate_cache,
        visible_start,
    )
    from generate_real_layer0_bf16_oracle import (
        PROMPT_IDS,
        REVISION,
        VERIFICATION_SHA256,
        rms_norm,
    )
    from generate_real_layer1_expert_oracle import ShardedCheckpoint
    from host_safety import HostSafetyMonitor
    from openrouter_reference import atomic_write_new, canonical_json


PW0091_SHA256 = "87466b59480a5a5b4256c490f1dfe670fe09f28d21d169085ab13bb1b4b7ab59"
PW0095_SHA256 = "75b4a5799bcc7dc898643c266d42a00b52c75be0f1fe1682ef253ce8fe4287a8"
DRAFT002_SHA256 = "cfae209566f433933097e1b4ca97f25e4019dab33851f5f46b294c5ab7709959"
DRAFT003_SHA256 = "0094235cbee8a19138b812a1edc40420925a198180f5cf81e9c644d14b31d5c6"
PW0102_TARGET_SHA256 = "cb30738d5a79d7d85587a68b53f876a59101d5ca09bbc7c895daaf501954f4d3"
PW0186_TARGET_SHA256 = "f773fa2859f08b57f851944aa8ba0ef9b502040058580a9344be4ce3ee1e1d1c"
FIRST_LOGITS_SHA256 = "c43be0909487235bddfe6e0de69aa42a98339faf43cd6b77d6ef4b5f1a853cab"
PW0206_PREFIX_SHA256 = "0002c617c5459d7531de99e779ecad7335afc1e6f86cbbb6071afa23da107807"
PW0206_DECODE_SHA256 = "f405225ea063bf3bfaf38a450fe752dc32c5afe54f69f5803c3ae61308caab2d"
PW0206_DFLASH_SHA256 = "e5084e606349fb9fe0b01f8e5505f43fa58969cae5398330b343f40dab7228c9"
PW0206_DFLASH_TARGET_SHA256 = "edf677be8406bd663e0d99b67c8cfb12fdad3914a100dbfd31f9d92b4787693e"
PW0206_FIRST_LOGITS_SHA256 = "2302a23b67083bad89f020302c87e8c6ec2409ced3ad640be0f7733e7fb71cce"
PROPOSED_BLOCK = [264, 1773, 102092, 102092, 102092, 1773, 1773, 1773]
PW0206_PROPOSED_BLOCK = [9707, 0, 0, 0, 0, 0, 0, 0]
PW0102_POSTERIOR = [13, 15, 18, 481, 15, 481, 15, 15]
PW0186_BLOCK = [264, 13, 15, 18, 481, 15, 481, 15]
PW0186_POSTERIOR = [13, 15, 13, 15, 15, 15, 15, 264]
Q = 8


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def authenticate_authorities(
    prefix_manifest_path: Path,
    cached_manifest_path: Path | None,
    draft_paths: list[Path],
    corrected_decode_path: Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any], int, int]:
    corrected = corrected_decode_path is not None
    expected_prefix_hash = PW0206_PREFIX_SHA256 if corrected else PW0091_SHA256
    expected_first_logits_hash = (
        PW0206_FIRST_LOGITS_SHA256 if corrected else FIRST_LOGITS_SHA256
    )
    expected_first_token = 9707 if corrected else 264
    if sha256_file(prefix_manifest_path) != expected_prefix_hash:
        raise ValueError("prefix manifest hash mismatch")
    prefix = json.loads(prefix_manifest_path.read_text())
    if (
        prefix.get("semantic") != "mimo_full_prefix_layer_final_rust_trace"
        or prefix.get("revision") != REVISION
        or prefix.get("prompt_token_ids") != PROMPT_IDS
        or prefix.get("captures", {}).get("last_logits", {}).get("sha256")
        != expected_first_logits_hash
    ):
        raise ValueError("prefix semantic identity mismatch")
    if corrected:
        if cached_manifest_path is not None:
            raise ValueError("legacy cached authority is incompatible with corrected mode")
        if sha256_file(corrected_decode_path) != PW0206_DECODE_SHA256:
            raise ValueError("PW-0206 corrected decode hash mismatch")
        decode = json.loads(corrected_decode_path.read_text())
        if (
            decode.get("semantic") != "mimo_v2_5_target_faithful_slow_chat_endpoint"
            or decode.get("revision") != REVISION
            or decode.get("prompt_token_ids") != PROMPT_IDS
            or decode.get("generated_token_ids") != [9707, 0]
        ):
            raise ValueError("PW-0206 corrected decode semantic identity mismatch")
        if len(draft_paths) != 1 or sha256_file(draft_paths[0]) != PW0206_DFLASH_SHA256:
            raise ValueError("PW-0206 corrected DFlash manifest hash mismatch")
        draft = json.loads(draft_paths[0].read_text())
        if (
            draft.get("status") != "passed"
            or draft.get("evidence_class")
            != "pw0206_corrected_qkv_exported_mask_dflash_proposal"
            or draft.get("proposed_block_token_ids") != PW0206_PROPOSED_BLOCK
            or draft.get("git_dirty") is not False
        ):
            raise ValueError("PW-0206 corrected DFlash semantic identity mismatch")
        return prefix, {
            "pw0206_prefix_manifest_sha256": PW0206_PREFIX_SHA256,
            "pw0206_target_decode_sha256": PW0206_DECODE_SHA256,
            "pw0206_dflash_manifest_sha256": PW0206_DFLASH_SHA256,
        }, expected_first_token, 0

    if cached_manifest_path is None:
        raise ValueError("legacy target verification requires cached authority")
    if sha256_file(cached_manifest_path) != PW0095_SHA256:
        raise ValueError("PW-0095 manifest hash mismatch")
    cached = json.loads(cached_manifest_path.read_text())
    if cached.get("revision") != REVISION or cached.get("output_token_id") != 13:
        raise ValueError("PW-0095 cached-token identity mismatch")
    if len(draft_paths) != 2:
        raise ValueError("exactly two draft manifests are required")
    expected_hashes = [DRAFT002_SHA256, DRAFT003_SHA256]
    drafts = []
    for path, expected in zip(draft_paths, expected_hashes, strict=True):
        if sha256_file(path) != expected:
            raise ValueError("Phase-B draft manifest hash mismatch")
        draft = json.loads(path.read_text())
        if draft.get("status") != "passed" or draft.get("proposed_block_token_ids") != PROPOSED_BLOCK:
            raise ValueError("Phase-B proposal identity mismatch")
        drafts.append(draft)
    projection_keys = ("proposed_block_token_ids", "semantic_adapter", "layer_states", "captures")
    if {key: drafts[0][key] for key in projection_keys} != {
        key: drafts[1][key] for key in projection_keys
    }:
        raise ValueError("Phase-B deterministic projection mismatch")
    return prefix, {
        "pw0091_manifest_sha256": PW0091_SHA256,
        "pw0095_manifest_sha256": PW0095_SHA256,
        "draft_manifest_sha256": expected_hashes,
    }, expected_first_token, 13


def jacobi_successor_block(proposed: list[int], posterior: list[int]) -> list[int]:
    """Shift one authenticated greedy target posterior into the next Jacobi row."""
    if len(proposed) != Q or len(posterior) != Q or proposed[0] < 0:
        raise ValueError("Jacobi successor shape or anchor mismatch")
    if any(not isinstance(token, int) or not 0 <= token < 152576 for token in posterior):
        raise ValueError("Jacobi successor posterior token domain mismatch")
    return [proposed[0], *posterior[:-1]]


def selected_proposed_block(arguments: argparse.Namespace) -> tuple[list[int], str, dict[str, Any]]:
    if getattr(arguments, "corrected_decode", None) is not None:
        second = getattr(arguments, "jacobi_second_iteration", False)
        third = getattr(arguments, "jacobi_third_iteration", False)
        prior_path = getattr(arguments, "prior_target_manifest", None)
        if second and third:
            raise ValueError("select exactly one corrected Jacobi iteration")
        if third:
            raise ValueError("corrected Jacobi third iteration lacks an authenticated predecessor")
        if second:
            if prior_path is None or sha256_file(prior_path) != PW0206_DFLASH_TARGET_SHA256:
                raise ValueError("corrected Jacobi prior target manifest hash mismatch")
            prior = json.loads(prior_path.read_text())
            if (
                prior.get("status") != "passed"
                or prior.get("evidence_class")
                != "pw0206_corrected_qkv_dflash_block_verification"
                or prior.get("proposed_block_token_ids") != PW0206_PROPOSED_BLOCK
                or prior.get("target_posterior_token_ids")
                != [0, 2585, 2585, 2585, 2585, 2585, 2585, 2585]
                or prior.get("greedy_verification", {}).get("accepted_length_a") != 2
            ):
                raise ValueError("corrected Jacobi prior target semantic identity mismatch")
            return (
                jacobi_successor_block(
                    PW0206_PROPOSED_BLOCK, prior["target_posterior_token_ids"]
                ),
                "pw0206_corrected_qkv_jacobi_second_iteration",
                {"pw0206_dflash_target_manifest_sha256": PW0206_DFLASH_TARGET_SHA256},
            )
        if prior_path is not None:
            raise ValueError("corrected prior target authority requires Jacobi iteration")
        return (
            PW0206_PROPOSED_BLOCK,
            "pw0206_corrected_qkv_dflash_block_verification",
            {},
        )
    second = getattr(arguments, "jacobi_second_iteration", False)
    third = getattr(arguments, "jacobi_third_iteration", False)
    if second and third:
        raise ValueError("select exactly one Jacobi iteration")
    if not second and not third:
        if getattr(arguments, "prior_target_manifest", None) is not None:
            raise ValueError("prior target authority requires Jacobi second iteration")
        return PROPOSED_BLOCK, "pw0102_source_target_dflash_block_verification", {}
    prior_path = getattr(arguments, "prior_target_manifest", None)
    expected_hash = PW0102_TARGET_SHA256 if second else PW0186_TARGET_SHA256
    if prior_path is None or sha256_file(prior_path) != expected_hash:
        raise ValueError("Jacobi prior target manifest hash mismatch")
    prior = json.loads(prior_path.read_text())
    if third:
        if (
            prior.get("status") != "passed"
            or prior.get("evidence_class") != "pw0186_source_target_jacobi_second_iteration"
            or prior.get("revision") != REVISION
            or prior.get("proposed_block_token_ids") != PW0186_BLOCK
            or prior.get("target_posterior_token_ids") != PW0186_POSTERIOR
            or prior.get("greedy_verification", {}).get("accepted_length_a") != 3
        ):
            raise ValueError("Jacobi second-iteration semantic identity mismatch")
        return (
            jacobi_successor_block(PW0186_BLOCK, PW0186_POSTERIOR),
            "pw0187_source_target_jacobi_third_iteration",
            {"pw0186_target_manifest_sha256": PW0186_TARGET_SHA256},
        )
    if (
        prior.get("status") != "passed"
        or prior.get("revision") != REVISION
        or prior.get("proposed_block_token_ids") != PROPOSED_BLOCK
        or prior.get("target_posterior_token_ids") != PW0102_POSTERIOR
        or prior.get("greedy_verification", {}).get("accepted_length_a") != 1
    ):
        raise ValueError("Jacobi prior target semantic identity mismatch")
    return (
        jacobi_successor_block(PROPOSED_BLOCK, PW0102_POSTERIOR),
        "pw0186_source_target_jacobi_second_iteration",
        {"pw0102_target_manifest_sha256": PW0102_TARGET_SHA256},
    )


def tensor_capture(output: Path, name: str, value: torch.Tensor, dtype: str) -> dict[str, Any]:
    widened = value.detach().float().cpu().contiguous().numpy().astype("<f4", copy=False)
    payload = widened.tobytes()
    atomic_write_new(output / f"{name}.f32", payload)
    return {"file": f"{name}.f32", "shape": list(value.shape), "dtype": dtype,
            "sha256": hashlib.sha256(payload).hexdigest()}


def apply_rope_positions(values: torch.Tensor, theta: float, first_position: int) -> torch.Tensor:
    if values.ndim != 3 or values.shape[2] != 192 or first_position < 0:
        raise ValueError("block RoPE shape or position mismatch")
    inv = 1.0 / (theta ** (torch.arange(0, 64, 2, dtype=torch.float32) / 64))
    positions = torch.arange(
        first_position, first_position + values.shape[0], dtype=torch.float32
    )
    frequencies = positions[:, None] * inv[None, :]
    cosine = torch.cat((frequencies, frequencies), dim=-1).cos().to(torch.bfloat16)
    sine = torch.cat((frequencies, frequencies), dim=-1).sin().to(torch.bfloat16)
    result = values.clone()
    rotating = values[:, :, :64]
    half = torch.cat((-rotating[:, :, 32:], rotating[:, :, :32]), dim=-1)
    result[:, :, :64] = rotating * cosine[:, None, :] + half * sine[:, None, :]
    return result


def attention_block(
    checkpoint: ShardedCheckpoint,
    layer: int,
    normalized: torch.Tensor,
    cache: LayerCache,
) -> torch.Tensor:
    if tuple(normalized.shape) != (Q, 4096):
        raise ValueError("target verification attention requires eight rows")
    prefix = f"model.layers.{layer}.self_attn"
    is_swa = PATTERN[layer] == 1
    kv_heads = 8 if is_swa else 4
    validate_cache(cache, len(PROMPT_IDS), kv_heads)
    qkv = checked_fp8(
        checkpoint, f"{prefix}.qkv_proj.weight", normalized, full_qkv=not is_swa
    )
    q_size, k_size = 64 * 192, kv_heads * 192
    theta = 10_000.0 if is_swa else 10_000_000.0
    q = apply_rope_positions(qkv[:, :q_size].reshape(Q, 64, 192), theta, len(PROMPT_IDS))
    new_k = apply_rope_positions(
        qkv[:, q_size:q_size + k_size].reshape(Q, kv_heads, 192),
        theta,
        len(PROMPT_IDS),
    )
    new_v = (qkv[:, q_size + k_size:].reshape(Q, kv_heads, 128) * 0.707).to(torch.bfloat16)
    cache.keys = torch.cat((cache.keys, new_k))
    cache.values = torch.cat((cache.values, new_v))
    validate_cache(cache, len(PROMPT_IDS) + Q, kv_heads)
    sinks = checkpoint.tensor(f"{prefix}.attention_sink_bias") if is_swa else None
    core = torch.empty((Q, 64, 128), dtype=torch.bfloat16)
    scale = 1.0 / math.sqrt(192)
    groups = 64 // kv_heads
    for row in range(Q):
        end = len(PROMPT_IDS) + row + 1
        start = visible_start(is_swa, end)
        for head in range(64):
            kv_head = head // groups
            scores = (q[row, head] @ cache.keys[start:end, kv_head].T) * scale
            if sinks is not None:
                scores = torch.cat((scores, sinks[head:head + 1]))
            scores = scores - scores.max()
            probabilities = torch.softmax(scores, dim=-1, dtype=torch.float32).to(torch.bfloat16)
            if sinks is not None:
                probabilities = probabilities[:-1]
            core[row, head] = probabilities @ cache.values[start:end, kv_head]
    return checked_bf16(checkpoint, f"{prefix}.o_proj.weight", core.reshape(Q, 8192))


def tensor_storage_bytes(checkpoint: ShardedCheckpoint, name: str, cache: dict[str, int]) -> int:
    if name in cache:
        return cache[name]
    from safetensors import safe_open
    with safe_open(checkpoint.shard(name), framework="pt", device="cpu") as source:
        view = source.get_slice(name)
        dtype = view.get_dtype()
        shape = view.get_shape()
    item_size = {"F8_E4M3": 1, "BF16": 2, "F32": 4}.get(dtype)
    if item_size is None:
        raise ValueError(f"unknown ledger dtype: {dtype}")
    result = math.prod(shape) * item_size
    cache[name] = result
    return result


def source_ledger(
    checkpoint: ShardedCheckpoint,
    traces: list[dict[str, Any]],
    rows: int,
) -> dict[str, int]:
    sizes: dict[str, int] = {}
    categories = {"dense": rows * 4096 * 2, "attention": 0, "routing": 0, "expert": 0}
    for layer, trace in enumerate(traces):
        prefix = f"model.layers.{layer}"
        categories["dense"] += tensor_storage_bytes(checkpoint, f"{prefix}.input_layernorm.weight", sizes)
        categories["dense"] += tensor_storage_bytes(checkpoint, f"{prefix}.post_attention_layernorm.weight", sizes)
        attn = f"{prefix}.self_attn"
        for suffix in ("qkv_proj.weight", "qkv_proj.weight_scale_inv", "o_proj.weight"):
            categories["attention"] += tensor_storage_bytes(checkpoint, f"{attn}.{suffix}", sizes)
        if PATTERN[layer]:
            categories["attention"] += tensor_storage_bytes(checkpoint, f"{attn}.attention_sink_bias", sizes)
        mlp = f"{prefix}.mlp"
        if layer == 0:
            for projection in ("gate_proj", "up_proj", "down_proj"):
                for suffix in ("weight", "weight_scale_inv"):
                    categories["dense"] += tensor_storage_bytes(
                        checkpoint, f"{mlp}.{projection}.{suffix}", sizes
                    )
        else:
            categories["routing"] += tensor_storage_bytes(checkpoint, f"{mlp}.gate.weight", sizes)
            categories["routing"] += tensor_storage_bytes(
                checkpoint, f"{mlp}.gate.e_score_correction_bias", sizes
            )
            for expert in trace["unique_experts"]:
                for projection in ("gate_proj", "up_proj", "down_proj"):
                    for suffix in ("weight", "weight_scale_inv"):
                        categories["expert"] += tensor_storage_bytes(
                            checkpoint, f"{mlp}.experts.{expert}.{projection}.{suffix}", sizes
                        )
    categories["dense"] += tensor_storage_bytes(checkpoint, "model.norm.weight", sizes)
    categories["dense"] += tensor_storage_bytes(checkpoint, "lm_head.weight", sizes)
    categories["total_source"] = sum(categories.values())
    return categories


def kv_ledger(rows: int, prefix_rows: int = 0) -> dict[str, int]:
    writes = reads = 0
    for layer in range(48):
        kv_heads = 8 if PATTERN[layer] else 4
        writes += rows * kv_heads * (192 + 128) * 2
        for row in range(rows):
            end = prefix_rows + row + 1
            start = visible_start(bool(PATTERN[layer]), end)
            reads += 64 * (end - start) * (192 + 128) * 2
    return {"cache_write_bytes": writes, "attention_cache_read_bytes": reads}


def layer_trace(layer: int, selected: list[list[int]], weights: list[list[float]], union: int,
                cache_length: int, wall_ms: float) -> dict[str, Any]:
    unique = sorted({expert for row in selected for expert in row})
    if layer == 0:
        if selected or union != 0:
            raise ValueError("dense layer unexpectedly routed")
    elif len(unique) != union:
        raise ValueError("routed expert union mismatch")
    return {
        "layer": layer,
        "attention": "sliding_window_128" if PATTERN[layer] else "full",
        "selected_experts_by_position": selected,
        "route_weights_by_position": weights,
        "unique_experts": unique,
        "expert_union_count": union,
        "normalized_union": union / 8.0 if layer else 0.0,
        "cache_length": cache_length,
        "wall_ms": wall_ms,
    }


def run(arguments: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    torch.set_num_threads(1)
    arguments.output.mkdir(parents=True, exist_ok=False)
    safety = HostSafetyMonitor()
    arguments._safety = safety
    (
        prefix_authority,
        identities,
        first_target_token,
        expected_first_posterior_token,
    ) = authenticate_authorities(
        arguments.prefix_manifest,
        arguments.cached_manifest,
        arguments.draft_manifest,
        arguments.corrected_decode,
    )
    proposed_block, evidence_class, proposal_identity = selected_proposed_block(arguments)
    identities.update(proposal_identity)
    checkpoint = ShardedCheckpoint(arguments.checkpoint, arguments.verification)
    safety.checkpoint("authorities_and_checkpoint_open")
    process_start_disk = safety.evidence()[0]["process_disk_bytes_read"]

    captures: dict[str, Any] = {}
    caches: list[LayerCache] = []
    prefill_traces: list[dict[str, Any]] = []
    hidden = embedding(checkpoint, PROMPT_IDS)
    prefill_started = time.monotonic()
    for layer in range(48):
        layer_started = time.monotonic()
        prefix = f"model.layers.{layer}"
        normalized = rms_norm(hidden, checkpoint.tensor(f"{prefix}.input_layernorm.weight"))
        projected, cache = attention_prefill(checkpoint, layer, normalized)
        hidden, selected, weights, union = decoder_layer(checkpoint, layer, hidden, projected)
        caches.append(cache)
        trace = layer_trace(
            layer, selected, weights, union, cache.keys.shape[0],
            (time.monotonic() - layer_started) * 1000,
        )
        authoritative = prefix_authority["layer_traces"][layer]
        if selected != authoritative["selected_experts_by_position"]:
            raise ValueError(f"authenticated prefill route mismatch at layer {layer}")
        name = f"prefill_layer_{layer:02}_final"
        capture = tensor_capture(arguments.output, name, hidden, "BF16_widened_F32")
        if capture["sha256"] != prefix_authority["captures"][f"layer_{layer:02}_final"]["sha256"]:
            raise ValueError(f"authenticated prefill hidden mismatch at layer {layer}")
        captures[name] = capture
        prefill_traces.append(trace)
        safety.checkpoint(f"prefill_layer_{layer}_complete")
    prefill_layers_ms = (time.monotonic() - prefill_started) * 1000

    final_norm = rms_norm(hidden, checkpoint.tensor("model.norm.weight"))
    first_logits = checked_bf16(checkpoint, "lm_head.weight", final_norm[-1:]).float().reshape(-1)
    captures["prefill_last_logits"] = tensor_capture(
        arguments.output, "prefill_last_logits", first_logits, "F32"
    )
    expected_first_logits_hash = (
        PW0206_FIRST_LOGITS_SHA256
        if arguments.corrected_decode is not None
        else FIRST_LOGITS_SHA256
    )
    if (
        captures["prefill_last_logits"]["sha256"] != expected_first_logits_hash
        or int(first_logits.argmax()) != first_target_token
    ):
        raise ValueError("authenticated first-token distribution mismatch")
    safety.checkpoint("prefill_first_token_verified")
    del first_logits, final_norm, hidden
    safety.release_checkpoint("prefill_lm_head_released", ["prefill hidden", "prefill LM-head/logits"])

    post_prefill_started = time.monotonic()
    block_hidden = embedding(checkpoint, proposed_block)
    verification_traces: list[dict[str, Any]] = []
    block_started = time.monotonic()
    for layer in range(48):
        layer_started = time.monotonic()
        prefix = f"model.layers.{layer}"
        normalized = rms_norm(block_hidden, checkpoint.tensor(f"{prefix}.input_layernorm.weight"))
        projected = attention_block(checkpoint, layer, normalized, caches[layer])
        block_hidden, selected, weights, union = decoder_layer(
            checkpoint, layer, block_hidden, projected
        )
        verification_traces.append(layer_trace(
            layer, selected, weights, union, caches[layer].keys.shape[0],
            (time.monotonic() - layer_started) * 1000,
        ))
        captures[f"verification_layer_{layer:02}_final"] = tensor_capture(
            arguments.output, f"verification_layer_{layer:02}_final", block_hidden,
            "BF16_widened_F32",
        )
        safety.checkpoint(f"verification_layer_{layer}_complete")
    target_layers_ms = (time.monotonic() - block_started) * 1000

    final_norm = rms_norm(block_hidden, checkpoint.tensor("model.norm.weight"))
    target_logits = checked_bf16(checkpoint, "lm_head.weight", final_norm).float()
    if target_logits.shape != (Q, 152576) or not torch.isfinite(target_logits).all():
        raise ValueError("target posterior logits mismatch")
    captures["target_logits"] = tensor_capture(arguments.output, "target_logits", target_logits, "F32")
    posterior_ids = torch.argmax(target_logits, dim=-1).unsqueeze(0)
    if int(posterior_ids[0, 0]) != expected_first_posterior_token:
        raise ValueError("authenticated first incremental token mismatch")
    verification = verify_greedy_block(
        torch.tensor([proposed_block], dtype=torch.long), posterior_ids
    )
    posterior_token_ids = posterior_ids[0].tolist()
    routed = verification_traces[1:]
    mean_u = sum(trace["normalized_union"] for trace in routed) / len(routed)
    leverage = verification.accepted_length_a / mean_u
    safety.checkpoint("target_logits_and_acceptance_complete")
    post_prefill_ms = (time.monotonic() - post_prefill_started) * 1000

    prefill_source = source_ledger(checkpoint, prefill_traces, len(PROMPT_IDS))
    verification_source = source_ledger(checkpoint, verification_traces, Q)
    del target_logits, posterior_ids, final_norm, block_hidden, caches, checkpoint
    safety.release_checkpoint(
        "target_buffers_released",
        ["target block hidden", "target logits", "all target K/V caches", "checkpoint views"],
    )
    safety.checkpoint("phase_c_final_health")
    snapshots = safety.evidence()
    git_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, text=True, capture_output=True
    ).stdout.strip()
    dirty = bool(subprocess.run(
        ["git", "status", "--porcelain"], check=True, text=True, capture_output=True
    ).stdout.strip())
    result = {
        "schema_version": 1,
        "evidence_class": evidence_class,
        "status": "passed",
        "revision": REVISION,
        "checkpoint_verification_sha256": VERIFICATION_SHA256,
        "identities": identities,
        "git_commit": git_commit,
        "git_dirty": dirty,
        "batch_size": 1,
        "concurrency": 1,
        "q": Q,
        "prompt_token_ids": PROMPT_IDS,
        "proposed_block_token_ids": proposed_block,
        "target_posterior_token_ids": posterior_token_ids,
        "greedy_verification": verification.to_dict(),
        "mean_normalized_union_u": mean_u,
        "acceptance_over_union_a_over_u": leverage,
        "pw0011_int4_otherwise_free_required_a_over_u": 7.548793,
        "routed_expert_byte_leverage": leverage > 1.0,
        "meets_pw0011_otherwise_free_threshold": leverage >= 7.548793,
        "numerics": NUMERICS,
        "captures": captures,
        "prefill_layer_traces": prefill_traces,
        "verification_layer_traces": verification_traces,
        "logical_bytes": {
            "draft_source": {
                "draft_model_tensor_bytes": 2936114304,
                "base_embedding_selected_rows_bytes": Q * 4096 * 2,
                "base_lm_head_source_bytes": 152576 * 4096 * 2,
            },
            "prefill_source": prefill_source,
            "verification_source": verification_source,
            "prefill_kv": kv_ledger(len(PROMPT_IDS)),
            "verification_kv": kv_ledger(Q, len(PROMPT_IDS)),
            "synchronization_payload_bytes": 0,
            "synchronization_note": "single-process CPU oracle; no cross-device synchronization payload",
        },
        "physical_io": {
            "draft_cold_process_disk_read_bytes": 3901050880,
            "draft_warm_process_disk_read_bytes": 26480640,
            "process_disk_bytes_read_delta": snapshots[-1]["process_disk_bytes_read"] - process_start_disk,
            "process_disk_bytes_written_delta": snapshots[-1]["process_disk_bytes_written"] - snapshots[0]["process_disk_bytes_written"],
        },
        "timings_ms": {
            "prefill_layers": prefill_layers_ms,
            "target_verification_layers": target_layers_ms,
            "post_prefill_complete": post_prefill_ms,
            "complete": (time.monotonic() - started) * 1000,
        },
        "single_trace_post_prefill_accepted_tps_diagnostic": (
            verification.accepted_length_a * 1000.0 / post_prefill_ms
        ),
        "accepted_tokens": verification.accepted_length_a,
        "performance_claim": None,
        "safety": snapshots,
    }
    atomic_write_new(arguments.output / "manifest.json", canonical_json(result))
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--verification", required=True, type=Path)
    parser.add_argument("--prefix-manifest", required=True, type=Path)
    parser.add_argument("--cached-manifest", type=Path)
    parser.add_argument("--draft-manifest", action="append", required=True, type=Path)
    parser.add_argument("--corrected-decode", type=Path)
    parser.add_argument("--jacobi-second-iteration", action="store_true")
    parser.add_argument("--jacobi-third-iteration", action="store_true")
    parser.add_argument("--prior-target-manifest", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    try:
        result = run(arguments)
        print(json.dumps({
            "output": str(arguments.output),
            "posterior": result["target_posterior_token_ids"],
            "A": result["greedy_verification"]["accepted_length_a"],
            "U": result["mean_normalized_union_u"],
            "A_over_U": result["acceptance_over_union_a_over_u"],
        }))
        return 0
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as error:
        safety = getattr(arguments, "_safety", None)
        failure_path = arguments.output / "failure.json"
        if safety is not None and arguments.output.is_dir() and not failure_path.exists():
            failure = {"schema_version": 1,
                "evidence_class": (
                    "pw0206_corrected_qkv_dflash_block_verification_failure"
                    if arguments.corrected_decode is not None
                    else "pw0102_source_target_dflash_block_verification_failure"
                ),
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
