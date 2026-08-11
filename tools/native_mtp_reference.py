"""Readable complete-history native-MTP reference primitives.

The production MTP block needs all causal K/V rows but only the final output
row when it is used as a draft layer.  These helpers preserve that distinction
so the CPU oracle does not spend three full dense MLP passes per proposal.
"""

from __future__ import annotations

from dataclasses import dataclass
import gc
import math
from pathlib import Path
import time
from typing import Callable

from safetensors import safe_open
import torch

try:
    from tools.generate_full_prefix_oracle import checked_bf16
    from tools.generate_native_mtp_first_proposal import (
        bf16_linear_path,
        checked_fp8_path,
        tensor,
    )
    from tools.generate_real_layer0_bf16_oracle import apply_rope, rms_norm
    from tools.generate_real_layer1_expert_oracle import ShardedCheckpoint
except ModuleNotFoundError:
    from generate_full_prefix_oracle import checked_bf16
    from generate_native_mtp_first_proposal import (
        bf16_linear_path,
        checked_fp8_path,
        tensor,
    )
    from generate_real_layer0_bf16_oracle import apply_rope, rms_norm
    from generate_real_layer1_expert_oracle import ShardedCheckpoint


VOCAB_SIZE = 152_576
HIDDEN_SIZE = 4096
MAX_HISTORY = 128


@dataclass(frozen=True)
class NativeMtpLayerResult:
    layer: int
    proposal_token_id: int
    top20: tuple[tuple[int, float], ...]
    logits: torch.Tensor
    timings_ms: dict[str, float]


def rotate_mtp_input_ids(input_ids: list[int], proposal_token_id: int) -> list[int]:
    """Apply SGLang's non-chain MiMo MTP input rotation for the next layer."""
    if not input_ids:
        raise ValueError("MTP input IDs must be nonempty")
    if any(not isinstance(token, int) or token < 0 or token >= VOCAB_SIZE for token in input_ids):
        raise ValueError("MTP input token ID is invalid")
    if not isinstance(proposal_token_id, int) or not 0 <= proposal_token_id < VOCAB_SIZE:
        raise ValueError("MTP proposal token ID is invalid")
    return [*input_ids[1:], proposal_token_id]


def q4_proposal_block(anchor_token_id: int, layer_proposals: list[int]) -> list[int]:
    """Form the trained three-layer native chain's four-token verifier block."""
    if len(layer_proposals) != 3:
        raise ValueError("native q4 requires exactly three MTP layer proposals")
    block = [anchor_token_id, *layer_proposals]
    if any(not isinstance(token, int) or not 0 <= token < VOCAB_SIZE for token in block):
        raise ValueError("native q4 token ID is invalid")
    return block


def load_embeddings(checkpoint: ShardedCheckpoint, token_ids: list[int]) -> torch.Tensor:
    if not token_ids or any(
        not isinstance(token, int) or token < 0 or token >= VOCAB_SIZE for token in token_ids
    ):
        raise ValueError("embedding token IDs are invalid")
    name = "model.embed_tokens.weight"
    with safe_open(checkpoint.shard(name), framework="pt", device="cpu") as source:
        view = source.get_slice(name)
        result = torch.cat([view[token:token + 1] for token in token_ids])
    if result.dtype != torch.bfloat16 or tuple(result.shape) != (len(token_ids), HIDDEN_SIZE):
        raise ValueError("embedding result layout mismatch")
    return result


def last_row_attention_core(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    sinks: torch.Tensor,
    window: int = MAX_HISTORY,
) -> torch.Tensor:
    """Compute only the final causal attention row from complete Q/K/V history."""
    rows = q.shape[0]
    if (
        rows < 1
        or tuple(q.shape) != (rows, 64, 192)
        or tuple(k.shape) != (rows, 8, 192)
        or tuple(v.shape) != (rows, 8, 128)
        or tuple(sinks.shape) != (64,)
        or window < 1
    ):
        raise ValueError("native MTP attention layout mismatch")
    start = max(0, rows - window)
    output = torch.empty((1, 64, 128), dtype=torch.bfloat16)
    scale = 1.0 / math.sqrt(192)
    for head in range(64):
        kv_head = head // 8
        scores = (q[-1, head] @ k[start:, kv_head].T) * scale
        scores = torch.cat((scores, sinks[head:head + 1]))
        scores = scores - scores.max()
        probabilities = torch.softmax(scores, dim=-1, dtype=torch.float32).to(torch.bfloat16)
        output[0, head] = probabilities[:-1] @ v[start:, kv_head]
    return output


def last_row_attention(path: Path, prefix: str, values: torch.Tensor) -> torch.Tensor:
    rows = values.shape[0]
    if values.dtype != torch.bfloat16 or tuple(values.shape) != (rows, HIDDEN_SIZE):
        raise ValueError("native MTP fused history layout mismatch")
    normalized = rms_norm(values, tensor(path, f"{prefix}.input_layernorm.weight"))
    qkv = checked_fp8_path(path, f"{prefix}.self_attn.qkv_proj.weight", normalized)
    q = apply_rope(qkv[:, :12288].reshape(rows, 64, 192), 10_000.0)
    k = apply_rope(qkv[:, 12288:13824].reshape(rows, 8, 192), 10_000.0)
    v = (qkv[:, 13824:].reshape(rows, 8, 128) * 0.707).to(torch.bfloat16)
    core = last_row_attention_core(
        q, k, v, tensor(path, f"{prefix}.self_attn.attention_sink_bias")
    )
    result = bf16_linear_path(path, f"{prefix}.self_attn.o_proj.weight", core.reshape(1, 8192))
    del normalized, qkv, q, k, v, core
    gc.collect()
    return result


def generate_layer_proposal(
    checkpoint: ShardedCheckpoint,
    mtp_path: Path,
    layer: int,
    target_hidden: torch.Tensor,
    input_token_ids: list[int],
    checkpoint_callback: Callable[[str], None] | None = None,
) -> NativeMtpLayerResult:
    """Run one MTP layer over complete target history and emit its final proposal."""
    if layer not in (0, 1, 2):
        raise ValueError("native MTP layer must be 0, 1, or 2")
    rows = target_hidden.shape[0]
    if (
        target_hidden.dtype != torch.bfloat16
        or tuple(target_hidden.shape) != (rows, HIDDEN_SIZE)
        or len(input_token_ids) != rows
        or rows < 1
        or not torch.isfinite(target_hidden.float()).all()
    ):
        raise ValueError("native MTP target history mismatch")
    mark = checkpoint_callback or (lambda _: None)
    prefix = f"model.mtp.layers.{layer}"
    timings: dict[str, float] = {}

    started = time.monotonic()
    token_embeddings = load_embeddings(checkpoint, input_token_ids)
    normalized_embedding = rms_norm(token_embeddings, tensor(mtp_path, f"{prefix}.enorm.weight"))
    normalized_hidden = rms_norm(target_hidden, tensor(mtp_path, f"{prefix}.hnorm.weight"))
    fused = bf16_linear_path(
        mtp_path,
        f"{prefix}.eh_proj.weight",
        torch.cat((normalized_embedding, normalized_hidden), dim=-1),
    )
    timings["fusion"] = (time.monotonic() - started) * 1000
    mark(f"mtp_layer_{layer}_fusion_complete")

    started = time.monotonic()
    attention_output = last_row_attention(mtp_path, prefix, fused)
    post_attention = (fused[-1:] + attention_output).to(torch.bfloat16)
    timings["attention"] = (time.monotonic() - started) * 1000
    mark(f"mtp_layer_{layer}_attention_complete")

    started = time.monotonic()
    last_mlp_input = rms_norm(
        post_attention, tensor(mtp_path, f"{prefix}.pre_mlp_layernorm.weight")
    )
    # PyTorch chooses a row-count-dependent GEMM reduction topology.  Preserve
    # the complete-history row shape with inert leading rows so the final row
    # is bit-identical to the readable full-block oracle without calculating
    # meaningful MLP activations for earlier positions.
    mlp_input = torch.zeros((rows, HIDDEN_SIZE), dtype=torch.bfloat16)
    mlp_input[-1:] = last_mlp_input
    gate = checked_fp8_path(mtp_path, f"{prefix}.mlp.gate_proj.weight", mlp_input)
    mark(f"mtp_layer_{layer}_gate_complete")
    up = checked_fp8_path(mtp_path, f"{prefix}.mlp.up_proj.weight", mlp_input)
    mark(f"mtp_layer_{layer}_up_complete")
    activated = (torch.nn.functional.silu(gate) * up).to(torch.bfloat16)
    mlp_output = checked_fp8_path(mtp_path, f"{prefix}.mlp.down_proj.weight", activated)[-1:]
    block_hidden = (post_attention + mlp_output).to(torch.bfloat16)
    final_norm = rms_norm(block_hidden, tensor(mtp_path, f"{prefix}.final_layernorm.weight"))
    timings["mlp"] = (time.monotonic() - started) * 1000
    mark(f"mtp_layer_{layer}_mlp_complete")

    started = time.monotonic()
    logits = checked_bf16(checkpoint, "lm_head.weight", final_norm).float().reshape(-1)
    timings["lm_head"] = (time.monotonic() - started) * 1000
    if tuple(logits.shape) != (VOCAB_SIZE,) or not torch.isfinite(logits).all():
        raise ValueError("native MTP logits layout mismatch")
    proposal = int(logits.argmax())
    top_values, top_ids = torch.topk(logits, 20)
    top20 = tuple(
        (int(token), float(value))
        for token, value in zip(top_ids.tolist(), top_values.tolist(), strict=True)
    )
    mark(f"mtp_layer_{layer}_logits_complete")
    del token_embeddings, normalized_embedding, normalized_hidden, fused
    del attention_output, post_attention, last_mlp_input, mlp_input, gate, up, activated
    del mlp_output, block_hidden, final_norm
    gc.collect()
    return NativeMtpLayerResult(layer, proposal, top20, logits, timings)
