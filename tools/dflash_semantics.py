#!/usr/bin/env python3
"""Small fail-closed semantic helpers derived from pinned DFlash source."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

import torch


BLOCK_SIZE = 8
MASK_TOKEN_ID = 151675
TARGET_LAYER_IDS = (0, 11, 23, 35, 47)


def extract_context_feature(
    hidden_states: Sequence[torch.Tensor],
    layer_ids: Sequence[int] = TARGET_LAYER_IDS,
) -> torch.Tensor:
    """Match DFlash's output-hidden-state offset and last-axis concatenation."""
    if not layer_ids or len(set(layer_ids)) != len(layer_ids):
        raise ValueError("target layer IDs must be nonempty and unique")
    if any(layer < 0 or layer + 1 >= len(hidden_states) for layer in layer_ids):
        raise ValueError("target layer ID is outside output_hidden_states")
    selected = [hidden_states[layer + 1] for layer in layer_ids]
    first = selected[0]
    if first.ndim != 3 or not first.is_floating_point() or not torch.isfinite(first).all():
        raise ValueError("target hidden state must be finite rank-3 floating point")
    for value in selected[1:]:
        if (
            value.ndim != 3
            or value.shape[:-1] != first.shape[:-1]
            or value.dtype != first.dtype
            or value.device != first.device
            or not torch.isfinite(value).all()
        ):
            raise ValueError("target hidden states are incompatible")
    return torch.cat(selected, dim=-1)


def initial_block_ids(
    first_target_token: int,
    *,
    block_size: int = BLOCK_SIZE,
    mask_token_id: int = MASK_TOKEN_ID,
) -> torch.Tensor:
    if block_size < 2 or first_target_token < 0 or mask_token_id < 0:
        raise ValueError("invalid DFlash block identity")
    result = torch.full((1, block_size), mask_token_id, dtype=torch.long)
    result[0, 0] = first_target_token
    return result


def first_block_position_ids(context_length: int, block_size: int = BLOCK_SIZE) -> torch.Tensor:
    if context_length < 1 or block_size < 2:
        raise ValueError("invalid DFlash position range")
    return torch.arange(context_length + block_size, dtype=torch.long).unsqueeze(0)


def install_greedy_draft_suffix(
    block_ids: torch.Tensor, draft_logits: torch.Tensor
) -> torch.Tensor:
    if block_ids.ndim != 2 or block_ids.shape[0] != 1 or block_ids.shape[1] < 2:
        raise ValueError("DFlash block must have shape [1, q], q >= 2")
    expected = (1, block_ids.shape[1] - 1)
    if draft_logits.ndim != 3 or draft_logits.shape[:2] != expected:
        raise ValueError("draft logits do not cover the seven-token suffix")
    if draft_logits.shape[-1] < 2 or not draft_logits.is_floating_point():
        raise ValueError("draft logits have invalid vocabulary/dtype")
    if not torch.isfinite(draft_logits).all():
        raise ValueError("draft logits are non-finite")
    result = block_ids.clone()
    result[:, 1:] = torch.argmax(draft_logits, dim=-1)
    return result


@dataclass(frozen=True)
class GreedyVerification:
    q: int
    matching_draft_tokens: int
    accepted_length_a: int
    accepted_block_token_ids: tuple[int, ...]
    correction_token_id: int
    rejected_draft_token_ids: tuple[int, ...]

    def to_dict(self) -> dict:
        return asdict(self)


def verify_greedy_block(
    proposed_block_ids: torch.Tensor, target_posterior_ids: torch.Tensor
) -> GreedyVerification:
    """Match the published temperature-zero prefix acceptance/start advance."""
    if (
        proposed_block_ids.ndim != 2
        or target_posterior_ids.ndim != 2
        or proposed_block_ids.shape != target_posterior_ids.shape
        or proposed_block_ids.shape[0] != 1
        or proposed_block_ids.shape[1] < 2
        or proposed_block_ids.dtype != torch.long
        or target_posterior_ids.dtype != torch.long
    ):
        raise ValueError("proposal and posterior must be equal [1, q] int64 blocks")
    if (proposed_block_ids < 0).any() or (target_posterior_ids < 0).any():
        raise ValueError("token IDs must be nonnegative")
    matches = proposed_block_ids[:, 1:] == target_posterior_ids[:, :-1]
    matching = int(matches.to(torch.int64).cumprod(dim=1).sum().item())
    accepted_length = matching + 1
    block = proposed_block_ids[0].tolist()
    posterior = target_posterior_ids[0].tolist()
    return GreedyVerification(
        q=len(block),
        matching_draft_tokens=matching,
        accepted_length_a=accepted_length,
        accepted_block_token_ids=tuple(block[:accepted_length]),
        correction_token_id=int(posterior[matching]),
        rejected_draft_token_ids=tuple(block[accepted_length:]),
    )


def validate_first_block_cache_lengths(
    cache_lengths: Sequence[int], context_length: int, block_size: int = BLOCK_SIZE
) -> None:
    expected = context_length + block_size
    if len(cache_lengths) != 5 or any(length != expected for length in cache_lengths):
        raise ValueError(f"DFlash cache lengths must be five copies of {expected}")
