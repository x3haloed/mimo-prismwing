#!/usr/bin/env python3
"""Analyze the PW-0329 corrected K4 joint-residency storage ceilings.

This program is deliberately an analytical falsifier.  It does not execute a
model, populate a cache, construct a K4 identity, or report endpoint TPS.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

try:
    from tools.host_safety import HostSafetyMonitor, HostSafetyViolation
    from tools.openrouter_reference import atomic_write_new, canonical_json
    from tools.reproduce_pw0311_k4_expert import verify_clean_commit
except ModuleNotFoundError:
    from host_safety import HostSafetyMonitor, HostSafetyViolation
    from openrouter_reference import atomic_write_new, canonical_json
    from reproduce_pw0311_k4_expert import verify_clean_commit


EXPERIMENT_ID = "PW-0329"
CONTRACT_PATH = "experiments/PW-0329-corrected-k4-joint-residency-bound.md"
CONTRACT_GIT_BLOB = "bb1bd1818b3136824d0a87f7a05b8fe1e20a5e73"
CONTRACT_SHA256 = "469debefbcb3fcec23a9e9f59844f6782e2c9f57bed94f06ab782c8012af4a27"
TARGET_SHA256 = "dda459684c194b03491f36e9b66521ff00c400a6cc38d23a567a5a92ef8fb17d"
RED_LINES_SHA256 = "cc261ad9bd67a865715e72cbbadf3b74c3f1f282e17a8ef86ed02c1a92fb8b36"

PW0328_MANIFEST_SHA256 = "36e4f10b6f807f766c87ee7078f5f18ea8fc339dd12e4dbc24f1f4ac6e824403"
PW0328_BUILDER_COMMIT = "26d2ea31852c0d63bd022df6d571fd722137c39f"
PW0318_SUMMARY_SHA256 = "a91af31bdea45749c9ae9d5d679260bcbcd8284c238479938206a7e7e0b5eb2f"
PW0318_MANIFEST_SHA256 = "ca2cd8005c3c8f712fabd0b2fc88183d740bd6613efa065cdd4b25738c4924c3"
PW0318_BUNDLE_SHA256 = "e87a0af2aba57f46b6a2f394d70e530533d04c18aa61650afbc8528a4b8bdc35"
PW0316_REJECTION_SHA256 = "7e5560cf2cdc2abdec8ec1a17af0462f69fa7204f8ba528808ce1f046d0e6ff4"
PW0207_OFFLINE_SHA256 = "1dedbef7c79aa23835d194f52760a1f2c65dcca1481bd6df2d5602615c3fdad6"
PW0136_RAW_SHA256 = "e6ab84cada19c6036ee7b83f318c3920631141b9ea5e882cc88eb9784d0b5a56"
PW0136_ANALYSIS_SHA256 = "7ebf2cde5c4a3f4931d2d705993f822e38af13ea66bc3efc91410296b14e2aab"
PW0308_MANIFEST_SHA256 = "d395cd1844ee46a938578063ab7c68ba156b6e3b1e53f29b29c58c6e33949613"
PW0308_REPEATED_SHA256 = "754cb36ba8d3831a3d7e3c59f5faebd7ea17c924b9d34f34343541ff3e7d9c4e"
PW0325_ANALYSIS_SHA256 = "9391b3b8bc8b4264ec1e74378743f00780f724eab953fb31841434d0516e81c1"
PW0325_ORDER_SHA256 = "d5a68bb4291076fbf62c8def45837b6a948d06438bbde298077fdec380f6b25a"

CHECKPOINT_REVISION = "63651580ca774f8504f676040460aed3e1244ac1"
CATEGORIES = ("code", "multilingual", "ordinary", "rare_route")
ROUTED_LAYERS = tuple(range(1, 48))
EXPERTS_PER_LAYER = 256
ALL_IDENTITY_COUNT = len(ROUTED_LAYERS) * EXPERTS_PER_LAYER
DENSITIES = (3, 4, 5, 6, 8)
RESIDENCY_GIB = (4, 6, 8, 12)
CATEGORY_TARGETS = (1.10, 1.25, 1.50)
MAXIMUM_BANK_IDENTITIES = 4096
GIB = 1024**3
ALIGNMENT_BYTES = 16 * 1024

SOURCE_LOGICAL_BYTES = 25_171_968
K4_LOGICAL_BYTES = 12_654_604
SOURCE_SCHEMA2_STRIDE = 25_214_976
K4_SCHEMA2_STRIDE = 12_877_824
SOURCE_REPACK_STRIDE = 25_182_208
K4_REPACK_STRIDE = 12_664_832
LOGICAL_SAVING = SOURCE_LOGICAL_BYTES - K4_LOGICAL_BYTES
SCHEMA2_SAVING = SOURCE_SCHEMA2_STRIDE - K4_SCHEMA2_STRIDE
REPACK_SAVING = SOURCE_REPACK_STRIDE - K4_REPACK_STRIDE

FIXED_LOGICAL_BYTES = 7_743_236_992
EMBEDDING_ROW_BYTES = 8_192
EMBEDDING_ROWS_PER_Q8 = 8
Q8_EXACT_SHARED_LOGICAL_BYTES = FIXED_LOGICAL_BYTES + EMBEDDING_ROWS_PER_Q8 * EMBEDDING_ROW_BYTES
FIXED_ALLOCATED_BYTES = 7_745_470_464
LARGEST_FIXED_OBJECT_BYTES = 1_249_902_592
K4_TLUT_LOGICAL_BYTES = 4_096
K4_TLUT_ALLOCATED_BYTES = 16_384

# Preserve both PW-0136 derivations.  Fractions make selector scores exact and
# make ties independent of platform floating-point reduction order.
BANDWIDTH_EXACT = Fraction(201_719_808_000_000_000, 58_125_375)
BANDWIDTH_EXACT_FLOAT = 3_470_425_919.832775
BANDWIDTH_FAVORABLE = Fraction("3470448309.677419")
BANDWIDTH_FAVORABLE_FLOAT = 3_470_448_309.677419
PW0308_REPEATED_P90_SECONDS = Fraction("0.351680083")

ARTIFACT_BYTES_PER_IDENTITY = 30_000_000
M1_SECONDS_PER_IDENTITY = 500

Identity = tuple[int, int]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fraction(value: int | float | str | Fraction) -> Fraction:
    if isinstance(value, Fraction):
        return value
    return Fraction(str(value))


def _seconds(moved_bytes: int, bandwidth: Fraction) -> Fraction:
    if moved_bytes < 0 or bandwidth <= 0:
        raise ValueError("invalid byte/time input")
    return Fraction(moved_bytes, 1) / bandwidth


def _tps(accepted: int, wall: Fraction) -> float | None:
    if accepted < 0 or wall < 0:
        raise ValueError("invalid aggregate input")
    return None if wall == 0 else float(Fraction(accepted, 1) / wall)


def _tps_at_least(value: float | None, threshold: float) -> bool:
    return value is None or value >= threshold


def _tps_strictly_above(value: float | None, threshold: float) -> bool:
    return value is None or value > threshold


def _summary_tps_compare(
    summary: Mapping[str, Any], threshold: float, *, strict: bool
) -> bool:
    exact = summary.get("optimistic_storage_tps_fraction")
    if exact is None:
        value = summary.get("optimistic_storage_tps")
        if value is None:
            return True
        return float(value) > threshold if strict else float(value) >= threshold
    value = Fraction(int(exact["numerator"]), int(exact["denominator"]))
    bound = _fraction(threshold)
    return value > bound if strict else value >= bound


def _p10_compare(overall: Mapping[str, Any], threshold: float, *, strict: bool) -> bool:
    exact = overall.get("nearest_rank_p10_window_optimistic_storage_tps_fraction")
    if exact is None:
        value = overall.get("nearest_rank_p10_window_optimistic_storage_tps")
        if value is None:
            return True
        return float(value) > threshold if strict else float(value) >= threshold
    value = Fraction(int(exact["numerator"]), int(exact["denominator"]))
    bound = _fraction(threshold)
    return value > bound if strict else value >= bound


def _identity_record(identity: Identity) -> dict[str, int]:
    return {"layer": identity[0], "expert": identity[1]}


def _route_row_record(rows: Sequence[RouteRow], index: int) -> dict[str, Any]:
    row = rows[index]
    return {
        "row_index": index,
        "category": row.category,
        "window_index": row.window_index,
        "transaction_index": row.transaction_index,
        "position": row.position,
        "layer": row.layer,
    }


def selection_order_sha256(order: Sequence[Identity]) -> str:
    return hashlib.sha256(canonical_json([_identity_record(item) for item in order])).hexdigest()


@dataclass(frozen=True)
class RouteRow:
    category: str
    window_index: int
    transaction_index: int
    position: int
    layer: int
    identities: tuple[Identity, ...]

    def __post_init__(self) -> None:
        if self.category not in CATEGORIES:
            raise ValueError("unknown route-row category")
        if not 0 <= self.transaction_index < 8 or not 0 <= self.position < 8:
            raise ValueError("route-row index outside q8")
        if self.layer not in ROUTED_LAYERS:
            raise ValueError("route-row layer mismatch")
        if len(self.identities) != 8 or len(set(self.identities)) != 8:
            raise ValueError("route row must contain eight distinct identities")
        if any(layer != self.layer or not 0 <= expert < EXPERTS_PER_LAYER for layer, expert in self.identities):
            raise ValueError("route-row identity mismatch")


@dataclass(frozen=True)
class Window:
    window_index: int
    category: str
    transaction_index: int
    accepted_tokens: int
    unique_experts_per_layer: float
    layer_unions: tuple[tuple[int, tuple[Identity, ...]], ...]

    def __post_init__(self) -> None:
        if self.category not in CATEGORIES:
            raise ValueError("unknown window category")
        if not 0 <= self.transaction_index < 8:
            raise ValueError("window transaction index outside q8 corpus")
        if not 1 <= self.accepted_tokens <= 8:
            raise ValueError("full verifier-authorized A is required")
        if not math.isfinite(self.unique_experts_per_layer) or self.unique_experts_per_layer <= 0:
            raise ValueError("invalid U")
        if tuple(layer for layer, _ in self.layer_unions) != ROUTED_LAYERS:
            raise ValueError("window must contain 47 ordered routed-layer unions")
        for layer, identities in self.layer_unions:
            if not identities or len(set(identities)) != len(identities):
                raise ValueError("invalid per-layer union")
            if any(item[0] != layer or not 0 <= item[1] < EXPERTS_PER_LAYER for item in identities):
                raise ValueError("per-layer union identity mismatch")

    @property
    def identities(self) -> frozenset[Identity]:
        return frozenset(identity for _, values in self.layer_unions for identity in values)


def validate_corpus_shape(windows: Sequence[Window], rows: Sequence[RouteRow]) -> None:
    """Validate the analyzer-side closure after the shared loader replay."""
    if len(windows) != 32:
        raise ValueError("PW-0328 requires exactly 32 primary windows")
    by_key = {(window.category, window.transaction_index): window for window in windows}
    if set(by_key) != {(category, index) for category in CATEGORIES for index in range(8)}:
        raise ValueError("PW-0328 category/transaction cardinality mismatch")
    if {window.window_index for window in windows} != set(range(32)):
        raise ValueError("PW-0328 window indices must be 0..31")
    grouped: dict[tuple[int, int], list[RouteRow]] = defaultdict(list)
    for row in rows:
        grouped[(row.window_index, row.layer)].append(row)
    if len(rows) != 32 * 8 * 47:
        raise ValueError("PW-0328 q8 route-row cardinality mismatch")
    for window in windows:
        union_by_layer = dict(window.layer_unions)
        for layer in ROUTED_LAYERS:
            layer_rows = grouped[(window.window_index, layer)]
            if len(layer_rows) != 8 or {row.position for row in layer_rows} != set(range(8)):
                raise ValueError("PW-0328 q8 row replay mismatch")
            reconstructed = tuple(sorted({item for row in layer_rows for item in row.identities}))
            if reconstructed != tuple(sorted(union_by_layer[layer])):
                raise ValueError("PW-0328 per-layer union replay mismatch")
        reconstructed_u = mean_q1_unique_experts(window.layer_unions)
        if not math.isclose(reconstructed_u, window.unique_experts_per_layer, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("PW-0328 U replay mismatch")
    if sum(window.accepted_tokens for window in windows) != 232:
        raise ValueError("PW-0328 full A must sum to 232")
    if not math.isclose(
        math.fsum(window.unique_experts_per_layer for window in windows),
        142.71808510638297,
        rel_tol=0.0,
        abs_tol=2e-13,
    ):
        raise ValueError("PW-0328 corpus U mismatch")


def mean_q1_unique_experts(
    layer_unions: Sequence[tuple[int, Sequence[Identity]]],
) -> float:
    """Derive PW-0328 U: mean q1 uniqueness, not mean q8 union size."""
    if tuple(layer for layer, _identities in layer_unions) != ROUTED_LAYERS:
        raise ValueError("U derivation requires 47 ordered routed layers")
    return math.fsum(len(identities) for _layer, identities in layer_unions) / (
        8 * len(ROUTED_LAYERS)
    )


def relaxed_k4_count(union_count: int, density: int) -> int:
    if union_count < 0 or density not in DENSITIES:
        raise ValueError("invalid relaxed placement input")
    return min(union_count, 8 * density)


def whole_record_stride(logical_bytes: int, alignment: int = ALIGNMENT_BYTES) -> int:
    if logical_bytes <= 0 or alignment <= 0:
        raise ValueError("invalid whole-record alignment input")
    return math.ceil(logical_bytes / alignment) * alignment


def fractional_miss(total_bytes: int, residency_bytes: int) -> int:
    if total_bytes < 0 or residency_bytes < 0:
        raise ValueError("invalid fractional-residency input")
    return max(0, total_bytes - residency_bytes)


def guarded_miss(total_allocated_bytes: int, residency_bytes: int, largest_object_bytes: int) -> int:
    if min(total_allocated_bytes, residency_bytes, largest_object_bytes) < 0:
        raise ValueError("invalid guarded-residency input")
    if total_allocated_bytes <= residency_bytes:
        return 0
    return total_allocated_bytes - max(0, residency_bytes - largest_object_bytes)


def _placement_counts(
    window: Window,
    *,
    density: int | None = None,
    selected: frozenset[Identity] | None = None,
) -> tuple[int, int, list[dict[str, int]]]:
    if (density is None) == (selected is None):
        raise ValueError("choose exactly one relaxed or fixed placement")
    k4 = 0
    source = 0
    per_layer = []
    for layer, identities in window.layer_unions:
        count = len(identities)
        selected_count = (
            relaxed_k4_count(count, int(density))
            if density is not None
            else sum(identity in selected for identity in identities)
        )
        k4 += selected_count
        source += count - selected_count
        per_layer.append(
            {
                "layer": layer,
                "union_identities": count,
                "k4_identities": selected_count,
                "source_identities": count - selected_count,
            }
        )
    return k4, source, per_layer


def window_storage_ledger(
    window: Window,
    *,
    residency_bytes: int,
    density: int | None = None,
    selected: frozenset[Identity] | None = None,
    rows: Sequence[RouteRow] = (),
) -> dict[str, Any]:
    """Compute all four predeclared storage ledgers for one window."""
    k4_count, source_count, per_layer = _placement_counts(window, density=density, selected=selected)
    expert_logical = k4_count * K4_LOGICAL_BYTES + source_count * SOURCE_LOGICAL_BYTES
    relaxed_total = FIXED_LOGICAL_BYTES + expert_logical
    exact_total = Q8_EXACT_SHARED_LOGICAL_BYTES + expert_logical
    current_expert = k4_count * K4_SCHEMA2_STRIDE + source_count * SOURCE_SCHEMA2_STRIDE
    current_total = FIXED_ALLOCATED_BYTES + current_expert
    repack_expert = k4_count * K4_REPACK_STRIDE + source_count * SOURCE_REPACK_STRIDE
    repack_total = FIXED_ALLOCATED_BYTES + repack_expert
    largest = max(LARGEST_FIXED_OBJECT_BYTES, K4_SCHEMA2_STRIDE if k4_count else 0, SOURCE_SCHEMA2_STRIDE if source_count else 0)

    relaxed_miss = fractional_miss(relaxed_total, residency_bytes)
    exact_miss = fractional_miss(exact_total, residency_bytes)
    current_miss = guarded_miss(current_total, residency_bytes, largest)
    repack_miss = guarded_miss(repack_total, residency_bytes, max(LARGEST_FIXED_OBJECT_BYTES, SOURCE_REPACK_STRIDE if source_count else K4_REPACK_STRIDE))

    row_hits: dict[str, int] | None = None
    if selected is not None and rows:
        values = [sum(identity in selected for identity in row.identities) for row in rows if row.window_index == window.window_index]
        row_hits = {str(hit): count for hit, count in sorted(Counter(values).items())}

    def model(
        total: int,
        miss: int,
        bandwidth: Fraction,
        *,
        shared: int,
        expert: int,
        semantic: str,
    ) -> dict[str, Any]:
        wall = _seconds(miss, bandwidth)
        optimistic = None if wall == 0 else Fraction(window.accepted_tokens, 1) / wall
        return {
            "semantic": semantic,
            "shared_bytes": shared,
            "expert_bytes": expert,
            "joint_total_bytes": total,
            "residency_budget_bytes": residency_bytes,
            "cache_credit_bytes": total - miss,
            "bytes_moved": miss,
            "storage_wall_seconds": float(wall),
            "storage_wall_fraction": {
                "numerator": wall.numerator,
                "denominator": wall.denominator,
            },
            "optimistic_storage_tps": None if optimistic is None else float(optimistic),
            "optimistic_storage_tps_fraction": (
                None
                if optimistic is None
                else {
                    "numerator": optimistic.numerator,
                    "denominator": optimistic.denominator,
                }
            ),
            "unbounded_storage_only": miss == 0,
            "bandwidth_bytes_per_second": float(bandwidth),
            "bandwidth_fraction": {
                "numerator": bandwidth.numerator,
                "denominator": bandwidth.denominator,
            },
        }

    route_identity_payload = [_identity_record(identity) for identity in sorted(window.identities)]
    return {
        "window_index": window.window_index,
        "category": window.category,
        "transaction_index": window.transaction_index,
        "A": window.accepted_tokens,
        "U": window.unique_experts_per_layer,
        "route_identities": len(window.identities),
        "route_identity_authority": {
            "pw0328_corpus_index": window.window_index,
            "canonical_identity_list_sha256": hashlib.sha256(
                canonical_json(route_identity_payload)
            ).hexdigest(),
            "representation_assignment": (
                "fixed_bank_selection_order"
                if selected is not None
                else "relaxed_counts_only_no_global_identity_assignment"
            ),
        },
        "k4_identity_layer_occurrences": k4_count,
        "source_identity_layer_occurrences": source_count,
        "per_layer": per_layer,
        "row_hit_histogram": row_hits,
        "fractional_relaxed": model(
            relaxed_total,
            relaxed_miss,
            BANDWIDTH_FAVORABLE,
            shared=FIXED_LOGICAL_BYTES,
            expert=expert_logical,
            semantic=(
                "logical_fractional_relaxed_density_embedding_omitted"
                if density is not None
                else "logical_fractional_fixed_bank_embedding_omitted"
            ),
        ),
        "exact_logical": model(
            exact_total,
            exact_miss,
            BANDWIDTH_EXACT,
            shared=Q8_EXACT_SHARED_LOGICAL_BYTES,
            expert=expert_logical,
            semantic="exact_logical_q8_embeddings_included",
        ),
        "current_layout_guarded": model(
            current_total,
            current_miss,
            BANDWIDTH_EXACT,
            shared=FIXED_ALLOCATED_BYTES,
            expert=current_expert,
            semantic="schema2_individual_payload_alignment_largest_object_guard",
        )
        | {"largest_object_guard_bytes": largest},
        "hypothetical_repack_guarded": model(
            repack_total,
            repack_miss,
            BANDWIDTH_EXACT,
            shared=FIXED_ALLOCATED_BYTES,
            expert=repack_expert,
            semantic="hypothetical_whole_record_16k_repack_not_executable",
        )
        | {"largest_object_guard_bytes": max(LARGEST_FIXED_OBJECT_BYTES, SOURCE_REPACK_STRIDE if source_count else K4_REPACK_STRIDE)},
        "explicit_allocation_omissions": {
            "embedding_rows_logical_bytes": EMBEDDING_ROWS_PER_Q8 * EMBEDDING_ROW_BYTES,
            "embedding_rows_allocated_bytes": None,
            "k4_tlut_logical_bytes": K4_TLUT_LOGICAL_BYTES,
            "k4_tlut_allocated_bytes": K4_TLUT_ALLOCATED_BYTES,
            "charged_to_current_layout": False,
        },
    }


def nearest_rank_p10(values: Sequence[float | None]) -> float | None:
    if len(values) != 32:
        raise ValueError("PW-0329 p10 requires exactly 32 windows")
    ordered = sorted(math.inf if value is None else float(value) for value in values)
    result = ordered[3]
    return None if math.isinf(result) else result


def aggregate_ledgers(windows: Sequence[dict[str, Any]], model_key: str) -> dict[str, Any]:
    if not windows or any(model_key not in window for window in windows):
        raise ValueError("aggregate requires complete ledgers")

    def summarize(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
        accepted = sum(int(row["A"]) for row in rows)
        moved = sum(int(row[model_key]["bytes_moved"]) for row in rows)
        bandwidths = {
            (
                int(row[model_key]["bandwidth_fraction"]["numerator"]),
                int(row[model_key]["bandwidth_fraction"]["denominator"]),
            )
            for row in rows
        }
        if rows and len(bandwidths) != 1:
            raise ValueError("aggregate bandwidth mismatch")
        bandwidth = Fraction(*next(iter(bandwidths))) if bandwidths else BANDWIDTH_EXACT
        exact_wall = _seconds(moved, bandwidth)
        wall = float(exact_wall)
        optimistic = None if exact_wall == 0 else Fraction(accepted, 1) / exact_wall
        return {
            "windows": len(rows),
            "accepted_tokens": accepted,
            "bytes_moved": moved,
            "storage_wall_seconds": wall,
            "storage_wall_fraction": {
                "numerator": exact_wall.numerator,
                "denominator": exact_wall.denominator,
            },
            "optimistic_storage_tps": None if optimistic is None else float(optimistic),
            "optimistic_storage_tps_fraction": (
                None
                if optimistic is None
                else {
                    "numerator": optimistic.numerator,
                    "denominator": optimistic.denominator,
                }
            ),
            "unbounded_storage_only": exact_wall == 0,
            "one_tps_headroom_seconds": accepted - wall,
            "one_tps_headroom_seconds_per_accepted_token": (
                None if accepted == 0 else (accepted - wall) / accepted
            ),
        }

    category = {
        name: summarize([window for window in windows if window["category"] == name])
        for name in CATEGORIES
    }
    summary = summarize(windows)
    if len(windows) == 32:
        exact_window_tps: list[Fraction | None] = []
        for window in windows:
            value = window[model_key]["optimistic_storage_tps_fraction"]
            exact_window_tps.append(
                None
                if value is None
                else Fraction(int(value["numerator"]), int(value["denominator"]))
            )
        ordered = sorted(
            exact_window_tps,
            key=lambda value: (value is None, Fraction() if value is None else value),
        )
        p10_exact = ordered[3]
        summary["nearest_rank_p10_window_optimistic_storage_tps"] = (
            None if p10_exact is None else float(p10_exact)
        )
        summary["nearest_rank_p10_window_optimistic_storage_tps_fraction"] = (
            None
            if p10_exact is None
            else {"numerator": p10_exact.numerator, "denominator": p10_exact.denominator}
        )
    else:
        summary["nearest_rank_p10_window_optimistic_storage_tps"] = None
        summary["nearest_rank_p10_window_optimistic_storage_tps_fraction"] = None
    return {"overall": summary, "category": category}


def _current_miss_for_window(window: Window, selected: frozenset[Identity], residency_bytes: int) -> int:
    k4_count, source_count, _ = _placement_counts(window, selected=selected)
    total = FIXED_ALLOCATED_BYTES + k4_count * K4_SCHEMA2_STRIDE + source_count * SOURCE_SCHEMA2_STRIDE
    largest = max(LARGEST_FIXED_OBJECT_BYTES, SOURCE_SCHEMA2_STRIDE if source_count else K4_SCHEMA2_STRIDE)
    return guarded_miss(total, residency_bytes, largest)


def _row_density_histogram(rows: Sequence[RouteRow], selected: frozenset[Identity]) -> dict[str, int]:
    hits = Counter(sum(identity in selected for identity in row.identities) for row in rows)
    return {str(value): hits.get(value, 0) for value in range(9)}


def _coverage(windows: Sequence[Window], rows: Sequence[RouteRow], selected: frozenset[Identity]) -> dict[str, Any]:
    layer_total: Counter[int] = Counter()
    layer_selected: Counter[int] = Counter()
    category_total: Counter[str] = Counter()
    category_selected: Counter[str] = Counter()
    for window in windows:
        for layer, identities in window.layer_unions:
            layer_total[layer] += len(identities)
            layer_selected[layer] += sum(identity in selected for identity in identities)
            category_total[window.category] += len(identities)
            category_selected[window.category] += sum(identity in selected for identity in identities)
    return {
        "row_density_histogram": _row_density_histogram(rows, selected),
        "layer": {
            str(layer): {
                "selected_identity_window_occurrences": layer_selected[layer],
                "total_identity_window_occurrences": layer_total[layer],
                "fraction": layer_selected[layer] / layer_total[layer],
            }
            for layer in ROUTED_LAYERS
        },
        "category": {
            category: {
                "selected_identity_window_occurrences": category_selected[category],
                "total_identity_window_occurrences": category_total[category],
                "fraction": category_selected[category] / category_total[category],
            }
            for category in CATEGORIES
        },
    }


@dataclass(frozen=True)
class SelectorResult:
    density: int
    residency_bytes: int
    target_tps: float
    order: tuple[Identity, ...]
    selection_order_sha256: str
    stop_reason: str
    initial_category_deficit_seconds: dict[str, float]
    remaining_category_deficit_seconds: dict[str, float]
    rejected_by_row_cap: tuple[dict[str, Any], ...]
    independent_recomputation_pass: bool
    metrics: dict[str, Any]
    windows: tuple[dict[str, Any], ...]
    coverage: dict[str, Any]
    installed_hybrid_expert_bank_bytes: int
    all_source_expert_bank_bytes: int
    construction_artifact_bytes: int
    estimated_m1_construction_seconds: int

    @property
    def selected(self) -> frozenset[Identity]:
        return frozenset(self.order)


def _selector(
    windows: Sequence[Window],
    rows: Sequence[RouteRow],
    *,
    density: int,
    residency_bytes: int,
    target_tps: float,
    maximum_identities: int = MAXIMUM_BANK_IDENTITIES,
    optimized: bool,
) -> SelectorResult:
    if density not in DENSITIES or residency_bytes < 0 or target_tps <= 0:
        raise ValueError("invalid selector scenario")
    if maximum_identities < 0 or maximum_identities > MAXIMUM_BANK_IDENTITIES:
        raise ValueError("selector cap exceeds 4096")
    validate_corpus_shape(windows, rows) if len(windows) == 32 else None
    window_by_index = {window.window_index: window for window in windows}
    if len(window_by_index) != len(windows):
        raise ValueError("duplicate selector window")
    universe = tuple(sorted({identity for window in windows for identity in window.identities}))
    occurrence_windows: dict[Identity, tuple[int, ...]] = {}
    for identity in universe:
        occurrence_windows[identity] = tuple(
            sorted(window.window_index for window in windows if identity in window.identities)
        )
    row_membership: dict[Identity, list[int]] = defaultdict(list)
    for row_index, row in enumerate(rows):
        for identity in row.identities:
            row_membership[identity].append(row_index)

    selected: set[Identity] = set()
    order: list[Identity] = []
    row_hits = [0] * len(rows)
    current_miss = {
        window.window_index: _current_miss_for_window(window, frozenset(), residency_bytes)
        for window in windows
    }
    target = _fraction(target_tps)

    def deficits(misses: Mapping[int, int]) -> dict[str, Fraction]:
        result: dict[str, Fraction] = {}
        for category in CATEGORIES:
            category_windows = [window for window in windows if window.category == category]
            exact_wall = sum((_seconds(misses[window.window_index], BANDWIDTH_EXACT) for window in category_windows), Fraction())
            allowed = Fraction(sum(window.accepted_tokens for window in category_windows), 1) / target
            result[category] = max(Fraction(), exact_wall - allowed)
        return result

    initial = deficits(current_miss)
    current_deficits = dict(initial)
    rejected: dict[Identity, dict[str, Any]] = {}
    stop_reason = "unknown"

    while True:
        if all(value == 0 for value in current_deficits.values()):
            stop_reason = "all_category_deficits_closed"
            break
        if len(order) >= maximum_identities:
            stop_reason = "maximum_4096_identity_cap"
            break
        best_identity: Identity | None = None
        best_score = Fraction()
        best_updates: dict[int, int] = {}

        for identity in universe:
            if identity in selected:
                continue
            blocking = [index for index in row_membership.get(identity, ()) if row_hits[index] >= density]
            if blocking:
                if identity not in rejected:
                    rejected[identity] = {
                        **_identity_record(identity),
                        "first_rejected_after_selected": len(order),
                        "blocking_rows": blocking,
                        "blocking_route_rows": [
                            _route_row_record(rows, index) for index in blocking
                        ],
                    }
                continue
            candidate_selected = frozenset(selected | {identity})
            affected = occurrence_windows[identity]
            updates: dict[int, int] = {}
            if optimized:
                for index in affected:
                    updates[index] = _current_miss_for_window(window_by_index[index], candidate_selected, residency_bytes)
            else:
                for window in windows:
                    updates[window.window_index] = _current_miss_for_window(window, candidate_selected, residency_bytes)

            reductions: dict[str, Fraction] = defaultdict(Fraction)
            check_indices = affected if optimized else tuple(window_by_index)
            for index in check_indices:
                old = current_miss[index]
                new = updates[index]
                if new > old:
                    raise AssertionError("K4 substitution increased guarded miss")
                category = window_by_index[index].category
                reductions[category] += _seconds(old - new, BANDWIDTH_EXACT)
            score = Fraction()
            for category in CATEGORIES:
                if current_deficits[category] == 0 or initial[category] == 0:
                    continue
                score += min(reductions[category], current_deficits[category]) / initial[category]
            if score > best_score or (score == best_score and score > 0 and (best_identity is None or identity < best_identity)):
                best_identity = identity
                best_score = score
                best_updates = updates

        if best_identity is None or best_score <= 0:
            stop_reason = "no_feasible_positive_score_identity"
            break
        selected.add(best_identity)
        order.append(best_identity)
        newly_saturated: list[int] = []
        for row_index in row_membership.get(best_identity, ()):
            row_hits[row_index] += 1
            if row_hits[row_index] > density:
                raise AssertionError("selector violated row density")
            if row_hits[row_index] == density:
                newly_saturated.append(row_index)
        for identity in universe:
            if identity in selected or identity in rejected:
                continue
            if not any(row_index in row_membership.get(identity, ()) for row_index in newly_saturated):
                continue
            blocking = [
                row_index
                for row_index in row_membership.get(identity, ())
                if row_hits[row_index] >= density
            ]
            rejected[identity] = {
                **_identity_record(identity),
                "first_rejected_after_selected": len(order),
                "blocking_rows": blocking,
                "blocking_route_rows": [
                    _route_row_record(rows, index) for index in blocking
                ],
            }
        if optimized:
            current_miss.update(best_updates)
        else:
            current_miss = best_updates
        current_deficits = deficits(current_miss)

    frozen = frozenset(selected)
    independent_miss = {
        window.window_index: _current_miss_for_window(window, frozen, residency_bytes)
        for window in windows
    }
    recomputed_deficits = deficits(independent_miss)
    row_histogram = _row_density_histogram(rows, frozen)
    independent_pass = (
        independent_miss == current_miss
        and recomputed_deficits == current_deficits
        and max((int(key) for key, count in row_histogram.items() if count), default=0) <= density
        and len(frozen) == len(order)
    )
    ledgers = [
        window_storage_ledger(window, residency_bytes=residency_bytes, selected=frozen, rows=rows)
        for window in windows
    ]
    metrics = {key: aggregate_ledgers(ledgers, key) for key in (
        "fractional_relaxed", "exact_logical", "current_layout_guarded", "hypothetical_repack_guarded"
    )}
    all_source = ALL_IDENTITY_COUNT * SOURCE_SCHEMA2_STRIDE
    installed = len(order) * K4_SCHEMA2_STRIDE + (ALL_IDENTITY_COUNT - len(order)) * SOURCE_SCHEMA2_STRIDE
    return SelectorResult(
        density=density,
        residency_bytes=residency_bytes,
        target_tps=float(target),
        order=tuple(order),
        selection_order_sha256=selection_order_sha256(order),
        stop_reason=stop_reason,
        initial_category_deficit_seconds={key: float(value) for key, value in initial.items()},
        remaining_category_deficit_seconds={key: float(value) for key, value in recomputed_deficits.items()},
        rejected_by_row_cap=tuple(rejected[key] for key in sorted(rejected)),
        independent_recomputation_pass=independent_pass,
        metrics=metrics,
        windows=tuple(ledgers),
        coverage=_coverage(windows, rows, frozen),
        installed_hybrid_expert_bank_bytes=installed,
        all_source_expert_bank_bytes=all_source,
        construction_artifact_bytes=len(order) * ARTIFACT_BYTES_PER_IDENTITY,
        estimated_m1_construction_seconds=len(order) * M1_SECONDS_PER_IDENTITY,
    )


def select_fixed_bank(
    windows: Sequence[Window],
    rows: Sequence[RouteRow],
    *,
    density: int,
    residency_bytes: int,
    target_tps: float,
    maximum_identities: int = MAXIMUM_BANK_IDENTITIES,
) -> SelectorResult:
    return _selector_optimized(
        windows,
        rows,
        density=density,
        residency_bytes=residency_bytes,
        target_tps=target_tps,
        maximum_identities=maximum_identities,
    )


def _selector_optimized(
    windows: Sequence[Window],
    rows: Sequence[RouteRow],
    *,
    density: int,
    residency_bytes: int,
    target_tps: float,
    maximum_identities: int = MAXIMUM_BANK_IDENTITIES,
) -> SelectorResult:
    """Indexed exact selector; a vector screen never decides the winner."""
    if density not in DENSITIES or residency_bytes < 0 or target_tps <= 0:
        raise ValueError("invalid selector scenario")
    if maximum_identities < 0 or maximum_identities > MAXIMUM_BANK_IDENTITIES:
        raise ValueError("selector cap exceeds 4096")
    validate_corpus_shape(windows, rows) if len(windows) == 32 else None
    if len({window.window_index for window in windows}) != len(windows):
        raise ValueError("duplicate selector window")

    ordered_windows = tuple(sorted(windows, key=lambda item: item.window_index))
    window_position = {
        window.window_index: position for position, window in enumerate(ordered_windows)
    }
    universe = tuple(sorted({identity for window in ordered_windows for identity in window.identities}))
    identity_index = {identity: index for index, identity in enumerate(universe)}
    candidate_count = len(universe)
    category_index = {category: index for index, category in enumerate(CATEGORIES)}
    window_candidates: list[np.ndarray] = []
    occurrence_windows: list[list[int]] = [[] for _ in universe]
    for position, window in enumerate(ordered_windows):
        indices = np.asarray(
            [identity_index[identity] for identity in sorted(window.identities)],
            dtype=np.int32,
        )
        window_candidates.append(indices)
        for index in indices:
            occurrence_windows[int(index)].append(position)

    row_candidates: list[tuple[int, ...]] = []
    row_membership: list[list[int]] = [[] for _ in universe]
    for row_index, row in enumerate(rows):
        indices = tuple(identity_index[identity] for identity in row.identities)
        row_candidates.append(indices)
        for index in indices:
            row_membership[index].append(row_index)

    base_total = np.asarray(
        [
            FIXED_ALLOCATED_BYTES
            + sum(len(identities) for _layer, identities in window.layer_unions)
            * SOURCE_SCHEMA2_STRIDE
            for window in ordered_windows
        ],
        dtype=np.int64,
    )
    selected_per_window = np.zeros(len(ordered_windows), dtype=np.int32)

    def miss_at(position: int, selected_count: int) -> int:
        total = int(base_total[position]) - selected_count * SCHEMA2_SAVING
        return guarded_miss(total, residency_bytes, LARGEST_FIXED_OBJECT_BYTES)

    current_miss_array = np.asarray(
        [miss_at(position, 0) for position in range(len(ordered_windows))],
        dtype=np.int64,
    )
    marginal = np.asarray(
        [
            int(current_miss_array[position]) - miss_at(position, 1)
            for position in range(len(ordered_windows))
        ],
        dtype=np.int64,
    )
    reductions = np.zeros((len(CATEGORIES), candidate_count), dtype=np.int64)
    for position, window in enumerate(ordered_windows):
        reductions[category_index[window.category], window_candidates[position]] += marginal[position]

    target = _fraction(target_tps)

    def deficit_vector() -> dict[str, Fraction]:
        result: dict[str, Fraction] = {}
        for category in CATEGORIES:
            positions = [
                position
                for position, window in enumerate(ordered_windows)
                if window.category == category
            ]
            wall = sum(
                (_seconds(int(current_miss_array[position]), BANDWIDTH_EXACT) for position in positions),
                Fraction(),
            )
            accepted = sum(ordered_windows[position].accepted_tokens for position in positions)
            allowed = Fraction(accepted, 1) / target
            result[category] = max(Fraction(), wall - allowed)
        return result

    initial = deficit_vector()
    current_deficits = dict(initial)
    selected_mask = np.zeros(candidate_count, dtype=bool)
    blocked_mask = np.zeros(candidate_count, dtype=bool)
    row_hits = np.zeros(len(rows), dtype=np.int8)
    rejected: dict[int, dict[str, Any]] = {}
    order_indices: list[int] = []
    stop_reason = "unknown"

    def exact_score(index: int) -> Fraction:
        score = Fraction()
        for category, category_position in category_index.items():
            if current_deficits[category] == 0 or initial[category] == 0:
                continue
            reduction = _seconds(int(reductions[category_position, index]), BANDWIDTH_EXACT)
            score += min(reduction, current_deficits[category]) / initial[category]
        return score

    while True:
        if all(value == 0 for value in current_deficits.values()):
            stop_reason = "all_category_deficits_closed"
            break
        if len(order_indices) >= maximum_identities:
            stop_reason = "maximum_4096_identity_cap"
            break

        scores = np.zeros(candidate_count, dtype=np.float64)
        for category, position in category_index.items():
            if current_deficits[category] == 0 or initial[category] == 0:
                continue
            reduction_seconds = reductions[position].astype(np.float64) / BANDWIDTH_EXACT_FLOAT
            scores += np.minimum(reduction_seconds, float(current_deficits[category])) / float(initial[category])
        scores[selected_mask | blocked_mask] = -np.inf
        best_float = float(np.max(scores)) if candidate_count else -math.inf
        if not math.isfinite(best_float) or best_float <= 0.0:
            stop_reason = "no_feasible_positive_score_identity"
            break
        tolerance = max(1e-15, abs(best_float) * 1e-12)
        finalists = np.flatnonzero(scores >= best_float - tolerance)
        best_index: int | None = None
        best_exact = Fraction()
        for raw_index in finalists:
            index = int(raw_index)
            value = exact_score(index)
            if value > best_exact or (
                value == best_exact
                and value > 0
                and (best_index is None or universe[index] < universe[best_index])
            ):
                best_index = index
                best_exact = value
        if best_index is None or best_exact <= 0:
            raise AssertionError("vector selector screen lost a positive exact candidate")
        outside = scores < best_float - tolerance
        if np.any(outside) and float(np.max(scores[outside])) > float(best_exact) + tolerance:
            raise AssertionError("vector selector screen excluded the exact winner")

        selected_mask[best_index] = True
        order_indices.append(best_index)
        newly_saturated: list[int] = []
        for row_index in row_membership[best_index]:
            row_hits[row_index] += 1
            if int(row_hits[row_index]) > density:
                raise AssertionError("selector violated row density")
            if int(row_hits[row_index]) == density:
                newly_saturated.append(row_index)
        for row_index in newly_saturated:
            for candidate_index in row_candidates[row_index]:
                if selected_mask[candidate_index] or blocked_mask[candidate_index]:
                    continue
                blocking = [
                    member
                    for member in row_membership[candidate_index]
                    if int(row_hits[member]) >= density
                ]
                blocked_mask[candidate_index] = True
                rejected[candidate_index] = {
                    **_identity_record(universe[candidate_index]),
                    "first_rejected_after_selected": len(order_indices),
                    "blocking_rows": blocking,
                    "blocking_route_rows": [
                        _route_row_record(rows, index) for index in blocking
                    ],
                }

        for position in occurrence_windows[best_index]:
            old_marginal = int(marginal[position])
            old_miss = int(current_miss_array[position])
            selected_per_window[position] += 1
            new_count = int(selected_per_window[position])
            new_miss = miss_at(position, new_count)
            if new_miss != old_miss - old_marginal:
                raise AssertionError("incremental guarded miss mismatch")
            current_miss_array[position] = new_miss
            if new_count < len(window_candidates[position]):
                new_marginal = new_miss - miss_at(position, new_count + 1)
            else:
                new_marginal = 0
            marginal[position] = new_marginal
            delta = int(new_marginal) - old_marginal
            if delta:
                category_position = category_index[ordered_windows[position].category]
                reductions[category_position, window_candidates[position]] += delta
        current_deficits = deficit_vector()

    order = tuple(universe[index] for index in order_indices)
    frozen = frozenset(order)
    current_miss = {
        window.window_index: int(current_miss_array[position])
        for position, window in enumerate(ordered_windows)
    }
    independent_miss = {
        window.window_index: _current_miss_for_window(window, frozen, residency_bytes)
        for window in ordered_windows
    }
    recomputed_deficits: dict[str, Fraction] = {}
    for category in CATEGORIES:
        category_windows = [window for window in ordered_windows if window.category == category]
        wall = sum(
            (_seconds(independent_miss[window.window_index], BANDWIDTH_EXACT) for window in category_windows),
            Fraction(),
        )
        allowed = Fraction(sum(window.accepted_tokens for window in category_windows), 1) / target
        recomputed_deficits[category] = max(Fraction(), wall - allowed)
    row_histogram = _row_density_histogram(rows, frozen)
    independent_pass = (
        independent_miss == current_miss
        and recomputed_deficits == current_deficits
        and max((int(key) for key, count in row_histogram.items() if count), default=0) <= density
        and len(frozen) == len(order)
    )
    ledgers = [
        window_storage_ledger(window, residency_bytes=residency_bytes, selected=frozen, rows=rows)
        for window in ordered_windows
    ]
    metrics = {
        key: aggregate_ledgers(ledgers, key)
        for key in (
            "fractional_relaxed",
            "exact_logical",
            "current_layout_guarded",
            "hypothetical_repack_guarded",
        )
    }
    all_source = ALL_IDENTITY_COUNT * SOURCE_SCHEMA2_STRIDE
    installed = len(order) * K4_SCHEMA2_STRIDE + (ALL_IDENTITY_COUNT - len(order)) * SOURCE_SCHEMA2_STRIDE
    return SelectorResult(
        density=density,
        residency_bytes=residency_bytes,
        target_tps=float(target),
        order=order,
        selection_order_sha256=selection_order_sha256(order),
        stop_reason=stop_reason,
        initial_category_deficit_seconds={key: float(value) for key, value in initial.items()},
        remaining_category_deficit_seconds={key: float(value) for key, value in recomputed_deficits.items()},
        rejected_by_row_cap=tuple(rejected[key] for key in sorted(rejected, key=lambda item: universe[item])),
        independent_recomputation_pass=independent_pass,
        metrics=metrics,
        windows=tuple(ledgers),
        coverage=_coverage(ordered_windows, rows, frozen),
        installed_hybrid_expert_bank_bytes=installed,
        all_source_expert_bank_bytes=all_source,
        construction_artifact_bytes=len(order) * ARTIFACT_BYTES_PER_IDENTITY,
        estimated_m1_construction_seconds=len(order) * M1_SECONDS_PER_IDENTITY,
    )


def select_fixed_bank_naive(
    windows: Sequence[Window],
    rows: Sequence[RouteRow],
    *,
    density: int,
    residency_bytes: int,
    target_tps: float,
    maximum_identities: int = MAXIMUM_BANK_IDENTITIES,
) -> SelectorResult:
    """Independent full-window reference used only by deterministic fixtures."""
    return _selector(
        windows,
        rows,
        density=density,
        residency_bytes=residency_bytes,
        target_tps=target_tps,
        maximum_identities=maximum_identities,
        optimized=False,
    )


def relaxed_scenario(
    windows: Sequence[Window],
    rows: Sequence[RouteRow],
    *,
    density: int,
    residency_bytes: int,
) -> dict[str, Any]:
    ledgers = [
        window_storage_ledger(window, residency_bytes=residency_bytes, density=density, rows=rows)
        for window in windows
    ]
    return {
        "density": density,
        "residency_bytes": residency_bytes,
        "residency_gib": residency_bytes / GIB,
        "placement": "relaxed_per_window_per_layer_fractional_density_ceiling",
        "metrics": {key: aggregate_ledgers(ledgers, key) for key in (
            "fractional_relaxed", "exact_logical", "current_layout_guarded", "hypothetical_repack_guarded"
        )},
        "windows": ledgers,
    }


def _strict_metrics_pass(metrics: Mapping[str, Any], threshold: float = 1.0) -> bool:
    return (
        _summary_tps_compare(metrics["overall"], threshold, strict=True)
        and all(
            _summary_tps_compare(metrics["category"][name], threshold, strict=True)
            for name in CATEGORIES
        )
        and _p10_compare(metrics["overall"], threshold, strict=True)
    )


def density_survival_summary(
    relaxed: Mapping[tuple[int, int], Mapping[str, Any]]
) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    for density in (4, 5, 6, 8):
        metrics = relaxed[(density, 12 * GIB)]["metrics"]["fractional_relaxed"]
        overall = _summary_tps_compare(metrics["overall"], 1.0, strict=True)
        categories = all(
            _summary_tps_compare(metrics["category"][category], 1.0, strict=True)
            for category in CATEGORIES
        )
        tail = _p10_compare(metrics["overall"], 1.0, strict=True)
        rows[str(density)] = {
            "strict_overall_above_one": overall,
            "strict_every_category_above_one": categories,
            "strict_fourth_lowest_p10_above_one": tail,
            "strict_all": overall and categories and tail,
        }
    return {
        "R_gib": 12,
        "by_density": rows,
        "earliest_density_by_gate": {
            gate: next(
                (
                    density
                    for density in (4, 5, 6)
                    if rows[str(density)][gate]
                ),
                None,
            )
            for gate in (
                "strict_overall_above_one",
                "strict_every_category_above_one",
                "strict_fourth_lowest_p10_above_one",
                "strict_all",
            )
        },
        "density8_diagnostic_only": True,
    }


def disposition(
    relaxed: Mapping[tuple[int, int], Mapping[str, Any]],
    banks: Sequence[SelectorResult],
    *,
    authority_complete: bool,
    gate8_pass: bool,
) -> dict[str, Any]:
    """Apply the frozen PW-0329 gates in their exact precedence order."""
    strongest = relaxed[(8, 12 * GIB)]["metrics"]["fractional_relaxed"]
    if not _summary_tps_compare(strongest["overall"], 1.0, strict=True) or any(
        not _summary_tps_compare(strongest["category"][name], 1.0, strict=True)
        for name in CATEGORIES
    ):
        return {
            "decision": "reject_current_k4_portfolio_absolute_corpus_byte_floor",
            "workflow_disposition": "rejected",
            "precedence_gate": 1,
            "work_order": None,
            "reason": "strongest relaxed d8/R12 overall or required-category storage ceiling is at or below one TPS",
        }
    if not _p10_compare(strongest["overall"], 1.0, strict=True):
        return {
            "decision": "reject_k4_construction_continuation_on_tail",
            "workflow_disposition": "rejected",
            "precedence_gate": 2,
            "work_order": None,
            "reason": "strongest relaxed d8/R12 fourth-lowest window ceiling is at or below one TPS",
        }

    density_pass = {
        density: _strict_metrics_pass(relaxed[(density, 12 * GIB)]["metrics"]["fractional_relaxed"])
        for density in (4, 5, 6)
    }
    earliest_density = next((density for density in (4, 5, 6) if density_pass[density]), None)
    if not density_pass[6]:
        return {
            "decision": "close_currently_evidenced_k4_route_density_seven_or_eight_only",
            "workflow_disposition": "rejected",
            "precedence_gate": 3,
            "work_order": None,
            "reason": "d8 survives but relaxed d6/R12 fails a strict corpus, category, or tail storage gate",
            "earliest_relaxed_surviving_density": earliest_density,
        }

    eligible: list[SelectorResult] = []
    for bank in banks:
        if bank.density > 6 or not math.isclose(bank.target_tps, 1.25, rel_tol=0.0, abs_tol=0.0):
            continue
        guarded = bank.metrics["current_layout_guarded"]
        fixed_fractional = bank.metrics["fractional_relaxed"]
        if (
            len(bank.order) <= MAXIMUM_BANK_IDENTITIES
            and _summary_tps_compare(guarded["overall"], 1.25, strict=False)
            and all(
                _summary_tps_compare(guarded["category"][name], 1.25, strict=False)
                for name in CATEGORIES
            )
            and _p10_compare(fixed_fractional["overall"], 1.0, strict=True)
            and bank.installed_hybrid_expert_bank_bytes < bank.all_source_expert_bank_bytes
            and bank.independent_recomputation_pass
            and all(value == 0.0 for value in bank.remaining_category_deficit_seconds.values())
            and authority_complete
            and gate8_pass
        ):
            eligible.append(bank)
    if not eligible:
        return {
            "decision": "retain_conditional_analytical_survivor_no_work_order",
            "workflow_disposition": "conditional",
            "precedence_gate": 4,
            "work_order": None,
            "reason": "relaxed storage survives but no row-feasible bank closes the exact 1.25 guarded prerequisites",
            "earliest_relaxed_surviving_density": earliest_density,
        }
    chosen = min(eligible, key=lambda item: (item.density, item.residency_bytes, len(item.order), item.selection_order_sha256))
    pressure_requalification = chosen.residency_bytes > 8 * GIB
    if chosen.density == 3:
        work = "authorize_only_q8_native_mtp_acceptance_then_arbitrary_identity_panel_then_mixed_layer"
        staged_falsifier = {
            "first": "native-MTP q8 acceptance and latency on PW-0328 histories",
            "second": "early/middle/deep arbitrary-identity fidelity panel",
            "third": "production-shaped mixed q8 layer",
        }
    elif chosen.density in (4, 5):
        work = "authorize_only_byte_neutral_rank1_correction_falsifier"
        staged_falsifier = {
            "starting_expert": 96,
            "frozen_route": [96, 64, 232, 31, 88, 245, 223, 151],
            "unchanged_exclusive_relative_l2_gate": 0.01,
            "four_of_eight_first": True,
            "separate_five_of_eight_row_after_four_pass": chosen.density == 5,
        }
    else:
        work = "authorize_only_bounded_six_of_eight_28_subset_falsifier"
        staged_falsifier = {
            "frozen_route": [96, 64, 232, 31, 88, 245, 223, 151],
            "missing_identities_to_complete": [88, 245, 223, 151],
            "six_of_eight_subsets": 28,
            "unchanged_exclusive_relative_l2_gate": 0.01,
        }
    return {
        "decision": work,
        "workflow_disposition": "conditional",
        "precedence_gate": 5 if chosen.density == 3 else 6 if chosen.density in (4, 5) else 7,
        "work_order": {
            "density": chosen.density,
            "residency_bytes": chosen.residency_bytes,
            "selected_identities": len(chosen.order),
            "selection_order_sha256": chosen.selection_order_sha256,
            "physical_authorization": False,
            "pressure_requalification_required": pressure_requalification,
            "five_of_eight_row_required_after_four_row": chosen.density == 5,
            "six_of_eight_subsets_required": 28 if chosen.density == 6 else None,
            "staged_falsifier": staged_falsifier,
        },
        "reason": "one row-feasible bank closes the frozen analytical 1.25 prerequisite; construction remains excluded",
        "earliest_relaxed_surviving_density": earliest_density,
    }


def selector_result_dict(result: SelectorResult) -> dict[str, Any]:
    value = asdict(result)
    value["order"] = [_identity_record(item) for item in result.order]
    value["selected_identities"] = len(result.order)
    value["diagnostic_only_density8"] = result.density == 8
    return value


def derive_schema2_layout(records: Any) -> dict[str, Any]:
    """Derive logical and executable record sizes without trusting constants."""
    if not isinstance(records, list) or not records:
        raise ValueError("PW-0318 schema-2 record cardinality mismatch")
    logical_by_format: dict[str, set[int]] = defaultdict(set)
    stride_by_format: dict[str, set[int]] = defaultdict(set)
    correction_payloads: list[dict[str, Any]] = []
    for record in records:
        format_name = record.get("format")
        if format_name == "qtip_k4_ldlq":
            projections = record.get("projections")
            if not isinstance(projections, dict) or set(projections) != {"gate", "up", "down"}:
                raise ValueError("PW-0318 K4 projection set mismatch")
            payloads = []
            for projection in projections.values():
                if projection.get("rank") != 1:
                    raise ValueError("PW-0318 correction rank is not one")
                panel = projection.get("payloads")
                if not isinstance(panel, dict):
                    raise ValueError("PW-0318 K4 payload panel mismatch")
                payloads.extend(panel.values())
                for role in ("correction_left", "correction_right"):
                    payload = panel.get(role)
                    if not isinstance(payload, dict):
                        raise ValueError("PW-0318 correction payload missing")
                    correction_payloads.append(payload)
        elif format_name == "source_fp8_e4m3_block128":
            panel = record.get("payloads")
            if not isinstance(panel, dict) or set(panel) != {
                "gate_weight", "gate_scales", "up_weight", "up_scales", "down_weight", "down_scales"
            }:
                raise ValueError("PW-0318 source payload panel mismatch")
            payloads = list(panel.values())
        else:
            raise ValueError("PW-0318 unknown record format")
        sizes = [int(payload["bytes"]) for payload in payloads]
        if any(int(payload.get("alignment", -1)) != ALIGNMENT_BYTES for payload in payloads):
            raise ValueError("PW-0318 payload alignment mismatch")
        logical_by_format[format_name].add(sum(sizes))
        stride_by_format[format_name].add(sum(whole_record_stride(size) for size in sizes))
    if logical_by_format != {
        "qtip_k4_ldlq": {K4_LOGICAL_BYTES},
        "source_fp8_e4m3_block128": {SOURCE_LOGICAL_BYTES},
    }:
        raise ValueError("PW-0318 logical record size mismatch")
    if stride_by_format != {
        "qtip_k4_ldlq": {K4_SCHEMA2_STRIDE},
        "source_fp8_e4m3_block128": {SOURCE_SCHEMA2_STRIDE},
    }:
        raise ValueError("PW-0318 current record stride mismatch")
    return {
        "logical_by_format": {key: next(iter(value)) for key, value in logical_by_format.items()},
        "stride_by_format": {key: next(iter(value)) for key, value in stride_by_format.items()},
        "correction_payloads": correction_payloads,
        "k4_records": sum(record["format"] == "qtip_k4_ldlq" for record in records),
        "source_records": sum(record["format"] == "source_fp8_e4m3_block128" for record in records),
    }


def verify_zero_correction_payloads(
    payloads: Sequence[Mapping[str, Any]], read_payload: Any
) -> None:
    """Authenticate both the declared digest and actual bytes of every factor."""
    for payload in payloads:
        count = int(payload["bytes"])
        if payload.get("sha256") != hashlib.sha256(bytes(count)).hexdigest():
            raise ValueError("PW-0318 correction hash is not all-zero F16")
        actual = read_payload(payload)
        if not isinstance(actual, bytes) or len(actual) != count or actual != bytes(count):
            raise ValueError("PW-0318 correction payload is not all zero")


def validate_schema2_authority(manifest_path: Path, bundle_path: Path) -> dict[str, Any]:
    if sha256_file(manifest_path) != PW0318_MANIFEST_SHA256:
        raise ValueError("PW-0318 schema-2 manifest mismatch")
    if bundle_path.stat().st_size != 164_724_736 or sha256_file(bundle_path) != PW0318_BUNDLE_SHA256:
        raise ValueError("PW-0318 schema-2 bundle mismatch")
    manifest = json.loads(manifest_path.read_text())
    if (
        manifest.get("schema_version") != 2
        or manifest.get("alignment_bytes") != ALIGNMENT_BYTES
        or manifest.get("bundle_bytes") != 164_724_736
        or manifest.get("bundle_sha256") != PW0318_BUNDLE_SHA256
        or manifest.get("semantic") != "prismwing_mixed_k4_source_layer_bundle_v2"
    ):
        raise ValueError("PW-0318 schema-2 semantic mismatch")
    records = manifest.get("records")
    if not isinstance(records, list) or len(records) != 8:
        raise ValueError("PW-0318 schema-2 record cardinality mismatch")
    derived = derive_schema2_layout(records)
    tlut = manifest.get("tlut")
    if (
        not isinstance(tlut, dict)
        or tlut.get("bytes") != K4_TLUT_LOGICAL_BYTES
        or tlut.get("alignment") != ALIGNMENT_BYTES
        or tlut.get("offset") != 0
        or derived["k4_records"] != 3
        or derived["source_records"] != 5
        or (
            derived["k4_records"] * K4_SCHEMA2_STRIDE
            + derived["source_records"] * SOURCE_SCHEMA2_STRIDE
            + K4_TLUT_ALLOCATED_BYTES
        )
        != manifest["bundle_bytes"]
    ):
        raise ValueError("PW-0318 schema-2 TLUT/allocation closure mismatch")
    with bundle_path.open("rb") as bundle:
        def read_payload(payload: Mapping[str, Any]) -> bytes:
            count = int(payload["bytes"])
            offset = int(payload["offset"])
            bundle.seek(offset)
            return bundle.read(count)

        verify_zero_correction_payloads(derived["correction_payloads"], read_payload)
    if whole_record_stride(K4_LOGICAL_BYTES) != K4_REPACK_STRIDE or whole_record_stride(SOURCE_LOGICAL_BYTES) != SOURCE_REPACK_STRIDE:
        raise AssertionError("hypothetical repack derivation mismatch")
    return {
        "manifest_sha256": PW0318_MANIFEST_SHA256,
        "bundle_sha256": PW0318_BUNDLE_SHA256,
        "bundle_bytes": 164_724_736,
        "k4_records": derived["k4_records"],
        "source_records": derived["source_records"],
        "k4_logical_bytes": K4_LOGICAL_BYTES,
        "source_logical_bytes": SOURCE_LOGICAL_BYTES,
        "k4_schema2_stride": K4_SCHEMA2_STRIDE,
        "source_schema2_stride": SOURCE_SCHEMA2_STRIDE,
        "k4_hypothetical_repack_stride": K4_REPACK_STRIDE,
        "source_hypothetical_repack_stride": SOURCE_REPACK_STRIDE,
        "record_payload_logical_bytes": (
            derived["k4_records"] * K4_LOGICAL_BYTES
            + derived["source_records"] * SOURCE_LOGICAL_BYTES
        ),
        "tlut_logical_bytes": K4_TLUT_LOGICAL_BYTES,
        "tlut_allocated_bytes": K4_TLUT_ALLOCATED_BYTES,
        "allocated_bundle_closure": True,
        "rank_one_correction_payloads": len(derived["correction_payloads"]),
        "all_correction_payloads_zero_f16": True,
        "byte_neutral_replacement": True,
    }


def validate_correction_execution_path(repo: Path) -> dict[str, str]:
    rust = repo / "src/k4_source_metal.rs"
    loader = repo / "src/k4_source_bundle.rs"
    metal = repo / "kernels/qtip_k4_bundle_batched.metal"
    rust_text = rust.read_text()
    loader_text = loader.read_text()
    metal_text = metal.read_text()
    rust_needles = (
        'payload("correction_left")',
        'payload("correction_right")',
        "self.pipelines.low_rank_inputs",
        "self.pipelines.low_rank_shared",
        "self.pipelines.finish",
    )
    metal_needles = (
        "kernel void qtip_k4_bundle_low_rank_shared",
        "kernel void qtip_k4_bundle_low_rank_inputs",
        "kernel void qtip_k4_bundle_finish",
        "low_rank_right_output",
        "+ correction",
    )
    if (
        any(needle not in rust_text for needle in rust_needles)
        or any(needle not in metal_text for needle in metal_needles)
        or '"correction_left"' not in loader_text
        or '"correction_right"' not in loader_text
    ):
        raise ValueError("schema-2 correction execution path mismatch")
    return {
        "rust_execution_sha256": sha256_file(rust),
        "rust_loader_sha256": sha256_file(loader),
        "metal_sha256": sha256_file(metal),
    }


def validate_pw0325_order(path: Path) -> tuple[tuple[Identity, ...], dict[str, Any]]:
    if sha256_file(path) != PW0325_ANALYSIS_SHA256:
        raise ValueError("PW-0325 canonical analysis mismatch")
    report = json.loads(path.read_text())
    if report.get("experiment_id") != "PW-0325" or report.get("candidate_selection_order_sha256") != PW0325_ORDER_SHA256:
        raise ValueError("PW-0325 analysis semantic mismatch")
    matches = [
        item for item in report.get("scenarios", [])
        if item.get("selection_order_sha256") == PW0325_ORDER_SHA256
        and item.get("selected_identities") == 3925
    ]
    if len(matches) != 1:
        raise ValueError("PW-0325 canonical selection scenario mismatch")
    order = tuple((int(item["layer"]), int(item["expert"])) for item in matches[0].get("selection_order", []))
    if (
        len(order) != 3925
        or len(set(order)) != len(order)
        or any(layer not in ROUTED_LAYERS or not 0 <= expert < EXPERTS_PER_LAYER for layer, expert in order)
        or selection_order_sha256(order) != PW0325_ORDER_SHA256
    ):
        raise ValueError("PW-0325 canonical selection order mismatch")
    return order, {
        "analysis_sha256": PW0325_ANALYSIS_SHA256,
        "selection_order_sha256": PW0325_ORDER_SHA256,
        "selected_identities": len(order),
    }


def historical_replay(
    windows: Sequence[Window], rows: Sequence[RouteRow], order: Sequence[Identity]
) -> dict[str, Any]:
    selected = frozenset(order)
    scenarios = []
    for gib in RESIDENCY_GIB:
        ledgers = [
            window_storage_ledger(window, residency_bytes=gib * GIB, selected=selected, rows=rows)
            for window in windows
        ]
        scenarios.append({
            "residency_gib": gib,
            "metrics": {key: aggregate_ledgers(ledgers, key) for key in (
                "fractional_relaxed", "exact_logical", "current_layout_guarded", "hypothetical_repack_guarded"
            )},
            "windows": ledgers,
        })
    histogram = _row_density_histogram(rows, selected)
    return {
        "semantic": "historical_pw0325_order_replayed_on_pw0328_not_current_bank_authority",
        "selected_identities": len(order),
        "selection_order_sha256": selection_order_sha256(order),
        "row_density_histogram": histogram,
        "maximum_row_density": max(int(key) for key, value in histogram.items() if value),
        "installed_hybrid_expert_bank_bytes": (
            len(order) * K4_SCHEMA2_STRIDE
            + (ALL_IDENTITY_COUNT - len(order)) * SOURCE_SCHEMA2_STRIDE
        ),
        "all_source_expert_bank_bytes": ALL_IDENTITY_COUNT * SOURCE_SCHEMA2_STRIDE,
        "construction_artifact_bytes": len(order) * ARTIFACT_BYTES_PER_IDENTITY,
        "estimated_m1_construction_seconds": len(order) * M1_SECONDS_PER_IDENTITY,
        "scenarios": scenarios,
    }


def _validate_hash(path: Path, expected: str, label: str) -> dict[str, Any]:
    observed = sha256_file(path)
    if observed != expected:
        raise ValueError(f"{label} mismatch")
    return {"path": str(path), "sha256": observed}


def validate_additional_authorities(
    *,
    repo: Path,
    pw0318_summary: Path,
    pw0316_analysis: Path,
    pw0308_manifest: Path,
    pw0308_repeated: Path,
) -> dict[str, Any]:
    result = {
        "contract": _validate_hash(repo / CONTRACT_PATH, CONTRACT_SHA256, "PW-0329 contract"),
        "target": _validate_hash(repo / "TARGET.md", TARGET_SHA256, "TARGET.md"),
        "red_lines": _validate_hash(repo / "RED_LINES.md", RED_LINES_SHA256, "RED_LINES.md"),
        "pw0318_summary": _validate_hash(pw0318_summary, PW0318_SUMMARY_SHA256, "PW-0318 summary"),
        "pw0316_rejection": _validate_hash(pw0316_analysis, PW0316_REJECTION_SHA256, "PW-0316 rejection"),
        "pw0308_manifest": _validate_hash(pw0308_manifest, PW0308_MANIFEST_SHA256, "PW-0308 manifest"),
        "pw0308_repeated": _validate_hash(pw0308_repeated, PW0308_REPEATED_SHA256, "PW-0308 repeated result"),
    }
    git_blob = __import__("subprocess").run(
        ["git", "hash-object", CONTRACT_PATH], cwd=repo, check=True, text=True, capture_output=True
    ).stdout.strip()
    if git_blob != CONTRACT_GIT_BLOB:
        raise ValueError("PW-0329 contract Git blob mismatch")
    result["contract"]["git_blob"] = git_blob
    rejection = json.loads(pw0316_analysis.read_text())
    routed = (
        rejection.get("semantic", {})
        .get("route_candidate_vs_source", {})
        .get("relative_l2")
    )
    if not math.isclose(float(routed), 0.0109888419, rel_tol=0.0, abs_tol=5e-11):
        raise ValueError("PW-0316 rejected four/four row metric mismatch")
    manifest = json.loads(pw0308_manifest.read_text())
    if (
        manifest.get("experiment_id") != "PW-0308"
        or manifest.get("files", {}).get("repeated-47.json") != PW0308_REPEATED_SHA256
        or not math.isclose(float(manifest.get("result", {}).get("repeated_complete_call_p90_ms")), 351.680083, rel_tol=0.0, abs_tol=5e-10)
        or manifest.get("result", {}).get("repeated_components") != 47
    ):
        raise ValueError("PW-0308 diagnostic semantic mismatch")
    return result


def _import_shared_authorities() -> tuple[Any, Any]:
    try:
        from tools.pw0328_corpus_authority import authenticate_pw0328_corpus
    except ModuleNotFoundError:
        from pw0328_corpus_authority import authenticate_pw0328_corpus
    try:
        from tools.prismwing_storage_authority import authenticate_prismwing_storage
    except ModuleNotFoundError:
        from prismwing_storage_authority import authenticate_prismwing_storage
    return authenticate_pw0328_corpus, authenticate_prismwing_storage


def _normalize_identity(value: Any, layer: int) -> Identity:
    if isinstance(value, Mapping):
        item = (int(value.get("layer", layer)), int(value["expert"]))
    elif isinstance(value, (tuple, list)) and len(value) == 2:
        item = (int(value[0]), int(value[1]))
    else:
        item = (layer, int(value))
    if item[0] != layer:
        raise ValueError("shared loader identity layer mismatch")
    return item


def normalize_pw0328_authority(authority: Mapping[str, Any]) -> tuple[list[Window], list[RouteRow]]:
    """Adapt only the frozen shared-loader schema, failing closed on drift."""
    if authority.get("manifest_sha256") != PW0328_MANIFEST_SHA256 or authority.get("builder_commit") != PW0328_BUILDER_COMMIT:
        raise ValueError("shared PW-0328 authority mismatch")
    control = authority.get("control", {})
    if (
        authority.get("artifact_count") != 24
        or len(authority.get("artifacts", [])) != 24
        or set(authority.get("categories", [])) != set(CATEGORIES)
        or len(authority.get("q1_events", [])) != 232
        or control.get("windows") != 32
        or control.get("sum_A") != 232
        or control.get("sum_observable_A") != 231
        or not math.isclose(
            float(control.get("sum_U", -1)),
            142.71808510638297,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or authority.get("builder_gate8", {}).get("pass") is not True
    ):
        raise ValueError("shared PW-0328 closure mismatch")
    source_windows = authority.get("windows")
    if not isinstance(source_windows, list):
        raise ValueError("shared PW-0328 window schema mismatch")
    windows: list[Window] = []
    rows: list[RouteRow] = []
    for source in source_windows:
        window_index = int(source["corpus_index"])
        category = str(source["category"])
        transaction_index = int(source["transaction_index"])
        accepted = int(source["A"])
        if (
            accepted != len(source.get("verifier_authorized_token_ids", []))
            or accepted != len(source.get("authorized_q1_rows", []))
        ):
            raise ValueError("full verifier-authorized A is required; observable_A is forbidden")
        per_layer_source = source.get("per_layer_q8")
        if not isinstance(per_layer_source, list) or len(per_layer_source) != 47:
            raise ValueError("shared PW-0328 per-layer schema mismatch")
        unions = []
        for panel in per_layer_source:
            layer = int(panel["layer"])
            raw_identities = panel.get(
                "identities", panel.get("union", panel.get("union_experts"))
            )
            if not isinstance(raw_identities, list):
                raise ValueError("shared PW-0328 union schema mismatch")
            identities = tuple(sorted(_normalize_identity(value, layer) for value in raw_identities))
            unions.append((layer, identities))
        all_rows = source.get("all_q8_rows")
        if not isinstance(all_rows, list) or len(all_rows) != 8:
            raise ValueError("shared PW-0328 all_q8_rows schema mismatch")
        for position, position_rows in enumerate(all_rows):
            if isinstance(position_rows, Mapping) and int(position_rows.get("position", -1)) != position:
                raise ValueError("shared PW-0328 q8 position order mismatch")
            panels = position_rows.get("layers") if isinstance(position_rows, Mapping) else position_rows
            if not isinstance(panels, list) or len(panels) != 47:
                raise ValueError("shared PW-0328 route row panel mismatch")
            for panel in panels:
                layer = int(panel["layer"])
                raw_ids = panel.get(
                    "identities",
                    panel.get("selected_experts", panel.get("experts", panel.get("expert_set"))),
                )
                if not isinstance(raw_ids, list):
                    raise ValueError("shared PW-0328 route identities mismatch")
                identities = tuple(_normalize_identity(value, layer) for value in raw_ids)
                rows.append(RouteRow(category, window_index, transaction_index, position, layer, identities))
        windows.append(Window(
            window_index=window_index,
            category=category,
            transaction_index=transaction_index,
            accepted_tokens=accepted,
            unique_experts_per_layer=float(source["U"]),
            layer_unions=tuple(unions),
        ))
    windows.sort(key=lambda item: item.window_index)
    validate_corpus_shape(windows, rows)
    return windows, rows


def analyze(
    *,
    pw0328_manifest: Path,
    checkpoint_root: Path,
    checkpoint_verification: Path,
    pw0207_offline: Path,
    pw0136_raw: Path,
    pw0136_analysis: Path,
    pw0318_summary: Path,
    pw0318_manifest: Path,
    pw0318_bundle: Path,
    pw0316_analysis: Path,
    pw0308_manifest: Path,
    pw0308_repeated: Path,
    pw0325_analysis: Path,
    output: Path,
    repo: Path,
    commit: str,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(output)
    repo = repo.resolve()
    verify_clean_commit(repo, commit)
    authenticate_pw0328_corpus, authenticate_prismwing_storage = _import_shared_authorities()
    corpus_authority = authenticate_pw0328_corpus(repo=repo, manifest_path=pw0328_manifest)
    storage_authority = authenticate_prismwing_storage(
        checkpoint_root=checkpoint_root,
        verification_path=checkpoint_verification,
        offline_path=pw0207_offline,
        bandwidth_raw_path=pw0136_raw,
        bandwidth_analysis_path=pw0136_analysis,
    )
    windows, rows = normalize_pw0328_authority(corpus_authority)
    safety = HostSafetyMonitor()
    safety.checkpoint("authenticated_authorities_loaded")

    fixed_authority = storage_authority.get("fixed", {})
    storage_identities = storage_authority.get("identities", {})
    bandwidth_authority = storage_authority.get("bandwidth", {})
    if (
        fixed_authority.get("logical_source_bytes") != FIXED_LOGICAL_BYTES
        or fixed_authority.get("page_aligned_allocation_bytes") != FIXED_ALLOCATED_BYTES
        or fixed_authority.get("object_count") != 381
        or fixed_authority.get("largest_object_bytes") != LARGEST_FIXED_OBJECT_BYTES
        or storage_identities.get("pw0207_offline_sha256") != PW0207_OFFLINE_SHA256
        or storage_identities.get("pw0136_raw_sha256") != PW0136_RAW_SHA256
        or storage_identities.get("pw0136_analysis_sha256") != PW0136_ANALYSIS_SHA256
        or not math.isclose(
            float(bandwidth_authority.get("raw_exact_bytes_per_second", -1)),
            BANDWIDTH_EXACT_FLOAT,
            rel_tol=0.0,
            abs_tol=1e-6,
        )
        or not math.isclose(
            float(bandwidth_authority.get("candidate_favorable_bytes_per_second", -1)),
            BANDWIDTH_FAVORABLE_FLOAT,
            rel_tol=0.0,
            abs_tol=1e-6,
        )
    ):
        raise ValueError("shared storage authority mismatch")
    additional = validate_additional_authorities(
        repo=repo,
        pw0318_summary=pw0318_summary,
        pw0316_analysis=pw0316_analysis,
        pw0308_manifest=pw0308_manifest,
        pw0308_repeated=pw0308_repeated,
    )
    schema2 = validate_schema2_authority(pw0318_manifest, pw0318_bundle)
    correction_path = validate_correction_execution_path(repo)
    historical_order, historical_authority = validate_pw0325_order(pw0325_analysis)

    relaxed: dict[tuple[int, int], dict[str, Any]] = {}
    for density in DENSITIES:
        for gib in RESIDENCY_GIB:
            relaxed[(density, gib * GIB)] = relaxed_scenario(
                windows, rows, density=density, residency_bytes=gib * GIB
            )
    safety.checkpoint("relaxed_grid_complete")

    banks: list[SelectorResult] = []
    for density in DENSITIES:
        for gib in RESIDENCY_GIB:
            for target in CATEGORY_TARGETS:
                banks.append(select_fixed_bank(
                    windows,
                    rows,
                    density=density,
                    residency_bytes=gib * GIB,
                    target_tps=target,
                ))
    safety.checkpoint("row_feasible_bank_grid_complete")
    replay = historical_replay(windows, rows, historical_order)
    survival = density_survival_summary(relaxed)
    diagnostic_windows = [
        {
            "window_index": window.window_index,
            "category": window.category,
            "U": window.unique_experts_per_layer,
            "diagnostic_seconds": float(PW0308_REPEATED_P90_SECONDS * _fraction(window.unique_experts_per_layer)),
        }
        for window in windows
    ]
    diagnostic_seconds = math.fsum(item["diagnostic_seconds"] for item in diagnostic_windows)
    safety.release_checkpoint("analysis_state_released", ["PW-0328 route rows", "fixed-bank selection indices"])
    safety.checkpoint("final_service_health")
    gate8_pass = True
    decision = disposition(relaxed, banks, authority_complete=True, gate8_pass=gate8_pass)

    report = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "status": "analytical_joint_residency_bound_complete",
        "decision": decision,
        "commit": commit,
        "authority": {
            "contract_sha256": CONTRACT_SHA256,
            "contract_git_blob": CONTRACT_GIT_BLOB,
            "pw0328": corpus_authority,
            "storage": storage_authority,
            "additional": additional,
            "schema2": schema2,
            "correction_execution_path": correction_path,
            "pw0325": historical_authority,
            "authority_complete": True,
        },
        "constants": {
            "density_grid": list(DENSITIES),
            "residency_gib_grid": list(RESIDENCY_GIB),
            "category_target_tps_grid": list(CATEGORY_TARGETS),
            "source_logical_bytes": SOURCE_LOGICAL_BYTES,
            "k4_logical_bytes": K4_LOGICAL_BYTES,
            "source_schema2_stride": SOURCE_SCHEMA2_STRIDE,
            "k4_schema2_stride": K4_SCHEMA2_STRIDE,
            "source_hypothetical_repack_stride": SOURCE_REPACK_STRIDE,
            "k4_hypothetical_repack_stride": K4_REPACK_STRIDE,
            "logical_saving_per_substitution": LOGICAL_SAVING,
            "schema2_saving_per_substitution": SCHEMA2_SAVING,
            "hypothetical_repack_saving_per_substitution": REPACK_SAVING,
            "fixed_shared_logical_bytes": FIXED_LOGICAL_BYTES,
            "q8_exact_shared_logical_bytes": Q8_EXACT_SHARED_LOGICAL_BYTES,
            "fixed_shared_allocated_bytes": FIXED_ALLOCATED_BYTES,
            "largest_fixed_object_bytes": LARGEST_FIXED_OBJECT_BYTES,
            "exact_bandwidth_bytes_per_second": BANDWIDTH_EXACT_FLOAT,
            "favorable_historical_bandwidth_bytes_per_second": BANDWIDTH_FAVORABLE_FLOAT,
            "selector_score_arithmetic": "exact rational guarded-current-layout wall deficits",
        },
        "measurement_context": {
            "hardware": "Apple M1 16 GiB",
            "checkpoint_revision": CHECKPOINT_REVISION,
            "batch_size": 1,
            "concurrency": 1,
            "cache_state": "analytical; no cache populated",
            "prefill_state": "authenticated post-prefill PW-0328 windows",
            "companion_hardware": "excluded",
            "twelve_gib_is_physical_authorization": False,
        },
        "relaxed_scenarios": [relaxed[key] for key in sorted(relaxed)],
        "density_survival": survival,
        "fixed_bank_scenarios": [selector_result_dict(item) for item in banks],
        "pw0325_historical_replay": replay,
        "pw0308_diagnostic": {
            "repeated_mixed_component_p90_seconds": float(PW0308_REPEATED_P90_SECONDS),
            "sum_U": math.fsum(window.unique_experts_per_layer for window in windows),
            "scaled_corpus_seconds": diagnostic_seconds,
            "windows": diagnostic_windows,
            "gate": False,
            "scope": "47 repeats of one U=1 mixed three-K4/five-source row; not distinct layers or higher-density theorem",
        },
        "safety_snapshots": safety.evidence(),
        "gate8_analyzer_pass": gate8_pass,
        "accepted_tokens": 0,
        "performance_claim": None,
        "claims_excluded": [
            "achieved or endpoint TPS",
            "target-faithful K4 labeling",
            "construction or cache allocation",
            "qualified proposer or compute overlap",
            "companion hardware",
        ],
    }
    output.mkdir(parents=True)
    report_path = output / "analysis.json"
    atomic_write_new(report_path, canonical_json(report))
    print(json.dumps({"output": str(report_path), "decision": decision["decision"]}))
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    for name in (
        "pw0328_manifest",
        "checkpoint_root",
        "checkpoint_verification",
        "pw0207_offline",
        "pw0136_raw",
        "pw0136_analysis",
        "pw0318_summary",
        "pw0318_manifest",
        "pw0318_bundle",
        "pw0316_analysis",
        "pw0308_manifest",
        "pw0308_repeated",
        "pw0325_analysis",
        "output",
        "repo",
    ):
        parser.add_argument("--" + name.replace("_", "-"), required=True, type=Path)
    parser.add_argument("--commit", required=True)
    try:
        analyze(**vars(parser.parse_args()))
        return 0
    except (
        FileExistsError,
        HostSafetyViolation,
        KeyError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ) as error:
        print(json.dumps({"error": str(error)}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
