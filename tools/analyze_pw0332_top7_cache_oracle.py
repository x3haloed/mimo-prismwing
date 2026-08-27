#!/usr/bin/env python3
"""Analyze the PW-0332 exact top-seven token-cache storage ceiling.

This is an analytical, impossible-favorable byte-floor falsifier.  It neither
builds a codec nor executes the target model, and it never reports endpoint
throughput.
"""

from __future__ import annotations

import argparse
from bisect import bisect_right
from collections import defaultdict
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import itertools
import json
import math
from pathlib import Path
import platform
import subprocess
from typing import Any, Hashable, Iterable, Mapping, Sequence

try:
    from tools.host_safety import HostSafetyMonitor, HostSafetyViolation
    from tools.analyze_pw0324_onboard_closure import replicate_pw0300_panel
    from tools.openrouter_reference import atomic_write_new, canonical_json
    from tools.pw0328_corpus_authority import authenticate_pw0328_corpus
    from tools.prismwing_storage_authority import (
        CHECKPOINT_VERIFICATION_PATH,
        authenticate_prismwing_storage,
    )
    from tools.reproduce_pw0311_k4_expert import verify_clean_commit
except ModuleNotFoundError:
    from host_safety import HostSafetyMonitor, HostSafetyViolation
    from analyze_pw0324_onboard_closure import replicate_pw0300_panel
    from openrouter_reference import atomic_write_new, canonical_json
    from pw0328_corpus_authority import authenticate_pw0328_corpus
    from prismwing_storage_authority import (
        CHECKPOINT_VERIFICATION_PATH,
        authenticate_prismwing_storage,
    )
    from reproduce_pw0311_k4_expert import verify_clean_commit


EXPERIMENT_ID = "PW-0332"
CONTRACT_PATH = "experiments/PW-0332-exact-top7-token-cache-oracle.md"
CONTRACT_ADDING_COMMIT = "d43efd82fb1bc1a01dc137958bc474158a721f5d"
CONTRACT_GIT_BLOB = "073fcb4fd52330acb8ed8d8d645f521ae2ded3b8"
CONTRACT_SHA256 = "e37d1586311989f2e23e1af5737774d1332077e705baaf8d53e96e63e75d90e1"
TARGET_SHA256 = "dda459684c194b03491f36e9b66521ff00c400a6cc38d23a567a5a92ef8fb17d"
RED_LINES_SHA256 = "cc261ad9bd67a865715e72cbbadf3b74c3f1f282e17a8ef86ed02c1a92fb8b36"

PW0328_MANIFEST_SHA256 = "36e4f10b6f807f766c87ee7078f5f18ea8fc339dd12e4dbc24f1f4ac6e824403"
PW0328_CAPTURE_COMMIT = "26d2ea31852c0d63bd022df6d571fd722137c39f"
PW0328_Q1_DEMAND_STREAM_SHA256 = (
    "91fd42fe48033a1b04c1b3d9cdba30a4e6847147064db9946e71c6595bf71db6"
)
PW0324_ANALYSIS_PATH = Path(
    "/Users/chad/Models/mimo-prismwing/evidence/PW-0324/analysis-002/analysis.json"
)
PW0324_ANALYSIS_SHA256 = "97d4d20a4c709d42429973e867138495756ce9d52d417f98a7edd40b282ccff3"
PW0212_ANALYSIS_PATH = Path(
    "/Volumes/Elements/mimo-prismwing/evidence/PW-0212/"
    "corrected-route-prefetch-oracle-001.json"
)
PW0212_ANALYSIS_SHA256 = "2365033116e194b6bac34d2017f644c3499c5fb92a3727f7db9162dce318587f"
PW0207_OFFLINE_SHA256 = "1dedbef7c79aa23835d194f52760a1f2c65dcca1481bd6df2d5602615c3fdad6"
PW0136_RAW_SHA256 = "e6ab84cada19c6036ee7b83f318c3920631141b9ea5e882cc88eb9784d0b5a56"
PW0136_ANALYSIS_SHA256 = "7ebf2cde5c4a3f4931d2d705993f822e38af13ea66bc3efc91410296b14e2aab"

CHECKPOINT_REVISION = "63651580ca774f8504f676040460aed3e1244ac1"
CHECKPOINT_RECEIPT_SHA256 = "9ddc8a99755f04ae2ea3c2484f6dd022d3f3a681b5a72c915ee4de833dbb0d03"
CHECKPOINT_INDEX_SHA256 = "f2e1774c9acf9a62338b68c144e6fc7a66495e59f2e64b3078c1b7ef5a196816"
DEFAULT_CHECKPOINT_ROOT = Path(
    "/Users/chad/Models/mimo-prismwing/checkpoints/MiMo-V2.5-63651580"
)

CATEGORIES = ("ordinary", "code", "multilingual", "rare_route")
EXPECTED_CATEGORY_A = {
    "ordinary": 50,
    "code": 58,
    "multilingual": 60,
    "rare_route": 64,
}
ROUTED_LAYERS = tuple(range(1, 48))
EXPERTS_PER_LAYER = 256
TOP_K = 8
WINDOWS_PER_CATEGORY = 8

GIB = 1024**3
RESIDENCY_BYTES = 12 * GIB
SOURCE_EXPERT_BYTES = 25_171_968
SOURCE_EXPERT_FP8_CODE_BYTES = 25_165_824
SOURCE_EXPERT_F32_SCALE_BYTES = 6_144
FIXED_LOGICAL_BYTES = 7_743_236_992
FIXED_FP8_CODE_BYTES = 3_073_376_256
FIXED_NON_FP8_BYTES = 4_669_860_736
FIXED_OBJECT_COUNT = 381
FIXED_FP8_OBJECT_COUNT = 51
FIXED_BF16_OBJECT_COUNT = 185
FIXED_F32_OBJECT_COUNT = 145
FIXED_BF16_BYTES = 4_471_927_680
FIXED_F32_BYTES = 197_933_056
FIXED_MAX_OBJECT_BYTES = 1_249_902_592

BLOCK_CODES = 128 * 128
ZERO_ESCAPE_BYTES = 14_340
OBSERVED_ESCAPE_BYTES = 14_510
ABSOLUTE_FLOOR_RATIO = Fraction(ZERO_ESCAPE_BYTES, BLOCK_CODES)
OBSERVED_MINIMUM_RATIO = Fraction(OBSERVED_ESCAPE_BYTES, BLOCK_CODES)

BANDWIDTH_EXACT = Fraction(201_719_808_000_000_000, 58_125_375)
BANDWIDTH_EXACT_FLOAT = 3_470_425_919.832775
BANDWIDTH_FAVORABLE = Fraction("3470448309.677419")
BANDWIDTH_FAVORABLE_FLOAT = 3_470_448_309.677419
PW0212_HISTORICAL_HIDDEN_FRACTION = 0.016168329780693005
PW0212_SOURCE_CORPUS_SHA256 = "a9bb6bd26bf048a2144133cc0a96023a8af112eae58122b666915149f2993a7b"
PW0212_IMPLEMENTATION_COMMIT = "098d43224a2cbbce706bca82b34bb2bc75a3033f"
TARGET_MEMORY_BYTES = 16 * GIB

Identity = tuple[int, int]
GenericIdentity = Hashable


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(repo: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *arguments],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def verify_execution_commit(repo: Path, commit: str) -> None:
    """Require clean HEAD, the frozen contract blob, and contract ancestry."""

    verify_clean_commit(repo, commit)
    if _git(repo, "rev-parse", f"{commit}:{CONTRACT_PATH}") != CONTRACT_GIT_BLOB:
        raise ValueError("PW-0332 contract Git blob mismatch")
    if sha256_file(repo / CONTRACT_PATH) != CONTRACT_SHA256:
        raise ValueError("PW-0332 contract SHA-256 mismatch")
    if subprocess.run(
        ["git", "-C", str(repo), "merge-base", "--is-ancestor", CONTRACT_ADDING_COMMIT, commit],
        check=False,
    ).returncode != 0:
        raise ValueError("PW-0332 execution commit does not descend from contract freeze")
    if sha256_file(repo / "TARGET.md") != TARGET_SHA256:
        raise ValueError("TARGET.md authority mismatch")
    if sha256_file(repo / "RED_LINES.md") != RED_LINES_SHA256:
        raise ValueError("RED_LINES.md authority mismatch")


def verify_target_hardware() -> dict[str, str]:
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise ValueError("PW-0332 requires the target Apple M1 host")
    brand = subprocess.run(
        ["/usr/sbin/sysctl", "-n", "machdep.cpu.brand_string"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if brand != "Apple M1":
        raise ValueError("PW-0332 hardware authority is not Apple M1")
    memory = int(
        subprocess.run(
            ["/usr/sbin/sysctl", "-n", "hw.memsize"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    if memory != TARGET_MEMORY_BYTES:
        raise ValueError("PW-0332 hardware authority is not 16 GiB")
    return {
        "system": "Darwin",
        "machine": "arm64",
        "processor": brand,
        "physical_memory_bytes": memory,
    }


def _identity_record(identity: Identity) -> dict[str, int]:
    return {"layer": identity[0], "expert": identity[1]}


def _identity_records(identities: Iterable[Identity]) -> list[dict[str, int]]:
    return [_identity_record(item) for item in sorted(identities)]


def _resident_sha256(identities: Iterable[Identity]) -> str:
    return hashlib.sha256(canonical_json(_identity_records(identities))).hexdigest()


def _fraction_record(value: Fraction) -> dict[str, int]:
    return {"numerator": value.numerator, "denominator": value.denominator}


def encoded_top7_bytes(escapes: int, *, codes: int = BLOCK_CODES) -> int:
    """Return ceil((7*N + 4*e + 28) / 8) for the named exact format."""

    if type(codes) is not int or codes <= 0:
        raise ValueError("top-seven code count must be positive")
    if type(escapes) is not int or not 0 <= escapes <= codes:
        raise ValueError("top-seven escape count is outside the block")
    bits = 7 * codes + 4 * escapes + 28
    return (bits + 7) // 8


def validate_codec_floor() -> dict[str, Any]:
    zero = encoded_top7_bytes(0)
    if zero != ZERO_ESCAPE_BYTES:
        raise ValueError("top-seven zero-escape formula mismatch")
    # The observed 14,510-byte block has 341 exact exponent escapes.  The byte
    # count, rather than this witness escape count, is the frozen PW-0324 ratio.
    observed = encoded_top7_bytes(341)
    if observed != OBSERVED_ESCAPE_BYTES:
        raise ValueError("top-seven observed escape formula mismatch")
    if any(encoded_top7_bytes(value) < zero for value in range(BLOCK_CODES + 1)):
        raise ValueError("top-seven encoded ratio fell below its zero-escape floor")
    return {
        "block_codes": BLOCK_CODES,
        "formula": "ceil((7*N + 4*e + 28) / 8)",
        "zero_escape_bytes": zero,
        "zero_escape_ratio": float(ABSOLUTE_FLOOR_RATIO),
        "zero_escape_ratio_fraction": _fraction_record(ABSOLUTE_FLOOR_RATIO),
        "observed_witness_escapes": 341,
        "observed_bytes": observed,
        "observed_ratio": float(OBSERVED_MINIMUM_RATIO),
        "observed_ratio_fraction": _fraction_record(OBSERVED_MINIMUM_RATIO),
        "ratio_below_floor_possible": False,
    }


@dataclass(frozen=True)
class Scenario:
    name: str
    fixed_fp8_ratio: Fraction
    expert_ratio: Fraction
    hard_kill_authority: bool = False

    def __post_init__(self) -> None:
        if not self.name or not 0 < self.fixed_fp8_ratio <= 1 or not 0 < self.expert_ratio <= 1:
            raise ValueError("invalid PW-0332 scenario")


SCENARIOS = (
    Scenario("uncompressed", Fraction(1), Fraction(1)),
    Scenario("observed_expert_only", Fraction(1), OBSERVED_MINIMUM_RATIO),
    Scenario("absolute_floor_all_fp8", ABSOLUTE_FLOOR_RATIO, ABSOLUTE_FLOOR_RATIO, True),
)


def scenario_layout(scenario: Scenario, *, residency_bytes: int = RESIDENCY_BYTES) -> dict[str, Any]:
    if type(residency_bytes) is not int or residency_bytes <= 0:
        raise ValueError("invalid residency capacity")
    encoded_fixed_fp8 = FIXED_FP8_CODE_BYTES * scenario.fixed_fp8_ratio
    encoded_fixed = Fraction(FIXED_NON_FP8_BYTES) + encoded_fixed_fp8
    encoded_expert = SOURCE_EXPERT_BYTES * scenario.expert_ratio
    if encoded_fixed.denominator != 1:
        raise ValueError("fixed encoded byte ledger is fractional")
    if encoded_fixed > residency_bytes:
        raise ValueError(f"{scenario.name}: encoded fixed set exceeds resident bytes")
    available = Fraction(residency_bytes) - encoded_fixed
    capacity = available // encoded_expert
    if capacity < TOP_K:
        raise ValueError(f"{scenario.name}: expert capacity is below one routed set")
    return {
        "scenario": scenario.name,
        "residency_bytes": residency_bytes,
        "fixed_object_count": FIXED_OBJECT_COUNT,
        "fixed_logical_bytes": FIXED_LOGICAL_BYTES,
        "fixed_fp8_logical_bytes": FIXED_FP8_CODE_BYTES,
        "fixed_bf16_logical_bytes": FIXED_BF16_BYTES,
        "fixed_f32_logical_bytes": FIXED_F32_BYTES,
        "fixed_non_fp8_logical_bytes": FIXED_NON_FP8_BYTES,
        "fixed_fp8_ratio": float(scenario.fixed_fp8_ratio),
        "fixed_fp8_ratio_fraction": _fraction_record(scenario.fixed_fp8_ratio),
        "encoded_fixed_fp8_bytes": int(encoded_fixed_fp8),
        "encoded_fixed_bytes": int(encoded_fixed),
        "source_expert_logical_bytes": SOURCE_EXPERT_BYTES,
        "source_expert_fp8_code_bytes": SOURCE_EXPERT_FP8_CODE_BYTES,
        "source_expert_f32_scale_bytes": SOURCE_EXPERT_F32_SCALE_BYTES,
        "expert_ratio": float(scenario.expert_ratio),
        "expert_ratio_fraction": _fraction_record(scenario.expert_ratio),
        "encoded_expert_bytes": float(encoded_expert),
        "encoded_expert_bytes_fraction": _fraction_record(encoded_expert),
        "expert_capacity": int(capacity),
        "unused_fractional_resident_bytes": float(available - capacity * encoded_expert),
        "unused_fractional_resident_bytes_fraction": _fraction_record(
            available - capacity * encoded_expert
        ),
        "fixed_set_pinned": True,
        "embedding_row_traffic_charged": False,
        "largest_object_guard_charged": False,
        "fractional_encoded_records_granted": True,
        "expert_ratio_applies_favorably_to_complete_record_including_scales": True,
        "hard_kill_authority": scenario.hard_kill_authority,
    }


def validate_layouts(layouts: Sequence[Mapping[str, Any]]) -> None:
    expected_names = [scenario.name for scenario in SCENARIOS]
    if [layout.get("scenario") for layout in layouts] != expected_names:
        raise ValueError("PW-0332 scenario order mismatch")
    if [layout.get("expert_capacity") for layout in layouts] != [204, 230, 250]:
        raise ValueError("PW-0332 exact capacity mismatch")
    if not (
        layouts[2]["encoded_fixed_bytes"] <= layouts[1]["encoded_fixed_bytes"]
        <= layouts[0]["encoded_fixed_bytes"]
        and layouts[2]["encoded_expert_bytes"] <= layouts[1]["encoded_expert_bytes"]
        <= layouts[0]["encoded_expert_bytes"]
    ):
        raise ValueError("absolute codec floor is not scenario-dominant")


@dataclass(frozen=True)
class Demand:
    category: str
    corpus_index: int
    transaction_index: int
    event_index: int
    position: int
    layer: int
    identities: tuple[Identity, ...]

    def __post_init__(self) -> None:
        if self.category not in CATEGORIES:
            raise ValueError("unknown demand category")
        if not 0 <= self.corpus_index < len(CATEGORIES) * WINDOWS_PER_CATEGORY:
            raise ValueError("demand corpus index mismatch")
        if not 0 <= self.transaction_index < WINDOWS_PER_CATEGORY or self.position < 0:
            raise ValueError("demand transaction/position mismatch")
        if self.layer not in ROUTED_LAYERS:
            raise ValueError("demand routed layer mismatch")
        if len(self.identities) != TOP_K or len(set(self.identities)) != TOP_K:
            raise ValueError("demand must contain eight distinct expert identities")
        if any(
            identity[0] != self.layer or not 0 <= identity[1] < EXPERTS_PER_LAYER
            for identity in self.identities
        ):
            raise ValueError("demand expert identity mismatch")


def normalize_corpus_authority(
    authority: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, list[Demand]]]:
    """Recheck the shared authority boundary and build only full-A q1 demands."""

    if authority.get("manifest_sha256") != PW0328_MANIFEST_SHA256:
        raise ValueError("PW-0328 manifest authority mismatch")
    if authority.get("builder_commit") != PW0328_CAPTURE_COMMIT:
        raise ValueError("PW-0328 builder authority mismatch")
    if authority.get("categories") != list(CATEGORIES):
        raise ValueError("PW-0328 category order mismatch")
    if authority.get("artifact_count") != 24:
        raise ValueError("PW-0328 artifact closure mismatch")
    windows = authority.get("windows")
    events = authority.get("q1_events")
    control = authority.get("control")
    if not isinstance(windows, list) or len(windows) != 32:
        raise ValueError("PW-0328 requires exactly 32 primary windows")
    if not isinstance(events, list) or len(events) != 232 or not isinstance(control, dict):
        raise ValueError("PW-0328 full-A q1 event cardinality mismatch")
    if control.get("sum_A") != 232 or control.get("sum_observable_A") != 231:
        raise ValueError("PW-0328 full-A versus clipped observable authority mismatch")

    window_rows: list[dict[str, Any]] = []
    window_by_index: dict[int, dict[str, Any]] = {}
    for index, window in enumerate(windows):
        if not isinstance(window, dict):
            raise ValueError("PW-0328 window schema mismatch")
        expected_category = CATEGORIES[index // WINDOWS_PER_CATEGORY]
        expected_transaction = index % WINDOWS_PER_CATEGORY
        a = window.get("A")
        authorized = window.get("verifier_authorized_token_ids")
        q1 = window.get("authorized_q1_rows")
        if (
            window.get("corpus_index") != index
            or window.get("category") != expected_category
            or window.get("transaction_index") != expected_transaction
            or type(a) is not int
            or not 1 <= a <= 8
            or not isinstance(authorized, list)
            or len(authorized) != a
            or not isinstance(q1, list)
            or len(q1) != a
            or [row.get("position") for row in q1 if isinstance(row, dict)] != list(range(a))
        ):
            raise ValueError("PW-0328 clipped, reordered, or incomplete full-A window")
        value_u = window.get("U")
        if type(value_u) not in (int, float) or not math.isfinite(float(value_u)) or value_u <= 0:
            raise ValueError("PW-0328 window U mismatch")
        record = {
            "corpus_index": index,
            "category": expected_category,
            "transaction_index": expected_transaction,
            "modeled_source_A": a,
            "modeled_source_U": float(value_u),
        }
        window_rows.append(record)
        window_by_index[index] = record
    for category in CATEGORIES:
        if sum(row["modeled_source_A"] for row in window_rows if row["category"] == category) != EXPECTED_CATEGORY_A[category]:
            raise ValueError(f"PW-0328 {category} full-A total mismatch")

    traces: dict[str, list[Demand]] = {category: [] for category in CATEGORIES}
    per_window_events: dict[int, list[int]] = defaultdict(list)
    expected_event_order = [
        (window["corpus_index"], position)
        for window in window_rows
        for position in range(window["modeled_source_A"])
    ]
    observed_event_order = [
        (event.get("corpus_index"), event.get("position"))
        if isinstance(event, dict)
        else (None, None)
        for event in events
    ]
    if observed_event_order != expected_event_order:
        raise ValueError("PW-0328 global q1 category/window/position order mismatch")
    for event_index, event in enumerate(events):
        if not isinstance(event, dict):
            raise ValueError("PW-0328 q1 event schema mismatch")
        if set(event) != {
            "event_index",
            "category",
            "corpus_index",
            "transaction_index",
            "authorized_token_id",
            "position",
            "layers",
        }:
            raise ValueError("PW-0328 q1 event schema drift or proposal-route contamination")
        corpus_index = event.get("corpus_index")
        if type(corpus_index) is not int or corpus_index not in window_by_index:
            raise ValueError("PW-0328 q1 event window mismatch")
        window = window_by_index[corpus_index]
        position = event.get("position")
        layers = event.get("layers")
        if type(position) is not int or not 0 <= position < window["modeled_source_A"]:
            raise ValueError("PW-0328 q1 event position mismatch")
        if (
            event.get("event_index") != event_index
            or event.get("category") != window["category"]
            or event.get("transaction_index") != window["transaction_index"]
            or type(event.get("authorized_token_id")) is not int
            or not 0 <= event["authorized_token_id"] < 152_576
            or event["authorized_token_id"]
            != windows[corpus_index]["verifier_authorized_token_ids"][position]
            or not isinstance(layers, list)
            or len(layers) != len(ROUTED_LAYERS)
        ):
            raise ValueError("PW-0328 q1 event ordering or proposal-route contamination")
        per_window_events[corpus_index].append(position)
        for expected_layer, layer_row in zip(ROUTED_LAYERS, layers, strict=True):
            if not isinstance(layer_row, dict):
                raise ValueError("PW-0328 q1 layer schema mismatch")
            if set(layer_row) != {"layer", "experts"}:
                raise ValueError("PW-0328 q1 layer schema drift or proposal-route contamination")
            experts = layer_row.get("experts")
            if (
                layer_row.get("layer") != expected_layer
                or not isinstance(experts, list)
                or len(experts) != TOP_K
                or experts != sorted(set(experts))
                or any(type(expert) is not int or not 0 <= expert < EXPERTS_PER_LAYER for expert in experts)
            ):
                raise ValueError("PW-0328 q1 layer has duplicate or invalid identities")
            traces[window["category"]].append(
                Demand(
                    category=window["category"],
                    corpus_index=corpus_index,
                    transaction_index=window["transaction_index"],
                    event_index=event_index,
                    position=position,
                    layer=expected_layer,
                    identities=tuple((expected_layer, expert) for expert in experts),
                )
            )
    for window in window_rows:
        if per_window_events[window["corpus_index"]] != list(range(window["modeled_source_A"])):
            raise ValueError("PW-0328 mismatch suffix or clipped-A q1 event boundary")
    if demand_stream_sha256(traces) != PW0328_Q1_DEMAND_STREAM_SHA256:
        raise ValueError("PW-0328 authorized q1 demand-stream hash mismatch")
    return window_rows, traces


def demand_stream_sha256(traces: Mapping[str, Sequence[Demand]]) -> str:
    stream = []
    for category in CATEGORIES:
        for demand in traces.get(category, ()):
            stream.append(
                {
                    "category": demand.category,
                    "corpus_index": demand.corpus_index,
                    "event_index": demand.event_index,
                    "layer": demand.layer,
                    "experts": [identity[1] for identity in demand.identities],
                }
            )
    return hashlib.sha256(canonical_json(stream)).hexdigest()


def _canonical_demands(
    demands: Sequence[Iterable[GenericIdentity]],
) -> tuple[tuple[GenericIdentity, ...], ...]:
    result: list[tuple[GenericIdentity, ...]] = []
    for demand in demands:
        raw = tuple(demand)
        try:
            canonical = tuple(sorted(raw))
        except TypeError as error:
            raise ValueError("Belady identities lack canonical ordering") from error
        if not canonical or len(canonical) != len(set(canonical)):
            raise ValueError("Belady demand must contain distinct identities")
        result.append(canonical)
    if not result:
        raise ValueError("Belady trace must not be empty")
    return tuple(result)


def indexed_belady(
    demands: Sequence[Iterable[GenericIdentity]], capacity: int
) -> dict[str, Any]:
    """Exact equal-record batch-set Belady with deterministic ties."""

    trace = _canonical_demands(demands)
    if type(capacity) is not int or capacity < max(map(len, trace)):
        raise ValueError("Belady capacity is below simultaneous demand width")
    uses: dict[GenericIdentity, list[int]] = defaultdict(list)
    for demand_index, demand in enumerate(trace):
        for identity in demand:
            uses[identity].append(demand_index)
    universe = tuple(sorted(uses))
    pointers = {identity: 0 for identity in universe}
    initial = tuple(
        sorted(universe, key=lambda identity: (uses[identity][0], identity))[
            : min(capacity, len(universe))
        ]
    )
    resident = set(initial)
    ledger: list[dict[str, Any]] = []
    misses_total = 0
    evictions_total = 0

    def next_use(identity: GenericIdentity) -> tuple[int, int]:
        pointer = pointers[identity]
        if pointer == len(uses[identity]):
            return (1, 0)
        return (0, uses[identity][pointer])

    for demand_index, demand in enumerate(trace):
        demand_set = set(demand)
        if any(
            pointers[identity] >= len(uses[identity])
            or uses[identity][pointers[identity]] != demand_index
            for identity in demand
        ):
            raise ValueError("Belady future-use index corruption")
        misses = tuple(sorted(demand_set - resident))
        resident.update(misses)
        overflow = len(resident) - capacity
        candidates = resident - demand_set
        if overflow < 0:
            overflow = 0
        if overflow > len(candidates):
            raise ValueError("Belady demand set did not coexist")
        # Infinity sorts before finite for eviction, then farther use, then
        # reverse canonical identity.  D is protected until the full set is served.
        evictions = tuple(
            sorted(
                candidates,
                key=lambda identity: (*next_use(identity), identity),
                reverse=True,
            )[:overflow]
        )
        resident.difference_update(evictions)
        for identity in demand:
            pointers[identity] += 1
        if len(resident) > capacity or not demand_set <= resident:
            raise ValueError("Belady residency invariant failed")
        misses_total += len(misses)
        evictions_total += len(evictions)
        ledger.append(
            {
                "demand_index": demand_index,
                "demand": list(demand),
                "misses": list(misses),
                "evictions": list(evictions),
                "resident_count": len(resident),
                "resident_sha256": hashlib.sha256(canonical_json(list(sorted(resident)))).hexdigest(),
            }
        )
    return {
        "capacity": capacity,
        "distinct_identities": len(universe),
        "free_initial_identities": list(initial),
        "miss_count": misses_total,
        "eviction_count": evictions_total,
        "demand_ledger": ledger,
        "final_resident_identities": list(sorted(resident)),
        "final_resident_sha256": hashlib.sha256(canonical_json(list(sorted(resident)))).hexdigest(),
    }


def replay_belady(
    demands: Sequence[Iterable[GenericIdentity]], capacity: int, result: Mapping[str, Any]
) -> dict[str, Any]:
    """Independently replay a ledger using binary-search future lookup."""

    trace = _canonical_demands(demands)
    uses: dict[GenericIdentity, list[int]] = defaultdict(list)
    for index, demand in enumerate(trace):
        for identity in demand:
            uses[identity].append(index)
    universe = tuple(sorted(uses))
    expected_initial = tuple(
        sorted(universe, key=lambda identity: (uses[identity][0], identity))[
            : min(capacity, len(universe))
        ]
    )
    initial = tuple(result.get("free_initial_identities", ()))
    ledger = result.get("demand_ledger")
    if initial != expected_initial or not isinstance(ledger, list) or len(ledger) != len(trace):
        raise ValueError("independent Belady replay initial/ledger mismatch")
    resident = set(initial)
    misses_total = 0
    evictions_total = 0
    for index, (demand, row) in enumerate(zip(trace, ledger, strict=True)):
        if not isinstance(row, dict) or row.get("demand_index") != index or tuple(row.get("demand", ())) != demand:
            raise ValueError("independent Belady demand replay mismatch")
        demand_set = set(demand)
        misses = tuple(sorted(demand_set - resident))
        resident.update(misses)

        def key(identity: GenericIdentity) -> tuple[int, int, GenericIdentity]:
            position = bisect_right(uses[identity], index)
            if position == len(uses[identity]):
                return (1, 0, identity)
            return (0, uses[identity][position], identity)

        overflow = max(0, len(resident) - capacity)
        evictions = tuple(
            sorted(resident - demand_set, key=key, reverse=True)[:overflow]
        )
        resident.difference_update(evictions)
        digest = hashlib.sha256(canonical_json(list(sorted(resident)))).hexdigest()
        if (
            tuple(row.get("misses", ())) != misses
            or tuple(row.get("evictions", ())) != evictions
            or row.get("resident_count") != len(resident)
            or row.get("resident_sha256") != digest
        ):
            raise ValueError("independent Belady miss/residency replay mismatch")
        misses_total += len(misses)
        evictions_total += len(evictions)
    final = tuple(sorted(resident))
    if (
        misses_total != result.get("miss_count")
        or evictions_total != result.get("eviction_count")
        or tuple(result.get("final_resident_identities", ())) != final
        or result.get("final_resident_sha256")
        != hashlib.sha256(canonical_json(list(final))).hexdigest()
    ):
        raise ValueError("independent Belady final replay mismatch")
    return {
        "pass": True,
        "miss_count": misses_total,
        "eviction_count": evictions_total,
        "final_resident_identities": list(final),
        "final_resident_sha256": result["final_resident_sha256"],
    }


def exhaustive_optimal_misses(
    demands: Sequence[Iterable[GenericIdentity]], capacity: int, *, maximum_universe: int = 14
) -> int:
    """Exhaustive tiny-state DP oracle with a free future-chosen initial cache."""

    trace = _canonical_demands(demands)
    if type(capacity) is not int or capacity < max(map(len, trace)):
        raise ValueError("exhaustive capacity is below simultaneous demand width")
    universe = tuple(sorted({identity for demand in trace for identity in demand}))
    if len(universe) > maximum_universe:
        raise ValueError("exhaustive reference is restricted to tiny universes")
    initial_size = min(capacity, len(universe))
    states = {frozenset(items): 0 for items in itertools.combinations(universe, initial_size)}
    for demand in trace:
        demand_set = frozenset(demand)
        next_states: dict[frozenset[GenericIdentity], int] = {}
        for resident, prior_cost in states.items():
            loaded = resident | demand_set
            cost = prior_cost + len(demand_set - resident)
            size = min(capacity, len(loaded))
            extras = tuple(sorted(loaded - demand_set))
            keep_extra = size - len(demand_set)
            for kept in itertools.combinations(extras, keep_extra):
                state = frozenset((*demand_set, *kept))
                previous = next_states.get(state)
                if previous is None or cost < previous:
                    next_states[state] = cost
        states = next_states
        if not states:
            raise ValueError("exhaustive reference has no feasible residency state")
    return min(states.values())


def fixed_pinned_reference_misses(
    expert_demands: Sequence[Iterable[GenericIdentity]], *, fixed_objects: int, capacity: int
) -> int:
    if type(fixed_objects) is not int or fixed_objects < 0 or capacity - fixed_objects < 1:
        raise ValueError("invalid fixed-pinned tiny schedule")
    return exhaustive_optimal_misses(expert_demands, capacity - fixed_objects)


def exhaustive_joint_schedule_misses(
    expert_demands: Sequence[Iterable[GenericIdentity]], *, fixed_objects: int, capacity: int
) -> int:
    """Tiny whole-object joint DP used to falsify fixed-eviction benefit."""

    if type(fixed_objects) is not int or fixed_objects < 0:
        raise ValueError("invalid tiny fixed-object count")
    fixed = tuple(("fixed", index) for index in range(fixed_objects))
    trace: list[tuple[GenericIdentity, ...]] = []
    for demand in expert_demands:
        # All fixed objects are demanded once per token row, then each expert set.
        if fixed:
            trace.append(fixed)
        trace.append(tuple(("expert", identity) for identity in demand))
    return exhaustive_optimal_misses(trace, capacity)


def validate_tiny_oracles() -> dict[str, Any]:
    """Execute the frozen deterministic DP/replay implementation fixtures."""

    fixtures = (
        (
            "single_record_reuse",
            (((1, 0),), ((1, 1),), ((1, 0),)),
            1,
        ),
        (
            "batch_set_future_reuse",
            (
                ((1, 0), (1, 1)),
                ((2, 0), (2, 1)),
                ((1, 0), (1, 2)),
                ((2, 0), (2, 2)),
            ),
            2,
        ),
        (
            "fit_all_zero_miss",
            (
                ((1, 0), (1, 1)),
                ((1, 1), (1, 2)),
                ((1, 2), (1, 0)),
            ),
            3,
        ),
    )
    records = []
    for name, trace, capacity in fixtures:
        indexed = indexed_belady(trace, capacity)
        exhaustive = exhaustive_optimal_misses(trace, capacity)
        replay = replay_belady(trace, capacity, indexed)
        if indexed["miss_count"] != exhaustive or replay["miss_count"] != exhaustive:
            raise ValueError(f"tiny Belady implementation disagreement: {name}")
        records.append(
            {
                "fixture": name,
                "trace_sha256": hashlib.sha256(canonical_json(trace)).hexdigest(),
                "capacity": capacity,
                "indexed_misses": indexed["miss_count"],
                "exhaustive_dp_misses": exhaustive,
                "independent_replay_misses": replay["miss_count"],
                "pass": True,
            }
        )
    expert_trace = (("a",), ("b",), ("a",), ("c",))
    pinned = fixed_pinned_reference_misses(
        expert_trace, fixed_objects=1, capacity=3
    )
    joint = exhaustive_joint_schedule_misses(
        expert_trace, fixed_objects=1, capacity=3
    )
    if pinned > joint:
        raise ValueError("tiny dynamic fixed eviction beats fixed-pinned representative")
    return {
        "pass": True,
        "fixtures": records,
        "fixed_residency_dominance_fixture": {
            "fixed_objects": 1,
            "joint_capacity": 3,
            "fixed_pinned_misses": pinned,
            "exhaustive_joint_dynamic_misses": joint,
            "dynamic_fixed_eviction_improves_misses": False,
        },
    }


def nearest_rank_p10(values: Sequence[Fraction | None]) -> Fraction | None:
    if not values:
        raise ValueError("p10 requires at least one value")
    if any(value is not None and (not isinstance(value, Fraction) or value < 0) for value in values):
        raise ValueError("p10 value mismatch")
    ordered = sorted(values, key=lambda value: (value is None, value or Fraction(0)))
    return ordered[math.ceil(Fraction(1, 10) * len(ordered)) - 1]


def serialize_tps(value: Fraction | None) -> str | float:
    return "infinity" if value is None else float(value)


def _tps_record(value: Fraction | None) -> dict[str, Any]:
    return {
        "storage_tps": serialize_tps(value),
        "storage_tps_fraction": None if value is None else _fraction_record(value),
    }


def _aggregate(
    events: Sequence[Mapping[str, Any]], *, token_p10: bool = False
) -> dict[str, Any]:
    if not events:
        raise ValueError("aggregate requires token events")
    encoded = sum((event["encoded_moved_fraction"] for event in events), Fraction(0))
    wall = encoded / BANDWIDTH_FAVORABLE
    tps = None if wall == 0 else Fraction(len(events), 1) / wall
    result = {
        "token_events": len(events),
        "misses": sum(int(event["misses"]) for event in events),
        "logical_moved_bytes": sum(int(event["logical_moved_bytes"]) for event in events),
        "encoded_moved_bytes": float(encoded),
        "encoded_moved_bytes_fraction": _fraction_record(encoded),
        "storage_wall_seconds": float(wall),
        "storage_wall_seconds_fraction": _fraction_record(wall),
        **_tps_record(tps),
    }
    if token_p10:
        p10 = nearest_rank_p10([event["storage_tps_fraction_internal"] for event in events])
        result.update(
            {
                "nearest_rank_p10_token_storage_tps": serialize_tps(p10),
                "nearest_rank_p10_token_storage_tps_fraction": (
                    None if p10 is None else _fraction_record(p10)
                ),
                "nearest_rank_p10_rank": math.ceil(0.10 * len(events)),
            }
        )
    return result


def _annotate_real_ledger(
    demands: Sequence[Demand], raw: Mapping[str, Any]
) -> list[dict[str, Any]]:
    rows = raw.get("demand_ledger")
    if not isinstance(rows, list) or len(rows) != len(demands):
        raise ValueError("real Belady ledger cardinality mismatch")
    result: list[dict[str, Any]] = []
    for demand, row in zip(demands, rows, strict=True):
        result.append(
            {
                "demand_index": row["demand_index"],
                "event_index": demand.event_index,
                "corpus_index": demand.corpus_index,
                "transaction_index": demand.transaction_index,
                "position": demand.position,
                "layer": demand.layer,
                "demand": _identity_records(row["demand"]),
                "misses": _identity_records(row["misses"]),
                "evictions": _identity_records(row["evictions"]),
                "resident_count": row["resident_count"],
                "resident_sha256": row["resident_sha256"],
            }
        )
    return result


def analyze_scenario(
    scenario: Scenario,
    layout: Mapping[str, Any],
    windows: Sequence[Mapping[str, Any]],
    traces: Mapping[str, Sequence[Demand]],
) -> dict[str, Any]:
    capacity = int(layout["expert_capacity"])
    encoded_expert_record = Fraction(
        layout["encoded_expert_bytes_fraction"]["numerator"],
        layout["encoded_expert_bytes_fraction"]["denominator"],
    )
    all_events: list[dict[str, Any]] = []
    category_results: dict[str, Any] = {}
    ledgers: dict[str, Any] = {}
    for category in CATEGORIES:
        demands = list(traces[category])
        raw = indexed_belady([demand.identities for demand in demands], capacity)
        replay = replay_belady([demand.identities for demand in demands], capacity, raw)
        annotated = _annotate_real_ledger(demands, raw)
        by_event: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for demand, row in zip(demands, annotated, strict=True):
            by_event[demand.event_index].append(row)
        events: list[dict[str, Any]] = []
        for event_index in sorted(by_event):
            rows = by_event[event_index]
            if [row["layer"] for row in rows] != list(ROUTED_LAYERS):
                raise ValueError("token event lost routed-layer execution order")
            misses = sum(len(row["misses"]) for row in rows)
            encoded = misses * encoded_expert_record
            wall = encoded / BANDWIDTH_FAVORABLE
            tps = None if wall == 0 else Fraction(1, 1) / wall
            first = rows[0]
            event = {
                "event_index": event_index,
                "category": category,
                "corpus_index": first["corpus_index"],
                "transaction_index": first["transaction_index"],
                "position": first["position"],
                "layer_demands": len(rows),
                "misses": misses,
                "logical_moved_bytes": misses * SOURCE_EXPERT_BYTES,
                "encoded_moved_bytes": float(encoded),
                "encoded_moved_bytes_fraction": _fraction_record(encoded),
                "storage_wall_seconds": float(wall),
                "storage_wall_seconds_fraction": _fraction_record(wall),
                **_tps_record(tps),
                "encoded_moved_fraction": encoded,
                "storage_tps_fraction_internal": tps,
            }
            events.append(event)
        if sum(event["misses"] for event in events) != raw["miss_count"]:
            raise ValueError("token and demand miss ledgers do not close")
        category_results[category] = {
            "category_reset": True,
            "modeled_source_A": len(events),
            "modeled_source_U": math.fsum(
                window["modeled_source_U"] for window in windows if window["category"] == category
            ),
            "aggregate": _aggregate(events, token_p10=True),
            "token_events": [
                {key: value for key, value in event.items() if not key.endswith("_internal") and key != "encoded_moved_fraction"}
                for event in events
            ],
        }
        ledgers[category] = {
            "capacity": capacity,
            "distinct_identities": raw["distinct_identities"],
            "free_initial_identity_count": len(raw["free_initial_identities"]),
            "free_initial_identities": _identity_records(raw["free_initial_identities"]),
            "miss_count": raw["miss_count"],
            "eviction_count": raw["eviction_count"],
            "demand_ledger": annotated,
            "final_resident_identities": _identity_records(raw["final_resident_identities"]),
            "final_resident_sha256": raw["final_resident_sha256"],
            "independent_replay": {
                **{key: value for key, value in replay.items() if key != "final_resident_identities"},
                "final_resident_identities": _identity_records(replay["final_resident_identities"]),
            },
        }
        all_events.extend(events)

    window_results: list[dict[str, Any]] = []
    for window in windows:
        selected = [event for event in all_events if event["corpus_index"] == window["corpus_index"]]
        if len(selected) != window["modeled_source_A"]:
            raise ValueError("window token aggregate does not close to full A")
        window_results.append(
            {
                **dict(window),
                "aggregate": _aggregate(selected, token_p10=True),
            }
        )
    overall = _aggregate(all_events, token_p10=True)
    window_tps = []
    for window in window_results:
        exact = window["aggregate"]["storage_tps_fraction"]
        window_tps.append(None if exact is None else Fraction(exact["numerator"], exact["denominator"]))
    fourth = nearest_rank_p10(window_tps)
    overall.update(
        {
            "fourth_lowest_window_storage_tps": serialize_tps(fourth),
            "fourth_lowest_window_storage_tps_fraction": (
                None if fourth is None else _fraction_record(fourth)
            ),
            "window_nearest_rank_p10_rank": 4,
        }
    )
    return {
        "scenario": scenario.name,
        "layout": dict(layout),
        "overall": overall,
        "categories": category_results,
        "windows": window_results,
        "category_oracles": ledgers,
    }


def _exact_tps(summary: Mapping[str, Any], field: str = "storage_tps_fraction") -> Fraction | None:
    value = summary.get(field)
    if value is None:
        return None
    if not isinstance(value, dict) or set(value) != {"numerator", "denominator"}:
        raise ValueError("TPS exact-fraction schema mismatch")
    return Fraction(int(value["numerator"]), int(value["denominator"]))


def strict_gates(result: Mapping[str, Any]) -> dict[str, Any]:
    overall = result.get("overall")
    categories = result.get("categories")
    if not isinstance(overall, dict) or not isinstance(categories, dict) or set(categories) != set(CATEGORIES):
        raise ValueError("scenario aggregate schema mismatch")

    def above(value: Fraction | None) -> bool:
        return value is None or value > 1

    category_aggregate = {
        category: above(_exact_tps(categories[category]["aggregate"])) for category in CATEGORIES
    }
    category_token_p10 = {
        category: above(
            _exact_tps(
                categories[category]["aggregate"],
                "nearest_rank_p10_token_storage_tps_fraction",
            )
        )
        for category in CATEGORIES
    }
    gates = {
        "overall_aggregate_strictly_above_one": above(_exact_tps(overall)),
        "required_category_aggregates_strictly_above_one": category_aggregate,
        "corpus_token_p10_strictly_above_one": above(
            _exact_tps(overall, "nearest_rank_p10_token_storage_tps_fraction")
        ),
        "category_token_p10_strictly_above_one": category_token_p10,
        "fourth_lowest_window_strictly_above_one": above(
            _exact_tps(overall, "fourth_lowest_window_storage_tps_fraction")
        ),
    }
    gates["all_strict_gates_pass"] = (
        gates["overall_aggregate_strictly_above_one"]
        and all(category_aggregate.values())
        and gates["corpus_token_p10_strictly_above_one"]
        and all(category_token_p10.values())
        and gates["fourth_lowest_window_strictly_above_one"]
    )
    return gates


def disposition(results: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    if set(results) != {scenario.name for scenario in SCENARIOS}:
        raise ValueError("disposition scenario set mismatch")
    gates = {name: strict_gates(result) for name, result in results.items()}
    if not gates["absolute_floor_all_fp8"]["all_strict_gates_pass"]:
        decision = "reject_exact_top7_token_cache_oracle"
        decoder_authorized = False
        analytical_survivor = False
    elif not gates["observed_expert_only"]["all_strict_gates_pass"]:
        decision = "retain_absolute_floor_survivor_reject_observed_ratio_diagnostic"
        decoder_authorized = False
        analytical_survivor = True
    else:
        decision = "retain_analytical_survivor"
        decoder_authorized = False
        analytical_survivor = True
    return {
        "decision": decision,
        "scenario_gates": gates,
        "decoder_authorized": decoder_authorized,
        "analytical_survivor": analytical_survivor,
        "runtime_default_changed": False,
    }


def validate_scenario_dominance(results: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    ordered = [results[scenario.name] for scenario in SCENARIOS]
    for category in CATEGORIES:
        misses = [item["categories"][category]["aggregate"]["misses"] for item in ordered]
        if not misses[2] <= misses[1] <= misses[0]:
            raise ValueError("larger exact-codec cache increased optimal category misses")
    encoded = [Fraction(
        item["overall"]["encoded_moved_bytes_fraction"]["numerator"],
        item["overall"]["encoded_moved_bytes_fraction"]["denominator"],
    ) for item in ordered]
    if not encoded[2] <= encoded[1] <= encoded[0]:
        raise ValueError("scenario three is worse than another byte scenario")
    return {
        "pass": True,
        "capacity_order": [item["layout"]["expert_capacity"] for item in ordered],
        "overall_encoded_moved_bytes_fractions": [_fraction_record(value) for value in encoded],
        "claim": "absolute_floor_all_fp8 is no worse than every less favorable scenario",
    }


def validate_pw0324_payload(report: Mapping[str, Any]) -> dict[str, Any]:
    replication = report.get("local_fp8_codec_replication")
    if (
        report.get("schema_version") != 1
        or report.get("experiment_id") != "PW-0324"
        or report.get("status") != "complete"
        or report.get("accepted_tokens") != 0
        or report.get("performance_claim") is not None
        or not isinstance(replication, dict)
        or replication.get("evidence_class")
        != "pw0324_local_replication_of_pw0300_fp8_symbol_mechanisms"
        or replication.get("checkpoint_revision") != CHECKPOINT_REVISION
        or replication.get("checkpoint_receipt_sha256") != CHECKPOINT_RECEIPT_SHA256
        or replication.get("checkpoint_index_sha256") != CHECKPOINT_INDEX_SHA256
        or replication.get("quantization_blocks") != 480
        or replication.get("exact_palette_6bit_blocks") != 0
        or replication.get("exact_palette_7bit_blocks") != 0
        or replication.get("idealized_top7_exponent_escape_ratio")
        != {
            "minimum": float(OBSERVED_MINIMUM_RATIO),
            "median": 0.89013671875,
            "maximum": 0.91650390625,
        }
        or replication.get("panel")
        != [
            {"layer": 4, "expert": 96},
            {"layer": 24, "expert": 22},
            {"layer": 46, "expert": 28},
        ]
        or "not the unavailable original remote JSON payloads or routed-output fidelity"
        not in str(replication.get("limitation"))
    ):
        raise ValueError("PW-0324 exact-codec authority mismatch")
    tiles = replication.get("tiles")
    if not isinstance(tiles, list) or len(tiles) != 18 or sum(int(tile.get("quantization_blocks", -1)) for tile in tiles if isinstance(tile, dict)) != 480:
        raise ValueError("PW-0324 deterministic block ledger mismatch")
    return {
        "local_exact_codec_replication_authenticated": True,
        "checkpoint_index_sha256": CHECKPOINT_INDEX_SHA256,
        "quantization_blocks": 480,
        "observed_minimum_top7_ratio": float(OBSERVED_MINIMUM_RATIO),
        "sample_is_routed_full_model_codec_census": False,
        "limitation": replication["limitation"],
    }


def authenticate_pw0324(
    checkpoint_root: Path,
    path: Path = PW0324_ANALYSIS_PATH,
) -> dict[str, Any]:
    if path.resolve() != PW0324_ANALYSIS_PATH.resolve():
        raise ValueError("PW-0324 caller replacement is not authorized")
    digest = sha256_file(path)
    if digest != PW0324_ANALYSIS_SHA256:
        raise ValueError("PW-0324 analysis SHA-256 mismatch")
    report = json.loads(path.read_text())
    result = validate_pw0324_payload(report)
    replay = replicate_pw0300_panel(checkpoint_root, CHECKPOINT_VERIFICATION_PATH)
    if replay != report["local_fp8_codec_replication"]:
        raise ValueError("PW-0324 local exact-codec replay mismatch")
    return {
        "file": str(path),
        "sha256": digest,
        **result,
        "local_exact_codec_replay_equal_to_canonical": True,
    }


def validate_pw0212_payload(report: Mapping[str, Any]) -> dict[str, Any]:
    oracle = report.get("bandwidth_bounded_offline_future_oracle")
    aggregate = oracle.get("aggregate") if isinstance(oracle, dict) else None
    by_category = oracle.get("by_category") if isinstance(oracle, dict) else None
    implementation = report.get("implementation")
    identities = report.get("identities")
    constants = report.get("constants")
    if (
        report.get("schema_version") != 1
        or report.get("evidence_class") != "pw0212_corrected_route_predictive_prefetch_oracle"
        or report.get("status") != "complete"
        or report.get("decision") != "reject_runtime_prefetch_under_frozen_tax_and_complete_wall_gate"
        or report.get("accepted_tokens") != 0
        or report.get("performance_claim") is not None
        or not isinstance(aggregate, dict)
        or aggregate.get("complete_wall_hidden_fraction") != PW0212_HISTORICAL_HIDDEN_FRACTION
        or aggregate.get("windows") != 16
        or aggregate.get("demand_records") != 26_710
        or aggregate.get("prefetched_records") != 6_386
        or aggregate.get("prefetch_bandwidth_tax") != 0.23908648446274802
        or not isinstance(by_category, dict)
        or set(by_category) != set(CATEGORIES)
        or any(
            not isinstance(by_category[category], dict)
            or by_category[category].get("windows") != 4
            for category in CATEGORIES
        )
        or implementation
        != {"commit": PW0212_IMPLEMENTATION_COMMIT, "dirty": False}
        or not isinstance(identities, dict)
        or identities.get("pw0208_corpus_sha256") != PW0212_SOURCE_CORPUS_SHA256
        or not isinstance(constants, dict)
        or constants.get("expert_source_bytes") != SOURCE_EXPERT_BYTES
        or constants.get("maximum_prefetch_bandwidth_tax") != 0.25
        or constants.get("required_complete_wall_hidden_fraction") != 0.1
        or report.get("gates", {}).get("offline_oracle_implementation_gate_passed") is not False
    ):
        raise ValueError("PW-0212 prefetch-oracle authority mismatch")
    return {
        "historical_complete_wall_hidden_fraction": PW0212_HISTORICAL_HIDDEN_FRACTION,
        "historical_complete_wall_hidden_percent": 100 * PW0212_HISTORICAL_HIDDEN_FRACTION,
        "imported_as_current_percentage": False,
        "current_oracle_grants_future_knowledge_and_zero_prefetch_cost_anew": True,
    }


def authenticate_pw0212(path: Path = PW0212_ANALYSIS_PATH) -> dict[str, Any]:
    if path.resolve() != PW0212_ANALYSIS_PATH.resolve():
        raise ValueError("PW-0212 caller replacement is not authorized")
    digest = sha256_file(path)
    if digest != PW0212_ANALYSIS_SHA256:
        raise ValueError("PW-0212 analysis SHA-256 mismatch")
    result = validate_pw0212_payload(json.loads(path.read_text()))
    return {"file": str(path), "sha256": digest, **result}


def validate_storage_authority(authority: Mapping[str, Any]) -> dict[str, Any]:
    fixed = authority.get("fixed")
    identities = authority.get("identities")
    bandwidth = authority.get("bandwidth")
    if not isinstance(fixed, dict) or not isinstance(identities, dict) or not isinstance(bandwidth, dict):
        raise ValueError("shared storage authority schema mismatch")
    objects = fixed.get("objects")
    if not isinstance(objects, list) or len(objects) != FIXED_OBJECT_COUNT:
        raise ValueError("shared storage fixed-object ledger mismatch")
    dtype_rows: dict[str, tuple[int, int]] = {}
    for dtype in ("F8_E4M3", "BF16", "F32"):
        selected = [row for row in objects if isinstance(row, dict) and row.get("dtype") == dtype]
        dtype_rows[dtype] = (len(selected), sum(int(row.get("logical_bytes", -1)) for row in selected))
    if (
        identities.get("revision") != CHECKPOINT_REVISION
        or identities.get("checkpoint_verification_sha256") != CHECKPOINT_RECEIPT_SHA256
        or identities.get("tensor_index_sha256") != CHECKPOINT_INDEX_SHA256
        or identities.get("pw0207_offline_sha256") != PW0207_OFFLINE_SHA256
        or identities.get("pw0136_raw_sha256") != PW0136_RAW_SHA256
        or identities.get("pw0136_analysis_sha256") != PW0136_ANALYSIS_SHA256
        or fixed.get("object_count") != FIXED_OBJECT_COUNT
        or fixed.get("logical_source_bytes") != FIXED_LOGICAL_BYTES
        or fixed.get("fp8_code_bytes") != FIXED_FP8_CODE_BYTES
        or fixed.get("non_fp8_bytes") != FIXED_NON_FP8_BYTES
        or fixed.get("largest_object_bytes") != FIXED_MAX_OBJECT_BYTES
        or bandwidth.get("raw_exact_bytes_per_second") != BANDWIDTH_EXACT_FLOAT
        or bandwidth.get("candidate_favorable_bytes_per_second") != BANDWIDTH_FAVORABLE_FLOAT
        or dtype_rows["F8_E4M3"] != (FIXED_FP8_OBJECT_COUNT, FIXED_FP8_CODE_BYTES)
        or dtype_rows["BF16"] != (FIXED_BF16_OBJECT_COUNT, FIXED_BF16_BYTES)
        or dtype_rows["F32"] != (FIXED_F32_OBJECT_COUNT, FIXED_F32_BYTES)
    ):
        raise ValueError("shared storage authority semantic mismatch")
    return {
        "fixed_census_independently_recomputed": True,
        "fixed_object_count": FIXED_OBJECT_COUNT,
        "fixed_logical_bytes": FIXED_LOGICAL_BYTES,
        "fixed_fp8_code_bytes": FIXED_FP8_CODE_BYTES,
        "fixed_non_fp8_bytes": FIXED_NON_FP8_BYTES,
        "fixed_dtype_census": {
            dtype: {"objects": values[0], "logical_bytes": values[1]}
            for dtype, values in dtype_rows.items()
        },
        "trace_specific_embedding_rows_excluded": 7,
        "trace_specific_embedding_bytes_excluded": 57_344,
        "raw_exact_bandwidth_bytes_per_second": BANDWIDTH_EXACT_FLOAT,
        "candidate_favorable_bandwidth_bytes_per_second": BANDWIDTH_FAVORABLE_FLOAT,
    }


TOP_LEVEL_REPORT_KEYS = {
    "A",
    "U",
    "accepted_tokens",
    "authority",
    "codec_floor",
    "commit",
    "constants",
    "decision",
    "experiment_id",
    "gate8_analyzer_pass",
    "measurement_context",
    "performance_claim",
    "safety_snapshots",
    "scenario_dominance",
    "scenarios",
    "schema_version",
    "status",
}


def reject_nonfinite_evidence(value: Any, *, path: str = "$") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"nonfinite evidence at {path}")
    if isinstance(value, dict):
        for key, item in value.items():
            reject_nonfinite_evidence(item, path=f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            reject_nonfinite_evidence(item, path=f"{path}[{index}]")


def validate_report_schema(report: Mapping[str, Any]) -> None:
    if set(report) != TOP_LEVEL_REPORT_KEYS:
        raise ValueError("PW-0332 evidence schema drift")
    authority = report.get("authority")
    decision = report.get("decision")
    context = report.get("measurement_context")
    if (
        report.get("schema_version") != 1
        or report.get("experiment_id") != EXPERIMENT_ID
        or report.get("status") != "analytical_token_cache_oracle_complete"
        or report.get("gate8_analyzer_pass") is not True
        or report.get("accepted_tokens") != 0
        or report.get("A") != 0
        or report.get("U") != 0
        or report.get("performance_claim") is not None
        or not isinstance(authority, dict)
        or authority.get("authority_complete") is not True
        or not isinstance(decision, dict)
        or decision.get("runtime_default_changed") is not False
        or decision.get("decoder_authorized") is not False
        or not isinstance(context, dict)
        or context.get("companion_hardware") != "excluded"
        or context.get("batch_size") != 1
        or context.get("concurrency") != 1
        or context.get("accepted_tokens_experiment") != 0
        or not isinstance(report.get("scenarios"), list)
        or [row.get("scenario") for row in report["scenarios"] if isinstance(row, dict)]
        != [scenario.name for scenario in SCENARIOS]
    ):
        raise ValueError("PW-0332 report semantic schema mismatch")
    reject_nonfinite_evidence(report)


def analyze(*, repo: Path, checkpoint_root: Path, output: Path, commit: str) -> dict[str, Any]:
    repo = repo.resolve()
    checkpoint_root = checkpoint_root.resolve()
    if output.exists():
        raise FileExistsError(output)
    verify_execution_commit(repo, commit)
    hardware = verify_target_hardware()
    safety = HostSafetyMonitor()
    corpus_authority = authenticate_pw0328_corpus(repo=repo)
    storage_authority = authenticate_prismwing_storage(checkpoint_root=checkpoint_root)
    storage_summary = validate_storage_authority(storage_authority)
    pw0324 = authenticate_pw0324(checkpoint_root)
    pw0212 = authenticate_pw0212()
    safety.checkpoint("frozen_authorities_authenticated")

    windows, traces = normalize_corpus_authority(corpus_authority)
    tiny_validation = validate_tiny_oracles()
    layouts = [scenario_layout(scenario) for scenario in SCENARIOS]
    validate_layouts(layouts)
    safety.checkpoint("q1_demand_stream_reconstructed")
    results: dict[str, dict[str, Any]] = {}
    for scenario, layout in zip(SCENARIOS, layouts, strict=True):
        results[scenario.name] = analyze_scenario(scenario, layout, windows, traces)
        safety.checkpoint(f"{scenario.name}_oracle_complete")
    dominance = validate_scenario_dominance(results)
    decision = disposition(results)
    safety.release_checkpoint(
        "analysis_state_released",
        ["PW-0328 q1 demand traces", "batch-set future-use indices", "residency ledgers"],
    )
    safety.checkpoint("final_service_health")

    report = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "status": "analytical_token_cache_oracle_complete",
        "decision": decision,
        "commit": commit,
        "authority": {
            "contract": {
                "path": CONTRACT_PATH,
                "adding_commit": CONTRACT_ADDING_COMMIT,
                "git_blob": CONTRACT_GIT_BLOB,
                "sha256": CONTRACT_SHA256,
            },
            "target_sha256": TARGET_SHA256,
            "red_lines_sha256": RED_LINES_SHA256,
            "pw0328": corpus_authority,
            "pw0324": pw0324,
            "pw0212": pw0212,
            "storage": storage_authority,
            "storage_summary": storage_summary,
            "oracle_implementation_validation": tiny_validation,
            "authority_complete": True,
        },
        "codec_floor": validate_codec_floor(),
        "constants": {
            "residency_bytes": RESIDENCY_BYTES,
            "residency_gib": 12,
            "source_expert_logical_bytes": SOURCE_EXPERT_BYTES,
            "fixed_logical_bytes": FIXED_LOGICAL_BYTES,
            "fixed_fp8_code_bytes": FIXED_FP8_CODE_BYTES,
            "fixed_non_fp8_bytes": FIXED_NON_FP8_BYTES,
            "exact_raw_bandwidth_bytes_per_second": BANDWIDTH_EXACT_FLOAT,
            "candidate_favorable_bandwidth_bytes_per_second": BANDWIDTH_FAVORABLE_FLOAT,
            "scenario_capacities": {layout["scenario"]: layout["expert_capacity"] for layout in layouts},
            "required_threshold_tps": 1.0,
            "threshold_comparison": "strictly_greater_than",
        },
        "measurement_context": {
            "hardware": "Apple M1 16 GiB",
            "hardware_probe": hardware,
            "checkpoint_revision": CHECKPOINT_REVISION,
            "batch_size": 1,
            "concurrency": 1,
            "accepted_tokens_modeled": 232,
            "accepted_tokens_experiment": 0,
            "prefill_state": "authenticated post-prefill PW-0328 q8 windows",
            "cache_state": "category-reset future-aware analytical oracle with free initial fill",
            "cold_warm_state": "not measured; analytical ceiling",
            "companion_hardware": "excluded",
            "non_storage_work": "free",
            "prefetch_and_decode_cost": "free",
            "fixed_encoded_objects": "pinned for free after initial placement",
        },
        "scenarios": [results[scenario.name] for scenario in SCENARIOS],
        "scenario_dominance": dominance,
        "safety_snapshots": safety.evidence(),
        "gate8_analyzer_pass": True,
        "accepted_tokens": 0,
        "A": 0,
        "U": 0,
        "performance_claim": None,
    }
    validate_report_schema(report)
    output.mkdir(parents=True, exist_ok=False)
    report_path = output / "analysis.json"
    atomic_write_new(report_path, canonical_json(report))
    print(json.dumps({"output": str(report_path), "decision": decision["decision"]}))
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--checkpoint-root", type=Path, default=DEFAULT_CHECKPOINT_ROOT)
    parser.add_argument("--output", required=True, type=Path)
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
        subprocess.CalledProcessError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ) as error:
        print(json.dumps({"error": str(error)}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
