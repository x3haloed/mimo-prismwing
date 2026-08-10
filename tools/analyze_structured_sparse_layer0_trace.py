#!/usr/bin/env python3
"""Authenticate and adjudicate PW-0176's 64K layer-0 structured oracle."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

try:
    from tools.analyze_owned_epyc_companion_envelope import authenticate_implementation_commit
    from tools.analyze_pw0116_corpus import sha256_file
    from tools.openrouter_reference import atomic_write_new, canonical_json
except ModuleNotFoundError:
    from analyze_owned_epyc_companion_envelope import authenticate_implementation_commit
    from analyze_pw0116_corpus import sha256_file
    from openrouter_reference import atomic_write_new, canonical_json


REVISION = "63651580ca774f8504f676040460aed3e1244ac1"
TARGET_SHA256 = "91fe6e0441bb0a0e1ab0852db60fb575d131b61ff069002c9c333f9b776e4950"
PW0175_SHA256 = "e5ac56b7f710285cdeb0088f9fa750748ad74cbc68cd6d4dcb627061209a37ab"
MINFERENCE_SHA256 = "b368e765fcc2021591f7cdf970e4e1a71731ed66f19e97023a13b70a02e6edc2"
VERIFICATION_SHA256 = "9ddc8a99755f04ae2ea3c2484f6dd022d3f3a681b5a72c915ee4de833dbb0d03"
TOKEN_SHA256 = "7a5c2d35b51d6a05b6d445d575bd08d68fed91a8997ec1e13cdc4c31e71cc507"
WORK_CEILING = 0.21056139043683178
PAIRS = (
    (30, 800, 0.026340859133157844),
    (100, 800, 0.028448877943292053),
    (500, 700, 0.037457400465338284),
    (3500, 100, 0.1080168610920644),
    (1000, 6096, 0.20599934654934904),
)
SAMPLES = tuple(
    sorted(
        set(
            (63, 127, 255)
            + tuple(range(4095, 65_536, 4096))
            + (65_509, 65_515, 65_520, 65_525, 65_530, 65_535)
        )
    )
)
HEADS = 64
EXPECTED_OBSERVATIONS = len(SAMPLES) * HEADS


def nearest_rank(values: list[float], quantile: float) -> float:
    if not values or not 0.0 < quantile <= 1.0:
        raise ValueError("invalid nearest-rank input")
    ordered = sorted(values)
    return ordered[math.ceil(quantile * len(ordered)) - 1]


def _aggregate(rows: list[tuple[dict, dict]]) -> float:
    reference_squared = sum(observation["reference_l2"] ** 2 for observation, _ in rows)
    error_squared = sum(candidate["error_l2"] ** 2 for _, candidate in rows)
    return math.sqrt(error_squared / max(reference_squared, 1.0e-40))


def summarize_pair(observations: list[dict], pair_index: int) -> dict:
    vertical, slash, work = PAIRS[pair_index]
    rows = [(observation, observation["candidates"][pair_index]) for observation in observations]
    relative = [candidate["relative_l2"] for _, candidate in rows]
    selected = [candidate["selected_positions"] for _, candidate in rows]
    by_position = {}
    for position in SAMPLES:
        position_rows = [
            row for row in rows if row[0]["absolute_query_position"] == position
        ]
        by_position[str(position)] = {
            "observations": len(position_rows),
            "aggregate_relative_l2": _aggregate(position_rows),
        }
    by_band = {}
    for band in ("early", "interval", "final_question"):
        band_rows = [row for row in rows if row[0]["band"] == band]
        by_band[band] = {
            "observations": len(band_rows),
            "aggregate_relative_l2": _aggregate(band_rows),
            "head_query_relative_l2_p99": nearest_rank(
                [candidate["relative_l2"] for _, candidate in band_rows], 0.99
            ),
        }
    aggregate = _aggregate(rows)
    position_maximum = max(
        row["aggregate_relative_l2"] for row in by_position.values()
    )
    p99 = nearest_rank(relative, 0.99)
    passes = (
        work <= WORK_CEILING
        and aggregate <= 0.01
        and position_maximum <= 0.02
        and p99 <= 0.05
    )
    return {
        "vertical_size": vertical,
        "slash_size": slash,
        "effective_work_fraction": work,
        "within_complete_system_work_ceiling": work <= WORK_CEILING,
        "observations": len(rows),
        "aggregate_relative_l2": aggregate,
        "maximum_position_aggregate_relative_l2": position_maximum,
        "head_query_relative_l2_p50": nearest_rank(relative, 0.50),
        "head_query_relative_l2_p99": p99,
        "head_query_relative_l2_max": max(relative),
        "maximum_absolute_error": max(
            candidate["maximum_absolute_error"] for _, candidate in rows
        ),
        "selected_positions_min": min(selected),
        "selected_positions_p50": nearest_rank(selected, 0.50),
        "selected_positions_max": max(selected),
        "bit_exact_values": sum(candidate["bit_exact_values"] for _, candidate in rows),
        "total_values": sum(candidate["total_values"] for _, candidate in rows),
        "positions": by_position,
        "bands": by_band,
        "passes": passes,
    }


def validate_safety(snapshots: object) -> dict:
    if not isinstance(snapshots, list) or len(snapshots) < 135:
        raise ValueError("missing PW-0176 Gate-8 phase snapshots")
    phases = [snapshot.get("phase") for snapshot in snapshots]
    required = {
        "process_start",
        "pw0176_fixture_and_checkpoint_authenticated",
        "pw0176_qkv_weight_decoded",
        "pw0176_qkv_projection_complete",
        "pw0176_observations_complete",
        "pw0176_checkpoint_and_buffers_released",
        "pw0176_final_service_health",
    }
    if not required <= set(phases):
        raise ValueError("missing PW-0176 Gate-8 authority/release/health phase")
    if sum(phase.startswith("pw0176_qkv_chunk_") for phase in phases if isinstance(phase, str)) != 64:
        raise ValueError("missing PW-0176 per-chunk safety coverage")
    if sum(phase.startswith("pw0176_selector_head_") for phase in phases if isinstance(phase, str)) != 64:
        raise ValueError("missing PW-0176 per-head safety coverage")
    baseline_services = {
        name for name, pids in snapshots[0].get("protected_service_pids", {}).items() if pids
    }
    previous_peak = 0
    for snapshot in snapshots:
        current_services = {
            name
            for name, pids in snapshot.get("protected_service_pids", {}).items()
            if pids
        }
        peak = snapshot.get("process_peak_resident_bytes", 1 << 63)
        if (
            snapshot.get("system_memory_free_percent", -1) < 20
            or snapshot.get("process_resident_bytes", 1 << 63) > 8 * 1024**3
            or snapshot.get("process_physical_footprint_bytes", 1 << 63) > 8 * 1024**3
            or peak > 8 * 1024**3
            or peak < previous_peak
            or snapshot.get("swap_growth_bytes", 1 << 63) > 512 * 1024**2
            or snapshot.get("new_throttled_pages", 1) != 0
            or not baseline_services <= current_services
        ):
            raise ValueError("PW-0176 Gate-8 safety violation")
        if (
            snapshot.get("release_boundary") is True
            and snapshot.get("process_physical_footprint_bytes", 1 << 63) > 4 * 1024**3
        ):
            raise ValueError("PW-0176 Gate-8 post-release footprint violation")
        previous_peak = peak
    return {
        "snapshots": len(snapshots),
        "minimum_system_memory_free_percent": min(
            row["system_memory_free_percent"] for row in snapshots
        ),
        "maximum_process_resident_bytes": max(
            row["process_resident_bytes"] for row in snapshots
        ),
        "maximum_process_physical_footprint_bytes": max(
            row["process_physical_footprint_bytes"] for row in snapshots
        ),
        "maximum_process_peak_resident_bytes": max(
            row["process_peak_resident_bytes"] for row in snapshots
        ),
        "maximum_swap_growth_bytes": max(row["swap_growth_bytes"] for row in snapshots),
        "maximum_new_throttled_pages": max(
            row["new_throttled_pages"] for row in snapshots
        ),
        "final_services_healthy": baseline_services
        <= {
            name
            for name, pids in snapshots[-1]["protected_service_pids"].items()
            if pids
        },
    }


def validate_raw(raw: dict, commit: str) -> None:
    expected_pairs = [
        {
            "vertical_size": vertical,
            "slash_size": slash,
            "effective_work_fraction": work,
            "within_complete_system_work_ceiling": True,
        }
        for vertical, slash, work in PAIRS
    ]
    if (
        raw.get("schema_version") != 1
        or raw.get("semantic")
        != "mimo_target_faithful_layer0_structured_sparse_shadow_trace"
        or raw.get("revision") != REVISION
        or raw.get("commit") != commit
        or raw.get("fixture_commit") != commit
        or raw.get("checkpoint_verification_sha256") != VERIFICATION_SHA256
        or raw.get("token_ids_sha256") != TOKEN_SHA256
        or raw.get("positions") != 65_536
        or raw.get("chunk_positions") != 1024
        or raw.get("qkv_chunks") != 64
        or tuple(raw.get("sampled_absolute_query_positions", ())) != SAMPLES
        or raw.get("observed_heads_per_sample") != HEADS
        or raw.get("selector_last_queries") != 64
        or raw.get("pairs") != expected_pairs
        or raw.get("batch_size") != 1
        or raw.get("concurrency") != 1
        or raw.get("accepted_tokens") != 0
        or raw.get("performance_claim") is not None
        or raw.get("endpoint_tps") is not None
        or raw.get("exactness")
        != "target_faithful_source_layer0_qkv_and_dense_samples_with_noncausal_L3_shadow_only"
    ):
        raise ValueError("PW-0176 raw identity mismatch")
    for name in (
        "fixture_manifest_sha256",
        "authority_fixture_sha256",
        "qkv_sha256",
        "query_samples_sha256",
        "selector_queries_sha256",
        "keys_sha256",
        "values_sha256",
    ):
        if not isinstance(raw.get(name), str) or len(raw[name]) != 64:
            raise ValueError(f"PW-0176 raw hash missing: {name}")
    ledger = raw.get("ledger", {})
    if (
        ledger.get("fp8_matrices_expanded") != 1
        or ledger.get("bf16_matrices_expanded") != 1
        or ledger.get("dynamic_activation_values") != 65_536 * 4096
        or not isinstance(ledger.get("actual_process_disk_bytes_read"), int)
        or ledger.get("peak_resident_bytes", 0) <= 0
        or not isinstance(raw.get("complete_wall_ms"), (int, float))
        or not math.isfinite(raw["complete_wall_ms"])
        or raw["complete_wall_ms"] <= 0.0
    ):
        raise ValueError("PW-0176 raw ledger or timing mismatch")
    observations = raw.get("observations")
    if not isinstance(observations, list) or len(observations) != EXPECTED_OBSERVATIONS:
        raise ValueError("PW-0176 raw observation count mismatch")
    expected_identities = {(position, head) for position in SAMPLES for head in range(HEADS)}
    identities = set()
    expected_band = {
        position: (
            "early"
            if position in (63, 127, 255)
            else "final_question"
            if position in (65_509, 65_515, 65_520, 65_525, 65_530, 65_535)
            else "interval"
        )
        for position in SAMPLES
    }
    for observation in observations:
        identity = (observation.get("absolute_query_position"), observation.get("head"))
        identities.add(identity)
        reference_l2 = observation.get("reference_l2")
        if (
            identity not in expected_identities
            or observation.get("band") != expected_band[identity[0]]
            or observation.get("visible_positions") != identity[0] + 1
            or not isinstance(reference_l2, (int, float))
            or not math.isfinite(reference_l2)
            or reference_l2 < 0.0
            or observation.get("full_selection_bit_exact_values") != 128
            or len(observation.get("candidates", ())) != len(PAIRS)
        ):
            raise ValueError("PW-0176 raw observation shape mismatch")
        for index, candidate in enumerate(observation["candidates"]):
            vertical, slash, _ = PAIRS[index]
            numerics = (
                candidate.get("selected_fraction"),
                candidate.get("candidate_l2"),
                candidate.get("error_l2"),
                candidate.get("relative_l2"),
                candidate.get("maximum_absolute_error"),
            )
            if (
                candidate.get("vertical_size") != vertical
                or candidate.get("slash_size") != slash
                or not isinstance(candidate.get("selected_positions"), int)
                or not 1 <= candidate["selected_positions"] <= observation["visible_positions"]
                or any(
                    not isinstance(value, (int, float))
                    or not math.isfinite(value)
                    or value < 0.0
                    for value in numerics
                )
                or not math.isclose(
                    candidate["selected_fraction"],
                    candidate["selected_positions"] / observation["visible_positions"],
                    rel_tol=1.0e-12,
                    abs_tol=1.0e-15,
                )
                or not math.isclose(
                    candidate["relative_l2"],
                    candidate["error_l2"] / max(reference_l2, 1.0e-20),
                    rel_tol=1.0e-12,
                    abs_tol=1.0e-15,
                )
                or candidate.get("total_values") != 128
                or not isinstance(candidate.get("bit_exact_values"), int)
                or not 0 <= candidate["bit_exact_values"] <= 128
            ):
                raise ValueError("PW-0176 raw candidate mismatch")
    if identities != expected_identities:
        raise ValueError("PW-0176 raw observation identities mismatch")
    validate_safety(raw.get("safety_snapshots"))


def run(
    target: Path,
    pw0175: Path,
    minference: Path,
    raw_path: Path,
    output: Path,
    commit: str,
) -> dict:
    if output.exists():
        raise ValueError(f"refusing to overwrite {output}")
    authenticate_implementation_commit(commit)
    fixed = {
        "target": (target, TARGET_SHA256),
        "pw0175": (pw0175, PW0175_SHA256),
        "minference": (minference, MINFERENCE_SHA256),
    }
    source_hashes = {}
    for name, (path, expected) in fixed.items():
        observed = sha256_file(path)
        if observed != expected:
            raise ValueError(f"PW-0176 source hash mismatch: {name}")
        source_hashes[name] = observed
    raw = json.loads(raw_path.read_text(errors="strict"))
    validate_raw(raw, commit)
    source_hashes["raw"] = sha256_file(raw_path)
    summaries = [summarize_pair(raw["observations"], index) for index in range(len(PAIRS))]
    passing = [
        {"vertical_size": row["vertical_size"], "slash_size": row["slash_size"]}
        for row in summaries
        if row["passes"]
    ]
    safety = validate_safety(raw["safety_snapshots"])
    manifest = {
        "schema_version": 1,
        "evidence_class": "pw0176_mimo_64k_structured_sparse_layer0_oracle",
        "revision": REVISION,
        "commit": commit,
        "source_hashes": source_hashes,
        "positions": 65_536,
        "sampled_positions": list(SAMPLES),
        "heads": HEADS,
        "observations": EXPECTED_OBSERVATIONS,
        "thresholds": {
            "complete_system_work_fraction_maximum": WORK_CEILING,
            "aggregate_relative_l2_maximum": 0.01,
            "per_position_aggregate_relative_l2_maximum": 0.02,
            "head_query_relative_l2_p99_maximum": 0.05,
        },
        "pair_summaries": summaries,
        "passing_pairs": passing,
        "continuation_gate_passes": bool(passing),
        "decision": (
            "promote_passing_pairs_to_deeper_global_and_accumulated_fidelity"
            if passing
            else "kill_released_minference_vertical_slash_pairs_on_mandatory_mimo_layer0_64k_slice"
        ),
        "safety": safety,
        "accepted_tokens": 0,
        "A": 0,
        "U": None,
        "performance_claim": None,
        "endpoint_tps": None,
        "limitations": (
            "one artificial deterministic 64K text prefix and source layer 0 only; "
            "no downstream state, route, logit, native modality, endpoint, TPS, hardware, "
            "or purchase conclusion"
        ),
    }
    atomic_write_new(output, canonical_json(manifest))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("target", type=Path)
    parser.add_argument("pw0175", type=Path)
    parser.add_argument("minference", type=Path)
    parser.add_argument("raw", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("commit")
    arguments = parser.parse_args()
    manifest = run(
        arguments.target,
        arguments.pw0175,
        arguments.minference,
        arguments.raw,
        arguments.output,
        arguments.commit,
    )
    print(canonical_json(manifest), end="")


if __name__ == "__main__":
    main()
