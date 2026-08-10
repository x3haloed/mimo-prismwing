#!/usr/bin/env python3
"""Authenticate and analyze PW-0162's global-attention sparsity oracle."""

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
CONFIG_SHA256 = "292a60e74ae9a6d53422b31b21468ce2111c0ab3f7f7a4f4e9c7cd5133b96587"
CORPUS_SHA256 = "3b5bc4e8f41fed2a13867bc96ea8236d1630bf994eee5608a8366f1f846a79d5"
VERIFICATION_SHA256 = "9ddc8a99755f04ae2ea3c2484f6dd022d3f3a681b5a72c915ee4de833dbb0d03"
PW0157_SHA256 = "32fa8954e875e6c8c53b5092827820940f51225d2bf24322caf5b782295004b9"
PW0158_SHA256 = "3b5b94cae112bee558ec46566ec09652c58bd434c3f47bebd3e0bc7c533fd315"
PW0161_SHA256 = "fc438d593d8ac99be3cc426496feb830256ffc48c75d58fc8bb9d6b09a2c6c8f"
INPUT_SHA256 = "9a8e422acb7b8762d86419adfe3234831614eee8a9f24c63648dccc4575d9e78"
ROUTES_SHA256 = "eff0dd3c993d132bd2ef66008c42c10e7b6b0b604ccad93ba0c72f894023a903"
FRACTIONS = (0.01, 0.05, 0.1, 0.2, 0.21056139043683178, 0.25, 1.0)
GLOBAL_LAYERS = (0, 5, 11, 17, 23, 29, 35, 41, 47)
SAMPLE_POSITIONS = tuple(range(63, 512, 32))
HEADS = 64
EXPECTED_OBSERVATIONS = len(GLOBAL_LAYERS) * len(SAMPLE_POSITIONS) * HEADS


def nearest_rank(values: list[float], quantile: float) -> float:
    if not values or not 0.0 < quantile <= 1.0:
        raise ValueError("invalid nearest-rank input")
    ordered = sorted(values)
    return ordered[math.ceil(quantile * len(ordered)) - 1]


def summarize_candidates(observations: list[dict], fraction_index: int) -> dict:
    if not 0 <= fraction_index < len(FRACTIONS):
        raise ValueError("fraction index out of range")
    rows = [observation["candidates"][fraction_index] for observation in observations]
    reference_squared = sum(row["reference_l2"] ** 2 for row in rows)
    error_squared = sum(row["error_l2"] ** 2 for row in rows)
    aggregate = math.sqrt(error_squared / max(reference_squared, 1.0e-40))
    relative = [row["relative_l2"] for row in rows]
    masses = [row["retained_probability_mass"] for row in rows]
    by_layer = {}
    for layer in GLOBAL_LAYERS:
        layer_rows = [
            observation["candidates"][fraction_index]
            for observation in observations
            if observation["layer"] == layer
        ]
        layer_reference_squared = sum(row["reference_l2"] ** 2 for row in layer_rows)
        layer_error_squared = sum(row["error_l2"] ** 2 for row in layer_rows)
        by_layer[str(layer)] = {
            "observations": len(layer_rows),
            "aggregate_relative_l2": math.sqrt(
                layer_error_squared / max(layer_reference_squared, 1.0e-40)
            ),
        }
    return {
        "retained_fraction": FRACTIONS[fraction_index],
        "observations": len(rows),
        "aggregate_relative_l2": aggregate,
        "head_query_relative_l2_p50": nearest_rank(relative, 0.50),
        "head_query_relative_l2_p99": nearest_rank(relative, 0.99),
        "head_query_relative_l2_max": max(relative),
        "retained_probability_mass_p01": nearest_rank(masses, 0.01),
        "retained_probability_mass_p50": nearest_rank(masses, 0.50),
        "maximum_absolute_error": max(row["maximum_absolute_error"] for row in rows),
        "bit_exact_values": sum(row["bit_exact_values"] for row in rows),
        "total_values": sum(row["total_values"] for row in rows),
        "layers": by_layer,
    }


def validate_safety(snapshots: object) -> None:
    if not isinstance(snapshots, list) or len(snapshots) < 4:
        raise ValueError("missing Gate-8 safety snapshots")
    phases = {snapshot.get("phase") for snapshot in snapshots}
    if not {"process_start", "checkpoint_released", "final_service_health"} <= phases:
        raise ValueError("missing Gate-8 release or service-health phase")
    baseline_services = {
        name for name, pids in snapshots[0].get("protected_service_pids", {}).items() if pids
    }
    for snapshot in snapshots:
        current_services = {
            name for name, pids in snapshot.get("protected_service_pids", {}).items() if pids
        }
        if (
            snapshot.get("system_memory_free_percent", -1) < 20
            or snapshot.get("process_physical_footprint_bytes", 1 << 63) > 8 * 1024**3
            or snapshot.get("process_peak_resident_bytes", 1 << 63) > 8 * 1024**3
            or snapshot.get("swap_growth_bytes", 1 << 63) > 512 * 1024**2
            or snapshot.get("new_throttled_pages", 1) != 0
            or not baseline_services <= current_services
        ):
            raise ValueError("Gate-8 safety violation in raw trace")
    release = next(row for row in snapshots if row["phase"] == "checkpoint_released")
    if release["process_physical_footprint_bytes"] > 4 * 1024**3:
        raise ValueError("Gate-8 post-release footprint violation")


def _authenticate(paths: dict[str, Path], commit: str) -> tuple[dict, dict]:
    fixed = {
        "target": TARGET_SHA256,
        "config": CONFIG_SHA256,
        "corpus": CORPUS_SHA256,
        "verification": VERIFICATION_SHA256,
        "pw0157": PW0157_SHA256,
        "pw0158": PW0158_SHA256,
        "pw0161": PW0161_SHA256,
    }
    for name, expected in fixed.items():
        if sha256_file(paths[name]) != expected:
            raise ValueError(f"PW-0162 source hash mismatch: {name}")
    target = paths["target"].read_text(errors="strict")
    if "1M-token smoke case" not in target or "USD $500 total" not in target:
        raise ValueError("TARGET authority mismatch")
    config = json.loads(paths["config"].read_text())
    if (
        config.get("max_position_embeddings", 0) < 1_000_000
        or tuple(index for index, kind in enumerate(config["hybrid_layer_pattern"]) if kind == 0)
        != GLOBAL_LAYERS
    ):
        raise ValueError("config global-attention authority mismatch")
    pw0158 = json.loads(paths["pw0158"].read_text())
    pw0161 = json.loads(paths["pw0161"].read_text())
    if (
        pw0158.get("evidence_class") != "pw0158_million_context_two_p100_attention_ceiling"
        or pw0158.get("attention_work_ledger", {}).get("global_attention_layers") != 9
        or pw0161.get("evidence_class") != "pw0161_volta_32gb_complete_system_envelope"
        or pw0161.get("positions") != 1_000_000
    ):
        raise ValueError("prior arithmetic authority mismatch")
    raw = json.loads(paths["raw"].read_text())
    if raw.get("commit") != commit:
        raise ValueError("raw trace implementation commit mismatch")
    source_hashes = {name: expected for name, expected in fixed.items()}
    source_hashes["raw"] = sha256_file(paths["raw"])
    return raw, source_hashes


def _validate_raw(raw: dict) -> None:
    if (
        raw.get("schema_version") != 1
        or raw.get("semantic") != "mimo_target_faithful_global_attention_sparsity_shadow_trace"
        or raw.get("revision") != REVISION
        or raw.get("fixture_sha256") != CORPUS_SHA256
        or raw.get("checkpoint_verification_sha256") != VERIFICATION_SHA256
        or raw.get("pw0157_prefix512_sha256") != PW0157_SHA256
        or raw.get("traced_prefix_positions") != 512
        or raw.get("input_token_ids_sha256") != INPUT_SHA256
        or raw.get("layer_routes_sha256") != ROUTES_SHA256
        or tuple(raw.get("observed_global_layers", ())) != GLOBAL_LAYERS
        or tuple(raw.get("sampled_absolute_query_positions", ())) != SAMPLE_POSITIONS
        or raw.get("observed_heads_per_sample") != HEADS
        or tuple(raw.get("retained_fractions", ())) != FRACTIONS
        or raw.get("accepted_tokens") != 0
        or raw.get("performance_claim") is not None
        or raw.get("exactness") != "target_faithful_source_state_with_noncausal_L3_shadow_only"
    ):
        raise ValueError("raw trace identity mismatch")
    observations = raw.get("observations")
    if not isinstance(observations, list) or len(observations) != EXPECTED_OBSERVATIONS:
        raise ValueError("raw observation count mismatch")
    expected = {
        (layer, position, head)
        for layer in GLOBAL_LAYERS
        for position in SAMPLE_POSITIONS
        for head in range(HEADS)
    }
    identities = set()
    for observation in observations:
        identity = (
            observation.get("layer"),
            observation.get("absolute_query_position"),
            observation.get("head"),
        )
        identities.add(identity)
        if (
            observation.get("visible_positions") != identity[1] + 1
            or len(observation.get("candidates", ())) != len(FRACTIONS)
        ):
            raise ValueError("raw observation shape mismatch")
        for index, candidate in enumerate(observation["candidates"]):
            numeric = (
                candidate.get("retained_probability_mass"),
                candidate.get("reference_l2"),
                candidate.get("candidate_l2"),
                candidate.get("error_l2"),
                candidate.get("relative_l2"),
                candidate.get("maximum_absolute_error"),
            )
            if (
                candidate.get("retained_fraction") != FRACTIONS[index]
                or candidate.get("retained_positions")
                != max(1, math.ceil(FRACTIONS[index] * observation["visible_positions"]))
                or any(not isinstance(value, (int, float)) or not math.isfinite(value) for value in numeric)
            ):
                raise ValueError("raw candidate value mismatch")
        control = observation["candidates"][-1]
        if (
            control["retained_positions"] != observation["visible_positions"]
            or control["bit_exact_values"] != control["total_values"]
            or control["error_l2"] != 0.0
            or control["maximum_absolute_error"] != 0.0
        ):
            raise ValueError("100% oracle control mismatch")
    if identities != expected:
        raise ValueError("raw observation identities mismatch")
    validate_safety(raw.get("safety_snapshots"))


def run(paths: dict[str, Path], output: Path, commit: str) -> dict:
    if output.exists():
        raise ValueError(f"refusing to overwrite {output}")
    authenticate_implementation_commit(commit)
    raw, source_hashes = _authenticate(paths, commit)
    _validate_raw(raw)
    summaries = [summarize_candidates(raw["observations"], index) for index in range(len(FRACTIONS))]
    top20 = summaries[3]
    exact_boundary = summaries[4]
    maximum_layer_relative_l2 = max(
        row["aggregate_relative_l2"] for row in top20["layers"].values()
    )
    passes = (
        top20["aggregate_relative_l2"] <= 0.01
        and maximum_layer_relative_l2 <= 0.02
        and top20["head_query_relative_l2_p99"] <= 0.05
    )
    manifest = {
        "schema_version": 1,
        "evidence_class": "pw0162_global_attention_top20_oracle",
        "revision": REVISION,
        "commit": commit,
        "source_hashes": source_hashes,
        "observations": EXPECTED_OBSERVATIONS,
        "fraction_summaries": summaries,
        "continuation_gate": {
            "retained_fraction": 0.2,
            "aggregate_relative_l2_threshold": 0.01,
            "aggregate_relative_l2": top20["aggregate_relative_l2"],
            "maximum_layer_relative_l2_threshold": 0.02,
            "maximum_layer_relative_l2": maximum_layer_relative_l2,
            "head_query_relative_l2_p99_threshold": 0.05,
            "head_query_relative_l2_p99": top20["head_query_relative_l2_p99"],
            "passes": passes,
        },
        "exact_two_p100_arithmetic_boundary": exact_boundary,
        "decision": (
            "promote_causal_accumulated_phase_b_only"
            if passes
            else "kill_simple_probability_ranked_20_percent_global_history_pruning"
        ),
        "accepted_tokens": 0,
        "A": 0,
        "U": None,
        "performance_claim": None,
        "endpoint_tps": None,
        "limitations": (
            "non-causal oracle shadow on exact source states at 512 positions; not a selector, "
            "accumulated candidate run, held-out result, one-million-token result, endpoint, or TPS"
        ),
    }
    atomic_write_new(output, canonical_json(manifest))
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("target", type=Path)
    parser.add_argument("config", type=Path)
    parser.add_argument("corpus", type=Path)
    parser.add_argument("verification", type=Path)
    parser.add_argument("pw0157", type=Path)
    parser.add_argument("pw0158", type=Path)
    parser.add_argument("pw0161", type=Path)
    parser.add_argument("raw", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("commit")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    names = (
        "target",
        "config",
        "corpus",
        "verification",
        "pw0157",
        "pw0158",
        "pw0161",
        "raw",
    )
    manifest = run({name: getattr(args, name) for name in names}, args.output, args.commit)
    print(canonical_json(manifest), end="")


if __name__ == "__main__":
    main()
