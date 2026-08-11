#!/usr/bin/env python3
"""Evaluate PW-0214's q=2..8 corrected-window scheduling oracle."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import subprocess

try:
    from tools.analyze_pw0116_corpus import sha256_file
    from tools.openrouter_reference import atomic_write_new, canonical_json
except ModuleNotFoundError:
    from analyze_pw0116_corpus import sha256_file
    from openrouter_reference import atomic_write_new, canonical_json


MANIFEST_SHA256 = "a9bb6bd26bf048a2144133cc0a96023a8af112eae58122b666915149f2993a7b"
WIDTHS = tuple(range(2, 9))
EXPERT_BYTES = 25_171_968
COLD_SINGLE_EXPERT_ACQUISITION_MS = 58.033833 / 8
REQUIRED_CAUSAL_GAIN = 0.05
CATEGORIES = ("ordinary", "code", "multilingual", "rare_route")


def accepted_tokens(proposal: list[int], posterior: list[int], q: int) -> int:
    if q not in WIDTHS or len(proposal) < q or len(posterior) < q:
        raise ValueError("PW-0214 invalid q or proposal/posterior width")
    mismatch = next(
        (index for index in range(q - 1) if posterior[index] != proposal[index + 1]),
        None,
    )
    return q - 1 if mismatch is None else mismatch + 1


def expert_units(transaction: dict, q: int) -> int:
    traces = transaction["verification_layer_traces"]
    if len(traces) != 48:
        raise ValueError("PW-0214 verification trace must contain 48 layers")
    units = 0
    for layer, trace in enumerate(traces[1:], 1):
        if trace.get("layer") != layer:
            raise ValueError("PW-0214 route layers are not contiguous")
        routes = trace.get("selected_experts_by_position", [])[:q]
        if len(routes) != q or any(len(route) != 8 for route in routes):
            raise ValueError("PW-0214 route prefix shape mismatch")
        units += len({expert for route in routes for expert in route})
    return units


def aggregate(rows: list[dict], choices: list[int]) -> dict:
    if not rows or len(rows) != len(choices):
        raise ValueError("PW-0214 aggregate dimensions mismatch")
    accepted = sum(row["q"][str(q)]["A"] for row, q in zip(rows, choices, strict=True))
    wall_ms = sum(row["q"][str(q)]["modeled_wall_ms"] for row, q in zip(rows, choices, strict=True))
    units = sum(row["q"][str(q)]["expert_units"] for row, q in zip(rows, choices, strict=True))
    return {
        "A": accepted,
        "modeled_wall_ms": wall_ms,
        "modeled_tps": accepted * 1000.0 / wall_ms,
        "expert_units": units,
        "unique_expert_bytes": units * EXPERT_BYTES,
        "q_choices": choices,
        "q_histogram": {str(q): count for q, count in sorted(Counter(choices).items())},
        "maximum_window_wall_ms": max(
            row["q"][str(q)]["modeled_wall_ms"]
            for row, q in zip(rows, choices, strict=True)
        ),
    }


def best_fixed_q(rows: list[dict]) -> tuple[int, dict]:
    candidates = [(q, aggregate(rows, [q] * len(rows))) for q in WIDTHS]
    return max(candidates, key=lambda item: (item[1]["modeled_tps"], -item[0]))


def optimal_ratio_choices(rows: list[dict]) -> tuple[list[int], dict]:
    if not rows:
        raise ValueError("PW-0214 ratio oracle requires rows")
    choices = [8] * len(rows)
    rate_per_ms = aggregate(rows, choices)["modeled_tps"] / 1000.0
    for _ in range(100):
        choices = [
            max(
                WIDTHS,
                key=lambda q: (
                    row["q"][str(q)]["A"]
                    - rate_per_ms * row["q"][str(q)]["modeled_wall_ms"],
                    -q,
                ),
            )
            for row in rows
        ]
        result = aggregate(rows, choices)
        next_rate = result["modeled_tps"] / 1000.0
        if abs(next_rate - rate_per_ms) <= 1.0e-15:
            return choices, result
        rate_per_ms = next_rate
    raise ValueError("PW-0214 ratio oracle did not converge")


def analyze(manifest_path: Path) -> dict:
    if sha256_file(manifest_path) != MANIFEST_SHA256:
        raise ValueError("PW-0214 corpus manifest hash mismatch")
    manifest = json.loads(manifest_path.read_text())
    if (
        manifest.get("schema_version") != 1
        or manifest.get("builder_git_dirty") is not False
        or manifest.get("control", {}).get("windows") != 32
        or len(manifest.get("primary_windows", [])) != 32
    ):
        raise ValueError("PW-0214 corpus authority mismatch")
    reports = {}
    source_identities = {}
    for source in manifest["sources"]:
        category = source["category"]
        if category not in CATEGORIES or category in reports:
            raise ValueError("PW-0214 source category mismatch")
        for path_key, hash_key in (
            ("report_file", "report_sha256"),
            ("progress_file", "progress_sha256"),
            ("hidden_file", "hidden_sha256"),
        ):
            if sha256_file(Path(source[path_key])) != source[hash_key]:
                raise ValueError(f"PW-0214 {path_key} hash mismatch")
        reports[category] = json.loads(Path(source["report_file"]).read_text())
        source_identities[category] = {
            key: source[key]
            for key in (
                "report_sha256",
                "progress_sha256",
                "hidden_sha256",
                "generation_commit",
            )
        }

    rows = []
    for window in manifest["primary_windows"]:
        category = window["category"]
        transaction = reports[category]["transactions"][window["transaction_index"]]
        if (
            transaction["proposal_token_ids"] != window["proposal_token_ids"]
            or transaction["posterior_token_ids"] != window["posterior_token_ids"]
            or len(transaction["emitted_token_ids"]) != window["A"]
            or transaction["proposal_wall_ms"] != window["proposal_wall_ms"]
            or transaction["verification_wall_ms"] != window["verification_wall_ms"]
        ):
            raise ValueError("PW-0214 window/report mismatch")
        q8_units = expert_units(transaction, 8)
        q8_acquisition_ms = q8_units * COLD_SINGLE_EXPERT_ACQUISITION_MS
        residual_q8_verifier_ms = window["verification_wall_ms"] - q8_acquisition_ms
        if residual_q8_verifier_ms <= 0:
            raise ValueError("PW-0214 measured verifier wall is below acquisition model")
        q_rows = {}
        for q in WIDTHS:
            units = expert_units(transaction, q)
            proposal_ms = window["proposal_wall_ms"] * (q - 1) / 7
            acquisition_ms = units * COLD_SINGLE_EXPERT_ACQUISITION_MS
            shared_spine_and_compute_ms = residual_q8_verifier_ms * q / 8
            verifier_ms = acquisition_ms + shared_spine_and_compute_ms
            accepted = accepted_tokens(
                transaction["proposal_token_ids"], transaction["posterior_token_ids"], q
            )
            q_rows[str(q)] = {
                "q": q,
                "A": accepted,
                "expert_units": units,
                "proposal_wall_ms": proposal_ms,
                "acquisition_wall_ms": acquisition_ms,
                "shared_spine_and_compute_wall_ms": shared_spine_and_compute_ms,
                "verifier_wall_ms": verifier_ms,
                "modeled_wall_ms": proposal_ms + verifier_ms,
                "modeled_tps": accepted * 1000.0 / (proposal_ms + verifier_ms),
            }
        if q_rows["8"]["A"] != window["A"]:
            raise ValueError("PW-0214 q8 acceptance reconstruction mismatch")
        rows.append(
            {
                "corpus_index": window["corpus_index"],
                "category": category,
                "transaction_index": window["transaction_index"],
                "q": q_rows,
            }
        )

    calibration = [row for row in rows if row["transaction_index"] <= 4]
    holdout = [row for row in rows if row["transaction_index"] >= 5]
    if len(calibration) != 16 or len(holdout) != 16:
        raise ValueError("PW-0214 calibration/holdout split mismatch")

    fixed = {str(q): aggregate(holdout, [q] * len(holdout)) for q in WIDTHS}
    calibration_fixed_q, calibration_fixed_fit = best_fixed_q(calibration)
    calibration_fixed_holdout = aggregate(holdout, [calibration_fixed_q] * len(holdout))
    category_q = {}
    for category in CATEGORIES:
        category_calibration = [row for row in calibration if row["category"] == category]
        category_q[category] = best_fixed_q(category_calibration)[0]
    category_choices = [category_q[row["category"]] for row in holdout]
    category_policy = aggregate(holdout, category_choices)

    local_oracle_choices = [
        max(WIDTHS, key=lambda q: (row["q"][str(q)]["modeled_tps"], -q))
        for row in holdout
    ]
    local_oracle = aggregate(holdout, local_oracle_choices)
    offline_choices, offline_oracle = optimal_ratio_choices(holdout)
    previous_choices = []
    by_category = {category: [] for category in CATEGORIES}
    for row in rows:
        by_category[row["category"]].append(row)
    prior_best = {
        category: max(
            WIDTHS,
            key=lambda q: (by_category[category][3]["q"][str(q)]["modeled_tps"], -q),
        )
        for category in CATEGORIES
    }
    for row in holdout:
        category = row["category"]
        previous_choices.append(prior_best[category])
        prior_best[category] = max(
            WIDTHS,
            key=lambda q: (row["q"][str(q)]["modeled_tps"], -q),
        )
    previous_window = aggregate(holdout, previous_choices)

    q8 = fixed["8"]
    def gain(candidate: dict) -> float:
        return candidate["modeled_tps"] / q8["modeled_tps"] - 1.0

    category_slices = {}
    every_category_improves = True
    for category in CATEGORIES:
        category_rows = [row for row in holdout if row["category"] == category]
        selected = aggregate(category_rows, [category_q[category]] * len(category_rows))
        control = aggregate(category_rows, [8] * len(category_rows))
        category_gain = selected["modeled_tps"] / control["modeled_tps"] - 1.0
        every_category_improves &= category_gain > 0.0
        category_slices[category] = {
            "selected_q": category_q[category],
            "gain_vs_q8": category_gain,
            "candidate": selected,
            "q8_control": control,
        }

    offline_gain = gain(offline_oracle)
    category_gain = gain(category_policy)
    tail_not_increased = (
        category_policy["maximum_window_wall_ms"] <= q8["maximum_window_wall_ms"]
    )
    cheap_falsifier_passed = offline_gain > 0.0
    implementation_gate_passed = (
        category_gain >= REQUIRED_CAUSAL_GAIN
        and every_category_improves
        and tail_not_increased
    )
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, text=True, capture_output=True
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"], check=True, text=True, capture_output=True
        ).stdout
    )
    return {
        "schema_version": 1,
        "evidence_class": "pw0214_corrected_cost_adaptive_verification_horizon_oracle",
        "implementation": {"commit": commit, "dirty": dirty},
        "identities": {
            "corpus_manifest_sha256": MANIFEST_SHA256,
            "sources": source_identities,
        },
        "model": {
            "q": list(WIDTHS),
            "expert_bytes": EXPERT_BYTES,
            "cold_single_expert_acquisition_ms": COLD_SINGLE_EXPERT_ACQUISITION_MS,
            "proposal_rule": "measured q8 same-model proposal wall multiplied by (q-1)/7",
            "verifier_rule": "exact prefix expert units times cold acquisition plus measured q8 residual shared-spine/compute wall multiplied by q/8",
            "required_causal_gain": REQUIRED_CAUSAL_GAIN,
        },
        "split": {
            "calibration": "transactions 1-4 in each category",
            "holdout": "transactions 5-8 in each category",
        },
        "fixed_q_holdout": fixed,
        "calibration_fixed": {
            "selected_q": calibration_fixed_q,
            "calibration_fit": calibration_fixed_fit,
            "holdout": calibration_fixed_holdout,
            "holdout_gain_vs_q8": gain(calibration_fixed_holdout),
        },
        "category_calibration_policy": {
            "selected_q": category_q,
            "holdout": category_policy,
            "holdout_gain_vs_q8": category_gain,
            "by_category": category_slices,
            "every_category_improves": every_category_improves,
            "tail_not_increased": tail_not_increased,
        },
        "previous_window_control": {
            **previous_window,
            "gain_vs_q8": gain(previous_window),
        },
        "local_per_window_future_oracle": {
            **local_oracle,
            "gain_vs_q8": gain(local_oracle),
        },
        "offline_future_oracle": {
            **offline_oracle,
            "gain_vs_q8": offline_gain,
        },
        "gates": {
            "cheap_falsifier_passed": cheap_falsifier_passed,
            "offline_ceiling_reaches_required_causal_gain": offline_gain
            >= REQUIRED_CAUSAL_GAIN,
            "implementation_gate_passed": implementation_gate_passed,
        },
        "status": "complete",
        "decision": (
            "authorize_runtime_policy"
            if implementation_gate_passed
            else "reject_runtime_policy; preserve_offline_and_code_slice_horizon_gains"
        ),
        "limitations": "modeled same-model q8-prefix transaction economics over corrected 32-window text corpus; no native-MTP q-specific proposal walls, multimodal/long context, endpoint execution, or TPS",
        "performance_claim": None,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    try:
        result = analyze(arguments.manifest)
        atomic_write_new(arguments.output, canonical_json(result))
        print(json.dumps({"output": str(arguments.output), "decision": result["decision"]}))
        return 0
    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        json.JSONDecodeError,
        subprocess.SubprocessError,
    ) as error:
        print(json.dumps({"error": str(error)}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
