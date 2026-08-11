#!/usr/bin/env python3
"""Compute PW-0208's perfect-proposal accepted-token/expert-byte upper bound."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

from safetensors import safe_open

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.audit_native_mtp_window_corpus import sha256

WIDTHS = (4, 8)
EXPERT_BYTES = 25_171_968
REQUIRED_GAIN = 2.0
PROJECTIONS = {
    "gate_proj.weight": ("F8_E4M3", (2048, 4096)),
    "gate_proj.weight_scale_inv": ("F32", (16, 32)),
    "up_proj.weight": ("F8_E4M3", (2048, 4096)),
    "up_proj.weight_scale_inv": ("F32", (16, 32)),
    "down_proj.weight": ("F8_E4M3", (4096, 2048)),
    "down_proj.weight_scale_inv": ("F32", (32, 16)),
}


def expert_units(transaction: dict[str, Any], q: int) -> int:
    if q not in WIDTHS:
        raise ValueError("q must be four or eight")
    traces = transaction["verification_layer_traces"]
    if len(traces) != 48:
        raise ValueError("verification trace must contain 48 layers")
    units = 0
    for expected_layer, trace in enumerate(traces[1:], 1):
        if trace["layer"] != expected_layer:
            raise ValueError("route layers are not contiguous")
        rows = trace["selected_experts_by_position"][:q]
        if len(rows) != q or any(len(row) != 8 for row in rows):
            raise ValueError("route row or top-k width mismatch")
        if any(not 0 <= expert < 256 for row in rows for expert in row):
            raise ValueError("expert id outside pinned bank")
        units += len({expert for row in rows for expert in row})
    return units


def perfect_schedule_bound(rows: list[dict[str, int]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("upper bound requires windows")
    control_a = sum(row["control_A"] for row in rows)
    control_units = sum(row["q8_units"] for row in rows)
    if control_a <= 0 or control_units <= 0:
        raise ValueError("invalid control totals")
    control_rate = control_a / control_units
    fixed = {}
    for q in WIDTHS:
        units = sum(row[f"q{q}_units"] for row in rows)
        accepted = len(rows) * (q - 1)
        fixed[str(q)] = {
            "q": q,
            "perfect_max_A": accepted,
            "expert_units": units,
            "gain_vs_control": (accepted / units) / control_rate,
        }
    frontier = {0: (0, [])}
    for row in rows:
        updated: dict[int, tuple[int, list[int]]] = {}
        for units, (accepted, choices) in frontier.items():
            for q in WIDTHS:
                next_units = units + row[f"q{q}_units"]
                candidate = (accepted + q - 1, [*choices, q])
                if next_units not in updated or candidate[0] > updated[next_units][0]:
                    updated[next_units] = candidate
        frontier = updated
    gain, accepted, units, choices = max(
        (
            (candidate_a / candidate_units) / control_rate,
            candidate_a,
            candidate_units,
            candidate_choices,
        )
        for candidate_units, (candidate_a, candidate_choices) in frontier.items()
    )
    return {
        "control_A": control_a,
        "control_expert_units": control_units,
        "fixed_q": fixed,
        "perfect_per_window_q_oracle": {
            "perfect_max_A": accepted,
            "expert_units": units,
            "q_choices": choices,
            "gain_vs_control": gain,
        },
    }


def validate_expert_layout(checkpoint: Path, selected: set[tuple[int, int]]) -> str:
    index_path = checkpoint / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())["weight_map"]
    by_shard: dict[str, list[tuple[str, tuple[str, tuple[int, ...]]]]] = defaultdict(list)
    for layer, expert in selected:
        for suffix, identity in PROJECTIONS.items():
            name = f"model.layers.{layer}.mlp.experts.{expert}.{suffix}"
            shard = index.get(name)
            if shard is None:
                raise ValueError(f"checkpoint index lacks {name}")
            by_shard[shard].append((name, identity))
    for shard, entries in by_shard.items():
        with safe_open(checkpoint / shard, framework="pt", device="cpu") as source:
            for name, expected in entries:
                view = source.get_slice(name)
                actual = (view.get_dtype(), tuple(view.get_shape()))
                if actual != expected:
                    raise ValueError(f"expert tensor layout mismatch: {name}: {actual}")
    derived_bytes = (
        3 * 2048 * 4096
        + (16 * 32 + 16 * 32 + 32 * 16) * 4
    )
    if derived_bytes != EXPERT_BYTES:
        raise ValueError("expert byte derivation mismatch")
    return sha256(index_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    try:
        manifest = json.loads(args.manifest.read_text())
        if (
            manifest["schema_version"] != 1
            or manifest["builder_git_dirty"]
            or manifest["control"]["windows"] != 32
            or len(manifest["primary_windows"]) != 32
            or "complete_segmented_target_hidden_history" not in manifest["semantic"]
        ):
            raise ValueError("balanced corpus manifest gate failed")
        source_reports = {}
        for source in manifest["sources"]:
            path = Path(source["report_file"])
            if sha256(path) != source["report_sha256"]:
                raise ValueError("source report hash mismatch")
            source_reports[source["category"]] = json.loads(path.read_text())
        rows = []
        selected: set[tuple[int, int]] = set()
        for window in manifest["primary_windows"]:
            transaction = source_reports[window["category"]]["transactions"][
                window["transaction_index"]
            ]
            if len(transaction["emitted_token_ids"]) != window["A"]:
                raise ValueError("manifest/control A mismatch")
            q4_units = expert_units(transaction, 4)
            q8_units = expert_units(transaction, 8)
            for trace in transaction["verification_layer_traces"][1:]:
                selected.update(
                    (trace["layer"], expert)
                    for row in trace["selected_experts_by_position"]
                    for expert in row
                )
            rows.append(
                {
                    "corpus_index": window["corpus_index"],
                    "control_A": window["A"],
                    "q4_units": q4_units,
                    "q8_units": q8_units,
                }
            )
        bound = perfect_schedule_bound(rows)
        index_hash = validate_expert_layout(args.checkpoint, selected)
        control = bound["control_A"]
        control_units = bound["control_expert_units"]
        oracle = bound["perfect_per_window_q_oracle"]
        maximum_gain = oracle["gain_vs_control"]
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, text=True, capture_output=True
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"], check=True, text=True, capture_output=True
            ).stdout
        )
        result = {
            "schema_version": 1,
            "evidence_class": "pw0208_native_mtp_perfect_proposal_cost_upper_bound",
            "semantic": "perfect_native_mtp_q4_q8_oracle_under_existing_verifier_commit_semantics",
            "status": "cost_efficiency_gate_impossible" if maximum_gain < REQUIRED_GAIN else "execution_required",
            "commit": commit,
            "git_dirty": dirty,
            "identities": {
                "corpus_manifest_sha256": sha256(args.manifest),
                "checkpoint_index_sha256": index_hash,
            },
            "contract": {
                "windows": 32,
                "q": list(WIDTHS),
                "maximum_observable_A_by_q": {"4": 3, "8": 7},
                "reason": "the endpoint commits q-1 tokens when every draft matches and otherwise replaces the first mismatch with one verifier correction",
                "expert_bytes_per_layer_expert": EXPERT_BYTES,
                "required_gain": REQUIRED_GAIN,
            },
            "control": {
                "A": control,
                "expert_units": control_units,
                "unique_expert_bytes": control_units * EXPERT_BYTES,
                "accepted_tokens_per_unique_expert_byte": control / (control_units * EXPERT_BYTES),
            },
            "perfect_fixed_q": bound["fixed_q"],
            "perfect_per_window_q_oracle": {
                **oracle,
                "unique_expert_bytes": oracle["expert_units"] * EXPERT_BYTES,
                "accepted_tokens_per_unique_expert_byte": oracle["perfect_max_A"]
                / (oracle["expert_units"] * EXPERT_BYTES),
            },
            "maximum_possible_gain": maximum_gain,
            "required_gain": REQUIRED_GAIN,
            "gate_passed": maximum_gain >= REQUIRED_GAIN,
            "disposition": "kill PW-0208 cost-aware expert-byte promotion gate; preserve native MTP proposal-latency research as a separately named lower milestone",
            "performance_claim": None,
        }
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError, subprocess.SubprocessError) as error:
        raise SystemExit(f"analysis failed: {error}") from error
    print(
        json.dumps(
            {
                "status": result["status"],
                "output": str(args.output),
                "sha256": sha256(args.output),
                "maximum_possible_gain": maximum_gain,
                "required_gain": REQUIRED_GAIN,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
