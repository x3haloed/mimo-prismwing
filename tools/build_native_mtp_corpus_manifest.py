#!/usr/bin/env python3
"""Build the balanced, hash-bound 32-window PW-0208 corpus manifest."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.audit_native_mtp_window_corpus import PRIMARY_WINDOWS, audit, sha256

CATEGORIES = ("ordinary", "code", "multilingual", "rare_route")
PROMPTS = {
    "ordinary": "pw0208-ordinary.txt",
    "code": "pw0208-code.txt",
    "multilingual": "pw0208-multilingual.txt",
    "rare_route": "pw0208-rare-route.txt",
}
WINDOW_BYTES = 8 * 4096 * 4


def route_counts(report: dict[str, Any]) -> Counter[tuple[int, int]]:
    counts: Counter[tuple[int, int]] = Counter()
    for transaction in report["transactions"][:PRIMARY_WINDOWS]:
        for layer in transaction["verification_layer_traces"][1:]:
            for row in layer["selected_experts_by_position"]:
                counts.update((layer["layer"], expert) for expert in row)
    return counts


def git_identity(repo: Path) -> tuple[str, bool]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, check=True, text=True, capture_output=True
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"], cwd=repo, check=True, text=True, capture_output=True
        ).stdout
    )
    return commit, dirty


def build_manifest(evidence_root: Path, repo: Path) -> dict[str, Any]:
    sources = []
    reports = {}
    primary_windows = []
    for category in CATEGORIES:
        category_root = evidence_root / category
        report_path = category_root / "report.json"
        progress_path = category_root / "report.progress.jsonl"
        hidden_path = category_root / "verification-layer47-hidden.f32"
        prompt_path = repo / "evals" / "fixtures" / "requests" / PROMPTS[category]
        report = json.loads(report_path.read_text())
        result = audit(
            report_path,
            progress_path,
            hidden_path,
            category=category,
            commit=report["commit"],
            prompt_path=prompt_path,
        )
        reports[category] = report
        sources.append(
            {
                "category": category,
                "generation_commit": report["commit"],
                "prompt_file": str(prompt_path.relative_to(repo)),
                "prompt_sha256": sha256(prompt_path),
                "report_file": str(report_path),
                "report_sha256": result["report_sha256"],
                "progress_file": str(progress_path),
                "progress_sha256": result["progress_sha256"],
                "hidden_file": str(hidden_path),
                "hidden_sha256": result["hidden_sha256"],
                "captured_windows": result["windows"],
                "complete_wall_ms": result["complete_wall_ms"],
                "peak_resident_bytes": result["peak_resident_bytes"],
            }
        )
        for transaction in report["transactions"][:PRIMARY_WINDOWS]:
            index = transaction["index"]
            accepted = len(transaction["emitted_token_ids"])
            primary_windows.append(
                {
                    "corpus_index": len(primary_windows),
                    "category": category,
                    "transaction_index": index,
                    "hidden_byte_offset": index * WINDOW_BYTES,
                    "hidden_byte_length": WINDOW_BYTES,
                    "proposal_token_ids": transaction["proposal_token_ids"],
                    "posterior_token_ids": transaction["posterior_token_ids"],
                    "verifier_authorized_token_ids": transaction["verifier_authorized_token_ids"],
                    "A": accepted,
                    "U": transaction["U"],
                    "A_per_U": accepted / transaction["U"],
                    "proposal_converged": transaction["proposal_converged"],
                    "proposal_wall_ms": transaction["proposal_wall_ms"],
                    "verification_wall_ms": transaction["verification_wall_ms"],
                }
            )

    controls = sum((route_counts(reports[category]) for category in CATEGORIES[:-1]), Counter())
    rare = route_counts(reports["rare_route"])
    novel = Counter({pair: count for pair, count in rare.items() if controls[pair] == 0})
    low_frequency = Counter({pair: count for pair, count in rare.items() if controls[pair] <= 1})
    total_a = sum(window["A"] for window in primary_windows)
    total_u = sum(window["U"] for window in primary_windows)
    category_metrics = {}
    for category in CATEGORIES:
        windows = [window for window in primary_windows if window["category"] == category]
        category_a = sum(window["A"] for window in windows)
        category_u = sum(window["U"] for window in windows)
        category_metrics[category] = {
            "windows": len(windows),
            "sum_A": category_a,
            "sum_U": category_u,
            "sum_A_per_sum_U": category_a / category_u,
        }
    commit, dirty = git_identity(repo)
    return {
        "schema_version": 1,
        "evidence_class": "pw0208_balanced_corrected_native_mtp_window_corpus",
        "semantic": "first_eight_chronological_width_eight_corrected_verifier_windows_per_category",
        "builder_commit": commit,
        "builder_git_dirty": dirty,
        "window_shape": [8, 4096],
        "hidden_dtype": "float32_little_endian",
        "sources": sources,
        "primary_windows": primary_windows,
        "control": {
            "windows": len(primary_windows),
            "sum_A": total_a,
            "sum_U": total_u,
            "sum_A_per_sum_U": total_a / total_u,
            "category_metrics": category_metrics,
        },
        "rare_route_evidence": {
            "control_categories": list(CATEGORIES[:-1]),
            "control_layer_expert_pairs": len(controls),
            "rare_route_layer_expert_pairs": len(rare),
            "novel_layer_expert_pairs": len(novel),
            "novel_route_selections": sum(novel.values()),
            "novel_routed_layers": sorted({layer for layer, _ in novel}),
            "control_frequency_at_most_one_pairs": len(low_frequency),
            "control_frequency_at_most_one_selections": sum(low_frequency.values()),
            "top_novel_pairs": [
                {"layer": layer, "expert": expert, "rare_route_selections": count}
                for (layer, expert), count in sorted(
                    novel.items(), key=lambda item: (-item[1], item[0])
                )[:32]
            ],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("evidence_root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    try:
        manifest = build_manifest(args.evidence_root, args.repo.resolve())
    except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError, subprocess.SubprocessError) as error:
        raise SystemExit(f"manifest build failed: {error}") from error
    args.output.write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        json.dumps(
            {
                "status": "complete",
                "output": str(args.output),
                "sha256": sha256(args.output),
                "windows": manifest["control"]["windows"],
                "sum_A_per_sum_U": manifest["control"]["sum_A_per_sum_U"],
                "novel_layer_expert_pairs": manifest["rare_route_evidence"]["novel_layer_expert_pairs"],
                "novel_routed_layers": len(manifest["rare_route_evidence"]["novel_routed_layers"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
