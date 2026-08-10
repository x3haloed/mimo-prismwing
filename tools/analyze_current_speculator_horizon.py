#!/usr/bin/env python3
"""Authenticate and adjudicate PW-0173's current speculator horizon audit."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

try:
    from tools.analyze_owned_epyc_companion_envelope import authenticate_implementation_commit
    from tools.analyze_pw0116_corpus import sha256_file
    from tools.host_safety import HostSafetyMonitor
    from tools.openrouter_reference import atomic_write_new, canonical_json
except ModuleNotFoundError:
    from analyze_owned_epyc_companion_envelope import authenticate_implementation_commit
    from analyze_pw0116_corpus import sha256_file
    from host_safety import HostSafetyMonitor
    from openrouter_reference import atomic_write_new, canonical_json


PW0170_SHA256 = "c8eba5c4348378177d0d297b8eb4713fd9be71aa2f5a7c2790895c35859af5af"
SOURCE_HASHES = {
    "eagle3": "8a178337a1b05067907167dd81a43a7184596d71d4b795d6a0c73e6235fe1a27",
    "p_eagle": "35310a5280cd01e9c4d85be65aef9506728a69b17f09560ade26716a86dfbcd7",
    "angelspec": "5eac19e3ac72136bdeab1f4d18e83c9e7a99ec34babe63514b812efebb69e324",
    "bastion": "90d3c6045f7e177838e8de81cb760ca6f3250b86a01ae62ab76fb8a9a08386d9",
}

REQUIRED_ACCEPTED = {
    "fast_storage_34_3_tps": 56,
    "slow_storage_34_3_tps": 77,
    "fast_storage_50_tps": 81,
    "slow_storage_50_tps": 113,
}


def published_configurations() -> list[dict]:
    """Return source-transcribed configurations with deliberately favorable path grants."""
    return [
        {
            "name": "EAGLE-3",
            "paper": "arXiv:2503.01840",
            "canonical_url": "https://arxiv.org/pdf/2503.01840",
            "source_coordinate": "Appendix A and Table 1",
            "target_scope": "Vicuna-13B, Llama-3.1-8B, Llama-3.3-70B, and DeepSeek-R1-Distill-Llama-8B",
            "decoding_scope": "temperature 0 and 1; five text benchmarks",
            "configured_tree_depth": 8,
            "total_tree_nodes": 60,
            "granted_target_bonus_tokens": 1,
            "maximum_granted_accepted_path": 9,
            "reported_mean_accepted_length": 6.62,
            "largest_reported_slice_mean_accepted_length": 7.54,
            "direct_mimo_checkpoint_available": False,
            "prerequisite": "target-specific trained EAGLE-3 weights and target hidden states",
        },
        {
            "name": "P-EAGLE",
            "paper": "arXiv:2602.01469",
            "canonical_url": "https://arxiv.org/pdf/2602.01469",
            "source_coordinate": "Tables 1, 9, and 11",
            "target_scope": "GPT-OSS-20B/120B and Qwen3-Coder-30B",
            "decoding_scope": "speculation depth K=5; HumanEval, MT-Bench, and GSM8K",
            "configured_tree_depth": 5,
            "total_tree_nodes": 5,
            "granted_target_bonus_tokens": 1,
            "maximum_granted_accepted_path": 6,
            "reported_mean_accepted_length": 3.7,
            "largest_reported_slice_mean_accepted_length": 4.5,
            "direct_mimo_checkpoint_available": False,
            "prerequisite": "target-specific 2- or 4-layer P-EAGLE weights and fused target hidden states",
        },
        {
            "name": "AngelSpec DFly",
            "paper": "arXiv:2607.25852",
            "canonical_url": "https://arxiv.org/pdf/2607.25852",
            "source_coordinate": "Tables 3 and 5 plus Appendix B.3",
            "target_scope": "Qwen3-8B and Hy3-A21B",
            "decoding_scope": "block size 8; temperature 0 and 1; six text benchmarks",
            "configured_tree_depth": 7,
            "total_tree_nodes": 7,
            "granted_target_bonus_tokens": 1,
            "maximum_granted_accepted_path": 8,
            "reported_mean_accepted_length": 5.41,
            "largest_reported_slice_mean_accepted_length": 6.42,
            "direct_mimo_checkpoint_available": False,
            "prerequisite": "target-specific five-layer DFly weights, hidden-state generation, and online target verification",
        },
        {
            "name": "BASTION",
            "paper": "arXiv:2605.29727",
            "canonical_url": "https://arxiv.org/pdf/2605.29727",
            "source_coordinate": "Table 1, Figure 2, and Appendix H",
            "target_scope": "Qwen3-4B/8B with target-specific DFlash drafters",
            "decoding_scope": "block size 16; temperature 0 and 1; eight text benchmarks",
            "configured_tree_depth": 16,
            "total_tree_nodes": 60,
            "granted_target_bonus_tokens": 1,
            "maximum_granted_accepted_path": 17,
            "reported_mean_accepted_length": 8.56,
            "largest_reported_slice_mean_accepted_length": 10.60,
            "direct_mimo_checkpoint_available": False,
            "prerequisite": "target-specific block-diffusion drafter, full position-wise distributions, and target verification",
        },
    ]


def validate_pw0170(report: dict) -> None:
    scenarios = report.get("storage_scenarios", [])
    extracted: dict[tuple[int, float], tuple[int, int]] = {}
    for row in scenarios:
        if row.get("lanes") == 4:
            speed = row.get("granted_nameplate_bytes_per_second_per_lane")
            targets = row.get("targets", {})
            if speed in (2_500_000_000.0, 3_500_000_000.0):
                extracted[(4, speed)] = (
                    targets.get("34.3", {}).get("minimum_integer_A"),
                    targets.get("50.0", {}).get("minimum_integer_A"),
                )
    if extracted != {
        (4, 2_500_000_000.0): (77, 113),
        (4, 3_500_000_000.0): (56, 81),
    }:
        raise ValueError("PW-0173 inherited acceptance horizons mismatch")


def adjudicate(configurations: list[dict]) -> dict:
    if not configurations:
        raise ValueError("PW-0173 requires published configurations")
    for row in configurations:
        depth = row.get("configured_tree_depth")
        bonus = row.get("granted_target_bonus_tokens")
        granted = row.get("maximum_granted_accepted_path")
        if (
            not isinstance(depth, int)
            or depth <= 0
            or bonus != 1
            or granted != depth + bonus
            or not isinstance(row.get("largest_reported_slice_mean_accepted_length"), (int, float))
            or row["largest_reported_slice_mean_accepted_length"] > granted
            or row.get("direct_mimo_checkpoint_available") is not False
        ):
            raise ValueError(f"invalid published configuration: {row.get('name')}")
    strongest_path = max(configurations, key=lambda row: row["maximum_granted_accepted_path"])
    strongest_mean = max(
        configurations,
        key=lambda row: row["largest_reported_slice_mean_accepted_length"],
    )
    minimum_required = min(REQUIRED_ACCEPTED.values())
    direct_configs_rejected = strongest_path["maximum_granted_accepted_path"] < minimum_required
    return {
        "least_demanding_required_A": minimum_required,
        "strongest_published_configuration_by_granted_path": {
            "name": strongest_path["name"],
            "maximum_granted_accepted_path": strongest_path["maximum_granted_accepted_path"],
            "shortfall_tokens": minimum_required - strongest_path["maximum_granted_accepted_path"],
        },
        "strongest_reported_slice_mean": {
            "name": strongest_mean["name"],
            "mean_accepted_length": strongest_mean["largest_reported_slice_mean_accepted_length"],
        },
        "diagnostic_required_A_over_strongest_reported_slice_mean": {
            name: required / strongest_mean["largest_reported_slice_mean_accepted_length"]
            for name, required in REQUIRED_ACCEPTED.items()
        },
        "all_audited_direct_configurations_structurally_below_minimum": direct_configs_rejected,
        "residual_branch": (
            "new_mimo_specific_q_at_least_137_scaled_or_trained_proposer_only;feasibility_unproven"
        ),
    }


def run(paths: dict[str, Path], output: Path, commit: str) -> dict:
    if output.exists():
        raise ValueError(f"refusing to overwrite {output}")
    authenticate_implementation_commit(commit)
    started = time.perf_counter()
    safety = HostSafetyMonitor()
    expected = {"pw0170": PW0170_SHA256, **SOURCE_HASHES}
    for name, digest in expected.items():
        if sha256_file(paths[name]) != digest:
            raise ValueError(f"PW-0173 source hash mismatch: {name}")
    pw0170 = json.loads(paths["pw0170"].read_text())
    validate_pw0170(pw0170)
    safety.checkpoint("pw0170_and_primary_papers_authenticated")
    configurations = published_configurations()
    result = adjudicate(configurations)
    safety.checkpoint("published_horizons_adjudicated")
    manifest = {
        "schema_version": 1,
        "evidence_class": "pw0173_current_speculator_horizon_audit",
        "commit": commit,
        "source_hashes": expected,
        "required_acceptance_horizons": REQUIRED_ACCEPTED,
        "published_configurations": configurations,
        "adjudication": result,
        "decision": (
            "reject_audited_released_configurations_as_direct_pw0170_proposer;"
            "retain_only_unproven_new_mimo_specific_q137_branch"
        ),
        "purchase_authorized": False,
        "accepted_tokens": 0,
        "A": 0,
        "U": None,
        "performance_claim": None,
        "endpoint_tps": None,
        "limitations": (
            "primary-source structural and cross-model diagnostic audit only; published acceptance "
            "means are not MiMo measurements or universal bounds; no trained MiMo proposer, target "
            "verification, hardware, accepted-token timing, or endpoint TPS"
        ),
    }
    safety.release_checkpoint("paper_and_predecessor_inputs_released", ["paper PDFs", "PW-0170 report"])
    safety.checkpoint("final_service_health")
    manifest["safety"] = safety.evidence()
    manifest["complete_wall_ms"] = (time.perf_counter() - started) * 1000.0
    atomic_write_new(output, canonical_json(manifest))
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    for name in ("pw0170", "eagle3", "p_eagle", "angelspec", "bastion"):
        parser.add_argument(name, type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("commit")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    names = ("pw0170", "eagle3", "p_eagle", "angelspec", "bastion")
    report = run({name: getattr(args, name) for name in names}, args.output, args.commit)
    print(canonical_json(report), end="")


if __name__ == "__main__":
    main()
