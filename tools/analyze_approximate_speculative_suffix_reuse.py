#!/usr/bin/env python3
"""Authenticate and adjudicate PW-0174's approximate suffix-reuse audit."""

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


TARGET_SHA256 = "91fe6e0441bb0a0e1ab0852db60fb575d131b61ff069002c9c333f9b776e4950"
PW0170_SHA256 = "c8eba5c4348378177d0d297b8eb4713fd9be71aa2f5a7c2790895c35859af5af"
PW0173_SHA256 = "15ec2cfa3ea80a3914ce500f3cb8288a2149cc1948469aeecde04922f6f7a16d"
ASD_PAPER_SHA256 = "76813e2e94d7be83d964df710729897e728cf7f25e9c330a3cf5aa502ff91724"
MINIMUM_REQUIRED_A = 56


def paper_observation() -> dict:
    return {
        "paper": "Approximate Speculative Decoding",
        "arxiv": "2608.03447",
        "canonical_url": "https://arxiv.org/pdf/2608.03447",
        "source_coordinates": "Experimental Setup; Tables 1-3; Figures 5-7",
        "exactness": "L3_changed_trajectory_from_budgeted_low_regret_mismatch_acceptance",
        "primary_target": "Qwen3-14B",
        "primary_drafter": "DSpark-14B-block7",
        "draft_tokens": 7,
        "granted_target_bonus_tokens": 1,
        "maximum_granted_accepted_path": 8,
        "selected_request_regret_budget_B": 8,
        "selected_local_regret_gate_g": 0.25,
        "selected_per_block_exception_cap_M": 2,
        "mean_strict_accepted_length": 3.85,
        "mean_asd_accepted_length": 4.20,
        "mean_throughput_gain_percent_over_strict": 7.78,
        "maximum_throughput_gain_percent_over_strict": 15.26,
        "maximum_reported_accepted_length_gain_tokens": 0.67,
        "reported_hash_divergence_above_95_percent_on_named_primary_tasks": True,
        "worst_reported_task_score_point_change_percentage_points": -1.52,
        "hosted_top20_logprob_gate_reported": False,
        "native_image_audio_video_mixed_slices_reported": False,
        "million_context_slice_reported": False,
        "prismwing_capability_confidence_intervals_reported": False,
        "direct_mimo_drafter_or_verifier_result_reported": False,
    }


def validate_predecessors(target: str, pw0170: dict, pw0173: dict) -> None:
    if (
        "USD $500 total" not in target
        or "mean Jensen-Shannon divergence at most 0.01" not in target
        or "L3 — Bounded approximation" not in target
    ):
        raise ValueError("PW-0174 TARGET authority mismatch")
    scenarios = {
        (
            row.get("lanes"),
            row.get("granted_nameplate_bytes_per_second_per_lane"),
        ): (
            row.get("targets", {}).get("34.3", {}).get("minimum_integer_A"),
            row.get("targets", {}).get("50.0", {}).get("minimum_integer_A"),
        )
        for row in pw0170.get("storage_scenarios", [])
        if row.get("lanes") == 4
    }
    if scenarios.get((4, 3_500_000_000.0)) != (56, 81):
        raise ValueError("PW-0174 inherited minimum A mismatch")
    if (
        pw0173.get("decision")
        != "reject_audited_released_configurations_as_direct_pw0170_proposer;retain_only_unproven_new_mimo_specific_q137_branch"
    ):
        raise ValueError("PW-0174 predecessor disposition mismatch")


def adjudicate(observation: dict) -> dict:
    if (
        observation.get("draft_tokens") != 7
        or observation.get("granted_target_bonus_tokens") != 1
        or observation.get("maximum_granted_accepted_path") != 8
        or observation.get("selected_request_regret_budget_B") != 8
    ):
        raise ValueError("PW-0174 ASD structural transcription mismatch")
    missing_gates = [
        name
        for name in (
            "hosted_top20_logprob_gate_reported",
            "native_image_audio_video_mixed_slices_reported",
            "million_context_slice_reported",
            "prismwing_capability_confidence_intervals_reported",
            "direct_mimo_drafter_or_verifier_result_reported",
        )
        if observation.get(name) is not True
    ]
    structural_pass = observation["maximum_granted_accepted_path"] >= MINIMUM_REQUIRED_A
    fidelity_pass = not missing_gates
    return {
        "minimum_required_A": MINIMUM_REQUIRED_A,
        "maximum_granted_accepted_path": observation["maximum_granted_accepted_path"],
        "structural_shortfall_tokens": (
            MINIMUM_REQUIRED_A - observation["maximum_granted_accepted_path"]
        ),
        "structural_gate_passes": structural_pass,
        "declared_L3_gate_evidence_complete": fidelity_pass,
        "missing_declared_L3_evidence": missing_gates,
        "released_configuration_passes": structural_pass and fidelity_pass,
        "residual_branch": (
            "scaled_mimo_specific_q_at_least_137_asd_with_full_hosted_and_capability_validation;feasibility_unproven"
        ),
    }


def run(paths: dict[str, Path], output: Path, commit: str) -> dict:
    if output.exists():
        raise ValueError(f"refusing to overwrite {output}")
    authenticate_implementation_commit(commit)
    started = time.perf_counter()
    safety = HostSafetyMonitor()
    expected = {
        "target": TARGET_SHA256,
        "pw0170": PW0170_SHA256,
        "pw0173": PW0173_SHA256,
        "asd_paper": ASD_PAPER_SHA256,
    }
    for name, digest in expected.items():
        if sha256_file(paths[name]) != digest:
            raise ValueError(f"PW-0174 source hash mismatch: {name}")
    target = paths["target"].read_text(errors="strict")
    pw0170 = json.loads(paths["pw0170"].read_text())
    pw0173 = json.loads(paths["pw0173"].read_text())
    validate_predecessors(target, pw0170, pw0173)
    safety.checkpoint("target_predecessors_and_asd_paper_authenticated")
    observation = paper_observation()
    result = adjudicate(observation)
    safety.checkpoint("asd_structure_and_fidelity_coverage_adjudicated")
    manifest = {
        "schema_version": 1,
        "evidence_class": "pw0174_approximate_speculative_suffix_reuse_audit",
        "commit": commit,
        "source_hashes": expected,
        "paper_observation": observation,
        "adjudication": result,
        "decision": (
            "reject_released_asd_configuration_as_direct_pw0170_proposer_and_as_prismwing_L3_evidence;"
            "retain_only_unproven_scaled_mimo_specific_q137_asd_branch"
        ),
        "purchase_authorized": False,
        "accepted_tokens": 0,
        "A": 0,
        "U": None,
        "performance_claim": None,
        "endpoint_tps": None,
        "limitations": (
            "primary-source structural and fidelity-coverage audit only; published cross-model "
            "throughput and task scores are not MiMo measurements, hosted distributional evidence, "
            "universal bounds, accepted Prismwing timing, or endpoint TPS"
        ),
    }
    safety.release_checkpoint("paper_and_predecessor_inputs_released", ["ASD paper", "predecessor reports"])
    safety.checkpoint("final_service_health")
    manifest["safety"] = safety.evidence()
    manifest["complete_wall_ms"] = (time.perf_counter() - started) * 1000.0
    atomic_write_new(output, canonical_json(manifest))
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    for name in ("target", "pw0170", "pw0173", "asd_paper"):
        parser.add_argument(name, type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("commit")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    names = ("target", "pw0170", "pw0173", "asd_paper")
    report = run({name: getattr(args, name) for name in names}, args.output, args.commit)
    print(canonical_json(report), end="")


if __name__ == "__main__":
    main()
