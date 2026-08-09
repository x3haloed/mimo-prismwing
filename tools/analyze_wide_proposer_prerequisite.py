#!/usr/bin/env python3
"""Run PW-0152's authenticated wide-proposer prerequisite analysis."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import platform
import time

try:
    from tools.analyze_owned_epyc_companion_envelope import (
        authenticate_implementation_commit,
    )
    from tools.analyze_pw0116_corpus import sha256_file
    from tools.host_safety import HostSafetyMonitor
    from tools.openrouter_reference import atomic_write_new, canonical_json
except ModuleNotFoundError:
    from analyze_owned_epyc_companion_envelope import authenticate_implementation_commit
    from analyze_pw0116_corpus import sha256_file
    from host_safety import HostSafetyMonitor
    from openrouter_reference import atomic_write_new, canonical_json


REVISION = "63651580ca774f8504f676040460aed3e1244ac1"
PW0151_SHA256 = "d6919e47f0f4495ccac2ad56ebcfe6662b3309aebd3296c6b546a50836829cb1"
PW0150_SHA256 = "72051c021ae1d93989508b0423ab1b0811072c24799b8e986d4543b4a513f04e"
DFLASH_PDF_SHA256 = "ffa514e6ce180eb1f7a39c49372f3b8170b99f8bc142d4a4daa0f087bf2ceb91"
WIDE_Q = 137
TARGETS = (34.3, 50.0)
PUBLISHED_DFLASH_TABLE6 = (
    {"draft_layers": 3, "task": "Math500", "tau": 5.64},
    {"draft_layers": 3, "task": "HumanEval", "tau": 4.61},
    {"draft_layers": 3, "task": "MT-Bench", "tau": 3.18},
    {"draft_layers": 5, "task": "Math500", "tau": 5.99},
    {"draft_layers": 5, "task": "HumanEval", "tau": 4.94},
    {"draft_layers": 5, "task": "MT-Bench", "tau": 3.37},
    {"draft_layers": 8, "task": "Math500", "tau": 6.33},
    {"draft_layers": 8, "task": "HumanEval", "tau": 5.29},
    {"draft_layers": 8, "task": "MT-Bench", "tau": 3.50},
)


def expected_accepted_positions(q: int, conditional_match_probability: float) -> float:
    """Constant-independent-match diagnostic including the target anchor."""
    if q <= 0:
        raise ValueError("q must be positive")
    if not 0.0 <= conditional_match_probability <= 1.0:
        raise ValueError("conditional match probability must be in [0, 1]")
    return math.fsum(
        conditional_match_probability**index for index in range(q)
    )


def solve_conditional_match_probability(q: int, expected_a: float) -> float:
    if not 1.0 <= expected_a <= q:
        raise ValueError("expected A must be in [1, q]")
    low = 0.0
    high = 1.0
    for _ in range(100):
        middle = (low + high) / 2.0
        if expected_accepted_positions(q, middle) < expected_a:
            low = middle
        else:
            high = middle
    return (low + high) / 2.0


def linear_block_shape(block_size: int, q: int) -> dict:
    if block_size <= 1 or q <= 0:
        raise ValueError("block size must exceed one and q must be positive")
    transactions = math.ceil(q / block_size)
    return {
        "block_size_including_clean_target_anchor": block_size,
        "maximum_A_per_target_transaction": block_size,
        "can_reach_q137_34_3_requirement_in_one_transaction": block_size >= 86,
        "can_reach_q137_50_requirement_in_one_transaction": block_size >= 125,
        "blocks_to_span_q_positions": transactions,
        "target_transactions_to_span_q_positions": transactions,
        "why_transactions_do_not_compose": (
            "the next conventional block requires the clean target bonus anchor "
            "emitted by the preceding verification"
        ),
    }


def tree_shape(q: int, required_a: int) -> dict:
    if q <= 0 or not 1 <= required_a <= q:
        raise ValueError("tree shape requires 1 <= A <= q")
    return {
        "q_nodes": q,
        "required_A": required_a,
        "minimum_root_to_leaf_depth_including_anchor": required_a,
        "maximum_off_path_nodes": q - required_a,
        "minimum_fraction_of_nodes_on_accepted_path": required_a / q,
    }


def _named_q137_scenario(pw0151: dict) -> dict:
    if (
        pw0151.get("evidence_class") != "pw0151_owned_epyc_companion_envelope"
        or pw0151.get("revision") != REVISION
    ):
        raise ValueError("PW-0151 report authority mismatch")
    windows = pw0151.get("route_windows", {}).get(str(WIDE_Q), [])
    if len(windows) != 1 or windows[0].get("q") != WIDE_Q:
        raise ValueError("PW-0151 q=137 window mismatch")
    scenarios = [
        row
        for row in windows[0].get("scenarios", [])
        if row.get("lanes") == 4
        and row.get("granted_nameplate_bytes_per_second_per_lane") == 2.5e9
    ]
    if len(scenarios) != 1:
        raise ValueError("PW-0151 named four-lane scenario mismatch")
    scenario = scenarios[0]
    requirements = {
        float(target): scenario.get("targets", {}).get(str(target), {}).get(
            "minimum_integer_A"
        )
        for target in TARGETS
    }
    if requirements != {34.3: 86, 50.0: 125}:
        raise ValueError("PW-0151 acceptance prerequisite changed")
    return scenario


def validate_supplied_proposer_report(pw0150: dict) -> int:
    proposed = pw0150.get("exported_mask_proposed_block_token_ids", [])
    if (
        pw0150.get("evidence_class")
        != "pw0150_exported_mask_dflash_control_analysis"
        or pw0150.get("revision") != REVISION
        or pw0150.get("A") != 1
        or pw0150.get("accepted_tokens") != 0
        or len(proposed) != 8
    ):
        raise ValueError("PW-0150 supplied proposer authority mismatch")
    return len(proposed)


def _authenticate_sources(pw0151_path: Path, pw0150_path: Path, pdf_path: Path) -> tuple[dict, dict]:
    expected = {
        pw0151_path: PW0151_SHA256,
        pw0150_path: PW0150_SHA256,
        pdf_path: DFLASH_PDF_SHA256,
    }
    for path, digest in expected.items():
        if sha256_file(path) != digest:
            raise ValueError(f"PW-0152 source hash mismatch: {path.name}")
    if pdf_path.read_bytes()[:5] != b"%PDF-":
        raise ValueError("DFlash source is not a PDF")
    pw0151 = json.loads(pw0151_path.read_text())
    pw0150 = json.loads(pw0150_path.read_text())
    _named_q137_scenario(pw0151)
    validate_supplied_proposer_report(pw0150)
    return pw0151, pw0150


def run(
    pw0151_path: Path,
    pw0150_path: Path,
    dflash_pdf_path: Path,
    output_path: Path,
    commit: str,
) -> dict:
    if output_path.exists():
        raise ValueError(f"refusing to overwrite {output_path}")
    authenticate_implementation_commit(commit)
    started = time.perf_counter()
    safety = HostSafetyMonitor()
    pw0151, pw0150 = _authenticate_sources(
        pw0151_path, pw0150_path, dflash_pdf_path
    )
    named_scenario = _named_q137_scenario(pw0151)
    safety.checkpoint("source_evidence_authenticated")

    required = {
        str(target): named_scenario["targets"][str(target)]["minimum_integer_A"]
        for target in TARGETS
    }
    target_diagnostics = {}
    for target in TARGETS:
        required_a = required[str(target)]
        probability = solve_conditional_match_probability(WIDE_Q, required_a)
        target_diagnostics[str(target)] = {
            "required_A": required_a,
            "required_fraction_of_q": required_a / WIDE_Q,
            "constant_independent_conditional_match_probability": probability,
            "constant_independent_mismatch_probability": 1.0 - probability,
            "tree_shape": tree_shape(WIDE_Q, required_a),
        }

    published = []
    for row in PUBLISHED_DFLASH_TABLE6:
        probability = solve_conditional_match_probability(16, row["tau"])
        published.append(
            {
                **row,
                "block_size": 16,
                "constant_independent_conditional_match_probability": probability,
                "constant_independent_mismatch_probability": 1.0 - probability,
                "diagnostic_expected_A_if_probability_held_for_q137": (
                    expected_accepted_positions(WIDE_Q, probability)
                ),
            }
        )
    strongest = max(published, key=lambda row: row["tau"])
    required_50_mismatch = target_diagnostics["50.0"][
        "constant_independent_mismatch_probability"
    ]
    published_mismatch = strongest["constant_independent_mismatch_probability"]
    safety.checkpoint("structural_and_diagnostic_analysis_complete")
    safety.release_checkpoint(
        "source_documents_released",
        ["PW-0151 report", "PW-0150 report", "DFlash PDF identity view"],
    )
    safety.checkpoint("final_service_health")

    report = {
        "schema_version": 1,
        "evidence_class": "pw0152_wide_proposer_acceptance_prerequisite",
        "revision": REVISION,
        "commit": commit,
        "source_hashes": {
            "pw0151_analysis_sha256": PW0151_SHA256,
            "pw0150_analysis_sha256": PW0150_SHA256,
            "dflash_arxiv_2602_06036v2_pdf_sha256": DFLASH_PDF_SHA256,
        },
        "source_authority": {
            "dflash_version": "arXiv:2602.06036v2; ICML 2026 camera-ready",
            "dflash_url": "https://arxiv.org/abs/2602.06036v2",
            "bound_semantics": [
                "tau includes the target bonus token and is at most gamma+1",
                "the clean target bonus token is the first position of each block",
                "the draft predicts block_size-1 following positions in parallel",
                "the main DFlash experiments use block size 16",
            ],
        },
        "pw0151_named_scenario": {
            "q": WIDE_Q,
            "lanes": named_scenario["lanes"],
            "aggregate_storage_bytes_per_second": named_scenario[
                "granted_aggregate_bytes_per_second"
            ],
            "serial_expert_plus_matrix_floor_seconds": named_scenario[
                "serial_expert_plus_matrix_floor_seconds"
            ],
            "required_A": required,
        },
        "linear_block_shapes": {
            "supplied_mimo_width_8": linear_block_shape(8, WIDE_Q),
            "published_dflash_width_16": linear_block_shape(16, WIDE_Q),
            "hypothetical_width_137": linear_block_shape(137, WIDE_Q),
        },
        "target_diagnostics": target_diagnostics,
        "published_dflash_table6": published,
        "strongest_published_row": strongest,
        "diagnostic_mismatch_reduction_factor_required_vs_strongest_published": (
            published_mismatch / required_50_mismatch
        ),
        "supplied_mimo_proposer": {
            "q": len(pw0150["exported_mask_proposed_block_token_ids"]),
            "A": pw0150["A"],
            "accepted_draft_suffix_tokens": pw0150["accepted_tokens"],
            "decision": pw0150["decision"],
        },
        "structural_findings": [
            "width-8 and width-16 blocks cannot reach A=86 or A=125 in one target transaction",
            "ordinary block chaining inserts target verification to obtain the next clean bonus anchor",
            "a q=137 tree for A=125 needs depth at least 125 and leaves at most 12 off-path nodes",
            "a q=137 tree for A=86 needs depth at least 86 and leaves at most 51 off-path nodes",
        ],
        "decision": (
            "reject_conventional_width_8_and_width_16_dflash_and_ordinary_chaining;"
            "retain_only_distinct_unproven_q137_or_depth125_base_aligned_proposer"
        ),
        "limitations": [
            "published acceptance values use other target models and are scale evidence, not a MiMo bound",
            "the constant independent-match model is diagnostic and is not a measured acceptance law",
            "a newly trained long-block or long-depth proposer remains logically possible",
            "no draft training, target execution, endpoint timing, or hardware measurement occurred",
        ],
        "A": 0,
        "accepted_tokens": 0,
        "performance_claim": "none; analytical proposer prerequisite only",
        "gates_passed": True,
        "platform": platform.platform(),
        "complete_wall_ms": (time.perf_counter() - started) * 1000.0,
        "safety": safety.evidence(),
    }
    atomic_write_new(output_path, canonical_json(report))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pw0151", type=Path, required=True)
    parser.add_argument("--pw0150", type=Path, required=True)
    parser.add_argument("--dflash-pdf", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--commit", required=True)
    arguments = parser.parse_args()
    result = run(
        arguments.pw0151,
        arguments.pw0150,
        arguments.dflash_pdf,
        arguments.output,
        arguments.commit,
    )
    print(canonical_json(result).decode(), end="")


if __name__ == "__main__":
    main()
