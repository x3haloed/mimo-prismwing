#!/usr/bin/env python3
"""Run PW-0140's pooled-calibration low-count GPTQ falsification."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import platform
import time

import mlx.core as mx
import numpy as np
import torch

try:
    from tools.generate_real_layer1_expert_oracle import ShardedCheckpoint
    from tools.host_safety import HostSafetyMonitor
    from tools.openrouter_reference import atomic_write_new, canonical_json
    from tools.run_best_rank_real_expert_control import dequant_weight, load_capture, sha256_file, source_linear
    from tools.run_global_hessian_gptq_rescue import global_hessian_gptq_fixed_grid, projected_workspace_bytes
    from tools.run_group_local_gptq_three_expert_control import (
        affine_grid,
        dense_expert,
        dense_projection,
        physical_ledger,
        source_hidden,
        validate_grid_membership,
    )
    from tools.run_real_activation_affine_int4_audit import (
        CORPUS_SHA256,
        REVISION,
        VERIFICATION_SHA256,
        error_metrics,
        validate_routes,
    )
except ModuleNotFoundError:
    from generate_real_layer1_expert_oracle import ShardedCheckpoint
    from host_safety import HostSafetyMonitor
    from openrouter_reference import atomic_write_new, canonical_json
    from run_best_rank_real_expert_control import dequant_weight, load_capture, sha256_file, source_linear
    from run_global_hessian_gptq_rescue import global_hessian_gptq_fixed_grid, projected_workspace_bytes
    from run_group_local_gptq_three_expert_control import (
        affine_grid,
        dense_expert,
        dense_projection,
        physical_ledger,
        source_hidden,
        validate_grid_membership,
    )
    from run_real_activation_affine_int4_audit import (
        CORPUS_SHA256,
        REVISION,
        VERIFICATION_SHA256,
        error_metrics,
        validate_routes,
    )


PW0139_SHA256 = "83bd204c9d5c35a684cab15a4ddacf48cf9b661563fb26223eb3655d0ef4a7b5"
PW0139_ANALYSIS_SHA256 = "9aecfdcd32e535b4b9d27fcac075dfd1c9014080d624aa3f4af2c678be3f3b6c"
SAMPLES = ((24, 39, 6, 29), (24, 128, 19, 21), (46, 140, 10, 17))


def _prior_expert(prior: dict, layer: int, expert: int) -> dict:
    matches = [
        row
        for layer_report in prior.get("layer_reports", [])
        if layer_report.get("layer") == layer
        for row in layer_report.get("expert_reports", [])
        if row.get("expert") == expert
    ]
    if len(matches) != 1:
        raise ValueError("PW-0140 prior expert authority mismatch")
    return matches[0]


def _gate(reports: list[dict]) -> dict:
    rows = []
    for report in reports:
        candidate = report["pooled_validation_metrics"]
        prior = report["pw0139_routed_calibration_validation_metrics"]
        conditions = {
            "improves_pw0139": candidate["relative_l2"] < prior["relative_l2"],
            "validation_relative_l2": candidate["relative_l2"] <= 0.08,
            "maximum_validation_row_relative_l2": candidate["maximum_row_relative_l2"] <= 0.12,
            "all_projection_calibration_improve": all(
                row["candidate_calibration_metrics"]["relative_l2"]
                < row["baseline_calibration_metrics"]["relative_l2"]
                for row in report["projection_reports"].values()
            ),
        }
        rows.append(
            {
                "layer": report["layer"],
                "expert": report["expert"],
                "passes": all(conditions.values()),
                "conditions": conditions,
                "prior_validation_relative_l2": prior["relative_l2"],
                "pooled_validation_relative_l2": candidate["relative_l2"],
                "pooled_maximum_row_relative_l2": candidate["maximum_row_relative_l2"],
            }
        )
    physical = physical_ledger()
    physical_passes = (
        physical["packed_bytes_per_expert"] == 13_369_344
        and physical["additional_runtime_macs"] == 0
    )
    return {
        "passes": all(row["passes"] for row in rows) and physical_passes,
        "experts": rows,
        "physical": physical,
        "physical_passes": physical_passes,
        "thresholds": {
            "maximum_validation_relative_l2": 0.08,
            "maximum_validation_row_relative_l2": 0.12,
            "must_improve_pw0139": True,
            "all_projection_calibration_must_improve": True,
        },
    }


def run(
    checkpoint_root: Path,
    verification_path: Path,
    corpus_manifest_path: Path,
    pw0139_path: Path,
    pw0139_analysis_path: Path,
    output_path: Path,
    commit: str,
) -> dict:
    if output_path.exists():
        raise ValueError(f"refusing to overwrite {output_path}")
    if len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit):
        raise ValueError("implementation commit must be lowercase 40-hex")
    for path, expected, label in (
        (verification_path, VERIFICATION_SHA256, "checkpoint verification"),
        (corpus_manifest_path, CORPUS_SHA256, "PW-0116 corpus"),
        (pw0139_path, PW0139_SHA256, "PW-0139 report"),
        (pw0139_analysis_path, PW0139_ANALYSIS_SHA256, "PW-0139 analysis"),
    ):
        if sha256_file(path) != expected:
            raise ValueError(f"PW-0140 {label} hash mismatch")
    started = time.perf_counter()
    safety = HostSafetyMonitor()
    corpus = json.loads(corpus_manifest_path.read_text())
    prior = json.loads(pw0139_path.read_text())
    analysis = json.loads(pw0139_analysis_path.read_text())
    if (
        corpus.get("revision") != REVISION
        or prior.get("decision") != "reject_global_hessian_fixed_grid_on_full_validation"
        or analysis.get("source_report_sha256") != PW0139_SHA256
        or prior.get("holdout_unsealed")
        or analysis.get("holdout_unsealed")
    ):
        raise ValueError("PW-0140 authority identity mismatch")
    checkpoint = ShardedCheckpoint(checkpoint_root, verification_path)
    root = corpus_manifest_path.parent
    reports = []
    for layer, expert, expected_train, expected_validation in SAMPLES:
        authority = next(row for row in corpus["layers"] if row["layer"] == layer)
        validate_routes(authority)
        schedule = next(row for row in authority["expert_schedule"] if row["expert"] == expert)
        train_count = sum(position < 112 for position in schedule["positions"])
        validation_local = [
            index for index, position in enumerate(schedule["positions"]) if 112 <= position < 168
        ]
        if train_count != expected_train or len(validation_local) != expected_validation:
            raise ValueError("PW-0140 frozen sample coverage mismatch")
        offset = sum(
            len(row["positions"])
            for row in authority["expert_schedule"][: authority["expert_schedule"].index(schedule)]
        )
        moe_input = load_capture(root, authority["captures"]["moe_input"])
        expert_down = load_capture(root, authority["captures"]["expert_down"])
        validation_positions = [schedule["positions"][index] for index in validation_local]
        expected_output = np.asarray(
            expert_down[[offset + index for index in validation_local]], dtype=np.float32
        ).copy()
        prefix = f"model.layers.{layer}.mlp.experts.{expert}"
        source_weights = {
            projection: dequant_weight(checkpoint, f"{prefix}.{projection}_proj.weight")
            for projection in ("gate", "up", "down")
        }
        pooled_input = moe_input[:112]
        pooled_hidden = source_hidden(source_weights, pooled_input)
        activations = {"gate": pooled_input, "up": pooled_input, "down": pooled_hidden}
        expected = {
            "gate": source_linear(
                source_weights["gate"],
                torch.from_numpy(np.asarray(pooled_input).copy()).to(torch.bfloat16),
            ).float().numpy(),
            "up": source_linear(
                source_weights["up"],
                torch.from_numpy(np.asarray(pooled_input).copy()).to(torch.bfloat16),
            ).float().numpy(),
            "down": source_linear(
                source_weights["down"],
                torch.from_numpy(np.asarray(pooled_hidden).copy()).to(torch.bfloat16),
            ).float().numpy(),
        }
        selected = {}
        projection_reports = {}
        for projection in ("gate", "up", "down"):
            weight = source_weights[projection]
            scales, biases, baseline, baseline_codes = affine_grid(weight)
            validate_grid_membership(baseline_codes, baseline, scales, biases)
            projected = projected_workspace_bytes(weight)
            before = safety.checkpoint(f"layer_{layer}_expert_{expert}_{projection}_preflight")
            if before.process_physical_footprint_bytes + projected > safety.policy.maximum_process_physical_footprint_bytes:
                raise RuntimeError(f"PW-0140 layer {layer} expert {expert} exceeds Gate 8 headroom")
            candidate, codes, diagnostics = global_hessian_gptq_fixed_grid(
                weight, activations[projection], scales, biases
            )
            candidate_output = dense_projection(activations[projection], candidate)
            baseline_output = dense_projection(activations[projection], baseline)
            selected[projection] = candidate
            projection_reports[projection] = {
                **diagnostics,
                "projected_workspace_bytes": projected,
                "candidate_calibration_metrics": error_metrics(candidate_output, expected[projection]),
                "baseline_calibration_metrics": error_metrics(baseline_output, expected[projection]),
            }
            del scales, biases, baseline, baseline_codes, codes, candidate_output, baseline_output
            gc.collect()
            safety.release_checkpoint(
                f"layer_{layer}_expert_{expert}_{projection}_workspace_released",
                ["full Hessian", "inverse Cholesky", "permuted working weight", "projection outputs"],
            )
        pooled_validation = dense_expert(moe_input[validation_positions], selected)
        prior_expert = _prior_expert(prior, layer, expert)
        reports.append(
            {
                "layer": layer,
                "expert": expert,
                "routed_train_placements": train_count,
                "pooled_calibration_placements": 112,
                "validation_placements": len(validation_positions),
                "projection_reports": projection_reports,
                "pooled_validation_metrics": error_metrics(pooled_validation, expected_output),
                "pw0139_routed_calibration_validation_metrics": prior_expert[
                    "candidate_validation_metrics"
                ],
            }
        )
        del moe_input, expert_down, expected_output, source_weights, pooled_input, pooled_hidden
        del activations, expected, selected, pooled_validation
        gc.collect()
        mx.clear_cache()
        safety.release_checkpoint(
            f"layer_{layer}_expert_{expert}_complete_released",
            ["source expert", "pooled candidate", "captured activations", "expert outputs"],
        )
    gate = _gate(reports)
    decision = (
        "authorize_train_only_hessian_shrinkage_policy"
        if gate["passes"]
        else "reject_pooled_only_low_count_gptq"
    )
    safety.release_checkpoint(
        "checkpoint_and_authorities_released",
        ["checkpoint mappings", "PW-0116 corpus", "PW-0139 authorities"],
    )
    safety.checkpoint("final_service_health")
    result = {
        "schema_version": 1,
        "evidence_class": "pw0140_pooled_calibration_low_count_gptq_falsification",
        "revision": REVISION,
        "commit": commit,
        "source_hashes": {
            "checkpoint_verification_sha256": VERIFICATION_SHA256,
            "corpus_manifest_sha256": CORPUS_SHA256,
            "pw0139_report_sha256": PW0139_SHA256,
            "pw0139_analysis_sha256": PW0139_ANALYSIS_SHA256,
        },
        "samples": [list(row) for row in SAMPLES],
        "reports": reports,
        "continuation_gate": gate,
        "holdout_unsealed": False,
        "decision": decision,
        "safety_snapshots": safety.evidence(),
        "complete_wall_ms": (time.perf_counter() - started) * 1000.0,
        "accepted_tokens": 0,
        "A": 0,
        "performance_claim": None,
        "limitations": (
            "three validation-visible low-count experts only; pooled-only capacity test, not a "
            "train-only selector; fixed affine grids; dense unpacked oracle; holdout sealed; no "
            "bank, runtime, accumulated model, modalities, endpoint, accepted tokens, or TPS"
        ),
        "platform": platform.platform(),
    }
    atomic_write_new(output_path, canonical_json(result))
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--verification", required=True, type=Path)
    parser.add_argument("--corpus-manifest", required=True, type=Path)
    parser.add_argument("--pw0139", required=True, type=Path)
    parser.add_argument("--pw0139-analysis", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--commit", required=True)
    args = parser.parse_args()
    try:
        result = run(
            args.checkpoint, args.verification, args.corpus_manifest, args.pw0139,
            args.pw0139_analysis, args.output, args.commit,
        )
        print(json.dumps({"output": str(args.output), "decision": result["decision"]}))
        return 0
    except (OSError, RuntimeError, ValueError, KeyError, TypeError, json.JSONDecodeError, np.linalg.LinAlgError) as error:
        print(json.dumps({"error": str(error)}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
