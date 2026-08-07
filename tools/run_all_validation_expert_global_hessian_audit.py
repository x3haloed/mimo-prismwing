#!/usr/bin/env python3
"""Run PW-0139's all-validation-expert global-Hessian routed-layer audit."""

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
    from tools.run_global_hessian_gptq_rescue import (
        BLOCK_SIZE,
        DAMPING,
        global_hessian_gptq_fixed_grid,
        projected_workspace_bytes,
    )
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
        LAYERS,
        PACKED_RATIO_MAX,
        REVISION,
        VALIDATION_AGGREGATE_MAX,
        VALIDATION_LAYER_MAX,
        VALIDATION_ROW_MAX,
        VERIFICATION_SHA256,
        _source_replay,
        error_metrics,
        reconstruct_routed,
        validate_routes,
    )
except ModuleNotFoundError:
    from generate_real_layer1_expert_oracle import ShardedCheckpoint
    from host_safety import HostSafetyMonitor
    from openrouter_reference import atomic_write_new, canonical_json
    from run_best_rank_real_expert_control import dequant_weight, load_capture, sha256_file, source_linear
    from run_global_hessian_gptq_rescue import (
        BLOCK_SIZE,
        DAMPING,
        global_hessian_gptq_fixed_grid,
        projected_workspace_bytes,
    )
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
        LAYERS,
        PACKED_RATIO_MAX,
        REVISION,
        VALIDATION_AGGREGATE_MAX,
        VALIDATION_LAYER_MAX,
        VALIDATION_ROW_MAX,
        VERIFICATION_SHA256,
        _source_replay,
        error_metrics,
        reconstruct_routed,
        validate_routes,
    )


PW0129_SHA256 = "1deb9dd85f0b598f31bc2d8bc1d41bf52cfabcda43de63a2ae5b3fdfad400306"
PW0138_SHA256 = "37fa27ce90d0dc46b4b9308ed708c99405eb7ad3d924b859489716b9771bde49"
PW0138_ANALYSIS_SHA256 = "7ed32546bfb042d5b863c23d812eeada89cafb7d65b9c1d86c30c7483022e14b"
CONTROL_EXPERTS = {4: 96, 24: 22, 46: 28}


def _prior_control(prior: dict, layer: int, expert: int) -> dict:
    matches = [
        row for row in prior.get("reports", [])
        if row.get("layer") == layer and row.get("expert") == expert
    ]
    if len(matches) != 1:
        raise ValueError("PW-0139 PW-0138 control authority mismatch")
    return matches[0]


def _validation_gate(layer_reports: list[dict]) -> dict:
    if [row["layer"] for row in layer_reports] != list(LAYERS):
        raise ValueError("PW-0139 validation gate lacks frozen layers")
    squared_error = sum(row["routed_output_metrics"]["squared_error"] for row in layer_reports)
    expected_norm = sum(
        row["routed_output_metrics"]["expected_squared_norm"] for row in layer_reports
    )
    aggregate = float(np.sqrt(squared_error / max(expected_norm, 1e-30)))
    physical = physical_ledger()
    projection_improvement = all(
        projection["candidate_calibration_metrics"]["relative_l2"]
        < projection["baseline_calibration_metrics"]["relative_l2"]
        for layer in layer_reports
        for expert in layer["expert_reports"]
        for projection in expert["projection_reports"].values()
    )
    routes_complete = all(
        row["validation_placements"] == 448
        and sum(expert["validation_placements"] for expert in row["expert_reports"]) == 448
        for row in layer_reports
    )
    return {
        "aggregate_relative_l2": aggregate,
        "aggregate_maximum": VALIDATION_AGGREGATE_MAX,
        "maximum_layer_relative_l2": max(
            row["routed_output_metrics"]["relative_l2"] for row in layer_reports
        ),
        "layer_maximum": VALIDATION_LAYER_MAX,
        "maximum_row_relative_l2": max(
            row["routed_output_metrics"]["maximum_row_relative_l2"] for row in layer_reports
        ),
        "row_maximum": VALIDATION_ROW_MAX,
        "projection_calibration_improvement": projection_improvement,
        "routes_complete": routes_complete,
        "pw0138_exact_reproduction": all(row["pw0138_control_reproduced"] for row in layer_reports),
        "physical": physical,
        "packed_ratio_maximum": PACKED_RATIO_MAX,
        "passes": (
            aggregate <= VALIDATION_AGGREGATE_MAX
            and all(row["routed_output_metrics"]["relative_l2"] <= VALIDATION_LAYER_MAX for row in layer_reports)
            and all(row["routed_output_metrics"]["maximum_row_relative_l2"] <= VALIDATION_ROW_MAX for row in layer_reports)
            and projection_improvement
            and routes_complete
            and all(row["pw0138_control_reproduced"] for row in layer_reports)
            and physical["packed_to_source_ratio"] <= PACKED_RATIO_MAX
            and physical["additional_runtime_macs"] == 0
        ),
    }


def _evaluate_layer(
    checkpoint: ShardedCheckpoint,
    root: Path,
    authority: dict,
    layer: int,
    pw0138: dict,
    safety: HostSafetyMonitor,
) -> tuple[dict, dict]:
    validate_routes(authority)
    moe_input = load_capture(root, authority["captures"]["moe_input"])
    expert_down = load_capture(root, authority["captures"]["expert_down"])
    routed_output = load_capture(root, authority["captures"]["routed_output"])
    reconstructed = reconstruct_routed(expert_down, authority, 0, 168)
    if not np.array_equal(reconstructed, np.asarray(routed_output[:168], dtype=np.float32)):
        raise ValueError(f"PW-0139 layer {layer} source prefix reconstruction mismatch")
    source_replay = _source_replay(checkpoint, layer, authority, moe_input, expert_down, 168)
    safety.checkpoint(f"layer_{layer}_source_authority_passed")

    candidate_routed = np.zeros((56, 4096), dtype=np.float32)
    expert_reports = []
    offset = 0
    selected = authority["selected_experts_by_position"]
    route_weights = authority["route_weights_by_position"]
    control_reproduced = False
    for schedule in authority["expert_schedule"]:
        expert = schedule["expert"]
        positions = schedule["positions"]
        validation_local = [
            index for index, position in enumerate(positions) if 112 <= position < 168
        ]
        if not validation_local:
            offset += len(positions)
            continue
        train_local = [index for index, position in enumerate(positions) if position < 112]
        fallback = not train_local
        calibration_positions = (
            [positions[index] for index in train_local] if train_local else list(range(112))
        )
        validation_positions = [positions[index] for index in validation_local]
        prefix = f"model.layers.{layer}.mlp.experts.{expert}"
        source_weights = {
            projection: dequant_weight(checkpoint, f"{prefix}.{projection}_proj.weight")
            for projection in ("gate", "up", "down")
        }
        calibration_hidden = source_hidden(source_weights, moe_input[calibration_positions])
        calibration_activations = {
            "gate": moe_input[calibration_positions],
            "up": moe_input[calibration_positions],
            "down": calibration_hidden,
        }
        if fallback:
            calibration_expected_down = source_linear(
                source_weights["down"],
                torch.from_numpy(np.asarray(calibration_hidden).copy()).to(torch.bfloat16),
            ).float().numpy()
        else:
            calibration_expected_down = np.asarray(
                expert_down[[offset + index for index in train_local]], dtype=np.float32
            ).copy()
        calibration_expected = {
            "gate": source_linear(
                source_weights["gate"],
                torch.from_numpy(np.asarray(moe_input[calibration_positions]).copy()).to(torch.bfloat16),
            ).float().numpy(),
            "up": source_linear(
                source_weights["up"],
                torch.from_numpy(np.asarray(moe_input[calibration_positions]).copy()).to(torch.bfloat16),
            ).float().numpy(),
            "down": calibration_expected_down,
        }
        candidate_weights = {}
        projection_reports = {}
        for projection in ("gate", "up", "down"):
            weight = source_weights[projection]
            scales, biases, baseline, baseline_codes = affine_grid(weight)
            validate_grid_membership(baseline_codes, baseline, scales, biases)
            projected = projected_workspace_bytes(weight)
            before = safety.checkpoint(f"layer_{layer}_expert_{expert}_{projection}_preflight")
            if before.process_physical_footprint_bytes + projected > safety.policy.maximum_process_physical_footprint_bytes:
                raise RuntimeError(f"PW-0139 layer {layer} expert {expert} exceeds Gate 8 headroom")
            candidate, codes, diagnostics = global_hessian_gptq_fixed_grid(
                weight, calibration_activations[projection], scales, biases
            )
            candidate_projection = dense_projection(calibration_activations[projection], candidate)
            baseline_projection = dense_projection(calibration_activations[projection], baseline)
            candidate_weights[projection] = candidate
            projection_reports[projection] = {
                **diagnostics,
                "projected_workspace_bytes": projected,
                "candidate_calibration_metrics": error_metrics(
                    candidate_projection, calibration_expected[projection]
                ),
                "baseline_calibration_metrics": error_metrics(
                    baseline_projection, calibration_expected[projection]
                ),
            }
            del scales, biases, baseline, baseline_codes, codes
            del candidate_projection, baseline_projection
            gc.collect()
            safety.release_checkpoint(
                f"layer_{layer}_expert_{expert}_{projection}_workspace_released",
                ["full Hessian", "inverse Cholesky", "permuted working weight", "projection outputs"],
            )
        candidate_validation = dense_expert(moe_input[validation_positions], candidate_weights)
        expected_validation = np.asarray(
            expert_down[[offset + index for index in validation_local]], dtype=np.float32
        ).copy()
        candidate_calibration = dense_expert(
            moe_input[calibration_positions], candidate_weights
        )
        for local, position in enumerate(validation_positions):
            slot = selected[position].index(expert)
            candidate_routed[position - 112] += candidate_validation[local] * np.float32(
                route_weights[position][slot]
            )
        expert_report = {
            "expert": expert,
            "calibration_placements": len(calibration_positions),
            "validation_placements": len(validation_positions),
            "train_absent_layer_fallback": fallback,
            "projection_reports": projection_reports,
            "candidate_calibration_metrics": error_metrics(
                candidate_calibration, calibration_expected_down
            ),
            "candidate_validation_metrics": error_metrics(
                candidate_validation, expected_validation
            ),
        }
        if expert == CONTROL_EXPERTS[layer]:
            control = _prior_control(pw0138, layer, expert)
            if (
                expert_report["candidate_calibration_metrics"] != control["global_gptq_train"]
                or expert_report["candidate_validation_metrics"] != control["global_gptq_validation"]
                or {
                    name: (row["grid_sha256"], row["activation_order_sha256"])
                    for name, row in projection_reports.items()
                }
                != {
                    name: (row["grid_sha256"], row["activation_order_sha256"])
                    for name, row in control["projection_reports"].items()
                }
            ):
                raise ValueError(f"PW-0139 layer {layer} does not reproduce PW-0138")
            control_reproduced = True
        expert_reports.append(expert_report)
        del source_weights, calibration_hidden, calibration_activations, calibration_expected
        del calibration_expected_down, candidate_weights, candidate_validation
        del expected_validation, candidate_calibration
        gc.collect()
        mx.clear_cache()
        safety.release_checkpoint(
            f"layer_{layer}_expert_{expert}_complete_released",
            ["source expert", "candidate expert", "expert activations", "expert outputs"],
        )
        offset += len(positions)
    candidate_routed = torch.from_numpy(candidate_routed).to(torch.bfloat16).float().numpy()
    routed_metrics = error_metrics(candidate_routed, np.asarray(routed_output[112:168], dtype=np.float32))
    result = {
        "layer": layer,
        "unique_validation_experts": len(expert_reports),
        "validation_placements": sum(row["validation_placements"] for row in expert_reports),
        "train_absent_fallback_experts": [
            row["expert"] for row in expert_reports if row["train_absent_layer_fallback"]
        ],
        "pw0138_control_reproduced": control_reproduced,
        "routed_output_metrics": routed_metrics,
        "expert_reports": expert_reports,
    }
    del moe_input, expert_down, routed_output, reconstructed, candidate_routed
    gc.collect()
    safety.release_checkpoint(
        f"layer_{layer}_corpus_released",
        ["moe input", "expert outputs", "routed output", "candidate routed output"],
    )
    return result, {"layer": layer, **source_replay}


def run(
    checkpoint_root: Path,
    verification_path: Path,
    corpus_manifest_path: Path,
    pw0129_path: Path,
    pw0138_path: Path,
    pw0138_analysis_path: Path,
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
        (pw0129_path, PW0129_SHA256, "PW-0129 report"),
        (pw0138_path, PW0138_SHA256, "PW-0138 report"),
        (pw0138_analysis_path, PW0138_ANALYSIS_SHA256, "PW-0138 analysis"),
    ):
        if sha256_file(path) != expected:
            raise ValueError(f"PW-0139 {label} hash mismatch")
    started = time.perf_counter()
    safety = HostSafetyMonitor()
    corpus = json.loads(corpus_manifest_path.read_text())
    pw0129 = json.loads(pw0129_path.read_text())
    pw0138 = json.loads(pw0138_path.read_text())
    pw0138_analysis = json.loads(pw0138_analysis_path.read_text())
    if (
        corpus.get("revision") != REVISION
        or corpus.get("target_layers") != list(LAYERS)
        or pw0129.get("decision") != "reject_naive_affine_int4_on_real_validation"
        or pw0138.get("decision") != "authorize_all_validation_expert_global_hessian_audit"
        or pw0138_analysis.get("source_report_sha256") != PW0138_SHA256
        or pw0129.get("holdout_unsealed")
        or pw0138.get("holdout_unsealed")
        or pw0138_analysis.get("holdout_unsealed")
    ):
        raise ValueError("PW-0139 authority identity mismatch")
    checkpoint = ShardedCheckpoint(checkpoint_root, verification_path)
    root = corpus_manifest_path.parent
    layer_reports = []
    source_replays = []
    for layer in LAYERS:
        authority = next(row for row in corpus["layers"] if row["layer"] == layer)
        report, replay = _evaluate_layer(checkpoint, root, authority, layer, pw0138, safety)
        layer_reports.append(report)
        source_replays.append(replay)
    gate = _validation_gate(layer_reports)
    decision = (
        "authorize_global_hessian_holdout_audit"
        if gate["passes"]
        else "reject_global_hessian_fixed_grid_on_full_validation"
    )
    safety.release_checkpoint(
        "checkpoint_and_authorities_released",
        ["checkpoint mappings", "PW-0116 corpus", "PW-0129/PW-0138 authorities"],
    )
    safety.checkpoint("final_service_health")
    result = {
        "schema_version": 1,
        "evidence_class": "pw0139_all_validation_expert_global_hessian_audit",
        "revision": REVISION,
        "commit": commit,
        "source_hashes": {
            "checkpoint_verification_sha256": VERIFICATION_SHA256,
            "corpus_manifest_sha256": CORPUS_SHA256,
            "pw0129_report_sha256": PW0129_SHA256,
            "pw0138_report_sha256": PW0138_SHA256,
            "pw0138_analysis_sha256": PW0138_ANALYSIS_SHA256,
        },
        "layers": list(LAYERS),
        "validation_range": [112, 168],
        "mechanism": {
            "damping": DAMPING,
            "order": "descending_full_hessian_diagonal",
            "block_size": BLOCK_SIZE,
            "static_original_group_grid": True,
            "cross_block_error_propagation": True,
            "train_absent_fallback": "all_112_layer_train_positions",
        },
        "source_replays": source_replays,
        "layer_reports": layer_reports,
        "validation_gate": gate,
        "holdout_unsealed": False,
        "decision": decision,
        "safety_snapshots": safety.evidence(),
        "complete_wall_ms": (time.perf_counter() - started) * 1000.0,
        "accepted_tokens": 0,
        "A": 0,
        "performance_claim": None,
        "limitations": (
            "three layer-local source-route validation slices from one English pilot; two "
            "train-absent experts use declared layer fallback; dense unpacked execution oracle; "
            "holdout sealed; no runtime artifact, kernel, bank, accumulated model, modalities, "
            "endpoint, accepted tokens, or TPS"
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
    parser.add_argument("--pw0129", required=True, type=Path)
    parser.add_argument("--pw0138", required=True, type=Path)
    parser.add_argument("--pw0138-analysis", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--commit", required=True)
    args = parser.parse_args()
    try:
        result = run(
            args.checkpoint, args.verification, args.corpus_manifest, args.pw0129,
            args.pw0138, args.pw0138_analysis, args.output, args.commit,
        )
        print(json.dumps({"output": str(args.output), "decision": result["decision"]}))
        return 0
    except (OSError, RuntimeError, ValueError, KeyError, TypeError, json.JSONDecodeError, np.linalg.LinAlgError) as error:
        print(json.dumps({"error": str(error)}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
