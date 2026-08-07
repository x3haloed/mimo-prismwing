#!/usr/bin/env python3
"""Run PW-0133's train-only source-FP8 exception-store audit."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
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
    from tools.run_best_rank_real_expert_control import (
        dequant_weight,
        load_capture,
        sha256_file,
        source_linear,
    )
    from tools.run_int4_output_affine_repair_oracle import _route_rows
    from tools.run_real_activation_affine_int4_audit import (
        CORPUS_SHA256,
        GROUP_SIZE,
        LAYERS,
        REVISION,
        SOURCE_EXPERT_BYTES,
        VERIFICATION_SHA256,
        _candidate_expert,
        _quantize_expert,
        error_metrics,
        quantized_projection,
        validate_routes,
    )
except ModuleNotFoundError:
    from generate_real_layer1_expert_oracle import ShardedCheckpoint
    from host_safety import HostSafetyMonitor
    from openrouter_reference import atomic_write_new, canonical_json
    from run_best_rank_real_expert_control import (
        dequant_weight,
        load_capture,
        sha256_file,
        source_linear,
    )
    from run_int4_output_affine_repair_oracle import _route_rows
    from run_real_activation_affine_int4_audit import (
        CORPUS_SHA256,
        GROUP_SIZE,
        LAYERS,
        REVISION,
        SOURCE_EXPERT_BYTES,
        VERIFICATION_SHA256,
        _candidate_expert,
        _quantize_expert,
        error_metrics,
        quantized_projection,
        validate_routes,
    )


PW0129_SHA256 = "1deb9dd85f0b598f31bc2d8bc1d41bf52cfabcda43de63a2ae5b3fdfad400306"
PW0132_SHA256 = "0499a40645452eab646276e1619fb2e94b74439ef4263a71f036fae61fd8a9fe"
INT4_BYTES = 13_369_344
FRACTIONS = (0.01, 0.02, 0.04, 0.06)
GROUPS_PER_PROJECTION = 65_536
TOTAL_GROUPS_PER_EXPERT = 3 * GROUPS_PER_PROJECTION


def exception_count(total_groups: int, fraction: float) -> int:
    if total_groups <= 0 or not 0.0 < fraction <= 1.0:
        raise ValueError("PW-0133 exception count arguments are invalid")
    return int(math.ceil(total_groups * fraction))


def rank_exception_groups(
    source_weight: np.ndarray,
    int4_weight: np.ndarray,
    second_moment: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if (
        source_weight.shape != int4_weight.shape
        or source_weight.ndim != 2
        or source_weight.shape[1] % GROUP_SIZE
        or second_moment.shape != (source_weight.shape[1],)
        or not np.isfinite(source_weight).all()
        or not np.isfinite(int4_weight).all()
        or not np.isfinite(second_moment).all()
        or np.any(second_moment < 0)
    ):
        raise ValueError("PW-0133 sensitivity inputs are invalid")
    rows, columns = source_weight.shape
    error = source_weight.astype(np.float32) - int4_weight.astype(np.float32)
    scores = np.sum(
        error.reshape(rows, columns // GROUP_SIZE, GROUP_SIZE) ** 2
        * second_moment.astype(np.float32).reshape(1, columns // GROUP_SIZE, GROUP_SIZE),
        axis=2,
        dtype=np.float64,
    ).reshape(-1)
    ordinals = np.arange(scores.size, dtype=np.uint32)
    order = np.lexsort((ordinals, -scores)).astype(np.uint32, copy=False)
    if not np.array_equal(np.sort(order), ordinals):
        raise ValueError("PW-0133 sensitivity ranking is not a permutation")
    return order, scores


def source_scale_block_count(ordinals: np.ndarray, columns: int) -> int:
    if (
        ordinals.ndim != 1
        or ordinals.dtype != np.uint32
        or columns <= 0
        or columns % GROUP_SIZE
    ):
        raise ValueError("PW-0133 source-scale block inputs are invalid")
    groups_per_row = columns // GROUP_SIZE
    rows = ordinals.astype(np.uint64) // groups_per_row
    column_groups = ordinals.astype(np.uint64) % groups_per_row
    block_ids = (rows // 128) * groups_per_row + column_groups
    return int(np.unique(block_ids).size)


def physical_ledger(fraction: float) -> dict:
    groups = exception_count(GROUPS_PER_PROJECTION, fraction)
    selected = 3 * groups
    raw_fp8_bytes = selected * GROUP_SIZE
    ordinal_bytes = selected * 4
    # Fail-closed full-bank charge: assume no scale is shared between exceptions.
    source_scale_bytes = selected * 4
    combined = INT4_BYTES + raw_fp8_bytes + ordinal_bytes + source_scale_bytes
    return {
        "fraction": fraction,
        "selected_groups_per_projection": groups,
        "selected_groups_per_expert": selected,
        "int4_bytes_per_expert": INT4_BYTES,
        "exception_raw_fp8_bytes_per_expert": raw_fp8_bytes,
        "exception_ordinal_bytes_per_expert": ordinal_bytes,
        "conservative_source_scale_bytes_per_expert": source_scale_bytes,
        "combined_bytes_per_expert": combined,
        "combined_to_source_ratio": combined / SOURCE_EXPERT_BYTES,
        "correction_to_source_expert_mac_ratio": selected / TOTAL_GROUPS_PER_EXPERT,
    }


def train_second_moment(values: np.ndarray, positions: list[int] | None = None) -> np.ndarray:
    if values.ndim != 2 or values.shape[0] < 112 or not np.isfinite(values).all():
        raise ValueError("PW-0133 training activation matrix is invalid")
    selected = list(range(112)) if positions is None else positions
    if (
        not selected
        or any(not isinstance(position, int) or not 0 <= position < 112 for position in selected)
    ):
        raise ValueError("PW-0133 training activation positions are invalid")
    return np.mean(np.asarray(values[selected], dtype=np.float32) ** 2, axis=0)


def dense_correction(
    source_weight: np.ndarray,
    int4_weight: np.ndarray,
    selected: np.ndarray,
) -> np.ndarray:
    if selected.ndim != 1 or selected.dtype != np.uint32:
        raise ValueError("PW-0133 selected group identity is invalid")
    rows, columns = source_weight.shape
    groups_per_row = columns // GROUP_SIZE
    if selected.size and int(selected.max()) >= rows * groups_per_row:
        raise ValueError("PW-0133 selected group is out of bounds")
    correction = np.zeros(source_weight.shape, dtype=np.float16)
    source16 = source_weight.astype(np.float16, copy=False)
    int416 = int4_weight.astype(np.float16, copy=False)
    for ordinal in selected:
        row = int(ordinal) // groups_per_row
        group = int(ordinal) % groups_per_row
        start = group * GROUP_SIZE
        correction[row, start : start + GROUP_SIZE] = (
            source16[row, start : start + GROUP_SIZE].astype(np.float32)
            - int416[row, start : start + GROUP_SIZE].astype(np.float32)
        ).astype(np.float16)
    return correction


def selection_digest(
    projection: str,
    selected: np.ndarray,
    source_weight: np.ndarray,
    int4_weight: np.ndarray,
) -> str:
    digest = hashlib.sha256()
    digest.update(projection.encode("ascii"))
    digest.update(selected.astype("<u4", copy=False).tobytes())
    rows, columns = source_weight.shape
    groups_per_row = columns // GROUP_SIZE
    for ordinal in selected:
        row = int(ordinal) // groups_per_row
        start = (int(ordinal) % groups_per_row) * GROUP_SIZE
        digest.update(np.ascontiguousarray(source_weight[row, start : start + GROUP_SIZE].astype("<f2")).tobytes())
        digest.update(np.ascontiguousarray(int4_weight[row, start : start + GROUP_SIZE].astype("<f2")).tobytes())
    return digest.hexdigest()


def _source_hidden(
    gate_weight: np.ndarray,
    up_weight: np.ndarray,
    inputs: np.ndarray,
) -> np.ndarray:
    values = torch.from_numpy(np.asarray(inputs).copy()).to(torch.bfloat16)
    gate = source_linear(gate_weight, values)
    up = source_linear(up_weight, values)
    return (torch.nn.functional.silu(gate) * up).to(torch.bfloat16).float().numpy()


def _candidate_with_exceptions(
    inputs: np.ndarray,
    projections: dict,
    corrections: dict[str, np.ndarray],
) -> np.ndarray:
    values = mx.array(np.asarray(inputs, dtype=np.float16))
    correction_arrays = {name: mx.array(value) for name, value in corrections.items()}
    gate = quantized_projection(values, projections["gate"], 4) + values @ correction_arrays["gate"].T
    up = quantized_projection(values, projections["up"], 4) + values @ correction_arrays["up"].T
    hidden = mx.sigmoid(gate) * gate * up
    output = quantized_projection(hidden, projections["down"], 4) + hidden @ correction_arrays["down"].T
    mx.eval(output)
    result = np.asarray(output).astype(np.float32, copy=True)
    del values, correction_arrays, gate, up, hidden, output
    return result


def _prior_validation(prior: dict, layer: int) -> dict:
    matches = [
        row["validation"]
        for row in prior["reports"]
        if row["layer"] == layer and row["bits"] == 4
    ]
    if len(matches) != 1:
        raise ValueError("PW-0133 PW-0129 validation authority mismatch")
    return matches[0]


def _gate(layer_reports: list[dict]) -> dict:
    candidates = []
    for fraction in FRACTIONS:
        rows = [row["fractions"][str(fraction)] for row in layer_reports]
        squared_error = sum(row["routed_output_metrics"]["squared_error"] for row in rows)
        expected_norm = sum(row["routed_output_metrics"]["expected_squared_norm"] for row in rows)
        aggregate = float(np.sqrt(squared_error / max(expected_norm, 1e-30)))
        physical = physical_ledger(fraction)
        strict = (
            aggregate <= 0.01
            and all(row["routed_output_metrics"]["relative_l2"] <= 0.02 for row in rows)
            and all(row["routed_output_metrics"]["maximum_row_relative_l2"] <= 0.05 for row in rows)
            and physical["combined_to_source_ratio"] <= 0.60
            and physical["correction_to_source_expert_mac_ratio"] <= 0.10
        )
        near_miss = (
            not strict
            and aggregate <= 0.02
            and all(row["routed_output_metrics"]["relative_l2"] <= 0.04 for row in rows)
            and all(row["routed_output_metrics"]["maximum_row_relative_l2"] <= 0.08 for row in rows)
            and physical["combined_to_source_ratio"] <= 0.60
            and physical["correction_to_source_expert_mac_ratio"] <= 0.10
        )
        candidates.append(
            {
                "fraction": fraction,
                "aggregate_relative_l2": aggregate,
                "maximum_layer_relative_l2": max(row["routed_output_metrics"]["relative_l2"] for row in rows),
                "maximum_row_relative_l2": max(row["routed_output_metrics"]["maximum_row_relative_l2"] for row in rows),
                "physical": physical,
                "strict_pass": strict,
                "near_miss": near_miss,
            }
        )
    strict = next((row for row in candidates if row["strict_pass"]), None)
    near = next((row for row in candidates if row["near_miss"]), None)
    return {
        "candidates": candidates,
        "smallest_strict_fraction": strict["fraction"] if strict else None,
        "smallest_near_miss_fraction": near["fraction"] if near else None,
        "thresholds": {
            "strict": {"aggregate": 0.01, "layer": 0.02, "row": 0.05},
            "near_miss": {"aggregate": 0.02, "layer": 0.04, "row": 0.08},
            "maximum_source_byte_ratio": 0.60,
            "maximum_correction_mac_ratio": 0.10,
        },
    }


def run(
    checkpoint_root: Path,
    verification_path: Path,
    corpus_manifest_path: Path,
    pw0129_path: Path,
    pw0132_path: Path,
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
        (pw0132_path, PW0132_SHA256, "PW-0132 report"),
    ):
        if sha256_file(path) != expected:
            raise ValueError(f"PW-0133 {label} hash mismatch")
    started = time.perf_counter()
    safety = HostSafetyMonitor()
    corpus = json.loads(corpus_manifest_path.read_text())
    prior = json.loads(pw0129_path.read_text())
    rejected = json.loads(pw0132_path.read_text())
    if (
        corpus.get("revision") != REVISION
        or corpus.get("target_layers") != list(LAYERS)
        or prior.get("decision") != "reject_naive_affine_int4_on_real_validation"
        or rejected.get("decision") != "reject_pilot_train_only_rank32_int4_repair"
        or prior.get("holdout_unsealed")
        or rejected.get("holdout_unsealed")
    ):
        raise ValueError("PW-0133 authority identity mismatch")

    checkpoint = ShardedCheckpoint(checkpoint_root, verification_path)
    root = corpus_manifest_path.parent
    layer_reports = []
    for layer in LAYERS:
        authority = next(row for row in corpus["layers"] if row["layer"] == layer)
        validate_routes(authority)
        moe_input = load_capture(root, authority["captures"]["moe_input"])
        expert_down = load_capture(root, authority["captures"]["expert_down"])
        routed_expected = load_capture(root, authority["captures"]["routed_output"])
        pooled_input_moment = train_second_moment(moe_input)
        baseline_rows = {}
        fraction_rows = {fraction: {} for fraction in FRACTIONS}
        expert_reports = []
        offset = 0
        for schedule in authority["expert_schedule"]:
            expert = schedule["expert"]
            train_positions = [position for position in schedule["positions"] if position < 112]
            validation_local = [
                index for index, position in enumerate(schedule["positions"])
                if 112 <= position < 168
            ]
            if validation_local:
                validation_positions = [schedule["positions"][index] for index in validation_local]
                projections, packed = _quantize_expert(checkpoint, layer, expert, 4)
                expected_hashes = {
                    row["expert"]: row["packed_sha256"]
                    for row in _prior_validation(prior, layer)["expert_reports"]
                }
                if packed["packed_bytes"] != INT4_BYTES or packed["packed_sha256"] != expected_hashes.get(expert):
                    raise ValueError("PW-0133 recomputed INT4 artifact mismatch")
                source_weights = {}
                int4_weights = {}
                rankings = {}
                scores = {}
                for projection in ("gate", "up", "down"):
                    name = f"model.layers.{layer}.mlp.experts.{expert}.{projection}_proj.weight"
                    source_weights[projection] = dequant_weight(checkpoint, name)
                    int4_weights[projection] = np.asarray(
                        mx.dequantize(
                            *projections[projection], group_size=GROUP_SIZE, bits=4,
                            mode="affine", dtype=mx.float16,
                        )
                    ).astype(np.float16, copy=True)
                if train_positions:
                    input_moment = train_second_moment(moe_input, train_positions)
                    source_hidden = _source_hidden(
                        source_weights["gate"], source_weights["up"], moe_input[train_positions]
                    )
                    down_moment = np.mean(source_hidden.astype(np.float32) ** 2, axis=0)
                    fallback = "none"
                else:
                    input_moment = pooled_input_moment
                    down_moment = np.ones(2048, dtype=np.float32)
                    fallback = "layer_pooled_gate_up_and_weight_only_down"
                for projection in ("gate", "up", "down"):
                    moment = down_moment if projection == "down" else input_moment
                    rankings[projection], scores[projection] = rank_exception_groups(
                        source_weights[projection], int4_weights[projection], moment
                    )

                baseline, _ = _candidate_expert(
                    moe_input[validation_positions], projections, 4
                )
                baseline_rows[expert] = {
                    "positions": validation_positions,
                    "candidate": baseline,
                }
                expected_rows = np.asarray(
                    expert_down[[offset + index for index in validation_local]], dtype=np.float32
                ).copy()
                fraction_report = {}
                for fraction in FRACTIONS:
                    selections = {
                        projection: rankings[projection][
                            : exception_count(rankings[projection].size, fraction)
                        ]
                        for projection in ("gate", "up", "down")
                    }
                    corrections = {
                        projection: dense_correction(
                            source_weights[projection], int4_weights[projection], selections[projection]
                        )
                        for projection in ("gate", "up", "down")
                    }
                    candidate = _candidate_with_exceptions(
                        moe_input[validation_positions], projections, corrections
                    )
                    fraction_rows[fraction][expert] = {
                        "positions": validation_positions,
                        "candidate": candidate,
                    }
                    projection_reports = {}
                    for projection in ("gate", "up", "down"):
                        selected = selections[projection]
                        total_score = float(np.sum(scores[projection], dtype=np.float64))
                        selected_score = float(np.sum(scores[projection][selected], dtype=np.float64))
                        projection_reports[projection] = {
                            "selected_groups": int(selected.size),
                            "selected_score_fraction": selected_score / max(total_score, 1e-30),
                            "observed_unique_source_scale_blocks": source_scale_block_count(
                                selected, source_weights[projection].shape[1]
                            ),
                            "selection_sha256": selection_digest(
                                projection, selected, source_weights[projection], int4_weights[projection]
                            ),
                        }
                    fraction_report[str(fraction)] = {
                        "expert_output_metrics": error_metrics(candidate, expected_rows),
                        "projections": projection_reports,
                    }
                    del corrections, candidate, selections
                    gc.collect()
                    mx.clear_cache()
                expert_reports.append(
                    {
                        "expert": expert,
                        "train_placements": len(train_positions),
                        "validation_placements": len(validation_positions),
                        "selection_fallback": fallback,
                        "packed_sha256": packed["packed_sha256"],
                        "baseline_expert_output_metrics": error_metrics(baseline, expected_rows),
                        "fractions": fraction_report,
                    }
                )
                del projections, source_weights, int4_weights, rankings, scores, expected_rows
                if train_positions:
                    del source_hidden
                gc.collect()
                mx.clear_cache()
                if mx.get_active_memory() != 0:
                    raise ValueError("PW-0133 MLX expert buffers did not release")
                safety.release_checkpoint(
                    f"layer_{layer}_expert_{expert}_exception_oracle_released",
                    ["source expert", "INT4 expert", "dense correction oracle", "expert activations"],
                )
            offset += len(schedule["positions"])

        baseline_routed = _route_rows(baseline_rows, authority, "candidate", 112, 168)
        baseline_metrics = error_metrics(
            baseline_routed, np.asarray(routed_expected[112:168], dtype=np.float32)
        )
        if baseline_metrics != _prior_validation(prior, layer)["routed_output_metrics"]:
            raise ValueError(f"PW-0133 layer {layer} PW-0129 baseline reproduction mismatch")
        fractions = {}
        for fraction in FRACTIONS:
            routed = _route_rows(fraction_rows[fraction], authority, "candidate", 112, 168)
            fractions[str(fraction)] = {
                "routed_output_metrics": error_metrics(
                    routed, np.asarray(routed_expected[112:168], dtype=np.float32)
                ),
                "physical": physical_ledger(fraction),
            }
        layer_reports.append(
            {
                "layer": layer,
                "baseline_validation": baseline_metrics,
                "fractions": fractions,
                "experts": expert_reports,
                "fallback_experts": sum(row["selection_fallback"] != "none" for row in expert_reports),
                "fallback_validation_placements": sum(
                    row["validation_placements"]
                    for row in expert_reports if row["selection_fallback"] != "none"
                ),
            }
        )
        del moe_input, expert_down, routed_expected, pooled_input_moment
        del baseline_rows, fraction_rows, baseline_routed
        gc.collect()
        mx.clear_cache()
        safety.release_checkpoint(
            f"layer_{layer}_corpus_and_oracle_buffers_released",
            ["corpus captures", "routed candidates", "selection reports"],
        )

    gate = _gate(layer_reports)
    decision = (
        "authorize_source_fp8_exception_holdout_and_kernel_probe"
        if gate["smallest_strict_fraction"] is not None
        else (
            "authorize_awq_exception_composition"
            if gate["smallest_near_miss_fraction"] is not None
            else "reject_diagonal_sensitivity_source_fp8_exception_store"
        )
    )
    safety.release_checkpoint(
        "checkpoint_and_authorities_released",
        ["checkpoint mappings", "PW-0116 corpus", "PW-0129/PW-0132 reports"],
    )
    safety.checkpoint("final_service_health")
    report = {
        "schema_version": 1,
        "evidence_class": "pw0133_train_only_int4_source_fp8_exception_store",
        "revision": REVISION,
        "commit": commit,
        "source_hashes": {
            "checkpoint_verification_sha256": VERIFICATION_SHA256,
            "corpus_manifest_sha256": CORPUS_SHA256,
            "pw0129_report_sha256": PW0129_SHA256,
            "pw0132_report_sha256": PW0132_SHA256,
        },
        "fractions": list(FRACTIONS),
        "selection_partition": {"start": 0, "end_exclusive": 112},
        "validation_partition": {"start": 112, "end_exclusive": 168},
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
            "dense F16 correction matrices are fidelity-oracle machinery only; sparse source-FP8 "
            "decode/correction kernel unmeasured; one correlated English pilot; holdout sealed; "
            "no accumulated model, modalities, endpoint, accepted tokens, or TPS claim"
        ),
        "platform": platform.platform(),
        "mlx_version": "0.31.2",
    }
    atomic_write_new(output_path, canonical_json(report))
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--verification", required=True, type=Path)
    parser.add_argument("--corpus-manifest", required=True, type=Path)
    parser.add_argument("--pw0129", required=True, type=Path)
    parser.add_argument("--pw0132", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--commit", required=True)
    arguments = parser.parse_args()
    try:
        result = run(
            arguments.checkpoint, arguments.verification, arguments.corpus_manifest,
            arguments.pw0129, arguments.pw0132, arguments.output, arguments.commit,
        )
        print(json.dumps({"output": str(arguments.output), "decision": result["decision"]}))
        return 0
    except (OSError, RuntimeError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
        print(json.dumps({"error": str(error)}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
