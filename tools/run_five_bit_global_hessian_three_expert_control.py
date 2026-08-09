#!/usr/bin/env python3
"""Run PW-0147's five-bit global-Hessian three-expert control."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import gc
import hashlib
import json
from pathlib import Path
import platform
import time
from typing import Callable

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
        source_hidden,
    )
    from tools.run_real_activation_affine_int4_audit import (
        CORPUS_SHA256,
        GROUP_SIZE,
        REVISION,
        SOURCE_EXPERT_BYTES,
        VERIFICATION_SHA256,
        error_metrics,
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
    from run_group_local_gptq_three_expert_control import affine_grid, dense_expert, dense_projection, source_hidden
    from run_real_activation_affine_int4_audit import (
        CORPUS_SHA256,
        GROUP_SIZE,
        REVISION,
        SOURCE_EXPERT_BYTES,
        VERIFICATION_SHA256,
        error_metrics,
        validate_routes,
    )


PW0138_SHA256 = "37fa27ce90d0dc46b4b9308ed708c99405eb7ad3d924b859489716b9771bde49"
PW0146_SHA256 = "7bb795455927295c673bfe65d06ae6311dbdd97b9d3517caa357307d189bdcf3"
SAMPLES = ((4, 96, 109, 56), (24, 22, 26, 56), (46, 28, 100, 56))
BITS = 5
MAXIMUM_CODE = (1 << BITS) - 1
PACKED_CODE_BYTES = 15_728_640
METADATA_BYTES = 786_432
PACKED_BYTES = PACKED_CODE_BYTES + METADATA_BYTES
PACKED_RATIO = PACKED_BYTES / SOURCE_EXPERT_BYTES


@dataclass(frozen=True)
class NBitControlConfig:
    experiment: str
    bits: int
    packed_code_bytes: int
    metadata_bytes: int
    metadata_label: str
    maximum_packed_ratio: float
    candidate_label: str
    evidence_class: str
    pass_decision: str
    reject_decision: str
    prerequisite_sha256: str
    prerequisite_label: str
    prerequisite_decision: str
    prerequisite_source_key: str
    prior_candidate_label: str | None = None

    @property
    def maximum_code(self) -> int:
        return (1 << self.bits) - 1

    @property
    def packed_bytes(self) -> int:
        return self.packed_code_bytes + self.metadata_bytes


FIVE_BIT_CONFIG = NBitControlConfig(
    experiment="PW-0147",
    bits=5,
    packed_code_bytes=PACKED_CODE_BYTES,
    metadata_bytes=METADATA_BYTES,
    metadata_label="f16_affine_metadata_bytes_per_expert",
    maximum_packed_ratio=0.70,
    candidate_label="five_bit",
    evidence_class="pw0147_five_bit_global_hessian_three_expert_control",
    pass_decision="authorize_all_validation_expert_five_bit_audit",
    reject_decision="reject_five_bit_global_hessian_three_expert_control",
    prerequisite_sha256=PW0146_SHA256,
    prerequisite_label="PW-0146 report",
    prerequisite_decision="reject_threshold_crossing_fixed_grid_code_qat",
    prerequisite_source_key="pw0146_report_sha256",
)


def affine_nbit_grid(
    weight: np.ndarray, bits: int = BITS
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if (
        weight.ndim != 2
        or weight.shape[1] % GROUP_SIZE
        or bits <= 0
        or bits > 8
        or not np.isfinite(weight).all()
    ):
        raise ValueError("PW-0147 affine grid input is invalid")
    rows, columns = weight.shape
    groups = columns // GROUP_SIZE
    maximum = (1 << bits) - 1
    grouped = np.asarray(weight, dtype=np.float32).reshape(rows, groups, GROUP_SIZE)
    minimum = grouped.min(axis=2)
    maximum_value = grouped.max(axis=2)
    scales = ((maximum_value - minimum) / maximum).astype(np.float16)
    biases = minimum.astype(np.float16)
    scale32 = scales.astype(np.float32)
    bias32 = biases.astype(np.float32)
    safe = np.where(scale32 == 0, np.float32(1.0), scale32)
    codes = np.clip(
        np.rint((grouped - bias32[:, :, None]) / safe[:, :, None]), 0, maximum
    )
    codes = np.where(scale32[:, :, None] == 0, 0, codes).astype(np.uint8)
    values = (codes.astype(np.float32) * scale32[:, :, None] + bias32[:, :, None])
    quantized = values.reshape(rows, columns).astype(np.float16)
    validate_nbit_grid(codes.reshape(rows, columns), quantized, scales, biases, bits)
    return scales, biases, quantized, codes.reshape(rows, columns)


def reconstruct_nbit_grid(
    codes: np.ndarray, scales: np.ndarray, biases: np.ndarray, bits: int = BITS
) -> np.ndarray:
    maximum = (1 << bits) - 1
    if (
        codes.ndim != 2
        or codes.dtype != np.uint8
        or np.any(codes > maximum)
        or scales.shape != biases.shape
        or scales.shape[0] != codes.shape[0]
        or codes.shape[1] != scales.shape[1] * GROUP_SIZE
    ):
        raise ValueError("PW-0147 grid reconstruction input is invalid")
    rows, columns = codes.shape
    groups = scales.shape[1]
    values = codes.reshape(rows, groups, GROUP_SIZE).astype(np.float32)
    values = values * scales.astype(np.float32)[:, :, None] + biases.astype(np.float32)[:, :, None]
    return values.reshape(rows, columns).astype(np.float16)


def validate_nbit_grid(
    codes: np.ndarray,
    values: np.ndarray,
    scales: np.ndarray,
    biases: np.ndarray,
    bits: int = BITS,
) -> None:
    if not np.array_equal(values, reconstruct_nbit_grid(codes, scales, biases, bits)):
        raise ValueError("PW-0147 grid membership failed")


def global_hessian_nbit_fixed_grid(
    weight: np.ndarray,
    activations: np.ndarray,
    scales: np.ndarray,
    biases: np.ndarray,
    *,
    bits: int = BITS,
    damping: float = DAMPING,
    block_size: int = BLOCK_SIZE,
    column_quantizer: Callable | None = None,
    result_validator: Callable | None = None,
    grid_payloads: tuple[np.ndarray, ...] | None = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    if (
        weight.ndim != 2
        or activations.ndim != 2
        or activations.shape[1] != weight.shape[1]
        or activations.shape[0] == 0
        or bits <= 0
        or bits > 8
        or damping <= 0
        or block_size <= 0
    ):
        raise ValueError("PW-0147 global-Hessian input is invalid")
    if column_quantizer is None and (
        scales.shape != biases.shape
        or scales.shape[0] != weight.shape[0]
        or weight.shape[1] != scales.shape[1] * GROUP_SIZE
    ):
        raise ValueError("PW-0147 global-Hessian affine metadata is invalid")
    rows, columns = weight.shape
    maximum = (1 << bits) - 1
    x = activations.astype(np.float64)
    hessian = (x.T @ x) / x.shape[0]
    diagonal = np.diag(hessian).copy()
    dead = diagonal == 0
    if np.any(dead):
        dead_indices = np.flatnonzero(dead)
        hessian[dead_indices, dead_indices] = 1.0
    damp = max(damping * float(np.mean(np.diag(hessian))), 1e-8)
    hessian.flat[:: columns + 1] += damp
    permutation = np.argsort(-diagonal, kind="stable")
    inverse = np.linalg.inv(hessian[np.ix_(permutation, permutation)])
    inverse_cholesky = np.linalg.cholesky(inverse).T
    del x, hessian, inverse
    working = weight.astype(np.float64)[:, permutation].copy()
    quantized = np.empty(weight.shape, dtype=np.float16)
    codes = np.empty(weight.shape, dtype=np.uint8)
    cross_block_update_squared = 0.0
    block_count = 0
    for start in range(0, columns, block_size):
        end = min(start + block_size, columns)
        count = end - start
        block_count += 1
        block_weight = working[:, start:end].copy()
        block_error = np.empty((rows, count), dtype=np.float64)
        block_factor = inverse_cholesky[start:end, start:end]
        for local_column in range(count):
            ordered_column = start + local_column
            original_column = int(permutation[ordered_column])
            if column_quantizer is None:
                group = original_column // GROUP_SIZE
                scale = scales[:, group].astype(np.float64)
                bias = biases[:, group].astype(np.float64)
                safe = np.where(scale == 0, 1.0, scale)
                code = np.clip(
                    np.rint((block_weight[:, local_column] - bias) / safe), 0, maximum
                )
                code = np.where(scale == 0, 0, code)
                value = code * scale + bias
            else:
                code, value = column_quantizer(
                    block_weight[:, local_column], original_column
                )
                if code.shape != (rows,) or value.shape != (rows,):
                    raise ValueError("global-Hessian column quantizer shape mismatch")
            codes[:, original_column] = code.astype(np.uint8)
            quantized[:, original_column] = value.astype(np.float16)
            error = (block_weight[:, local_column] - value) / block_factor[local_column, local_column]
            block_weight[:, local_column:] -= (
                error[:, None] * block_factor[local_column, local_column:][None, :]
            )
            block_error[:, local_column] = error
        if end < columns:
            update = block_error @ inverse_cholesky[start:end, end:]
            cross_block_update_squared += float(np.sum(update * update, dtype=np.float64))
            working[:, end:] -= update
    if result_validator is None:
        validate_nbit_grid(codes, quantized, scales, biases, bits)
    else:
        result_validator(codes, quantized)
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(codes).tobytes())
    for payload in grid_payloads or (scales, biases):
        digest.update(np.ascontiguousarray(payload).tobytes())
    return quantized, codes, {
        "bits": bits,
        "maximum_code": maximum,
        "damping": damping,
        "block_size": block_size,
        "block_count": block_count,
        "dead_activation_columns": int(np.count_nonzero(dead)),
        "cross_block_update_l2": cross_block_update_squared**0.5,
        "activation_order_sha256": hashlib.sha256(
            np.ascontiguousarray(permutation.astype("<u4")).tobytes()
        ).hexdigest(),
        "grid_sha256": digest.hexdigest(),
    }


def physical_ledger(config: NBitControlConfig = FIVE_BIT_CONFIG) -> dict:
    packed_bytes = config.packed_bytes
    return {
        "bits": config.bits,
        "packed_code_bytes_per_expert": config.packed_code_bytes,
        config.metadata_label: config.metadata_bytes,
        "packed_bytes_per_expert": packed_bytes,
        "packed_to_source_ratio": packed_bytes / SOURCE_EXPERT_BYTES,
        "full_routed_bank_bytes": 47 * 256 * packed_bytes,
        "additional_runtime_macs": 0,
    }


def _prior_report(prior: dict, layer: int, expert: int) -> dict:
    matches = [
        row for row in prior.get("reports", [])
        if row.get("layer") == layer and row.get("expert") == expert
    ]
    if len(matches) != 1:
        raise ValueError("n-bit prior control authority mismatch")
    return matches[0]


def _gate(
    reports: list[dict], config: NBitControlConfig = FIVE_BIT_CONFIG
) -> dict:
    rows = []
    for report in reports:
        candidate = report[f"{config.candidate_label}_validation_metrics"]
        if config.prior_candidate_label is None:
            prior = report["pw0138_four_bit_validation_metrics"]
            prior_condition = "improves_four_bit_validation"
            prior_output = "four_bit_validation_relative_l2"
        else:
            prior = report["prior_candidate_validation_metrics"]
            prior_condition = f"improves_{config.prior_candidate_label}_validation"
            prior_output = f"{config.prior_candidate_label}_validation_relative_l2"
        conditions = {
            "validation_relative_l2": candidate["relative_l2"] <= 0.02,
            "maximum_validation_row_relative_l2": candidate["maximum_row_relative_l2"] <= 0.05,
            "train_improves_round_to_nearest": (
                report[f"{config.candidate_label}_train_metrics"]["relative_l2"]
                < report[f"{config.candidate_label}_rtn_train_metrics"]["relative_l2"]
            ),
            prior_condition: candidate["relative_l2"] < prior["relative_l2"],
            "four_bit_control_reproduced": report["four_bit_control_reproduced"],
            "code_domain": report["code_domain_valid"],
        }
        row = {
            "layer": report["layer"],
            "expert": report["expert"],
            "passes": all(conditions.values()),
            "conditions": conditions,
            f"{config.candidate_label}_validation_relative_l2": candidate["relative_l2"],
            f"{config.candidate_label}_maximum_row_relative_l2": candidate["maximum_row_relative_l2"],
            prior_output: prior["relative_l2"],
        }
        rows.append(row)
    physical = physical_ledger(config)
    physical_passes = (
        physical["packed_bytes_per_expert"] == config.packed_bytes
        and physical["packed_to_source_ratio"] <= config.maximum_packed_ratio
        and physical["additional_runtime_macs"] == 0
    )
    return {
        "passes": all(row["passes"] for row in rows) and physical_passes,
        "experts": rows,
        "physical": physical,
        "physical_passes": physical_passes,
    }


def run(
    checkpoint_root: Path,
    verification_path: Path,
    corpus_manifest_path: Path,
    pw0138_path: Path,
    prerequisite_path: Path,
    output_path: Path,
    commit: str,
    config: NBitControlConfig = FIVE_BIT_CONFIG,
    grid_builder: Callable = affine_nbit_grid,
    assignment_builder: Callable = global_hessian_nbit_fixed_grid,
) -> dict:
    if output_path.exists():
        raise ValueError(f"refusing to overwrite {output_path}")
    if len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit):
        raise ValueError("implementation commit must be lowercase 40-hex")
    for path, expected, label in (
        (verification_path, VERIFICATION_SHA256, "checkpoint verification"),
        (corpus_manifest_path, CORPUS_SHA256, "PW-0116 corpus"),
        (pw0138_path, PW0138_SHA256, "PW-0138 report"),
        (prerequisite_path, config.prerequisite_sha256, config.prerequisite_label),
    ):
        if sha256_file(path) != expected:
            raise ValueError(f"{config.experiment} {label} hash mismatch")
    started = time.perf_counter()
    safety = HostSafetyMonitor()
    corpus = json.loads(corpus_manifest_path.read_text())
    prior = json.loads(pw0138_path.read_text())
    prerequisite = json.loads(prerequisite_path.read_text())
    if (
        corpus.get("revision") != REVISION
        or prior.get("decision") != "authorize_all_validation_expert_global_hessian_audit"
        or prerequisite.get("decision") != config.prerequisite_decision
        or prior.get("holdout_unsealed")
        or prerequisite.get("holdout_unsealed")
    ):
        raise ValueError(f"{config.experiment} authority identity mismatch")
    checkpoint = ShardedCheckpoint(checkpoint_root, verification_path)
    root = corpus_manifest_path.parent
    reports = []
    for layer, expert, expected_train, expected_validation in SAMPLES:
        authority = next(row for row in corpus["layers"] if row["layer"] == layer)
        validate_routes(authority)
        schedule = next(row for row in authority["expert_schedule"] if row["expert"] == expert)
        train_local = [index for index, position in enumerate(schedule["positions"]) if position < 112]
        validation_local = [
            index for index, position in enumerate(schedule["positions"])
            if 112 <= position < 168
        ]
        if len(train_local) != expected_train or len(validation_local) != expected_validation:
            raise ValueError(f"{config.experiment} frozen sample coverage mismatch")
        offset = sum(
            len(row["positions"])
            for row in authority["expert_schedule"][: authority["expert_schedule"].index(schedule)]
        )
        moe_input = load_capture(root, authority["captures"]["moe_input"])
        expert_down = load_capture(root, authority["captures"]["expert_down"])
        train_positions = [schedule["positions"][index] for index in train_local]
        validation_positions = [schedule["positions"][index] for index in validation_local]
        train_expected = np.asarray(
            expert_down[[offset + index for index in train_local]], dtype=np.float32
        ).copy()
        validation_expected = np.asarray(
            expert_down[[offset + index for index in validation_local]], dtype=np.float32
        ).copy()
        prefix = f"model.layers.{layer}.mlp.experts.{expert}"
        source_weights = {
            projection: dequant_weight(checkpoint, f"{prefix}.{projection}_proj.weight")
            for projection in ("gate", "up", "down")
        }
        hidden = source_hidden(source_weights, moe_input[train_positions])
        activations = {"gate": moe_input[train_positions], "up": moe_input[train_positions], "down": hidden}
        expected_projection = {
            "gate": source_linear(
                source_weights["gate"],
                torch.from_numpy(np.asarray(moe_input[train_positions]).copy()).to(torch.bfloat16),
            ).float().numpy(),
            "up": source_linear(
                source_weights["up"],
                torch.from_numpy(np.asarray(moe_input[train_positions]).copy()).to(torch.bfloat16),
            ).float().numpy(),
            "down": train_expected,
        }
        five_weights = {}
        five_rtn_weights = {}
        four_weights = {}
        projection_reports = {}
        for projection in ("gate", "up", "down"):
            weight = source_weights[projection]
            projected = projected_workspace_bytes(weight)
            before = safety.checkpoint(f"layer_{layer}_expert_{expert}_{projection}_preflight")
            if before.process_physical_footprint_bytes + projected > safety.policy.maximum_process_physical_footprint_bytes:
                raise RuntimeError(f"{config.experiment} layer {layer} expert {expert} exceeds Gate 8 headroom")
            scales5, biases5, rtn5, rtn_codes5 = grid_builder(weight, config.bits)
            candidate5, codes5, diagnostics5 = assignment_builder(
                weight, activations[projection], scales5, biases5, bits=config.bits
            )
            scales4, biases4, _, _ = affine_grid(weight)
            candidate4, _, diagnostics4 = global_hessian_gptq_fixed_grid(
                weight, activations[projection], scales4, biases4
            )
            five_weights[projection] = candidate5
            five_rtn_weights[projection] = rtn5
            four_weights[projection] = candidate4
            projection_reports[projection] = {
                config.candidate_label: diagnostics5,
                f"{config.candidate_label}_candidate_train_metrics": error_metrics(
                    dense_projection(activations[projection], candidate5),
                    expected_projection[projection],
                ),
                f"{config.candidate_label}_rtn_train_metrics": error_metrics(
                    dense_projection(activations[projection], rtn5),
                    expected_projection[projection],
                ),
                "four_bit_grid_sha256": diagnostics4["grid_sha256"],
                f"{config.candidate_label}_rtn_code_sha256": hashlib.sha256(
                    np.ascontiguousarray(rtn_codes5).tobytes()
                ).hexdigest(),
            }
            del scales5, biases5, rtn_codes5, codes5, scales4, biases4
            gc.collect()
            safety.release_checkpoint(
                f"layer_{layer}_expert_{expert}_{projection}_workspace_released",
                ["full Hessians", "inverse Cholesky", "permuted working weights", "projection outputs"],
            )
        train_inputs = np.asarray(moe_input[train_positions], dtype=np.float32)
        validation_inputs = np.asarray(moe_input[validation_positions], dtype=np.float32)
        five_train = dense_expert(train_inputs, five_weights)
        five_validation = dense_expert(validation_inputs, five_weights)
        five_rtn_train = dense_expert(train_inputs, five_rtn_weights)
        four_train = dense_expert(train_inputs, four_weights)
        four_validation = dense_expert(validation_inputs, four_weights)
        prior_report = _prior_report(prior, layer, expert)
        four_train_metrics = error_metrics(four_train, train_expected)
        four_validation_metrics = error_metrics(four_validation, validation_expected)
        reproduced = (
            four_train_metrics == prior_report["global_gptq_train"]
            and four_validation_metrics == prior_report["global_gptq_validation"]
            and {
                name: row["four_bit_grid_sha256"] for name, row in projection_reports.items()
            }
            == {
                name: row["grid_sha256"]
                for name, row in prior_report["projection_reports"].items()
            }
        )
        report = {
            "layer": layer,
            "expert": expert,
            "train_placements": len(train_positions),
            "validation_placements": len(validation_positions),
            "projection_reports": projection_reports,
            f"{config.candidate_label}_train_metrics": error_metrics(five_train, train_expected),
            f"{config.candidate_label}_validation_metrics": error_metrics(five_validation, validation_expected),
            f"{config.candidate_label}_rtn_train_metrics": error_metrics(five_rtn_train, train_expected),
            "four_bit_train_metrics": four_train_metrics,
            "pw0138_four_bit_validation_metrics": prior_report["global_gptq_validation"],
            "four_bit_control_reproduced": reproduced,
            "code_domain_valid": all(
                row[config.candidate_label]["maximum_code"] == config.maximum_code
                for row in projection_reports.values()
            ),
        }
        if config.prior_candidate_label is not None:
            prior_candidate = _prior_report(prerequisite, layer, expert)
            report["prior_candidate_validation_metrics"] = prior_candidate[
                f"{config.prior_candidate_label}_validation_metrics"
            ]
        reports.append(report)
        del moe_input, expert_down, train_expected, validation_expected, source_weights
        del hidden, activations, expected_projection, five_weights, five_rtn_weights
        del four_weights, five_train, five_validation, five_rtn_train, four_train
        del four_validation, train_inputs, validation_inputs
        gc.collect()
        mx.clear_cache()
        safety.release_checkpoint(
            f"layer_{layer}_expert_{expert}_complete_released",
            ["source expert", "candidate/control weights", "captured activations", "expert outputs"],
        )
    gate = _gate(reports, config)
    decision = (
        config.pass_decision
        if gate["passes"]
        else config.reject_decision
    )
    safety.release_checkpoint(
        "checkpoint_and_authorities_released",
        ["checkpoint mappings", "PW-0116 corpus", "frozen control authorities"],
    )
    safety.checkpoint("final_service_health")
    result = {
        "schema_version": 1,
        "evidence_class": config.evidence_class,
        "revision": REVISION,
        "commit": commit,
        "source_hashes": {
            "checkpoint_verification_sha256": VERIFICATION_SHA256,
            "corpus_manifest_sha256": CORPUS_SHA256,
            "pw0138_report_sha256": PW0138_SHA256,
            config.prerequisite_source_key: config.prerequisite_sha256,
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
            "three validation-visible experts; dense unpacked F16 oracle; holdout sealed; "
            "no full bank, packed kernel, accumulated model, companion hardware, endpoint, "
            "accepted tokens, or TPS"
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
    parser.add_argument("--pw0138", required=True, type=Path)
    parser.add_argument("--pw0146", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--commit", required=True)
    args = parser.parse_args()
    try:
        result = run(
            args.checkpoint,
            args.verification,
            args.corpus_manifest,
            args.pw0138,
            args.pw0146,
            args.output,
            args.commit,
        )
        print(json.dumps({"output": str(args.output), "decision": result["decision"]}))
        return 0
    except (OSError, RuntimeError, ValueError, KeyError, TypeError, json.JSONDecodeError, np.linalg.LinAlgError) as error:
        print(json.dumps({"error": str(error)}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
