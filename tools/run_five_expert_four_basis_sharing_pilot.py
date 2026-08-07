#!/usr/bin/env python3
"""Run PW-0123's five-expert/four-basis forced-sharing pilot."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np
import torch

try:
    from tools.generate_real_layer1_expert_oracle import ShardedCheckpoint, dynamic_input
    from tools.host_safety import HostSafetyMonitor
    from tools.openrouter_reference import atomic_write_new, canonical_json
    from tools.run_best_rank_real_expert_control import (
        load_capture,
        parity,
        partition_metrics,
        sha256_file,
        source_linear,
        svd_control,
    )
    from tools.run_rank768_activation_weighted_expert_pilot import (
        MPS_MEMORY_FRACTION,
        P,
        D,
        RANK,
        balanced_factors,
        dequant_weight,
        factor_expert,
        factor_hash,
        normalized_mse,
        tensor_hash,
        train_projection,
    )
except ModuleNotFoundError:
    from generate_real_layer1_expert_oracle import ShardedCheckpoint, dynamic_input
    from host_safety import HostSafetyMonitor
    from openrouter_reference import atomic_write_new, canonical_json
    from run_best_rank_real_expert_control import (
        load_capture,
        parity,
        partition_metrics,
        sha256_file,
        source_linear,
        svd_control,
    )
    from run_rank768_activation_weighted_expert_pilot import (
        MPS_MEMORY_FRACTION,
        P,
        D,
        RANK,
        balanced_factors,
        dequant_weight,
        factor_expert,
        factor_hash,
        normalized_mse,
        tensor_hash,
        train_projection,
    )


EXPERIMENT_ID = "PW-0123"
REVISION = "63651580ca774f8504f676040460aed3e1244ac1"
VERIFICATION_SHA256 = "9ddc8a99755f04ae2ea3c2484f6dd022d3f3a681b5a72c915ee4de833dbb0d03"
CORPUS_SHA256 = "b9df976876d63c1ffbbe0c70507aea8b939a749ce5b1db27cbca0b5d82cf802e"
PW0122_ANALYSIS_SHA256 = "5b5a21be9438e81e9b05a155ca365cd0dc4180be1b06a18a873362e88f60e0eb"
LAYER = 46
EXPERTS = [28, 249, 213, 125, 57]
BASES = 4
EXPECTED_COUNTS = {
    28: {"train": 100, "validation": 56, "pilot_holdout": 56},
    249: {"train": 90, "validation": 56, "pilot_holdout": 56},
    213: {"train": 94, "validation": 48, "pilot_holdout": 46},
    125: {"train": 57, "validation": 48, "pilot_holdout": 56},
    57: {"train": 17, "validation": 56, "pilot_holdout": 56},
}
SEED = 260123
SHARED_LEARNING_RATE = 0.0005
SHARED_MAX_STEPS = 150
SHARED_VALIDATE_EVERY = 5
SHARED_PATIENCE = 6
PW0122_EXPERT28 = {
    "overall": 0.17183543253342182,
    "train": 0.0741091414825072,
    "validation": 0.19566708093351987,
    "pilot_holdout": 0.2881280111180422,
}


def _mps_memory() -> dict:
    return {
        "current_allocated_bytes": int(torch.mps.current_allocated_memory()),
        "driver_allocated_bytes": int(torch.mps.driver_allocated_memory()),
        "recommended_max_bytes": int(torch.mps.recommended_max_memory()),
    }


def _equal_expert_loss(
    predictions: list[torch.Tensor], targets: list[torch.Tensor]
) -> torch.Tensor:
    if len(predictions) != len(targets) or not predictions:
        raise ValueError("PW-0123 equal-expert loss requires paired experts")
    return torch.stack(
        [normalized_mse(actual, expected) for actual, expected in zip(predictions, targets)]
    ).mean()


def _shared_predict(
    projection: str,
    expert_index: int,
    values: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    coefficients: torch.Tensor,
) -> torch.Tensor:
    combined = torch.einsum("m,mrd->rd", coefficients[expert_index], b)
    if projection == "down":
        return (values @ a[expert_index]) @ combined
    return (values @ combined.T) @ a[expert_index].T


def _shared_factor_hash(
    a: np.ndarray, b: np.ndarray, coefficients: np.ndarray
) -> str:
    digest = hashlib.sha256()
    for values in (a, b, coefficients):
        digest.update(np.ascontiguousarray(values).astype("<f4", copy=False).tobytes())
    return digest.hexdigest()


def _select_fifth_basis_from_train(
    projection: str,
    left: np.ndarray,
    bases: np.ndarray,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    train_indices: list[int],
) -> tuple[int, list[float]]:
    losses = []
    values = inputs[train_indices]
    expected = targets[train_indices]
    left_tensor = torch.from_numpy(left)
    with torch.no_grad():
        for basis in bases:
            right = torch.from_numpy(basis)
            prediction = (
                (values @ left_tensor) @ right
                if projection == "down"
                else (values @ right.T) @ left_tensor.T
            )
            losses.append(float(normalized_mse(prediction, expected)))
    return int(np.argmin(losses)), losses


def _train_shared_projection(
    projection: str,
    independent: list[tuple[np.ndarray, np.ndarray]],
    inputs: list[torch.Tensor],
    targets: list[torch.Tensor],
    partitions: list[dict[str, list[int]]],
    safety: HostSafetyMonitor,
) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray], dict]:
    if len(independent) != len(EXPERTS) or len(EXPERTS) <= BASES:
        raise ValueError("PW-0123 topology does not force sharing")
    a_initial = np.stack([factors[0] for factors in independent]).astype(np.float32)
    b_initial = np.stack([independent[index][1] for index in range(BASES)]).astype(np.float32)
    coefficients_initial = np.zeros((len(EXPERTS), BASES), dtype=np.float32)
    for index in range(BASES):
        coefficients_initial[index, index] = 1.0

    fifth = len(EXPERTS) - 1
    selected_fifth_basis, fifth_train_losses = _select_fifth_basis_from_train(
        projection,
        a_initial[fifth],
        b_initial,
        inputs[fifth],
        targets[fifth],
        partitions[fifth]["train"],
    )
    coefficients_initial[fifth, selected_fifth_basis] = 1.0

    device = torch.device("mps")
    a = torch.nn.Parameter(torch.from_numpy(a_initial).to(device))
    b = torch.nn.Parameter(torch.from_numpy(b_initial).to(device))
    coefficients = torch.nn.Parameter(torch.from_numpy(coefficients_initial).to(device))
    optimizer = torch.optim.Adam([a, b, coefficients], lr=SHARED_LEARNING_RATE)
    train_inputs = [values[part["train"]].to(device) for values, part in zip(inputs, partitions)]
    train_targets = [values[part["train"]].to(device) for values, part in zip(targets, partitions)]
    validation_inputs = [
        values[part["validation"]].to(device) for values, part in zip(inputs, partitions)
    ]
    validation_targets = [
        values[part["validation"]].to(device) for values, part in zip(targets, partitions)
    ]
    torch.mps.synchronize()
    safety.checkpoint(f"shared_{projection}_parameters_migrated")
    memory = [{"phase": "parameters_migrated", **_mps_memory()}]

    def predictions(batch: list[torch.Tensor]) -> list[torch.Tensor]:
        return [
            _shared_predict(projection, index, values, a, b, coefficients)
            for index, values in enumerate(batch)
        ]

    history = []
    best_loss = float("inf")
    best_step = None
    best = None
    stale = 0
    coefficient_gradient_norm = None
    started = time.perf_counter()
    for step in range(SHARED_MAX_STEPS + 1):
        if step % SHARED_VALIDATE_EVERY == 0:
            with torch.no_grad():
                per_expert = [
                    float(normalized_mse(actual, expected).cpu())
                    for actual, expected in zip(
                        predictions(validation_inputs), validation_targets
                    )
                ]
                validation_loss = float(np.mean(per_expert))
            if not np.isfinite(validation_loss) or not np.isfinite(per_expert).all():
                raise ValueError(f"PW-0123 shared {projection} validation is non-finite")
            improved = validation_loss < best_loss
            history.append(
                {
                    "step": step,
                    "equal_expert_validation_normalized_mse": validation_loss,
                    "per_expert_validation_normalized_mse": {
                        str(expert): value for expert, value in zip(EXPERTS, per_expert)
                    },
                    "improved": improved,
                }
            )
            if improved:
                best_loss = validation_loss
                best_step = step
                best = (
                    a.detach().cpu().numpy().copy(),
                    b.detach().cpu().numpy().copy(),
                    coefficients.detach().cpu().numpy().copy(),
                )
                stale = 0
            else:
                stale += 1
            torch.mps.synchronize()
            safety.checkpoint(f"shared_{projection}_validation_checkpoint_{step}")
            memory.append({"phase": f"validation_{step}", **_mps_memory()})
            if stale >= SHARED_PATIENCE:
                break
        if step == SHARED_MAX_STEPS:
            break
        optimizer.zero_grad(set_to_none=True)
        loss = _equal_expert_loss(predictions(train_inputs), train_targets)
        if not bool(torch.isfinite(loss).item()):
            raise ValueError(f"PW-0123 shared {projection} train loss is non-finite")
        loss.backward()
        if step == 0:
            coefficient_gradient_norm = float(torch.linalg.vector_norm(coefficients.grad).cpu())
            if not np.isfinite(coefficient_gradient_norm) or coefficient_gradient_norm <= 0:
                raise ValueError(f"PW-0123 shared {projection} coefficient gradient is invalid")
            torch.mps.synchronize()
            safety.checkpoint(f"shared_{projection}_first_backward_complete")
            memory.append({"phase": "first_backward", **_mps_memory()})
        optimizer.step()
    improved_over_initial = best_loss < history[0][
        "equal_expert_validation_normalized_mse"
    ]
    if best is None or best_step is None:
        raise ValueError(f"PW-0123 shared {projection} lacks a finite checkpoint")

    with torch.no_grad():
        a.copy_(torch.from_numpy(best[0]).to(device))
        b.copy_(torch.from_numpy(best[1]).to(device))
        coefficients.copy_(torch.from_numpy(best[2]).to(device))
        selected_validation = [
            float(normalized_mse(actual, expected).cpu())
            for actual, expected in zip(predictions(validation_inputs), validation_targets)
        ]
        selected_train = [
            float(normalized_mse(actual, expected).cpu())
            for actual, expected in zip(predictions(train_inputs), train_targets)
        ]
    wall_ms = (time.perf_counter() - started) * 1000.0
    loss = None
    del a, b, coefficients, optimizer
    del train_inputs, train_targets, validation_inputs, validation_targets
    gc.collect()
    torch.mps.synchronize()
    torch.mps.empty_cache()
    release = _mps_memory()
    safety.release_checkpoint(
        f"shared_{projection}_released",
        [
            f"shared {projection} expert factors",
            f"shared {projection} bases",
            f"shared {projection} coefficients",
            f"shared {projection} Adam state",
            f"shared {projection} activation batches",
        ],
    )
    if release["current_allocated_bytes"] != 0:
        raise ValueError(f"PW-0123 shared {projection} MPS allocation did not release")
    return best, {
        "initial_equal_expert_validation_normalized_mse": history[0][
            "equal_expert_validation_normalized_mse"
        ],
        "selected_equal_expert_validation_normalized_mse": best_loss,
        "selected_step": best_step,
        "improved_over_initial": improved_over_initial,
        "selected_per_expert_validation_normalized_mse": {
            str(expert): value for expert, value in zip(EXPERTS, selected_validation)
        },
        "selected_per_expert_train_normalized_mse": {
            str(expert): value for expert, value in zip(EXPERTS, selected_train)
        },
        "fifth_expert_train_only_basis_losses": fifth_train_losses,
        "fifth_expert_selected_basis": selected_fifth_basis,
        "coefficient_gradient_norm_at_first_backward": coefficient_gradient_norm,
        "history": history,
        "wall_ms": wall_ms,
        "memory": memory,
        "release_memory": release,
        "initial_factor_sha256": _shared_factor_hash(
            a_initial, b_initial, coefficients_initial
        ),
        "selected_factor_sha256": _shared_factor_hash(*best),
    }


def _factor_metrics(
    factors: dict[str, tuple[np.ndarray, np.ndarray]],
    inputs: torch.Tensor,
    expected: torch.Tensor,
    positions: list[int],
) -> dict:
    output = factor_expert(factors, inputs)
    return {
        "overall": parity(output, expected),
        "partitions": partition_metrics(output, expected, positions),
    }


def _relative_l2(metrics: dict, partition: str) -> float:
    if partition == "overall":
        return metrics["overall"]["relative_l2"]
    return metrics["partitions"][partition]["metrics"]["relative_l2"]


def _full_bank_ledger() -> dict:
    source_weight_bytes = 256 * P * D
    source_scale_bytes = 256 * (P // 128) * (D // 128) * 4
    factor_weight_bytes = 256 * P * RANK + BASES * RANK * D
    factor_scale_bytes = (
        256 * (P // 128) * (RANK // 128) * 4
        + BASES * (RANK // 128) * (D // 128) * 4
    )
    coefficient_bytes = 256 * BASES * 2
    source_bytes = source_weight_bytes + source_scale_bytes
    candidate_bytes = factor_weight_bytes + factor_scale_bytes + coefficient_bytes
    source_multiplications = 8 * P * D
    candidate_multiplications = BASES * RANK * D + 8 * P * RANK + 8 * BASES * RANK
    return {
        "hypothesis": "FP8 E4M3FN factors with F32 group-128 scales and F16 coefficients",
        "source_projection_bytes": source_bytes,
        "candidate_projection_bytes": candidate_bytes,
        "candidate_to_source_byte_ratio": candidate_bytes / source_bytes,
        "source_projection_multiplications": source_multiplications,
        "candidate_projection_multiplications": candidate_multiplications,
        "candidate_to_source_multiplication_ratio": candidate_multiplications
        / source_multiplications,
        "byte_gate_passed": candidate_bytes / source_bytes <= 0.25,
        "multiplication_gate_passed": candidate_multiplications / source_multiplications
        <= 0.5,
    }


def run(
    checkpoint_root: Path,
    verification_path: Path,
    corpus_manifest_path: Path,
    pw0122_analysis_path: Path,
    output_path: Path,
    commit: str,
) -> dict:
    if output_path.exists():
        raise ValueError(f"refusing to overwrite {output_path}")
    if len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit):
        raise ValueError("implementation commit must be lowercase 40-hex")
    if sha256_file(verification_path) != VERIFICATION_SHA256:
        raise ValueError("PW-0123 checkpoint verification hash mismatch")
    if sha256_file(corpus_manifest_path) != CORPUS_SHA256:
        raise ValueError("PW-0123 corpus manifest hash mismatch")
    if sha256_file(pw0122_analysis_path) != PW0122_ANALYSIS_SHA256:
        raise ValueError("PW-0123 PW-0122 analysis hash mismatch")
    if not torch.backends.mps.is_built() or not torch.backends.mps.is_available():
        raise ValueError("PW-0123 requires an available PyTorch MPS backend")

    complete_started = time.perf_counter()
    safety = HostSafetyMonitor()
    torch.mps.set_per_process_memory_fraction(MPS_MEMORY_FRACTION)
    torch.mps.empty_cache()
    corpus = json.loads(corpus_manifest_path.read_text())
    authority = next(row for row in corpus["layers"] if row["layer"] == LAYER)
    root = corpus_manifest_path.parent
    moe_input = load_capture(root, authority["captures"]["moe_input"])
    expert_down = load_capture(root, authority["captures"]["expert_down"])
    schedule_by_expert = {}
    offset = 0
    for schedule in authority["expert_schedule"]:
        schedule_by_expert[schedule["expert"]] = (offset, schedule["positions"])
        offset += len(schedule["positions"])
    safety.checkpoint("authenticated_corpus_loaded")

    checkpoint = ShardedCheckpoint(checkpoint_root, verification_path)
    expert_data = []
    independent_factors = []
    independent_reports = []
    svd_reports = []
    torch.manual_seed(SEED)
    for expert in EXPERTS:
        row_offset, positions = schedule_by_expert[expert]
        partitions = {
            "train": [index for index, position in enumerate(positions) if position < 112],
            "validation": [
                index for index, position in enumerate(positions) if 112 <= position < 168
            ],
            "pilot_holdout": [
                index for index, position in enumerate(positions) if position >= 168
            ],
        }
        if {name: len(rows) for name, rows in partitions.items()} != EXPECTED_COUNTS[expert]:
            raise ValueError(f"PW-0123 expert {expert} partition mismatch")
        inputs = torch.from_numpy(np.asarray(moe_input[positions]).copy()).to(torch.bfloat16)
        expected = torch.from_numpy(
            np.asarray(expert_down[row_offset : row_offset + len(positions)]).copy()
        ).to(torch.bfloat16)
        prefix = f"model.layers.{LAYER}.mlp.experts.{expert}"
        weights = {
            projection: dequant_weight(checkpoint, f"{prefix}.{projection}_proj.weight")
            for projection in ("gate", "up", "down")
        }
        source_gate = source_linear(weights["gate"], inputs)
        source_up = source_linear(weights["up"], inputs)
        source_hidden = (torch.nn.functional.silu(source_gate) * source_up).to(torch.bfloat16)
        source_output = source_linear(weights["down"], source_hidden)
        source_parity = parity(source_output, expected)
        if source_parity["equality_fraction"] != 1.0:
            raise ValueError(f"PW-0123 expert {expert} source oracle is not bit exact")
        projection_inputs = {
            "gate": dynamic_input(inputs).float(),
            "up": dynamic_input(inputs).float(),
            "down": dynamic_input(source_hidden).float(),
        }
        projection_targets = {
            "gate": source_gate.float(),
            "up": source_up.float(),
            "down": expected.float(),
        }
        decompositions = {}
        initial = {}
        selected = {}
        projection_training = {}
        projection_svd = {}
        for projection in ("gate", "up", "down"):
            canonical = np.ascontiguousarray(
                weights[projection].T if projection == "down" else weights[projection]
            )
            started = time.perf_counter()
            decomposition = np.linalg.svd(canonical, full_matrices=False)
            svd_wall_ms = (time.perf_counter() - started) * 1000.0
            safety.checkpoint(f"expert_{expert}_{projection}_svd_complete")
            decompositions[projection] = decomposition
            initial[projection] = balanced_factors(decomposition, RANK)
            fitted, training_report = train_projection(
                EXPERIMENT_ID,
                projection,
                initial[projection],
                projection_inputs[projection],
                projection_targets[projection],
                partitions["train"],
                partitions["validation"],
                safety,
                phase_prefix=f"expert_{expert}_{projection}",
                require_improvement=False,
            )
            selected[projection] = fitted
            projection_training[projection] = training_report
            projection_svd[projection] = {"wall_ms": svd_wall_ms}
            del canonical
        baseline_gate = svd_control(decompositions["gate"], inputs, RANK, down=False)
        baseline_up = svd_control(decompositions["up"], inputs, RANK, down=False)
        baseline_hidden = (
            torch.nn.functional.silu(baseline_gate) * baseline_up
        ).to(torch.bfloat16)
        baseline_output = svd_control(
            decompositions["down"], baseline_hidden, RANK, down=True
        )
        svd_metrics = {
            "overall": parity(baseline_output, expected),
            "partitions": partition_metrics(baseline_output, expected, positions),
        }
        independent_metrics = _factor_metrics(selected, inputs, expected, positions)
        if expert == 28:
            for name, frozen in PW0122_EXPERT28.items():
                if abs(_relative_l2(independent_metrics, name) - frozen) > 1e-6:
                    raise ValueError(f"PW-0123 expert-28 independent {name} mismatch")
        expert_data.append(
            {
                "expert": expert,
                "positions": positions,
                "partitions": partitions,
                "inputs": inputs,
                "expected": expected,
                "projection_inputs": projection_inputs,
                "projection_targets": projection_targets,
            }
        )
        independent_factors.append(selected)
        independent_reports.append(
            {
                "expert": expert,
                "source_oracle_parity": source_parity,
                "projection_training": projection_training,
                "complete_expert_metrics": independent_metrics,
                "factor_sha256": {
                    projection: factor_hash(selected[projection])
                    for projection in ("gate", "up", "down")
                },
                "training_tensor_sha256": {
                    projection: {
                        "inputs": tensor_hash(projection_inputs[projection]),
                        "targets": tensor_hash(projection_targets[projection]),
                    }
                    for projection in ("gate", "up", "down")
                },
            }
        )
        svd_reports.append(
            {
                "expert": expert,
                "projection_svd": projection_svd,
                "complete_expert_metrics": svd_metrics,
            }
        )
        del weights, decompositions, decomposition, initial
        gc.collect()
        safety.release_checkpoint(
            f"expert_{expert}_source_and_svd_released",
            [f"expert {expert} decoded weights", f"expert {expert} SVD decompositions"],
        )

    shared_factors = {}
    shared_training = {}
    projection_gates = {}
    for projection in ("gate", "up", "down"):
        factors, report = _train_shared_projection(
            projection,
            [row[projection] for row in independent_factors],
            [row["projection_inputs"][projection] for row in expert_data],
            [row["projection_targets"][projection] for row in expert_data],
            [row["partitions"] for row in expert_data],
            safety,
        )
        shared_factors[projection] = factors
        shared_training[projection] = report
        independent_validation = [
            row["projection_training"][projection][
                "selected_validation_normalized_mse"
            ]
            for row in independent_reports
        ]
        shared_validation = [
            report["selected_per_expert_validation_normalized_mse"][str(expert)]
            for expert in EXPERTS
        ]
        aggregate_ratio = float(np.mean(shared_validation) / np.mean(independent_validation))
        per_expert_ratios = {
            str(expert): shared / independent
            for expert, shared, independent in zip(
                EXPERTS, shared_validation, independent_validation
            )
        }
        projection_gates[projection] = {
            "independent_equal_expert_validation_normalized_mse": float(
                np.mean(independent_validation)
            ),
            "shared_equal_expert_validation_normalized_mse": float(
                np.mean(shared_validation)
            ),
            "shared_to_independent_aggregate_ratio": aggregate_ratio,
            "shared_to_independent_per_expert_ratio": per_expert_ratios,
            "aggregate_gate_passed": aggregate_ratio <= 1.25,
            "per_expert_gate_passed": all(value <= 1.5 for value in per_expert_ratios.values()),
        }

    shared_reports = []
    for index, data in enumerate(expert_data):
        factors = {}
        for projection in ("gate", "up", "down"):
            a, b, coefficients = shared_factors[projection]
            combined = np.einsum("m,mrd->rd", coefficients[index], b, optimize=True)
            factors[projection] = (
                np.ascontiguousarray(a[index]),
                np.ascontiguousarray(combined),
            )
        metrics = _factor_metrics(
            factors, data["inputs"], data["expected"], data["positions"]
        )
        shared_reports.append({"expert": data["expert"], "complete_expert_metrics": metrics})

    complete_gates = {}
    for partition in ("validation", "pilot_holdout"):
        independent_values = [
            _relative_l2(row["complete_expert_metrics"], partition)
            for row in independent_reports
        ]
        shared_values = [
            _relative_l2(row["complete_expert_metrics"], partition)
            for row in shared_reports
        ]
        ratio = float(np.mean(shared_values) / np.mean(independent_values))
        complete_gates[partition] = {
            "independent_equal_expert_relative_l2": float(np.mean(independent_values)),
            "shared_equal_expert_relative_l2": float(np.mean(shared_values)),
            "shared_to_independent_ratio": ratio,
            "aggregate_gate_passed": ratio <= 1.25,
        }
    holdout_improvement_gates = {}
    for svd, shared in zip(svd_reports, shared_reports):
        expert = svd["expert"]
        svd_error = _relative_l2(svd["complete_expert_metrics"], "pilot_holdout")
        shared_error = _relative_l2(shared["complete_expert_metrics"], "pilot_holdout")
        holdout_improvement_gates[str(expert)] = {
            "svd_relative_l2": svd_error,
            "shared_relative_l2": shared_error,
            "relative_error_reduction": 1.0 - shared_error / svd_error,
            "passed": shared_error <= 0.75 * svd_error,
        }

    independent_improvement_passed = all(
        projection_report["improved_over_initial"]
        for expert_report in independent_reports
        for projection_report in expert_report["projection_training"].values()
    )
    physical_ledger = _full_bank_ledger()
    passed = (
        independent_improvement_passed
        and all(
            row["aggregate_gate_passed"] and row["per_expert_gate_passed"]
            for row in projection_gates.values()
        )
        and all(row["aggregate_gate_passed"] for row in complete_gates.values())
        and all(row["passed"] for row in holdout_improvement_gates.values())
        and physical_ledger["byte_gate_passed"]
        and physical_ledger["multiplication_gate_passed"]
    )
    safety.checkpoint("complete_shared_expert_evaluation")
    del a, b, coefficients, combined, factors
    del selected, projection_inputs, projection_targets
    del fitted, training_report, independent_metrics, svd_metrics
    del source_gate, source_up, source_hidden, source_output
    del baseline_gate, baseline_up, baseline_hidden, baseline_output
    del inputs, expected
    del shared_factors, independent_factors, expert_data
    gc.collect()
    torch.mps.synchronize()
    torch.mps.empty_cache()
    final_mps_memory = _mps_memory()
    safety.release_checkpoint(
        "all_pilot_factors_and_activations_released",
        [
            "independent fitted factors",
            "shared fitted factors",
            "all expert activation targets",
            "MPS cache",
        ],
    )
    safety.checkpoint("final_service_health")
    if final_mps_memory["current_allocated_bytes"] != 0:
        raise ValueError("PW-0123 final MPS allocation did not release")

    report = {
        "schema_version": 1,
        "evidence_class": "pw0123_five_expert_four_basis_sharing_pilot",
        "revision": REVISION,
        "commit": commit,
        "checkpoint_verification_sha256": VERIFICATION_SHA256,
        "corpus_manifest_sha256": CORPUS_SHA256,
        "pw0122_analysis_sha256": PW0122_ANALYSIS_SHA256,
        "layer": LAYER,
        "experts": EXPERTS,
        "basis_count": BASES,
        "rank": RANK,
        "expected_partition_counts": {
            str(expert): counts for expert, counts in EXPECTED_COUNTS.items()
        },
        "configuration": {
            "seed": SEED,
            "shared_optimizer": "Adam",
            "shared_learning_rate": SHARED_LEARNING_RATE,
            "shared_maximum_steps": SHARED_MAX_STEPS,
            "shared_validate_every": SHARED_VALIDATE_EVERY,
            "shared_patience_checks": SHARED_PATIENCE,
            "mps_memory_fraction": MPS_MEMORY_FRACTION,
            "shared_projection_parameter_values": len(EXPERTS) * P * RANK
            + BASES * RANK * D
            + len(EXPERTS) * BASES,
            "shared_projection_semantic_adam_bytes": (
                len(EXPERTS) * P * RANK
                + BASES * RANK * D
                + len(EXPERTS) * BASES
            )
            * 4
            * 4,
        },
        "svd_controls": svd_reports,
        "independent_activation_weighted_controls": independent_reports,
        "shared_projection_training": shared_training,
        "shared_complete_experts": shared_reports,
        "projection_sharing_gates": projection_gates,
        "complete_expert_sharing_gates": complete_gates,
        "per_expert_holdout_improvement_gates": holdout_improvement_gates,
        "all_independent_projection_validation_improvement_gate_passed": independent_improvement_passed,
        "full_bank_physical_ledger": physical_ledger,
        "gates_passed": passed,
        "final_mps_memory": final_mps_memory,
        "safety_snapshots": safety.evidence(),
        "complete_wall_ms": (time.perf_counter() - complete_started) * 1000.0,
        "batch_size": 1,
        "concurrency": 1,
        "accepted_tokens": 0,
        "A": 0,
        "decision": (
            "authorize_broader_same_layer_shared_basis_fit_contract"
            if passed
            else "reject_rank768_four_basis_sharing_under_current_objective"
        ),
        "limitations": "five well-covered layer-46 experts on one English sequential corpus; no other experts, rare identities, broader modalities, persisted artifact, quantized factor evaluation, kernel, endpoint, or TPS",
        "performance_claim": None,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
    }
    atomic_write_new(output_path, canonical_json(report))
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--verification", required=True, type=Path)
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--pw0122-analysis", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--commit", required=True)
    arguments = parser.parse_args()
    try:
        result = run(
            arguments.checkpoint,
            arguments.verification,
            arguments.corpus,
            arguments.pw0122_analysis,
            arguments.output,
            arguments.commit,
        )
        print(json.dumps({"output": str(arguments.output), "decision": result["decision"]}))
        return 0
    except (OSError, RuntimeError, ValueError, KeyError, TypeError, StopIteration, json.JSONDecodeError) as error:
        print(json.dumps({"error": str(error)}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
