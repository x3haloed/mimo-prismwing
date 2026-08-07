#!/usr/bin/env python3
"""Run PW-0121's rank-768 activation-weighted expert pilot."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
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


REVISION = "63651580ca774f8504f676040460aed3e1244ac1"
VERIFICATION_SHA256 = "9ddc8a99755f04ae2ea3c2484f6dd022d3f3a681b5a72c915ee4de833dbb0d03"
CORPUS_SHA256 = "b9df976876d63c1ffbbe0c70507aea8b939a749ce5b1db27cbca0b5d82cf802e"
PW0119_SOURCE_SHA256 = "3e7729dfff3d9ab6793d8e74d29ad20bb3c877bea328ae53d9325737c717c8fb"
RANK = 768
P, D = 2048, 4096
LEARNING_RATE = 0.001
MAX_STEPS = 100
VALIDATE_EVERY = 5
PATIENCE_CHECKS = 4
MPS_MEMORY_FRACTION = 0.60


@dataclass(frozen=True)
class PilotSpec:
    experiment_id: str
    evidence_class: str
    layer: int
    expert: int
    rank: int
    seed: int
    partition_counts: dict[str, int]
    pw0119_baseline: dict[str, float]
    validation_maximum: float
    holdout_maximum: float
    pass_decision: str
    fail_decision: str
    limitations: str


PW0121_SPEC = PilotSpec(
    experiment_id="PW-0121",
    evidence_class="pw0121_rank768_activation_weighted_expert_pilot",
    layer=24,
    expert=23,
    rank=768,
    seed=260121,
    partition_counts={"train": 65, "validation": 46, "pilot_holdout": 56},
    pw0119_baseline={
        "overall": 0.7097717469432467,
        "train": 0.731277796685779,
        "validation": 0.7103805967306607,
        "pilot_holdout": 0.6849577905886747,
    },
    validation_maximum=0.5327854475479955,
    holdout_maximum=0.5137183429415060,
    pass_decision="authorize_layer46_activation_weighted_rank768_pilot",
    fail_decision="reject_current_activation_weighted_rank768_factor_fit",
    limitations="one hot middle-layer expert on one English sequential corpus; independent factors only, no shared bases, broad corpus, kernel, endpoint, or TPS",
)


PW0122_SPEC = PilotSpec(
    experiment_id="PW-0122",
    evidence_class="pw0122_layer46_rank768_activation_weighted_expert_pilot",
    layer=46,
    expert=28,
    rank=768,
    seed=260122,
    partition_counts={"train": 100, "validation": 56, "pilot_holdout": 56},
    pw0119_baseline={
        "overall": 0.5694250892637611,
        "train": 0.5775213424014214,
        "validation": 0.572330134931118,
        "pilot_holdout": 0.5458150398186078,
    },
    validation_maximum=0.4292476011983385,
    holdout_maximum=0.4093612798639559,
    pass_decision="authorize_multi_expert_shared_basis_pilot_contract",
    fail_decision="reject_depth_general_activation_weighted_rank768_factor_fit",
    limitations="one hot late-layer expert on one English sequential corpus; independent factors only, no shared bases, broad corpus, kernel, endpoint, or TPS",
)


PW0125_SPEC = PilotSpec(
    experiment_id="PW-0125",
    evidence_class="pw0125_rank512_activation_weighted_capacity_control",
    layer=46,
    expert=28,
    rank=512,
    seed=260125,
    partition_counts={"train": 100, "validation": 56, "pilot_holdout": 56},
    pw0119_baseline={
        "overall": 0.6747763876584113,
        "train": 0.6822727543140975,
        "validation": 0.6730991256068856,
        "pilot_holdout": 0.6568507915821798,
    },
    validation_maximum=0.24458385116689985,
    holdout_maximum=0.36016001389755276,
    pass_decision="authorize_rank512_eight_basis_forced_sharing_contract",
    fail_decision="reject_rank512_eight_basis_branch_on_independent_capacity",
    limitations="one hot late-layer expert on one English sequential corpus; independent rank-512 factors only, no shared bases, broad corpus, kernel, endpoint, or TPS",
)


def dequant_weight(checkpoint: ShardedCheckpoint, name: str) -> np.ndarray:
    weight = checkpoint.tensor(name)
    scale = checkpoint.tensor(name + "_scale_inv")
    expected = ((weight.shape[0] + 127) // 128, (weight.shape[1] + 127) // 128)
    if (
        weight.dtype != torch.float8_e4m3fn
        or tuple(scale.shape) != expected
        or weight.shape[0] % 128
        or weight.shape[1] % 128
    ):
        raise ValueError(f"{name}: source FP8 layout mismatch")
    expanded = scale.float().repeat_interleave(128, 0).repeat_interleave(128, 1)
    result = (weight.float() * expanded).numpy()
    if not np.isfinite(result).all():
        raise ValueError(f"{name}: non-finite decoded weight")
    return result


def balanced_factors(
    decomposition: tuple[np.ndarray, np.ndarray, np.ndarray], rank: int
) -> tuple[np.ndarray, np.ndarray]:
    u, singular, vt = decomposition
    root = np.sqrt(singular[:rank]).astype(np.float32, copy=False)
    left = np.ascontiguousarray(u[:, :rank] * root)
    right = np.ascontiguousarray(root[:, None] * vt[:rank])
    return left, right


def normalized_mse(actual: torch.Tensor, expected: torch.Tensor) -> torch.Tensor:
    return torch.mean((actual - expected) ** 2) / torch.mean(expected**2).clamp(min=1e-30)


def factor_linear(
    factors: tuple[np.ndarray, np.ndarray], values: torch.Tensor, *, down: bool
) -> torch.Tensor:
    left = torch.from_numpy(factors[0])
    right = torch.from_numpy(factors[1])
    quantized = dynamic_input(values).float()
    output = (quantized @ left) @ right if down else (quantized @ right.T) @ left.T
    return output.to(torch.bfloat16)


def factor_expert(
    factors: dict[str, tuple[np.ndarray, np.ndarray]], values: torch.Tensor
) -> torch.Tensor:
    gate = factor_linear(factors["gate"], values, down=False)
    up = factor_linear(factors["up"], values, down=False)
    hidden = (torch.nn.functional.silu(gate) * up).to(torch.bfloat16)
    return factor_linear(factors["down"], hidden, down=True)


def factor_hash(factors: tuple[np.ndarray, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for factor in factors:
        digest.update(np.ascontiguousarray(factor).astype("<f4", copy=False).tobytes())
    return digest.hexdigest()


def tensor_hash(values: torch.Tensor) -> str:
    array = values.detach().cpu().numpy().astype("<f4", copy=False)
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _mps_memory() -> dict:
    return {
        "current_allocated_bytes": int(torch.mps.current_allocated_memory()),
        "driver_allocated_bytes": int(torch.mps.driver_allocated_memory()),
        "recommended_max_bytes": int(torch.mps.recommended_max_memory()),
    }


def train_projection(
    experiment_id: str,
    projection: str,
    initial: tuple[np.ndarray, np.ndarray],
    inputs: torch.Tensor,
    targets: torch.Tensor,
    train_indices: list[int],
    validation_indices: list[int],
    safety: HostSafetyMonitor,
    *,
    phase_prefix: str | None = None,
    require_improvement: bool = True,
) -> tuple[tuple[np.ndarray, np.ndarray], dict]:
    phase_name = phase_prefix or projection
    device = torch.device("mps")
    left = torch.nn.Parameter(torch.from_numpy(initial[0]).to(device))
    right = torch.nn.Parameter(torch.from_numpy(initial[1]).to(device))
    optimizer = torch.optim.Adam([left, right], lr=LEARNING_RATE)
    train_input = inputs[train_indices].to(device)
    train_target = targets[train_indices].to(device)
    validation_input = inputs[validation_indices].to(device)
    validation_target = targets[validation_indices].to(device)
    torch.mps.synchronize()
    safety.checkpoint(f"{phase_name}_parameters_migrated")
    memory = [{"phase": "parameters_migrated", **_mps_memory()}]

    def predict(values: torch.Tensor) -> torch.Tensor:
        return (values @ left) @ right if projection == "down" else (values @ right.T) @ left.T

    history = []
    best_loss = float("inf")
    best_step = None
    best = None
    stale_checks = 0
    started = time.perf_counter()
    for step in range(MAX_STEPS + 1):
        if step % VALIDATE_EVERY == 0:
            with torch.no_grad():
                validation_loss = float(
                    normalized_mse(predict(validation_input), validation_target).cpu()
                )
            if not np.isfinite(validation_loss):
                raise ValueError(f"{experiment_id} {projection} non-finite validation loss")
            improved = validation_loss < best_loss
            history.append(
                {"step": step, "validation_normalized_mse": validation_loss, "improved": improved}
            )
            if improved:
                best_loss = validation_loss
                best_step = step
                best = (
                    left.detach().cpu().numpy().copy(),
                    right.detach().cpu().numpy().copy(),
                )
                stale_checks = 0
            else:
                stale_checks += 1
            torch.mps.synchronize()
            safety.checkpoint(f"{phase_name}_validation_checkpoint_{step}")
            memory.append({"phase": f"validation_{step}", **_mps_memory()})
            if stale_checks >= PATIENCE_CHECKS:
                break
        if step == MAX_STEPS:
            break
        optimizer.zero_grad(set_to_none=True)
        loss = normalized_mse(predict(train_input), train_target)
        if not bool(torch.isfinite(loss).item()):
            raise ValueError(f"{experiment_id} {projection} non-finite train loss")
        loss.backward()
        if step == 0:
            torch.mps.synchronize()
            safety.checkpoint(f"{phase_name}_first_backward_complete")
            memory.append({"phase": "first_backward", **_mps_memory()})
        optimizer.step()
    improved_over_initial = best_loss < history[0]["validation_normalized_mse"]
    if best is None or best_step is None or (require_improvement and not improved_over_initial):
        raise ValueError(f"{experiment_id} {projection} did not improve validation loss")
    with torch.no_grad():
        left.copy_(torch.from_numpy(best[0]).to(device))
        right.copy_(torch.from_numpy(best[1]).to(device))
        train_loss = float(normalized_mse(predict(train_input), train_target).cpu())
    wall_ms = (time.perf_counter() - started) * 1000.0
    loss = None
    del left, right, optimizer, train_input, train_target, validation_input, validation_target
    gc.collect()
    torch.mps.synchronize()
    torch.mps.empty_cache()
    release_memory = _mps_memory()
    safety.release_checkpoint(
        f"{phase_name}_projection_released",
        [f"{projection} MPS factors", f"{projection} Adam state", f"{projection} activation batches"],
    )
    if release_memory["current_allocated_bytes"] != 0:
        raise ValueError(f"{experiment_id} {projection} MPS allocation did not release")
    return best, {
        "initial_validation_normalized_mse": history[0]["validation_normalized_mse"],
        "selected_validation_normalized_mse": best_loss,
        "selected_step": best_step,
        "improved_over_initial": improved_over_initial,
        "final_parameter_train_normalized_mse": train_loss,
        "validation_history": history,
        "wall_ms": wall_ms,
        "memory": memory,
        "release_memory": release_memory,
        "initial_factor_sha256": factor_hash(initial),
        "selected_factor_sha256": factor_hash(best),
    }


def run(
    checkpoint_root: Path,
    verification_path: Path,
    corpus_manifest_path: Path,
    pw0119_path: Path,
    output_path: Path,
    commit: str,
    spec: PilotSpec = PW0121_SPEC,
) -> dict:
    if output_path.exists():
        raise ValueError(f"refusing to overwrite {output_path}")
    if len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit):
        raise ValueError("implementation commit must be lowercase 40-hex")
    if spec.rank not in {128, 512, 768}:
        raise ValueError(f"{spec.experiment_id} unsupported rank")
    if sha256_file(verification_path) != VERIFICATION_SHA256:
        raise ValueError(f"{spec.experiment_id} checkpoint verification hash mismatch")
    if sha256_file(corpus_manifest_path) != CORPUS_SHA256:
        raise ValueError(f"{spec.experiment_id} corpus manifest hash mismatch")
    if sha256_file(pw0119_path) != PW0119_SOURCE_SHA256:
        raise ValueError(f"{spec.experiment_id} PW-0119 authority hash mismatch")
    if not torch.backends.mps.is_built() or not torch.backends.mps.is_available():
        raise ValueError(f"{spec.experiment_id} requires an available PyTorch MPS backend")

    complete_started = time.perf_counter()
    safety = HostSafetyMonitor()
    torch.mps.set_per_process_memory_fraction(MPS_MEMORY_FRACTION)
    torch.mps.empty_cache()
    corpus = json.loads(corpus_manifest_path.read_text())
    if corpus.get("revision") != REVISION:
        raise ValueError(f"{spec.experiment_id} corpus revision mismatch")
    authority = next(row for row in corpus["layers"] if row["layer"] == spec.layer)
    root = corpus_manifest_path.parent
    moe_input = load_capture(root, authority["captures"]["moe_input"])
    expert_down = load_capture(root, authority["captures"]["expert_down"])
    offset = 0
    positions = None
    for schedule in authority["expert_schedule"]:
        if schedule["expert"] == spec.expert:
            positions = schedule["positions"]
            break
        offset += len(schedule["positions"])
    if positions is None or len(positions) != sum(spec.partition_counts.values()):
        raise ValueError(f"{spec.experiment_id} expert schedule mismatch")
    partition_indices = {
        "train": [index for index, position in enumerate(positions) if position < 112],
        "validation": [index for index, position in enumerate(positions) if 112 <= position < 168],
        "pilot_holdout": [index for index, position in enumerate(positions) if position >= 168],
    }
    if {name: len(rows) for name, rows in partition_indices.items()} != spec.partition_counts:
        raise ValueError(f"{spec.experiment_id} partition coverage mismatch")
    inputs = torch.from_numpy(np.asarray(moe_input[positions]).copy()).to(torch.bfloat16)
    expected = torch.from_numpy(
        np.asarray(expert_down[offset : offset + len(positions)]).copy()
    ).to(torch.bfloat16)
    safety.checkpoint("authenticated_corpus_loaded")

    checkpoint = ShardedCheckpoint(checkpoint_root, verification_path)
    prefix = f"model.layers.{spec.layer}.mlp.experts.{spec.expert}"
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
        raise ValueError(f"{spec.experiment_id} source oracle is not bit exact")
    training_inputs = {
        "gate": dynamic_input(inputs).float(),
        "up": dynamic_input(inputs).float(),
        "down": dynamic_input(source_hidden).float(),
    }
    training_targets = {
        "gate": source_gate.float(),
        "up": source_up.float(),
        "down": expected.float(),
    }
    training_tensor_hashes = {
        projection: {
            "inputs": tensor_hash(training_inputs[projection]),
            "targets": tensor_hash(training_targets[projection]),
        }
        for projection in ("gate", "up", "down")
    }
    safety.checkpoint("source_projection_targets_derived")

    initial_factors = {}
    svd_decompositions = {}
    selected_factors = {}
    projection_reports = {}
    torch.manual_seed(spec.seed)
    for projection in ("gate", "up", "down"):
        canonical = np.ascontiguousarray(
            weights[projection].T if projection == "down" else weights[projection]
        )
        started = time.perf_counter()
        decomposition = np.linalg.svd(canonical, full_matrices=False)
        svd_wall_ms = (time.perf_counter() - started) * 1000.0
        safety.checkpoint(f"{projection}_svd_complete")
        initial = balanced_factors(decomposition, spec.rank)
        svd_decompositions[projection] = decomposition
        initial_factors[projection] = initial
        selected, training_report = train_projection(
            spec.experiment_id,
            projection,
            initial,
            training_inputs[projection],
            training_targets[projection],
            partition_indices["train"],
            partition_indices["validation"],
            safety,
        )
        selected_factors[projection] = selected
        projection_reports[projection] = {
            "svd_wall_ms": svd_wall_ms,
            **training_report,
        }
        del canonical
        gc.collect()

    baseline_gate = svd_control(svd_decompositions["gate"], inputs, spec.rank, down=False)
    baseline_up = svd_control(svd_decompositions["up"], inputs, spec.rank, down=False)
    baseline_hidden = (
        torch.nn.functional.silu(baseline_gate) * baseline_up
    ).to(torch.bfloat16)
    baseline_output = svd_control(
        svd_decompositions["down"], baseline_hidden, spec.rank, down=True
    )
    baseline = {
        "overall": parity(baseline_output, expected),
        "partitions": partition_metrics(baseline_output, expected, positions),
    }
    for name, expected_relative_l2 in spec.pw0119_baseline.items():
        actual = (
            baseline["overall"]["relative_l2"]
            if name == "overall"
            else baseline["partitions"][name]["metrics"]["relative_l2"]
        )
        if abs(actual - expected_relative_l2) > 1e-6:
            raise ValueError(f"{spec.experiment_id} rank-{spec.rank} baseline mismatch for {name}")
    balanced_output = factor_expert(initial_factors, inputs)
    balanced_baseline = {
        "overall": parity(balanced_output, expected),
        "partitions": partition_metrics(balanced_output, expected, positions),
    }
    for name in spec.pw0119_baseline:
        authority_value = (
            baseline["overall"]["relative_l2"]
            if name == "overall"
            else baseline["partitions"][name]["metrics"]["relative_l2"]
        )
        balanced_value = (
            balanced_baseline["overall"]["relative_l2"]
            if name == "overall"
            else balanced_baseline["partitions"][name]["metrics"]["relative_l2"]
        )
        if abs(balanced_value - authority_value) > 5e-6:
            raise ValueError(f"{spec.experiment_id} balanced initialization mismatch for {name}")
    candidate_output = factor_expert(selected_factors, inputs)
    candidate = {
        "overall": parity(candidate_output, expected),
        "partitions": partition_metrics(candidate_output, expected, positions),
    }
    validation_error = candidate["partitions"]["validation"]["metrics"]["relative_l2"]
    holdout_error = candidate["partitions"]["pilot_holdout"]["metrics"]["relative_l2"]
    validation_gate = validation_error <= spec.validation_maximum
    holdout_gate = holdout_error <= spec.holdout_maximum
    safety.checkpoint("complete_expert_evaluation")
    del initial_factors, selected_factors, svd_decompositions, weights, training_inputs, training_targets
    gc.collect()
    torch.mps.synchronize()
    torch.mps.empty_cache()
    final_mps_memory = _mps_memory()
    safety.release_checkpoint(
        "all_factors_and_activations_released",
        ["initial factors", "selected factors", "decoded weights", "activation targets", "MPS cache"],
    )
    safety.checkpoint("final_service_health")
    if final_mps_memory["current_allocated_bytes"] != 0:
        raise ValueError(f"{spec.experiment_id} final MPS allocation did not release")

    passed = validation_gate and holdout_gate
    report = {
        "schema_version": 1,
        "evidence_class": spec.evidence_class,
        "revision": REVISION,
        "commit": commit,
        "checkpoint_verification_sha256": VERIFICATION_SHA256,
        "corpus_manifest_sha256": CORPUS_SHA256,
        "pw0119_source_sha256": PW0119_SOURCE_SHA256,
        "layer": spec.layer,
        "expert": spec.expert,
        "positions": positions,
        "partition_counts": {name: len(rows) for name, rows in partition_indices.items()},
        "configuration": {
            "rank": spec.rank,
            "seed": spec.seed,
            "optimizer": "Adam",
            "learning_rate": LEARNING_RATE,
            "maximum_steps": MAX_STEPS,
            "validate_every": VALIDATE_EVERY,
            "patience_checks": PATIENCE_CHECKS,
            "mps_memory_fraction": MPS_MEMORY_FRACTION,
            "active_projection_parameter_values": P * spec.rank + spec.rank * D,
            "active_projection_semantic_adam_bytes": (P * spec.rank + spec.rank * D) * 4 * 4,
        },
        "source_oracle_parity": source_parity,
        "projection_training_tensor_sha256": training_tensor_hashes,
        "projection_training": projection_reports,
        f"rank{spec.rank}_svd_control": baseline,
        f"balanced_rank{spec.rank}_initialization_control": balanced_baseline,
        "activation_weighted_candidate": candidate,
        "validation_relative_l2_gate": {"maximum": spec.validation_maximum, "passed": validation_gate},
        "pilot_holdout_relative_l2_gate": {"maximum": spec.holdout_maximum, "passed": holdout_gate},
        "gates_passed": passed,
        "final_mps_memory": final_mps_memory,
        "safety_snapshots": safety.evidence(),
        "complete_wall_ms": (time.perf_counter() - complete_started) * 1000.0,
        "batch_size": 1,
        "concurrency": 1,
        "accepted_tokens": 0,
        "A": 0,
        "decision": spec.pass_decision if passed else spec.fail_decision,
        "limitations": spec.limitations,
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
    parser.add_argument("--pw0119", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--commit", required=True)
    arguments = parser.parse_args()
    try:
        result = run(
            arguments.checkpoint,
            arguments.verification,
            arguments.corpus,
            arguments.pw0119,
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
