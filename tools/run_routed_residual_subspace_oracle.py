#!/usr/bin/env python3
"""Run PW-0126's routed-residual output-subspace capacity oracle."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import platform
import time

import numpy as np

try:
    from tools.host_safety import HostSafetyMonitor
    from tools.openrouter_reference import atomic_write_new, canonical_json
    from tools.run_best_rank_real_expert_control import load_capture, sha256_file
except ModuleNotFoundError:
    from host_safety import HostSafetyMonitor
    from openrouter_reference import atomic_write_new, canonical_json
    from run_best_rank_real_expert_control import load_capture, sha256_file


REVISION = "63651580ca774f8504f676040460aed3e1244ac1"
CORPUS_SHA256 = "b9df976876d63c1ffbbe0c70507aea8b939a749ce5b1db27cbca0b5d82cf802e"
LAYERS = [4, 24, 46]
RANKS = [16, 32, 64, 96, 111]
WIDTH = 4096
SOURCE_EXPERT_BYTES = 25_171_968
SOURCE_LAYER_BANK_BYTES = 256 * SOURCE_EXPERT_BYTES
SOURCE_MIXTURE_MULTIPLICATIONS = 8 * 3 * 2048 * WIDTH
AGGREGATE_MAXIMUM = 0.01
SLICE_MAXIMUM = 0.02


def relative_l2(actual: np.ndarray, expected: np.ndarray) -> float:
    denominator = float(np.linalg.norm(expected.reshape(-1)))
    return float(np.linalg.norm((actual - expected).reshape(-1))) / max(denominator, 1e-30)


def route_slices(
    selected_experts_by_position: list[list[int]], positions: list[int]
) -> dict[str, list[int]]:
    training_experts = {
        int(expert)
        for position in range(112)
        for expert in selected_experts_by_position[position]
    }
    unseen = [
        position
        for position in positions
        if any(int(expert) not in training_experts for expert in selected_experts_by_position[position])
    ]
    unseen_set = set(unseen)
    return {
        "touches_training_unseen_expert": unseen,
        "all_experts_seen_in_training": [position for position in positions if position not in unseen_set],
    }


def projection(mean: np.ndarray, basis: np.ndarray, values: np.ndarray) -> np.ndarray:
    centered = values - mean
    return mean + (centered @ basis.T) @ basis


def metrics(
    expected: np.ndarray,
    reconstructed: np.ndarray,
    positions: list[int],
    slices: dict[str, list[int]],
) -> dict:
    local = {position: index for index, position in enumerate(positions)}
    result = {"aggregate_relative_l2": relative_l2(reconstructed, expected), "slices": {}}
    for name, members in slices.items():
        indices = [local[position] for position in members]
        result["slices"][name] = {
            "positions": len(indices),
            "relative_l2": (
                relative_l2(reconstructed[indices], expected[indices]) if indices else None
            ),
        }
    return result


def passes(report: dict) -> bool:
    return report["aggregate_relative_l2"] <= AGGREGATE_MAXIMUM and all(
        row["relative_l2"] is None or row["relative_l2"] <= SLICE_MAXIMUM
        for row in report["slices"].values()
    )


def select_rank(validation_by_rank: dict[str, dict]) -> int | None:
    for rank in RANKS:
        if passes(validation_by_rank[str(rank)]):
            return rank
    return None


def physical_ledger(rank: int) -> dict:
    artifact_bytes = 4 * WIDTH * (rank + 1)
    multiplications = rank * WIDTH
    return {
        "rank": rank,
        "f32_mean_and_basis_bytes": artifact_bytes,
        "source_layer_bank_bytes": SOURCE_LAYER_BANK_BYTES,
        "artifact_to_source_bank_ratio": artifact_bytes / SOURCE_LAYER_BANK_BYTES,
        "oracle_output_synthesis_multiplications": multiplications,
        "source_selected_mixture_multiplications": SOURCE_MIXTURE_MULTIPLICATIONS,
        "synthesis_to_source_multiplication_ratio": (
            multiplications / SOURCE_MIXTURE_MULTIPLICATIONS
        ),
        "byte_gate_passed": artifact_bytes / SOURCE_LAYER_BANK_BYTES <= 0.25,
        "multiplication_gate_passed": (
            multiplications / SOURCE_MIXTURE_MULTIPLICATIONS <= 0.25
        ),
        "omitted": (
            "coefficient prediction, routing, quantization, BF16 staging, "
            "artifact loading, and executable wall time"
        ),
    }


def run(corpus_path: Path, output_path: Path, commit: str) -> dict:
    if output_path.exists():
        raise ValueError(f"refusing to overwrite {output_path}")
    if len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit):
        raise ValueError("implementation commit must be lowercase 40-hex")
    if sha256_file(corpus_path) != CORPUS_SHA256:
        raise ValueError("PW-0126 corpus manifest hash mismatch")
    corpus = json.loads(corpus_path.read_text())
    if corpus.get("revision") != REVISION:
        raise ValueError("PW-0126 corpus revision mismatch")

    started = time.perf_counter()
    safety = HostSafetyMonitor()
    root = corpus_path.parent
    layer_reports = []
    for layer_number in LAYERS:
        authority = next(row for row in corpus["layers"] if row["layer"] == layer_number)
        selected = authority["selected_experts_by_position"]
        if len(selected) != 224 or any(len(row) != 8 for row in selected):
            raise ValueError(f"PW-0126 layer {layer_number} route authority mismatch")
        captured = load_capture(root, authority["captures"]["routed_output"])
        if tuple(captured.shape) != (224, WIDTH):
            raise ValueError(f"PW-0126 layer {layer_number} capture shape mismatch")
        values = np.asarray(captured, dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError(f"PW-0126 layer {layer_number} non-finite capture")
        safety.checkpoint(f"layer_{layer_number}_capture_authenticated")

        training = values[:112]
        mean = training.mean(axis=0, keepdims=True)
        centered = training - mean
        svd_started = time.perf_counter()
        _, singular, vt = np.linalg.svd(centered, full_matrices=False)
        svd_wall_ms = (time.perf_counter() - svd_started) * 1000.0
        if not np.isfinite(singular).all() or np.any(np.diff(singular) > 0):
            raise ValueError(f"PW-0126 layer {layer_number} invalid singular spectrum")
        safety.checkpoint(f"layer_{layer_number}_training_svd_complete")

        training_by_rank = {}
        previous = float("inf")
        for rank in RANKS:
            reconstructed = projection(mean, vt[:rank], training)
            error = relative_l2(reconstructed, training)
            if error > previous + 1e-12:
                raise ValueError(f"PW-0126 layer {layer_number} non-monotonic training error")
            training_by_rank[str(rank)] = error
            previous = error
        if training_by_rank["111"] > 1e-10:
            raise ValueError(f"PW-0126 layer {layer_number} full centered rank mismatch")

        validation_positions = list(range(112, 168))
        validation = values[validation_positions]
        validation_slices = route_slices(selected, validation_positions)
        validation_by_rank = {}
        for rank in RANKS:
            reconstructed = projection(mean, vt[:rank], validation)
            validation_by_rank[str(rank)] = metrics(
                validation, reconstructed, validation_positions, validation_slices
            )
        selected_rank = select_rank(validation_by_rank)
        evaluated_rank = selected_rank or RANKS[-1]
        ledger = physical_ledger(evaluated_rank)
        if not ledger["byte_gate_passed"] or not ledger["multiplication_gate_passed"]:
            raise ValueError(f"PW-0126 layer {layer_number} physical gate failed")
        layer_reports.append(
            {
                "layer": layer_number,
                "capture_sha256": authority["captures"]["routed_output"]["sha256"],
                "training_expert_count": len(
                    {expert for position in selected[:112] for expert in position}
                ),
                "singular_values": singular.tolist(),
                "svd_wall_ms": svd_wall_ms,
                "training_relative_l2_by_rank": training_by_rank,
                "validation_by_rank": validation_by_rank,
                "selected_rank": selected_rank,
                "physical_ledger": ledger,
                "holdout": None,
            }
        )
        del values, training, mean, centered, singular, vt
        gc.collect()
        safety.release_checkpoint(
            f"layer_{layer_number}_validation_resources_released",
            [f"layer {layer_number} F64 capture", f"layer {layer_number} SVD factors"],
        )

    validation_passed = all(row["selected_rank"] is not None for row in layer_reports)
    holdout_passed = False
    holdout_unsealed = False
    if validation_passed:
        holdout_unsealed = True
        for report in layer_reports:
            layer_number = report["layer"]
            authority = next(row for row in corpus["layers"] if row["layer"] == layer_number)
            selected = authority["selected_experts_by_position"]
            captured = load_capture(root, authority["captures"]["routed_output"])
            values = np.asarray(captured, dtype=np.float64)
            training = values[:112]
            mean = training.mean(axis=0, keepdims=True)
            _, _, vt = np.linalg.svd(training - mean, full_matrices=False)
            holdout_positions = list(range(168, 224))
            holdout = values[holdout_positions]
            reconstructed = projection(mean, vt[: report["selected_rank"]], holdout)
            report["holdout"] = metrics(
                holdout,
                reconstructed,
                holdout_positions,
                route_slices(selected, holdout_positions),
            )
            del values, training, mean, vt, holdout, reconstructed
            gc.collect()
            safety.release_checkpoint(
                f"layer_{layer_number}_holdout_resources_released",
                [f"layer {layer_number} holdout capture", f"layer {layer_number} SVD factors"],
            )
        holdout_passed = all(passes(row["holdout"]) for row in layer_reports)

    if not validation_passed:
        decision = "reject_fixed_linear_routed_residual_dictionary_on_validation"
    elif not holdout_passed:
        decision = "reject_fixed_linear_routed_residual_dictionary_on_holdout"
    else:
        decision = "authorize_routed_residual_coefficient_predictor_contract"
    gates_passed = validation_passed and holdout_passed
    safety.checkpoint("final_service_health")
    report = {
        "schema_version": 1,
        "evidence_class": "pw0126_routed_residual_subspace_oracle",
        "revision": REVISION,
        "commit": commit,
        "corpus_manifest_sha256": CORPUS_SHA256,
        "configuration": {
            "layers": LAYERS,
            "ranks": RANKS,
            "train_positions": [0, 112],
            "validation_positions": [112, 168],
            "holdout_positions": [168, 224],
            "aggregate_relative_l2_maximum": AGGREGATE_MAXIMUM,
            "nonempty_slice_relative_l2_maximum": SLICE_MAXIMUM,
            "basis_dtype_ledger": "F32",
        },
        "layers": layer_reports,
        "validation_passed": validation_passed,
        "holdout_unsealed": holdout_unsealed,
        "holdout_passed": holdout_passed,
        "gates_passed": gates_passed,
        "decision": decision,
        "safety_snapshots": safety.evidence(),
        "complete_wall_ms": (time.perf_counter() - started) * 1000.0,
        "batch_size": 1,
        "concurrency": 1,
        "accepted_tokens": 0,
        "A": 0,
        "performance_claim": None,
        "limitations": (
            "three layer-local English residual traces; oracle coefficients only, no "
            "coefficient predictor, executable artifact, kernel, multimodal or broad "
            "corpus evidence, accumulated model output, endpoint, or TPS"
        ),
        "platform": platform.platform(),
        "numpy_version": np.__version__,
    }
    atomic_write_new(output_path, canonical_json(report))
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--commit", required=True)
    arguments = parser.parse_args()
    try:
        result = run(arguments.corpus, arguments.output, arguments.commit)
        print(json.dumps({"output": str(arguments.output), "decision": result["decision"]}))
        return 0
    except (OSError, RuntimeError, ValueError, KeyError, TypeError, StopIteration, json.JSONDecodeError) as error:
        print(json.dumps({"error": str(error)}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
