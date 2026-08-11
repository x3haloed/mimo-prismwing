#!/usr/bin/env python3
"""Run PW-0178's activation-metric input-subvector capacity oracle."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import time

import numpy as np
import torch
from sklearn.cluster import MiniBatchKMeans

try:
    from tools.generate_real_layer1_expert_oracle import ShardedCheckpoint
    from tools.host_safety import HostSafetyMonitor
    from tools.openrouter_reference import atomic_write_new, canonical_json
    from tools.run_best_rank_real_expert_control import dequant_weight, load_capture, sha256_file, source_linear
except ModuleNotFoundError:
    from generate_real_layer1_expert_oracle import ShardedCheckpoint
    from host_safety import HostSafetyMonitor
    from openrouter_reference import atomic_write_new, canonical_json
    from run_best_rank_real_expert_control import dequant_weight, load_capture, sha256_file, source_linear


REVISION = "63651580ca774f8504f676040460aed3e1244ac1"
VERIFICATION_SHA256 = "9ddc8a99755f04ae2ea3c2484f6dd022d3f3a681b5a72c915ee4de833dbb0d03"
CORPUS_SHA256 = "b9df976876d63c1ffbbe0c70507aea8b939a749ce5b1db27cbca0b5d82cf802e"
LAYER, EXPERT = 46, 28
VECTOR_DIM, CENTROIDS = 4, 256
HIDDEN, INTERMEDIATE = 4096, 2048
EXPERT_CODE_BYTES = (2 * INTERMEDIATE * HIDDEN + HIDDEN * INTERMEDIATE) // VECTOR_DIM
LAYER_CODEBOOK_BYTES = ((2 * HIDDEN // VECTOR_DIM) + (INTERMEDIATE // VECTOR_DIM)) * CENTROIDS * VECTOR_DIM * 2


def decode_input_subvectors(indices: np.ndarray, codebooks: np.ndarray) -> np.ndarray:
    indices = np.asarray(indices)
    codebooks = np.asarray(codebooks)
    if indices.ndim != 2 or indices.dtype != np.uint8 or codebooks.ndim != 3:
        raise ValueError("PW-0178 code shape or dtype mismatch")
    groups, centroids, vector_dim = codebooks.shape
    if indices.shape[1] != groups or centroids > 256:
        raise ValueError("PW-0178 code shape mismatch")
    result = np.empty((indices.shape[0], groups * vector_dim), dtype=codebooks.dtype)
    rows = np.arange(indices.shape[0])
    for group in range(groups):
        result[:, group * vector_dim:(group + 1) * vector_dim] = codebooks[group, indices[:, group]]
    return result


def error_metrics(actual: np.ndarray, expected: np.ndarray) -> dict:
    actual = np.asarray(actual, dtype=np.float32)
    expected = np.asarray(expected, dtype=np.float32)
    difference = actual - expected
    row_error = np.linalg.norm(difference.astype(np.float64), axis=1) / np.maximum(np.linalg.norm(expected.astype(np.float64), axis=1), 1e-30)
    return {
        "relative_l2": float(np.linalg.norm(difference.astype(np.float64)) / max(np.linalg.norm(expected.astype(np.float64)), 1e-30)),
        "maximum_row_relative_l2": float(np.max(row_error)),
        "maximum_absolute_error": float(np.max(np.abs(difference))),
    }


def fit_projection(weight: np.ndarray, calibration: np.ndarray, seed_offset: int, progress_name: str) -> tuple[np.ndarray, np.ndarray, dict]:
    weight = np.asarray(weight, dtype=np.float32)
    calibration = np.asarray(calibration, dtype=np.float32)
    if weight.ndim != 2 or calibration.ndim != 2 or weight.shape[1] != calibration.shape[1] or weight.shape[1] % VECTOR_DIM:
        raise ValueError("PW-0178 projection shape mismatch")
    groups = weight.shape[1] // VECTOR_DIM
    indices = np.empty((weight.shape[0], groups), dtype=np.uint8)
    codebooks = np.empty((groups, CENTROIDS, VECTOR_DIM), dtype=np.float16)
    reconstructed = np.empty_like(weight)
    started = time.perf_counter()
    for group in range(groups):
        span = slice(group * VECTOR_DIM, (group + 1) * VECTOR_DIM)
        values = weight[:, span].astype(np.float64)
        activations = calibration[:, span].astype(np.float64)
        covariance = activations.T @ activations / max(activations.shape[0], 1)
        ridge = max(float(np.trace(covariance)) * 1e-6 / VECTOR_DIM, 1e-8)
        transform = np.linalg.cholesky(covariance + np.eye(VECTOR_DIM) * ridge)
        transformed = values @ transform
        fit = MiniBatchKMeans(
            n_clusters=CENTROIDS, batch_size=min(4096, weight.shape[0]), max_iter=50,
            n_init=1, random_state=seed_offset + group, reassignment_ratio=0.0,
        ).fit(transformed)
        centers = fit.cluster_centers_ @ np.linalg.inv(transform)
        labels = fit.labels_.astype(np.uint8)
        stored = centers.astype(np.float16)
        indices[:, group] = labels
        codebooks[group] = stored
        reconstructed[:, span] = stored[labels].astype(np.float32)
        if (group + 1) % 128 == 0 or group + 1 == groups:
            print(json.dumps({"phase": progress_name, "groups_complete": group + 1, "groups_total": groups}), flush=True)
    difference = reconstructed.astype(np.float64) - weight.astype(np.float64)
    report = {
        "groups": groups,
        "seconds": time.perf_counter() - started,
        "weight_relative_l2": float(np.linalg.norm(difference) / max(np.linalg.norm(weight.astype(np.float64)), 1e-30)),
        "index_bytes": int(indices.nbytes),
        "codebook_f16_bytes": int(codebooks.nbytes),
    }
    return indices, codebooks, report


def source_expert(weights: dict[str, np.ndarray], values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    tensor = torch.from_numpy(np.asarray(values, dtype=np.float32).copy()).to(torch.bfloat16)
    gate = source_linear(weights["gate"], tensor)
    up = source_linear(weights["up"], tensor)
    hidden = (torch.nn.functional.silu(gate) * up).to(torch.bfloat16)
    down = source_linear(weights["down"], hidden)
    return gate.float().numpy(), up.float().numpy(), hidden.float().numpy(), down.float().numpy()


def artifact_record(path: Path) -> dict:
    return {"file": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def run(checkpoint_root: Path, verification_path: Path, corpus_path: Path, artifact_root: Path, output_path: Path) -> dict:
    if output_path.exists() or artifact_root.exists():
        raise ValueError("PW-0178 refuses to overwrite evidence")
    if sha256_file(verification_path) != VERIFICATION_SHA256 or sha256_file(corpus_path) != CORPUS_SHA256:
        raise ValueError("PW-0178 authority hash mismatch")
    safety = HostSafetyMonitor()
    corpus = json.loads(corpus_path.read_text())
    if corpus.get("revision") != REVISION:
        raise ValueError("PW-0178 revision mismatch")
    authority = next(row for row in corpus["layers"] if row["layer"] == LAYER)
    inputs = load_capture(corpus_path.parent, authority["captures"]["moe_input"])
    expected_all = load_capture(corpus_path.parent, authority["captures"]["expert_down"])
    offset = 0
    for schedule in authority["expert_schedule"]:
        if schedule["expert"] == EXPERT:
            break
        offset += len(schedule["positions"])
    train_local = [i for i, p in enumerate(schedule["positions"]) if p < 112]
    validation_local = [i for i, p in enumerate(schedule["positions"]) if 112 <= p < 168]
    train_positions = [schedule["positions"][i] for i in train_local]
    validation_positions = [schedule["positions"][i] for i in validation_local]
    if validation_positions != list(range(112, 168)):
        raise ValueError("PW-0178 validation identity mismatch")
    expected_validation = np.asarray(expected_all[[offset + i for i in validation_local]], dtype=np.float32).copy()
    checkpoint = ShardedCheckpoint(checkpoint_root, verification_path)
    prefix = f"model.layers.{LAYER}.mlp.experts.{EXPERT}"
    weights = {name: dequant_weight(checkpoint, f"{prefix}.{name}_proj.weight") for name in ("gate", "up", "down")}
    source_train = source_expert(weights, np.asarray(inputs[train_positions]))
    source_validation = source_expert(weights, np.asarray(inputs[validation_positions]))
    source_capture_metrics = error_metrics(source_validation[3], expected_validation)
    if source_capture_metrics["relative_l2"] > 0.001 or source_capture_metrics["maximum_row_relative_l2"] > 0.002:
        raise ValueError("PW-0178 source control failed")
    safety.checkpoint("source_controls_complete")

    artifact_root.mkdir(parents=True)
    reconstructed = {}
    fits, artifacts = {}, {}
    calibrations = {"gate": np.asarray(inputs[train_positions]), "up": np.asarray(inputs[train_positions]), "down": source_train[2]}
    for projection, seed in (("gate", 178000), ("up", 179000), ("down", 180000)):
        indices, codebooks, fit = fit_projection(weights[projection], calibrations[projection], seed, projection)
        index_path = artifact_root / f"{projection}.indices.u8"
        codebook_path = artifact_root / f"{projection}.codebooks.f16"
        indices.tofile(index_path)
        codebooks.astype("<f2", copy=False).tofile(codebook_path)
        artifacts[projection] = {"indices": artifact_record(index_path), "codebooks": artifact_record(codebook_path)}
        fits[projection] = fit
        reconstructed[projection] = decode_input_subvectors(indices, codebooks).astype(np.float32)
        safety.checkpoint(f"{projection}_fit_complete")

    candidate_train = source_expert(reconstructed, np.asarray(inputs[train_positions]))
    candidate_validation = source_expert(reconstructed, np.asarray(inputs[validation_positions]))
    validation = {
        "gate": error_metrics(candidate_validation[0], source_validation[0]),
        "up": error_metrics(candidate_validation[1], source_validation[1]),
        "complete_expert": error_metrics(candidate_validation[3], expected_validation),
    }
    training = {"complete_expert": error_metrics(candidate_train[3], source_train[3])}
    index_bytes = sum(row["index_bytes"] for row in fits.values())
    private_codebook_bytes = sum(row["codebook_f16_bytes"] for row in fits.values())
    if index_bytes != EXPERT_CODE_BYTES or private_codebook_bytes != LAYER_CODEBOOK_BYTES:
        raise ValueError("PW-0178 physical ledger mismatch")
    physical = {
        "expert_index_bytes": index_bytes,
        "shared_layer_codebook_f16_bytes": private_codebook_bytes,
        "code_bytes_per_token_for_376_executions": index_bytes * 376,
        "resident_codebook_bytes_all_47_layers": private_codebook_bytes * 47,
        "gate_up_shared_centroid_table_macs_per_layer": 2 * (HIDDEN // VECTOR_DIM) * CENTROIDS * VECTOR_DIM,
        "down_centroid_table_macs_per_layer_top8": 8 * (INTERMEDIATE // VECTOR_DIM) * CENTROIDS * VECTOR_DIM,
        "index_accumulations_per_layer_top8": 8 * (2 * INTERMEDIATE * (HIDDEN // VECTOR_DIM) + HIDDEN * (INTERMEDIATE // VECTOR_DIM)),
    }
    numerical_pass = validation["complete_expert"]["relative_l2"] <= 0.02 and validation["complete_expert"]["maximum_row_relative_l2"] <= 0.05 and validation["gate"]["relative_l2"] <= 0.02 and validation["up"]["relative_l2"] <= 0.02
    report = {
        "schema_version": 1, "experiment": "PW-0178", "mode": "L3_favorable_private_codebook_oracle",
        "revision": REVISION, "layer": LAYER, "expert": EXPERT, "pilot_holdout_unsealed": False,
        "train_positions": train_positions, "validation_positions": validation_positions,
        "batch_size": 1, "concurrency": 1, "accepted_tokens": 0, "A": 1, "U": 1,
        "fit": fits, "artifacts": artifacts, "source_capture_control": source_capture_metrics,
        "training": training, "validation": validation, "physical": physical,
        "gates": {"numerical_pass": numerical_pass, "physical_ledger_pass": True},
        "decision": "promote_shared_layer_codebook_fit" if numerical_pass else ("kill_single_codebook_two_bit_family" if validation["complete_expert"]["relative_l2"] > 0.10 else "retain_only_trained_assignment_repair"),
        "hardware": {"machine": platform.machine(), "platform": platform.platform()},
        "software": {"python": platform.python_version(), "numpy": np.__version__},
        "implementation": {"commit": subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip(), "dirty": bool(subprocess.run(["git", "status", "--porcelain"], check=True, capture_output=True, text=True).stdout)},
    }
    del weights, reconstructed, source_train, source_validation, candidate_train, candidate_validation
    gc.collect()
    safety.release_checkpoint("candidate_released", ["source weights", "reconstructed weights", "candidate activations"])
    safety.checkpoint("final_service_health")
    report["host_safety"] = [snapshot.to_dict() for snapshot in safety.snapshots]
    atomic_write_new(output_path, canonical_json(report))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--verification", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.checkpoint_root, args.verification, args.corpus, args.artifact_root, args.output), sort_keys=True))


if __name__ == "__main__":
    main()
