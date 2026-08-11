#!/usr/bin/env python3
"""Run PW-0177's real Core ML scaled-vector expert transaction."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import time

import numpy as np

try:
    from tools.generate_real_layer1_expert_oracle import ShardedCheckpoint
    from tools.openrouter_reference import atomic_write_new, canonical_json
    from tools.run_best_rank_real_expert_control import dequant_weight, load_capture, sha256_file
except ModuleNotFoundError:
    from generate_real_layer1_expert_oracle import ShardedCheckpoint
    from openrouter_reference import atomic_write_new, canonical_json
    from run_best_rank_real_expert_control import dequant_weight, load_capture, sha256_file


REVISION = "63651580ca774f8504f676040460aed3e1244ac1"
VERIFICATION_SHA256 = "9ddc8a99755f04ae2ea3c2484f6dd022d3f3a681b5a72c915ee4de833dbb0d03"
CORPUS_SHA256 = "b9df976876d63c1ffbbe0c70507aea8b939a749ce5b1db27cbca0b5d82cf802e"
LAYER = 46
EXPERT = 28
HIDDEN = 4096
INTERMEDIATE = 2048
SOURCE_EXPERT_BYTES = 3 * HIDDEN * INTERMEDIATE


def normalize_projection(weight: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    value = np.asarray(weight, dtype=np.float32)
    if value.ndim != 2 or not np.isfinite(value).all():
        raise ValueError("PW-0177 projection is invalid")
    scale = np.max(np.abs(value), axis=1)
    scale[scale == 0] = 1.0
    normalized = value / scale[:, None]
    return normalized, scale


def scaled_linear(values: np.ndarray, normalized: np.ndarray, scale: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if normalized.shape[1] != values.shape[1] or scale.shape != (normalized.shape[0],):
        raise ValueError("PW-0177 scaled-linear shape mismatch")
    return (values @ normalized.T) * scale[None, :]


def metrics(actual: np.ndarray, expected: np.ndarray) -> dict:
    actual = np.asarray(actual, dtype=np.float32)
    expected = np.asarray(expected, dtype=np.float32)
    if actual.shape != expected.shape or actual.ndim != 2:
        raise ValueError("PW-0177 metric shape mismatch")
    difference = actual - expected
    row_denominators = np.linalg.norm(expected.astype(np.float64), axis=1)
    row_errors = np.linalg.norm(difference.astype(np.float64), axis=1) / np.maximum(row_denominators, 1e-30)
    return {
        "relative_l2": float(np.linalg.norm(difference.astype(np.float64)) / max(np.linalg.norm(expected.astype(np.float64)), 1e-30)),
        "maximum_absolute_error": float(np.max(np.abs(difference))),
        "maximum_row_relative_l2": float(np.max(row_errors)),
    }


def package_manifest(path: Path) -> dict:
    records = []
    digest = hashlib.sha256()
    for file in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
        relative = file.relative_to(path).as_posix()
        file_hash = sha256_file(file)
        size = file.stat().st_size
        records.append({"path": relative, "bytes": size, "sha256": file_hash})
        digest.update(relative.encode("utf-8"))
        digest.update(bytes.fromhex(file_hash))
        digest.update(size.to_bytes(8, "little"))
    return {"bytes": sum(row["bytes"] for row in records), "sha256": digest.hexdigest(), "files": records}


def _git_identity() -> dict:
    commit = subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain"], check=True, capture_output=True, text=True).stdout)
    return {"commit": commit, "dirty": dirty}


def run(checkpoint_root: Path, verification_path: Path, corpus_path: Path, artifact_root: Path, output_path: Path) -> dict:
    import coremltools as ct
    import torch
    from coremltools.optimize.coreml import OptimizationConfig, OpPalettizerConfig, palettize_weights

    if output_path.exists() or artifact_root.exists():
        raise ValueError("PW-0177 refuses to overwrite output or artifact root")
    if sha256_file(verification_path) != VERIFICATION_SHA256 or sha256_file(corpus_path) != CORPUS_SHA256:
        raise ValueError("PW-0177 authority hash mismatch")
    corpus = json.loads(corpus_path.read_text())
    if corpus.get("revision") != REVISION:
        raise ValueError("PW-0177 revision mismatch")
    authority = next((row for row in corpus["layers"] if row["layer"] == LAYER), None)
    if authority is None:
        raise ValueError("PW-0177 layer authority missing")
    capture_root = corpus_path.parent
    inputs = load_capture(capture_root, authority["captures"]["moe_input"])
    expected_all = load_capture(capture_root, authority["captures"]["expert_down"])
    offset = 0
    schedule = None
    for row in authority["expert_schedule"]:
        if row["expert"] == EXPERT:
            schedule = row
            break
        offset += len(row["positions"])
    if schedule is None:
        raise ValueError("PW-0177 expert schedule missing")
    local = [index for index, position in enumerate(schedule["positions"]) if 112 <= position < 168]
    positions = [schedule["positions"][index] for index in local]
    if positions != list(range(112, 168)):
        raise ValueError("PW-0177 validation placement identity mismatch")
    expected = np.asarray(expected_all[[offset + index for index in local]], dtype=np.float32).copy()

    checkpoint = ShardedCheckpoint(checkpoint_root, verification_path)
    prefix = f"model.layers.{LAYER}.mlp.experts.{EXPERT}"
    weights = {name: dequant_weight(checkpoint, f"{prefix}.{name}_proj.weight") for name in ("gate", "up", "down")}

    class ScaledExpert(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            for name in ("gate", "up", "down"):
                normalized, scale = normalize_projection(weights[name])
                linear = torch.nn.Linear(normalized.shape[1], normalized.shape[0], bias=False, dtype=torch.float16)
                linear.weight.data.copy_(torch.from_numpy(normalized).to(torch.float16))
                setattr(self, name, linear)
                self.register_buffer(name + "_scale", torch.from_numpy(scale).to(torch.float16))

        def forward(self, value):
            gate = self.gate(value) * self.gate_scale
            up = self.up(value) * self.up_scale
            activated = torch.nn.functional.silu(gate) * up
            return self.down(activated) * self.down_scale

    artifact_root.mkdir(parents=True)
    fp16_path = artifact_root / "expert-scaled-fp16.mlpackage"
    candidate_path = artifact_root / "expert-scaled-vector8-dim2-group16.mlpackage"
    example = torch.zeros((1, HIDDEN), dtype=torch.float16)
    started = time.perf_counter()
    control = ct.convert(
        torch.jit.trace(ScaledExpert().eval(), example),
        inputs=[ct.TensorType(name="input", shape=example.shape, dtype=np.float16)],
        outputs=[ct.TensorType(name="output", dtype=np.float16)],
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.macOS15,
        compute_precision=ct.precision.FLOAT16,
    )
    control.save(fp16_path)
    operation_counts = {}
    for function in control.get_spec().mlProgram.functions.values():
        for block in function.block_specializations.values():
            for operation in block.operations:
                operation_counts[operation.type] = operation_counts.get(operation.type, 0) + 1
    if operation_counts.get("linear") != 3 or operation_counts.get("mul", 0) < 4:
        raise ValueError("PW-0177 scaled control graph was folded unexpectedly")
    config = OptimizationConfig(global_config=OpPalettizerConfig(
        mode="kmeans", nbits=8, granularity="per_grouped_channel", group_size=16,
        cluster_dim=2, enable_per_channel_scale=False, num_kmeans_workers=1,
    ))
    compressed = palettize_weights(ct.models.MLModel(str(fp16_path), skip_model_load=True), config)
    compressed.save(candidate_path)

    reports = {}
    for name, path in (("fp16_control", fp16_path), ("vector_candidate", candidate_path)):
        loaded_at = time.perf_counter()
        model = ct.models.MLModel(str(path), compute_units=ct.ComputeUnit.ALL)
        load_ms = (time.perf_counter() - loaded_at) * 1000
        predictions, timings = [], []
        for position in positions:
            predicted_at = time.perf_counter()
            output = model.predict({"input": np.asarray(inputs[position:position + 1], dtype=np.float16)})["output"]
            timings.append((time.perf_counter() - predicted_at) * 1000)
            predictions.append(output[0])
        prediction = np.asarray(predictions, dtype=np.float32)
        warm = timings[8:]
        reports[name] = {
            "package": package_manifest(path),
            "model_load_ms": load_ms,
            "first_prediction_ms": timings[0],
            "warmup_predictions": 7,
            "measured_warm_predictions": len(warm),
            "warm_median_ms": float(np.median(warm)),
            "warm_p95_ms": float(np.quantile(warm, 0.95)),
            "validation": metrics(prediction, expected),
        }
    fp16_bytes = reports["fp16_control"]["package"]["bytes"]
    candidate = reports["vector_candidate"]
    candidate["package_to_fp16_ratio"] = candidate["package"]["bytes"] / fp16_bytes
    candidate["package_to_source_ratio"] = candidate["package"]["bytes"] / SOURCE_EXPERT_BYTES
    numerical_pass = candidate["validation"]["relative_l2"] <= 0.05 and candidate["validation"]["maximum_row_relative_l2"] <= 0.07
    physical_pass = candidate["package_to_fp16_ratio"] <= 0.35 and candidate["warm_median_ms"] <= 2.5 and candidate["warm_p95_ms"] <= 3.5
    topology_pass = candidate["model_load_ms"] + candidate["first_prediction_ms"] <= 2.5
    report = {
        "schema_version": 1,
        "experiment": "PW-0177",
        "mode": "L3_shadow_lossy_candidate",
        "revision": REVISION,
        "layer": LAYER,
        "expert": EXPERT,
        "partition": "validation_112_168",
        "pilot_holdout_unsealed": False,
        "batch_size": 1,
        "concurrency": 1,
        "accepted_tokens": 0,
        "A_expert_executions": 1,
        "U_unique_experts": 1,
        "candidate": {"nbits": 8, "cluster_dim": 2, "group_size": 16, "effective_index_bits_per_weight": 4.0},
        "control_graph_operation_counts": operation_counts,
        "reports": reports,
        "gates": {"numerical_pass": numerical_pass, "component_physical_pass": physical_pass, "per_expert_topology_pass": topology_pass},
        "decision": "promote_resident_layer_transaction" if numerical_pass and physical_pass and topology_pass else "reject_exact_candidate_and_or_topology",
        "hardware": {"machine": platform.machine(), "platform": platform.platform(), "processor": platform.processor()},
        "software": {"coremltools": ct.__version__, "python": platform.python_version()},
        "implementation": _git_identity(),
        "wall_seconds": time.perf_counter() - started,
    }
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
