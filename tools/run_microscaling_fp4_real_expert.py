#!/usr/bin/env python3
"""Run PW-0182's directly executable microscaling-FP4 expert control."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import time

import mlx.core as mx
import numpy as np
import torch

try:
    from tools.generate_real_layer1_expert_oracle import ShardedCheckpoint
    from tools.host_safety import HostSafetyMonitor
    from tools.openrouter_reference import atomic_write_new, canonical_json
    from tools.run_best_rank_real_expert_control import dequant_weight, load_capture, sha256_file, source_linear
    from tools.run_input_subvector_code_capacity_oracle import error_metrics
except ModuleNotFoundError:
    from generate_real_layer1_expert_oracle import ShardedCheckpoint
    from host_safety import HostSafetyMonitor
    from openrouter_reference import atomic_write_new, canonical_json
    from run_best_rank_real_expert_control import dequant_weight, load_capture, sha256_file, source_linear
    from run_input_subvector_code_capacity_oracle import error_metrics


REVISION = "63651580ca774f8504f676040460aed3e1244ac1"
VERIFICATION_SHA256 = "9ddc8a99755f04ae2ea3c2484f6dd022d3f3a681b5a72c915ee4de833dbb0d03"
CORPUS_SHA256 = "b9df976876d63c1ffbbe0c70507aea8b939a749ce5b1db27cbca0b5d82cf802e"
LAYER, EXPERT = 46, 28
HIDDEN, INTERMEDIATE = 4096, 2048
INT4_EXPERT_BYTES = 13_369_344
MODES = {
    "mxfp4": {"mode": "mxfp4", "group_size": 32, "bits": 4},
    "nvfp4": {"mode": "nvfp4", "group_size": 16, "bits": 4},
    "affine4_group32": {"mode": "affine", "group_size": 32, "bits": 4},
}


def quantize_projection(weight: np.ndarray, config: dict) -> tuple[mx.array, mx.array, mx.array | None]:
    value = mx.array(np.asarray(weight, dtype=np.float16))
    result = mx.quantize(value, group_size=config["group_size"], bits=config["bits"], mode=config["mode"])
    mx.eval(*result)
    return result[0], result[1], result[2] if len(result) == 3 else None


def quantized_linear(values: mx.array, arrays: tuple[mx.array, mx.array, mx.array | None], config: dict) -> mx.array:
    weight, scales, biases = arrays
    return mx.quantized_matmul(values, weight, scales=scales, biases=biases, transpose=True, group_size=config["group_size"], bits=config["bits"], mode=config["mode"])


def candidate_expert(values: np.ndarray, projections: dict, config: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = mx.array(np.asarray(values, dtype=np.float16))
    gate = quantized_linear(x, projections["gate"], config)
    up = quantized_linear(x, projections["up"], config)
    hidden = mx.sigmoid(gate) * gate * up
    down = quantized_linear(hidden, projections["down"], config)
    mx.eval(gate, up, down)
    return np.asarray(gate, dtype=np.float32), np.asarray(up, dtype=np.float32), np.asarray(down, dtype=np.float32)


def _source_expert(weights: dict[str, np.ndarray], values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tensor = torch.from_numpy(np.asarray(values, dtype=np.float32).copy()).to(torch.bfloat16)
    gate = source_linear(weights["gate"], tensor); up = source_linear(weights["up"], tensor)
    hidden = (torch.nn.functional.silu(gate) * up).to(torch.bfloat16)
    down = source_linear(weights["down"], hidden)
    return gate.float().numpy(), up.float().numpy(), down.float().numpy()


def _array_record(path: Path, value: mx.array) -> dict:
    array = np.asarray(value)
    array.tofile(path)
    digest = hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()
    return {"file": path.name, "shape": list(array.shape), "dtype": str(array.dtype), "bytes": int(array.nbytes), "sha256": digest}


def run(checkpoint_root: Path, verification_path: Path, corpus_path: Path, artifact_root: Path, output_path: Path) -> dict:
    if output_path.exists() or artifact_root.exists():
        raise ValueError("PW-0182 refuses to overwrite evidence")
    if sha256_file(verification_path) != VERIFICATION_SHA256 or sha256_file(corpus_path) != CORPUS_SHA256:
        raise ValueError("PW-0182 authority hash mismatch")
    safety = HostSafetyMonitor()
    corpus = json.loads(corpus_path.read_text())
    if corpus.get("revision") != REVISION:
        raise ValueError("PW-0182 revision mismatch")
    authority = next(row for row in corpus["layers"] if row["layer"] == LAYER)
    inputs = load_capture(corpus_path.parent, authority["captures"]["moe_input"])
    expected_all = load_capture(corpus_path.parent, authority["captures"]["expert_down"])
    offset = 0
    for schedule in authority["expert_schedule"]:
        if schedule["expert"] == EXPERT: break
        offset += len(schedule["positions"])
    local = [index for index, position in enumerate(schedule["positions"]) if 112 <= position < 168]
    positions = [schedule["positions"][index] for index in local]
    if positions != list(range(112, 168)):
        raise ValueError("PW-0182 validation identity mismatch")
    expected = np.asarray(expected_all[[offset + index for index in local]], dtype=np.float32).copy()
    checkpoint = ShardedCheckpoint(checkpoint_root, verification_path)
    prefix = f"model.layers.{LAYER}.mlp.experts.{EXPERT}"
    source_weights = {name: dequant_weight(checkpoint, f"{prefix}.{name}_proj.weight") for name in ("gate", "up", "down")}
    source = _source_expert(source_weights, np.asarray(inputs[positions]))
    source_control = error_metrics(source[2], expected)
    if source_control["relative_l2"] != 0 or source_control["maximum_row_relative_l2"] != 0:
        raise ValueError("PW-0182 source control failed")
    safety.checkpoint("source_control_complete")
    artifact_root.mkdir(parents=True)
    mode_reports = {}
    for mode_name, config in MODES.items():
        projections = {name: quantize_projection(source_weights[name], config) for name in ("gate", "up", "down")}
        mode_root = artifact_root / mode_name; mode_root.mkdir()
        artifacts, packed_bytes = {}, 0
        reconstruction = {}
        for projection, arrays in projections.items():
            artifacts[projection] = {}
            for label, array in (("weight", arrays[0]), ("scales", arrays[1]), ("biases", arrays[2])):
                if array is None: continue
                record = _array_record(mode_root / f"{projection}.{label}.bin", array)
                artifacts[projection][label] = record; packed_bytes += record["bytes"]
            dequantized = mx.dequantize(arrays[0], arrays[1], biases=arrays[2], group_size=config["group_size"], bits=config["bits"], mode=config["mode"], dtype=mx.float16)
            mx.eval(dequantized); reconstruction[projection] = error_metrics(np.asarray(dequantized, dtype=np.float32), source_weights[projection])["relative_l2"]
        actual = candidate_expert(np.asarray(inputs[positions]), projections, config)
        validation = {"gate": error_metrics(actual[0], source[0]), "up": error_metrics(actual[1], source[1]), "complete_expert": error_metrics(actual[2], expected)}
        sample = np.asarray(inputs[positions[0]:positions[0] + 1])
        for _ in range(10): candidate_expert(sample, projections, config)
        timings = []
        for _ in range(50):
            started = time.perf_counter(); candidate_expert(sample, projections, config); timings.append((time.perf_counter() - started) * 1000)
        physical_pass = packed_bytes <= INT4_EXPERT_BYTES and float(np.median(timings)) <= 0.75
        numerical_pass = validation["complete_expert"]["relative_l2"] <= 0.02 and validation["complete_expert"]["maximum_row_relative_l2"] <= 0.05 and validation["gate"]["relative_l2"] <= 0.02 and validation["up"]["relative_l2"] <= 0.02
        mode_reports[mode_name] = {"config": config, "artifacts": artifacts, "packed_expert_bytes": packed_bytes, "to_affine_int4_ratio": packed_bytes / INT4_EXPERT_BYTES, "weight_reconstruction_relative_l2": reconstruction, "validation": validation, "warm_median_ms": float(np.median(timings)), "warm_p95_ms": float(np.quantile(timings, 0.95)), "gates": {"physical_pass": physical_pass, "numerical_pass": numerical_pass}}
        safety.checkpoint(f"{mode_name}_complete")
        del projections, reconstruction, actual
        gc.collect(); mx.clear_cache()
    passing = [name for name, report in mode_reports.items() if report["gates"]["physical_pass"] and report["gates"]["numerical_pass"]]
    report = {"schema_version": 1, "experiment": "PW-0182", "mode": "L3_shadow_microscaling_fp4", "revision": REVISION, "layer": LAYER, "expert": EXPERT, "pilot_holdout_unsealed": False, "batch_size": 1, "concurrency": 1, "accepted_tokens": 0, "A": 0, "U": 0, "source_control": source_control, "modes": mode_reports, "passing_modes": passing, "decision": "promote_all_validation_and_fused_gather" if passing else "reject_tested_fp4_modes", "hardware": {"machine": platform.machine(), "platform": platform.platform()}, "software": {"mlx": "0.31.2", "python": platform.python_version()}, "implementation": {"commit": subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip(), "dirty": bool(subprocess.run(["git", "status", "--porcelain"], check=True, capture_output=True, text=True).stdout)}}
    del source_weights, source
    gc.collect(); mx.clear_cache(); safety.release_checkpoint("experiment_released", ["source weights", "quantized weights", "validation activations"]); safety.checkpoint("final_service_health")
    report["host_safety"] = [snapshot.to_dict() for snapshot in safety.snapshots]
    atomic_write_new(output_path, canonical_json(report)); return report


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--checkpoint-root", type=Path, required=True); parser.add_argument("--verification", type=Path, required=True); parser.add_argument("--corpus", type=Path, required=True); parser.add_argument("--artifact-root", type=Path, required=True); parser.add_argument("--output", type=Path, required=True); args = parser.parse_args()
    print(json.dumps(run(args.checkpoint_root, args.verification, args.corpus, args.artifact_root, args.output), sort_keys=True))


if __name__ == "__main__": main()
