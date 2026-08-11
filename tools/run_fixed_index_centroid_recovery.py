#!/usr/bin/env python3
"""Run PW-0180's fixed-index vector-centroid recovery preflight."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import platform
import subprocess

import numpy as np
import torch

try:
    from tools.generate_real_layer1_expert_oracle import ShardedCheckpoint
    from tools.host_safety import HostSafetyMonitor
    from tools.openrouter_reference import atomic_write_new, canonical_json
    from tools.run_best_rank_real_expert_control import dequant_weight, load_capture, sha256_file, source_linear
    from tools.run_input_subvector_code_capacity_oracle import decode_input_subvectors, error_metrics
except ModuleNotFoundError:
    from generate_real_layer1_expert_oracle import ShardedCheckpoint
    from host_safety import HostSafetyMonitor
    from openrouter_reference import atomic_write_new, canonical_json
    from run_best_rank_real_expert_control import dequant_weight, load_capture, sha256_file, source_linear
    from run_input_subvector_code_capacity_oracle import decode_input_subvectors, error_metrics


REVISION = "63651580ca774f8504f676040460aed3e1244ac1"
VERIFICATION_SHA256 = "9ddc8a99755f04ae2ea3c2484f6dd022d3f3a681b5a72c915ee4de833dbb0d03"
CORPUS_SHA256 = "b9df976876d63c1ffbbe0c70507aea8b939a749ce5b1db27cbca0b5d82cf802e"
PW0178_SHA256 = "1311a8ced8ea4d376229efc9e1508542e5023d41b8f9cec546fcaab3548ac559"
LAYER, EXPERT = 46, 28
HIDDEN, INTERMEDIATE = 4096, 2048
STEPS, LEARNING_RATE, ANCHOR = 64, 5e-4, 1e-5


def torch_decode(indices: torch.Tensor, codebooks: torch.Tensor) -> torch.Tensor:
    if indices.ndim != 2 or codebooks.ndim != 3 or indices.shape[1] != codebooks.shape[0]:
        raise ValueError("PW-0180 code shape mismatch")
    groups = codebooks.shape[0]
    selected = codebooks[torch.arange(groups, device=codebooks.device)[:, None], indices.T]
    return selected.permute(1, 0, 2).reshape(indices.shape[0], -1)


def _digest(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _source_expert(weights: dict[str, np.ndarray], values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tensor = torch.from_numpy(np.asarray(values, dtype=np.float32).copy()).to(torch.bfloat16)
    gate = source_linear(weights["gate"], tensor); up = source_linear(weights["up"], tensor)
    hidden = (torch.nn.functional.silu(gate) * up).to(torch.bfloat16)
    down = source_linear(weights["down"], hidden)
    return gate.float().numpy(), up.float().numpy(), down.float().numpy()


def _load_codes(root: Path, projection: str, output: int, inputs: int, authority: dict) -> tuple[np.ndarray, np.ndarray]:
    ip = root / authority[projection]["indices"]["file"]; cp = root / authority[projection]["codebooks"]["file"]
    for path, record in ((ip, authority[projection]["indices"]), (cp, authority[projection]["codebooks"])):
        if path.stat().st_size != record["bytes"] or sha256_file(path) != record["sha256"]:
            raise ValueError("PW-0180 artifact mismatch")
    return np.fromfile(ip, dtype=np.uint8).reshape(output, inputs // 4), np.fromfile(cp, dtype="<f2").reshape(inputs // 4, 256, 4)


def run(checkpoint_root: Path, verification_path: Path, corpus_path: Path, prior_report_path: Path, prior_artifact_root: Path, artifact_root: Path, output_path: Path) -> dict:
    if output_path.exists() or artifact_root.exists():
        raise ValueError("PW-0180 refuses to overwrite evidence")
    if sha256_file(verification_path) != VERIFICATION_SHA256 or sha256_file(corpus_path) != CORPUS_SHA256 or sha256_file(prior_report_path) != PW0178_SHA256:
        raise ValueError("PW-0180 authority hash mismatch")
    if not torch.backends.mps.is_available():
        raise ValueError("PW-0180 requires onboard MPS")
    safety = HostSafetyMonitor()
    corpus = json.loads(corpus_path.read_text()); prior = json.loads(prior_report_path.read_text())
    if corpus.get("revision") != REVISION or prior.get("decision") != "kill_single_codebook_two_bit_family" or prior.get("pilot_holdout_unsealed"):
        raise ValueError("PW-0180 authority identity mismatch")
    authority = next(row for row in corpus["layers"] if row["layer"] == LAYER)
    inputs = load_capture(corpus_path.parent, authority["captures"]["moe_input"])
    expected_all = load_capture(corpus_path.parent, authority["captures"]["expert_down"])
    offset = 0
    for schedule in authority["expert_schedule"]:
        if schedule["expert"] == EXPERT: break
        offset += len(schedule["positions"])
    train_local = [i for i, p in enumerate(schedule["positions"]) if p < 112]
    val_local = [i for i, p in enumerate(schedule["positions"]) if 112 <= p < 168]
    train_positions = [schedule["positions"][i] for i in train_local]
    val_positions = [schedule["positions"][i] for i in val_local]
    if val_positions != list(range(112, 168)):
        raise ValueError("PW-0180 validation identity mismatch")
    expected_train = np.asarray(expected_all[[offset + i for i in train_local]], dtype=np.float32).copy()
    # Validation targets are intentionally not sliced until optimization is complete.
    shapes = {"gate": (INTERMEDIATE, HIDDEN), "up": (INTERMEDIATE, HIDDEN), "down": (HIDDEN, INTERMEDIATE)}
    codes = {name: _load_codes(prior_artifact_root, name, *shapes[name], prior["artifacts"]) for name in shapes}
    device = torch.device("mps")
    indices = {name: torch.from_numpy(row[0].astype(np.int64)).to(device) for name, row in codes.items()}
    initial = {name: torch.from_numpy(row[1].astype(np.float32)).to(device) for name, row in codes.items()}
    parameters = {name: torch.nn.Parameter(value.clone()) for name, value in initial.items()}
    optimizer = torch.optim.Adam(parameters.values(), lr=LEARNING_RATE, betas=(0.9, 0.999), eps=1e-8)
    x = torch.from_numpy(np.asarray(inputs[train_positions], dtype=np.float32).copy()).to(device)
    target = torch.from_numpy(expected_train).to(device)
    denominator = torch.sum(target * target).clamp_min(1e-30)
    history = []
    for step in range(STEPS + 1):
        optimizer.zero_grad(set_to_none=True)
        weights = {name: torch_decode(indices[name], parameters[name]) for name in shapes}
        gate = x @ weights["gate"].T; up = x @ weights["up"].T
        output = (torch.nn.functional.silu(gate) * up) @ weights["down"].T
        data_loss = torch.sum((output - target) ** 2) / denominator
        anchor = sum(torch.mean((parameters[name] - initial[name]) ** 2) for name in shapes)
        loss = data_loss + ANCHOR * anchor
        if step % 8 == 0:
            torch.mps.synchronize()
            row = {"step": step, "data_loss": float(data_loss.detach().cpu()), "loss": float(loss.detach().cpu()), "mps_current_bytes": int(torch.mps.current_allocated_memory()), "mps_driver_bytes": int(torch.mps.driver_allocated_memory())}
            history.append(row); print(json.dumps(row), flush=True); safety.checkpoint(f"step_{step}")
        if step == STEPS: break
        loss.backward(); optimizer.step()
    final_codebooks = {name: parameters[name].detach().to("cpu").numpy().astype(np.float16) for name in shapes}
    initial_loss, final_loss = history[0]["data_loss"], history[-1]["data_loss"]
    del weights, gate, up, output, data_loss, loss, x, target, optimizer, parameters, initial, indices
    torch.mps.empty_cache(); gc.collect()
    safety.release_checkpoint("mps_training_released", ["MPS optimizer", "dense training weights", "training activations"])

    artifact_root.mkdir(parents=True)
    artifacts, candidate = {}, {}
    for name in shapes:
        path = artifact_root / f"{name}.trained-codebooks.f16"
        final_codebooks[name].astype("<f2", copy=False).tofile(path)
        artifacts[name] = {"file": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path), "initial_sha256": _digest(codes[name][1]), "maximum_absolute_displacement": float(np.max(np.abs(final_codebooks[name].astype(np.float32) - codes[name][1].astype(np.float32))))}
        candidate[name] = decode_input_subvectors(codes[name][0], final_codebooks[name]).astype(np.float32)
    expected_validation = np.asarray(expected_all[[offset + i for i in val_local]], dtype=np.float32).copy()
    source_checkpoint = ShardedCheckpoint(checkpoint_root, verification_path)
    prefix = f"model.layers.{LAYER}.mlp.experts.{EXPERT}"
    source_weights = {name: dequant_weight(source_checkpoint, f"{prefix}.{name}_proj.weight") for name in shapes}
    source = _source_expert(source_weights, np.asarray(inputs[val_positions]))
    source_control = error_metrics(source[2], expected_validation)
    if source_control["relative_l2"] != 0:
        raise ValueError("PW-0180 source control failed")
    actual = _source_expert(candidate, np.asarray(inputs[val_positions]))
    validation = {"gate": error_metrics(actual[0], source[0]), "up": error_metrics(actual[1], source[1]), "complete_expert": error_metrics(actual[2], expected_validation)}
    pass_gate = final_loss <= initial_loss * 0.5 and validation["complete_expert"]["relative_l2"] <= 0.02 and validation["complete_expert"]["maximum_row_relative_l2"] <= 0.05 and validation["gate"]["relative_l2"] <= 0.02 and validation["up"]["relative_l2"] <= 0.02
    decision = "promote_shared_codebook_training" if pass_gate else ("kill_fixed_index_centroid_recovery" if validation["complete_expert"]["relative_l2"] > 0.10 else "retain_broader_training_only")
    report = {"schema_version": 1, "experiment": "PW-0180", "mode": "L3_private_codebook_training_capacity", "revision": REVISION, "layer": LAYER, "expert": EXPERT, "pilot_holdout_unsealed": False, "train_positions": train_positions, "validation_positions": val_positions, "schedule": {"steps": STEPS, "learning_rate": LEARNING_RATE, "anchor": ANCHOR, "optimizer": "Adam"}, "history": history, "train_loss_reduction": 1 - final_loss / initial_loss, "artifacts": artifacts, "source_control": source_control, "validation": validation, "decision": decision, "accepted_tokens": 0, "A": 0, "U": 0, "hardware": {"machine": platform.machine(), "platform": platform.platform()}, "implementation": {"commit": subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip(), "dirty": bool(subprocess.run(["git", "status", "--porcelain"], check=True, capture_output=True, text=True).stdout)}}
    del candidate, source_weights, source, actual, final_codebooks
    gc.collect(); safety.release_checkpoint("validation_released", ["source weights", "candidate weights", "validation activations"]); safety.checkpoint("final_service_health")
    report["host_safety"] = [snapshot.to_dict() for snapshot in safety.snapshots]
    atomic_write_new(output_path, canonical_json(report)); return report


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--checkpoint-root", type=Path, required=True); parser.add_argument("--verification", type=Path, required=True); parser.add_argument("--corpus", type=Path, required=True); parser.add_argument("--pw0178-report", type=Path, required=True); parser.add_argument("--pw0178-artifact-root", type=Path, required=True); parser.add_argument("--artifact-root", type=Path, required=True); parser.add_argument("--output", type=Path, required=True); args = parser.parse_args()
    print(json.dumps(run(args.checkpoint_root, args.verification, args.corpus, args.pw0178_report, args.pw0178_artifact_root, args.artifact_root, args.output), sort_keys=True))


if __name__ == "__main__": main()
