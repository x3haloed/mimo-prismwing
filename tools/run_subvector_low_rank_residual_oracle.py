#!/usr/bin/env python3
"""Run PW-0179's weight-domain low-rank residual capacity oracle."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import platform
import subprocess

import numpy as np
import torch
from sklearn.utils.extmath import randomized_svd

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
RANKS = (16, 32, 64, 96, 128)
CORE_BYTES = 6_291_456
INT4_EXPERT_BYTES = 13_369_344
SOURCE_MACS = 3 * HIDDEN * INTERMEDIATE


def add_low_rank(core: np.ndarray, left: np.ndarray, right: np.ndarray) -> np.ndarray:
    core, left, right = (np.asarray(value, dtype=np.float32) for value in (core, left, right))
    if core.ndim != 2 or left.shape[0] != core.shape[0] or right.shape[1] != core.shape[1] or left.shape[1] != right.shape[0]:
        raise ValueError("PW-0179 low-rank shape mismatch")
    return core + left @ right


def _source_expert(weights: dict[str, np.ndarray], values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tensor = torch.from_numpy(np.asarray(values, dtype=np.float32).copy()).to(torch.bfloat16)
    gate = source_linear(weights["gate"], tensor)
    up = source_linear(weights["up"], tensor)
    hidden = (torch.nn.functional.silu(gate) * up).to(torch.bfloat16)
    down = source_linear(weights["down"], hidden)
    return gate.float().numpy(), up.float().numpy(), down.float().numpy()


def _load_core(root: Path, projection: str, output: int, inputs: int, authority: dict) -> np.ndarray:
    index_path = root / authority[projection]["indices"]["file"]
    codebook_path = root / authority[projection]["codebooks"]["file"]
    for path, record in ((index_path, authority[projection]["indices"]), (codebook_path, authority[projection]["codebooks"])):
        if path.stat().st_size != record["bytes"] or sha256_file(path) != record["sha256"]:
            raise ValueError("PW-0179 PW-0178 artifact mismatch")
    indices = np.fromfile(index_path, dtype=np.uint8).reshape(output, inputs // 4)
    codebooks = np.fromfile(codebook_path, dtype="<f2").reshape(inputs // 4, 256, 4)
    return decode_input_subvectors(indices, codebooks).astype(np.float32)


def run(checkpoint_root: Path, verification_path: Path, corpus_path: Path, pw0178_report_path: Path, pw0178_artifact_root: Path, artifact_root: Path, output_path: Path) -> dict:
    if output_path.exists() or artifact_root.exists():
        raise ValueError("PW-0179 refuses to overwrite evidence")
    if sha256_file(verification_path) != VERIFICATION_SHA256 or sha256_file(corpus_path) != CORPUS_SHA256 or sha256_file(pw0178_report_path) != PW0178_SHA256:
        raise ValueError("PW-0179 authority hash mismatch")
    safety = HostSafetyMonitor()
    corpus = json.loads(corpus_path.read_text())
    prior = json.loads(pw0178_report_path.read_text())
    if corpus.get("revision") != REVISION or prior.get("decision") != "kill_single_codebook_two_bit_family" or prior.get("pilot_holdout_unsealed"):
        raise ValueError("PW-0179 authority identity mismatch")
    authority = next(row for row in corpus["layers"] if row["layer"] == LAYER)
    inputs = load_capture(corpus_path.parent, authority["captures"]["moe_input"])
    expected_all = load_capture(corpus_path.parent, authority["captures"]["expert_down"])
    offset = 0
    for schedule in authority["expert_schedule"]:
        if schedule["expert"] == EXPERT:
            break
        offset += len(schedule["positions"])
    local = [i for i, position in enumerate(schedule["positions"]) if 112 <= position < 168]
    positions = [schedule["positions"][i] for i in local]
    if positions != list(range(112, 168)):
        raise ValueError("PW-0179 validation identity mismatch")
    expected = np.asarray(expected_all[[offset + i for i in local]], dtype=np.float32).copy()
    checkpoint = ShardedCheckpoint(checkpoint_root, verification_path)
    prefix = f"model.layers.{LAYER}.mlp.experts.{EXPERT}"
    weights = {name: dequant_weight(checkpoint, f"{prefix}.{name}_proj.weight") for name in ("gate", "up", "down")}
    cores = {
        "gate": _load_core(pw0178_artifact_root, "gate", INTERMEDIATE, HIDDEN, prior["artifacts"]),
        "up": _load_core(pw0178_artifact_root, "up", INTERMEDIATE, HIDDEN, prior["artifacts"]),
        "down": _load_core(pw0178_artifact_root, "down", HIDDEN, INTERMEDIATE, prior["artifacts"]),
    }
    source = _source_expert(weights, np.asarray(inputs[positions]))
    source_control = error_metrics(source[2], expected)
    if source_control["relative_l2"] != 0 or source_control["maximum_row_relative_l2"] != 0:
        raise ValueError("PW-0179 source control failed")
    artifact_root.mkdir(parents=True)
    factors, factor_artifacts = {}, {}
    for projection, seed in (("gate", 1791), ("up", 1792), ("down", 1793)):
        residual = weights[projection].astype(np.float32) - cores[projection]
        u, singular, vt = randomized_svd(residual, n_components=128, n_iter=4, random_state=seed, flip_sign=True)
        left = (u * singular[None, :]).astype(np.float16)
        right = vt.astype(np.float16)
        left_path, right_path = artifact_root / f"{projection}.left-r128.f16", artifact_root / f"{projection}.right-r128.f16"
        left.astype("<f2", copy=False).tofile(left_path); right.astype("<f2", copy=False).tofile(right_path)
        factors[projection] = (left, right)
        factor_artifacts[projection] = {
            "left": {"file": left_path.name, "bytes": left_path.stat().st_size, "sha256": sha256_file(left_path)},
            "right": {"file": right_path.name, "bytes": right_path.stat().st_size, "sha256": sha256_file(right_path)},
            "rank128_residual_energy_captured": float(np.sum(singular.astype(np.float64) ** 2) / max(np.linalg.norm(residual.astype(np.float64)) ** 2, 1e-30)),
        }
        safety.checkpoint(f"{projection}_rank128_complete")
    rank_reports = {}
    for rank in RANKS:
        candidate = {name: add_low_rank(cores[name], factors[name][0][:, :rank], factors[name][1][:rank]) for name in ("gate", "up", "down")}
        actual = _source_expert(candidate, np.asarray(inputs[positions]))
        factor_bytes = 3 * 2 * rank * (HIDDEN + INTERMEDIATE)
        factor_macs = 3 * rank * (HIDDEN + INTERMEDIATE)
        rank_reports[str(rank)] = {
            "gate": error_metrics(actual[0], source[0]), "up": error_metrics(actual[1], source[1]),
            "complete_expert": error_metrics(actual[2], expected),
            "factor_bytes_per_expert": factor_bytes, "combined_core_factor_bytes_per_expert": CORE_BYTES + factor_bytes,
            "combined_to_int4_ratio": (CORE_BYTES + factor_bytes) / INT4_EXPERT_BYTES,
            "factor_macs_per_expert": factor_macs, "factor_to_source_mac_ratio": factor_macs / SOURCE_MACS,
        }
        del candidate, actual
        gc.collect()
        safety.checkpoint(f"rank_{rank}_evaluation_complete")
    passing = [rank for rank in RANKS if rank <= 96 and rank_reports[str(rank)]["complete_expert"]["relative_l2"] <= 0.02 and rank_reports[str(rank)]["complete_expert"]["maximum_row_relative_l2"] <= 0.05 and rank_reports[str(rank)]["gate"]["relative_l2"] <= 0.02 and rank_reports[str(rank)]["up"]["relative_l2"] <= 0.02 and rank_reports[str(rank)]["combined_to_int4_ratio"] <= 0.75 and rank_reports[str(rank)]["factor_to_source_mac_ratio"] <= 0.08]
    rank128_error = rank_reports["128"]["complete_expert"]["relative_l2"]
    decision = "promote_packed_residual_kernel" if passing else ("kill_low_rank_residual_on_two_bit_core" if rank128_error > 0.05 else "retain_non_low_rank_trained_residual_only")
    report = {
        "schema_version": 1, "experiment": "PW-0179", "mode": "L3_favorable_weight_residual_oracle",
        "revision": REVISION, "layer": LAYER, "expert": EXPERT, "pilot_holdout_unsealed": False,
        "batch_size": 1, "concurrency": 1, "accepted_tokens": 0, "A": 1, "U": 1,
        "source_control": source_control, "factor_artifacts": factor_artifacts, "ranks": rank_reports,
        "passing_ranks": passing, "decision": decision,
        "hardware": {"machine": platform.machine(), "platform": platform.platform()},
        "software": {"python": platform.python_version(), "numpy": np.__version__},
        "implementation": {"commit": subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip(), "dirty": bool(subprocess.run(["git", "status", "--porcelain"], check=True, capture_output=True, text=True).stdout)},
    }
    del weights, cores, factors, source
    gc.collect()
    safety.release_checkpoint("residual_oracle_released", ["source weights", "code weights", "rank factors", "validation activations"])
    safety.checkpoint("final_service_health")
    report["host_safety"] = [snapshot.to_dict() for snapshot in safety.snapshots]
    atomic_write_new(output_path, canonical_json(report))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-root", type=Path, required=True); parser.add_argument("--verification", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True); parser.add_argument("--pw0178-report", type=Path, required=True)
    parser.add_argument("--pw0178-artifact-root", type=Path, required=True); parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True); args = parser.parse_args()
    print(json.dumps(run(args.checkpoint_root, args.verification, args.corpus, args.pw0178_report, args.pw0178_artifact_root, args.artifact_root, args.output), sort_keys=True))


if __name__ == "__main__":
    main()
