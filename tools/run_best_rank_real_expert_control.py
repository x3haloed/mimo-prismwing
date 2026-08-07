#!/usr/bin/env python3
"""Run PW-0119's streamed best-rank real-expert activation control."""

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
except ModuleNotFoundError:
    from generate_real_layer1_expert_oracle import ShardedCheckpoint, dynamic_input
    from host_safety import HostSafetyMonitor
    from openrouter_reference import atomic_write_new, canonical_json


REVISION = "63651580ca774f8504f676040460aed3e1244ac1"
VERIFICATION_SHA256 = "9ddc8a99755f04ae2ea3c2484f6dd022d3f3a681b5a72c915ee4de833dbb0d03"
CORPUS_SHA256 = "b9df976876d63c1ffbbe0c70507aea8b939a749ce5b1db27cbca0b5d82cf802e"
SAMPLES = [(4, 64, "hot"), (4, 10, "rare"), (24, 23, "hot"), (24, 101, "rare"), (46, 28, "hot"), (46, 0, "rare")]
RANKS = [128, 512, 768]
P, D = 2048, 4096


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_capture(root: Path, record: dict) -> np.ndarray:
    path = root / record["file"]
    if (
        record.get("dtype") != "BF16_widened_F32"
        or path.stat().st_size != record.get("bytes")
        or sha256_file(path) != record.get("sha256")
    ):
        raise ValueError("PW-0119 capture authority mismatch")
    values = np.memmap(path, dtype="<f4", mode="r")
    expected = int(np.prod(record["shape"]))
    if values.size != expected:
        raise ValueError("PW-0119 capture shape mismatch")
    return values.reshape(record["shape"])


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


def source_linear(weight: np.ndarray, values: torch.Tensor) -> torch.Tensor:
    quantized = dynamic_input(values)
    return (quantized @ torch.from_numpy(weight).T).to(torch.bfloat16)


def svd_control(
    decomposition: tuple[np.ndarray, np.ndarray, np.ndarray],
    values: torch.Tensor,
    rank: int,
    *,
    down: bool,
) -> torch.Tensor:
    u, singular, vt = decomposition
    quantized = dynamic_input(values).float()
    u_rank = torch.from_numpy(u[:, :rank])
    singular_rank = torch.from_numpy(singular[:rank])
    vt_rank = torch.from_numpy(vt[:rank])
    if down:
        output = ((quantized @ u_rank) * singular_rank) @ vt_rank
    else:
        output = ((quantized @ vt_rank.T) * singular_rank) @ u_rank.T
    return output.to(torch.bfloat16)


def parity(actual: torch.Tensor, expected: torch.Tensor) -> dict:
    actual_f32 = actual.float()
    expected_f32 = expected.float()
    difference = actual_f32 - expected_f32
    denominator = float(torch.linalg.vector_norm(expected_f32))
    return {
        "relative_l2": float(torch.linalg.vector_norm(difference)) / max(denominator, 1e-30),
        "maximum_absolute_error": float(difference.abs().max()),
        "equality_fraction": float((actual_f32 == expected_f32).float().mean()),
    }


def partition_metrics(
    actual: torch.Tensor, expected: torch.Tensor, positions: list[int]
) -> dict:
    partitions = [("train", 0, 112), ("validation", 112, 168), ("pilot_holdout", 168, 224)]
    result = {}
    for name, start, end in partitions:
        indices = [local for local, position in enumerate(positions) if start <= position < end]
        result[name] = {
            "positions": len(indices),
            "metrics": parity(actual[indices], expected[indices]) if indices else None,
        }
    return result


def run(
    checkpoint_root: Path,
    verification_path: Path,
    corpus_manifest_path: Path,
    output_path: Path,
    commit: str,
) -> dict:
    if output_path.exists():
        raise ValueError(f"refusing to overwrite {output_path}")
    if len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit):
        raise ValueError("implementation commit must be lowercase 40-hex")
    if sha256_file(verification_path) != VERIFICATION_SHA256:
        raise ValueError("PW-0119 checkpoint verification hash mismatch")
    if sha256_file(corpus_manifest_path) != CORPUS_SHA256:
        raise ValueError("PW-0119 corpus manifest hash mismatch")
    corpus = json.loads(corpus_manifest_path.read_text())
    if corpus.get("revision") != REVISION or corpus.get("target_layers") != [4, 24, 46]:
        raise ValueError("PW-0119 corpus identity mismatch")
    layer_authority = {row["layer"]: row for row in corpus["layers"]}
    root = corpus_manifest_path.parent
    checkpoint = ShardedCheckpoint(checkpoint_root, verification_path)
    safety = HostSafetyMonitor()
    complete_started = time.perf_counter()
    reports = []
    for layer, expert, frequency_class in SAMPLES:
        authority = layer_authority[layer]
        moe_input = load_capture(root, authority["captures"]["moe_input"])
        expert_down = load_capture(root, authority["captures"]["expert_down"])
        offset = 0
        positions = None
        for schedule in authority["expert_schedule"]:
            if schedule["expert"] == expert:
                positions = schedule["positions"]
                break
            offset += len(schedule["positions"])
        if positions is None or len(positions) != authority["expert_access_counts"][str(expert)]:
            raise ValueError("PW-0119 expert schedule identity mismatch")
        expected = torch.from_numpy(
            np.asarray(expert_down[offset : offset + len(positions)]).copy()
        ).to(torch.bfloat16)
        inputs = torch.from_numpy(np.asarray(moe_input[positions]).copy()).to(torch.bfloat16)
        prefix = f"model.layers.{layer}.mlp.experts.{expert}"
        matrices = {}
        decompositions = {}
        projection_reports = {}
        for projection in ["gate", "up", "down"]:
            weight = dequant_weight(checkpoint, f"{prefix}.{projection}_proj.weight")
            canonical = np.ascontiguousarray(weight.T if projection == "down" else weight)
            if canonical.shape != (P, D):
                raise ValueError("PW-0119 canonical projection shape mismatch")
            started = time.perf_counter()
            u, singular, vt = np.linalg.svd(canonical, full_matrices=False)
            wall_ms = (time.perf_counter() - started) * 1000.0
            if (
                not np.isfinite(singular).all()
                or np.any(singular[:-1] < singular[1:])
                or singular.size != P
            ):
                raise ValueError("PW-0119 singular spectrum mismatch")
            total_energy = float(np.dot(singular, singular))
            residuals = {
                str(rank): float(np.sqrt(np.dot(singular[rank:], singular[rank:]) / total_energy))
                for rank in RANKS
            }
            if not (residuals["128"] >= residuals["512"] >= residuals["768"]):
                raise ValueError("PW-0119 rank residual monotonicity mismatch")
            matrices[projection] = weight
            decompositions[projection] = (u, singular, vt)
            projection_reports[projection] = {
                "svd_wall_ms": wall_ms,
                "relative_frobenius_residual_by_rank": residuals,
                "largest_singular_value": float(singular[0]),
                "smallest_singular_value": float(singular[-1]),
            }
            safety.checkpoint(f"layer_{layer}_expert_{expert}_{projection}_spectrum_complete")
        gate = source_linear(matrices["gate"], inputs)
        up = source_linear(matrices["up"], inputs)
        activated = (torch.nn.functional.silu(gate) * up).to(torch.bfloat16)
        source_output = source_linear(matrices["down"], activated)
        source_parity = parity(source_output, expected)
        if source_parity["relative_l2"] > 1e-3 or source_parity["maximum_absolute_error"] > 0.02:
            raise ValueError("PW-0119 source oracle parity gate failed")
        rank_reports = {}
        for rank in RANKS:
            candidate_gate = svd_control(decompositions["gate"], inputs, rank, down=False)
            candidate_up = svd_control(decompositions["up"], inputs, rank, down=False)
            candidate_activated = (
                torch.nn.functional.silu(candidate_gate) * candidate_up
            ).to(torch.bfloat16)
            candidate_output = svd_control(
                decompositions["down"], candidate_activated, rank, down=True
            )
            rank_reports[str(rank)] = {
                "overall": parity(candidate_output, expected),
                "partitions": partition_metrics(candidate_output, expected, positions),
            }
        reports.append(
            {
                "layer": layer,
                "expert": expert,
                "frequency_class": frequency_class,
                "placements": len(positions),
                "positions": positions,
                "source_oracle_parity": source_parity,
                "projections": projection_reports,
                "rank_controls": rank_reports,
            }
        )
        del matrices, decompositions, gate, up, activated, source_output, inputs, expected
        gc.collect()
        safety.release_checkpoint(
            f"layer_{layer}_expert_{expert}_released",
            ["decoded weights", "SVD factors", "activation rows"],
        )
    safety.checkpoint("final_service_health")
    report = {
        "schema_version": 1,
        "evidence_class": "pw0119_best_rank_real_expert_activation_control",
        "revision": REVISION,
        "commit": commit,
        "checkpoint_verification_sha256": VERIFICATION_SHA256,
        "corpus_manifest_sha256": CORPUS_SHA256,
        "samples": reports,
        "ranks": RANKS,
        "safety_snapshots": safety.evidence(),
        "complete_wall_ms": (time.perf_counter() - complete_started) * 1000.0,
        "batch_size": 1,
        "concurrency": 1,
        "accepted_tokens": 0,
        "A": 0,
        "decision": "establish_matched_best_rank_control_for_identity_basis_fitting",
        "limitations": "six expert activation control; SVD minimizes matrix Frobenius rather than activation error; no shared fit, mixture residual, kernel, endpoint, or TPS",
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
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--commit", required=True)
    arguments = parser.parse_args()
    try:
        result = run(
            arguments.checkpoint,
            arguments.verification,
            arguments.corpus,
            arguments.output,
            arguments.commit,
        )
        print(json.dumps({"output": str(arguments.output), "decision": result["decision"]}))
        return 0
    except (OSError, RuntimeError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
        print(json.dumps({"error": str(error)}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
