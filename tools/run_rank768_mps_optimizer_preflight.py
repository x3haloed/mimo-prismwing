#!/usr/bin/env python3
"""Run PW-0120's bounded rank-768 MPS optimizer preflight."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np
import torch
from safetensors import safe_open

try:
    from tools.host_safety import HostSafetyMonitor, HostSafetyViolation
    from tools.openrouter_reference import atomic_write_new, canonical_json
    from tools.run_identity_basis_mps_preflight import IdentityBasis, sha256_file
except ModuleNotFoundError:
    from host_safety import HostSafetyMonitor, HostSafetyViolation
    from openrouter_reference import atomic_write_new, canonical_json
    from run_identity_basis_mps_preflight import IdentityBasis, sha256_file


VERIFICATION_SHA256 = "9ddc8a99755f04ae2ea3c2484f6dd022d3f3a681b5a72c915ee4de833dbb0d03"
REVISION = "63651580ca774f8504f676040460aed3e1244ac1"
SEED = 260120
LAYER = 24
EXPERT = 23
ROWS = 8
COLUMNS = 128
RANK = 768
BASES = 4
ALL_EXPERTS = 256
P = 2048
D = 4096
LEARNING_RATE = 0.001
MPS_MEMORY_FRACTION = 0.60
PARAMETER_VALUES = ALL_EXPERTS * P * RANK + BASES * RANK * D + ALL_EXPERTS * BASES
SEMANTIC_ADAM_BYTES = PARAMETER_VALUES * 4 * 4


def _source_tile(checkpoint_root: Path) -> tuple[torch.Tensor, dict]:
    index_path = checkpoint_root / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())["weight_map"]
    name = f"model.layers.{LAYER}.mlp.experts.{EXPERT}.gate_proj.weight"
    scale_name = f"{name}_scale_inv"
    if index.get(name) != index.get(scale_name):
        raise ValueError("PW-0120 weight and scale shard mismatch")
    shard = checkpoint_root / index[name]
    with safe_open(shard, framework="pt", device="cpu") as source:
        weight = source.get_tensor(name)
        scale = source.get_tensor(scale_name)
    if (
        tuple(weight.shape) != (P, D)
        or weight.dtype != torch.float8_e4m3fn
        or tuple(scale.shape) != (16, 32)
        or scale.dtype != torch.float32
    ):
        raise ValueError("PW-0120 source tensor authority mismatch")
    tile = (weight[:ROWS, :COLUMNS].float() * scale[0, 0]).contiguous()
    return tile, {
        "expert": EXPERT,
        "weight": name,
        "scale": scale_name,
        "shard": shard.name,
        "shard_size": shard.stat().st_size,
    }


def _mps_memory() -> dict:
    return {
        "current_allocated_bytes": int(torch.mps.current_allocated_memory()),
        "driver_allocated_bytes": int(torch.mps.driver_allocated_memory()),
        "recommended_max_bytes": int(torch.mps.recommended_max_memory()),
    }


def run(
    checkpoint_root: Path,
    verification_path: Path,
    output_path: Path,
    commit: str,
) -> dict:
    if output_path.exists():
        raise ValueError(f"refusing to overwrite {output_path}")
    if len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit):
        raise ValueError("implementation commit must be lowercase 40-hex")
    if sha256_file(verification_path) != VERIFICATION_SHA256:
        raise ValueError("PW-0120 checkpoint verification hash mismatch")
    if not torch.backends.mps.is_built() or not torch.backends.mps.is_available():
        raise ValueError("PW-0120 requires an available PyTorch MPS backend")
    if PARAMETER_VALUES != 415_237_120 or SEMANTIC_ADAM_BYTES != 6_643_793_920:
        raise ValueError("PW-0120 production allocation identity mismatch")

    complete_started = time.perf_counter()
    safety = HostSafetyMonitor()
    phase = "initialization"
    phase_wall_ms = {}
    memory_by_phase = {}
    failure = None
    loss_value = None
    maximum_selected_parameter_delta = None
    authority = None
    tile_sha256 = None
    model = optimizer = target = expert_ids = prediction = loss = target_cpu = None

    def observe(name: str, started: float) -> None:
        nonlocal phase
        torch.mps.synchronize()
        phase_wall_ms[name] = (time.perf_counter() - started) * 1000.0
        memory_by_phase[name] = _mps_memory()
        phase = name
        safety.checkpoint(name)

    try:
        torch.mps.set_per_process_memory_fraction(MPS_MEMORY_FRACTION)
        torch.mps.empty_cache()
        started = time.perf_counter()
        target_cpu, authority = _source_tile(checkpoint_root)
        tile_sha256 = hashlib.sha256(
            target_cpu.numpy().astype("<f4", copy=False).tobytes()
        ).hexdigest()
        phase_wall_ms["source_tile_loaded"] = (time.perf_counter() - started) * 1000.0
        phase = "source_tile_loaded"
        safety.checkpoint(phase)

        torch.manual_seed(SEED)
        device = torch.device("mps")
        started = time.perf_counter()
        model = IdentityBasis(ALL_EXPERTS, P, RANK, BASES, D, device=device)
        observe("production_parameters_allocated", started)

        started = time.perf_counter()
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        target = target_cpu.unsqueeze(0).to(device)
        expert_ids = torch.tensor([EXPERT], dtype=torch.long, device=device)
        observe("optimizer_and_source_ready", started)

        before = model.a[EXPERT, :ROWS, :8].detach().cpu().clone()
        started = time.perf_counter()
        prediction = model.tile(expert_ids, ROWS, COLUMNS)
        loss = torch.nn.functional.mse_loss(prediction, target)
        if not bool(torch.isfinite(loss).item()):
            raise ValueError("PW-0120 produced non-finite loss")
        loss_value = float(loss.detach().cpu())
        observe("forward_loss_complete", started)

        started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        observe("backward_gradients_complete", started)

        started = time.perf_counter()
        optimizer.step()
        observe("adam_step_complete", started)
        after = model.a[EXPERT, :ROWS, :8].detach().cpu()
        maximum_selected_parameter_delta = float((after - before).abs().max())
        if not np.isfinite(maximum_selected_parameter_delta) or maximum_selected_parameter_delta <= 0:
            raise ValueError("PW-0120 Adam step did not change selected parameters")
    except (OSError, RuntimeError, ValueError, HostSafetyViolation) as error:
        failure = {"phase": phase, "type": type(error).__name__, "message": str(error)}

    prediction = loss = target = expert_ids = target_cpu = None
    optimizer = model = None
    try:
        torch.mps.synchronize()
        torch.mps.empty_cache()
        memory_by_phase["optimizer_and_parameters_released"] = _mps_memory()
        safety.release_checkpoint(
            "optimizer_and_parameters_released",
            ["production parameters", "dense gradients", "Adam state", "source tile", "MPS cache"],
        )
        safety.checkpoint("final_service_health")
    except (RuntimeError, HostSafetyViolation) as error:
        if failure is None:
            failure = {
                "phase": "optimizer_and_parameters_released",
                "type": type(error).__name__,
                "message": str(error),
            }

    succeeded = failure is None
    report = {
        "schema_version": 1,
        "evidence_class": "pw0120_rank768_mps_optimizer_preflight",
        "revision": REVISION,
        "commit": commit,
        "checkpoint_verification_sha256": VERIFICATION_SHA256,
        "source_tile_sha256": tile_sha256,
        "source_authority": authority,
        "seed": SEED,
        "device": "mps",
        "torch_version": torch.__version__,
        "platform": platform.platform(),
        "configuration": {
            "experts": ALL_EXPERTS,
            "rows": P,
            "rank": RANK,
            "basis_count": BASES,
            "columns": D,
            "tile_layer": LAYER,
            "tile_expert": EXPERT,
            "tile_rows": ROWS,
            "tile_columns": COLUMNS,
            "optimizer": "Adam",
            "learning_rate": LEARNING_RATE,
            "steps": 1,
            "parameter_dtype": "F32",
            "activation": "identity",
            "mps_memory_fraction": MPS_MEMORY_FRACTION,
        },
        "parameter_values": PARAMETER_VALUES,
        "parameter_bytes": PARAMETER_VALUES * 4,
        "semantic_parameter_gradient_adam_bytes": SEMANTIC_ADAM_BYTES,
        "loss": loss_value,
        "maximum_selected_parameter_delta": maximum_selected_parameter_delta,
        "phase_wall_ms": phase_wall_ms,
        "memory_by_phase": memory_by_phase,
        "safety_snapshots": safety.evidence(),
        "complete_wall_ms": (time.perf_counter() - complete_started) * 1000.0,
        "succeeded": succeeded,
        "failure": failure,
        "batch_size": 1,
        "concurrency": 1,
        "accepted_tokens": 0,
        "A": 0,
        "decision": (
            "authorize_rank768_activation_weighted_fitting_contract"
            if succeeded
            else "reject_direct_full_state_rank768_mps_adam"
        ),
        "limitations": "production rank-768 parameter/gradient/Adam embodiment with one fixed source tile and one update only; no convergence, representation fidelity, inference, endpoint, or TPS evidence",
        "performance_claim": None,
    }
    atomic_write_new(output_path, canonical_json(report))
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--verification", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--commit", required=True)
    arguments = parser.parse_args()
    try:
        result = run(
            arguments.checkpoint,
            arguments.verification,
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
