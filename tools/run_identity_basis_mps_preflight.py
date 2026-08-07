#!/usr/bin/env python3
"""Run the bounded PW-0118 production-parameter MPS optimizer preflight."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import time

import numpy as np
import torch
from safetensors import safe_open

try:
    from tools.host_safety import HostSafetyMonitor
    from tools.openrouter_reference import atomic_write_new, canonical_json
except ModuleNotFoundError:
    from host_safety import HostSafetyMonitor
    from openrouter_reference import atomic_write_new, canonical_json


VERIFICATION_SHA256 = "9ddc8a99755f04ae2ea3c2484f6dd022d3f3a681b5a72c915ee4de833dbb0d03"
REVISION = "63651580ca774f8504f676040460aed3e1244ac1"
SEED = 260118
EXPERTS = [64, 10]
ROWS = 32
COLUMNS = 128
RANK = 128
BASES = 32
ALL_EXPERTS = 256
P = 2048
D = 4096
STEPS = 5
LEARNING_RATE = 0.01


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


class IdentityBasis(torch.nn.Module):
    def __init__(
        self,
        experts: int,
        rows: int,
        rank: int,
        bases: int,
        columns: int,
        *,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.a = torch.nn.Parameter(
            torch.empty((experts, rows, rank), dtype=torch.float32, device=device)
        )
        self.b = torch.nn.Parameter(
            torch.empty((bases, rank, columns), dtype=torch.float32, device=device)
        )
        self.alpha = torch.nn.Parameter(
            torch.zeros((experts, bases), dtype=torch.float32, device=device)
        )
        with torch.no_grad():
            self.a.normal_(mean=0.0, std=0.005)
            self.b.normal_(mean=0.0, std=0.005)

    def tile(self, expert_ids: torch.Tensor, row_count: int, column_count: int) -> torch.Tensor:
        coefficients = torch.softmax(self.alpha[expert_ids], dim=-1)
        combined = torch.einsum(
            "em,mrc->erc", coefficients, self.b[:, :, :column_count]
        )
        return torch.matmul(self.a[expert_ids, :row_count], combined)


def _source_tiles(checkpoint_root: Path) -> tuple[torch.Tensor, list[dict]]:
    index_path = checkpoint_root / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())["weight_map"]
    tiles = []
    authorities = []
    for expert in EXPERTS:
        name = f"model.layers.4.mlp.experts.{expert}.gate_proj.weight"
        scale_name = f"{name}_scale_inv"
        if index.get(name) != index.get(scale_name):
            raise ValueError("PW-0118 weight and scale shard mismatch")
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
            raise ValueError("PW-0118 source tensor authority mismatch")
        tile = weight[:ROWS, :COLUMNS].float() * scale[0, 0]
        tiles.append(tile)
        authorities.append(
            {
                "expert": expert,
                "weight": name,
                "scale": scale_name,
                "shard": shard.name,
                "shard_size": shard.stat().st_size,
            }
        )
    target = torch.stack(tiles).contiguous()
    return target, authorities


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
        raise ValueError("PW-0118 checkpoint verification hash mismatch")
    if not torch.backends.mps.is_built() or not torch.backends.mps.is_available():
        raise ValueError("PW-0118 requires an available PyTorch MPS backend")
    complete_started = time.perf_counter()
    safety = HostSafetyMonitor()
    target_cpu, authorities = _source_tiles(checkpoint_root)
    target_bytes = target_cpu.numpy().astype("<f4", copy=False).tobytes()
    safety.checkpoint("source_tiles_loaded")
    torch.manual_seed(SEED)
    device = torch.device("mps")
    torch.mps.empty_cache()
    model = IdentityBasis(ALL_EXPERTS, P, RANK, BASES, D, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    target = target_cpu.to(device)
    expert_ids = torch.tensor(EXPERTS, dtype=torch.long, device=device)
    torch.mps.synchronize()
    memory_after_parameters = _mps_memory()
    safety.checkpoint("production_parameters_allocated")
    losses = []
    step_wall_ms = []
    memory_after_steps = []
    for step in range(STEPS):
        started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        prediction = model.tile(expert_ids, ROWS, COLUMNS)
        loss = torch.nn.functional.mse_loss(prediction, target)
        if not bool(torch.isfinite(loss).item()):
            raise ValueError("PW-0118 produced non-finite loss")
        loss.backward()
        optimizer.step()
        torch.mps.synchronize()
        losses.append(float(loss.detach().cpu()))
        step_wall_ms.append((time.perf_counter() - started) * 1000.0)
        memory_after_steps.append(_mps_memory())
        safety.checkpoint(f"optimizer_step_{step + 1}_complete")
    if not all(np.isfinite(losses)) or losses[-1] >= losses[0]:
        raise ValueError("PW-0118 bounded optimizer did not reduce loss")
    del prediction, loss, target, expert_ids, target_cpu, optimizer, model
    torch.mps.synchronize()
    torch.mps.empty_cache()
    release_memory = _mps_memory()
    safety.release_checkpoint(
        "optimizer_and_parameters_released",
        ["production parameters", "Adam state", "source tiles", "MPS cache"],
    )
    report = {
        "schema_version": 1,
        "evidence_class": "pw0118_identity_basis_mps_optimizer_preflight",
        "revision": REVISION,
        "commit": commit,
        "checkpoint_verification_sha256": VERIFICATION_SHA256,
        "source_tile_sha256": hashlib.sha256(target_bytes).hexdigest(),
        "source_authorities": authorities,
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
            "tile_experts": EXPERTS,
            "tile_rows": ROWS,
            "tile_columns": COLUMNS,
            "optimizer": "Adam",
            "learning_rate": LEARNING_RATE,
            "steps": STEPS,
            "parameter_dtype": "F32",
            "activation": "identity",
        },
        "parameter_values": ALL_EXPERTS * P * RANK + BASES * RANK * D + ALL_EXPERTS * BASES,
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "losses": losses,
        "step_wall_ms": step_wall_ms,
        "memory_after_parameters": memory_after_parameters,
        "memory_after_steps": memory_after_steps,
        "memory_after_release": release_memory,
        "safety_snapshots": safety.evidence(),
        "complete_wall_ms": (time.perf_counter() - complete_started) * 1000.0,
        "batch_size": len(EXPERTS),
        "concurrency": 1,
        "accepted_tokens": 0,
        "A": 0,
        "decision": "authorize_streamed_identity_basis_weight_fitting_contract",
        "limitations": "production parameter/Adam embodiment with a fixed real source tile only; not converged layer training or fidelity evidence",
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
