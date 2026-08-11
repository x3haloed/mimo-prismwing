#!/usr/bin/env python3
"""Generate PW-0209's 128-row layer-43 routed-MoE reference directly."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from safetensors import safe_open
import torch

try:
    from tools.checkpoint_lock import validate_verified_install_file
    from tools.openrouter_reference import atomic_write_new, canonical_json
except ModuleNotFoundError:
    from checkpoint_lock import validate_verified_install_file
    from openrouter_reference import atomic_write_new, canonical_json


AUTHORITY_SHA256 = "14ab8792e4ead565ec91d5768737e5c6518bc2a7d2fdd2cae2a3efa93c5126c9"
REVISION = "63651580ca774f8504f676040460aed3e1244ac1"
LAYER = 43
ROWS = 128
TOP_K = 8


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_new(path: Path, payload: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def build_schedule(selected: np.ndarray, weights: np.ndarray) -> dict[int, dict[str, list]]:
    if selected.shape != (ROWS, TOP_K) or weights.shape != (ROWS, TOP_K):
        raise ValueError("PW-0209 route shape mismatch")
    schedule: dict[int, dict[str, list]] = {}
    for position in range(ROWS):
        if len(set(selected[position].tolist())) != TOP_K:
            raise ValueError("PW-0209 duplicate expert in route row")
        if abs(float(weights[position].sum()) - 1.0) > 2.0e-6:
            raise ValueError("PW-0209 route weights are not normalized")
        for slot in range(TOP_K):
            expert = int(selected[position, slot])
            if not 0 <= expert < 256:
                raise ValueError("PW-0209 expert index is out of range")
            entry = schedule.setdefault(expert, {"positions": [], "weights": []})
            entry["positions"].append(position)
            entry["weights"].append(float(weights[position, slot]))
    return schedule


def dequantize(tensors_by_shard: dict, index: dict[str, str], name: str) -> torch.Tensor:
    tensors = tensors_by_shard[index[name]]
    weight = tensors.get_tensor(name).float()
    scale = tensors.get_tensor(name + "_scale_inv").float()
    if weight.ndim != 2 or tuple(scale.shape) != (
        weight.shape[0] // 128,
        weight.shape[1] // 128,
    ):
        raise ValueError(f"{name}: source block-FP8 layout mismatch")
    return weight * scale.repeat_interleave(128, 0).repeat_interleave(128, 1)


def generate(
    checkpoint_dir: Path,
    lock_path: Path,
    verification_path: Path,
    authority_path: Path,
    output_path: Path,
    report_path: Path,
) -> dict:
    if sha256_file(authority_path) != AUTHORITY_SHA256:
        raise ValueError("PW-0209 full-width authority hash mismatch")
    authority = json.loads(authority_path.read_text())
    if (
        authority.get("schema_version") != 1
        or authority.get("semantic")
        != "mimo_pw0209_layer43_context128_full_width_source_authority"
        or authority.get("revision") != REVISION
        or authority.get("layer") != LAYER
        or authority.get("query_count") != ROWS
    ):
        raise ValueError("PW-0209 full-width authority identity mismatch")
    input_record = authority["artifacts"]["moe_input_f32"]
    input_path = authority_path.parent / input_record["file"]
    if sha256_file(input_path) != input_record["sha256"] or input_record["shape"] != [ROWS, 4096]:
        raise ValueError("PW-0209 MoE input artifact mismatch")
    inputs = np.fromfile(input_path, dtype="<f4").reshape(ROWS, 4096)
    selected = np.asarray(authority["selected_experts_by_position"], dtype=np.int64)
    route_weights = np.asarray(authority["route_weights_by_position"], dtype=np.float32)
    schedule = build_schedule(selected, route_weights)

    lock = json.loads(lock_path.read_text())
    verification = json.loads(verification_path.read_text())
    index_path = checkpoint_dir / "model.safetensors.index.json"
    if (
        lock.get("revision") != REVISION
        or verification.get("revision") != REVISION
        or verification.get("complete") is not True
        or verification.get("lock_sha256") != sha256_file(lock_path)
    ):
        raise ValueError("PW-0209 checkpoint authority mismatch")
    locked = {row["path"]: row for row in lock["files"]}
    verified = {row["path"]: row for row in verification["files"]}
    index = json.loads(index_path.read_text())["weight_map"]

    tensor_names = [
        f"model.layers.{LAYER}.mlp.experts.{expert}.{kind}_proj.weight{suffix}"
        for expert in schedule
        for kind in ("gate", "up", "down")
        for suffix in ("", "_scale_inv")
    ]
    shards = sorted({index[name] for name in tensor_names})
    output = torch.zeros((ROWS, 4096), dtype=torch.float32)
    maximum_scalar_error = 0.0
    source_bytes = 0
    for shard in shards:
        if shard not in locked or shard not in verified:
            raise ValueError(f"PW-0209 unverified shard {shard}")
        record = verified[shard]
        if record.get("status") != "verified" or record.get("sha256") != locked[shard]["sha256"]:
            raise ValueError(f"PW-0209 shard verification mismatch {shard}")
        validate_verified_install_file(checkpoint_dir / shard, record)
    with ExitStack() as stack:
        tensors_by_shard = {
            shard: stack.enter_context(
                safe_open(checkpoint_dir / shard, framework="pt", device="cpu")
            )
            for shard in shards
        }
        for expert in sorted(schedule):
            placement = schedule[expert]
            positions = placement["positions"]
            values = torch.from_numpy(inputs[positions].copy())
            prefix = f"model.layers.{LAYER}.mlp.experts.{expert}"
            gate = dequantize(tensors_by_shard, index, f"{prefix}.gate_proj.weight")
            up = dequantize(tensors_by_shard, index, f"{prefix}.up_proj.weight")
            down = dequantize(tensors_by_shard, index, f"{prefix}.down_proj.weight")
            source_bytes += 25_171_968
            gate_values = values @ gate.T
            up_values = values @ up.T
            activated = torch.sigmoid(gate_values) * gate_values * up_values
            projected = activated @ down.T
            maximum_scalar_error = max(
                maximum_scalar_error,
                abs(
                    float(gate_values[0, 0])
                    - float(torch.dot(values[0].double(), gate[0].double()))
                ),
                abs(
                    float(up_values[0, 0])
                    - float(torch.dot(values[0].double(), up[0].double()))
                ),
                abs(
                    float(projected[0, 0])
                    - float(torch.dot(activated[0].double(), down[0].double()))
                ),
            )
            weights = torch.tensor(placement["weights"], dtype=torch.float32).unsqueeze(1)
            output.index_add_(0, torch.tensor(positions), projected * weights)
    if maximum_scalar_error > 2.0e-4:
        raise ValueError(f"PW-0209 scalar parity failed: {maximum_scalar_error}")
    output_values = output.numpy().astype("<f4")
    if not np.isfinite(output_values).all():
        raise ValueError("PW-0209 reference output is non-finite")
    payload = output_values.tobytes()
    write_new(output_path, payload)
    report = {
        "schema_version": 1,
        "evidence_class": "pw0209_layer43_context128_full_width_source_moe_reference",
        "authority_sha256": AUTHORITY_SHA256,
        "output_sha256": hashlib.sha256(payload).hexdigest(),
        "rows": ROWS,
        "top_k": TOP_K,
        "unique_experts": len(schedule),
        "maximum_expert_positions": max(len(row["positions"]) for row in schedule.values()),
        "source_expert_bytes": source_bytes,
        "maximum_projection_scalar_absolute_error": maximum_scalar_error,
        "accepted_tokens": 0,
        "performance_claim": None,
    }
    atomic_write_new(report_path, canonical_json(report))
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", required=True, type=Path)
    parser.add_argument("--lock", required=True, type=Path)
    parser.add_argument("--verification-manifest", required=True, type=Path)
    parser.add_argument("--authority", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    try:
        report = generate(
            args.checkpoint_dir,
            args.lock,
            args.verification_manifest,
            args.authority,
            args.output,
            args.report,
        )
        print(json.dumps(report))
        return 0
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        print(json.dumps({"error": str(error)}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
