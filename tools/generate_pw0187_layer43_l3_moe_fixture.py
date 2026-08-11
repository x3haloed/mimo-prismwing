#!/usr/bin/env python3
"""Freeze PW-0187 layer-43 routes with a deterministic L3 MoE authority."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from safetensors import safe_open
import torch


PW0187_SHA256 = "a1066fafa979b923f9c2f5d259ff85b2f3d5aa2e77400e8b7075a48f3fa67950"
VERIFICATION_SHA256 = "9ddc8a99755f04ae2ea3c2484f6dd022d3f3a681b5a72c915ee4de833dbb0d03"
REVISION = "63651580ca774f8504f676040460aed3e1244ac1"
LAYER = 43


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def read_json_authority(path: Path, expected_sha256: str) -> dict:
    payload = path.read_bytes()
    if sha256_bytes(payload) != expected_sha256:
        raise ValueError(f"authority SHA-256 mismatch: {path}")
    return json.loads(payload)


def write_new(path: Path, payload: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as output:
        output.write(payload)
        output.flush()
        os.fsync(output.fileno())


def canonical_json(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def dequantize(shard: Path, name: str) -> torch.Tensor:
    with safe_open(shard, framework="pt", device="cpu") as tensors:
        weight = tensors.get_tensor(name).float()
        scale = tensors.get_tensor(name + "_scale_inv").float()
    if weight.ndim != 2 or scale.shape != (weight.shape[0] // 128, weight.shape[1] // 128):
        raise ValueError(f"invalid source-FP8 layout: {name}")
    return weight * scale.repeat_interleave(128, 0).repeat_interleave(128, 1)


def dynamic_input(values: torch.Tensor) -> torch.Tensor:
    rows, columns = values.shape
    grouped = values.float().reshape(rows, columns // 128, 128)
    scales = grouped.abs().amax(-1).clamp(min=1e-10) / 448.0
    encoded = torch.clamp(grouped / scales.unsqueeze(-1), -448.0, 448.0).to(
        torch.float8_e4m3fn
    )
    return (encoded.float() * scales.unsqueeze(-1)).reshape(rows, columns)


def source_projection(values: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return (dynamic_input(values) @ weight.T).to(torch.bfloat16)


def generate(
    checkpoint: Path,
    index_path: Path,
    verification_path: Path,
    prior_path: Path,
    input_path: Path,
    reference_path: Path,
    manifest_path: Path,
    source_bf16: bool,
) -> None:
    prior = read_json_authority(prior_path, PW0187_SHA256)
    verification = read_json_authority(verification_path, VERIFICATION_SHA256)
    index = json.loads(index_path.read_bytes())
    if prior.get("revision") != REVISION or verification.get("revision") != REVISION:
        raise ValueError("revision mismatch")
    traces = prior.get("verification_layer_traces")
    if not isinstance(traces, list) or traces[LAYER].get("layer") != LAYER:
        raise ValueError("PW-0187 layer-43 trace absent")
    trace = traces[LAYER]
    selected = trace["selected_experts_by_position"]
    route_weights = trace["route_weights_by_position"]
    if len(selected) != 8 or any(len(row) != 8 for row in selected):
        raise ValueError("route matrix must be 8x8")
    input_bytes = input_path.read_bytes()
    inputs = torch.from_numpy(np.frombuffer(input_bytes, dtype="<f4").copy().reshape(8, 4096))
    weight_map = index["weight_map"]
    verified = {
        Path(record["path"]).name: record["sha256"]
        for record in verification["files"]
        if record.get("status") == "verified"
    }
    output = torch.zeros((8, 4096), dtype=torch.float32)
    experts = []
    artifact_sha256: dict[str, str] = {}
    for expert in sorted({value for row in selected for value in row}):
        prefix = f"model.layers.{LAYER}.mlp.experts.{expert}"
        tensor_files = {}
        for projection in ("gate", "up", "down"):
            name = f"{prefix}.{projection}_proj.weight"
            shard_name = weight_map[name]
            if weight_map[name + "_scale_inv"] != shard_name or shard_name not in verified:
                raise ValueError(f"unauthenticated projection shard: {name}")
            tensor_files[f"{projection}_weight"] = shard_name
            tensor_files[f"{projection}_scale"] = shard_name
            artifact_sha256[shard_name] = verified[shard_name]
        positions, slots, weights = [], [], []
        for position, row in enumerate(selected):
            for slot, value in enumerate(row):
                if value == expert:
                    positions.append(position)
                    slots.append(slot)
                    weights.append(route_weights[position][slot])
        gathered = inputs[positions]
        gate = dequantize(checkpoint / tensor_files["gate_weight"], f"{prefix}.gate_proj.weight")
        up = dequantize(checkpoint / tensor_files["up_weight"], f"{prefix}.up_proj.weight")
        down = dequantize(checkpoint / tensor_files["down_weight"], f"{prefix}.down_proj.weight")
        if source_bf16:
            gate_values = source_projection(gathered, gate)
            up_values = source_projection(gathered, up)
            hidden = (torch.nn.functional.silu(gate_values) * up_values).to(torch.bfloat16)
            projected = source_projection(hidden, down)
        else:
            gate_values = gathered @ gate.T
            up_values = gathered @ up.T
            hidden = torch.sigmoid(gate_values) * gate_values * up_values
            projected = hidden @ down.T
        for local, position in enumerate(positions):
            output[position] += projected[local] * weights[local]
        experts.append({
            "expert": expert,
            "prefix": prefix,
            "positions": positions,
            "slots": slots,
            "route_weights": weights,
            "tensor_files": tensor_files,
        })
    if source_bf16:
        output = output.to(torch.bfloat16).float()
    reference = output.numpy().astype("<f4").tobytes()
    write_new(reference_path, reference)
    manifest = {
        "schema_version": 1,
        "semantic": (
            "mimo_layer43_pw0187_static_routes_direct_checkpoint_source_bf16_moe_block"
            if source_bf16
            else "mimo_layer43_pw0187_static_routes_direct_checkpoint_l3_moe_block"
        ),
        "revision": REVISION,
        "layer": LAYER,
        "batch_size": 8,
        "top_k": 8,
        "input_sha256": sha256_bytes(input_bytes),
        "reference_sha256": sha256_bytes(reference),
        "selected_experts_by_position": selected,
        "route_weights_by_position": route_weights,
        "real_expert_positions": 64,
        "padded_expert_positions": len(experts) * 8,
        "experts": experts,
        "artifact_sha256": artifact_sha256,
        "scheduling": (
            "pw0187_static_route_replay_source_bf16"
            if source_bf16
            else "pw0187_static_route_replay_l3"
        ),
        "pw0187_manifest_sha256": PW0187_SHA256,
        "checkpoint_verification_sha256": VERIFICATION_SHA256,
    }
    write_new(manifest_path, canonical_json(manifest))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--index", required=True, type=Path)
    parser.add_argument("--verification", required=True, type=Path)
    parser.add_argument("--prior", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--source-bf16", action="store_true")
    args = parser.parse_args()
    generate(
        args.checkpoint,
        args.index,
        args.verification,
        args.prior,
        args.input,
        args.reference,
        args.manifest,
        args.source_bf16,
    )


if __name__ == "__main__":
    main()
