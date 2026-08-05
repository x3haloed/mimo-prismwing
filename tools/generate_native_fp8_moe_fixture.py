#!/usr/bin/env python3
"""Freeze the PW-0016 heterogeneous routes and complete source-FP8 MoE oracle."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from safetensors import safe_open
import torch

try:
    from tools.openrouter_reference import atomic_write_new, canonical_json
except ModuleNotFoundError:
    from openrouter_reference import atomic_write_new, canonical_json


LAYER = 43
BATCH = 8
TOP_K = 8
REVISION = "63651580ca774f8504f676040460aed3e1244ac1"


def digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


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


def dequantize(values: dict[str, torch.Tensor], name: str) -> torch.Tensor:
    weight = values[name].float()
    scale = values[name + "_scale_inv"].float()
    expected_scale = (weight.shape[0] // 128, weight.shape[1] // 128)
    if weight.ndim != 2 or tuple(scale.shape) != expected_scale:
        raise ValueError(f"{name}: source FP8 layout mismatch")
    return weight * scale.repeat_interleave(128, 0).repeat_interleave(128, 1)


def generate(
    router_path: Path,
    expert_root: Path,
    extraction_manifest_path: Path,
    input_path: Path,
    expected_path: Path,
) -> tuple[dict, dict]:
    extraction = json.loads(extraction_manifest_path.read_text(encoding="utf-8"))
    if (
        extraction.get("schema_version") != 1
        or extraction.get("revision") != REVISION
        or extraction.get("layer") != LAYER
    ):
        raise ValueError("selected-expert extraction identity mismatch")
    locked_outputs = {
        item["output_file"]: item for item in extraction["source_slices"]
    }
    if len(locked_outputs) != len(extraction["source_slices"]):
        raise ValueError("duplicate selected-expert output file")

    tensor_values: dict[str, torch.Tensor] = {}
    tensor_files: dict[str, str] = {}
    artifact_hashes: dict[str, str] = {}
    for name, authority in sorted(locked_outputs.items()):
        path = expert_root / name
        actual_hash = sha256_file(path)
        if actual_hash != authority["output_sha256"]:
            raise ValueError(f"selected-expert artifact hash mismatch: {name}")
        artifact_hashes[name] = actual_hash
        with safe_open(path, framework="pt", device="cpu") as tensors:
            for tensor_name in tensors.keys():
                if tensor_name in tensor_values:
                    raise ValueError(f"duplicate selected tensor: {tensor_name}")
                tensor_values[tensor_name] = tensors.get_tensor(tensor_name)
                tensor_files[tensor_name] = name

    router_weight_name = f"model.layers.{LAYER}.mlp.gate.weight"
    correction_name = f"model.layers.{LAYER}.mlp.gate.e_score_correction_bias"
    with safe_open(router_path, framework="pt", device="cpu") as tensors:
        router_weight = tensors.get_tensor(router_weight_name).float()
        correction = tensors.get_tensor(correction_name).float()
    inputs_f16 = np.array(
        [
            [np.sin((column + 19 * row) / 17.0) * 0.01 for column in range(4096)]
            for row in range(BATCH)
        ],
        dtype=np.float16,
    )
    inputs = torch.from_numpy(inputs_f16).float()
    scores = torch.sigmoid(inputs @ router_weight.T)
    indices = torch.topk(scores + correction, TOP_K, dim=-1, sorted=False).indices
    selected_scores = scores.gather(1, indices)
    route_weights = selected_scores / (selected_scores.sum(dim=-1, keepdim=True) + 1e-20)

    schedule: dict[int, dict[str, list]] = {}
    for position in range(BATCH):
        for slot in range(TOP_K):
            expert = int(indices[position, slot])
            entry = schedule.setdefault(
                expert,
                {"positions": [], "slots": [], "route_weights": []},
            )
            entry["positions"].append(position)
            entry["slots"].append(slot)
            entry["route_weights"].append(float(route_weights[position, slot]))
    counts = sorted(len(entry["positions"]) for entry in schedule.values())
    if counts != [3, 5, 8, 8, 8, 8, 8, 8, 8]:
        raise ValueError(f"unexpected PW-0016 route counts: {counts}")

    reference = torch.zeros_like(inputs)
    experts = []
    for expert, placement in sorted(schedule.items()):
        prefix = f"model.layers.{LAYER}.mlp.experts.{expert}"
        names = {kind: f"{prefix}.{kind}_proj.weight" for kind in ("gate", "up", "down")}
        for name in names.values():
            if name not in tensor_values or name + "_scale_inv" not in tensor_values:
                raise ValueError(f"missing selected expert tensor: {name}")
        gate = dequantize(tensor_values, names["gate"])
        up = dequantize(tensor_values, names["up"])
        down = dequantize(tensor_values, names["down"])
        positions = placement["positions"]
        values = inputs[positions]
        gate_values = values @ gate.T
        up_values = values @ up.T
        output = (torch.sigmoid(gate_values) * gate_values * up_values) @ down.T
        weights = torch.tensor(placement["route_weights"]).unsqueeze(-1)
        reference.index_add_(0, torch.tensor(positions), output * weights)
        experts.append(
            {
                "expert": expert,
                "prefix": prefix,
                "positions": positions,
                "slots": placement["slots"],
                "route_weights": placement["route_weights"],
                "tensor_files": {
                    f"{kind}_{part}": tensor_files[
                        name if part == "weight" else name + "_scale_inv"
                    ]
                    for kind, name in names.items()
                    for part in ("weight", "scale")
                },
            }
        )

    input_f32 = inputs_f16.astype("<f4")
    expected = reference.numpy().astype("<f4")
    if not np.isfinite(input_f32).all() or not np.isfinite(expected).all():
        raise ValueError("heterogeneous MoE fixture is non-finite")
    input_bytes = input_f32.tobytes()
    expected_bytes = expected.tobytes()
    write_new(input_path, input_bytes)
    write_new(expected_path, expected_bytes)
    manifest = {
        "schema_version": 1,
        "semantic": "mimo_layer43_fixture_scheduled_source_fp8_moe_block",
        "revision": REVISION,
        "layer": LAYER,
        "batch_size": BATCH,
        "top_k": TOP_K,
        "input_sha256": digest(input_bytes),
        "reference_sha256": digest(expected_bytes),
        "selected_experts_by_position": indices.tolist(),
        "route_weights_by_position": route_weights.tolist(),
        "real_expert_positions": BATCH * TOP_K,
        "padded_expert_positions": len(experts) * BATCH,
        "experts": experts,
        "artifact_sha256": artifact_hashes,
        "scheduling": "fixture_static_source_routes",
    }
    report = {
        "schema_version": 1,
        "semantic": manifest["semantic"],
        "input_f32_count": int(input_f32.size),
        "expected_f32_count": int(expected.size),
        "input_sha256": manifest["input_sha256"],
        "expected_sha256": manifest["reference_sha256"],
        "expected_first8": expected.reshape(-1)[:8].tolist(),
        "selected_experts": [entry["expert"] for entry in experts],
        "expert_position_counts": {
            str(entry["expert"]): len(entry["positions"]) for entry in experts
        },
        "artifact_count": len(artifact_hashes),
    }
    return manifest, report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--router", required=True, type=Path)
    parser.add_argument("--expert-root", required=True, type=Path)
    parser.add_argument("--extraction-manifest", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--expected", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    manifest, report = generate(
        args.router,
        args.expert_root,
        args.extraction_manifest,
        args.input,
        args.expected,
    )
    atomic_write_new(args.manifest, canonical_json(manifest))
    atomic_write_new(args.report, canonical_json(report))


if __name__ == "__main__":
    main()
