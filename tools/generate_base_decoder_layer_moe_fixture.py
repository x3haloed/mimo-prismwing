#!/usr/bin/env python3
"""Complete the PW-0049 source-FP8 MoE and final-layer reference artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from safetensors import safe_open
import torch

try:
    from tools.openrouter_reference import atomic_write_new, canonical_json
except ModuleNotFoundError:
    from openrouter_reference import atomic_write_new, canonical_json


REVISION = "63651580ca774f8504f676040460aed3e1244ac1"
LAYER = 43
BATCH = 8
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


def load_f32_artifact(
    root: Path,
    record: dict[str, Any],
    expected_shape: tuple[int, ...],
) -> np.ndarray:
    path = root / record["file"]
    if sha256_file(path) != record.get("sha256"):
        raise ValueError(f"attention artifact hash mismatch: {path.name}")
    values = np.fromfile(path, dtype="<f4")
    if values.size != int(np.prod(expected_shape)) or record.get("shape") != list(expected_shape):
        raise ValueError(f"attention artifact shape mismatch: {path.name}")
    values = values.reshape(expected_shape)
    if not np.isfinite(values).all():
        raise ValueError(f"attention artifact is non-finite: {path.name}")
    return values


def dequantize(values: dict[str, torch.Tensor], name: str) -> torch.Tensor:
    weight = values[name].float()
    scale = values[name + "_scale_inv"].float()
    expected_scale = (weight.shape[0] // 128, weight.shape[1] // 128)
    if weight.ndim != 2 or tuple(scale.shape) != expected_scale:
        raise ValueError(f"{name}: source FP8 layout mismatch")
    return weight * scale.repeat_interleave(128, 0).repeat_interleave(128, 1)


def build_schedule(
    selected: np.ndarray,
    route_weights: np.ndarray,
) -> dict[int, dict[str, list]]:
    if selected.shape != (BATCH, TOP_K) or route_weights.shape != (BATCH, TOP_K):
        raise ValueError("base-layer route shape mismatch")
    if len(set(map(tuple, selected.tolist()))) == 0 or not np.isfinite(route_weights).all():
        raise ValueError("invalid base-layer routes")
    schedule: dict[int, dict[str, list]] = {}
    for position in range(BATCH):
        row = selected[position].tolist()
        if len(set(row)) != TOP_K or any(expert < 0 or expert >= 256 for expert in row):
            raise ValueError("invalid selected expert row")
        if abs(float(route_weights[position].sum()) - 1.0) > 2e-6:
            raise ValueError("route weights are not normalized")
        for slot in range(TOP_K):
            expert = int(selected[position, slot])
            entry = schedule.setdefault(
                expert, {"positions": [], "slots": [], "route_weights": []}
            )
            entry["positions"].append(position)
            entry["slots"].append(slot)
            entry["route_weights"].append(float(route_weights[position, slot]))
    return schedule


def validate_extraction_authority(
    extraction: dict[str, Any], schedule: dict[int, dict[str, list]]
) -> None:
    if (
        extraction.get("schema_version") != 1
        or extraction.get("revision") != REVISION
        or extraction.get("layer") != LAYER
        or extraction.get("experts") != sorted(schedule)
    ):
        raise ValueError("selected-expert extraction identity mismatch")
    source_slices = extraction.get("source_slices")
    if not isinstance(source_slices, list) or not source_slices:
        raise ValueError("selected-expert extraction authority is empty")
    outputs = [item.get("output_file") for item in source_slices]
    if None in outputs or len(set(outputs)) != len(outputs):
        raise ValueError("duplicate selected-expert output file")
    if any(
        item.get("evidence_class")
        != "pinned_local_verified_lossless_tensor_ranges"
        for item in source_slices
    ):
        raise ValueError("expert artifact lacks local verification authority")


def generate(
    attention_manifest_path: Path,
    expert_root: Path,
    extraction_manifest_path: Path,
    moe_expected_path: Path,
    final_expected_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    attention_manifest = json.loads(attention_manifest_path.read_text(encoding="utf-8"))
    if (
        attention_manifest.get("schema_version") != 1
        or attention_manifest.get("semantic")
        != "mimo_base_layer43_source_attention_to_dynamic_routes"
        or attention_manifest.get("revision") != REVISION
        or attention_manifest.get("layer") != LAYER
        or attention_manifest.get("query_count") != BATCH
    ):
        raise ValueError("attention-to-routing manifest identity mismatch")
    artifact_root = attention_manifest_path.parent
    moe_input = load_f32_artifact(
        artifact_root, attention_manifest["artifacts"]["moe_input_f32"], (BATCH, 4096)
    )
    post_attention = load_f32_artifact(
        artifact_root,
        attention_manifest["artifacts"]["post_attention_f32"],
        (BATCH, 4096),
    )
    selected = np.asarray(attention_manifest["selected_experts_by_position"], dtype=np.int64)
    route_weights = np.asarray(attention_manifest["route_weights_by_position"], dtype=np.float32)
    schedule = build_schedule(selected, route_weights)

    extraction = json.loads(extraction_manifest_path.read_text(encoding="utf-8"))
    validate_extraction_authority(extraction, schedule)
    locked_outputs = {item["output_file"]: item for item in extraction["source_slices"]}

    tensor_values: dict[str, torch.Tensor] = {}
    tensor_files: dict[str, str] = {}
    artifact_hashes: dict[str, str] = {}
    for name, authority in sorted(locked_outputs.items()):
        path = expert_root / name
        actual_hash = sha256_file(path)
        if actual_hash != authority.get("output_sha256"):
            raise ValueError(f"selected-expert artifact hash mismatch: {name}")
        artifact_hashes[name] = actual_hash
        with safe_open(path, framework="pt", device="cpu") as tensors:
            for tensor_name in tensors.keys():
                if tensor_name in tensor_values:
                    raise ValueError(f"duplicate selected tensor: {tensor_name}")
                tensor_values[tensor_name] = tensors.get_tensor(tensor_name)
                tensor_files[tensor_name] = name

    inputs = torch.from_numpy(moe_input.copy())
    reference = torch.zeros_like(inputs)
    experts = []
    scalar_errors = []
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
        activated = torch.sigmoid(gate_values) * gate_values * up_values
        output = activated @ down.T
        first = positions[0]
        scalar_errors.extend(
            (
                abs(
                    float(gate_values[0, 0])
                    - float(torch.dot(inputs[first].double(), gate[0].double()))
                ),
                abs(
                    float(up_values[0, 0])
                    - float(torch.dot(inputs[first].double(), up[0].double()))
                ),
                abs(
                    float(output[0, 0])
                    - float(torch.dot(activated[0].double(), down[0].double()))
                ),
            )
        )
        weights = torch.tensor(placement["route_weights"], dtype=torch.float32).unsqueeze(-1)
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
    maximum_scalar_error = max(scalar_errors)
    if maximum_scalar_error > 2e-4:
        raise ValueError(f"expert scalar parity failed: {maximum_scalar_error}")

    moe_output = reference.numpy().astype("<f4")
    final_output = (post_attention + moe_output).astype("<f4")
    if not np.isfinite(moe_output).all() or not np.isfinite(final_output).all():
        raise ValueError("complete base-layer fixture is non-finite")
    moe_bytes = moe_output.tobytes()
    final_bytes = final_output.tobytes()
    write_new(moe_expected_path, moe_bytes)
    write_new(final_expected_path, final_bytes)
    input_record = attention_manifest["artifacts"]["moe_input_f32"]
    manifest = {
        "schema_version": 1,
        "semantic": "mimo_layer43_real_attention_dynamic_source_fp8_moe_block",
        "revision": REVISION,
        "layer": LAYER,
        "batch_size": BATCH,
        "top_k": TOP_K,
        "input_sha256": input_record["sha256"],
        "reference_sha256": hashlib.sha256(moe_bytes).hexdigest(),
        "selected_experts_by_position": selected.tolist(),
        "route_weights_by_position": route_weights.tolist(),
        "real_expert_positions": BATCH * TOP_K,
        "padded_expert_positions": len(experts) * BATCH,
        "experts": experts,
        "artifact_sha256": artifact_hashes,
        "scheduling": "independent_real_attention_source_routes",
        "parent_attention_manifest_sha256": sha256_file(attention_manifest_path),
        "extraction_manifest_sha256": sha256_file(extraction_manifest_path),
        "post_attention_sha256": attention_manifest["artifacts"]["post_attention_f32"]["sha256"],
        "final_reference_sha256": hashlib.sha256(final_bytes).hexdigest(),
        "maximum_projection_scalar_absolute_error": maximum_scalar_error,
    }
    report = {
        "schema_version": 1,
        "semantic": manifest["semantic"],
        "unique_experts": len(experts),
        "expert_position_counts": {
            str(expert): len(placement["positions"])
            for expert, placement in sorted(schedule.items())
        },
        "moe_output_sha256": manifest["reference_sha256"],
        "final_output_sha256": manifest["final_reference_sha256"],
        "maximum_projection_scalar_absolute_error": maximum_scalar_error,
        "moe_output_first8": moe_output.reshape(-1)[:8].tolist(),
        "final_output_first8": final_output.reshape(-1)[:8].tolist(),
        "performance_claim": None,
    }
    return manifest, report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--attention-manifest", required=True, type=Path)
    parser.add_argument("--expert-root", required=True, type=Path)
    parser.add_argument("--extraction-manifest", required=True, type=Path)
    parser.add_argument("--moe-expected", required=True, type=Path)
    parser.add_argument("--final-expected", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    arguments = parser.parse_args()
    try:
        manifest, report = generate(
            arguments.attention_manifest,
            arguments.expert_root,
            arguments.extraction_manifest,
            arguments.moe_expected,
            arguments.final_expected,
        )
        atomic_write_new(arguments.manifest, canonical_json(manifest))
        atomic_write_new(arguments.report, canonical_json(report))
        return 0
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=__import__("sys").stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
