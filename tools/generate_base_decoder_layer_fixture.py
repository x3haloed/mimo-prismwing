#!/usr/bin/env python3
"""Generate the PW-0049 learned attention-to-routing base-layer fixture."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import struct
from typing import Any

import mlx.core as mx
import numpy as np
from safetensors import safe_open
import torch

try:
    from tools.generate_mtp_decoder_block_fixture import install_scales, project, rms_norm
    from tools.openrouter_reference import atomic_write_new, canonical_json
except ModuleNotFoundError:
    from generate_mtp_decoder_block_fixture import install_scales, project, rms_norm
    from openrouter_reference import atomic_write_new, canonical_json


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_bytes(values: np.ndarray) -> bytes:
    return np.asarray(values, dtype="<f4").tobytes()


def array_sha256(values: np.ndarray) -> str:
    return hashlib.sha256(array_bytes(values)).hexdigest()


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


def rope(values: np.ndarray, position: int, rope_dim: int, theta: int) -> np.ndarray:
    if values.ndim != 2 or rope_dim <= 0 or rope_dim % 2 or rope_dim > values.shape[1]:
        raise ValueError("invalid partial RoPE shape")
    result = values.astype(np.float32, copy=True)
    half = rope_dim // 2
    pair = np.arange(half, dtype=np.float32)
    angle = np.float32(position) / np.power(np.float32(theta), 2 * pair / rope_dim)
    cosine = np.cos(angle).astype(np.float32)
    sine = np.sin(angle).astype(np.float32)
    first = result[:, :half].copy()
    second = result[:, half:rope_dim].copy()
    result[:, :half] = first * cosine - second * sine
    result[:, half:rope_dim] = second * cosine + first * sine
    return result


def attention_query(
    query: np.ndarray,
    keys: np.ndarray,
    values: np.ndarray,
    sinks: np.ndarray,
) -> np.ndarray:
    if (
        query.ndim != 2
        or keys.ndim != 3
        or values.ndim != 3
        or query.shape != (64, 192)
        or keys.shape[1:] != (8, 192)
        or values.shape != (keys.shape[0], 8, 128)
        or sinks.shape != (64,)
    ):
        raise ValueError("invalid MiMo SWA attention shape")
    outputs = np.empty((64, 128), dtype=np.float32)
    scale = np.float32(1 / math.sqrt(192))
    for head in range(64):
        kv_head = head // 8
        scores = (keys[:, kv_head] @ query[head]) * scale
        logits = np.concatenate((scores, sinks[head : head + 1]))
        probabilities = np.exp(logits - logits.max()).astype(np.float32)
        probabilities /= probabilities.sum(dtype=np.float32)
        outputs[head] = probabilities[:-1] @ values[:, kv_head]
    return outputs


def noaux_tc_route(
    hidden: np.ndarray,
    router_weight: np.ndarray,
    correction: np.ndarray,
    top_k: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    if (
        hidden.ndim != 2
        or router_weight.shape != (256, hidden.shape[1])
        or correction.shape != (256,)
        or top_k != 8
    ):
        raise ValueError("invalid noaux-tc routing shape")
    hidden_tensor = torch.from_numpy(np.asarray(hidden, dtype=np.float32))
    weight_tensor = torch.from_numpy(np.asarray(router_weight, dtype=np.float32))
    correction_tensor = torch.from_numpy(np.asarray(correction, dtype=np.float32))
    scores = torch.sigmoid(hidden_tensor @ weight_tensor.T)
    corrected = scores + correction_tensor
    selected = torch.topk(corrected, top_k, dim=-1, sorted=False).indices
    selected_scores = scores.gather(1, selected)
    weights = selected_scores / (selected_scores.sum(dim=-1, keepdim=True) + 1e-20)
    ordered = torch.sort(corrected, dim=-1, descending=True).values
    margin = float(torch.min(ordered[:, top_k - 1] - ordered[:, top_k]))
    return selected.numpy(), weights.numpy(), margin


def require_verified_source(
    lock_path: Path,
    verification_path: Path,
    checkpoint_dir: Path,
    relative_path: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    verification = json.loads(verification_path.read_text(encoding="utf-8"))
    if (
        verification.get("schema_version") != 1
        or verification.get("evidence_class") != "local_checkpoint_lock_verification"
        or not verification.get("complete")
        or verification.get("lock_sha256") != sha256_file(lock_path)
        or verification.get("repository") != lock.get("repository")
        or verification.get("revision") != lock.get("revision")
    ):
        raise ValueError("checkpoint verification manifest identity mismatch")
    locked = {item["path"]: item for item in lock.get("files", [])}.get(relative_path)
    observed = {item["path"]: item for item in verification.get("files", [])}.get(relative_path)
    if not isinstance(locked, dict) or not isinstance(observed, dict):
        raise ValueError(f"source is absent from lock verification: {relative_path}")
    if (
        observed.get("status") != "verified"
        or observed.get("bytes") != locked.get("bytes")
        or observed.get("sha256") != locked.get("sha256")
    ):
        raise ValueError(f"source did not pass the lock: {relative_path}")
    stat = (checkpoint_dir / relative_path).stat()
    if (
        stat.st_size != observed.get("bytes")
        or stat.st_dev != observed.get("device")
        or stat.st_ino != observed.get("inode")
        or stat.st_mtime_ns != observed.get("modified_ns")
    ):
        raise ValueError(f"source identity changed after verification: {relative_path}")
    return locked, observed


def read_header(path: Path) -> tuple[int, dict[str, Any]]:
    with path.open("rb") as source:
        header_bytes = struct.unpack("<Q", source.read(8))[0]
        if header_bytes <= 0 or header_bytes > 256 * 1024 * 1024:
            raise ValueError("invalid safetensors header size")
        header = json.loads(source.read(header_bytes))
    if not isinstance(header, dict):
        raise ValueError("safetensors header is not an object")
    return header_bytes, header


def generate(
    checkpoint_dir: Path,
    lock_path: Path,
    verification_path: Path,
    semantic_fixture_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    fixture = json.loads(semantic_fixture_path.read_text(encoding="utf-8"))
    if (
        fixture.get("schema_version") != 1
        or fixture.get("semantic") != "mimo_base_layer43_source_attention_to_dynamic_routes"
        or fixture.get("layer") != 43
        or fixture.get("context") != 128
        or fixture.get("query_start") != 120
        or fixture.get("query_count") != 8
    ):
        raise ValueError("unknown base-layer semantic fixture")
    source_name = fixture["source_file"]
    source_lock, _ = require_verified_source(
        lock_path, verification_path, checkpoint_dir, source_name
    )
    model_lock, _ = require_verified_source(
        lock_path, verification_path, checkpoint_dir, fixture["model_source"]
    )
    if (
        source_lock["sha256"] != fixture["source_file_sha256"]
        or model_lock["sha256"] != fixture["model_source_sha256"]
    ):
        raise ValueError("base-layer source fixture hash mismatch")
    source_path = checkpoint_dir / source_name
    _, header = read_header(source_path)
    for name, metadata in fixture["tensors"].items():
        if header.get(name) != metadata:
            raise ValueError(f"base-layer tensor metadata mismatch: {name}")

    layer = fixture["layer"]
    prefix = f"model.layers.{layer}"
    names = {
        "input_norm": f"{prefix}.input_layernorm.weight",
        "post_norm": f"{prefix}.post_attention_layernorm.weight",
        "qkv": f"{prefix}.self_attn.qkv_proj.weight",
        "qkv_scale": f"{prefix}.self_attn.qkv_proj.weight_scale_inv",
        "sink": f"{prefix}.self_attn.attention_sink_bias",
        "out": f"{prefix}.self_attn.o_proj.weight",
        "router": f"{prefix}.mlp.gate.weight",
        "correction": f"{prefix}.mlp.gate.e_score_correction_bias",
    }
    with safe_open(source_path, framework="pt", device="cpu") as tensors:
        input_norm = tensors.get_tensor(names["input_norm"]).float().numpy()
        post_norm = tensors.get_tensor(names["post_norm"]).float().numpy()
        raw_qkv = tensors.get_tensor(names["qkv"]).float().numpy()
        qkv_scales = tensors.get_tensor(names["qkv_scale"]).float().numpy()
        sinks = tensors.get_tensor(names["sink"]).float().numpy()
        output_weight = tensors.get_tensor(names["out"]).float().numpy()
        router_weight = tensors.get_tensor(names["router"]).float().numpy()
        correction = tensors.get_tensor(names["correction"]).float().numpy()

    parameters = fixture["parameters"]
    rng = np.random.default_rng(fixture["seed"])
    hidden = rng.standard_normal(
        (fixture["context"], parameters["hidden_size"]), dtype=np.float32
    )
    normalized = rms_norm(hidden, input_norm)
    qkv_weight = install_scales(raw_qkv, qkv_scales)
    qkv = project(qkv_weight, normalized)
    sample_rows = (0, 1, 12288, 13824)
    scalar = np.asarray(
        [
            np.dot(normalized[0].astype(np.float64), qkv_weight[row].astype(np.float64))
            for row in sample_rows
        ]
    )
    sample_error = float(np.max(np.abs(qkv[0, list(sample_rows)].astype(np.float64) - scalar)))
    if sample_error > 2e-4:
        raise ValueError(f"QKV scalar parity failed: {sample_error}")

    q = qkv[:, :12288].reshape(128, 64, 192)
    k = qkv[:, 12288:13824].reshape(128, 8, 192)
    v = qkv[:, 13824:].reshape(128, 8, 128) * np.float32(parameters["value_scale"])
    for position in range(fixture["context"]):
        q[position] = rope(q[position], position, parameters["rope_dim"], parameters["rope_theta"])
        k[position] = rope(k[position], position, parameters["rope_dim"], parameters["rope_theta"])

    attention_outputs = []
    for position in range(fixture["query_start"], fixture["context"]):
        start = max(0, position + 1 - parameters["sliding_window"])
        attention_outputs.append(
            attention_query(q[position], k[start : position + 1], v[start : position + 1], sinks)
        )
    attention_output = np.asarray(attention_outputs, dtype=np.float32).reshape(8, 8192)
    projected = project(output_weight, attention_output)
    post_attention = hidden[fixture["query_start"] :] + projected
    moe_input = rms_norm(post_attention, post_norm)
    selected, route_weights, boundary_margin = noaux_tc_route(
        moe_input, router_weight, correction, parameters["top_k"]
    )
    union, counts = np.unique(selected, return_counts=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "hidden_f32": hidden,
        "normalized_f32": normalized,
        "attention_f32": attention_output,
        "post_attention_f32": post_attention,
        "moe_input_f32": moe_input,
    }
    artifact_records = {}
    for name, values in artifacts.items():
        payload = array_bytes(values)
        filename = f"{name}.bin"
        write_new(output_dir / filename, payload)
        artifact_records[name] = {
            "file": filename,
            "shape": list(values.shape),
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }

    result = {
        "schema_version": 1,
        "semantic": fixture["semantic"],
        "revision": fixture["revision"],
        "layer": layer,
        "context": fixture["context"],
        "query_start": fixture["query_start"],
        "query_count": fixture["query_count"],
        "seed": fixture["seed"],
        "source_file": source_name,
        "source_file_sha256": source_lock["sha256"],
        "model_source_sha256": model_lock["sha256"],
        "semantic_fixture_sha256": sha256_file(semantic_fixture_path),
        "parameters": parameters,
        "projection_sample_maximum_absolute_error": sample_error,
        "selected_experts_by_position": selected.tolist(),
        "route_weights_by_position": route_weights.tolist(),
        "selected_expert_union": union.tolist(),
        "expert_position_counts": {
            str(int(expert)): int(count) for expert, count in zip(union, counts)
        },
        "minimum_topk_boundary_margin": boundary_margin,
        "artifacts": artifact_records,
        "boundary_sha256": {
            "qkv": array_sha256(qkv),
            "projected_attention": array_sha256(projected),
            "route_ids": hashlib.sha256(selected.astype("<u4").tobytes()).hexdigest(),
            "route_weights": array_sha256(route_weights),
        },
        "oracle": f"MLX {mx.__version__ if hasattr(mx, '__version__') else 'runtime'} projections; NumPy source attention; Torch {torch.__version__} noaux-tc",
        "performance_claim": None,
    }
    atomic_write_new(output_dir / "manifest.json", canonical_json(result))
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", required=True, type=Path)
    parser.add_argument("--lock", required=True, type=Path)
    parser.add_argument("--verification-manifest", required=True, type=Path)
    parser.add_argument("--semantic-fixture", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    arguments = parser.parse_args()
    try:
        result = generate(
            arguments.checkpoint_dir,
            arguments.lock,
            arguments.verification_manifest,
            arguments.semantic_fixture,
            arguments.output_dir,
        )
        print(canonical_json(result).decode("utf-8"), end="")
        return 0
    except (OSError, ValueError, json.JSONDecodeError, struct.error) as error:
        print(f"error: {error}", file=__import__("sys").stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
