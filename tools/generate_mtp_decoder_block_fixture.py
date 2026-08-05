#!/usr/bin/env python3
"""Run a complete learned MTP decoder block with affine8 Metal attention."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
import struct

import mlx.core as mx
import numpy as np
from safetensors import safe_open

try:
    from tools.generate_mtp_attention_fixture import (
        EXPECTED as ATTENTION_EXPECTED,
        OUT,
        QKV,
        REVISION,
        SCALE,
        SINK,
        SOURCE_SHA256,
        attention,
        rope,
        sha256_file,
    )
    from tools.openrouter_reference import atomic_write_new, canonical_json
except ModuleNotFoundError:
    from generate_mtp_attention_fixture import (
        EXPECTED as ATTENTION_EXPECTED,
        OUT,
        QKV,
        REVISION,
        SCALE,
        SINK,
        SOURCE_SHA256,
        attention,
        rope,
        sha256_file,
    )
    from openrouter_reference import atomic_write_new, canonical_json

MODEL_SOURCE_SHA256 = "a8c3cb3aae473bcc15f023010547c919f15eba6546e6ed7efb61a8937b12f3ad"
ATTENTION_FIXTURE_SHA256 = "dbb2a67bb4573cddb4d4b6cfcea55f224190308605ca9bf40594c4f7f489acd1"
METAL_ATTENTION_SHA256 = "fdb8ea39872939fd44bd01d383f384239ed074b156ee8cbbffe078fdba9a6108"
INPUT_NORM = "model.mtp.layers.0.input_layernorm.weight"
PRE_MLP_NORM = "model.mtp.layers.0.pre_mlp_layernorm.weight"
GATE = "model.mtp.layers.0.mlp.gate_proj.weight"
GATE_SCALE = "model.mtp.layers.0.mlp.gate_proj.weight_scale_inv"
UP = "model.mtp.layers.0.mlp.up_proj.weight"
UP_SCALE = "model.mtp.layers.0.mlp.up_proj.weight_scale_inv"
DOWN = "model.mtp.layers.0.mlp.down_proj.weight"
DOWN_SCALE = "model.mtp.layers.0.mlp.down_proj.weight_scale_inv"
EXPECTED = {
    **ATTENTION_EXPECTED,
    PRE_MLP_NORM: {"dtype":"BF16","shape":[4096],"data_offsets":[67333632,67341824]},
    GATE: {"dtype":"F8_E4M3","shape":[16384,4096],"data_offsets":[470077312,537186176]},
    GATE_SCALE: {"dtype":"F32","shape":[128,32],"data_offsets":[16384,32768]},
    UP: {"dtype":"F8_E4M3","shape":[16384,4096],"data_offsets":[537186176,604295040]},
    UP_SCALE: {"dtype":"F32","shape":[128,32],"data_offsets":[32768,49152]},
    DOWN: {"dtype":"F8_E4M3","shape":[4096,16384],"data_offsets":[402968448,470077312]},
    DOWN_SCALE: {"dtype":"F32","shape":[32,128],"data_offsets":[0,16384]},
}


def digest(array: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(array, dtype="<f4").tobytes()).hexdigest()


def rms_norm(values: np.ndarray, weight: np.ndarray) -> np.ndarray:
    variance = np.mean(values * values, axis=-1, keepdims=True, dtype=np.float32)
    return values * np.reciprocal(np.sqrt(variance + np.float32(1e-5))) * weight


def install_scales(weight: np.ndarray, scales: np.ndarray) -> np.ndarray:
    expected = (weight.shape[0] // 128, weight.shape[1] // 128)
    if scales.shape != expected:
        raise ValueError(f"FP8 scale grid mismatch: expected {expected}, got {scales.shape}")
    return weight * np.repeat(np.repeat(scales, 128, axis=0), 128, axis=1)


def raw_bf16_matches(
    checkpoint: Path,
    payload_start: int,
    name: str,
    decoded: np.ndarray,
) -> None:
    start, end = EXPECTED[name]["data_offsets"]
    raw = np.memmap(
        checkpoint, dtype="<u2", mode="r", offset=payload_start + start,
        shape=((end - start) // 2,),
    )
    flat = decoded.reshape(-1)
    for offset in range(0, raw.size, 1_000_000):
        bits = np.asarray(raw[offset:offset + 1_000_000], dtype=np.uint16)
        manual = (bits.astype(np.uint32) << 16).view(np.float32)
        if not np.array_equal(manual, flat[offset:offset + manual.size]):
            raise ValueError(f"BF16 decode mismatch: {name}")


def project(weight: np.ndarray, values: np.ndarray) -> np.ndarray:
    result = mx.matmul(mx.array(values), mx.array(weight).T)
    mx.eval(result)
    return np.array(result, copy=False).astype(np.float32)


def generate(
    checkpoint: Path,
    model_source: Path,
    attention_fixture_path: Path,
    metal_attention_path: Path,
) -> dict:
    if sha256_file(checkpoint) != SOURCE_SHA256:
        raise ValueError("MTP source SHA-256 mismatch")
    if sha256_file(model_source) != MODEL_SOURCE_SHA256:
        raise ValueError("model source SHA-256 mismatch")
    if sha256_file(attention_fixture_path) != ATTENTION_FIXTURE_SHA256:
        raise ValueError("attention fixture SHA-256 mismatch")
    if sha256_file(metal_attention_path) != METAL_ATTENTION_SHA256:
        raise ValueError("Metal attention artifact SHA-256 mismatch")
    with checkpoint.open("rb") as source:
        header_size = struct.unpack("<Q", source.read(8))[0]
        header = json.loads(source.read(header_size))
    for name, metadata in EXPECTED.items():
        if header.get(name) != metadata:
            raise ValueError(f"MTP metadata mismatch: {name}")
    fixture = json.loads(attention_fixture_path.read_text())
    if (
        fixture.get("schema_version") != 1
        or fixture.get("semantic") != "mimo_mtp_real_attention_context17"
        or fixture.get("source_revision") != REVISION
        or fixture.get("source_sha256") != SOURCE_SHA256
        or len(fixture.get("wht_affine8_expected_attention_f32", [])) != 8192
    ):
        raise ValueError("attention fixture identity or shape mismatch")
    metal_attention = np.fromfile(metal_attention_path, dtype="<f4")
    expected_affine_attention = np.asarray(
        fixture["wht_affine8_expected_attention_f32"], dtype=np.float32
    )
    if metal_attention.shape != (8192,) or not np.isfinite(metal_attention).all():
        raise ValueError("Metal attention artifact shape or finiteness mismatch")
    metal_parity = float(
        np.linalg.norm(metal_attention - expected_affine_attention)
        / np.linalg.norm(expected_affine_attention)
    )
    if metal_parity > 4e-4:
        raise ValueError("Metal attention artifact parity mismatch")

    with safe_open(checkpoint, framework="pt", device="cpu") as tensors:
        input_norm = tensors.get_tensor(INPUT_NORM).float().numpy()
        pre_mlp_norm = tensors.get_tensor(PRE_MLP_NORM).float().numpy()
        sinks = tensors.get_tensor(SINK).float().numpy()
        output_weight = tensors.get_tensor(OUT).float().numpy()
        raw_qkv = tensors.get_tensor(QKV).float().numpy()
        qkv_scales = tensors.get_tensor(SCALE).float().numpy()
    payload_start = 8 + header_size
    for name, decoded in (
        (INPUT_NORM, input_norm), (PRE_MLP_NORM, pre_mlp_norm),
        (SINK, sinks), (OUT, output_weight),
    ):
        raw_bf16_matches(checkpoint, payload_start, name, decoded)

    qkv_weight = install_scales(raw_qkv, qkv_scales)
    rng = np.random.default_rng(260026)
    hidden = rng.standard_normal((17, 4096), dtype=np.float32)
    normalized = rms_norm(hidden, input_norm)
    qkv = project(qkv_weight, normalized)
    sample_rows = [0, 1, 12288, 13824]
    qkv_scalar = np.array([
        np.dot(normalized[0].astype(np.float64), qkv_weight[row].astype(np.float64))
        for row in sample_rows
    ])
    qkv_sample_error = float(np.max(np.abs(qkv[0, sample_rows].astype(np.float64) - qkv_scalar)))
    q = qkv[:, :12288].reshape(17, 64, 192)
    k = qkv[:, 12288:13824].reshape(17, 8, 192)
    v = qkv[:, 13824:].reshape(17, 8, 128) * np.float32(0.707)
    for token in range(17):
        for head in range(64):
            q[token, head] = rope(q[token, head], token)
        for head in range(8):
            k[token, head] = rope(k[token, head], token)
    source_attention = attention(q[-1], k, v, sinks).reshape(-1)
    if digest(source_attention) != fixture.get("source_attention_sha256"):
        raise ValueError("source attention identity mismatch")
    source_projection = project(output_weight, source_attention.reshape(1, -1))[0]
    candidate_projection = project(output_weight, metal_attention.reshape(1, -1))[0]
    residual = hidden[-1]
    source_post_attention = residual + source_projection
    candidate_post_attention = residual + candidate_projection
    source_mlp_input = rms_norm(source_post_attention, pre_mlp_norm)
    candidate_mlp_input = rms_norm(candidate_post_attention, pre_mlp_norm)
    mlp_inputs = np.stack([source_mlp_input, candidate_mlp_input])

    projection_errors = {"qkv": qkv_sample_error}
    with safe_open(checkpoint, framework="pt", device="cpu") as tensors:
        raw_gate = tensors.get_tensor(GATE).float().numpy()
        gate_scales = tensors.get_tensor(GATE_SCALE).float().numpy()
    gate_weight = install_scales(raw_gate, gate_scales)
    gate = project(gate_weight, mlp_inputs)
    gate_scalar = np.dot(mlp_inputs[0].astype(np.float64), gate_weight[0].astype(np.float64))
    projection_errors["gate"] = float(abs(np.float64(gate[0, 0]) - gate_scalar))
    del raw_gate, gate_scales, gate_weight
    gc.collect()

    with safe_open(checkpoint, framework="pt", device="cpu") as tensors:
        raw_up = tensors.get_tensor(UP).float().numpy()
        up_scales = tensors.get_tensor(UP_SCALE).float().numpy()
    up_weight = install_scales(raw_up, up_scales)
    up = project(up_weight, mlp_inputs)
    up_scalar = np.dot(mlp_inputs[0].astype(np.float64), up_weight[0].astype(np.float64))
    projection_errors["up"] = float(abs(np.float64(up[0, 0]) - up_scalar))
    del raw_up, up_scales, up_weight
    gc.collect()
    activated = (gate / (np.float32(1) + np.exp(-gate))) * up

    with safe_open(checkpoint, framework="pt", device="cpu") as tensors:
        raw_down = tensors.get_tensor(DOWN).float().numpy()
        down_scales = tensors.get_tensor(DOWN_SCALE).float().numpy()
    down_weight = install_scales(raw_down, down_scales)
    mlp_output = project(down_weight, activated)
    down_scalar = np.dot(activated[0].astype(np.float64), down_weight[0].astype(np.float64))
    projection_errors["down"] = float(abs(np.float64(mlp_output[0, 0]) - down_scalar))
    if max(projection_errors.values()) > 2e-4:
        raise ValueError(f"projection scalar gate failed: {projection_errors}")
    source_final = source_post_attention + mlp_output[0]
    candidate_final = candidate_post_attention + mlp_output[1]

    def boundary(source: np.ndarray, candidate: np.ndarray) -> dict:
        return {
            "source_sha256": digest(source),
            "candidate_sha256": digest(candidate),
            "candidate_relative_l2_vs_source": float(
                np.linalg.norm(candidate - source) / np.linalg.norm(source)
            ),
        }

    return {
        "schema_version": 1,
        "semantic": "mimo_mtp_layer0_complete_decoder_block_context17_final_token",
        "mode": "modified_wht_affine8_attention",
        "source_revision": REVISION,
        "source_sha256": SOURCE_SHA256,
        "model_source_sha256": MODEL_SOURCE_SHA256,
        "attention_fixture_sha256": ATTENTION_FIXTURE_SHA256,
        "metal_attention_sha256": METAL_ATTENTION_SHA256,
        "context": 17,
        "hidden_size": 4096,
        "intermediate_size": 16384,
        "projection_sample_max_abs_errors": projection_errors,
        "metal_attention_relative_l2_vs_packed_scalar": metal_parity,
        "boundaries": {
            "attention": boundary(source_attention, metal_attention),
            "post_attention_residual": boundary(source_post_attention, candidate_post_attention),
            "pre_mlp_normalized": boundary(source_mlp_input, candidate_mlp_input),
            "mlp_output": boundary(mlp_output[0], mlp_output[1]),
            "final_block_state": boundary(source_final, candidate_final),
        },
        "source_final_first8": source_final[:8].tolist(),
        "candidate_final_first8": candidate_final[:8].tolist(),
        "performance_claim": None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--model-source", required=True, type=Path)
    parser.add_argument("--attention-fixture", required=True, type=Path)
    parser.add_argument("--metal-attention", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = generate(
        args.checkpoint, args.model_source, args.attention_fixture, args.metal_attention
    )
    atomic_write_new(args.output, canonical_json(result))


if __name__ == "__main__":
    main()
