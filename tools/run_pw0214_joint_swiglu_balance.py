#!/usr/bin/env python3
"""Run PW-0214's joint up-row/down-column balance control."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional

try:
    from tools.openrouter_reference import atomic_write_new, canonical_json
    from tools.run_pw0213_projection_probe import (
        CORPUS_SHA256, decode_weight, metrics, sha256_file,
    )
except ModuleNotFoundError:
    from openrouter_reference import atomic_write_new, canonical_json
    from run_pw0213_projection_probe import (
        CORPUS_SHA256, decode_weight, metrics, sha256_file,
    )


ALPHAS = tuple(round(index / 10, 1) for index in range(11))
TORCH_WHEEL = "torch-2.13.0+cpu-cp313-cp313-manylinux_2_28_x86_64.whl"
TORCH_WHEEL_BYTES = 191815667
TORCH_WHEEL_SHA256 = "3fbf9c9d1f3c10c2d59d04aca426dee9ccc6ceb32d255c61e93acc3b4f75fae6"


def dynamic_input(values: torch.Tensor) -> torch.Tensor:
    rows, columns = values.shape
    grouped = values.float().reshape(rows, columns // 128, 128)
    scales = grouped.abs().amax(-1).clamp(min=1e-10) / 448.0
    encoded = torch.clamp(
        grouped / scales.unsqueeze(-1), -448.0, 448.0
    ).to(torch.float8_e4m3fn)
    return (encoded.float() * scales.unsqueeze(-1)).reshape(rows, columns)


def source_linear(weight: np.ndarray, values: torch.Tensor) -> torch.Tensor:
    return (dynamic_input(values) @ torch.from_numpy(weight).T).to(torch.bfloat16)


def expert_output(inputs: np.ndarray, weights: dict[str, np.ndarray]) -> np.ndarray:
    values = torch.from_numpy(np.asarray(inputs, dtype=np.float32))
    gate = source_linear(weights["gate"], values)
    up = source_linear(weights["up"], values)
    hidden = (functional.silu(gate) * up).to(torch.bfloat16)
    return source_linear(weights["down"], hidden).float().numpy()


def affine6(weight: np.ndarray) -> np.ndarray:
    """Group-128 affine RTN with installed F16 scale and bias."""
    rows, columns = weight.shape
    if columns % 128 or not np.isfinite(weight).all():
        raise ValueError("invalid affine6 weight")
    grouped = np.asarray(weight, dtype=np.float32).reshape(rows, columns // 128, 128)
    minimum = grouped.min(axis=-1).astype(np.float16).astype(np.float32)
    maximum = grouped.max(axis=-1)
    step = ((maximum - grouped.min(axis=-1)) / 63.0).astype(np.float16).astype(np.float32)
    denominator = np.where(step[..., None] != 0, step[..., None], 1.0)
    codes = np.rint((grouped - minimum[..., None]) / denominator)
    return (
        minimum[..., None] + np.clip(codes, 0, 63) * step[..., None]
    ).astype(np.float16).astype(np.float32).reshape(rows, columns)


def balance_scales(up: np.ndarray, down: np.ndarray, alpha: float) -> np.ndarray:
    up_rms = np.sqrt(np.mean(np.asarray(up, dtype=np.float64) ** 2, axis=1))
    down_rms = np.sqrt(np.mean(np.asarray(down, dtype=np.float64) ** 2, axis=0))
    scales = np.clip((up_rms / np.maximum(down_rms, 1e-12)) ** (alpha / 2.0), 1 / 16, 16)
    grouped = scales.reshape(-1, 128)
    geometric = np.exp(np.mean(np.log(grouped), axis=1))
    scales = (grouped / geometric[:, None]).reshape(-1)
    if not np.all(np.isfinite(scales)) or np.any(scales <= 0):
        raise ValueError("invalid balance scale")
    return scales.astype(np.float32)


def run(source: Path, corpus_path: Path, layer: int, expert: int) -> dict:
    if sha256_file(corpus_path) != CORPUS_SHA256:
        raise ValueError("PW-0116 corpus authority mismatch")
    corpus = json.loads(corpus_path.read_text(encoding="utf-8"))
    authority = next(record for record in corpus["layers"] if record["layer"] == layer)
    schedule_index = next(
        index for index, record in enumerate(authority["expert_schedule"])
        if record["expert"] == expert
    )
    schedule = authority["expert_schedule"][schedule_index]
    offset = sum(len(record["positions"]) for record in authority["expert_schedule"][:schedule_index])
    train_local = [i for i, position in enumerate(schedule["positions"]) if position < 112]
    validation_local = [i for i, position in enumerate(schedule["positions"]) if 112 <= position < 168]
    train_positions = [schedule["positions"][i] for i in train_local]
    validation_positions = [schedule["positions"][i] for i in validation_local]
    root = corpus_path.parent
    moe_record = authority["captures"]["moe_input"]
    down_record = authority["captures"]["expert_down"]
    for record in (moe_record, down_record):
        if sha256_file(root / record["file"]) != record["sha256"]:
            raise ValueError("PW-0116 capture hash mismatch")
    moe = np.memmap(root / moe_record["file"], dtype="<f4", mode="r", shape=tuple(moe_record["shape"]))
    captured = np.memmap(root / down_record["file"], dtype="<f4", mode="r", shape=tuple(down_record["shape"]))
    train_expected = np.asarray(captured[[offset + i for i in train_local]], dtype=np.float32)
    validation_expected = np.asarray(captured[[offset + i for i in validation_local]], dtype=np.float32)
    arrays = np.load(source, allow_pickle=False)
    weights = {}
    for projection in ("gate", "up", "down"):
        key = f"layers__{layer}__mlp__experts__{expert}__{projection}_proj__weight"
        weights[projection] = decode_weight(arrays[f"{key}__codes"], arrays[f"{key}__scales"])
    train_inputs = np.asarray(moe[train_positions], dtype=np.float32)
    validation_inputs = np.asarray(moe[validation_positions], dtype=np.float32)
    source_train = expert_output(train_inputs, weights)
    source_validation = expert_output(validation_inputs, weights)
    parity_train = metrics(source_train, train_expected)
    parity_validation = metrics(source_validation, validation_expected)
    if parity_train["relative_l2"] > 0.002 or parity_validation["relative_l2"] > 0.002:
        raise ValueError(
            f"source parity insufficient: {parity_train['relative_l2']}, "
            f"{parity_validation['relative_l2']}"
        )
    symmetry_curve = []
    symmetry_failure = None
    for alpha in ALPHAS:
        scales = balance_scales(weights["up"], weights["down"], alpha)
        transformed_source = {
            "gate": weights["gate"],
            "up": weights["up"] / scales[:, None],
            "down": weights["down"] * scales[None, :],
        }
        train_symmetry = metrics(expert_output(train_inputs, transformed_source), source_train)
        validation_symmetry = metrics(
            expert_output(validation_inputs, transformed_source), source_validation
        )
        symmetry_curve.append({
            "alpha": alpha,
            "train_metrics": train_symmetry,
            "validation_metrics": validation_symmetry,
        })
        if (
            train_symmetry["relative_l2"] > 0.002
            or validation_symmetry["relative_l2"] > 0.002
        ):
            symmetry_failure = alpha
            break
    toolchain = {
        "torch_version": torch.__version__,
        "torch_wheel": TORCH_WHEEL,
        "torch_wheel_bytes": TORCH_WHEEL_BYTES,
        "torch_wheel_sha256": TORCH_WHEEL_SHA256,
    }
    if symmetry_failure is not None:
        return {
            "schema_version": 1,
            "evidence_class": "pw0214_joint_swiglu_neuron_balance_control",
            "layer": layer,
            "expert": expert,
            "train_positions": train_positions,
            "validation_positions": validation_positions,
            "source_parity_train": parity_train,
            "source_parity_validation": parity_validation,
            "source_symmetry_curve": symmetry_curve,
            "source_symmetry_threshold": 0.002,
            "first_failing_alpha": symmetry_failure,
            "gate_passed": False,
            "disposition": "rejected_before_quantized_search",
            "toolchain": toolchain,
        }
    gate_quantized = affine6(weights["gate"])
    curve = []
    best = None
    for alpha in ALPHAS:
        scales = balance_scales(weights["up"], weights["down"], alpha)
        transformed = {
            "gate": gate_quantized,
            "up": affine6(weights["up"] / scales[:, None]),
            "down": affine6(weights["down"] * scales[None, :]),
        }
        train_output = expert_output(train_inputs, transformed)
        train_metrics = metrics(train_output, train_expected)
        record = {
            "alpha": alpha,
            "scale_minimum": float(scales.min()),
            "scale_maximum": float(scales.max()),
            "scale_sha256": __import__("hashlib").sha256(scales.tobytes()).hexdigest(),
            "train_metrics": train_metrics,
        }
        curve.append(record)
        key = (train_metrics["relative_l2"], alpha)
        if best is None or key < best[0]:
            best = (key, alpha, scales.copy(), transformed)
    assert best is not None
    _, selected_alpha, selected_scales, selected = best
    identity = {name: affine6(weight) for name, weight in weights.items()}
    return {
        "schema_version": 1,
        "evidence_class": "pw0214_joint_swiglu_neuron_balance_control",
        "layer": layer,
        "expert": expert,
        "train_positions": train_positions,
        "validation_positions": validation_positions,
        "source_parity_train": parity_train,
        "source_parity_validation": parity_validation,
        "source_symmetry_curve": symmetry_curve,
        "source_symmetry_threshold": 0.002,
        "gate_passed": True,
        "toolchain": toolchain,
        "curve": curve,
        "selected_alpha": selected_alpha,
        "selected_scale_sha256": __import__("hashlib").sha256(selected_scales.tobytes()).hexdigest(),
        "affine6_train": metrics(expert_output(train_inputs, identity), train_expected),
        "affine6_validation": metrics(expert_output(validation_inputs, identity), validation_expected),
        "candidate_train": metrics(expert_output(train_inputs, selected), train_expected),
        "candidate_validation": metrics(expert_output(validation_inputs, selected), validation_expected),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--layer", required=True, type=int)
    parser.add_argument("--expert", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    result = run(arguments.source, arguments.corpus, arguments.layer, arguments.expert)
    atomic_write_new(arguments.output, canonical_json(result))
    print(arguments.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
