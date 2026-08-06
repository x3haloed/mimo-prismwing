#!/usr/bin/env python3
"""Extract PW-0090's hash-bound BF16 LM-head dot fixture."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import struct

import numpy as np
import torch

try:
    from tools.generate_pytorch_bf16_dot_fixture import (
        forward_dot, source_four_lane_dot, source_specialized_vector_dot,
    )
    from tools.generate_real_layer1_expert_oracle import ShardedCheckpoint
    from tools.openrouter_reference import atomic_write_new, canonical_json
except ModuleNotFoundError:
    from generate_pytorch_bf16_dot_fixture import forward_dot, source_four_lane_dot, source_specialized_vector_dot
    from generate_real_layer1_expert_oracle import ShardedCheckpoint
    from openrouter_reference import atomic_write_new, canonical_json


ORACLE_SHA256 = "081550060338070eaa00730877065d2752824c589c22f74eaa7e921448c61573"
RUST_SHA256 = "0e8b14621a5e3e3715c8136bbef53ae94da674df9a0e9435e3ae881fb5d11f80"
TOKEN = 15_745


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def f32_bits(value: np.float32) -> int:
    return struct.unpack("<I", struct.pack("<f", float(value)))[0]


def capture(root: Path, manifest: dict, name: str, shape: list[int]) -> np.ndarray:
    record = manifest["captures"][name]
    path = root / record["file"]
    if record.get("shape") != shape or sha256(path) != record.get("sha256"):
        raise ValueError(f"{name} capture authority mismatch")
    return np.fromfile(path, dtype="<f4").reshape(shape)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oracle", required=True, type=Path)
    parser.add_argument("--rust", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--verification", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if sha256(args.oracle) != ORACLE_SHA256 or sha256(args.rust) != RUST_SHA256:
        raise ValueError("LM-head fixture manifest hash mismatch")
    oracle = json.loads(args.oracle.read_text())
    rust = json.loads(args.rust.read_text())
    if (oracle.get("semantic") != "mimo_full_prefix_layer_final_oracle"
            or rust.get("semantic") != "mimo_full_prefix_layer_final_rust_trace"):
        raise ValueError("LM-head fixture manifest semantic mismatch")
    input_values = capture(args.oracle.parent, oracle, "final_norm", [27, 4096])[-1].copy()
    expected_logits = capture(args.oracle.parent, oracle, "last_logits", [152576])
    rust_logits = capture(args.rust.parent, rust, "last_logits", [152576])
    checkpoint = ShardedCheckpoint(args.checkpoint, args.verification)
    weight = checkpoint.tensor("lm_head.weight")[TOKEN].float().numpy().copy()
    input_tensor = torch.from_numpy(input_values).to(torch.bfloat16)
    weight_tensor = torch.from_numpy(weight).to(torch.bfloat16)
    dot = input_tensor @ weight_tensor
    specialized = source_specialized_vector_dot(input_values, weight)
    generic = source_four_lane_dot(input_values, weight)
    forward = forward_dot(input_values, weight)
    expected_bits = int(dot.view(torch.uint16))
    if expected_bits != int(torch.tensor(float(expected_logits[TOKEN])).to(torch.bfloat16).view(torch.uint16)):
        raise ValueError("PyTorch dot does not reproduce frozen oracle logit")
    if int(torch.tensor(float(specialized)).to(torch.bfloat16).view(torch.uint16)) != expected_bits:
        raise ValueError("specialized topology does not reproduce PyTorch LM-head dot")
    rust_bits = int(torch.tensor(float(rust_logits[TOKEN])).to(torch.bfloat16).view(torch.uint16))
    if rust_bits == expected_bits:
        raise ValueError("selected LM-head row no longer discriminates current Rust")
    output = {
        "schema_version": 1,
        "semantic": "pytorch_aarch64_bf16_lm_head_specialized_dot",
        "torch_version": torch.__version__,
        "torch_commit": "cf30153c4c131c8164ee7798e5022d810682e2cb",
        "oracle_manifest_sha256": ORACLE_SHA256,
        "rust_manifest_sha256": RUST_SHA256,
        "final_norm_capture_sha256": oracle["captures"]["final_norm"]["sha256"],
        "lm_head_token": TOKEN,
        "width": 4096,
        "input_bf16_u16": input_tensor.view(torch.uint16).tolist(),
        "weight_bf16_u16": weight_tensor.view(torch.uint16).tolist(),
        "source_specialized_vector_f32_u32": f32_bits(specialized),
        "source_generic_four_lane_f32_u32": f32_bits(generic),
        "forward_f32_u32": f32_bits(forward),
        "pytorch_dot_bf16_u16": expected_bits,
        "pw0089_rust_logit_bf16_u16": rust_bits,
    }
    atomic_write_new(args.output, canonical_json(output))


if __name__ == "__main__":
    main()
