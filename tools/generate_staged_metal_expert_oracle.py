#!/usr/bin/env python3
"""Generate the independent source-faithful oracle for PW-0097."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

try:
    from tools.generate_real_layer0_bf16_oracle import REVISION, VERIFICATION_SHA256, Safety
    from tools.generate_real_layer1_expert_oracle import ShardedCheckpoint, expert_linear
    from tools.openrouter_reference import atomic_write_new, canonical_json
except ModuleNotFoundError:
    from generate_real_layer0_bf16_oracle import REVISION, VERIFICATION_SHA256, Safety
    from generate_real_layer1_expert_oracle import ShardedCheckpoint, expert_linear
    from openrouter_reference import atomic_write_new, canonical_json


PREFIX = "model.layers.43.mlp.experts.32"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def generate(checkpoint_root: Path, verification: Path, input_path: Path,
             output: Path) -> None:
    started = time.monotonic()
    torch.set_num_threads(1)
    if sha256(input_path) != "ad261a98cc64c34277a40168f45654cabb1c1059e88771c3c71092ae6ffee5ba":
        raise ValueError("PW-0034 input SHA-256 mismatch")
    raw = np.fromfile(input_path, dtype="<f4")
    if raw.shape != (4096,) or not np.isfinite(raw).all():
        raise ValueError("PW-0034 input shape or values mismatch")
    values = torch.from_numpy(raw.copy()).reshape(1, 4096).to(torch.bfloat16)
    checkpoint = ShardedCheckpoint(checkpoint_root, verification)
    output.mkdir(parents=True, exist_ok=False)
    safety = Safety()
    gate = expert_linear(checkpoint, f"{PREFIX}.gate_proj.weight", values)
    safety.check("gate_complete")
    up = expert_linear(checkpoint, f"{PREFIX}.up_proj.weight", values)
    safety.check("up_complete")
    hidden = (torch.nn.functional.silu(gate) * up).to(torch.bfloat16)
    safety.check("swiglu_complete")
    result = expert_linear(checkpoint, f"{PREFIX}.down_proj.weight", hidden)
    safety.check("down_complete")
    widened = result.float().numpy().astype("<f4", copy=False).tobytes()
    result_path = output / "expected.f32"
    atomic_write_new(result_path, widened)
    manifest = {
        "schema_version": 1,
        "semantic": "mimo_layer43_expert32_dynamic_fp8_bf16_staged_oracle",
        "revision": REVISION,
        "checkpoint_verification_sha256": VERIFICATION_SHA256,
        "input_file": input_path.name,
        "input_sha256": sha256(input_path),
        "output_file": result_path.name,
        "output_sha256": sha256(result_path),
        "output_shape": [4096],
        "output_dtype": "BF16_widened_F32",
        "numerics": "BF16 input; dynamic E4M3FN per group 128 before each linear; BF16 gate/up/SwiGLU/down",
        "torch_version": torch.__version__,
        "safety_snapshots": safety.snapshots,
        "wall_ms": (time.monotonic() - started) * 1000.0,
        "performance_claim": None,
    }
    atomic_write_new(output / "manifest.json", canonical_json(manifest))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--verification", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    generate(args.checkpoint, args.verification, args.input, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
