#!/usr/bin/env python3
"""Generate the independent verified-checkpoint oracle for PW-0098."""

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


INPUT_SHA256 = "ac6776035eee0537ab0d7d7975d4ad92e08bf67930b58d47a4d9f2e051113150"
MANIFEST_SHA256 = "a2b30be7ab767c754fd4680887420246dcd28d314bff242b144f664a8ff12470"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def generate(checkpoint_root: Path, verification: Path, manifest_path: Path,
             input_path: Path, output: Path) -> None:
    started = time.monotonic()
    torch.set_num_threads(1)
    if sha256(manifest_path) != MANIFEST_SHA256 or sha256(input_path) != INPUT_SHA256:
        raise ValueError("PW-0037 authority hash mismatch")
    manifest = json.loads(manifest_path.read_text())
    raw = np.fromfile(input_path, dtype="<f4")
    if raw.shape != (8 * 4096,) or not np.isfinite(raw).all():
        raise ValueError("PW-0037 input shape/value mismatch")
    values = torch.from_numpy(raw[:4096].copy()).reshape(1, 4096).to(torch.bfloat16)
    checkpoint = ShardedCheckpoint(checkpoint_root, verification)
    output.mkdir(parents=True, exist_ok=False)
    safety = Safety()

    router_prefix = "model.layers.43.mlp.gate"
    router_weight = checkpoint.tensor(router_prefix + ".weight").float()
    correction = checkpoint.tensor(router_prefix + ".e_score_correction_bias").float()
    logits = values.float() @ router_weight.T
    scores = torch.sigmoid(logits)
    _, selected = torch.topk(scores + correction, k=8, dim=-1, sorted=False)
    route_weights = scores.gather(1, selected)
    route_weights = route_weights / (route_weights.sum(-1, keepdim=True) + 1e-20)
    selected_list = selected[0].tolist()
    weights_list = route_weights[0].tolist()
    if selected_list != manifest["selected_experts_by_position"][0]:
        raise ValueError(f"independent route order mismatch: {selected_list}")
    safety.check("routing_complete")

    routed = torch.zeros((1, 4096), dtype=torch.float32)
    expert_outputs = {}
    for slot, expert in enumerate(selected_list):
        prefix = f"model.layers.43.mlp.experts.{expert}"
        gate = expert_linear(checkpoint, prefix + ".gate_proj.weight", values)
        up = expert_linear(checkpoint, prefix + ".up_proj.weight", values)
        hidden = (torch.nn.functional.silu(gate) * up).to(torch.bfloat16)
        down = expert_linear(checkpoint, prefix + ".down_proj.weight", hidden)
        routed += down.float() * float(weights_list[slot])
        expert_bytes = down.float().numpy().astype("<f4", copy=False).tobytes()
        expert_path = output / f"expert_{expert}.f32"
        atomic_write_new(expert_path, expert_bytes)
        expert_outputs[str(expert)] = {
            "file": expert_path.name,
            "sha256": sha256(expert_path),
            "shape": [4096],
            "dtype": "BF16_widened_F32",
        }
        safety.check(f"expert_{expert}_complete")
    result = routed.to(torch.bfloat16)
    result_bytes = result.float().numpy().astype("<f4", copy=False).tobytes()
    result_path = output / "expected.f32"
    atomic_write_new(result_path, result_bytes)
    safety.check("routed_output_complete")
    manifest_out = {
        "schema_version": 1,
        "semantic": "mimo_layer43_verified_checkpoint_bf16_staged_routed_row_oracle",
        "revision": REVISION,
        "checkpoint_verification_sha256": VERIFICATION_SHA256,
        "source_manifest_sha256": MANIFEST_SHA256,
        "input_sha256": INPUT_SHA256,
        "selected_experts": selected_list,
        "route_weights": weights_list,
        "expert_outputs": expert_outputs,
        "output_file": result_path.name,
        "output_sha256": sha256(result_path),
        "output_shape": [4096],
        "output_dtype": "BF16_widened_F32",
        "torch_version": torch.__version__,
        "safety_snapshots": safety.snapshots,
        "wall_ms": (time.monotonic() - started) * 1000.0,
        "performance_claim": None,
    }
    atomic_write_new(output / "manifest.json", canonical_json(manifest_out))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--verification", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    generate(args.checkpoint, args.verification, args.manifest, args.input, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
