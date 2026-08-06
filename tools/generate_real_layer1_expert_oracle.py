#!/usr/bin/env python3
"""Generate the independent PW-0059 selected-expert layer-1 trace."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from safetensors import safe_open
import torch

try:
    from tools.generate_real_layer0_bf16_oracle import (
        PROMPT_IDS, REVISION, VERIFICATION_SHA256, Safety, dynamic_input, write_capture,
    )
    from tools.openrouter_reference import atomic_write_new, canonical_json
except ModuleNotFoundError:
    from generate_real_layer0_bf16_oracle import (
        PROMPT_IDS, REVISION, VERIFICATION_SHA256, Safety, dynamic_input, write_capture,
    )
    from openrouter_reference import atomic_write_new, canonical_json


NUMERICS = "dynamic_fp8_e4m3fn_per_token_group_128_bf16_boundaries"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_file(path: Path, record: dict) -> None:
    stat = path.stat()
    if (record.get("status") != "verified" or stat.st_size != record.get("bytes")
            or stat.st_dev != record.get("device") or stat.st_ino != record.get("inode")
            or stat.st_mtime_ns != record.get("modified_ns")):
        raise ValueError(f"verified file identity changed: {path.name}")


class ShardedCheckpoint:
    def __init__(self, root: Path, verification_path: Path) -> None:
        if sha256(verification_path) != VERIFICATION_SHA256:
            raise ValueError("checkpoint verification SHA-256 mismatch")
        verification = json.loads(verification_path.read_text())
        if verification.get("revision") != REVISION or not verification.get("complete"):
            raise ValueError("checkpoint verification identity mismatch")
        self.root = root
        self.records = {record["path"]: record for record in verification["files"]}
        index_path = root / "model.safetensors.index.json"
        validate_file(index_path, self.records.get(index_path.name, {}))
        index = json.loads(index_path.read_text())
        self.weight_map = index.get("weight_map")
        if not isinstance(self.weight_map, dict) or len(self.weight_map) != 73_081:
            raise ValueError("checkpoint index mismatch")
        self.validated_shards: set[str] = set()

    def shard(self, name: str) -> Path:
        shard = self.weight_map.get(name)
        if not isinstance(shard, str) or Path(shard).name != shard:
            raise ValueError(f"{name}: checkpoint index authority mismatch")
        path = self.root / shard
        if shard not in self.validated_shards:
            validate_file(path, self.records.get(shard, {}))
            self.validated_shards.add(shard)
        return path

    def tensor(self, name: str) -> torch.Tensor:
        path = self.shard(name)
        with safe_open(path, framework="pt", device="cpu") as source:
            if name not in source.keys():
                raise ValueError(f"{name}: indexed tensor absent from shard")
            return source.get_tensor(name)


def load_capture(manifest_path: Path, manifest: dict, name: str,
                 shape: tuple[int, ...]) -> torch.Tensor:
    record = manifest["captures"][name]
    path = manifest_path.parent / record["file"]
    if (record.get("shape") != list(shape) or record.get("dtype") != "BF16_widened_F32"
            or sha256(path) != record.get("sha256")):
        raise ValueError(f"{name}: PW-0058 capture authority mismatch")
    values = np.fromfile(path, dtype="<f4")
    if values.size != int(np.prod(shape)) or not np.isfinite(values).all():
        raise ValueError(f"{name}: PW-0058 capture values mismatch")
    return torch.from_numpy(values.copy()).reshape(shape).to(torch.bfloat16)


def load_routing_authority(manifest_path: Path) -> tuple[dict, torch.Tensor, torch.Tensor]:
    manifest = json.loads(manifest_path.read_text())
    if (manifest.get("semantic") != "mimo_real_layer1_attention_to_routing_oracle"
            or manifest.get("revision") != REVISION
            or manifest.get("checkpoint_verification_sha256") != VERIFICATION_SHA256
            or manifest.get("prompt_token_ids") != PROMPT_IDS
            or manifest.get("numerics") != NUMERICS):
        raise ValueError("PW-0058 oracle authority mismatch")
    moe_input = load_capture(manifest_path, manifest, "post_attention_norm", (27, 4096))
    post_attention = load_capture(manifest_path, manifest, "post_attention", (27, 4096))
    selected = manifest.get("selected_experts_by_position")
    weights = manifest.get("route_weights_by_position")
    if (not isinstance(selected, list) or not isinstance(weights, list)
            or len(selected) != 27 or len(weights) != 27
            or any(len(row) != 8 or len(set(row)) != 8
                   or any(not isinstance(expert, int) or not 0 <= expert < 256 for expert in row)
                   for row in selected)
            or any(len(row) != 8 or any(not np.isfinite(weight) for weight in row)
                   for row in weights)
            or manifest["captures"]["incoming"].get("sha256") != manifest.get("source_input_sha256")):
        raise ValueError("PW-0058 routing shape mismatch")
    return manifest, moe_input, post_attention


def expert_linear(checkpoint: ShardedCheckpoint, name: str,
                  values: torch.Tensor) -> torch.Tensor:
    weight = checkpoint.tensor(name).float()
    scale = checkpoint.tensor(name + "_scale_inv").float()
    expected = ((weight.shape[0] + 127) // 128, (weight.shape[1] + 127) // 128)
    if (tuple(scale.shape) != expected or weight.shape[0] % 128
            or weight.shape[1] % 128):
        raise ValueError(f"{name}: FP8 layout mismatch")
    expanded = scale.repeat_interleave(128, 0).repeat_interleave(128, 1)
    inputs = dynamic_input(values)
    output = (inputs @ (weight * expanded).T).to(torch.bfloat16)
    del inputs, weight, scale, expanded
    gc.collect()
    return output


def generate(checkpoint_root: Path, verification: Path, routing_manifest: Path,
             output: Path) -> None:
    started = time.monotonic()
    torch.set_num_threads(1)
    authority, moe_input, post_attention = load_routing_authority(routing_manifest)
    checkpoint = ShardedCheckpoint(checkpoint_root, verification)
    output.mkdir(parents=True, exist_ok=False)
    safety = Safety()
    selected = authority["selected_experts_by_position"]
    route_weights = authority["route_weights_by_position"]
    schedule: dict[int, list[tuple[int, float]]] = {}
    for position, (experts, weights) in enumerate(zip(selected, route_weights, strict=True)):
        for expert, weight in zip(experts, weights, strict=True):
            schedule.setdefault(expert, []).append((position, weight))
    if len(schedule) != 28 or sum(map(len, schedule.values())) != 216:
        raise ValueError("PW-0059 frozen route union mismatch")

    gates, ups, activated_rows, downs = [], [], [], []
    routed = torch.zeros((27, 4096), dtype=torch.float32)
    schedule_manifest = []
    for expert in sorted(schedule):
        placements = schedule[expert]
        positions = [position for position, _ in placements]
        gathered = moe_input[positions]
        prefix = f"model.layers.1.mlp.experts.{expert}"
        gate = expert_linear(checkpoint, f"{prefix}.gate_proj.weight", gathered)
        up = expert_linear(checkpoint, f"{prefix}.up_proj.weight", gathered)
        activated = (torch.nn.functional.silu(gate) * up).to(torch.bfloat16)
        down = expert_linear(checkpoint, f"{prefix}.down_proj.weight", activated)
        gates.append(gate); ups.append(up); activated_rows.append(activated); downs.append(down)
        for local, (position, weight) in enumerate(placements):
            routed[position] += down[local].float() * float(weight)
        schedule_manifest.append({"expert": expert, "positions": positions})
        safety.check(f"layer_1_expert_{expert}_complete")
    routed = routed.to(torch.bfloat16)
    final = (post_attention + routed).to(torch.bfloat16)
    safety.check("layer_1_complete")
    captures = {
        "moe_input": write_capture(output, "moe_input", moe_input, safety),
        "expert_gate": write_capture(output, "expert_gate", torch.cat(gates), safety),
        "expert_up": write_capture(output, "expert_up", torch.cat(ups), safety),
        "expert_swiglu": write_capture(output, "expert_swiglu", torch.cat(activated_rows), safety),
        "expert_down": write_capture(output, "expert_down", torch.cat(downs), safety),
        "routed_output": write_capture(output, "routed_output", routed, safety),
        "final": write_capture(output, "final", final, safety),
    }
    manifest = {"schema_version": 1,
        "semantic": "mimo_real_layer1_selected_experts_oracle", "revision": REVISION,
        "checkpoint_verification_sha256": VERIFICATION_SHA256,
        "prompt_token_ids": PROMPT_IDS,
        "source_input_sha256": authority["source_input_sha256"], "numerics": NUMERICS,
        "captures": captures, "selected_experts_by_position": selected,
        "route_weights_by_position": route_weights, "expert_schedule": schedule_manifest,
        "torch_version": torch.__version__, "safety_snapshots": safety.snapshots,
        "wall_ms": (time.monotonic() - started) * 1000.0, "performance_claim": None}
    atomic_write_new(output / "manifest.json", canonical_json(manifest))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--verification", required=True, type=Path)
    parser.add_argument("--routing-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    generate(args.checkpoint, args.verification, args.routing_manifest, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
