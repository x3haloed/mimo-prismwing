#!/usr/bin/env python3
"""Verify and inventory the pinned DFlash draft without loading its tensors."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

from safetensors import safe_open
import torch

try:
    from tools.host_safety import HostSafetyMonitor
    from tools.openrouter_reference import atomic_write_new, canonical_json
except ModuleNotFoundError:
    from host_safety import HostSafetyMonitor
    from openrouter_reference import atomic_write_new, canonical_json


REPOSITORY = "XiaomiMiMo/MiMo-V2.5-DFlash"
REVISION = "1f58446181abcaa01030fdbde835fbd38ae9a2b1"
REQUIRED_PATHS = (
    "dflash/config.json",
    "dflash/dflash.py",
    "dflash/dflash_draft_model.safetensors",
    "dflash/mask_embedding.pt",
    "dflash/model.safetensors.index.json",
)


def expected_inventory() -> dict[str, tuple[str, tuple[int, ...]]]:
    expected = {
        "fc.weight": ("BF16", (4096, 20480)),
        "hidden_norm.weight": ("BF16", (4096,)),
        "norm.weight": ("BF16", (4096,)),
    }
    per_layer = {
        "input_layernorm.weight": (4096,),
        "post_attention_layernorm.weight": (4096,),
        "mlp.down_proj.weight": (4096, 16384),
        "mlp.gate_proj.weight": (16384, 4096),
        "mlp.up_proj.weight": (16384, 4096),
        "self_attn.attention_sink_bias": (64,),
        "self_attn.k_norm.weight": (128,),
        "self_attn.k_proj.weight": (1024, 4096),
        "self_attn.o_proj.weight": (4096, 8192),
        "self_attn.q_norm.weight": (128,),
        "self_attn.q_proj.weight": (8192, 4096),
        "self_attn.v_proj.weight": (1024, 4096),
    }
    for layer in range(5):
        for suffix, shape in per_layer.items():
            expected[f"layers.{layer}.{suffix}"] = ("BF16", shape)
    return expected


def validate_config(config: dict) -> None:
    exact = {
        "model_type": "qwen3",
        "hidden_size": 4096,
        "intermediate_size": 16384,
        "num_hidden_layers": 5,
        "num_attention_heads": 64,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "partial_rotary_factor": 0.5,
        "block_size": 8,
        "num_target_layers": 48,
        "vocab_size": 152576,
        "max_position_embeddings": 262144,
        "rope_theta": 10000,
        "sliding_window": 1024,
        "rms_norm_eps": 1e-6,
        "torch_dtype": "bfloat16",
        "hidden_act": "silu",
        "attention_bias": False,
        "attention_dropout": 0.0,
        "tie_word_embeddings": False,
    }
    for key, expected in exact.items():
        if config.get(key) != expected:
            raise ValueError(f"DFlash config mismatch for {key}")
    dflash = config.get("dflash_config")
    if not isinstance(dflash, dict):
        raise ValueError("DFlash config lacks dflash_config")
    required = {
        "target_layer_ids": [0, 11, 23, 35, 47],
        "mask_token_id": 151675,
        "block_size": 8,
        "use_swa": True,
        "swa_window_size": 1024,
        "backbone_rotary_base": 5000000,
        "attention_value_scale": 0.612,
        "attention_sink_bias": True,
    }
    for key, expected in required.items():
        if dflash.get(key) != expected:
            raise ValueError(f"DFlash nested config mismatch for {key}")


def validate_inventory(inventory: dict[str, tuple[str, tuple[int, ...]]]) -> None:
    expected = expected_inventory()
    if inventory.keys() != expected.keys():
        missing = sorted(expected.keys() - inventory.keys())
        extra = sorted(inventory.keys() - expected.keys())
        raise ValueError(f"DFlash tensor names mismatch: missing={missing}, extra={extra}")
    for name, identity in inventory.items():
        if identity != expected[name]:
            raise ValueError(f"DFlash tensor layout mismatch for {name}: {identity}")


def sha256_file(path: Path, safety: HostSafetyMonitor, phase: str) -> str:
    digest = hashlib.sha256()
    bytes_read = 0
    next_checkpoint = 512 * 1024**2
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024**2), b""):
            digest.update(chunk)
            bytes_read += len(chunk)
            if bytes_read >= next_checkpoint:
                safety.checkpoint(f"{phase}_{bytes_read}_bytes")
                next_checkpoint += 512 * 1024**2
    safety.checkpoint(f"{phase}_complete")
    return digest.hexdigest()


def audit(lock_path: Path, root: Path, output: Path) -> dict:
    started = time.monotonic()
    safety = HostSafetyMonitor()
    lock_bytes = lock_path.read_bytes()
    lock = json.loads(lock_bytes)
    if (
        lock.get("schema_version") != 1
        or lock.get("repository") != REPOSITORY
        or lock.get("revision") != REVISION
    ):
        raise ValueError("DFlash lock identity mismatch")
    records = {item.get("path"): item for item in lock.get("files", [])}
    if any(path not in records for path in REQUIRED_PATHS):
        raise ValueError("DFlash lock lacks required draft artifacts")

    verified_files = []
    for relative in REQUIRED_PATHS:
        record = records[relative]
        path = root / relative
        stat = path.stat()
        if not path.is_file() or stat.st_size != record.get("bytes"):
            raise ValueError(f"DFlash artifact size mismatch: {relative}")
        actual_hash = sha256_file(path, safety, relative.replace("/", "_"))
        if actual_hash != record.get("sha256"):
            raise ValueError(f"DFlash artifact SHA-256 mismatch: {relative}")
        verified_files.append(
            {
                "path": relative,
                "bytes": stat.st_size,
                "sha256": actual_hash,
                "device": stat.st_dev,
                "inode": stat.st_ino,
                "modified_ns": stat.st_mtime_ns,
            }
        )

    config = json.loads((root / "dflash/config.json").read_text())
    validate_config(config)
    index = json.loads((root / "dflash/model.safetensors.index.json").read_text())
    expected_names = expected_inventory().keys()
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or weight_map.keys() != expected_names:
        raise ValueError("DFlash index tensor names mismatch")
    if set(weight_map.values()) != {"dflash_draft_model.safetensors"}:
        raise ValueError("DFlash index shard mapping mismatch")
    if index.get("metadata", {}).get("total_size") != 2_936_114_304:
        raise ValueError("DFlash index tensor byte total mismatch")

    draft_path = root / "dflash/dflash_draft_model.safetensors"
    with safe_open(draft_path, framework="pt", device="cpu") as source:
        if source.metadata() is not None:
            raise ValueError("unexpected DFlash safetensors metadata")
        inventory = {
            name: (source.get_slice(name).get_dtype(), tuple(source.get_slice(name).get_shape()))
            for name in source.keys()
        }
    validate_inventory(inventory)
    safety.checkpoint("tensor_header_inventory_complete")

    mask = torch.load(root / "dflash/mask_embedding.pt", map_location="cpu", weights_only=True)
    if (
        not isinstance(mask, dict)
        or mask.keys() != {"mask_token_id", "embedding"}
        or mask["mask_token_id"] != 151675
        or not isinstance(mask["embedding"], torch.Tensor)
        or mask["embedding"].dtype != torch.bfloat16
        or tuple(mask["embedding"].shape) != (4096,)
        or not torch.isfinite(mask["embedding"]).all()
    ):
        raise ValueError("DFlash mask embedding identity mismatch")
    del mask
    safety.release_checkpoint("artifact_audit_release", ["mask embedding", "tensor header views"])

    result = {
        "schema_version": 1,
        "evidence_class": "pinned_dflash_draft_artifact_audit",
        "repository": REPOSITORY,
        "revision": REVISION,
        "lock_sha256": hashlib.sha256(lock_bytes).hexdigest(),
        "verified_files": verified_files,
        "tensor_count": len(inventory),
        "tensor_inventory": [
            {"name": name, "dtype": dtype, "shape": list(shape)}
            for name, (dtype, shape) in sorted(inventory.items())
        ],
        "config": config,
        "mask_embedding": {"token_id": 151675, "dtype": "BF16", "shape": [4096]},
        "safety": safety.evidence(),
        "wall_ms": (time.monotonic() - started) * 1000,
    }
    atomic_write_new(output, canonical_json(result))
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", required=True, type=Path)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    try:
        result = audit(arguments.lock, arguments.root, arguments.output)
        print(json.dumps({"output": str(arguments.output), "tensors": result["tensor_count"]}))
        return 0
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as error:
        print(json.dumps({"error": str(error)}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
