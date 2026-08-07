#!/usr/bin/env python3
"""Run PW-0127's authenticated impossible-perfect R720 CPU ceiling."""

from __future__ import annotations

import argparse
from collections import Counter
import gc
import json
from pathlib import Path
import platform
import time

import torch

try:
    from tools.generate_real_layer1_expert_oracle import ShardedCheckpoint
    from tools.host_safety import HostSafetyMonitor
    from tools.openrouter_reference import atomic_write_new, canonical_json
except ModuleNotFoundError:
    from generate_real_layer1_expert_oracle import ShardedCheckpoint
    from host_safety import HostSafetyMonitor
    from openrouter_reference import atomic_write_new, canonical_json


REVISION = "63651580ca774f8504f676040460aed3e1244ac1"
VERIFICATION_SHA256 = "9ddc8a99755f04ae2ea3c2484f6dd022d3f3a681b5a72c915ee4de833dbb0d03"
TARGETS = [10.0, 12.5, 34.3, 50.0]
SELECTED_EXPERT_BYTES_PER_TOKEN = 9_464_659_968


def matrix_macs(shape: tuple[int, ...]) -> int:
    if len(shape) != 2 or min(shape) <= 0:
        raise ValueError("matrix MAC accounting requires a positive 2D shape")
    return int(shape[0] * shape[1])


def peak_flops(sockets: int, cores_per_socket: int, maximum_ghz: float, sp_ops_cycle: int) -> float:
    if sockets <= 0 or cores_per_socket <= 0 or maximum_ghz <= 0 or sp_ops_cycle <= 0:
        raise ValueError("candidate peak specification must be positive")
    return sockets * cores_per_socket * maximum_ghz * 1e9 * sp_ops_cycle


def ceiling(peak: float, macs_per_token: int) -> dict:
    if peak <= 0 or macs_per_token <= 0:
        raise ValueError("ceiling inputs must be positive")
    flops_per_token = 2 * macs_per_token
    maximum_tps = peak / flops_per_token
    return {
        "mandatory_macs_per_token": macs_per_token,
        "mandatory_flops_per_token": flops_per_token,
        "impossible_peak_flops_per_second": peak,
        "impossible_maximum_tps": maximum_tps,
        "targets": {
            str(target): {
                "required_flops_per_second": flops_per_token * target,
                "required_fraction_of_impossible_peak": flops_per_token * target / peak,
                "arithmetically_possible_at_impossible_peak": target <= maximum_tps,
            }
            for target in TARGETS
        },
    }


def tensor_shape(checkpoint: ShardedCheckpoint, name: str, dtype: torch.dtype) -> tuple[int, int]:
    tensor = checkpoint.tensor(name)
    if tensor.dtype != dtype or tensor.ndim != 2:
        raise ValueError(f"{name}: matrix dtype/shape mismatch")
    return tuple(int(value) for value in tensor.shape)


def run(
    checkpoint_root: Path,
    verification_path: Path,
    output_path: Path,
    commit: str,
) -> dict:
    if output_path.exists():
        raise ValueError(f"refusing to overwrite {output_path}")
    if len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit):
        raise ValueError("implementation commit must be lowercase 40-hex")
    started = time.perf_counter()
    safety = HostSafetyMonitor()
    checkpoint = ShardedCheckpoint(checkpoint_root, verification_path)
    categories = {
        "attention_projections": 0,
        "dense_layer0_mlp": 0,
        "routers": 0,
        "selected_experts": 0,
        "lm_head": 0,
    }
    shape_counts: Counter[tuple[str, tuple[int, int], str]] = Counter()

    for layer in range(48):
        for projection, dtype in (
            ("qkv_proj", torch.float8_e4m3fn),
            ("o_proj", torch.bfloat16),
        ):
            name = f"model.layers.{layer}.self_attn.{projection}.weight"
            shape = tensor_shape(checkpoint, name, dtype)
            categories["attention_projections"] += matrix_macs(shape)
            shape_counts[(projection, shape, str(dtype))] += 1
        if layer == 0:
            for projection in ("gate_proj", "up_proj", "down_proj"):
                name = f"model.layers.0.mlp.{projection}.weight"
                shape = tensor_shape(checkpoint, name, torch.float8_e4m3fn)
                categories["dense_layer0_mlp"] += matrix_macs(shape)
                shape_counts[(f"dense_{projection}", shape, str(torch.float8_e4m3fn))] += 1
        else:
            router_name = f"model.layers.{layer}.mlp.gate.weight"
            router_shape = tensor_shape(checkpoint, router_name, torch.float32)
            categories["routers"] += matrix_macs(router_shape)
            shape_counts[("router", router_shape, str(torch.float32))] += 1
            for projection in ("gate_proj", "up_proj", "down_proj"):
                name = f"model.layers.{layer}.mlp.experts.0.{projection}.weight"
                shape = tensor_shape(checkpoint, name, torch.float8_e4m3fn)
                categories["selected_experts"] += 8 * matrix_macs(shape)
                shape_counts[(f"expert_{projection}", shape, str(torch.float8_e4m3fn))] += 1
        safety.checkpoint(f"layer_{layer}_matrix_shapes_authenticated")

    lm_shape = tensor_shape(checkpoint, "lm_head.weight", torch.bfloat16)
    categories["lm_head"] = matrix_macs(lm_shape)
    shape_counts[("lm_head", lm_shape, str(torch.bfloat16))] += 1
    expected_shapes = {
        ("qkv_proj", (14_848, 4_096), str(torch.float8_e4m3fn)): 39,
        ("qkv_proj", (13_568, 4_096), str(torch.float8_e4m3fn)): 9,
        ("o_proj", (4_096, 8_192), str(torch.bfloat16)): 48,
        ("dense_gate_proj", (16_384, 4_096), str(torch.float8_e4m3fn)): 1,
        ("dense_up_proj", (16_384, 4_096), str(torch.float8_e4m3fn)): 1,
        ("dense_down_proj", (4_096, 16_384), str(torch.float8_e4m3fn)): 1,
        ("router", (256, 4_096), str(torch.float32)): 47,
        ("expert_gate_proj", (2_048, 4_096), str(torch.float8_e4m3fn)): 47,
        ("expert_up_proj", (2_048, 4_096), str(torch.float8_e4m3fn)): 47,
        ("expert_down_proj", (4_096, 2_048), str(torch.float8_e4m3fn)): 47,
        ("lm_head", (152_576, 4_096), str(torch.bfloat16)): 1,
    }
    if dict(shape_counts) != expected_shapes:
        raise ValueError("PW-0127 checkpoint matrix topology mismatch")

    mandatory_macs = sum(categories.values())
    peak = peak_flops(2, 10, 3.60, 16)
    arithmetic = ceiling(peak, mandatory_macs)
    bandwidth = {
        "per_socket_official_maximum_bytes_per_second": 59.7e9,
        "impossible_dual_socket_bytes_per_second": 119.4e9,
        "selected_expert_bytes_per_ordinary_token": SELECTED_EXPERT_BYTES_PER_TOKEN,
        "expert_only_ordinary_token_tps_ceiling": (
            119.4e9 / SELECTED_EXPERT_BYTES_PER_TOKEN
        ),
    }
    bandwidth["headroom_over_10_tps"] = (
        bandwidth["expert_only_ordinary_token_tps_ceiling"] / 10.0 - 1.0
    )
    bandwidth["requires_wide_or_expert_major_measurement"] = (
        bandwidth["headroom_over_10_tps"] < 0.25
    )
    if arithmetic["impossible_maximum_tps"] >= 50:
        raise ValueError("PW-0127 expected CPU-only Prismwing-50 rejection absent")
    safety.release_checkpoint(
        "checkpoint_matrix_metadata_released",
        ["checkpoint tensor mappings", "matrix topology ledger"],
    )
    safety.checkpoint("final_service_health")
    report = {
        "schema_version": 1,
        "evidence_class": "pw0127_r720_cpu_arithmetic_ceiling",
        "revision": REVISION,
        "commit": commit,
        "checkpoint_verification_sha256": VERIFICATION_SHA256,
        "candidate": {
            "system_class": "Dell PowerEdge R720 CPU-only",
            "sockets": 2,
            "processor": "Intel Xeon E5-2680 v2",
            "cores_per_socket": 10,
            "granted_all_core_ghz": 3.60,
            "granted_sp_operations_per_cycle_per_core": 16,
            "memory_gib": 512,
            "candidate_is_owned_or_measured": False,
        },
        "matrix_shape_counts": [
            {"kind": key[0], "shape": list(key[1]), "dtype": key[2], "count": count}
            for key, count in sorted(shape_counts.items())
        ],
        "mandatory_macs_by_category": categories,
        "arithmetic_ceiling": arithmetic,
        "ordinary_token_bandwidth_ceiling": bandwidth,
        "gates_passed": False,
        "decision": "reject_cpu_only_dual_e5_2680v2_for_prismwing_50",
        "safety_snapshots": safety.evidence(),
        "complete_wall_ms": (time.perf_counter() - started) * 1000.0,
        "accepted_tokens": 0,
        "A": 0,
        "performance_claim": None,
        "omitted_work": [
            "embedding lookup",
            "RMSNorm",
            "RoPE",
            "attention score and value work",
            "softmax and KV traffic",
            "nonlinearities",
            "BF16 and FP8 conversion and scales",
            "route selection",
            "threading and NUMA exchange",
            "network, rollback, and sampling",
        ],
        "limitations": (
            "analytical CPU-only candidate-class ceiling, not an active BOM, owned "
            "machine, measured stage, executable runtime, accepted-token result, or TPS"
        ),
        "platform": platform.platform(),
    }
    atomic_write_new(output_path, canonical_json(report))
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--verification", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--commit", required=True)
    arguments = parser.parse_args()
    try:
        result = run(
            arguments.checkpoint,
            arguments.verification,
            arguments.output,
            arguments.commit,
        )
        print(json.dumps({"output": str(arguments.output), "decision": result["decision"]}))
        return 0
    except (OSError, RuntimeError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
        print(json.dumps({"error": str(error)}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
