#!/usr/bin/env python3
"""Run PW-0207's byte-accurate corrected-route residency falsifier."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import subprocess
from typing import Any

from safetensors import safe_open

try:
    from tools.openrouter_reference import atomic_write_new, canonical_json
except ModuleNotFoundError:
    from openrouter_reference import atomic_write_new, canonical_json


TRACE_SHA256 = "e5c0b93d039ec8d8c6b1f7a0087ec3991ba55df2a1cee7d388f08d6e668d830b"
TRACE_PROGRESS_SHA256 = "f637223240b529058a3ab314d71cc412b6f2a2c301c16aca8d9de14325dd7cd3"
REFERENCE_SHA256 = "c87f2a12809c1accc52fc5d5092765ad4cb90cb9d1fa0a2f916a2ccb6d23e1b9"
VERIFICATION_SHA256 = "9ddc8a99755f04ae2ea3c2484f6dd022d3f3a681b5a72c915ee4de833dbb0d03"
INDEX_SHA256 = "f2e1774c9acf9a62338b68c144e6fc7a66495e59f2e64b3078c1b7ef5a196816"
REVISION = "63651580ca774f8504f676040460aed3e1244ac1"
TRACE_COMMIT = "871db1ff3c3f654d9184de22c958c877e1402006"
CAPACITY_BYTES = 12 * 1024**3
EXPERT_BYTES = 25_171_968
EMBEDDING_ROW_BYTES = 4096 * 2
ITEM_BYTES = {"F8_E4M3": 1, "BF16": 2, "F32": 4}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fixed_tensor_names(weight_map: dict[str, str]) -> list[str]:
    names: list[str] = []
    for layer in range(48):
        prefix = f"model.layers.{layer}"
        names.extend(
            [
                f"{prefix}.input_layernorm.weight",
                f"{prefix}.self_attn.qkv_proj.weight",
                f"{prefix}.self_attn.qkv_proj.weight_scale_inv",
                f"{prefix}.self_attn.o_proj.weight",
                f"{prefix}.post_attention_layernorm.weight",
            ]
        )
        sink = f"{prefix}.self_attn.attention_sink_bias"
        if sink in weight_map:
            names.append(sink)
        if layer == 0:
            for projection in ("gate_proj", "up_proj", "down_proj"):
                names.extend(
                    [
                        f"{prefix}.mlp.{projection}.weight",
                        f"{prefix}.mlp.{projection}.weight_scale_inv",
                    ]
                )
        else:
            names.extend(
                [
                    f"{prefix}.mlp.gate.weight",
                    f"{prefix}.mlp.gate.e_score_correction_bias",
                ]
            )
    names.extend(["model.norm.weight", "lm_head.weight"])
    if len(names) != len(set(names)) or any(name not in weight_map for name in names):
        raise ValueError("fixed tensor authority is incomplete or duplicated")
    return names


def expert_tensor_names(layer: int, expert: int) -> list[str]:
    prefix = f"model.layers.{layer}.mlp.experts.{expert}"
    return [
        f"{prefix}.{projection}.{suffix}"
        for projection in ("gate_proj", "up_proj", "down_proj")
        for suffix in ("weight", "weight_scale_inv")
    ]


def unique_trace_experts(trace: dict[str, Any]) -> list[int]:
    rows = trace.get("selected_experts_by_position")
    if (
        trace.get("layer") == 0
        or not isinstance(rows, list)
        or not rows
        or any(
            not isinstance(row, list)
            or len(row) != 8
            or len(set(row)) != 8
            or any(not isinstance(expert, int) or not 0 <= expert < 256 for expert in row)
            for row in rows
        )
    ):
        raise ValueError("routed trace expert identity mismatch")
    return sorted({expert for row in rows for expert in row})


def solve_attributed_rates(
    proposal_shared_bytes: int,
    proposal_expert_bytes: int,
    verification_shared_bytes: int,
    verification_expert_bytes: int,
    proposal_wall_ms: float,
    verification_wall_ms: float,
) -> tuple[float, float]:
    determinant = (
        proposal_shared_bytes * verification_expert_bytes
        - verification_shared_bytes * proposal_expert_bytes
    )
    if determinant == 0:
        raise ValueError("attribution equations are singular")
    shared = (
        proposal_wall_ms * verification_expert_bytes
        - verification_wall_ms * proposal_expert_bytes
    ) / determinant
    expert = (
        proposal_shared_bytes * verification_wall_ms
        - verification_shared_bytes * proposal_wall_ms
    ) / determinant
    if not math.isfinite(shared) or not math.isfinite(expert) or shared <= 0 or expert <= 0:
        raise ValueError("attributed acquisition rates are non-positive")
    return shared, expert


def select_static_residents(objects: list[dict[str, Any]], capacity: int) -> list[dict[str, Any]]:
    if capacity <= 0 or any(row["bytes"] <= 0 for row in objects):
        raise ValueError("residency capacity or object bytes are invalid")
    ranked = sorted(
        objects,
        key=lambda row: (
            -row["avoided_stall_ms_per_resident_byte"],
            -row["avoided_stall_ms"],
            row["identity"],
        ),
    )
    selected: list[dict[str, Any]] = []
    used = 0
    for row in ranked:
        if row["avoided_logical_read_bytes"] <= 0 or used + row["bytes"] > capacity:
            continue
        selected.append(row)
        used += row["bytes"]
    return selected


def authenticate_inputs(
    trace_path: Path,
    trace_progress_path: Path,
    reference_path: Path,
    verification_path: Path,
    checkpoint: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, str], dict[str, Any]]:
    expected = {
        trace_path: TRACE_SHA256,
        trace_progress_path: TRACE_PROGRESS_SHA256,
        reference_path: REFERENCE_SHA256,
        verification_path: VERIFICATION_SHA256,
        checkpoint / "model.safetensors.index.json": INDEX_SHA256,
    }
    for path, digest in expected.items():
        if sha256_file(path) != digest:
            raise ValueError(f"input hash mismatch: {path}")
    trace = json.loads(trace_path.read_text())
    reference = json.loads(reference_path.read_text())
    verification = json.loads(verification_path.read_text())
    index = json.loads((checkpoint / "model.safetensors.index.json").read_text())
    if (
        trace.get("schema_version") != 3
        or trace.get("route_trace_captured") is not True
        or trace.get("revision") != REVISION
        or trace.get("commit") != TRACE_COMMIT
        or trace.get("git_dirty") is not False
        or trace.get("progress_sha256") != TRACE_PROGRESS_SHA256
        or trace.get("accepted_tokens") != 8
        or len(trace.get("transactions", [])) != 1
        or trace.get("peak_resident_bytes", 0) > 8 * 1024**3
        or any(
            row.get("swap_growth_bytes") != 0 or row.get("new_throttled_pages") != 0
            for row in trace.get("safety_snapshots", [])
        )
    ):
        raise ValueError("PW-0207 trace semantic or safety identity mismatch")
    transaction = trace["transactions"][0]
    reference_transaction = reference.get("transactions", [None])[0]
    comparable = (
        "proposal_token_ids",
        "posterior_token_ids",
        "verifier_authorized_token_ids",
        "emitted_token_ids",
        "verifier_retained_proposal_rows",
        "retained_proposal_rows",
        "proposal_converged",
        "U",
    )
    if (
        reference.get("revision") != REVISION
        or any(transaction.get(key) != reference_transaction.get(key) for key in comparable)
        or len(transaction.get("proposal_layer_traces", [])) != 7
        or any(len(step) != 48 for step in transaction["proposal_layer_traces"])
        or len(transaction.get("verification_layer_traces", [])) != 48
    ):
        raise ValueError("route trace does not reproduce PW-0205 run 009 transaction zero")
    files = {row["path"]: row for row in verification.get("files", [])}
    if (
        verification.get("complete") is not True
        or verification.get("revision") != REVISION
        or files.get("model.safetensors.index.json", {}).get("sha256") != INDEX_SHA256
        or not isinstance(index.get("weight_map"), dict)
    ):
        raise ValueError("checkpoint verification or tensor index identity mismatch")
    shard_hashes = {
        path: row["sha256"]
        for path, row in files.items()
        if path.endswith(".safetensors") and row.get("status") == "verified"
    }
    return trace, transaction, verification, shard_hashes, index


def tensor_metadata(
    checkpoint: Path,
    weight_map: dict[str, str],
    shard_hashes: dict[str, str],
    names: set[str],
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for name in names:
        shard = weight_map.get(name)
        if shard is None or shard not in shard_hashes:
            raise ValueError(f"tensor lacks verified backing shard: {name}")
        grouped[shard].append(name)
    result: dict[str, dict[str, Any]] = {}
    for shard, shard_names in grouped.items():
        with safe_open(checkpoint / shard, framework="pt", device="cpu") as source:
            for name in shard_names:
                view = source.get_slice(name)
                dtype = view.get_dtype()
                shape = view.get_shape()
                if dtype not in ITEM_BYTES or any(value <= 0 for value in shape):
                    raise ValueError(f"unsupported tensor metadata: {name}")
                result[name] = {
                    "tensor": name,
                    "dtype": dtype,
                    "shape": shape,
                    "bytes": math.prod(shape) * ITEM_BYTES[dtype],
                    "backing_file": shard,
                    "backing_file_sha256": shard_hashes[shard],
                }
    return result


def analyze(
    trace_path: Path,
    trace_progress_path: Path,
    reference_path: Path,
    verification_path: Path,
    checkpoint: Path,
) -> dict[str, Any]:
    trace, transaction, _verification, shard_hashes, index = authenticate_inputs(
        trace_path, trace_progress_path, reference_path, verification_path, checkpoint
    )
    weight_map = index["weight_map"]
    fixed_names = fixed_tensor_names(weight_map)
    used_experts: set[tuple[int, int]] = set()
    for step in transaction["proposal_layer_traces"]:
        for layer_trace in step[1:]:
            used_experts.update(
                (layer_trace["layer"], expert)
                for expert in unique_trace_experts(layer_trace)
            )
    for layer_trace in transaction["verification_layer_traces"][1:]:
        used_experts.update(
            (layer_trace["layer"], expert)
            for expert in unique_trace_experts(layer_trace)
        )
    tensor_names = set(fixed_names)
    tensor_names.add("model.embed_tokens.weight")
    for layer, expert in used_experts:
        tensor_names.update(expert_tensor_names(layer, expert))
    metadata = tensor_metadata(checkpoint, weight_map, shard_hashes, tensor_names)

    object_authority: dict[str, dict[str, Any]] = {}
    for name in fixed_names:
        row = metadata[name]
        identity = f"tensor:{name}"
        object_authority[identity] = {
            "identity": identity,
            "category": "shared_spine",
            "bytes": row["bytes"],
            "tensor_metadata_sha256": hashlib.sha256(canonical_json(row)).hexdigest(),
            "tensors": [row],
        }
    for layer, expert in used_experts:
        rows = [metadata[name] for name in expert_tensor_names(layer, expert)]
        identity = f"expert:{layer}:{expert}"
        size = sum(row["bytes"] for row in rows)
        if size != EXPERT_BYTES:
            raise ValueError("expert bundle byte identity changed")
        object_authority[identity] = {
            "identity": identity,
            "category": "routed_expert",
            "layer": layer,
            "expert": expert,
            "bytes": size,
            "tensor_metadata_sha256": hashlib.sha256(canonical_json(rows)).hexdigest(),
            "tensors": rows,
        }
    for token in set(transaction["proposal_token_ids"]):
        source = metadata.get("model.embed_tokens.weight")
        if source is None:
            source = tensor_metadata(
                checkpoint,
                weight_map,
                shard_hashes,
                {"model.embed_tokens.weight"},
            )["model.embed_tokens.weight"]
        identity = f"embedding_row:{token}"
        row_authority = {
            "tensor": "model.embed_tokens.weight",
            "row": token,
            "dtype": source["dtype"],
            "shape": [4096],
            "bytes": EMBEDDING_ROW_BYTES,
            "backing_file": source["backing_file"],
            "backing_file_sha256": source["backing_file_sha256"],
        }
        object_authority[identity] = {
            "identity": identity,
            "category": "shared_spine",
            "bytes": EMBEDDING_ROW_BYTES,
            "tensor_metadata_sha256": hashlib.sha256(canonical_json(row_authority)).hexdigest(),
            "tensors": [row_authority],
        }

    accesses: list[str] = []
    access_phases: list[str] = []

    def append_call(tokens: list[int], traces: list[dict[str, Any]], phase: str) -> None:
        for token in tokens:
            accesses.append(f"embedding_row:{token}")
            access_phases.append(phase)
        for layer, layer_trace in enumerate(traces):
            prefix = f"model.layers.{layer}"
            layer_names = [name for name in fixed_names if name.startswith(prefix + ".")]
            for name in layer_names:
                accesses.append(f"tensor:{name}")
                access_phases.append(phase)
            if layer:
                for expert in unique_trace_experts(layer_trace):
                    accesses.append(f"expert:{layer}:{expert}")
                    access_phases.append(phase)
        for name in ("model.norm.weight", "lm_head.weight"):
            accesses.append(f"tensor:{name}")
            access_phases.append(phase)

    proposal = transaction["proposal_token_ids"]
    for index_value, traces in enumerate(transaction["proposal_layer_traces"]):
        append_call([proposal[index_value]], traces, "proposal")
    append_call(proposal, transaction["verification_layer_traces"], "verification")

    counts = Counter(accesses)
    positions: dict[str, list[int]] = defaultdict(list)
    for position, identity in enumerate(accesses):
        positions[identity].append(position)
    logical_by_phase = Counter()
    expert_by_phase = Counter()
    for identity, phase in zip(accesses, access_phases, strict=True):
        size = object_authority[identity]["bytes"]
        logical_by_phase[phase] += size
        if object_authority[identity]["category"] == "routed_expert":
            expert_by_phase[phase] += size
    if sum(logical_by_phase.values()) != transaction["logical_source_bytes"]:
        raise ValueError("object access bytes do not close to endpoint logical ledger")
    shared_by_phase = {
        phase: logical_by_phase[phase] - expert_by_phase[phase]
        for phase in ("proposal", "verification")
    }
    shared_rate, expert_rate = solve_attributed_rates(
        shared_by_phase["proposal"],
        expert_by_phase["proposal"],
        shared_by_phase["verification"],
        expert_by_phase["verification"],
        transaction["proposal_wall_ms"],
        transaction["verification_wall_ms"],
    )
    rates = {"shared_spine": shared_rate, "routed_expert": expert_rate}
    candidates = []
    for identity, authority in object_authority.items():
        count = counts[identity]
        avoided = max(0, count - 1) * authority["bytes"]
        reuse = [right - left for left, right in zip(positions[identity], positions[identity][1:])]
        stall = avoided * rates[authority["category"]]
        candidates.append(
            {
                **authority,
                "access_count": count,
                "reuse_distance_accesses": reuse,
                "expected_avoided_reads": max(0, count - 1),
                "avoided_logical_read_bytes": avoided,
                "avoided_stall_ms": stall,
                "avoided_stall_ms_per_resident_byte": stall / authority["bytes"],
                "lifetime": "one_repeated_decoder_transaction",
            }
        )
    selected = select_static_residents(candidates, CAPACITY_BYTES)
    selected_bytes = sum(row["bytes"] for row in selected)
    avoided_logical = sum(row["avoided_logical_read_bytes"] for row in selected)
    avoided_wall = sum(row["avoided_stall_ms"] for row in selected)
    control_logical = transaction["logical_source_bytes"]
    control_physical = transaction["process_disk_bytes_read"]
    physical_scale = control_physical / control_logical
    candidate_logical = control_logical - avoided_logical
    candidate_physical = candidate_logical * physical_scale
    control_wall = transaction["proposal_wall_ms"] + transaction["verification_wall_ms"]
    candidate_wall = control_wall - avoided_wall
    if candidate_logical <= 0 or candidate_physical <= 0 or candidate_wall <= 0:
        raise ValueError("residency prediction removed more than the measured control")
    for eviction_order, row in enumerate(reversed(selected), start=1):
        row["warning_eviction_order"] = eviction_order
    bytes_reduction = control_physical / candidate_physical
    wall_speedup = control_wall / candidate_wall
    gate = bytes_reduction >= 4.0 or wall_speedup >= 2.0
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    return {
        "schema_version": 1,
        "evidence_class": "pw0207_pressure_elastic_offline_residency_falsifier",
        "status": "passed",
        "revision": REVISION,
        "git_commit": commit,
        "git_dirty": dirty,
        "identities": {
            "route_trace_sha256": TRACE_SHA256,
            "route_trace_progress_sha256": TRACE_PROGRESS_SHA256,
            "pw0205_run009_report_sha256": REFERENCE_SHA256,
            "checkpoint_verification_sha256": VERIFICATION_SHA256,
            "tensor_index_sha256": INDEX_SHA256,
        },
        "trace_scope": "PW-0205 run-009-identical first repeated decoder transaction",
        "batch_size": 1,
        "concurrency": 1,
        "accepted_tokens": len(transaction["emitted_token_ids"]),
        "A": len(transaction["emitted_token_ids"]),
        "U": transaction["U"],
        "accesses": len(accesses),
        "unique_objects": len(object_authority),
        "control": {
            "logical_source_bytes": control_logical,
            "physical_read_bytes": control_physical,
            "proposal_wall_ms": transaction["proposal_wall_ms"],
            "verification_wall_ms": transaction["verification_wall_ms"],
            "attributed_transaction_wall_ms": control_wall,
        },
        "attribution": {
            "shared_spine_ms_per_byte": shared_rate,
            "routed_expert_ms_per_byte": expert_rate,
            "shared_spine_ns_per_byte": shared_rate * 1_000_000,
            "routed_expert_ns_per_byte": expert_rate * 1_000_000,
            "proposal_shared_bytes": shared_by_phase["proposal"],
            "proposal_expert_bytes": expert_by_phase["proposal"],
            "verification_shared_bytes": shared_by_phase["verification"],
            "verification_expert_bytes": expert_by_phase["verification"],
            "method": "positive two-equation solution from measured proposal/verification walls and exact shared/expert source bytes",
        },
        "residency_manifest": {
            "capacity_bytes": CAPACITY_BYTES,
            "selected_bytes": selected_bytes,
            "unallocated_bytes": CAPACITY_BYTES - selected_bytes,
            "persistent_lifetime": "one repeated decoder transaction",
            "warning_eviction_order": "ascending warning_eviction_order; lowest predicted stall avoided per resident byte first",
            "critical_pressure_action": "stop; no allocation is performed by this offline experiment",
            "objects": selected,
        },
        "prediction": {
            "avoided_logical_read_bytes": avoided_logical,
            "candidate_logical_source_bytes": candidate_logical,
            "candidate_physical_read_bytes": candidate_physical,
            "physical_read_reduction_factor": bytes_reduction,
            "avoided_attributed_wall_ms": avoided_wall,
            "candidate_attributed_wall_ms": candidate_wall,
            "attributed_wall_speedup": wall_speedup,
        },
        "gates": {
            "four_x_physical_read_reduction": bytes_reduction >= 4.0,
            "two_x_attributed_acquisition_wall_speedup": wall_speedup >= 2.0,
            "implementation_authorized": gate,
        },
        "decision": (
            "authorize_pressure_observer_and_one_transaction_residency_implementation"
            if gate
            else "kill_high_residency_implementation_on_offline_bound"
        ),
        "limitations": "single corrected first transaction; static noncausal selection; predicted attributed acquisition wall, not endpoint TPS; no high-residency allocation or pressure event is performed",
        "performance_claim": None,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", required=True, type=Path)
    parser.add_argument("--trace-progress", required=True, type=Path)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--verification", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    try:
        result = analyze(
            arguments.trace,
            arguments.trace_progress,
            arguments.reference,
            arguments.verification,
            arguments.checkpoint,
        )
        atomic_write_new(arguments.output, canonical_json(result))
        print(
            json.dumps(
                {
                    "output": str(arguments.output),
                    "physical_read_reduction_factor": result["prediction"][
                        "physical_read_reduction_factor"
                    ],
                    "attributed_wall_speedup": result["prediction"][
                        "attributed_wall_speedup"
                    ],
                    "implementation_authorized": result["gates"][
                        "implementation_authorized"
                    ],
                }
            )
        )
        return 0
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as error:
        print(json.dumps({"error": str(error)}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
