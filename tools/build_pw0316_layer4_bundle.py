#!/usr/bin/env python3
"""Build the receipt-bound PW-0316 layer-4 four-K4/four-source bundle."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import time
from typing import Any

import numpy as np

try:
    from tools.analyze_pw0315_layer4_bank import replace_expert_outputs
    from tools.construct_pw0314_layer4_k4 import (
        CHECKPOINT_RECEIPT_SHA256,
        CHECKPOINT_REVISION,
        CORPUS_SHA256,
        HIDDEN,
        LAYER,
        load_capture,
        metric,
        reconstruct_route,
        selected_rows,
    )
    from tools.host_safety import HostSafetyMonitor, HostSafetyViolation
    from tools.openrouter_reference import atomic_write_new, canonical_json
    from tools.reproduce_pw0311_k4_expert import (
        TLUT_ARRAY_SHA256,
        _load_authority_modules,
        array_sha256,
        authority_paths,
        sha256_file,
        verify_clean_commit,
    )
except ModuleNotFoundError:
    from analyze_pw0315_layer4_bank import replace_expert_outputs
    from construct_pw0314_layer4_k4 import (
        CHECKPOINT_RECEIPT_SHA256,
        CHECKPOINT_REVISION,
        CORPUS_SHA256,
        HIDDEN,
        LAYER,
        load_capture,
        metric,
        reconstruct_route,
        selected_rows,
    )
    from host_safety import HostSafetyMonitor, HostSafetyViolation
    from openrouter_reference import atomic_write_new, canonical_json
    from reproduce_pw0311_k4_expert import (
        TLUT_ARRAY_SHA256,
        _load_authority_modules,
        array_sha256,
        authority_paths,
        sha256_file,
        verify_clean_commit,
    )


EXPERIMENT_ID = "PW-0316"
POSITION = 1
K4_EXPERTS = (96, 64, 232, 31)
SOURCE_EXPERTS = (88, 245, 223, 151)
ROUTE = K4_EXPERTS + SOURCE_EXPERTS
MAXIMUM_MIXED_ROW_RELATIVE_L2 = 0.01
PW0315_SUMMARY_SHA256 = "07b3d3793a6750a030eb5b7e12a0add1b603d48758a85e6f45b44504e404d0e8"
TLUT_FILE_SHA256 = "8c76b28d00a94d037c8699d823abefbea12ebfd0c9039a47098f5b21f9e54293"
ALIGNMENT = 16 * 1024
K4_ROLES = (
    "packed", "left_sign", "right_sign", "global_scale", "row_scale",
    "correction_left", "correction_right",
)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def align(stream: Any) -> int:
    offset = stream.tell()
    padding = (-offset) % ALIGNMENT
    if padding:
        stream.write(bytes(padding))
    return offset + padding


def append_file(stream: Any, path: Path, expected: dict[str, Any]) -> dict[str, Any]:
    if path.stat().st_size != int(expected["bytes"]) or sha256_file(path) != expected["sha256"]:
        raise ValueError(f"payload authority mismatch: {path}")
    offset = align(stream)
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            stream.write(chunk)
    return {
        "offset": offset,
        "bytes": int(expected["bytes"]),
        "sha256": expected["sha256"],
        "alignment": ALIGNMENT,
    }


def raw_tensor(checkpoint: Any, name: str) -> tuple[bytes, dict[str, Any], str]:
    shard, base, meta = checkpoint.metadata(name)
    start, end = map(int, meta["data_offsets"])
    payload = checkpoint._pread(shard, end - start, base + start)
    return payload, {"dtype": meta["dtype"], "shape": meta["shape"]}, shard


def replay_source_expert(
    panel: Any,
    moe_input: np.ndarray,
    positions: np.ndarray,
    exact: dict[str, np.ndarray],
    expected: np.ndarray,
) -> np.ndarray:
    """Replay the complete expert-major GEMM shape used by PW-0116.

    Accelerate may select a different accumulation path for a one-row GEMM.
    The BF16 boundary can expose that otherwise-small difference, so selecting
    one row before execution is not equivalent to selecting it afterward.
    """
    actual = panel.complete_outputs(
        np.asarray(moe_input[positions], dtype=np.float32), exact
    )["candidate_output_bf16_f32"]
    if actual.shape != expected.shape or not np.array_equal(actual, expected):
        raise ValueError(f"source expert-major replay mismatch: {metric(expected, actual)}")
    return actual


def mixed_row_qualified(
    route_metric: dict[str, float], final_metric: dict[str, float]
) -> bool:
    return (
        route_metric["relative_l2"] < MAXIMUM_MIXED_ROW_RELATIVE_L2
        and final_metric["relative_l2"] < MAXIMUM_MIXED_ROW_RELATIVE_L2
    )


def verify_install_for_sources(checkpoint_root: Path, receipt_path: Path, modules: dict[str, Any]) -> dict[str, Any]:
    if sha256_file(receipt_path) != CHECKPOINT_RECEIPT_SHA256:
        raise ValueError("checkpoint receipt mismatch")
    receipt = json.loads(receipt_path.read_text())
    if receipt.get("complete") is not True or receipt.get("revision") != CHECKPOINT_REVISION:
        raise ValueError("checkpoint receipt contract mismatch")
    by_path = {row["path"]: row for row in receipt["files"]}
    observations: dict[str, Any] = {}
    with modules["checkpoint"].Checkpoint(checkpoint_root) as checkpoint:
        for expert in SOURCE_EXPERTS:
            for projection in ("gate", "up", "down"):
                name = f"model.layers.{LAYER}.mlp.experts.{expert}.{projection}_proj.weight"
                for tensor_name in (name, name + "_scale_inv"):
                    shard, _, _ = checkpoint.metadata(tensor_name)
                    row = by_path.get(shard)
                    if row is None or row.get("status") != "verified":
                        raise ValueError(f"source shard absent from receipt: {shard}")
                    stat = (checkpoint_root / shard).stat()
                    if (
                        stat.st_size != int(row["bytes"])
                        or stat.st_ino != int(row["inode"])
                        or stat.st_mtime_ns != int(row["modified_ns"])
                    ):
                        raise ValueError(f"source shard identity mismatch: {shard}")
                    observations[shard] = {
                        "bytes": stat.st_size,
                        "inode": stat.st_ino,
                        "modified_ns": stat.st_mtime_ns,
                        "sha256_from_receipt": row["sha256"],
                    }
    return observations


def build(
    *,
    authority_root: Path,
    checkpoint_root: Path,
    checkpoint_receipt: Path,
    corpus_manifest: Path,
    pw0315_summary: Path,
    pw0315_evidence_root: Path,
    output: Path,
    repo: Path,
    commit: str,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(output)
    verify_clean_commit(repo.resolve(), commit)
    started = time.monotonic()
    safety = HostSafetyMonitor()
    paths = authority_paths(authority_root.resolve())
    modules = _load_authority_modules(paths)
    if sha256_file(corpus_manifest) != CORPUS_SHA256:
        raise ValueError("PW-0116 corpus mismatch")
    if sha256_file(pw0315_summary) != PW0315_SUMMARY_SHA256:
        raise ValueError("PW-0315 summary mismatch")
    summary = json.loads(pw0315_summary.read_text())
    shard_observations = verify_install_for_sources(
        checkpoint_root.resolve(), checkpoint_receipt.resolve(), modules
    )
    corpus = json.loads(corpus_manifest.read_text())
    layer_row = next(row for row in corpus["layers"] if int(row["layer"]) == LAYER)
    if tuple(layer_row["selected_experts_by_position"][POSITION]) != ROUTE:
        raise ValueError("PW-0316 route identity mismatch")
    weights = np.asarray(layer_row["route_weights_by_position"][POSITION], dtype=np.float32)
    corpus_root = corpus_manifest.parent
    moe_input = load_capture(corpus_root, layer_row, "moe_input")
    expert_down = load_capture(corpus_root, layer_row, "expert_down")
    source_routed = load_capture(corpus_root, layer_row, "routed_output")
    post_attention = load_capture(corpus_root, layer_row, "post_attention")
    source_final = load_capture(corpus_root, layer_row, "final")
    if not np.array_equal(reconstruct_route(expert_down, layer_row, modules["panel"].bf16), source_routed):
        raise ValueError("source route reconstruction mismatch")
    if not np.array_equal(modules["panel"].bf16(post_attention + source_routed), source_final):
        raise ValueError("source final reconstruction mismatch")
    safety.checkpoint("authorities_and_source_route_verified")

    output.mkdir(parents=True)
    source_root = output / "source-fixtures"
    source_root.mkdir()
    source_records: dict[int, dict[str, Any]] = {}
    with modules["checkpoint"].Checkpoint(checkpoint_root) as checkpoint:
        for expert in SOURCE_EXPERTS:
            positions, _, _, offsets = selected_rows(layer_row, expert)
            local = int(np.flatnonzero(positions == POSITION)[0])
            expected = expert_down[offsets]
            exact: dict[str, np.ndarray] = {}
            tensors: dict[str, Any] = {}
            expert_root = source_root / f"expert-{expert:03d}"
            expert_root.mkdir()
            for projection in ("gate", "up", "down"):
                name = f"model.layers.{LAYER}.mlp.experts.{expert}.{projection}_proj.weight"
                exact[projection] = checkpoint.read_dequantized_fp8(name)
                for suffix, role, filename in (
                    ("", f"{projection}_weight", f"{projection}.fp8"),
                    ("_scale_inv", f"{projection}_scales", f"{projection}.scales.f32le"),
                ):
                    payload, meta, shard = raw_tensor(checkpoint, name + suffix)
                    path = expert_root / filename
                    path.write_bytes(payload)
                    tensors[role] = {
                        "file": filename,
                        "bytes": len(payload),
                        "sha256": digest(payload),
                        "shard": shard,
                        **meta,
                    }
            actual = replay_source_expert(
                modules["panel"], moe_input, positions, exact, expected
            )
            selected_actual = actual[local : local + 1]
            fixture = {
                "schema_version": 1,
                "semantic": "source_fp8_exception_complete_expert_e3",
                "layer": LAYER,
                "expert": expert,
                "position": POSITION,
                "tensors": tensors,
                "expert_major_positions": positions.tolist(),
                "expert_major_output_sha256": array_sha256(actual),
                "output_sha256": array_sha256(selected_actual),
            }
            fixture_path = expert_root / "fixture.json"
            fixture_path.write_bytes(canonical_json(fixture))
            source_records[expert] = {"fixture": fixture, "fixture_path": fixture_path}
            safety.checkpoint(f"source_expert_{expert}_exported")

    candidate_down = expert_down.copy()
    k4_roots: dict[int, Path] = {}
    for expert in K4_EXPERTS:
        report_path = pw0315_evidence_root / f"expert-{expert:03d}-run-001" / "construction.json"
        expected_report_hash = summary["identities"][str(expert)]["report_sha256"][0]
        if sha256_file(report_path) != expected_report_hash:
            raise ValueError(f"PW-0315 report mismatch: {expert}")
        report = json.loads(report_path.read_text())
        if report["semantic"]["gates"]["pass"] is not True:
            raise ValueError(f"unqualified K4 identity: {expert}")
        root = report_path.parent / f"layer-{LAYER:02d}-expert-{expert:03d}"
        positions, _, _, _ = selected_rows(layer_row, expert)
        candidate = np.fromfile(root / "candidate-output.f32le", dtype="<f4").reshape(len(positions), HIDDEN)
        if array_sha256(candidate) != report["semantic"]["array_sha256"]["candidate_output_f32"]:
            raise ValueError(f"K4 candidate output mismatch: {expert}")
        candidate_down = replace_expert_outputs(candidate_down, layer_row, expert, candidate)
        k4_roots[expert] = root

    candidate_route = reconstruct_route(candidate_down, layer_row, modules["panel"].bf16)
    candidate_final = modules["panel"].bf16(post_attention + candidate_route)
    route_metric = metric(source_routed[POSITION : POSITION + 1], candidate_route[POSITION : POSITION + 1])
    final_metric = metric(source_final[POSITION : POSITION + 1], candidate_final[POSITION : POSITION + 1])
    if not mixed_row_qualified(route_metric, final_metric):
        source_fixture_sha256 = {
            str(expert): sha256_file(row["fixture_path"])
            for expert, row in sorted(source_records.items())
        }
        del (
            exact,
            candidate_down,
            candidate_route,
            candidate_final,
            moe_input,
            expert_down,
            source_routed,
            post_attention,
            source_final,
        )
        gc.collect()
        safety.release_checkpoint(
            "mixed_row_gate_rejected_buffers_released",
            ["source weights", "PW-0116 captures", "candidate route staging"],
        )
        safety.checkpoint("final_service_health")
        rejection = {
            "schema_version": 1,
            "experiment_id": EXPERIMENT_ID,
            "status": "rejected_mixed_row_semantic_gate",
            "commit": commit,
            "authority": {
                "checkpoint_receipt_sha256": CHECKPOINT_RECEIPT_SHA256,
                "corpus_sha256": CORPUS_SHA256,
                "pw0315_summary_sha256": PW0315_SUMMARY_SHA256,
            },
            "route": list(ROUTE),
            "k4_experts": list(K4_EXPERTS),
            "source_experts": list(SOURCE_EXPERTS),
            "source_fixture_sha256": source_fixture_sha256,
            "semantic": {
                "route_candidate_vs_source": route_metric,
                "final_candidate_vs_source": final_metric,
                "maximum_relative_l2_exclusive": MAXIMUM_MIXED_ROW_RELATIVE_L2,
                "pass": False,
            },
            "safety_snapshots": safety.evidence(),
            "host": {
                "machine": platform.machine(),
                "platform": platform.platform(),
                "total_memory_bytes": int(
                    os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
                ),
            },
            "complete_seconds": time.monotonic() - started,
            "peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
            "batch_size": 1,
            "concurrency": 1,
            "accepted_tokens": 0,
            "performance_claim": None,
            "decision": "kill_four_k4_four_source_mixed_transaction",
        }
        atomic_write_new(output / "rejection.json", canonical_json(rejection))
        raise ValueError(f"mixed row semantic gate failed: route={route_metric}, final={final_metric}")

    tlut_path = paths["reference_export"] / "tlut.f32le"
    if sha256_file(tlut_path) != TLUT_FILE_SHA256:
        raise ValueError("TLUT file mismatch")
    bundle_path = output / "layer04-position001.k4-source.bin"
    records: list[dict[str, Any]] = []
    with bundle_path.open("xb") as stream:
        tlut = append_file(stream, tlut_path, {"bytes": 4096, "sha256": TLUT_FILE_SHA256})
        for expert in K4_EXPERTS:
            record = {"expert": expert, "format": "qtip_k4_ldlq", "projections": {}}
            for projection in ("gate", "up", "down"):
                directory = k4_roots[expert] / projection
                manifest_path = directory / "manifest.json"
                manifest = json.loads(manifest_path.read_text())
                payloads = {
                    role: append_file(stream, directory / manifest["files"][role]["file"], manifest["files"][role])
                    for role in K4_ROLES
                }
                record["projections"][projection] = {
                    "rows": manifest["rows"],
                    "columns": manifest["columns"],
                    "rank": manifest["rank"],
                    "payloads": payloads,
                    "source_manifest_sha256": sha256_file(manifest_path),
                }
            records.append(record)
        for expert in SOURCE_EXPERTS:
            row = source_records[expert]
            payloads = {}
            for role, source in row["fixture"]["tensors"].items():
                payload = append_file(stream, row["fixture_path"].parent / source["file"], source)
                payload.update({"dtype": source["dtype"], "shape": source["shape"]})
                payloads[role] = payload
            records.append({
                "expert": expert,
                "format": "source_fp8_e4m3_block128",
                "payloads": payloads,
                "source_fixture_sha256": sha256_file(row["fixture_path"]),
            })
        logical_end = stream.tell()
        align(stream)

    spec = {
        "experiment_id": EXPERIMENT_ID,
        "layer": LAYER,
        "position": POSITION,
        "k4_experts": list(K4_EXPERTS),
        "source_experts": list(SOURCE_EXPERTS),
        "checkpoint_receipt_sha256": CHECKPOINT_RECEIPT_SHA256,
        "corpus_sha256": CORPUS_SHA256,
        "pw0315_summary_sha256": PW0315_SUMMARY_SHA256,
        "tlut_array_sha256": TLUT_ARRAY_SHA256,
    }
    spec_path = output / "build-spec.json"
    spec_path.write_bytes(canonical_json(spec))
    manifest = {
        "schema_version": 2,
        "experiment_id": EXPERIMENT_ID,
        "semantic": "prismwing_mixed_k4_source_layer_bundle_v2",
        "layer": LAYER,
        "alignment_bytes": ALIGNMENT,
        "bundle_bytes": bundle_path.stat().st_size,
        "logical_end_bytes": logical_end,
        "bundle_sha256": sha256_file(bundle_path),
        "tlut": tlut,
        "records": sorted(records, key=lambda row: row["expert"]),
        "k4_experts": list(K4_EXPERTS),
        "source_experts": list(SOURCE_EXPERTS),
        "route_authority": {
            "experts": list(ROUTE),
            "weights": weights.tolist(),
            "candidate_relative_l2": route_metric["relative_l2"],
            "maximum_relative_l2": 0.01,
            "correctness_qualified": True,
        },
        "identity_policy": "selected expert IDs must match bundle records exactly; no substitution",
        "spec_sha256": sha256_file(spec_path),
        "claims_excluded": ["arbitrary routes", "complete bank", "endpoint TPS", "Prismwing completion"],
    }
    manifest_path = output / "layer04-position001.k4-source.manifest.json"
    manifest_path.write_bytes(canonical_json(manifest))
    fixture = {
        "schema_version": 1,
        "semantic": "pw0316_layer4_four_k4_four_source_fixture",
        "layer": LAYER,
        "position": POSITION,
        "input_f32": np.asarray(moe_input[POSITION], dtype=np.float32).tolist(),
        "native_router_experts": list(ROUTE),
        "native_router_weights": weights.tolist(),
        "source_routed_f32": np.asarray(source_routed[POSITION], dtype=np.float32).tolist(),
        "candidate_routed_f32": np.asarray(candidate_route[POSITION], dtype=np.float32).tolist(),
        "source_final_f32": np.asarray(source_final[POSITION], dtype=np.float32).tolist(),
        "candidate_final_f32": np.asarray(candidate_final[POSITION], dtype=np.float32).tolist(),
        "route_candidate_vs_source": route_metric,
        "final_candidate_vs_source": final_metric,
    }
    fixture_path = output / "layer04-position001.fixture.json"
    fixture_path.write_bytes(canonical_json(fixture))
    safety.release_checkpoint("builder_buffers_released", ["source weights", "PW-0116 captures", "bundle staging"])
    safety.checkpoint("final_service_health")
    result = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "status": "layer4_four_four_bundle_ready",
        "commit": commit,
        "authority": {
            "shards": shard_observations,
            "checkpoint_receipt_sha256": CHECKPOINT_RECEIPT_SHA256,
            "corpus_sha256": CORPUS_SHA256,
            "pw0315_summary_sha256": PW0315_SUMMARY_SHA256,
        },
        "bundle": {"file": bundle_path.name, "bytes": bundle_path.stat().st_size, "sha256": sha256_file(bundle_path)},
        "manifest": {"file": manifest_path.name, "sha256": sha256_file(manifest_path)},
        "fixture": {"file": fixture_path.name, "sha256": sha256_file(fixture_path)},
        "semantic": {"route_candidate_vs_source": route_metric, "final_candidate_vs_source": final_metric},
        "complete_seconds": time.monotonic() - started,
        "peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "safety_snapshots": safety.evidence(),
        "host": {"machine": platform.machine(), "platform": platform.platform(), "total_memory_bytes": int(os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"))},
        "batch_size": 1,
        "concurrency": 1,
        "accepted_tokens": 0,
        "performance_claim": None,
    }
    atomic_write_new(output / "build.json", canonical_json(result))
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--authority-root", required=True, type=Path)
    parser.add_argument("--checkpoint-root", required=True, type=Path)
    parser.add_argument("--checkpoint-receipt", required=True, type=Path)
    parser.add_argument("--corpus-manifest", required=True, type=Path)
    parser.add_argument("--pw0315-summary", required=True, type=Path)
    parser.add_argument("--pw0315-evidence-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--commit", required=True)
    args = parser.parse_args()
    try:
        result = build(**vars(args))
        print(json.dumps({"output": str(args.output), "status": result["status"]}))
        return 0
    except (FileExistsError, HostSafetyViolation, KeyError, OSError, RuntimeError, TypeError, ValueError, json.JSONDecodeError) as error:
        print(json.dumps({"error": str(error)}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
