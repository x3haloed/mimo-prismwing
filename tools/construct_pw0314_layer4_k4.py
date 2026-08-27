#!/usr/bin/env python3
"""Construct and gate layer-4 expert 64 from the verified full checkpoint."""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import platform
import resource
import time
from typing import Any

import numpy as np

try:
    from tools.construct_pw0313_m1_native_k4_expert import (
        MAXIMUM_DECODE_RELATIVE_L2,
        REVISION,
        deterministic_tree_manifest,
        metric,
    )
    from tools.host_safety import HostSafetyMonitor, HostSafetyViolation
    from tools.openrouter_reference import atomic_write_new, canonical_json
    from tools.reproduce_pw0311_k4_expert import (
        ANCHOR_MANIFEST_SHA256,
        NAMES,
        PANEL_CONTRACT_SHA256,
        PANEL_EXPORT_SHA256,
        QTIP_COMMIT,
        TLUT_ARRAY_SHA256,
        _load_authority_modules,
        array_sha256,
        authority_paths,
        sha256_file,
        verify_clean_commit,
    )
except ModuleNotFoundError:
    from construct_pw0313_m1_native_k4_expert import (
        MAXIMUM_DECODE_RELATIVE_L2,
        REVISION,
        deterministic_tree_manifest,
        metric,
    )
    from host_safety import HostSafetyMonitor, HostSafetyViolation
    from openrouter_reference import atomic_write_new, canonical_json
    from reproduce_pw0311_k4_expert import (
        ANCHOR_MANIFEST_SHA256,
        NAMES,
        PANEL_CONTRACT_SHA256,
        PANEL_EXPORT_SHA256,
        QTIP_COMMIT,
        TLUT_ARRAY_SHA256,
        _load_authority_modules,
        array_sha256,
        authority_paths,
        sha256_file,
        verify_clean_commit,
    )


EXPERIMENT_ID = "PW-0314"
CHECKPOINT_REVISION = "63651580ca774f8504f676040460aed3e1244ac1"
CHECKPOINT_RECEIPT_SHA256 = "9ddc8a99755f04ae2ea3c2484f6dd022d3f3a681b5a72c915ee4de833dbb0d03"
CHECKPOINT_INDEX_SHA256 = "f2e1774c9acf9a62338b68c144e6fc7a66495e59f2e64b3078c1b7ef5a196816"
SOURCE_SHARD = "model_pp0_ep2_shard0.safetensors"
SOURCE_SHARD_SHA256 = "70639d2d3ad4bd80a3b3843632e17a5089baa3b2ac5565e571fb5ad7bafb0be0"
CORPUS_SHA256 = "b9df976876d63c1ffbbe0c70507aea8b939a749ce5b1db27cbca0b5d82cf802e"
LAYER = 4
EXPERT = 64
ROWS = 224
HIDDEN = 4096
TOP_K = 8
PARTITIONS = {"train": (0, 112), "validation": (112, 168), "pilot_holdout": (168, 224)}
MAXIMUM_ROUTE_RELATIVE_L2 = 0.01
MAXIMUM_ROW_RELATIVE_L2 = 0.05
EXPECTED_PROJECTION_HASHES: dict[str, dict[str, str]] | None = None
EXPECTED_PLACEMENT_COUNT: int | None = 181


def verify_full_checkpoint_install(
    checkpoint_root: Path,
    receipt_path: Path,
    layer: int = LAYER,
    expert: int = EXPERT,
) -> dict[str, Any]:
    if sha256_file(receipt_path) != CHECKPOINT_RECEIPT_SHA256:
        raise ValueError("full-checkpoint receipt mismatch")
    receipt = json.loads(receipt_path.read_text())
    if (
        receipt.get("schema_version") != 1
        or receipt.get("evidence_class") != "local_checkpoint_lock_verification"
        or receipt.get("complete") is not True
        or receipt.get("missing_files") != []
        or receipt.get("revision") != CHECKPOINT_REVISION
    ):
        raise ValueError("full-checkpoint receipt contract mismatch")
    by_path = {row["path"]: row for row in receipt["files"]}
    index_path = checkpoint_root / "model.safetensors.index.json"
    if sha256_file(index_path) != CHECKPOINT_INDEX_SHA256:
        raise ValueError("full-checkpoint index mismatch")
    index = json.loads(index_path.read_text())
    tensor_names = [
        f"model.layers.{layer}.mlp.experts.{expert}.{name}_proj.weight{suffix}"
        for name in NAMES
        for suffix in ("", "_scale_inv")
    ]
    mapped = {index["weight_map"].get(name) for name in tensor_names}
    if mapped != {SOURCE_SHARD}:
        raise ValueError("layer-4 expert tensor-to-shard mapping mismatch")
    observations = {}
    for relative, expected_hash in (
        ("model.safetensors.index.json", CHECKPOINT_INDEX_SHA256),
        (SOURCE_SHARD, SOURCE_SHARD_SHA256),
    ):
        if relative not in by_path or by_path[relative].get("sha256") != expected_hash:
            raise ValueError(f"receipt file authority mismatch: {relative}")
        row = by_path[relative]
        artifact = checkpoint_root / relative
        observed = artifact.stat()
        if (
            row.get("status") != "verified"
            or observed.st_size != int(row["bytes"])
            or observed.st_ino != int(row["inode"])
            or observed.st_mtime_ns != int(row["modified_ns"])
        ):
            raise ValueError(f"installed file identity mismatch: {relative}")
        observations[relative] = {
            "bytes": observed.st_size,
            "inode": observed.st_ino,
            "modified_ns": observed.st_mtime_ns,
            "observed_device": observed.st_dev,
            "receipt_device": int(row["device"]),
            "sha256_from_receipt": row["sha256"],
            "content_rescanned": relative == "model.safetensors.index.json",
        }
    return {
        "revision": CHECKPOINT_REVISION,
        "receipt_sha256": CHECKPOINT_RECEIPT_SHA256,
        "lock_sha256": receipt["lock_sha256"],
        "index_sha256": CHECKPOINT_INDEX_SHA256,
        "source_shard_sha256_from_receipt": SOURCE_SHARD_SHA256,
        "tensor_names": tensor_names,
        "observations": observations,
    }


def load_capture(corpus_root: Path, row: dict[str, Any], name: str) -> np.ndarray:
    capture = row["captures"][name]
    path = corpus_root / capture["file"]
    if sha256_file(path) != capture["sha256"] or path.stat().st_size != int(capture["bytes"]):
        raise ValueError(f"PW-0116 capture mismatch: {name}")
    return np.fromfile(path, dtype="<f4").reshape(capture["shape"])


def selected_rows(
    layer_row: dict[str, Any], expert: int = EXPERT
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    positions = []
    slots = []
    weights = []
    source_offsets = []
    offset = 0
    matches = [row for row in layer_row["expert_schedule"] if int(row["expert"]) == expert]
    if len(matches) != 1:
        raise ValueError("expert-major schedule identity mismatch")
    for schedule in layer_row["expert_schedule"]:
        if int(schedule["expert"]) == expert:
            for local, position in enumerate(schedule["positions"]):
                route = layer_row["selected_experts_by_position"][position]
                slot = route.index(expert)
                positions.append(position)
                slots.append(slot)
                weights.append(layer_row["route_weights_by_position"][position][slot])
                source_offsets.append(offset + local)
        offset += len(schedule["positions"])
    return (
        np.asarray(positions, dtype=np.int64),
        np.asarray(slots, dtype=np.int64),
        np.asarray(weights, dtype=np.float32),
        np.asarray(source_offsets, dtype=np.int64),
    )


def reconstruct_route(
    expert_outputs: np.ndarray,
    layer_row: dict[str, Any],
    bf16: Any,
) -> np.ndarray:
    outputs = np.asarray(expert_outputs, dtype=np.float32)
    if outputs.shape != (ROWS * TOP_K, HIDDEN):
        raise ValueError("expert-major output shape mismatch")
    selected = layer_row["selected_experts_by_position"]
    weights = layer_row["route_weights_by_position"]
    schedule = layer_row["expert_schedule"]
    if (
        len(selected) != ROWS
        or len(weights) != ROWS
        or sum(len(row["positions"]) for row in schedule) != ROWS * TOP_K
    ):
        raise ValueError("expert-major route authority mismatch")
    routed = np.zeros((ROWS, HIDDEN), dtype=np.float32)
    offset = 0
    for row in schedule:
        expert = int(row["expert"])
        for local, position in enumerate(row["positions"]):
            slot = selected[position].index(expert)
            routed[position] += outputs[offset + local] * np.float32(weights[position][slot])
        offset += len(row["positions"])
    return bf16(routed)


def partition_metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    left = np.asarray(reference, dtype=np.float32)
    right = np.asarray(candidate, dtype=np.float32)
    if left.shape != right.shape or left.ndim != 2:
        raise ValueError("partition metric shape mismatch")
    delta = (right.astype(np.float64) - left.astype(np.float64))
    numerator = np.linalg.norm(delta, axis=1)
    denominator = np.maximum(np.linalg.norm(left.astype(np.float64), axis=1), 1e-30)
    row_relative = numerator / denominator
    return {
        **metric(left, right),
        "median_row_relative_l2": float(np.median(row_relative)),
        "maximum_row_relative_l2": float(np.max(row_relative, initial=0.0)),
    }


def sliced_metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    result = {"overall": partition_metrics(reference, candidate)}
    for name, (start, end) in PARTITIONS.items():
        result[name] = partition_metrics(reference[start:end], candidate[start:end])
    return result


def metrics_pass(rows: dict[str, Any]) -> bool:
    return all(
        row["relative_l2"] < MAXIMUM_ROUTE_RELATIVE_L2
        and row["maximum_row_relative_l2"] < MAXIMUM_ROW_RELATIVE_L2
        for row in rows.values()
    )


def construct(
    *,
    authority_root: Path,
    qtip_repo: Path,
    checkpoint_root: Path,
    checkpoint_receipt: Path,
    corpus_manifest: Path,
    output: Path,
    repo: Path,
    commit: str,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    started = time.monotonic()
    phase = "preflight"
    safety: HostSafetyMonitor | None = None
    torch = codebook = fit_x = exact = candidate = decoded_weights = None
    failure: dict[str, str] | None = None
    authority: dict[str, Any] = {}
    projections: dict[str, Any] = {}
    semantic: dict[str, Any] = {}
    paths = authority_paths(authority_root.resolve())
    report_path = output / "construction.json"

    try:
        verify_clean_commit(repo.resolve(), commit)
        safety = HostSafetyMonitor()
        if sha256_file(paths["contract"]) != PANEL_CONTRACT_SHA256:
            raise ValueError("PW-0352 recipe contract mismatch")
        if sha256_file(paths["reference_export"] / "export.json") != PANEL_EXPORT_SHA256:
            raise ValueError("PW-0352 recipe export mismatch")
        if sha256_file(paths["anchor"]) != ANCHOR_MANIFEST_SHA256:
            raise ValueError("MRL-0147 recipe anchor mismatch")
        checkpoint_authority = verify_full_checkpoint_install(
            checkpoint_root.resolve(), checkpoint_receipt.resolve()
        )
        if sha256_file(corpus_manifest) != CORPUS_SHA256:
            raise ValueError("PW-0116 corpus manifest mismatch")
        corpus = json.loads(corpus_manifest.read_text())
        if (
            corpus.get("revision") != CHECKPOINT_REVISION
            or corpus.get("checkpoint_verification_sha256") != CHECKPOINT_RECEIPT_SHA256
            or corpus.get("target_layers") != [4, 24, 46]
        ):
            raise ValueError("PW-0116 corpus authority mismatch")
        layer_rows = [row for row in corpus["layers"] if int(row["layer"]) == LAYER]
        if len(layer_rows) != 1:
            raise ValueError("PW-0116 layer-4 authority mismatch")
        layer_row = layer_rows[0]
        modules = _load_authority_modules(paths)
        contract = json.loads(paths["contract"].read_text())
        if sha256_file(paths["work"] / "tools/export_selected_k4_panel.py") != contract["authority"]["implementation_sha256"]:
            raise ValueError("K4 recipe implementation mismatch")
        pw0333 = json.loads(Path(contract["authority"]["pw0333_contract"]).read_text())
        pilot_contract = json.loads(Path(contract["authority"]["pilot_contract"]).read_text())
        verified_qtip = modules["qtip"]._verify_qtip(qtip_repo.resolve(), pw0333["authority"])
        if verified_qtip["commit"] != QTIP_COMMIT:
            raise ValueError("QTIP commit mismatch")
        authority = {
            "checkpoint": checkpoint_authority,
            "corpus_sha256": CORPUS_SHA256,
            "recipe_contract_sha256": PANEL_CONTRACT_SHA256,
            "recipe_export_sha256": PANEL_EXPORT_SHA256,
            "recipe_anchor_sha256": ANCHOR_MANIFEST_SHA256,
            "qtip": verified_qtip,
            "expected_projection_hashes": EXPECTED_PROJECTION_HASHES,
        }
        phase = "authorities_verified"
        safety.checkpoint(phase)

        torch = __import__("torch")
        if not torch.backends.mps.is_available():
            raise RuntimeError("PW-0314 construction requires Apple Metal")
        torch.set_grad_enabled(False)
        device = torch.device("mps:0")
        settings = pw0333["qtip_settings"]
        bitshift = modules["qtip"]._load_bitshift_module(qtip_repo.resolve())
        official_ldlq = modules["pilot"]._load_official_ldlq(qtip_repo.resolve(), torch)
        official_math = modules["pilot"]._load_official_math(qtip_repo.resolve())
        torch.manual_seed(int(pw0333["codebook_seed"]))
        codebook = bitshift.bitshift_codebook(
            L=int(settings["L"]), K=int(settings["K"]), V=int(settings["V"]),
            tlut_bits=int(settings["tlut_bits"]), decode_mode=settings["decode_mode"],
        )
        tlut = codebook.tlut.detach().cpu().contiguous().numpy().astype(np.float32, copy=False)
        if array_sha256(tlut) != TLUT_ARRAY_SHA256:
            raise ValueError("K4 TLUT mismatch")
        codebook = codebook.to(device)

        atlas_digest, manifest_paths = modules["atlas"]._canonical_manifest_set(
            Path(pilot_contract["authority"]["atlas_root"])
        )
        if atlas_digest != pilot_contract["authority"]["atlas_manifest_set_sha256"]:
            raise ValueError("calibration atlas mismatch")
        fit_x, fit_authority = modules["atlas"]._sample_inputs(
            manifest_paths,
            pilot_contract["evaluation_regime"]["splits"]["calibration"],
            LAYER,
            int(pilot_contract["rows_per_record"]),
            int(pilot_contract["seed"]),
        )
        phase = "calibration_loaded"
        safety.checkpoint(phase)

        corpus_root = corpus_manifest.parent
        moe_input = load_capture(corpus_root, layer_row, "moe_input")
        expert_down = load_capture(corpus_root, layer_row, "expert_down")
        source_routed = load_capture(corpus_root, layer_row, "routed_output")
        post_attention = load_capture(corpus_root, layer_row, "post_attention")
        source_final = load_capture(corpus_root, layer_row, "final")
        positions, slots, selected_weights, source_offsets = selected_rows(
            layer_row, EXPERT
        )
        partition_counts = {
            name: int(np.sum((start <= positions) & (positions < end)))
            for name, (start, end) in PARTITIONS.items()
        }
        scheduled_count = sum(
            len(row["positions"])
            for row in layer_row["expert_schedule"]
            if int(row["expert"]) == EXPERT
        )
        if (
            len(positions) == 0
            or len(positions) != scheduled_count
            or (
                EXPECTED_PLACEMENT_COUNT is not None
                and len(positions) != EXPECTED_PLACEMENT_COUNT
            )
        ):
            raise ValueError(
                f"layer-{LAYER} expert-{EXPERT} placement authority mismatch"
            )
        selected_input = np.asarray(moe_input[positions], dtype=np.float32)
        expected_source_output = np.asarray(
            expert_down[source_offsets], dtype=np.float32
        )

        reconstructed_source_route = reconstruct_route(
            expert_down, layer_row, modules["panel"].bf16
        )
        if not np.array_equal(reconstructed_source_route, source_routed):
            raise ValueError("PW-0116 source route reconstruction mismatch")
        reconstructed_source_final = modules["panel"].bf16(post_attention + source_routed)
        if not np.array_equal(reconstructed_source_final, source_final):
            raise ValueError("PW-0116 source final reconstruction mismatch")
        phase = "corpus_loaded_and_reconstructed"
        safety.checkpoint(phase)

        with modules["checkpoint"].Checkpoint(checkpoint_root) as checkpoint:
            prefix = f"model.layers.{LAYER}.mlp.experts.{EXPERT}"
            exact = {
                name: checkpoint.read_dequantized_fp8(f"{prefix}.{name}_proj.weight")
                for name in NAMES
            }
        source_outputs = modules["panel"].complete_outputs(selected_input, exact)
        source_output = source_outputs["candidate_output_bf16_f32"]
        source_replay = metric(expected_source_output, source_output)
        if not np.array_equal(source_output, expected_source_output):
            raise ValueError(f"PW-0116 source expert replay mismatch: {source_replay}")
        phase = "source_expert_replayed"
        safety.checkpoint(phase)

        calibration = {
            "gate": fit_x,
            "up": fit_x,
            "down": modules["activation"]._staged_activations(
                fit_x, exact["gate"], exact["up"]
            ),
        }
        artifact_root = output / f"layer-{LAYER:02d}-expert-{EXPERT:03d}"
        artifact_root.mkdir()
        candidate = {}
        decoded_weights = {}
        validation_x = modules["panel"].dynamic_input(selected_input[0:1])
        for index, name in enumerate(NAMES):
            quantized = modules["export"]._quantize(
                name,
                exact[name],
                calibration[name],
                int(pilot_contract["seed"]) + index * 1000,
                codebook,
                settings,
                official_ldlq,
                official_math,
                torch,
                device,
            )
            decoded = modules["export"]._decode_k4(
                quantized["packed"], tlut, quantized["rows"], quantized["columns"],
                quantized["scale"], quantized["left_sign"], quantized["right_sign"],
            )
            independent = metric(quantized["candidate"], decoded)
            if independent["relative_l2"] > MAXIMUM_DECODE_RELATIVE_L2:
                raise ValueError(f"independent decode mismatch: {name}")
            candidate[name] = quantized["candidate"]
            decoded_weights[name] = decoded
            validation_input = (
                validation_x[0]
                if name in ("gate", "up")
                else modules["panel"].complete_outputs(
                    selected_input[0:1], decoded_weights
                )["dynamic_hidden_f32"][0]
            )
            serialized = modules["export"]._serialize_projection(
                artifact_root,
                quantized,
                tlut,
                validation_input,
                [
                    "source-exact or L1 weights",
                    "other identities or layers",
                    "complete K4 bank",
                    "endpoint TPS",
                ],
            )
            projections[name] = {
                "quantization_seconds": quantized["seconds"],
                "independent_decode": independent,
                "candidate_array_sha256": serialized["candidate_array_sha256"],
                "packed_trellis_array_sha256": serialized["packed_trellis_array_sha256"],
                "manifest_sha256": serialized["manifest_sha256"],
                "model_storage_bytes": serialized["model_storage_bytes"],
            }
            phase = f"{name}_projection_constructed"
            safety.checkpoint(phase)
            del quantized, decoded
            gc.collect()
            torch.mps.empty_cache()

        if EXPECTED_PROJECTION_HASHES is not None:
            for name, expected in EXPECTED_PROJECTION_HASHES.items():
                observed = projections.get(name, {})
                for field, digest in expected.items():
                    if observed.get(field) != digest:
                        raise ValueError(
                            f"immutable projection control mismatch: {name}.{field}"
                        )

        candidate_output = modules["panel"].complete_outputs(
            selected_input, decoded_weights
        )["candidate_output_bf16_f32"]
        candidate_output_path = artifact_root / "candidate-output.f32le"
        candidate_output_path.write_bytes(
            np.asarray(candidate_output, dtype="<f4").tobytes(order="C")
        )
        candidate_expert = partition_metrics(source_output, candidate_output)
        candidate_expert["selected_placements"] = int(len(positions))
        candidate_down = expert_down.copy()
        candidate_down[source_offsets] = candidate_output
        candidate_route = reconstruct_route(
            candidate_down, layer_row, modules["panel"].bf16
        )
        candidate_final = modules["panel"].bf16(post_attention + candidate_route)
        route_metrics = sliced_metrics(source_routed, candidate_route)
        final_metrics = sliced_metrics(source_final, candidate_final)
        semantic = {
            "source_replay": source_replay,
            "source_route_reconstruction_exact": True,
            "source_final_reconstruction_exact": True,
            "expert_candidate_vs_source": candidate_expert,
            "route_candidate_vs_source": route_metrics,
            "final_candidate_vs_source": final_metrics,
            "placements": {
                "total": int(len(positions)),
                **partition_counts,
                "minimum_route_weight": float(np.min(selected_weights)),
                "median_route_weight": float(np.median(selected_weights)),
                "maximum_route_weight": float(np.max(selected_weights)),
            },
            "array_sha256": {
                "candidate_output_f32": array_sha256(candidate_output),
                "candidate_route_f32": array_sha256(candidate_route),
                "candidate_final_f32": array_sha256(candidate_final),
            },
        }
        semantic_pass = metrics_pass(route_metrics) and metrics_pass(final_metrics)
        semantic["gates"] = {
            "route_pass": metrics_pass(route_metrics),
            "final_pass": metrics_pass(final_metrics),
            "pass": semantic_pass,
        }
        status = (
            f"layer{LAYER}_expert{EXPERT}_semantically_qualified"
            if semantic_pass
            else f"layer{LAYER}_expert{EXPERT}_semantic_gate_failed"
        )
        decision = (
            "require_local_repeat"
            if semantic_pass
            else f"reject_layer{LAYER}_expert{EXPERT}_k4"
        )
    except (
        FileNotFoundError,
        HostSafetyViolation,
        KeyError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ) as error:
        status = f"layer{LAYER}_expert{EXPERT}_construction_failed"
        decision = "keep_cross_layer_k4_unproven"
        failure = {"phase": phase, "type": type(error).__name__, "message": str(error)}
    finally:
        fit_x = exact = candidate = decoded_weights = codebook = None
        if torch is not None:
            try:
                torch.mps.synchronize()
                torch.mps.empty_cache()
            except RuntimeError as error:
                if failure is None:
                    status = f"layer{LAYER}_expert{EXPERT}_construction_failed"
                    decision = "keep_cross_layer_k4_unproven"
                    failure = {"phase": "mps_release", "type": type(error).__name__, "message": str(error)}
        if safety is not None:
            try:
                safety.release_checkpoint(
                    "construction_buffers_released",
                    ["calibration activations", "source expert", "decoded K4 expert", "PW-0116 captures", "QTIP codebook", "MPS cache"],
                )
                safety.checkpoint("final_service_health")
            except (HostSafetyViolation, RuntimeError) as error:
                if failure is None:
                    status = f"layer{LAYER}_expert{EXPERT}_construction_failed"
                    decision = "keep_cross_layer_k4_unproven"
                    failure = {"phase": "construction_buffers_released", "type": type(error).__name__, "message": str(error)}

    tree = deterministic_tree_manifest(output)
    result = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "revision": REVISION,
        "status": status,
        "decision": decision,
        "layer": LAYER,
        "expert": EXPERT,
        "exactness_class": "L3 bounded target-native K4 approximation",
        "commit": commit,
        "tool_sha256": sha256_file(Path(__file__).resolve()),
        "authority": authority,
        "calibration_authority": locals().get("fit_authority"),
        "projections": projections,
        "semantic": semantic,
        "deterministic_tree": tree,
        "safety_snapshots": safety.evidence() if safety is not None else [],
        "complete_seconds": time.monotonic() - started,
        "peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "host": {
            "machine": platform.machine(),
            "platform": platform.platform(),
            "total_memory_bytes": int(os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")),
            "torch_version": getattr(torch, "__version__", None),
            "device": "mps:0",
            "target_host_proof": True,
        },
        "batch_size": 1,
        "concurrency": 1,
        "accepted_tokens": 0,
        "A": 0,
        "U": 0,
        "failure": failure,
        "claims_excluded": [
            "source-exact or L1 weights",
            "other identities or layers",
            "complete K4 bank",
            "hosted, multimodal, or capability equivalence",
            "ordinary endpoint execution",
            "accepted-token TPS",
            "Prismwing-2, 34.3 TPS, or Prismwing 50 completion",
        ],
        "performance_claim": None,
    }
    atomic_write_new(report_path, canonical_json(result))
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--authority-root", required=True, type=Path)
    parser.add_argument("--qtip-repo", required=True, type=Path)
    parser.add_argument("--checkpoint-root", required=True, type=Path)
    parser.add_argument("--checkpoint-receipt", required=True, type=Path)
    parser.add_argument("--corpus-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--commit", required=True)
    arguments = parser.parse_args()
    try:
        result = construct(
            authority_root=arguments.authority_root,
            qtip_repo=arguments.qtip_repo,
            checkpoint_root=arguments.checkpoint_root,
            checkpoint_receipt=arguments.checkpoint_receipt,
            corpus_manifest=arguments.corpus_manifest,
            output=arguments.output,
            repo=arguments.repo,
            commit=arguments.commit,
        )
        print(json.dumps({"output": str(arguments.output), "status": result["status"]}))
        return 0 if result["failure"] is None else 1
    except (FileExistsError, OSError, RuntimeError, ValueError) as error:
        print(json.dumps({"error": str(error)}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
