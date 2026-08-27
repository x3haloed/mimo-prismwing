#!/usr/bin/env python3
"""Construct and semantically gate one explicit ``m1-native-k4-v1`` expert.

This is deliberately separate from the PW-0311/PW-0312 bit-exact reproducer.
Cross-device payload differences remain visible here, but do not abort before
decoded-weight and complete-expert comparisons can classify the difference.
"""

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
    from tools.host_safety import HostSafetyMonitor, HostSafetyViolation
    from tools.openrouter_reference import atomic_write_new, canonical_json
    from tools.reproduce_pw0311_k4_expert import (
        ANCHOR_MANIFEST_SHA256,
        NAMES,
        PANEL_CONTRACT_SHA256,
        PANEL_EXPORT_SHA256,
        QTIP_COMMIT,
        SOURCE_MANIFEST_SHA256,
        TLUT_ARRAY_SHA256,
        _load_authority_modules,
        array_sha256,
        authority_paths,
        select_reference_slot,
        sha256_file,
        verify_clean_commit,
        verify_source_checkpoint,
    )
except ModuleNotFoundError:
    from host_safety import HostSafetyMonitor, HostSafetyViolation
    from openrouter_reference import atomic_write_new, canonical_json
    from reproduce_pw0311_k4_expert import (
        ANCHOR_MANIFEST_SHA256,
        NAMES,
        PANEL_CONTRACT_SHA256,
        PANEL_EXPORT_SHA256,
        QTIP_COMMIT,
        SOURCE_MANIFEST_SHA256,
        TLUT_ARRAY_SHA256,
        _load_authority_modules,
        array_sha256,
        authority_paths,
        select_reference_slot,
        sha256_file,
        verify_clean_commit,
        verify_source_checkpoint,
    )


EXPERIMENT_ID = "PW-0313"
REVISION = "m1-native-k4-v1"
SUPPORTED_EXPERTS = (199, 41)
ROUTE_EXPERTS = (188, 199, 252)
MAXIMUM_DECODE_RELATIVE_L2 = 2e-5
MAXIMUM_M4_OUTPUT_RELATIVE_L2 = 0.005
MAXIMUM_SOURCE_DEGRADATION = 0.005
MAXIMUM_ROUTE_M4_RELATIVE_L2 = 0.001
MAXIMUM_ROUTE_SOURCE_RELATIVE_L2 = 0.01
PW0424_SHA256 = "05439a232c2002530002d95ac29831b38a5c74b1049903406747620b3ce4f64e"


def metric(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    left = np.asarray(reference, dtype=np.float32)
    right = np.asarray(candidate, dtype=np.float32)
    if left.shape != right.shape:
        raise ValueError(f"metric shape mismatch: {left.shape} != {right.shape}")
    delta = right.astype(np.float64) - left.astype(np.float64)
    denominator = max(float(np.linalg.norm(left.astype(np.float64))), 1e-30)
    return {
        "relative_l2": float(np.linalg.norm(delta) / denominator),
        "maximum_absolute_error": float(np.max(np.abs(delta), initial=0.0)),
    }


def projection_payload_files(directory: Path) -> list[Path]:
    manifest = json.loads((directory / "manifest.json").read_text())
    files = [directory / "manifest.json", directory / manifest["fixture"]["file"]]
    files.extend(directory / row["file"] for row in manifest["files"].values())
    return files


def compare_payload_trees(candidate: Path, reference: Path) -> dict[str, Any]:
    candidate_manifest = json.loads((candidate / "manifest.json").read_text())
    reference_manifest = json.loads((reference / "manifest.json").read_text())
    candidate_files = {path.name: path for path in projection_payload_files(candidate)}
    reference_files = {path.name: path for path in projection_payload_files(reference)}
    if set(candidate_files) != set(reference_files):
        raise ValueError("projection payload file set mismatch")
    differences = []
    matching_bytes = 0
    for name in sorted(candidate_files):
        candidate_path = candidate_files[name]
        reference_path = reference_files[name]
        if candidate_path.read_bytes() == reference_path.read_bytes():
            matching_bytes += candidate_path.stat().st_size
        else:
            differences.append(name)
    model_keys = (
        "packed",
        "left_sign",
        "right_sign",
        "global_scale",
        "row_scale",
        "correction_left",
        "correction_right",
    )
    model_names = {
        candidate_manifest["files"][key]["file"] for key in model_keys
    }
    if model_names != {
        reference_manifest["files"][key]["file"] for key in model_keys
    }:
        raise ValueError("projection model-payload file set mismatch")
    model_differences = [name for name in differences if name in model_names]
    return {
        "files_compared": len(candidate_files),
        "differing_files": differences,
        "model_payload_differing_files": model_differences,
        "matching_bytes": matching_bytes,
        "all_files_identical": not differences,
        "payload_identical": not model_differences,
    }


def load_decoded_projection(directory: Path, tlut: np.ndarray, decode: Any) -> np.ndarray:
    manifest = json.loads((directory / "manifest.json").read_text())

    def load(name: str) -> np.ndarray:
        row = manifest["files"][name]
        value = np.fromfile(directory / row["file"], dtype=row["dtype"])
        return value.reshape(row["shape"])

    packed = load("packed")
    left_sign = load("left_sign").astype(np.float32)
    right_sign = load("right_sign").astype(np.float32)
    scale = float(load("global_scale")[0])
    return decode(
        packed,
        tlut,
        int(manifest["rows"]),
        int(manifest["columns"]),
        scale,
        left_sign,
        right_sign,
    )


def classify_projection(payload: dict[str, Any], decoded_metric: dict[str, float]) -> str:
    if payload["payload_identical"]:
        return "payload_identical"
    if decoded_metric["relative_l2"] == 0.0:
        return "semantic_alias"
    return "numerical_drift"


def deterministic_tree_manifest(root: Path) -> dict[str, Any]:
    records = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        if path.name == "construction.json":
            continue
        records.append(
            {
                "path": str(path.relative_to(root)),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return {"files": records, "total_bytes": sum(row["bytes"] for row in records)}


def _source_expert(checkpoint: Any, layer: int, expert: int) -> dict[str, np.ndarray]:
    prefix = f"model.layers.{layer}.mlp.experts.{expert}"
    return {
        name: checkpoint.read_dequantized_fp8(f"{prefix}.{name}_proj.weight")
        for name in NAMES
    }


def substitute_exact_frozen_route(
    frozen_route: np.ndarray,
    frozen_expert_output: np.ndarray,
    replacement_expert_output: np.ndarray,
) -> np.ndarray:
    """Prove an exact substitution without inventing the lost PW-0424 assembler."""
    if metric(frozen_expert_output, replacement_expert_output)["relative_l2"] != 0.0:
        raise ValueError("non-identical expert output requires a new authenticated route assembler")
    return np.asarray(frozen_route, dtype=np.float32).copy()


def construct(
    *,
    authority_root: Path,
    qtip_repo: Path,
    route_fixture: Path,
    output: Path,
    expert: int,
    repo: Path,
    commit: str,
) -> dict[str, Any]:
    if expert not in SUPPORTED_EXPERTS:
        raise ValueError(f"PW-0313 supports only experts {SUPPORTED_EXPERTS}")
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    started = time.monotonic()
    phase = "preflight"
    safety: HostSafetyMonitor | None = None
    torch = codebook = fit_x = exact = local_weights = None
    failure: dict[str, str] | None = None
    projections: dict[str, Any] = {}
    semantic: dict[str, Any] = {}
    authority: dict[str, Any] = {}
    paths = authority_paths(authority_root.resolve())
    report_path = output / "construction.json"

    try:
        verify_clean_commit(repo.resolve(), commit)
        safety = HostSafetyMonitor()
        if sha256_file(paths["contract"]) != PANEL_CONTRACT_SHA256:
            raise ValueError("PW-0352 panel contract mismatch")
        if sha256_file(paths["reference_export"] / "export.json") != PANEL_EXPORT_SHA256:
            raise ValueError("PW-0352 export report mismatch")
        if sha256_file(paths["anchor"]) != ANCHOR_MANIFEST_SHA256:
            raise ValueError("MRL-0147 anchor mismatch")
        if sha256_file(route_fixture) != PW0424_SHA256:
            raise ValueError("PW-0424 route fixture mismatch")
        contract = json.loads(paths["contract"].read_text())
        reference_report = json.loads((paths["reference_export"] / "export.json").read_text())
        reference_slot = select_reference_slot(reference_report, expert)
        source_authority = verify_source_checkpoint(paths["source_checkpoint"])
        if source_authority["manifest_sha256"] != SOURCE_MANIFEST_SHA256:
            raise ValueError("source authority mismatch")
        modules = _load_authority_modules(paths)
        if sha256_file(paths["work"] / "tools/export_selected_k4_panel.py") != contract["authority"]["implementation_sha256"]:
            raise ValueError("PW-0352 exporter implementation mismatch")
        pw0333 = json.loads(Path(contract["authority"]["pw0333_contract"]).read_text())
        pilot_contract = json.loads(Path(contract["authority"]["pilot_contract"]).read_text())
        verified_qtip = modules["qtip"]._verify_qtip(qtip_repo.resolve(), pw0333["authority"])
        if verified_qtip["commit"] != QTIP_COMMIT:
            raise ValueError("QTIP commit mismatch")
        authority = {
            "contract_sha256": PANEL_CONTRACT_SHA256,
            "reference_export_sha256": PANEL_EXPORT_SHA256,
            "source": source_authority,
            "anchor_sha256": ANCHOR_MANIFEST_SHA256,
            "route_fixture_sha256": PW0424_SHA256,
            "qtip": verified_qtip,
        }
        phase = "authorities_verified"
        safety.checkpoint(phase)

        torch = __import__("torch")
        if not torch.backends.mps.is_available():
            raise RuntimeError("PW-0313 construction requires Apple Metal")
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
        layer = int(contract["layer"])
        fit_x, fit_authority = modules["atlas"]._sample_inputs(
            manifest_paths,
            pilot_contract["evaluation_regime"]["splits"]["calibration"],
            layer,
            int(pilot_contract["rows_per_record"]),
            int(pilot_contract["seed"]),
        )
        phase = "calibration_loaded"
        safety.checkpoint(phase)

        anchor = json.loads(paths["anchor"].read_text())
        layer_row = modules["activation"]._layer(anchor, layer)
        moe_input = np.asarray(
            modules["panel"]._capture(
                paths["anchor"].parent,
                layer_row["captures"]["moe_input"],
                (modules["panel"].ROWS, modules["panel"].HIDDEN),
            )
        )
        x = np.asarray(moe_input[0:1], dtype=np.float32)
        dynamic_x = modules["panel"].dynamic_input(x)
        expert_dir = output / f"expert-{expert:03d}"
        expert_dir.mkdir()
        local_weights = {}
        m4_weights = {}

        with modules["checkpoint"].Checkpoint(paths["source_checkpoint"]) as checkpoint:
            exact = _source_expert(checkpoint, layer, expert)
            calibration = {
                "gate": fit_x,
                "up": fit_x,
                "down": modules["activation"]._staged_activations(
                    fit_x, exact["gate"], exact["up"]
                ),
            }
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
                    raise ValueError(f"independent decode mismatch: {expert}.{name}")
                local_weights[name] = decoded
                validation_input = (
                    dynamic_x[0]
                    if name in ("gate", "up")
                    else modules["panel"].complete_outputs(x, local_weights)["dynamic_hidden_f32"][0]
                )
                serialized = modules["export"]._serialize_projection(
                    expert_dir,
                    quantized,
                    tlut,
                    validation_input,
                    [
                        "source-exact or L1 weights",
                        "cross-device payload identity",
                        "complete-bank construction",
                        "endpoint TPS",
                    ],
                )
                reference_dir = paths["reference_export"] / f"expert-{expert:03d}" / name
                m4_decoded = load_decoded_projection(reference_dir, tlut, modules["export"]._decode_k4)
                m4_weights[name] = m4_decoded
                payload = compare_payload_trees(expert_dir / name, reference_dir)
                decoded_vs_m4 = metric(m4_decoded, decoded)
                expected = reference_slot["projection_reports"][name]
                projections[name] = {
                    "classification": classify_projection(payload, decoded_vs_m4),
                    "quantization_seconds": quantized["seconds"],
                    "independent_decode": independent,
                    "candidate_array_sha256": serialized["candidate_array_sha256"],
                    "m4_candidate_array_sha256": expected["candidate_array_sha256"],
                    "packed_trellis_array_sha256": serialized["packed_trellis_array_sha256"],
                    "m4_packed_trellis_array_sha256": expected["packed_trellis_array_sha256"],
                    "decoded_vs_m4": decoded_vs_m4,
                    "payload": payload,
                }
                phase = f"{name}_projection_constructed"
                safety.checkpoint(phase)
                del quantized, decoded, m4_decoded
                gc.collect()
                torch.mps.empty_cache()

            local_output = modules["panel"].complete_outputs(x, local_weights)["candidate_output_bf16_f32"]
            m4_output = modules["panel"].complete_outputs(x, m4_weights)["candidate_output_bf16_f32"]
            source_output = modules["panel"].complete_outputs(x, exact)["candidate_output_bf16_f32"]
            m4_vs_source = metric(source_output, m4_output)
            local_vs_source = metric(source_output, local_output)
            local_vs_m4 = metric(m4_output, local_output)
            frozen_slot = next(row for row in json.loads(route_fixture.read_text())["slots"] if int(row["expert"]) == expert)
            frozen_m4_output = np.asarray(frozen_slot["candidate_output_bf16_f32"], dtype=np.float32)[None, :]
            frozen_m4_check = metric(frozen_m4_output, m4_output)
            if frozen_m4_check["relative_l2"] != 0.0:
                raise ValueError("decoded M4 output does not match frozen PW-0424 slot")
            semantic["expert_output"] = {
                "m1_vs_m4": local_vs_m4,
                "m1_vs_source": local_vs_source,
                "m4_vs_source": m4_vs_source,
                "source_degradation": local_vs_source["relative_l2"] - m4_vs_source["relative_l2"],
                "frozen_m4_check": frozen_m4_check,
            }

            if expert == 199:
                fixture = json.loads(route_fixture.read_text())
                frozen_route = np.asarray(fixture["candidate_routed_f32"], dtype=np.float32)[None, :]
                local_route = substitute_exact_frozen_route(
                    frozen_route, frozen_m4_output, local_output
                )
                source_route = np.asarray(fixture["exact_reached_routed_f32"], dtype=np.float32)[None, :]
                semantic["policy_route"] = {
                    "m1_vs_m4": metric(frozen_route, local_route),
                    "m1_vs_source": metric(source_route, local_route),
                    "proof": "bit-identical expert output substituted into authenticated frozen route",
                    "lost_assembler_not_reconstructed": True,
                    "candidate_route_array_sha256": array_sha256(local_route[0]),
                }

        expert_gate_pass = (
            semantic["expert_output"]["m1_vs_m4"]["relative_l2"]
            <= MAXIMUM_M4_OUTPUT_RELATIVE_L2
            and semantic["expert_output"]["source_degradation"] <= MAXIMUM_SOURCE_DEGRADATION
        )
        route_gate_pass = expert != 199 or (
            semantic["policy_route"]["m1_vs_m4"]["relative_l2"]
            <= MAXIMUM_ROUTE_M4_RELATIVE_L2
            and semantic["policy_route"]["m1_vs_source"]["relative_l2"]
            < MAXIMUM_ROUTE_SOURCE_RELATIVE_L2
        )
        semantic["gates"] = {
            "expert_output_pass": expert_gate_pass,
            "policy_route_pass": route_gate_pass,
            "pass": expert_gate_pass and route_gate_pass,
        }
        status = "m1_native_expert_semantically_qualified" if semantic["gates"]["pass"] else "m1_native_expert_semantic_gate_failed"
        decision = "require_local_repeat" if semantic["gates"]["pass"] else "reject_or_restrict_m1_native_k4"
    except (
        FileNotFoundError,
        HostSafetyViolation,
        KeyError,
        OSError,
        RuntimeError,
        StopIteration,
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ) as error:
        status = "m1_native_expert_construction_failed"
        decision = "keep_m1_native_k4_unproven"
        failure = {"phase": phase, "type": type(error).__name__, "message": str(error)}
    finally:
        fit_x = exact = local_weights = codebook = None
        if torch is not None:
            try:
                torch.mps.synchronize()
                torch.mps.empty_cache()
            except RuntimeError as error:
                if failure is None:
                    status = "m1_native_expert_construction_failed"
                    decision = "keep_m1_native_k4_unproven"
                    failure = {"phase": "mps_release", "type": type(error).__name__, "message": str(error)}
        if safety is not None:
            try:
                safety.release_checkpoint(
                    "construction_buffers_released",
                    ["calibration activations", "source experts", "decoded K4 experts", "QTIP codebook", "MPS cache"],
                )
                safety.checkpoint("final_service_health")
            except (HostSafetyViolation, RuntimeError) as error:
                if failure is None:
                    status = "m1_native_expert_construction_failed"
                    decision = "keep_m1_native_k4_unproven"
                    failure = {"phase": "construction_buffers_released", "type": type(error).__name__, "message": str(error)}

    tree = deterministic_tree_manifest(output)
    result = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "revision": REVISION,
        "status": status,
        "decision": decision,
        "expert": expert,
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
            "cross-device payload identity",
            "arbitrary experts or layers",
            "complete K4 bank",
            "hosted or multimodal equivalence",
            "ordinary endpoint execution",
            "accepted-token TPS",
            "Prismwing-2 or Prismwing 50 completion",
        ],
        "performance_claim": None,
    }
    atomic_write_new(report_path, canonical_json(result))
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--authority-root", required=True, type=Path)
    parser.add_argument("--qtip-repo", required=True, type=Path)
    parser.add_argument("--route-fixture", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expert", required=True, type=int)
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--commit", required=True)
    arguments = parser.parse_args()
    try:
        result = construct(
            authority_root=arguments.authority_root,
            qtip_repo=arguments.qtip_repo,
            route_fixture=arguments.route_fixture,
            output=arguments.output,
            expert=arguments.expert,
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
