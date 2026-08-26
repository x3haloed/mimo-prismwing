#!/usr/bin/env python3
"""Reproduce one authenticated PW-0352 K4 expert on the target M1."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time
from typing import Any

import numpy as np

try:
    from tools.host_safety import HostSafetyMonitor, HostSafetyViolation
    from tools.openrouter_reference import atomic_write_new, canonical_json
except ModuleNotFoundError:
    from host_safety import HostSafetyMonitor, HostSafetyViolation
    from openrouter_reference import atomic_write_new, canonical_json


EXPERIMENT_ID = "PW-0311"
PANEL_CONTRACT_SHA256 = "dddcafc5cbac96246c62e401d6eed00706595a2918ef01b97e03fa3047280258"
PANEL_EXPORT_SHA256 = "054c5fd41031ab91b234e7817864db9af0d3b8756ec189186ed1f2a67c5a51a5"
SOURCE_MANIFEST_SHA256 = "c567a637e643476820ed07960385a9de84010ab48d9428441a08a84687b29ac8"
ANCHOR_MANIFEST_SHA256 = "14331fa6e6314d0b82b5a5b7085870e549db2dc9810d03d9251565ca5b281d9a"
QTIP_COMMIT = "e90c6688c8dfae326a3a81b5eb032db7c6680ec0"
TLUT_ARRAY_SHA256 = "21bab03171fb4ccaf2b4fb86f3b48efb2d7daa526f2b6dd3b01ceef9db95a9d8"
SUPPORTED_EXPERTS = (114, 188, 93, 199, 248, 41, 252)
NAMES = ("gate", "up", "down")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(value: np.ndarray) -> str:
    array = np.asarray(value)
    digest = hashlib.sha256()
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _run_git(repo: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *arguments],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def verify_clean_commit(repo: Path, commit: str) -> None:
    if len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit):
        raise ValueError("commit must be lowercase 40-hex")
    if _run_git(repo, "rev-parse", "HEAD") != commit:
        raise ValueError("declared commit is not repository HEAD")
    if _run_git(repo, "status", "--porcelain"):
        raise ValueError("PW-0311 reproduction requires a clean worktree")


def authority_paths(authority_root: Path) -> dict[str, Path]:
    work = authority_root / "work/prismwing-exact-boundary"
    lab = authority_root / "outputs/mimo-representation-lab"
    return {
        "work": work,
        "lab": lab,
        "lab_source": lab / "src",
        "authority_tools": work / "tools",
        "contract": work / "spec/pw0352-distinct-selected-k4-panel.json",
        "reference_export": work / "evidence/PW-0352/export",
        "source_checkpoint": work / "evidence/PW-0351/source-experts",
        "anchor": lab / "evidence/MRL-0147/native-code-07/manifest.json",
    }


def verify_source_checkpoint(source_root: Path) -> dict[str, Any]:
    manifest_path = source_root / "manifest.json"
    if sha256_file(manifest_path) != SOURCE_MANIFEST_SHA256:
        raise ValueError("PW-0351 source manifest mismatch")
    manifest = json.loads(manifest_path.read_text())
    index_path = source_root / "model.safetensors.index.json"
    if sha256_file(index_path) != manifest["local_index_sha256"]:
        raise ValueError("PW-0351 local index mismatch")
    verified_bytes = index_path.stat().st_size
    for shard in manifest["shards"]:
        artifact = source_root / shard["artifact_file"]
        range_manifest = source_root / shard["range_manifest"]
        if artifact.stat().st_size != int(shard["artifact_bytes"]):
            raise ValueError(f"source artifact byte mismatch: {artifact.name}")
        if sha256_file(artifact) != shard["artifact_sha256"]:
            raise ValueError(f"source artifact hash mismatch: {artifact.name}")
        if sha256_file(range_manifest) != shard["range_manifest_sha256"]:
            raise ValueError(f"source range-manifest mismatch: {range_manifest.name}")
        verified_bytes += artifact.stat().st_size + range_manifest.stat().st_size
    return {
        "revision": manifest["revision"],
        "experts": manifest["experts"],
        "tensor_count": manifest["tensor_count"],
        "verified_bytes": verified_bytes,
        "manifest_sha256": SOURCE_MANIFEST_SHA256,
    }


def select_reference_slot(report: dict[str, Any], expert: int) -> dict[str, Any]:
    if expert not in SUPPORTED_EXPERTS:
        raise ValueError(f"expert {expert} is outside the authenticated PW-0352 panel")
    slots = [slot for slot in report["slots"] if int(slot["expert"]) == expert]
    if len(slots) != 1:
        raise ValueError(f"expected exactly one PW-0352 slot for expert {expert}")
    if set(slots[0]["projection_reports"]) != set(NAMES):
        raise ValueError("reference slot projection set mismatch")
    return slots[0]


def compare_projection_directory(
    candidate: Path,
    reference: Path,
    expected: dict[str, Any],
) -> dict[str, Any]:
    candidate_manifest = candidate / "manifest.json"
    reference_manifest = reference / "manifest.json"
    expected_manifest_hash = expected["manifest_sha256"]
    if sha256_file(reference_manifest) != expected_manifest_hash:
        raise ValueError(f"reference manifest authority mismatch: {reference}")
    if sha256_file(candidate_manifest) != expected_manifest_hash:
        raise ValueError(f"candidate manifest mismatch: {candidate}")
    manifest = json.loads(reference_manifest.read_text())
    relative_files = ["manifest.json", manifest["fixture"]["file"]]
    relative_files.extend(row["file"] for row in manifest["files"].values())
    total_bytes = 0
    hashes: dict[str, str] = {}
    for relative in relative_files:
        candidate_file = candidate / relative
        reference_file = reference / relative
        if candidate_file.read_bytes() != reference_file.read_bytes():
            raise ValueError(f"candidate payload mismatch: {candidate_file}")
        digest = sha256_file(candidate_file)
        hashes[relative] = digest
        total_bytes += candidate_file.stat().st_size
    return {
        "files_bit_exact": len(relative_files),
        "bytes_bit_exact": total_bytes,
        "file_sha256": hashes,
    }


def _load_authority_modules(paths: dict[str, Path]) -> dict[str, Any]:
    for path in (paths["authority_tools"], paths["lab_source"]):
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)
    return {
        "activation": importlib.import_module("mimo_lab.activation_selected_neurons"),
        "atlas": importlib.import_module("compress_cross_layer_corrected_qtip_expert"),
        "checkpoint": importlib.import_module("mimo_lab.safetensors"),
        "export": importlib.import_module("export_qtip_k4_ldlq_expert"),
        "panel": importlib.import_module("export_selected_k4_panel"),
        "pilot": importlib.import_module("run_qtip_ldlq_expert_pilot"),
        "qtip": importlib.import_module("mimo_lab.qtip_incoherent_projection_pilot"),
    }


def reproduce(
    *,
    authority_root: Path,
    qtip_repo: Path,
    output: Path,
    expert: int,
    repo: Path,
    commit: str,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    started = time.monotonic()
    phase = "preflight"
    safety: HostSafetyMonitor | None = None
    torch = codebook = fit_x = exact = candidate = None
    projection_results: dict[str, Any] = {}
    failure: dict[str, str] | None = None
    paths = authority_paths(authority_root.resolve())
    report_path = output / "reproduction.json"
    authority: dict[str, Any] = {}

    try:
        verify_clean_commit(repo.resolve(), commit)
        safety = HostSafetyMonitor()
        if sha256_file(paths["contract"]) != PANEL_CONTRACT_SHA256:
            raise ValueError("PW-0352 panel contract mismatch")
        if sha256_file(paths["reference_export"] / "export.json") != PANEL_EXPORT_SHA256:
            raise ValueError("PW-0352 export report mismatch")
        if sha256_file(paths["anchor"]) != ANCHOR_MANIFEST_SHA256:
            raise ValueError("MRL-0147 anchor mismatch")
        contract = json.loads(paths["contract"].read_text())
        if list(map(int, contract["experts"])) != list(SUPPORTED_EXPERTS):
            raise ValueError("PW-0352 expert-order mismatch")
        reference_report = json.loads((paths["reference_export"] / "export.json").read_text())
        reference_slot = select_reference_slot(reference_report, expert)
        source_authority = verify_source_checkpoint(paths["source_checkpoint"])
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
            "qtip": verified_qtip,
        }
        phase = "authorities_verified"
        safety.checkpoint(phase)

        torch = importlib.import_module("torch")
        if not torch.backends.mps.is_available():
            raise RuntimeError("PW-0311 expert reproduction requires Apple Metal")
        torch.set_grad_enabled(False)
        device = torch.device("mps:0")
        settings = pw0333["qtip_settings"]
        bitshift = modules["qtip"]._load_bitshift_module(qtip_repo.resolve())
        official_ldlq = modules["pilot"]._load_official_ldlq(qtip_repo.resolve(), torch)
        official_math = modules["pilot"]._load_official_math(qtip_repo.resolve())
        torch.manual_seed(int(pw0333["codebook_seed"]))
        codebook = bitshift.bitshift_codebook(
            L=int(settings["L"]),
            K=int(settings["K"]),
            V=int(settings["V"]),
            tlut_bits=int(settings["tlut_bits"]),
            decode_mode=settings["decode_mode"],
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

        prefix = f"model.layers.{layer}.mlp.experts.{expert}"
        with modules["checkpoint"].Checkpoint(paths["source_checkpoint"]) as checkpoint:
            exact = {
                name: checkpoint.read_dequantized_fp8(f"{prefix}.{name}_proj.weight")
                for name in NAMES
            }
        calibration = {
            "gate": fit_x,
            "up": fit_x,
            "down": modules["activation"]._staged_activations(fit_x, exact["gate"], exact["up"]),
        }
        phase = "source_expert_loaded"
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
        candidate = {}
        expert_dir = output / f"expert-{expert:03d}"
        expert_dir.mkdir()
        for index, name in enumerate(NAMES):
            result = modules["export"]._quantize(
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
                result["packed"],
                tlut,
                result["rows"],
                result["columns"],
                result["scale"],
                result["left_sign"],
                result["right_sign"],
            )
            parity = modules["panel"]._metric(result["candidate"], decoded)
            if parity["relative_l2"] > contract["gates"]["maximum_independent_decode_relative_l2"]:
                raise ValueError(f"independent decode mismatch: {expert}.{name}")
            candidate[name] = result["candidate"]
            if name in ("gate", "up"):
                validation_input = dynamic_x[0]
            elif len(candidate) == 3:
                validation_input = modules["panel"].complete_outputs(x, candidate)["dynamic_hidden_f32"][0]
            else:
                raise ValueError("down projection serialized before complete candidate")
            serialized = modules["export"]._serialize_projection(
                expert_dir,
                result,
                tlut,
                validation_input,
                contract["claims_excluded"],
            )
            expected = reference_slot["projection_reports"][name]
            if serialized["candidate_array_sha256"] != expected["candidate_array_sha256"]:
                raise ValueError(f"candidate array mismatch: {expert}.{name}")
            if serialized["packed_trellis_array_sha256"] != expected["packed_trellis_array_sha256"]:
                raise ValueError(f"packed trellis mismatch: {expert}.{name}")
            comparison = compare_projection_directory(
                expert_dir / name,
                paths["reference_export"] / f"expert-{expert:03d}" / name,
                expected,
            )
            projection_results[name] = {
                "quantization_seconds": result["seconds"],
                "independent_decode": parity,
                "candidate_array_sha256": serialized["candidate_array_sha256"],
                "packed_trellis_array_sha256": serialized["packed_trellis_array_sha256"],
                **comparison,
            }
            phase = f"{name}_projection_reproduced"
            safety.checkpoint(phase)
            del result, decoded
            gc.collect()
            torch.mps.empty_cache()

        status = "one_expert_payload_bit_exact"
        decision = "authorize_arbitrary_expert_constructor_generalization"
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
        status = "one_expert_reproduction_failed"
        decision = "keep_arbitrary_expert_construction_unproven"
        failure = {"phase": phase, "type": type(error).__name__, "message": str(error)}
    finally:
        fit_x = exact = candidate = codebook = None
        if torch is not None:
            try:
                torch.mps.synchronize()
                torch.mps.empty_cache()
            except RuntimeError as error:
                if failure is None:
                    status = "one_expert_reproduction_failed"
                    decision = "keep_arbitrary_expert_construction_unproven"
                    failure = {"phase": "mps_release", "type": type(error).__name__, "message": str(error)}
        if safety is not None:
            try:
                safety.release_checkpoint(
                    "construction_buffers_released",
                    ["calibration activations", "source expert", "QTIP codebook", "MPS cache"],
                )
                safety.checkpoint("final_service_health")
            except (HostSafetyViolation, RuntimeError) as error:
                if failure is None:
                    status = "one_expert_reproduction_failed"
                    decision = "keep_arbitrary_expert_construction_unproven"
                    failure = {"phase": "construction_buffers_released", "type": type(error).__name__, "message": str(error)}

    result = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "status": status,
        "decision": decision,
        "expert": expert,
        "exactness_class": "L3 artifact construction; payload reproduction is bit exact",
        "commit": commit,
        "tool_sha256": sha256_file(Path(__file__).resolve()),
        "authority": authority,
        "projections": projection_results,
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
            "second-expert or cross-layer construction",
            "complete K4 bank",
            "general fidelity or modalities",
            "ordinary endpoint execution",
            "accepted-token TPS",
            "Prismwing-2 completion",
            "Prismwing 50 completion",
        ],
        "performance_claim": None,
    }
    atomic_write_new(report_path, canonical_json(result))
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--authority-root", required=True, type=Path)
    parser.add_argument("--qtip-repo", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expert", required=True, type=int)
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--commit", required=True)
    arguments = parser.parse_args()
    try:
        result = reproduce(
            authority_root=arguments.authority_root,
            qtip_repo=arguments.qtip_repo,
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
