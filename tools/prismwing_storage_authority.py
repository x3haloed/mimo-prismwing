#!/usr/bin/env python3
"""Shared fail-closed fixed-census and storage-bandwidth authority."""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics
from typing import Any

from safetensors import safe_open


REVISION = "63651580ca774f8504f676040460aed3e1244ac1"
CHECKPOINT_VERIFICATION_PATH = Path(
    "/Volumes/Elements/mimo-prismwing/cold-assets/internal-ssd-migration-2026-08-26/"
    "Users/chad/Models/mimo-prismwing/evidence/PW-0049/checkpoint-verification.json"
)
CHECKPOINT_VERIFICATION_SHA256 = (
    "9ddc8a99755f04ae2ea3c2484f6dd022d3f3a681b5a72c915ee4de833dbb0d03"
)
INDEX_NAME = "model.safetensors.index.json"
INDEX_SHA256 = "f2e1774c9acf9a62338b68c144e6fc7a66495e59f2e64b3078c1b7ef5a196816"
PW0207_OFFLINE_PATH = Path("/Volumes/Elements/mimo-prismwing/evidence/PW-0207/offline-002.json")
PW0207_OFFLINE_SHA256 = "1dedbef7c79aa23835d194f52760a1f2c65dcca1481bd6df2d5602615c3fdad6"
PW0136_RAW_PATH = Path("/Users/chad/Models/mimo-prismwing/evidence/PW-0136/run-001.json")
PW0136_RAW_SHA256 = "e6ab84cada19c6036ee7b83f318c3920631141b9ea5e882cc88eb9784d0b5a56"
PW0136_ANALYSIS_PATH = Path(
    "/Users/chad/Models/mimo-prismwing/evidence/PW-0136/analysis-001/manifest.json"
)
PW0136_ANALYSIS_SHA256 = "7ebf2cde5c4a3f4931d2d705993f822e38af13ea66bc3efc91410296b14e2aab"
PW0136_CAPTURE_COMMIT = "cebc5150b0bd92f6f4098b1d7d1f39c53364e05b"

ITEM_BYTES = {"F8_E4M3": 1, "BF16": 2, "F32": 4}
RESIDENT_ALIGNMENT_BYTES = 16 * 1024
FIXED_OBJECT_COUNT = 381
FIXED_LOGICAL_BYTES = 7_743_236_992
FIXED_ALLOCATED_BYTES = 7_745_470_464
FIXED_FP8_CODE_BYTES = 3_073_376_256
FIXED_NON_FP8_BYTES = 4_669_860_736
FIXED_MAX_OBJECT_BYTES = 1_249_902_592
FIXED_MAX_OBJECT = "lm_head.weight"

BANDWIDTH_ARTIFACT_BYTES = 201_719_808
BANDWIDTH_EXACT_MEDIAN_MS = 58.125375
BANDWIDTH_ROUNDED_MEDIAN_MS = 58.125
BANDWIDTH_EXACT_BYTES_PER_SECOND = 3_470_425_919.832775
BANDWIDTH_FAVORABLE_BYTES_PER_SECOND = 3_470_448_309.677419
PW0136_WORKERS = (1, 2, 4, 8)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _mapping(value: Any, message: str) -> dict[str, Any]:
    _require(isinstance(value, dict), message)
    return value


def _sequence(value: Any, message: str) -> list[Any]:
    _require(isinstance(value, list), message)
    return value


def _integer(value: Any, message: str, *, minimum: int = 0) -> int:
    _require(type(value) is int and value >= minimum, message)
    return value


def _json(path: Path, label: str) -> dict[str, Any]:
    return _mapping(json.loads(path.read_text()), f"{label}: JSON root")


def _strict_hash(path: Path, expected: str, label: str) -> str:
    actual = sha256_file(path)
    _require(actual == expected, f"{label}: SHA-256 mismatch")
    return actual


def resident_allocation_bytes(source_bytes: int) -> int:
    source_bytes = _integer(source_bytes, "resident source bytes", minimum=1)
    return (
        (source_bytes + RESIDENT_ALIGNMENT_BYTES - 1) // RESIDENT_ALIGNMENT_BYTES
    ) * RESIDENT_ALIGNMENT_BYTES


def fixed_tensor_names(weight_map: dict[str, str]) -> list[str]:
    """Derive the exact target fixed-spine tensor set from model structure."""

    _require(isinstance(weight_map, dict), "weight map")
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
    names.extend(["model.norm.weight", FIXED_MAX_OBJECT])
    _require(len(names) == len(set(names)), "duplicate fixed tensor identity")
    _require(all(name in weight_map for name in names), "fixed tensor authority is incomplete")
    _require("model.embed_tokens.weight" not in names, "embedding table entered fixed census")
    return names


def fixed_census_from_metadata(metadata: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Pure byte census over already authenticated tensor metadata."""

    _require(isinstance(metadata, dict) and bool(metadata), "fixed tensor metadata")
    objects: list[dict[str, Any]] = []
    for name in sorted(metadata):
        row = _mapping(metadata[name], f"fixed metadata: {name}")
        dtype = row.get("dtype")
        shape = row.get("shape")
        _require(dtype in ITEM_BYTES, f"fixed metadata dtype: {name}")
        _require(
            isinstance(shape, list)
            and bool(shape)
            and all(type(value) is int and value > 0 for value in shape),
            f"fixed metadata shape: {name}",
        )
        derived_bytes = math.prod(shape) * ITEM_BYTES[dtype]
        _require(row.get("bytes") == derived_bytes, f"fixed metadata bytes: {name}")
        objects.append(
            {
                "tensor": name,
                "dtype": dtype,
                "shape": list(shape),
                "logical_bytes": derived_bytes,
                "allocated_bytes": resident_allocation_bytes(derived_bytes),
                "backing_file": row.get("backing_file"),
                "backing_file_sha256": row.get("backing_file_sha256"),
            }
        )
    logical = sum(row["logical_bytes"] for row in objects)
    fp8 = sum(row["logical_bytes"] for row in objects if row["dtype"] == "F8_E4M3")
    maximum = max(objects, key=lambda row: (row["logical_bytes"], row["tensor"]))
    return {
        "object_count": len(objects),
        "logical_source_bytes": logical,
        "page_aligned_allocation_bytes": sum(row["allocated_bytes"] for row in objects),
        "fp8_code_bytes": fp8,
        "non_fp8_bytes": logical - fp8,
        "largest_object": maximum["tensor"],
        "largest_object_bytes": maximum["logical_bytes"],
        "allocation_alignment_bytes": RESIDENT_ALIGNMENT_BYTES,
        "objects": objects,
    }


def _tensor_metadata(
    checkpoint_root: Path,
    weight_map: dict[str, str],
    receipt_files: dict[str, dict[str, Any]],
    names: list[str],
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for name in names:
        shard = weight_map.get(name)
        record = receipt_files.get(str(shard))
        _require(
            isinstance(shard, str)
            and isinstance(record, dict)
            and record.get("status") == "verified"
            and isinstance(record.get("sha256"), str),
            f"tensor lacks receipt-authenticated shard: {name}",
        )
        shard_path = checkpoint_root / shard
        _require(
            shard_path.is_file() and shard_path.stat().st_size == record.get("bytes"),
            f"backing shard size mismatch: {shard}",
        )
        grouped[shard].append(name)
    result: dict[str, dict[str, Any]] = {}
    for shard in sorted(grouped):
        record = receipt_files[shard]
        with safe_open(checkpoint_root / shard, framework="pt", device="cpu") as source:
            available = set(source.keys())
            for name in sorted(grouped[shard]):
                _require(name in available, f"tensor absent from safetensors header: {name}")
                view = source.get_slice(name)
                dtype = view.get_dtype()
                shape = list(view.get_shape())
                _require(
                    dtype in ITEM_BYTES
                    and bool(shape)
                    and all(type(value) is int and value > 0 for value in shape),
                    f"unsupported fixed tensor metadata: {name}",
                )
                result[name] = {
                    "tensor": name,
                    "dtype": dtype,
                    "shape": shape,
                    "bytes": math.prod(shape) * ITEM_BYTES[dtype],
                    "backing_file": shard,
                    "backing_file_sha256": record["sha256"],
                }
    _require(set(result) == set(names), "fixed tensor metadata cardinality")
    return result


def _authenticate_fixed_census(
    checkpoint_root: Path,
    verification_path: Path,
    offline_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    receipt_hash = _strict_hash(
        verification_path, CHECKPOINT_VERIFICATION_SHA256, "checkpoint receipt"
    )
    index_path = checkpoint_root / INDEX_NAME
    index_hash = _strict_hash(index_path, INDEX_SHA256, "checkpoint tensor index")
    offline_hash = _strict_hash(offline_path, PW0207_OFFLINE_SHA256, "PW-0207 offline")
    receipt = _json(verification_path, "checkpoint receipt")
    index = _json(index_path, "checkpoint tensor index")
    offline = _json(offline_path, "PW-0207 offline")
    receipt_rows = _sequence(receipt.get("files"), "checkpoint receipt files")
    receipt_files: dict[str, dict[str, Any]] = {}
    for raw in receipt_rows:
        row = _mapping(raw, "checkpoint receipt row")
        path = row.get("path")
        _require(isinstance(path, str) and path not in receipt_files, "receipt path identity")
        receipt_files[path] = row
    _require(
        receipt.get("schema_version") == 1
        and receipt.get("evidence_class") == "local_checkpoint_lock_verification"
        and receipt.get("complete") is True
        and receipt.get("require_complete") is True
        and receipt.get("repository") == "XiaomiMiMo/MiMo-V2.5"
        and receipt.get("revision") == REVISION
        and receipt.get("missing_files") == []
        and receipt.get("verified_files") == len(receipt_rows)
        and all(row.get("status") == "verified" for row in receipt_rows),
        "checkpoint receipt identity",
    )
    _require(
        receipt_files.get(INDEX_NAME, {}).get("sha256") == INDEX_SHA256
        and receipt_files[INDEX_NAME].get("bytes") == index_path.stat().st_size,
        "checkpoint index receipt identity",
    )
    weight_map = index.get("weight_map")
    _require(
        isinstance(weight_map, dict)
        and len(weight_map) == 73_081
        and index.get("metadata")
        == {"save_format": "fp8", "total_size": 315_031_102_208, "tp_size": 4},
        "checkpoint index schema",
    )
    names = fixed_tensor_names(weight_map)
    _require(len(names) == FIXED_OBJECT_COUNT, "fixed tensor count")
    metadata = _tensor_metadata(checkpoint_root, weight_map, receipt_files, names)
    census = fixed_census_from_metadata(metadata)
    _require(
        census["object_count"] == FIXED_OBJECT_COUNT
        and census["logical_source_bytes"] == FIXED_LOGICAL_BYTES
        and census["page_aligned_allocation_bytes"] == FIXED_ALLOCATED_BYTES
        and census["fp8_code_bytes"] == FIXED_FP8_CODE_BYTES
        and census["non_fp8_bytes"] == FIXED_NON_FP8_BYTES
        and census["largest_object"] == FIXED_MAX_OBJECT
        and census["largest_object_bytes"] == FIXED_MAX_OBJECT_BYTES,
        "fixed census constants",
    )
    _require(
        offline.get("schema_version") == 1
        and offline.get("evidence_class")
        == "pw0207_pressure_elastic_offline_residency_falsifier"
        and offline.get("revision") == REVISION
        and offline.get("status") == "passed"
        and offline.get("git_dirty") is False
        and offline.get("identities", {}).get("checkpoint_verification_sha256")
        == CHECKPOINT_VERIFICATION_SHA256
        and offline.get("identities", {}).get("tensor_index_sha256") == INDEX_SHA256,
        "PW-0207 offline identity",
    )
    authority_objects = _sequence(
        _mapping(offline.get("residency_manifest"), "PW-0207 residency manifest").get("objects"),
        "PW-0207 residency objects",
    )
    object_map = {
        row.get("identity"): row
        for row in authority_objects
        if isinstance(row, dict) and isinstance(row.get("identity"), str)
    }
    _require(len(object_map) == len(authority_objects), "PW-0207 duplicate object identity")
    for name in names:
        authority = _mapping(object_map.get(f"tensor:{name}"), f"PW-0207 fixed object: {name}")
        row = metadata[name]
        _require(
            authority.get("category") == "shared_spine"
            and authority.get("source_bytes") == row["bytes"]
            and authority.get("bytes") == resident_allocation_bytes(row["bytes"])
            and authority.get("tensor_metadata_sha256")
            == hashlib.sha256(canonical_json(row)).hexdigest()
            and authority.get("tensors") == [row],
            f"PW-0207 fixed object mismatch: {name}",
        )
    identities = {
        "revision": REVISION,
        "checkpoint_verification_file": str(verification_path),
        "checkpoint_verification_sha256": receipt_hash,
        "tensor_index_file": str(index_path),
        "tensor_index_sha256": index_hash,
        "pw0207_offline_file": str(offline_path),
        "pw0207_offline_sha256": offline_hash,
        "checkpoint_weight_count": len(weight_map),
        "verified_receipt_files": len(receipt_rows),
    }
    return census, identities


def _validate_pw0136_safety(snapshots: Any) -> dict[str, Any]:
    rows = _sequence(snapshots, "PW-0136 safety snapshots")
    _require(bool(rows) and rows[-1].get("phase") == "buffer_release", "PW-0136 release")
    services = {
        name
        for name, pids in _mapping(rows[0].get("protected_service_pids"), "PW-0136 services").items()
        if isinstance(pids, list) and pids
    }
    _require(bool(services), "PW-0136 protected services")
    summary = {
        "snapshot_count": len(rows),
        "minimum_system_memory_free_percent": min(
            _integer(row.get("system_memory_free_percent"), "PW-0136 free memory", minimum=1)
            for row in rows
        ),
        "maximum_process_peak_resident_bytes": max(
            _integer(row.get("process_peak_resident_bytes"), "PW-0136 peak RSS", minimum=1)
            for row in rows
        ),
        "maximum_process_physical_footprint_bytes": max(
            _integer(row.get("process_physical_footprint_bytes"), "PW-0136 footprint", minimum=1)
            for row in rows
        ),
        "final_process_physical_footprint_bytes": _integer(
            rows[-1].get("process_physical_footprint_bytes"), "PW-0136 final footprint", minimum=1
        ),
        "maximum_swap_growth_bytes": max(
            _integer(row.get("swap_growth_bytes"), "PW-0136 swap growth") for row in rows
        ),
        "maximum_new_throttled_pages": max(
            _integer(row.get("new_throttled_pages"), "PW-0136 throttling") for row in rows
        ),
        "protected_services_remained_resident": all(
            all(_mapping(row.get("protected_service_pids"), "PW-0136 services").get(name) for name in services)
            for row in rows
        ),
        "protected_service_pid_sets_stable": all(
            row.get("protected_service_pids") == rows[0].get("protected_service_pids")
            for row in rows
        ),
    }
    _require(
        summary["minimum_system_memory_free_percent"] >= 20
        and summary["maximum_process_peak_resident_bytes"] <= 8 * 1024**3
        and summary["maximum_process_physical_footprint_bytes"] <= 8 * 1024**3
        and summary["final_process_physical_footprint_bytes"] <= 4 * 1024**3
        and summary["maximum_swap_growth_bytes"] <= 512 * 1024**2
        and summary["maximum_new_throttled_pages"] == 0
        and summary["protected_services_remained_resident"]
        and summary["protected_service_pid_sets_stable"],
        "PW-0136 safety mismatch",
    )
    return summary


def derive_pw0136_bandwidth(raw: dict[str, Any]) -> dict[str, Any]:
    """Pure reconstruction of the exact and favorable PW-0136 constants."""

    raw = _mapping(raw, "PW-0136 raw")
    _require(
        raw.get("schema_version") == 1
        and raw.get("commit") == PW0136_CAPTURE_COMMIT
        and raw.get("metal_device") == "Apple M1"
        and raw.get("semantic") == "mimo_v2_5_layer4_page_aligned_pread_expert_slot_acquisition"
        and raw.get("artifact_manifest_sha256")
        == "40179385a571a19b135a4740122744ae3d8ea2c97ef265ac20968296e98822b8"
        and raw.get("artifact_sha256")
        == "fac61c2cfad4b00248c96a52b68360fecd39e2c912e6ffd6643e3f06ade00d21"
        and raw.get("artifact_bytes") == BANDWIDTH_ARTIFACT_BYTES
        and raw.get("expert_stride_bytes") == 25_214_976
        and raw.get("expert_count") == 8
        and raw.get("worker_counts") == list(PW0136_WORKERS)
        and raw.get("slot_capacity_bytes") == BANDWIDTH_ARTIFACT_BYTES
        and raw.get("slot_alignment_bytes") == 2 * 1024 * 1024
        and raw.get("slot_buffer_pointer_identity") == [True] * 8
        and raw.get("slot_buffer_lengths") == [25_214_976] * 8
        and raw.get("batch_size") == 1
        and raw.get("concurrency") == 1
        and raw.get("accepted_tokens") == 0
        and raw.get("A") == 0
        and raw.get("U") == 8
        and raw.get("performance_claim") is None,
        "PW-0136 source identity",
    )
    selected_experts = _sequence(raw.get("selected_experts"), "PW-0136 experts")
    _require(
        len(selected_experts) == 8
        and len(set(selected_experts)) == 8
        and all(type(expert) is int and 0 <= expert < 256 for expert in selected_experts),
        "PW-0136 expert identities",
    )
    trials = _sequence(raw.get("trials"), "PW-0136 trials")
    _require(len(trials) == 24, "PW-0136 trial cardinality")
    distributions: dict[str, dict[int, list[float]]] = {
        "cold": {},
        "warm": {},
    }
    for state in ("cold", "warm"):
        for workers in PW0136_WORKERS:
            rows = [
                _mapping(row, "PW-0136 trial")
                for row in trials
                if row.get("cache_state") == state and row.get("workers") == workers
            ]
            _require(
                sorted(row.get("repetition") for row in rows) == [0, 1, 2],
                "PW-0136 interleaved trial identity",
            )
            walls: list[float] = []
            for row in rows:
                _require(
                    row.get("requested_bytes") == BANDWIDTH_ARTIFACT_BYTES
                    and row.get("returned_bytes") == BANDWIDTH_ARTIFACT_BYTES
                    and row.get("pread_calls") == 8
                    and row.get("slot_stream_sha256")
                    == "fac61c2cfad4b00248c96a52b68360fecd39e2c912e6ffd6643e3f06ade00d21",
                    "PW-0136 transfer integrity",
                )
                expert_reads = _sequence(row.get("expert_reads"), "PW-0136 expert reads")
                _require(
                    len(expert_reads) == 8
                    and all(isinstance(item, dict) for item in expert_reads),
                    "PW-0136 expert transfer integrity",
                )
                for slot, item in enumerate(expert_reads):
                    _require(
                        item.get("expert") == selected_experts[slot]
                        and item.get("slot") == slot
                        and item.get("source_offset") == slot * 25_214_976
                        and item.get("requested_bytes") == 25_214_976
                        and item.get("returned_bytes") == 25_214_976
                        and item.get("pread_calls") == 1
                        and type(item.get("wall_ms")) in (int, float)
                        and not isinstance(item.get("wall_ms"), bool)
                        and math.isfinite(float(item["wall_ms"]))
                        and float(item["wall_ms"]) > 0.0,
                        "PW-0136 expert transfer integrity",
                    )
                activity = _mapping(row.get("activity"), "PW-0136 activity")
                _require(
                    (
                        state == "cold"
                        and activity.get("disk_bytes_read") >= 0.95 * BANDWIDTH_ARTIFACT_BYTES
                    )
                    or (state == "warm" and activity.get("disk_bytes_read") == 0),
                    "PW-0136 cache-state physical reads",
                )
                wall = row.get("transfer_wall_ms")
                _require(
                    type(wall) in (int, float)
                    and not isinstance(wall, bool)
                    and math.isfinite(float(wall))
                    and float(wall) > 0,
                    "PW-0136 transfer wall",
                )
                walls.append(float(wall))
            distributions[state][workers] = walls
    cold_medians = {
        workers: float(statistics.median(distributions["cold"][workers]))
        for workers in PW0136_WORKERS
    }
    selected_workers = min(PW0136_WORKERS, key=lambda value: (cold_medians[value], value))
    exact_ms = cold_medians[selected_workers]
    exact = BANDWIDTH_ARTIFACT_BYTES / (exact_ms / 1000.0)
    favorable = BANDWIDTH_ARTIFACT_BYTES / (BANDWIDTH_ROUNDED_MEDIAN_MS / 1000.0)
    _require(
        selected_workers == 2
        and exact_ms == BANDWIDTH_EXACT_MEDIAN_MS
        and math.isclose(exact, BANDWIDTH_EXACT_BYTES_PER_SECOND, rel_tol=0.0, abs_tol=1.0e-6)
        and math.isclose(
            favorable, BANDWIDTH_FAVORABLE_BYTES_PER_SECOND, rel_tol=0.0, abs_tol=1.0e-6
        ),
        "PW-0136 bandwidth constants",
    )
    safety = _validate_pw0136_safety(raw.get("safety_snapshots"))
    return {
        "artifact_bytes": BANDWIDTH_ARTIFACT_BYTES,
        "selected_workers": selected_workers,
        "raw_exact_median_ms": exact_ms,
        "raw_exact_bytes_per_second": exact,
        "rounded_historical_median_ms": BANDWIDTH_ROUNDED_MEDIAN_MS,
        "candidate_favorable_bytes_per_second": favorable,
        "cold_trial_walls_ms": distributions["cold"][selected_workers],
        "safety": safety,
    }


def _authenticate_bandwidth(raw_path: Path, analysis_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    raw_hash = _strict_hash(raw_path, PW0136_RAW_SHA256, "PW-0136 raw")
    analysis_hash = _strict_hash(analysis_path, PW0136_ANALYSIS_SHA256, "PW-0136 analysis")
    raw = _json(raw_path, "PW-0136 raw")
    analysis = _json(analysis_path, "PW-0136 analysis")
    result = derive_pw0136_bandwidth(raw)
    gate = _mapping(analysis.get("physical_continuation_gate"), "PW-0136 analysis gate")
    _require(
        analysis.get("schema_version") == 1
        and analysis.get("evidence_class")
        == "pw0136_validated_page_aligned_pread_acquisition_rejection"
        and analysis.get("source_report_sha256") == PW0136_RAW_SHA256
        and analysis.get("evidence_valid") is True
        and analysis.get("experiment_passed") is False
        and analysis.get("performance_claim") is None
        and gate.get("selected_workers") == result["selected_workers"]
        and gate.get("cold_median_ms") == result["raw_exact_median_ms"]
        and gate.get("passes") is False
        and analysis.get("safety") == result["safety"],
        "PW-0136 validated analysis identity",
    )
    identities = {
        "pw0136_raw_file": str(raw_path),
        "pw0136_raw_sha256": raw_hash,
        "pw0136_analysis_file": str(analysis_path),
        "pw0136_analysis_sha256": analysis_hash,
    }
    return result, identities


def authenticate_prismwing_storage(
    checkpoint_root: Path,
    verification_path: Path = CHECKPOINT_VERIFICATION_PATH,
    offline_path: Path = PW0207_OFFLINE_PATH,
    bandwidth_raw_path: Path = PW0136_RAW_PATH,
    bandwidth_analysis_path: Path = PW0136_ANALYSIS_PATH,
) -> dict[str, Any]:
    """Authenticate and rederive the fixed census and storage bandwidth."""

    checkpoint_root = checkpoint_root.resolve()
    verification_path = verification_path.resolve()
    offline_path = offline_path.resolve()
    bandwidth_raw_path = bandwidth_raw_path.resolve()
    bandwidth_analysis_path = bandwidth_analysis_path.resolve()
    fixed, fixed_identities = _authenticate_fixed_census(
        checkpoint_root, verification_path, offline_path
    )
    bandwidth, bandwidth_identities = _authenticate_bandwidth(
        bandwidth_raw_path, bandwidth_analysis_path
    )
    return {
        "identities": {**fixed_identities, **bandwidth_identities},
        "fixed": fixed,
        "bandwidth": bandwidth,
    }
