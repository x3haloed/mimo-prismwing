#!/usr/bin/env python3
"""Run PW-0113 exact fine-grained SwiGLU neuron canonicalization."""

from __future__ import annotations

import argparse
import hashlib
import json
import mmap
from pathlib import Path
import shutil
import struct
import subprocess
import tempfile
import time
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

try:
    from tools.host_safety import HostSafetyMonitor
    from tools.openrouter_reference import atomic_write_new, canonical_json
except ModuleNotFoundError:
    from host_safety import HostSafetyMonitor
    from openrouter_reference import atomic_write_new, canonical_json


ARTIFACT_SHA256 = "fac61c2cfad4b00248c96a52b68360fecd39e2c912e6ffd6643e3f06ade00d21"
MANIFEST_SHA256 = "40179385a571a19b135a4740122744ae3d8ea2c97ef265ac20968296e98822b8"
CONTRACT_COMMIT = "e6cd914bee4b448d04864e2473e4e573698756d3"
FP8_FIXTURE_SHA256 = "feb5d20d36a561e9011563edf6896216f49cbea6023a8689c58be39ce3c21a67"
INTERMEDIATE = 2048
HIDDEN = 4096
SCALE_BLOCK = 128
SCALE_GROUPS = 32
WEIGHT_COMPONENT_BYTES = HIDDEN
SCALE_COMPONENT_BYTES = SCALE_GROUPS * 4
COMPONENT_BYTES = (
    WEIGHT_COMPONENT_BYTES,
    SCALE_COMPONENT_BYTES,
    WEIGHT_COMPONENT_BYTES,
    SCALE_COMPONENT_BYTES,
    WEIGHT_COMPONENT_BYTES,
    SCALE_COMPONENT_BYTES,
)
NEURON_BYTES = sum(COMPONENT_BYTES)
SOURCE_EXPERT_BYTES = 25_171_968
EXPANDED_EXPERT_BYTES = INTERMEDIATE * NEURON_BYTES
EXPANDED_OVERHEAD_BYTES = EXPANDED_EXPERT_BYTES - SOURCE_EXPERT_BYTES
PERMUTATION_BYTES = INTERMEDIATE * 2
EXPERTS = 8
LOGICAL_BYTES = EXPERTS * EXPANDED_EXPERT_BYTES
SOURCE_LOGICAL_BYTES = EXPERTS * SOURCE_EXPERT_BYTES
POPCOUNT = np.array([value.bit_count() for value in range(256)], dtype=np.uint8)


def sha256_bytes(data: bytes | bytearray | memoryview) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fp8_lut(path: Path) -> np.ndarray:
    if sha256_file(path) != FP8_FIXTURE_SHA256:
        raise ValueError("FP8 exhaustive fixture SHA-256 mismatch")
    payload = json.loads(path.read_text())
    bits = payload.get("expected_f32_bits")
    if not isinstance(bits, list) or len(bits) != 256:
        raise ValueError("FP8 exhaustive fixture shape mismatch")
    values = np.array(bits, dtype=np.uint32).view(np.float32)
    return values


def components(record: bytes | memoryview) -> tuple[memoryview, ...]:
    if len(record) != NEURON_BYTES:
        raise ValueError("neuron record byte count mismatch")
    view = memoryview(record)
    result = []
    cursor = 0
    for length in COMPONENT_BYTES:
        result.append(view[cursor : cursor + length])
        cursor += length
    if cursor != NEURON_BYTES:
        raise ValueError("neuron component ledger mismatch")
    return tuple(result)


def scale_row(data: memoryview, row: int) -> bytes:
    start = row * SCALE_COMPONENT_BYTES
    end = start + SCALE_COMPONENT_BYTES
    if row < 0 or len(data) != 16 * SCALE_COMPONENT_BYTES or end > len(data):
        raise ValueError("gate/up scale row layout mismatch")
    return bytes(data[start:end])


def down_scale_column(data: memoryview, column: int) -> bytes:
    if column < 0 or column >= 16 or len(data) != SCALE_GROUPS * 16 * 4:
        raise ValueError("down scale column layout mismatch")
    result = bytearray(SCALE_COMPONENT_BYTES)
    for row in range(SCALE_GROUPS):
        source = (row * 16 + column) * 4
        result[row * 4 : (row + 1) * 4] = data[source : source + 4]
    return bytes(result)


def down_weight_column(data: memoryview, column: int) -> bytes:
    if column < 0 or column >= INTERMEDIATE or len(data) != HIDDEN * INTERMEDIATE:
        raise ValueError("down weight column layout mismatch")
    result = bytearray(HIDDEN)
    for row in range(HIDDEN):
        result[row] = data[row * INTERMEDIATE + column]
    return bytes(result)


def extract_neurons(tensors: dict[str, memoryview]) -> list[bytes]:
    expected = {
        "gate/weight": INTERMEDIATE * HIDDEN,
        "gate/scale": 16 * SCALE_GROUPS * 4,
        "up/weight": INTERMEDIATE * HIDDEN,
        "up/scale": 16 * SCALE_GROUPS * 4,
        "down/weight": HIDDEN * INTERMEDIATE,
        "down/scale": SCALE_GROUPS * 16 * 4,
    }
    if set(tensors) != set(expected) or any(len(tensors[key]) != size for key, size in expected.items()):
        raise ValueError("expert tensor shape or byte layout mismatch")
    result = []
    for neuron in range(INTERMEDIATE):
        scale_group = neuron // SCALE_BLOCK
        gate_start = neuron * HIDDEN
        record = b"".join(
            (
                bytes(tensors["gate/weight"][gate_start : gate_start + HIDDEN]),
                scale_row(tensors["gate/scale"], scale_group),
                bytes(tensors["up/weight"][gate_start : gate_start + HIDDEN]),
                scale_row(tensors["up/scale"], scale_group),
                down_weight_column(tensors["down/weight"], neuron),
                down_scale_column(tensors["down/scale"], scale_group),
            )
        )
        if len(record) != NEURON_BYTES:
            raise ValueError("neuron record construction mismatch")
        result.append(record)
    if sum(map(len, result)) != EXPANDED_EXPERT_BYTES:
        raise ValueError("expanded expert byte ledger mismatch")
    return result


def collapse_identical_scale_rows(rows: list[bytes], role: str) -> bytes:
    if len(rows) != INTERMEDIATE or any(len(row) != SCALE_COMPONENT_BYTES for row in rows):
        raise ValueError(f"{role} scale replica shape mismatch")
    output = bytearray(16 * SCALE_COMPONENT_BYTES)
    for group in range(16):
        replicas = rows[group * SCALE_BLOCK : (group + 1) * SCALE_BLOCK]
        if any(replica != replicas[0] for replica in replicas[1:]):
            raise ValueError(f"{role} scale replicas differ")
        output[group * SCALE_COMPONENT_BYTES : (group + 1) * SCALE_COMPONENT_BYTES] = replicas[0]
    return bytes(output)


def collapse_down_scale_columns(columns: list[bytes]) -> bytes:
    grouped = collapse_identical_scale_rows(columns, "down")
    output = bytearray(SCALE_GROUPS * 16 * 4)
    for group in range(16):
        vector = grouped[group * SCALE_COMPONENT_BYTES : (group + 1) * SCALE_COMPONENT_BYTES]
        for row in range(SCALE_GROUPS):
            destination = (row * 16 + group) * 4
            output[destination : destination + 4] = vector[row * 4 : (row + 1) * 4]
    return bytes(output)


def reconstruct_tensors(records: list[bytes]) -> dict[str, bytes]:
    if len(records) != INTERMEDIATE:
        raise ValueError("inverse canonicalization requires 2,048 neurons")
    gate_weight = bytearray(INTERMEDIATE * HIDDEN)
    up_weight = bytearray(INTERMEDIATE * HIDDEN)
    down_weight = bytearray(HIDDEN * INTERMEDIATE)
    gate_scales = []
    up_scales = []
    down_scales = []
    for neuron, record in enumerate(records):
        gate_w, gate_s, up_w, up_s, down_w, down_s = components(record)
        start = neuron * HIDDEN
        gate_weight[start : start + HIDDEN] = gate_w
        up_weight[start : start + HIDDEN] = up_w
        for row in range(HIDDEN):
            down_weight[row * INTERMEDIATE + neuron] = down_w[row]
        gate_scales.append(bytes(gate_s))
        up_scales.append(bytes(up_s))
        down_scales.append(bytes(down_s))
    return {
        "gate/weight": bytes(gate_weight),
        "gate/scale": collapse_identical_scale_rows(gate_scales, "gate"),
        "up/weight": bytes(up_weight),
        "up/scale": collapse_identical_scale_rows(up_scales, "up"),
        "down/weight": bytes(down_weight),
        "down/scale": collapse_down_scale_columns(down_scales),
    }


def decoded_projection_features(codes: memoryview, scale_bytes: memoryview, lut: np.ndarray) -> np.ndarray:
    raw = np.frombuffer(codes, dtype=np.uint8).reshape(SCALE_GROUPS, SCALE_BLOCK)
    scales = np.frombuffer(scale_bytes, dtype="<f4").astype(np.float64)
    if len(scales) != SCALE_GROUPS or not np.isfinite(scales).all():
        raise ValueError("neuron scale vector is invalid")
    values = lut[raw].astype(np.float64) * scales[:, None]
    if not np.isfinite(values).all():
        raise ValueError("neuron contains non-finite dequantized weights")
    return np.stack(
        (
            values.sum(axis=1),
            np.abs(values).sum(axis=1),
            np.square(values).sum(axis=1),
            np.abs(values).max(axis=1),
        ),
        axis=1,
    ).reshape(-1)


def neuron_features(records: list[bytes], lut: np.ndarray) -> np.ndarray:
    features = np.empty((INTERMEDIATE, 3 * SCALE_GROUPS * 4), dtype=np.float64)
    for neuron, record in enumerate(records):
        gate_w, gate_s, up_w, up_s, down_w, down_s = components(record)
        features[neuron] = np.concatenate(
            (
                decoded_projection_features(gate_w, gate_s, lut),
                decoded_projection_features(up_w, up_s, lut),
                decoded_projection_features(down_w, down_s, lut),
            )
        )
    return features


def assignment(reference: np.ndarray, candidate: np.ndarray) -> tuple[list[int], dict[str, Any]]:
    if reference.shape != (INTERMEDIATE, 384) or candidate.shape != reference.shape:
        raise ValueError("neuron feature matrix shape mismatch")
    center = reference.mean(axis=0)
    scale = reference.std(axis=0)
    scale = np.maximum(scale, 1.0e-12)
    left = (reference - center) / scale
    right = (candidate - center) / scale
    left_norm = np.square(left).sum(axis=1)
    right_norm = np.square(right).sum(axis=1)
    distances = left_norm[:, None] + right_norm[None, :] - 2.0 * left @ right.T
    np.maximum(distances, 0.0, out=distances)
    maximum = float(distances.max())
    if not math_is_finite_positive(maximum):
        raise ValueError("neuron assignment distance range is invalid")
    primary = np.rint(distances / maximum * 1_000_000_000).astype(np.int64)
    rows = np.arange(INTERMEDIATE, dtype=np.int64)[:, None]
    columns = np.arange(INTERMEDIATE, dtype=np.int64)[None, :]
    secondary = np.abs(rows - columns) * INTERMEDIATE + columns
    combined = primary * (INTERMEDIATE * INTERMEDIATE) + secondary
    cost_sha256 = hashlib.sha256(combined.astype("<i8", copy=False).tobytes()).hexdigest()
    assigned_rows, assigned_columns = linear_sum_assignment(combined)
    if assigned_rows.tolist() != list(range(INTERMEDIATE)) or sorted(assigned_columns.tolist()) != list(range(INTERMEDIATE)):
        raise ValueError("neuron assignment is not a complete bijection")
    assigned_primary = primary[assigned_rows, assigned_columns]
    return assigned_columns.tolist(), {
        "feature_dimensions": reference.shape[1],
        "normalization": "reference_dimension_mean_and_max_std_1e-12",
        "primary_quantization_levels": 1_000_000_000,
        "tie_break": "absolute_index_distance_then_candidate_index",
        "maximum_squared_standardized_distance": maximum,
        "combined_cost_sha256": cost_sha256,
        "assigned_primary_cost_sum": int(assigned_primary.sum(dtype=np.int64)),
    }


def math_is_finite_positive(value: float) -> bool:
    return np.isfinite(value) and value > 0.0


def xor_bytes(left: bytes, right: bytes) -> bytes:
    if len(left) != len(right):
        raise ValueError("XOR operands differ in length")
    return np.bitwise_xor(np.frombuffer(left, dtype=np.uint8), np.frombuffer(right, dtype=np.uint8)).tobytes()


def write_stream(
    path: Path,
    mode: str,
    experts: list[int],
    records: dict[int, list[bytes]],
    permutations: dict[int, list[int]],
) -> dict[str, Any]:
    reference = experts[0]
    digest = hashlib.sha256()
    written = 0
    xor_popcounts = {str(expert): 0 for expert in experts}
    with path.open("xb") as output:
        def emit(payload: bytes) -> None:
            nonlocal written
            output.write(payload)
            digest.update(payload)
            written += len(payload)

        if mode == "expanded_expert_major":
            for expert in experts:
                for record in records[expert]:
                    emit(record)
        elif mode in ("identity_delta", "aligned_delta"):
            for slot in range(INTERMEDIATE):
                reference_record = records[reference][slot]
                emit(reference_record)
                for expert in experts[1:]:
                    source = slot if mode == "identity_delta" else permutations[expert][slot]
                    residual = xor_bytes(records[expert][source], reference_record)
                    if xor_bytes(residual, reference_record) != records[expert][source]:
                        raise ValueError("neuron XOR reversal failed")
                    xor_popcounts[str(expert)] += int(
                        POPCOUNT[np.frombuffer(residual, dtype=np.uint8)].sum(dtype=np.int64)
                    )
                    emit(residual)
        else:
            raise ValueError(f"unknown stream mode {mode}")
    if written != LOGICAL_BYTES:
        raise ValueError("neuron stream byte ledger mismatch")
    return {"bytes": written, "sha256": digest.hexdigest(), "xor_popcounts": xor_popcounts}


def codec_measurement(zstd: str, stream: Path, level: int) -> dict[str, Any]:
    compressed = stream.with_suffix(f".zst{level}")
    started = time.monotonic()
    subprocess.run([zstd, "-q", "-f", f"-{level}", "-T1", str(stream), "-o", str(compressed)], check=True)
    compression_ms = (time.monotonic() - started) * 1000
    compressed_bytes = compressed.stat().st_size
    compressed_sha256 = sha256_file(compressed)
    subprocess.run([zstd, "-q", "-t", str(compressed)], check=True)
    started = time.monotonic()
    with open("/dev/null", "wb") as sink:
        subprocess.run([zstd, "-q", "-d", "-c", str(compressed)], stdout=sink, check=True)
    decompression_ms = (time.monotonic() - started) * 1000
    verify = hashlib.sha256()
    process = subprocess.Popen([zstd, "-q", "-d", "-c", str(compressed)], stdout=subprocess.PIPE)
    assert process.stdout is not None
    for chunk in iter(lambda: process.stdout.read(8 * 1024**2), b""):
        verify.update(chunk)
    if process.wait() != 0 or verify.hexdigest() != sha256_file(stream):
        raise ValueError("codec decompression identity mismatch")
    compressed.unlink()
    ratio_to_expanded = compressed_bytes / LOGICAL_BYTES
    ratio_to_source = compressed_bytes / SOURCE_LOGICAL_BYTES
    return {
        "level": level,
        "threads": 1,
        "compressed_bytes": compressed_bytes,
        "compressed_sha256": compressed_sha256,
        "compressed_ratio_to_expanded": ratio_to_expanded,
        "compressed_ratio_to_source": ratio_to_source,
        "compression_ms": compression_ms,
        "decompression_ms": decompression_ms,
        "optimistic_transformed_bound_ms": 58.033833 * ratio_to_source + decompression_ms,
    }


def run(arguments: argparse.Namespace) -> dict[str, Any]:
    if arguments.output.exists():
        raise ValueError(f"refusing to overwrite {arguments.output}")
    if sha256_file(arguments.manifest) != MANIFEST_SHA256 or sha256_file(arguments.artifact) != ARTIFACT_SHA256:
        raise ValueError("PW-0113 artifact authority mismatch")
    if EXPANDED_OVERHEAD_BYTES + PERMUTATION_BYTES > SOURCE_EXPERT_BYTES * 0.10:
        raise ValueError("expanded representation exceeds frozen overhead gate")
    manifest = json.loads(arguments.manifest.read_text())
    if manifest.get("artifact_sha256") != ARTIFACT_SHA256 or manifest.get("artifact_bytes") != 201_719_808 or len(manifest.get("tensors", [])) != 48:
        raise ValueError("PW-0113 artifact manifest layout mismatch")
    zstd = shutil.which("zstd")
    if not zstd:
        raise ValueError("zstd executable is unavailable")
    zstd_version = subprocess.run([zstd, "--version"], check=True, text=True, capture_output=True).stdout.strip()
    lut = fp8_lut(arguments.fp8_fixture)
    safety = HostSafetyMonitor()
    started = time.monotonic()
    experts = sorted(manifest["selected_experts"])
    if len(experts) != EXPERTS:
        raise ValueError("PW-0113 requires eight selected experts")
    tensor_records = {
        (record["expert"], f'{record["projection"]}/{record["role"]}'): record
        for record in manifest["tensors"]
    }
    records: dict[int, list[bytes]] = {}
    source_hashes: dict[int, dict[str, str]] = {}
    features: dict[int, np.ndarray] = {}
    with arguments.artifact.open("rb") as source:
        mapping = mmap.mmap(source.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            for expert in experts:
                tensors = {}
                source_hashes[expert] = {}
                for key in ("gate/weight", "gate/scale", "up/weight", "up/scale", "down/weight", "down/scale"):
                    record = tensor_records.get((expert, key))
                    if record is None:
                        raise ValueError(f"expert {expert} lacks {key}")
                    start, end = record["artifact_metadata"]["data_offsets"]
                    view = memoryview(mapping)[start:end]
                    if sha256_bytes(view) != record["artifact_tensor_sha256"]:
                        raise ValueError(f"expert {expert} {key} hash mismatch")
                    tensors[key] = view
                    source_hashes[expert][key] = record["artifact_tensor_sha256"]
                records[expert] = extract_neurons(tensors)
                if {key: sha256_bytes(value) for key, value in reconstruct_tensors(records[expert]).items()} != source_hashes[expert]:
                    raise ValueError(f"expert {expert} expanded reconstruction mismatch")
                features[expert] = neuron_features(records[expert], lut)
                del tensors
                del view
            safety.checkpoint("expanded_records_and_features_extracted")
        finally:
            mapping.close()
    if sum(len(record) for expert in experts for record in records[expert]) != LOGICAL_BYTES:
        raise ValueError("expanded selected-route byte ledger mismatch")
    reference = experts[0]
    permutations = {reference: list(range(INTERMEDIATE))}
    assignment_evidence = {}
    peak_assignment_matrix_bytes = 0
    for expert in experts[1:]:
        permutation, evidence = assignment(features[reference], features[expert])
        permutations[expert] = permutation
        assignment_evidence[str(expert)] = evidence
        peak_assignment_matrix_bytes = max(
            peak_assignment_matrix_bytes,
            INTERMEDIATE * INTERMEDIATE * 8 * 3,
        )
        recovered: list[bytes | None] = [None] * INTERMEDIATE
        for slot, source in enumerate(permutation):
            recovered[source] = records[expert][source]
        if any(record is None for record in recovered):
            raise ValueError("inverse neuron assignment contains a hole")
        reconstructed = reconstruct_tensors([record for record in recovered if record is not None])
        if {key: sha256_bytes(value) for key, value in reconstructed.items()} != source_hashes[expert]:
            raise ValueError(f"expert {expert} inverse neuron reconstruction mismatch")
        safety.checkpoint(f"expert_{expert}_assignment_verified")
    del features
    stream_results = {}
    with tempfile.TemporaryDirectory(prefix="pw0113-", dir=arguments.output.parent) as temporary:
        temporary_path = Path(temporary)
        for mode in ("expanded_expert_major", "identity_delta", "aligned_delta"):
            stream = temporary_path / f"{mode}.bin"
            result = write_stream(stream, mode, experts, records, permutations)
            result["codec"] = [codec_measurement(zstd, stream, level) for level in (1, 19)]
            stream_results[mode] = result
            stream.unlink()
            safety.checkpoint(f"{mode}_codec_measurements_complete")
    del records
    safety.release_checkpoint("neuron_canonicalization_buffers_released", ["expanded neuron records", "features", "assignment matrices", "temporary streams"])
    return {
        "schema_version": 1,
        "evidence_class": "pw0113_exact_selected_expert_neuron_canonicalization",
        "contract_commit": CONTRACT_COMMIT,
        "implementation_commit": arguments.commit,
        "artifact_manifest_sha256": MANIFEST_SHA256,
        "artifact_sha256": ARTIFACT_SHA256,
        "fp8_fixture_sha256": FP8_FIXTURE_SHA256,
        "experts": experts,
        "reference_expert": reference,
        "neurons_per_expert": INTERMEDIATE,
        "neuron_bytes": NEURON_BYTES,
        "source_expert_bytes": SOURCE_EXPERT_BYTES,
        "expanded_expert_bytes": EXPANDED_EXPERT_BYTES,
        "expanded_overhead_bytes_per_expert": EXPANDED_OVERHEAD_BYTES,
        "permutation_bytes_per_expert": PERMUTATION_BYTES,
        "representation_overhead_fraction": (EXPANDED_OVERHEAD_BYTES + PERMUTATION_BYTES) / SOURCE_EXPERT_BYTES,
        "source_logical_bytes": SOURCE_LOGICAL_BYTES,
        "expanded_logical_bytes": LOGICAL_BYTES,
        "permutations_reference_slot_to_source_neuron": permutations,
        "assignment_evidence": assignment_evidence,
        "peak_assignment_matrix_bytes": peak_assignment_matrix_bytes,
        "source_tensor_hashes": source_hashes,
        "streams": stream_results,
        "codec_binary": zstd,
        "codec_version": zstd_version,
        "safety_snapshots": safety.evidence(),
        "complete_wall_ms": (time.monotonic() - started) * 1000,
        "accepted_tokens": 0,
        "performance_claim": None,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--fp8-fixture", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--commit", required=True)
    arguments = parser.parse_args()
    try:
        result = run(arguments)
        atomic_write_new(arguments.output, canonical_json(result))
        print(json.dumps({"output": str(arguments.output), "complete_wall_ms": result["complete_wall_ms"]}))
        return 0
    except (OSError, ValueError, KeyError, TypeError, subprocess.CalledProcessError, json.JSONDecodeError) as error:
        print(json.dumps({"error": str(error)}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
