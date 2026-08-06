#!/usr/bin/env python3
"""Run the exact PW-0109 selected-expert block canonicalization experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import mmap
from pathlib import Path
import shutil
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
CONTRACT_COMMIT = "e3d47f1cdbc866cf70a056ba0dfe87b643ee4e82"
BLOCK_NEURONS = 128
INTERMEDIATE = 2048
HIDDEN = 4096
BLOCK_COUNT = INTERMEDIATE // BLOCK_NEURONS
SCALE_BLOCK = 128
GATE_WEIGHT_BLOCK = BLOCK_NEURONS * HIDDEN
GATE_SCALE_BLOCK = HIDDEN // SCALE_BLOCK * 4
UP_WEIGHT_BLOCK = GATE_WEIGHT_BLOCK
UP_SCALE_BLOCK = GATE_SCALE_BLOCK
DOWN_WEIGHT_BLOCK = HIDDEN * BLOCK_NEURONS
DOWN_SCALE_BLOCK = HIDDEN // SCALE_BLOCK * 4
COMPONENT_BYTES = (
    GATE_WEIGHT_BLOCK,
    GATE_SCALE_BLOCK,
    UP_WEIGHT_BLOCK,
    UP_SCALE_BLOCK,
    DOWN_WEIGHT_BLOCK,
    DOWN_SCALE_BLOCK,
)
BLOCK_BYTES = sum(COMPONENT_BYTES)
LOGICAL_BYTES = 8 * BLOCK_COUNT * BLOCK_BYTES
POPCOUNT = np.array([value.bit_count() for value in range(256)], dtype=np.uint8)


def sha256_bytes(data: bytes | bytearray | memoryview) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def components(block: bytes) -> tuple[memoryview, ...]:
    if len(block) != BLOCK_BYTES:
        raise ValueError("canonical neuron block byte count mismatch")
    view = memoryview(block)
    result = []
    cursor = 0
    for length in COMPONENT_BYTES:
        result.append(view[cursor : cursor + length])
        cursor += length
    if cursor != len(block):
        raise ValueError("canonical neuron block component ledger mismatch")
    return tuple(result)


def extract_row_block(
    data: memoryview, block: int, block_rows: int, row_bytes: int
) -> bytes:
    start = block * block_rows * row_bytes
    end = start + block_rows * row_bytes
    if block < 0 or block_rows <= 0 or row_bytes <= 0 or end > len(data):
        raise ValueError("row-block extraction exceeds tensor")
    return bytes(data[start:end])


def extract_column_block(
    data: memoryview,
    block: int,
    rows: int,
    columns: int,
    block_columns: int,
    element_bytes: int,
) -> bytes:
    if (
        block < 0
        or rows <= 0
        or columns <= 0
        or block_columns <= 0
        or element_bytes <= 0
        or len(data) != rows * columns * element_bytes
        or (block + 1) * block_columns > columns
    ):
        raise ValueError("column-block extraction layout mismatch")
    result = bytearray(rows * block_columns * element_bytes)
    width = block_columns * element_bytes
    for row in range(rows):
        source = (row * columns + block * block_columns) * element_bytes
        result[row * width : (row + 1) * width] = data[source : source + width]
    return bytes(result)


def scatter_column_block(
    destination: bytearray,
    payload: memoryview,
    block: int,
    rows: int,
    columns: int,
    block_columns: int,
    element_bytes: int,
) -> None:
    if (
        len(destination) != rows * columns * element_bytes
        or len(payload) != rows * block_columns * element_bytes
        or block < 0
        or (block + 1) * block_columns > columns
    ):
        raise ValueError("column-block scatter layout mismatch")
    width = block_columns * element_bytes
    for row in range(rows):
        destination_start = (row * columns + block * block_columns) * element_bytes
        destination[destination_start : destination_start + width] = payload[
            row * width : (row + 1) * width
        ]


def extract_blocks(tensors: dict[str, memoryview]) -> list[bytes]:
    expected = {
        "gate/weight": INTERMEDIATE * HIDDEN,
        "gate/scale": BLOCK_COUNT * (HIDDEN // SCALE_BLOCK) * 4,
        "up/weight": INTERMEDIATE * HIDDEN,
        "up/scale": BLOCK_COUNT * (HIDDEN // SCALE_BLOCK) * 4,
        "down/weight": HIDDEN * INTERMEDIATE,
        "down/scale": (HIDDEN // SCALE_BLOCK) * BLOCK_COUNT * 4,
    }
    if set(tensors) != set(expected) or any(len(tensors[key]) != size for key, size in expected.items()):
        raise ValueError("expert tensor shape or byte layout mismatch")
    blocks = []
    for block in range(BLOCK_COUNT):
        payload = b"".join(
            (
                extract_row_block(tensors["gate/weight"], block, BLOCK_NEURONS, HIDDEN),
                extract_row_block(tensors["gate/scale"], block, 1, GATE_SCALE_BLOCK),
                extract_row_block(tensors["up/weight"], block, BLOCK_NEURONS, HIDDEN),
                extract_row_block(tensors["up/scale"], block, 1, UP_SCALE_BLOCK),
                extract_column_block(
                    tensors["down/weight"], block, HIDDEN, INTERMEDIATE, BLOCK_NEURONS, 1
                ),
                extract_column_block(
                    tensors["down/scale"], block, HIDDEN // SCALE_BLOCK, BLOCK_COUNT, 1, 4
                ),
            )
        )
        if len(payload) != BLOCK_BYTES:
            raise ValueError("canonical block construction mismatch")
        blocks.append(payload)
    return blocks


def reconstruct_tensors(blocks: list[bytes]) -> dict[str, bytes]:
    if len(blocks) != BLOCK_COUNT:
        raise ValueError("inverse canonicalization requires sixteen blocks")
    gate_weight = bytearray(INTERMEDIATE * HIDDEN)
    gate_scale = bytearray(BLOCK_COUNT * (HIDDEN // SCALE_BLOCK) * 4)
    up_weight = bytearray(INTERMEDIATE * HIDDEN)
    up_scale = bytearray(BLOCK_COUNT * (HIDDEN // SCALE_BLOCK) * 4)
    down_weight = bytearray(HIDDEN * INTERMEDIATE)
    down_scale = bytearray((HIDDEN // SCALE_BLOCK) * BLOCK_COUNT * 4)
    for block_index, block in enumerate(blocks):
        gate_w, gate_s, up_w, up_s, down_w, down_s = components(block)
        gate_start = block_index * GATE_WEIGHT_BLOCK
        scale_start = block_index * GATE_SCALE_BLOCK
        gate_weight[gate_start : gate_start + GATE_WEIGHT_BLOCK] = gate_w
        gate_scale[scale_start : scale_start + GATE_SCALE_BLOCK] = gate_s
        up_weight[gate_start : gate_start + UP_WEIGHT_BLOCK] = up_w
        up_scale[scale_start : scale_start + UP_SCALE_BLOCK] = up_s
        scatter_column_block(
            down_weight, down_w, block_index, HIDDEN, INTERMEDIATE, BLOCK_NEURONS, 1
        )
        scatter_column_block(
            down_scale, down_s, block_index, HIDDEN // SCALE_BLOCK, BLOCK_COUNT, 1, 4
        )
    return {
        "gate/weight": bytes(gate_weight),
        "gate/scale": bytes(gate_scale),
        "up/weight": bytes(up_weight),
        "up/scale": bytes(up_scale),
        "down/weight": bytes(down_weight),
        "down/scale": bytes(down_scale),
    }


def xor_bytes(left: bytes, right: bytes) -> bytes:
    if len(left) != len(right):
        raise ValueError("XOR operands differ in length")
    return np.bitwise_xor(np.frombuffer(left, dtype=np.uint8), np.frombuffer(right, dtype=np.uint8)).tobytes()


def assignment(reference: list[bytes], candidate: list[bytes]) -> tuple[list[int], list[list[int]]]:
    if len(reference) != BLOCK_COUNT or len(candidate) != BLOCK_COUNT:
        raise ValueError("assignment block count mismatch")
    reference_sketches = [np.frombuffer(block[::1024], dtype=np.uint8) for block in reference]
    candidate_sketches = [np.frombuffer(block[::1024], dtype=np.uint8) for block in candidate]
    primary = np.zeros((BLOCK_COUNT, BLOCK_COUNT), dtype=np.int64)
    for row, left in enumerate(reference_sketches):
        for column, right in enumerate(candidate_sketches):
            primary[row, column] = int(POPCOUNT[np.bitwise_xor(left, right)].sum(dtype=np.int64))
    secondary = np.fromfunction(
        lambda row, column: (row - column) ** 2 * 32 + column,
        (BLOCK_COUNT, BLOCK_COUNT),
        dtype=np.int64,
    ).astype(np.int64)
    rows, columns = linear_sum_assignment(primary * 1_000_000 + secondary)
    if rows.tolist() != list(range(BLOCK_COUNT)) or sorted(columns.tolist()) != list(range(BLOCK_COUNT)):
        raise ValueError("assignment is not a complete bijection")
    return columns.tolist(), primary.tolist()


def write_stream(
    path: Path,
    mode: str,
    experts: list[int],
    blocks: dict[int, list[bytes]],
    permutations: dict[int, list[int]],
) -> dict[str, Any]:
    reference = experts[0]
    digest = hashlib.sha256()
    written = 0
    with path.open("xb") as output:
        def emit(payload: bytes) -> None:
            nonlocal written
            output.write(payload)
            digest.update(payload)
            written += len(payload)

        if mode == "unmodified_expert_major":
            for expert in experts:
                for block in blocks[expert]:
                    emit(block)
        elif mode in ("identity_delta", "aligned_delta"):
            for slot in range(BLOCK_COUNT):
                reference_block = blocks[reference][slot]
                emit(reference_block)
                for expert in experts[1:]:
                    source_index = slot if mode == "identity_delta" else permutations[expert][slot]
                    residual = xor_bytes(blocks[expert][source_index], reference_block)
                    if xor_bytes(residual, reference_block) != blocks[expert][source_index]:
                        raise ValueError("XOR reversal failed")
                    emit(residual)
        else:
            raise ValueError(f"unknown stream mode {mode}")
    if written != LOGICAL_BYTES:
        raise ValueError("canonical stream byte ledger mismatch")
    return {"bytes": written, "sha256": digest.hexdigest()}


def codec_measurement(zstd: str, stream: Path, level: int) -> dict[str, Any]:
    compressed = stream.with_suffix(f".zst{level}")
    started = time.monotonic()
    subprocess.run(
        [zstd, "-q", "-f", f"-{level}", "-T1", str(stream), "-o", str(compressed)],
        check=True,
    )
    compression_ms = (time.monotonic() - started) * 1000
    compressed_bytes = compressed.stat().st_size
    compressed_sha256 = sha256_file(compressed)
    subprocess.run([zstd, "-q", "-t", str(compressed)], check=True)
    started = time.monotonic()
    with open("/dev/null", "wb") as sink:
        subprocess.run([zstd, "-q", "-d", "-c", str(compressed)], stdout=sink, check=True)
    decompression_ms = (time.monotonic() - started) * 1000
    compressed.unlink()
    ratio = compressed_bytes / stream.stat().st_size
    return {
        "level": level,
        "threads": 1,
        "compressed_bytes": compressed_bytes,
        "compressed_sha256": compressed_sha256,
        "compressed_ratio": ratio,
        "compression_ms": compression_ms,
        "decompression_ms": decompression_ms,
        "optimistic_transformed_bound_ms": 58.033833 * ratio + decompression_ms,
    }


def run(arguments: argparse.Namespace) -> dict[str, Any]:
    if arguments.output.exists():
        raise ValueError(f"refusing to overwrite {arguments.output}")
    if sha256_file(arguments.manifest) != MANIFEST_SHA256 or sha256_file(arguments.artifact) != ARTIFACT_SHA256:
        raise ValueError("PW-0109 artifact authority mismatch")
    manifest = json.loads(arguments.manifest.read_text())
    if (
        manifest.get("artifact_sha256") != ARTIFACT_SHA256
        or manifest.get("artifact_bytes") != 201_719_808
        or len(manifest.get("tensors", [])) != 48
    ):
        raise ValueError("PW-0109 manifest layout mismatch")
    zstd = shutil.which("zstd")
    if not zstd:
        raise ValueError("zstd executable is unavailable")
    zstd_version = subprocess.run([zstd, "--version"], check=True, text=True, capture_output=True).stdout.strip()
    safety = HostSafetyMonitor()
    started = time.monotonic()
    experts = sorted(manifest["selected_experts"])
    records = {
        (record["expert"], f'{record["projection"]}/{record["role"]}'): record
        for record in manifest["tensors"]
    }
    blocks: dict[int, list[bytes]] = {}
    source_hashes: dict[int, dict[str, str]] = {}
    with arguments.artifact.open("rb") as source:
        mapping = mmap.mmap(source.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            for expert in experts:
                tensors = {}
                source_hashes[expert] = {}
                for key in ("gate/weight", "gate/scale", "up/weight", "up/scale", "down/weight", "down/scale"):
                    record = records.get((expert, key))
                    if record is None:
                        raise ValueError(f"expert {expert} lacks {key}")
                    start, end = record["artifact_metadata"]["data_offsets"]
                    view = memoryview(mapping)[start:end]
                    if sha256_bytes(view) != record["artifact_tensor_sha256"]:
                        raise ValueError(f"expert {expert} {key} hash mismatch")
                    tensors[key] = view
                    source_hashes[expert][key] = record["artifact_tensor_sha256"]
                blocks[expert] = extract_blocks(tensors)
                del tensors
                del view
            safety.checkpoint("canonical_blocks_extracted")
        finally:
            mapping.close()
    if sum(len(block) for expert in experts for block in blocks[expert]) != LOGICAL_BYTES:
        raise ValueError("canonical blocks do not cover the selected tensor bytes")
    reference = experts[0]
    permutations = {reference: list(range(BLOCK_COUNT))}
    sketch_distance_matrices = {}
    exact_assigned_xor_popcount = {reference: 0}
    for expert in experts[1:]:
        permutation, distances = assignment(blocks[reference], blocks[expert])
        permutations[expert] = permutation
        sketch_distance_matrices[expert] = distances
        exact_assigned_xor_popcount[expert] = sum(
            int(
                POPCOUNT[
                    np.bitwise_xor(
                        np.frombuffer(blocks[reference][slot], dtype=np.uint8),
                        np.frombuffer(blocks[expert][source], dtype=np.uint8),
                    )
                ].sum(dtype=np.int64)
            )
            for slot, source in enumerate(permutation)
        )
    for expert in experts:
        recovered = [None] * BLOCK_COUNT
        for slot, source in enumerate(permutations[expert]):
            recovered[source] = blocks[expert][source]
        reconstructed = reconstruct_tensors(recovered)
        if {key: sha256_bytes(value) for key, value in reconstructed.items()} != source_hashes[expert]:
            raise ValueError(f"expert {expert} inverse reconstruction mismatch")
    safety.checkpoint("assignments_and_inverse_reconstruction_verified")
    stream_results = {}
    with tempfile.TemporaryDirectory(prefix="pw0109-", dir=arguments.output.parent) as temporary:
        temporary = Path(temporary)
        for mode in ("unmodified_expert_major", "identity_delta", "aligned_delta"):
            stream = temporary / f"{mode}.bin"
            stream_result = write_stream(stream, mode, experts, blocks, permutations)
            stream_result["codec"] = [codec_measurement(zstd, stream, level) for level in (1, 19)]
            stream_results[mode] = stream_result
            stream.unlink()
            safety.checkpoint(f"{mode}_codec_measurements_complete")
    del blocks
    safety.release_checkpoint("canonicalization_buffers_released", ["block records", "temporary streams"])
    return {
        "schema_version": 1,
        "evidence_class": "pw0109_exact_selected_expert_block_canonicalization",
        "contract_commit": CONTRACT_COMMIT,
        "implementation_commit": arguments.commit,
        "artifact_manifest_sha256": MANIFEST_SHA256,
        "artifact_sha256": ARTIFACT_SHA256,
        "experts": experts,
        "reference_expert": reference,
        "block_neurons": BLOCK_NEURONS,
        "blocks_per_expert": BLOCK_COUNT,
        "block_bytes": BLOCK_BYTES,
        "logical_bytes": LOGICAL_BYTES,
        "permutations_reference_slot_to_source_block": permutations,
        "sketch_stride_bytes": 1024,
        "sketch_distance_matrices": sketch_distance_matrices,
        "exact_assigned_xor_popcount": exact_assigned_xor_popcount,
        "source_tensor_hashes": source_hashes,
        "streams": stream_results,
        "codec_binary": zstd,
        "codec_version": zstd_version,
        "safety_snapshots": safety.evidence(),
        "complete_wall_ms": (time.monotonic() - started) * 1000,
        "accepted_tokens": 0,
        "A": 0,
        "U": 8,
        "performance_claim": None,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--commit", required=True)
    arguments = parser.parse_args()
    if len(arguments.commit) != 40 or any(character not in "0123456789abcdef" for character in arguments.commit):
        raise SystemExit("implementation commit must be lowercase 40-hex")
    try:
        result = run(arguments)
        atomic_write_new(arguments.output, canonical_json(result))
        print(json.dumps({"output": str(arguments.output)}))
        return 0
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        print(json.dumps({"error": str(error)}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
