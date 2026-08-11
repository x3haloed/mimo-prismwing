#!/usr/bin/env python3
"""Census deterministic remote FP8 quantization-block samples with bounded I/O."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import struct
import time
from typing import Any, Callable

try:
    from tools.openrouter_reference import atomic_write_new, canonical_json
    from tools.remote_safetensors_audit import fetch_range, load_lock
except ModuleNotFoundError:
    from openrouter_reference import atomic_write_new, canonical_json
    from remote_safetensors_audit import fetch_range, load_lock


SCHEMA_VERSION = 1
BLOCK = 128
FetchRange = Callable[[str, str, str, int, int], bytes]


def retry_fetch_range(repository: str, revision: str, path: str, start: int, end: int) -> bytes:
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            return fetch_range(repository, revision, path, start, end)
        except (OSError, ValueError) as error:
            last_error = error
            if attempt < 2:
                time.sleep(0.5 * (attempt + 1))
    assert last_error is not None
    raise last_error


def decode_e4m3fn(code: int) -> float:
    if code < 0 or code > 255:
        raise ValueError("FP8 code must fit in one byte")
    sign = -1.0 if code & 0x80 else 1.0
    exponent = (code >> 3) & 0xF
    mantissa = code & 0x7
    if exponent == 0:
        return sign * (mantissa / 8.0) * (2.0 ** -6)
    if exponent == 0xF and mantissa == 0x7:
        return math.nan
    return sign * (1.0 + mantissa / 8.0) * (2.0 ** (exponent - 7))


def entropy(counts: Counter[int]) -> float:
    total = sum(counts.values())
    return -sum((count / total) * math.log2(count / total) for count in counts.values())


def component_entropy(counts: Counter[int], shift: int, mask: int) -> float:
    projected: Counter[int] = Counter()
    for value, count in counts.items():
        projected[(value >> shift) & mask] += count
    return entropy(projected)


def component_counts(counts: Counter[int], shift: int, mask: int) -> Counter[int]:
    projected: Counter[int] = Counter()
    for value, count in counts.items():
        projected[(value >> shift) & mask] += count
    return projected


def exact_palette_bytes(symbol_count: int, values: int = BLOCK * BLOCK) -> int | None:
    bits = max(1, (symbol_count - 1).bit_length())
    if bits >= 8:
        return None
    return (values * bits + 7) // 8 + symbol_count


def escape_bytes(counts: Counter[int], common_symbols: int) -> int:
    """Ideal packed fixed-code+literal size, excluding offset-table metadata."""
    values = sum(counts.values())
    covered = sum(count for _, count in counts.most_common(common_symbols))
    code_bits = (common_symbols + 1 - 1).bit_length()
    escaped = values - covered
    return (values * code_bits + escaped * 8 + 7) // 8 + common_symbols


def analyze_block(payload: bytes) -> dict[str, Any]:
    if len(payload) != BLOCK * BLOCK:
        raise ValueError("FP8 block must contain exactly 128x128 bytes")
    counts = Counter(payload)
    total = len(payload)
    values = [decode_e4m3fn(value) for value in payload]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("FP8 block contains E4M3FN NaN codes")
    affine_error = 0.0
    reference_squared = sum(value * value for value in values)
    for row in range(BLOCK):
        row_values = values[row * BLOCK : (row + 1) * BLOCK]
        minimum = min(row_values)
        maximum = max(row_values)
        step = (maximum - minimum) / 63.0
        if step:
            for value in row_values:
                code = min(63, max(0, round((value - minimum) / step)))
                reconstructed = minimum + code * step
                affine_error += (value - reconstructed) ** 2
    signs = component_counts(counts, 7, 0x1)
    exponents = component_counts(counts, 3, 0xF)
    mantissas = component_counts(counts, 0, 0x7)
    result: dict[str, Any] = {
        "symbol_counts": [counts[value] for value in range(256)],
        "reference_squared_sum": reference_squared,
        "affine6_rtn_squared_error": affine_error,
        "distinct_symbols": len(counts),
        "entropy_bits_per_weight": entropy(counts),
        "zero_frequency": counts[0] / total,
        "sign_entropy": entropy(signs),
        "exponent_entropy": entropy(exponents),
        "mantissa_entropy": entropy(mantissas),
        "distinct_signs": len(signs),
        "distinct_exponents": len(exponents),
        "distinct_mantissas": len(mantissas),
        "exponent_top_coverage": {
            str(k): sum(count for _, count in exponents.most_common(k)) / total
            for k in (3, 7)
        },
        "exponent_top7_escape_bytes": (
            (total * 7 + (total - sum(count for _, count in exponents.most_common(7))) * 4 + 7 * 4 + 7) // 8
        ),
        "exact_split_bits_per_weight": (
            1 + max(1, (len(exponents) - 1).bit_length()) + 3
        ),
        "top_coverage": {
            str(k): sum(count for _, count in counts.most_common(k)) / total
            for k in (8, 15, 31, 63)
        },
    }
    result["exact_palette_bytes"] = {
        str(bits): (
            (BLOCK * BLOCK * bits + 7) // 8 + len(counts)
            if len(counts) <= 1 << bits
            else None
        )
        for bits in (4, 5, 6, 7)
    }
    result["top_k_escape_bytes"] = {
        str(k): escape_bytes(counts, k) for k in (15, 31, 63)
    }
    return result


def analyze_row_tile(payload: bytes, columns: int) -> list[dict[str, Any]]:
    if columns <= 0 or columns % BLOCK or len(payload) != BLOCK * columns:
        raise ValueError("row tile must be 128 complete rows with 128-aligned columns")
    blocks = []
    for column in range(0, columns, BLOCK):
        block = b"".join(
            payload[row * columns + column : row * columns + column + BLOCK]
            for row in range(BLOCK)
        )
        record = analyze_block(block)
        record["column_block"] = column // BLOCK
        blocks.append(record)
    return blocks


def load_header(
    lock: dict[str, Any], shard: str, fetch: FetchRange
) -> tuple[int, dict[str, Any]]:
    files = {item["path"]: item for item in lock["files"]}
    if shard not in files:
        raise ValueError(f"shard absent from lock: {shard}")
    repository = lock["repository"]
    revision = lock["revision"]
    header_bytes = struct.unpack("<Q", fetch(repository, revision, shard, 0, 7))[0]
    if header_bytes <= 0 or 8 + header_bytes > files[shard]["bytes"]:
        raise ValueError(f"invalid safetensors header: {shard}")
    header = json.loads(fetch(repository, revision, shard, 8, 7 + header_bytes))
    if not isinstance(header, dict):
        raise ValueError(f"malformed safetensors header: {shard}")
    return header_bytes, header


def census(
    lock: dict[str, Any],
    index: dict[str, Any],
    samples: list[tuple[str, int]],
    fetch: FetchRange = retry_fetch_range,
) -> dict[str, Any]:
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        raise ValueError("index lacks weight_map")
    headers: dict[str, tuple[int, dict[str, Any]]] = {}
    scale_payloads: dict[str, tuple[list[int], bytes]] = {}
    records = []
    fetched_bytes = 0
    for name, row_block in samples:
        shard = weight_map.get(name)
        if not isinstance(shard, str):
            raise ValueError(f"tensor absent from index: {name}")
        if shard not in headers:
            headers[shard] = load_header(lock, shard, fetch)
            fetched_bytes += 8 + headers[shard][0]
        header_bytes, header = headers[shard]
        metadata = header.get(name)
        if not isinstance(metadata, dict):
            raise ValueError(f"tensor absent from shard header: {name}")
        dtype = metadata.get("dtype")
        shape = metadata.get("shape")
        offsets = metadata.get("data_offsets")
        if (
            dtype != "F8_E4M3"
            or not isinstance(shape, list)
            or len(shape) != 2
            or not all(isinstance(value, int) and value > 0 for value in shape)
            or shape[0] % BLOCK
            or shape[1] % BLOCK
            or not isinstance(offsets, list)
            or len(offsets) != 2
        ):
            raise ValueError(f"unsupported FP8 tensor layout: {name}")
        rows, columns = shape
        if row_block < 0 or row_block >= rows // BLOCK:
            raise ValueError(f"row block out of range for {name}: {row_block}")
        tile_bytes = BLOCK * columns
        absolute_start = 8 + header_bytes + offsets[0] + row_block * tile_bytes
        payload = fetch(
            lock["repository"], lock["revision"], shard,
            absolute_start, absolute_start + tile_bytes - 1,
        )
        fetched_bytes += len(payload)
        scale_name = f"{name}_scale_inv"
        scale_metadata = header.get(scale_name)
        expected_scale_shape = [rows // BLOCK, columns // BLOCK]
        if (
            not isinstance(scale_metadata, dict)
            or scale_metadata.get("dtype") != "F32"
            or scale_metadata.get("shape") != expected_scale_shape
            or not isinstance(scale_metadata.get("data_offsets"), list)
            or len(scale_metadata["data_offsets"]) != 2
        ):
            raise ValueError(f"unsupported source scale layout: {scale_name}")
        if scale_name not in scale_payloads:
            scale_start = 8 + header_bytes + scale_metadata["data_offsets"][0]
            scale_bytes = scale_metadata["data_offsets"][1] - scale_metadata["data_offsets"][0]
            scale_payload = fetch(
                lock["repository"], lock["revision"], shard,
                scale_start, scale_start + scale_bytes - 1,
            )
            fetched_bytes += len(scale_payload)
            scale_payloads[scale_name] = (expected_scale_shape, scale_payload)
        _, scale_payload = scale_payloads[scale_name]
        scale_columns = expected_scale_shape[1]
        scale_row_bytes = scale_payload[row_block * scale_columns * 4 : (row_block + 1) * scale_columns * 4]
        scale_values = list(struct.unpack(f"<{scale_columns}f", scale_row_bytes))
        if not all(math.isfinite(value) and value > 0 for value in scale_values):
            raise ValueError(f"invalid source scales: {scale_name}")
        records.append({
            "tensor": name,
            "shard": shard,
            "shape": shape,
            "row_block": row_block,
            "source_offset": absolute_start,
            "fetched_bytes": len(payload),
            "scale_tensor": scale_name,
            "scale_values": scale_values,
            "blocks": analyze_row_tile(payload, columns),
        })
    return {
        "schema_version": SCHEMA_VERSION,
        "evidence_class": "pinned_remote_deterministic_fp8_row_tile_samples",
        "repository": lock["repository"],
        "revision": lock["revision"],
        "sample_count": len(records),
        "quantization_block_count": sum(len(record["blocks"]) for record in records),
        "network_bytes": fetched_bytes,
        "samples": records,
    }


def parse_sample(value: str) -> tuple[str, int]:
    name, separator, row_block = value.rpartition(":")
    if not separator or not name:
        raise argparse.ArgumentTypeError("sample must be TENSOR:ROW_BLOCK")
    try:
        return name, int(row_block)
    except ValueError as error:
        raise argparse.ArgumentTypeError("ROW_BLOCK must be an integer") from error


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", required=True, type=Path)
    parser.add_argument("--index", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--sample", required=True, action="append", type=parse_sample)
    arguments = parser.parse_args()
    try:
        result = census(
            load_lock(arguments.lock),
            json.loads(arguments.index.read_text(encoding="utf-8")),
            arguments.sample,
        )
        atomic_write_new(arguments.output, canonical_json(result))
        print(arguments.output)
        return 0
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"error: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
