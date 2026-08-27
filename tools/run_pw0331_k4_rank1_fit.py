#!/usr/bin/env python3
"""Construct PW-0331's fit-only byte-neutral K4 rank-one factors.

This process deliberately cannot load a complete PW-0116 capture.  It reads
only the 108 predeclared expert-96 fit rows with ``pread`` and leaves position
1, validation, and pilot payloads for the separate held-out analyzer.
"""

from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import resource
import secrets
import subprocess
import time
from typing import Any, Callable

import numpy as np

try:
    import llvmlite
    import numba
    from numba import njit, prange, types
    from numba.extending import intrinsic
except ModuleNotFoundError:  # pragma: no cover - exercised by the real fail-closed path
    llvmlite = numba = None
    njit = prange = types = intrinsic = None

try:
    from tools.construct_pw0314_layer4_k4 import selected_rows
    from tools.host_safety import HostSafetyMonitor, HostSafetyViolation
    from tools.openrouter_reference import atomic_write_new, canonical_json
    from tools.reproduce_pw0311_k4_expert import (
        PANEL_CONTRACT_SHA256,
        PANEL_EXPORT_SHA256,
        _load_authority_modules,
        authority_paths,
        sha256_file,
        verify_clean_commit,
    )
except ModuleNotFoundError:
    from construct_pw0314_layer4_k4 import selected_rows
    from host_safety import HostSafetyMonitor, HostSafetyViolation
    from openrouter_reference import atomic_write_new, canonical_json
    from reproduce_pw0311_k4_expert import (
        PANEL_CONTRACT_SHA256,
        PANEL_EXPORT_SHA256,
        _load_authority_modules,
        authority_paths,
        sha256_file,
        verify_clean_commit,
    )


EXPERIMENT_ID = "PW-0331"
SEMANTIC = "m1-native-k4-r1-down-v1"
CONTRACT_COMMIT = "424055c6379864c199f157d2ff2bd48970dbf34d"
CONTRACT_RELATIVE_PATH = "experiments/PW-0331-byte-neutral-k4-rank1-repair.md"
CONTRACT_GIT_BLOB = "6256b039c5adc9fe8f6ef888a2131794087b4591"
CONTRACT_SHA256 = "0a8a7f96e12ab8ec1407f8b2b8ffeb99702cd8385188def3d0cb4ed70a5f9d95"
SERIALIZED_DENSE_CONTROL_RELATIVE_PATH = (
    "evals/fixtures/tiny/pw0331-serialized-dense-control.json"
)
SERIALIZED_DENSE_CONTROL_GIT_BLOB = "d1a5dca8833f228c19602867a1968f620201caf6"
SERIALIZED_DENSE_CONTROL_SHA256 = (
    "d6f3a30271fdafec67941161fb5b096239e51554ea239a7bd47ebe401c36d569"
)
SERIALIZED_DENSE_CONTROL_DIAGNOSTIC_SHA256 = (
    "9c4b542414466664e3f7af9fbe011e1827d4b05da89edaaa88e9cdb90db0e551"
)
SERIALIZED_DENSE_CONTROL_STAGES_SHA256 = (
    "726bb2337ba46443f38af0a87d3c70efd2d868b32ec39fefcfdbbae767086a4e"
)
TARGET_SHA256 = "dda459684c194b03491f36e9b66521ff00c400a6cc38d23a567a5a92ef8fb17d"
RED_LINES_SHA256 = "cc261ad9bd67a865715e72cbbadf3b74c3f1f282e17a8ef86ed02c1a92fb8b36"
CORPUS_SHA256 = "b9df976876d63c1ffbbe0c70507aea8b939a749ce5b1db27cbca0b5d82cf802e"
PW0315_SUMMARY_SHA256 = "07b3d3793a6750a030eb5b7e12a0add1b603d48758a85e6f45b44504e404d0e8"
PW0315_EXPERT96_REPORT_SHA256 = "eb033a9a60f304a4c54069484247b31801a203c70142417afe3055161b940609"
PW0316_REJECTION_SHA256 = "7e5560cf2cdc2abdec8ec1a17af0462f69fa7204f8ba528808ce1f046d0e6ff4"
PW0318_SUMMARY_SHA256 = "a91af31bdea45749c9ae9d5d679260bcbcd8284c238479938206a7e7e0b5eb2f"
PW0318_MANIFEST_SHA256 = "ca2cd8005c3c8f712fabd0b2fc88183d740bd6613efa065cdd4b25738c4924c3"
PW0318_BUNDLE_SHA256 = "e87a0af2aba57f46b6a2f394d70e530533d04c18aa61650afbc8528a4b8bdc35"
PW0318_FIXTURE_SHA256 = "0189a8c15299410537cd43f934c4aefbda1c160e7c9f6920790cabfd812a6706"
CHECKPOINT_REVISION = "63651580ca774f8504f676040460aed3e1244ac1"
CHECKPOINT_RECEIPT_SHA256 = "9ddc8a99755f04ae2ea3c2484f6dd022d3f3a681b5a72c915ee4de833dbb0d03"
CHECKPOINT_INDEX_SHA256 = "f2e1774c9acf9a62338b68c144e6fc7a66495e59f2e64b3078c1b7ef5a196816"
CHECKPOINT_EXPERT96_SHARD_SHA256 = (
    "f8c8ab1b22da717ed0360c8248da84d0f9a58af7a89deeb6d4021a67ae98a046"
)
ZERO_F16_4096_SHA256 = "ad7facb2586fc6e966c004d7d1d16b024f5805ff7cb47c7a85dabd8b48892ca7"
ZERO_F16_8192_SHA256 = "9f1dcbc35c350d6027f98be0f5c8b43b42ca52b7604459c0c42be3aa88913d47"
IMPLEMENTATION_HASHES = {
    "src/k4_source_bundle.rs": "fd863e53514afa0eecaf06ce0a43d7fef93ad88d4d22e465f447b707da81c9cb",
    "src/k4_source_metal.rs": "01f4837d71da370b25e64ccaa9ebce8b4262fcb2999728707551b739fd771c8b",
    "kernels/qtip_k4_bundle_batched.metal": "50c835699e7f80403d8127bdbe19e572acbf89774144f3bc079cd3a9c68b58c8",
    "kernels/mixed_route_reduce.metal": "d20446229683edb5855e6e2b9cf1aadc0183f5d10b976fe52f165cb03384ac84",
}
PANEL_IMPLEMENTATION_SHA256 = (
    "1cf0950e878bcb7f2b5259b346a661ac6e6cc4319bc0a0b39b17ad57130b9106"
)

LAYER = 4
EXPERT = 96
ROWS = 224
HIDDEN = 4096
INTERMEDIATE = 2048
FIT_END = 112
PRIMARY_POSITION = 1
VALIDATION = (112, 168)
PILOT = (168, 224)
EXPECTED_COUNTS = {"fit": 108, "primary": 1, "validation": 56, "pilot": 9}
RCOND = 1e-12
STAGE_A_BASE_SEMANTIC = (
    "serialized-k4-signed-fwht-lane64-tree-output-fwht-finish-stage-a-v1"
)


def stage_a_numerics_authority() -> dict[str, Any]:
    """Return the complete, fail-closed numerical contract emitted by Stage A."""
    if numba is None or llvmlite is None:
        raise RuntimeError("PW-0331 exact serialized Stage A requires numba and llvmlite")
    return {
        "fit_algebra": "float64_svd_weighted_residual_then_pinv",
        "application": "ascending_float32_fma_rank1_then_bf16",
        "stage_a_base_semantic": STAGE_A_BASE_SEMANTIC,
        "stage_a_base_application": (
            "authenticated_packed_k4_signed_fwht_then_64_lane_explicit_f32_"
            "fma_then_tree_then_output_fwht_then_finish"
        ),
        "stage_a_serialized_execution_reference": True,
        "stage_a_base_is_metal_answer_key": False,
        "historical_dense_control": (
            "exact_fit_only_fingerprint_diagnostic_not_cross_order_equality"
        ),
        "stage_b_metal_answer_key": (
            "deferred_until_stage_a_pass; must cover_signed_fwht_projection_"
            "lane_tree_output_fwht_finish"
        ),
        "numpy_version": np.__version__,
        "numba_version": numba.__version__,
        "llvmlite_version": llvmlite.__version__,
    }


K4_MODEL_ROLES = (
    "packed",
    "left_sign",
    "right_sign",
    "global_scale",
    "row_scale",
    "correction_left",
    "correction_right",
)
PROCESS_STARTED_NS = time.time_ns()
PROCESS_NONCE = secrets.token_hex(32)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    return sha256_bytes(array.view(np.uint8).tobytes())


CONTROL_STAGE_NAMES = (
    "dynamic_input_f32",
    "gate_bf16_f32",
    "up_bf16_f32",
    "hidden_bf16_f32",
    "dynamic_hidden_f32",
    "candidate_output_bf16_f32",
)


def stage_control_metrics(serialized: np.ndarray, historical: np.ndarray) -> dict[str, Any]:
    """Fingerprint two exact F32 paths without assuming their reductions agree."""
    left = np.asarray(serialized)
    right = np.asarray(historical)
    if (
        left.dtype != np.float32
        or right.dtype != np.float32
        or left.shape != right.shape
        or left.ndim != 2
        or not np.isfinite(left).all()
        or not np.isfinite(right).all()
    ):
        raise ValueError("PW-0331 serialized/dense control array mismatch")
    # The frozen preflight diagnostic subtracts in the source F32 domain, then
    # widens only for the norm. Preserve that reporting definition exactly.
    delta = np.asarray(left - right, dtype=np.float64)
    denominator = float(np.linalg.norm(np.asarray(right, dtype=np.float64).ravel()))
    numerator = float(np.linalg.norm(delta.ravel()))
    if not math.isfinite(denominator) or denominator <= 0.0 or not math.isfinite(numerator):
        raise ValueError("PW-0331 serialized/dense control norm mismatch")
    return {
        "elements": int(left.size),
        "bit_mismatches": int(
            np.count_nonzero(left.view(np.uint32) != right.view(np.uint32))
        ),
        "max_abs": float(np.max(np.abs(delta))),
        "relative_l2": numerator / denominator,
        "serialized_sha256": array_sha256(left),
        "historical_dense_sha256": array_sha256(right),
    }


def require_control_stage_metrics(
    serialized: dict[str, np.ndarray],
    historical: dict[str, np.ndarray],
    expected: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Require exact per-stage hashes/counts and stable diagnostic scalars."""
    if set(expected) != set(CONTROL_STAGE_NAMES):
        raise ValueError("PW-0331 serialized/dense control stage contract mismatch")
    if any(name not in serialized or name not in historical for name in CONTROL_STAGE_NAMES):
        raise ValueError("PW-0331 serialized/dense control stage missing")
    observed = {
        name: stage_control_metrics(serialized[name], historical[name])
        for name in CONTROL_STAGE_NAMES
    }
    for name in CONTROL_STAGE_NAMES:
        actual = observed[name]
        frozen = expected[name]
        if set(frozen) != set(actual):
            raise ValueError(f"PW-0331 {name} control metric schema mismatch")
        for field in (
            "elements",
            "bit_mismatches",
            "max_abs",
            "serialized_sha256",
            "historical_dense_sha256",
        ):
            if actual[field] != frozen[field]:
                raise ValueError(f"PW-0331 {name} control {field} mismatch")
        if not math.isclose(
            actual["relative_l2"],
            frozen["relative_l2"],
            rel_tol=0.0,
            abs_tol=1.0e-18,
        ):
            raise ValueError(f"PW-0331 {name} control relative_l2 mismatch")
    return observed


def load_serialized_dense_control_fixture(repo: Path) -> dict[str, Any]:
    root = repo.resolve()
    path = root / SERIALIZED_DENSE_CONTROL_RELATIVE_PATH
    if sha256_file(path) != SERIALIZED_DENSE_CONTROL_SHA256:
        raise ValueError("PW-0331 serialized/dense fixture content mismatch")
    fixture = json.loads(path.read_text())
    diagnostic = fixture.get("diagnostic", {})
    if (
        fixture.get("schema_version") != 1
        or fixture.get("experiment_id") != EXPERIMENT_ID
        or fixture.get("semantic")
        != "fit_only_zero_factor_serialized_vs_historical_dense_control_v1"
        or fixture.get("failed_execution", {}).get("output_created") is not False
        or fixture.get("failed_execution", {}).get("held_out_payloads_opened") is not False
        or diagnostic.get("independent_process_replays") != 2
        or diagnostic.get("fit_rows") != EXPECTED_COUNTS["fit"]
        or diagnostic.get("held_out_payloads_opened") is not False
        or sha256_bytes(canonical_json(diagnostic))
        != SERIALIZED_DENSE_CONTROL_DIAGNOSTIC_SHA256
        or sha256_bytes(canonical_json(diagnostic.get("stages")))
        != SERIALIZED_DENSE_CONTROL_STAGES_SHA256
    ):
        raise ValueError("PW-0331 serialized/dense fixture authority mismatch")
    return fixture


def authenticate_serialized_dense_control(
    serialized: dict[str, np.ndarray],
    historical: dict[str, np.ndarray],
    fixture: dict[str, Any],
) -> dict[str, Any]:
    diagnostic = fixture["diagnostic"]
    observed = require_control_stage_metrics(
        serialized, historical, diagnostic["stages"]
    )
    return {
        "semantic": fixture["semantic"],
        "fixture_sha256": SERIALIZED_DENSE_CONTROL_SHA256,
        "diagnostic_sha256": SERIALIZED_DENSE_CONTROL_DIAGNOSTIC_SHA256,
        "stages_sha256": SERIALIZED_DENSE_CONTROL_STAGES_SHA256,
        "independent_process_replays": diagnostic["independent_process_replays"],
        "fit_rows": diagnostic["fit_rows"],
        "held_out_payloads_opened": False,
        "stages": observed,
        "pass": True,
    }


def legacy_framed_array_sha256(values: np.ndarray) -> str:
    """Hash an array exactly as the frozen PW-0311/PW-0315 authorities do."""
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(str(tuple(array.shape)).encode())
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def require_legacy_framed_array_sha256(
    values: np.ndarray, expected: str, label: str
) -> None:
    if legacy_framed_array_sha256(values) != expected:
        raise ValueError(f"PW-0331 {label} legacy framed array hash mismatch")


def bf16(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    bits = array.view(np.uint32)
    rounded = bits + np.uint32(0x7FFF) + ((bits >> 16) & np.uint32(1))
    return ((rounded & np.uint32(0xFFFF0000)).view(np.float32)).copy()


def partition_fit_positions(positions: np.ndarray) -> dict[str, np.ndarray]:
    values = np.asarray(positions)
    if (
        values.ndim != 1
        or values.dtype.kind not in "iu"
        or values.size != 174
        or len(set(map(int, values))) != values.size
        or np.any(values < 0)
        or np.any(values >= ROWS)
    ):
        raise ValueError("PW-0331 expert-96 placement authority mismatch")
    result = {
        "fit": np.flatnonzero((values < FIT_END) & (values != PRIMARY_POSITION)),
        "primary": np.flatnonzero(values == PRIMARY_POSITION),
        "validation": np.flatnonzero(
            (values >= VALIDATION[0]) & (values < VALIDATION[1])
        ),
        "pilot": np.flatnonzero((values >= PILOT[0]) & (values < PILOT[1])),
    }
    observed = {name: int(indices.size) for name, indices in result.items()}
    if observed != EXPECTED_COUNTS:
        raise ValueError(f"PW-0331 split mismatch: {observed}")
    covered = np.concatenate(list(result.values()))
    if sorted(map(int, covered)) != list(range(values.size)):
        raise ValueError("PW-0331 split is not a disjoint bijection")
    return result


def pread_f32_rows(
    path: Path,
    row_indices: np.ndarray | list[int],
    *,
    total_rows: int,
    columns: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Read exactly the requested F32 rows and report every byte range touched."""
    indices = [int(value) for value in np.asarray(row_indices).tolist()]
    if (
        total_rows <= 0
        or columns <= 0
        or not indices
        or len(indices) != len(set(indices))
        or any(index < 0 or index >= total_rows for index in indices)
    ):
        raise ValueError("PW-0331 selected row request is invalid")
    row_bytes = columns * np.dtype("<f4").itemsize
    expected_bytes = total_rows * row_bytes
    if path.stat().st_size != expected_bytes:
        raise ValueError(f"PW-0331 capture size mismatch: {path}")
    payloads: list[bytes] = []
    ranges = []
    descriptor = os.open(path, os.O_RDONLY)
    try:
        for index in indices:
            offset = index * row_bytes
            payload = os.pread(descriptor, row_bytes, offset)
            if len(payload) != row_bytes:
                raise ValueError(f"PW-0331 short selected-row read: {path}")
            payloads.append(payload)
            ranges.append({"row": index, "offset": offset, "bytes": row_bytes})
    finally:
        os.close(descriptor)
    joined = b"".join(payloads)
    result = np.frombuffer(joined, dtype="<f4").reshape(len(indices), columns).copy()
    return result, {
        "path": str(path),
        "total_file_bytes": expected_bytes,
        "whole_payload_rescanned": False,
        "selected_rows": indices,
        "selected_bytes": len(joined),
        "selected_bytes_sha256": sha256_bytes(joined),
        "pread_ranges": ranges,
    }


def _libc_fmaf() -> Callable[[float, float, float], float]:
    library_name = ctypes.util.find_library("m")
    library = ctypes.CDLL(library_name or None)
    function = library.fmaf
    function.argtypes = (ctypes.c_float, ctypes.c_float, ctypes.c_float)
    function.restype = ctypes.c_float

    def call(left: float, right: float, accumulator: float) -> float:
        return float(function(left, right, accumulator))

    return call


if numba is not None:

    @intrinsic
    def _numba_fma32(typing_context, left, right, accumulator):
        """Emit an explicit scalar ``llvm.fma.f32`` with no reassociation."""
        del typing_context
        if left != types.float32 or right != types.float32 or accumulator != types.float32:
            raise TypeError("PW-0331 FMA operands must all be F32")
        signature = types.float32(left, right, accumulator)

        def codegen(context, builder, compiled_signature, arguments):
            del context, compiled_signature
            return builder.fma(*arguments)

        return signature, codegen

    @njit(parallel=True, cache=False)
    def _serialized_lane64_tree_projection(inputs, transformed_weights):
        output = np.empty(
            (inputs.shape[0], transformed_weights.shape[0]), dtype=np.float32
        )
        total = inputs.shape[0] * transformed_weights.shape[0]
        for flat_index in prange(total):
            input_index = flat_index // transformed_weights.shape[0]
            output_index = flat_index % transformed_weights.shape[0]
            partials = np.zeros(64, dtype=np.float32)
            for lane in range(64):
                accumulator = np.float32(0.0)
                for column in range(lane, inputs.shape[1], 64):
                    accumulator = _numba_fma32(
                        transformed_weights[output_index, column],
                        inputs[input_index, column],
                        accumulator,
                    )
                partials[lane] = accumulator
            offset = 32
            while offset > 0:
                for lane in range(offset):
                    partials[lane] = np.float32(
                        partials[lane] + partials[lane + offset]
                    )
                offset //= 2
            output[input_index, output_index] = partials[0]
        return output

    @njit(parallel=True, cache=False)
    def _serialized_finish_base(
        transformed_output, left_sign, global_scale, row_scale
    ):
        result = np.empty_like(transformed_output)
        total = transformed_output.shape[0] * transformed_output.shape[1]
        for flat_index in prange(total):
            input_index = flat_index // transformed_output.shape[1]
            output_index = flat_index % transformed_output.shape[1]
            signed = np.float32(
                transformed_output[input_index, output_index]
                * np.float32(left_sign[output_index])
            )
            scaled = np.float32(signed * global_scale)
            result[input_index, output_index] = _numba_fma32(
                scaled, row_scale[output_index], np.float32(0.0)
            )
        return result


def _kernel_fwht_rows(values: np.ndarray) -> np.ndarray:
    """Mirror qtip_k4_bundle_fwht/qtip_fwht_fused_batched F32 order."""
    result = np.ascontiguousarray(values, dtype=np.float32).copy()
    if result.ndim != 2:
        raise ValueError("PW-0331 serialized FWHT requires a rank-2 array")
    width = result.shape[1]
    if width <= 0 or width & (width - 1):
        raise ValueError("PW-0331 serialized FWHT width must be a power of two")
    stride = 1
    while stride < width:
        shaped = result.reshape(result.shape[0], width // (2 * stride), 2, stride)
        left = shaped[:, :, 0, :].copy()
        right = shaped[:, :, 1, :].copy()
        shaped[:, :, 0, :] = np.asarray(left + right, dtype=np.float32)
        shaped[:, :, 1, :] = np.asarray(left - right, dtype=np.float32)
        stride <<= 1
    normalization = np.float32(
        np.float32(1.0) / np.sqrt(np.float32(width))
    )
    return np.asarray(result * normalization, dtype=np.float32)


def unpack_serialized_k4_transformed(
    packed: np.ndarray, tlut: np.ndarray, rows: int, columns: int
) -> np.ndarray:
    """Decode exactly the unscaled 16x16 tiles consumed by the Metal kernel."""
    words = np.asarray(packed)
    lookup = np.asarray(tlut)
    if (
        rows <= 0
        or columns <= 0
        or rows % 16
        or columns % 16
        or words.dtype != np.dtype("<u2")
        or words.shape != (rows * columns // 256, 64)
        or lookup.dtype != np.float32
        or lookup.shape != (512, 2)
        or not np.isfinite(lookup).all()
    ):
        raise ValueError("PW-0331 serialized K4 tile authority mismatch")
    words32 = words.astype(np.uint32)
    state_index = np.arange(128, dtype=np.uint32)
    bit_offset = state_index * np.uint32(8)
    word_index = bit_offset >> np.uint32(4)
    word_shift = bit_offset & np.uint32(15)
    joined = (
        (words32[:, word_index] << np.uint32(16))
        | words32[:, (word_index + np.uint32(1)) & np.uint32(63)]
    )
    states = (joined >> (np.uint32(16) - word_shift)) & np.uint32(0xFFFF)
    mixed = states * (states + np.uint32(1))
    tlut_index = (mixed >> np.uint32(6)) & np.uint32(511)
    quantized_permuted = np.empty((words.shape[0], 256), dtype=np.float32)
    sign = np.where(
        ((mixed >> np.uint32(15)) & np.uint32(1)) != 0,
        np.float32(-1.0),
        np.float32(1.0),
    )
    quantized_permuted[:, 0::2] = np.asarray(
        lookup[tlut_index, 0] * sign, dtype=np.float32
    )
    quantized_permuted[:, 1::2] = lookup[tlut_index, 1]
    permutation = (
        np.arange(256)
        .reshape(2, 8, 2, 4, 2)
        .transpose(1, 3, 2, 0, 4)
        .reshape(-1)
    )
    inverse = np.empty_like(permutation)
    inverse[permutation] = np.arange(256)
    tiled = quantized_permuted[:, inverse].reshape(
        rows // 16, columns // 16, 16, 16
    )
    return np.ascontiguousarray(
        tiled.transpose(0, 2, 1, 3).reshape(rows, columns), dtype=np.float32
    )


def serialized_k4_projection_base(
    inputs: np.ndarray, projection: dict[str, Any], tlut: np.ndarray
) -> np.ndarray:
    """Emulate the current zero-correction schema-2 projection order."""
    if numba is None:
        raise RuntimeError("PW-0331 serialized reference requires numba/llvmlite")
    x = np.asarray(inputs)
    rows = projection.get("rows")
    columns = projection.get("columns")
    left_sign = np.asarray(projection.get("left_sign"))
    right_sign = np.asarray(projection.get("right_sign"))
    row_scale = np.asarray(projection.get("row_scale"))
    global_scale = np.asarray(projection.get("global_scale"))
    if (
        not isinstance(rows, int)
        or not isinstance(columns, int)
        or x.ndim != 2
        or x.dtype != np.float32
        or x.shape[1] != columns
        or left_sign.dtype != np.int8
        or left_sign.shape != (rows,)
        or right_sign.dtype != np.int8
        or right_sign.shape != (columns,)
        or row_scale.dtype != np.float16
        or row_scale.shape != (rows,)
        or global_scale.dtype != np.float32
        or global_scale.shape != (1,)
        or not np.isin(left_sign, (-1, 1)).all()
        or not np.isin(right_sign, (-1, 1)).all()
        or not np.isfinite(x).all()
        or not np.isfinite(row_scale).all()
        or not np.isfinite(global_scale).all()
    ):
        raise ValueError("PW-0331 serialized K4 projection authority mismatch")
    transformed_weights = unpack_serialized_k4_transformed(
        projection["packed"], tlut, rows, columns
    )
    signed_input = np.asarray(
        x * right_sign.astype(np.float32)[None, :], dtype=np.float32
    )
    transformed_input = _kernel_fwht_rows(signed_input)
    raw = _serialized_lane64_tree_projection(
        transformed_input, transformed_weights
    )
    transformed_output = _kernel_fwht_rows(raw)
    result = _serialized_finish_base(
        transformed_output,
        left_sign,
        np.float32(global_scale[0]),
        row_scale.astype(np.float32),
    )
    if result.dtype != np.float32 or not np.isfinite(result).all():
        raise ValueError("PW-0331 serialized K4 projection produced nonfinite output")
    return result


def stage_a_candidate_stages(
    values: np.ndarray,
    projections: dict[str, dict[str, Any]],
    tlut: np.ndarray,
    dynamic_input: Callable[[np.ndarray], np.ndarray],
) -> dict[str, np.ndarray]:
    """Evaluate all serialized-K4 Stage A projections without BLAS batching."""
    if set(projections) != {"gate", "up", "down"}:
        raise ValueError("PW-0331 Stage A projection set mismatch")
    source = np.asarray(values)
    if source.ndim != 2 or source.dtype != np.float32 or not np.isfinite(source).all():
        raise ValueError("PW-0331 Stage A source input mismatch")
    dynamic_values = np.asarray(dynamic_input(source), dtype=np.float32)
    if dynamic_values.shape != source.shape or not np.isfinite(dynamic_values).all():
        raise ValueError("PW-0331 Stage A dynamic input mismatch")
    if not np.array_equal(
        np.asarray(dynamic_input(dynamic_values), dtype=np.float32), dynamic_values
    ):
        raise ValueError("PW-0331 Stage A input dynamic-FP8 boundary is not idempotent")
    gate_raw = serialized_k4_projection_base(dynamic_values, projections["gate"], tlut)
    up_raw = serialized_k4_projection_base(dynamic_values, projections["up"], tlut)
    gate = bf16(gate_raw)
    up = bf16(up_raw)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        silu = bf16(gate / (np.float32(1.0) + np.exp(-gate)))
        hidden = bf16(silu * up)
    if not np.isfinite(hidden).all():
        raise ValueError("PW-0331 Stage A SwiGLU produced nonfinite output")
    dynamic_hidden = np.asarray(dynamic_input(hidden), dtype=np.float32)
    if dynamic_hidden.shape != hidden.shape or not np.isfinite(dynamic_hidden).all():
        raise ValueError("PW-0331 Stage A dynamic hidden mismatch")
    if not np.array_equal(
        np.asarray(dynamic_input(dynamic_hidden), dtype=np.float32), dynamic_hidden
    ):
        raise ValueError("PW-0331 Stage A hidden dynamic-FP8 boundary is not idempotent")
    down_raw = serialized_k4_projection_base(
        dynamic_hidden, projections["down"], tlut
    )
    return {
        "dynamic_input_f32": dynamic_values,
        "gate_base_raw_f32": gate_raw,
        "gate_bf16_f32": gate,
        "up_base_raw_f32": up_raw,
        "up_bf16_f32": up,
        "hidden_bf16_f32": hidden,
        "dynamic_hidden_f32": dynamic_hidden,
        "down_base_raw_f32": down_raw,
        "candidate_output_bf16_f32": bf16(down_raw),
    }


def apply_serialized_rank_one(
    inputs: np.ndarray,
    base_raw: np.ndarray,
    correction_left: np.ndarray,
    correction_right: np.ndarray,
    *,
    fma: Callable[[float, float, float], float] | None = None,
) -> np.ndarray:
    """Apply the rank-one slot in the shader's ascending F32/FMA order."""
    x = np.asarray(inputs, dtype=np.float32)
    base = np.asarray(base_raw, dtype=np.float32)
    left = np.asarray(correction_left)
    right = np.asarray(correction_right)
    if (
        x.ndim != 2
        or base.ndim != 2
        or base.shape[0] != x.shape[0]
        or left.shape != (base.shape[1], 1)
        or right.shape != (1, x.shape[1])
        or left.dtype != np.float16
        or right.dtype != np.float16
        or not np.isfinite(x).all()
        or not np.isfinite(base).all()
        or not np.isfinite(left).all()
        or not np.isfinite(right).all()
    ):
        raise ValueError("PW-0331 serialized rank-one application mismatch")
    fused = fma or _libc_fmaf()
    left32 = left[:, 0].astype(np.float32)
    right32 = right[0].astype(np.float32)
    result = np.empty_like(base)
    for row in range(x.shape[0]):
        scalar = 0.0
        for column in range(x.shape[1]):
            scalar = fused(float(right32[column]), float(x[row, column]), scalar)
        for output in range(base.shape[1]):
            correction = fused(float(left32[output]), scalar, 0.0)
            result[row, output] = np.float32(
                fused(float(base[row, output]), 1.0, correction)
            )
    return bf16(result)


def fit_rank_one(
    inputs: np.ndarray,
    base_raw: np.ndarray,
    source_bf16: np.ndarray,
    route_weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Fit the exact PW-0331 weighted rank-one embodiment from fit rows only."""
    x = np.asarray(inputs, dtype=np.float32)
    base = np.asarray(base_raw, dtype=np.float32)
    source = np.asarray(source_bf16, dtype=np.float32)
    weights = np.asarray(route_weights, dtype=np.float32)
    if (
        x.ndim != 2
        or base.ndim != 2
        or source.shape != base.shape
        or base.shape[0] != x.shape[0]
        or weights.shape != (x.shape[0],)
        or x.shape[0] == 0
        or not np.isfinite(x).all()
        or not np.isfinite(base).all()
        or not np.isfinite(source).all()
        or not np.isfinite(weights).all()
        or np.any(weights < 0.0)
    ):
        raise ValueError("PW-0331 fit matrices are invalid")
    x64 = x.astype(np.float64)
    residual64 = source.astype(np.float64) - bf16(base).astype(np.float64)
    weights64 = weights.astype(np.float64)
    weighted_x = weights64[:, None] * x64
    weighted_residual = weights64[:, None] * residual64
    u, singular, vt = np.linalg.svd(weighted_residual, full_matrices=False)
    if singular.size == 0 or not np.isfinite(singular).all() or singular[0] <= 0.0:
        raise ValueError("PW-0331 weighted residual has no rank-one signal")
    score = u[:, 0] * singular[0]
    gram = weighted_x @ weighted_x.T
    right64 = weighted_x.T @ np.linalg.pinv(gram, rcond=RCOND) @ score
    left64 = vt[0].copy()
    maximum = float(np.max(np.abs(left64), initial=0.0))
    pivots = np.flatnonzero(np.abs(left64) == maximum)
    if maximum == 0.0 or pivots.size == 0:
        raise ValueError("PW-0331 leading-vector sign is ambiguous")
    pivot = int(pivots[0])
    if left64[pivot] < 0.0:
        left64 *= -1.0
        right64 *= -1.0
    left = np.asarray(left64[:, None], dtype="<f2")
    right = np.asarray(right64[None, :], dtype="<f2")
    if (
        not np.isfinite(left).all()
        or not np.isfinite(right).all()
        or not np.any(left)
        or not np.any(right)
        or float(np.max(np.abs(left.astype(np.float32))))
        * float(np.max(np.abs(right.astype(np.float32))))
        == 0.0
    ):
        raise ValueError("PW-0331 serialized rank-one product is zero or nonfinite")
    diagnostics = {
        "fit_rows": int(x.shape[0]),
        "input_columns": int(x.shape[1]),
        "output_rows": int(base.shape[1]),
        "rcond": RCOND,
        "leading_singular_value_f64": float(singular[0]),
        "second_singular_value_f64": float(singular[1]) if singular.size > 1 else 0.0,
        "sign_pivot": pivot,
        "weighted_input_rank": int(np.linalg.matrix_rank(weighted_x)),
        "continuous_weighted_residual_l2": float(np.linalg.norm(weighted_residual)),
    }
    return left, right, diagnostics


def schema2_layout_ledger(k4_experts: int, source_experts: int) -> dict[str, int]:
    if k4_experts < 0 or source_experts < 0 or k4_experts + source_experts != 8:
        raise ValueError("PW-0331 schema-2 route must contain eight experts")
    k4_stride = 12_877_824
    source_stride = 25_214_976
    tlut_page = 16_384
    return {
        "k4_logical_bytes": 12_654_604,
        "k4_stride_bytes": k4_stride,
        "source_stride_bytes": source_stride,
        "tlut_page_bytes": tlut_page,
        "bundle_bytes": tlut_page + k4_experts * k4_stride + source_experts * source_stride,
        "down_factor_bytes": 12_288,
    }


def _git(repo: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *arguments],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def verify_execution_authority(repo: Path, commit: str) -> dict[str, Any]:
    root = repo.resolve()
    verify_clean_commit(root, commit)
    if subprocess.run(
        ["git", "-C", str(root), "merge-base", "--is-ancestor", CONTRACT_COMMIT, commit]
    ).returncode != 0:
        raise ValueError("PW-0331 execution commit does not descend from its contract")
    contract = root / CONTRACT_RELATIVE_PATH
    if sha256_file(contract) != CONTRACT_SHA256:
        raise ValueError("PW-0331 contract content mismatch")
    if _git(root, "rev-parse", f"HEAD:{CONTRACT_RELATIVE_PATH}") != CONTRACT_GIT_BLOB:
        raise ValueError("PW-0331 contract Git blob mismatch")
    load_serialized_dense_control_fixture(root)
    if (
        _git(root, "rev-parse", f"HEAD:{SERIALIZED_DENSE_CONTROL_RELATIVE_PATH}")
        != SERIALIZED_DENSE_CONTROL_GIT_BLOB
    ):
        raise ValueError("PW-0331 serialized/dense fixture Git blob mismatch")
    if sha256_file(root / "TARGET.md") != TARGET_SHA256:
        raise ValueError("PW-0331 TARGET.md mismatch")
    if sha256_file(root / "RED_LINES.md") != RED_LINES_SHA256:
        raise ValueError("PW-0331 RED_LINES.md mismatch")
    observed = {}
    for relative, expected in IMPLEMENTATION_HASHES.items():
        digest = sha256_file(root / relative)
        if digest != expected:
            raise ValueError(f"PW-0331 unchanged implementation mismatch: {relative}")
        observed[relative] = digest
    return {
        "contract_commit": CONTRACT_COMMIT,
        "contract_git_blob": CONTRACT_GIT_BLOB,
        "contract_sha256": CONTRACT_SHA256,
        "serialized_dense_control_git_blob": SERIALIZED_DENSE_CONTROL_GIT_BLOB,
        "serialized_dense_control_sha256": SERIALIZED_DENSE_CONTROL_SHA256,
        "target_sha256": TARGET_SHA256,
        "red_lines_sha256": RED_LINES_SHA256,
        "unchanged_implementation_sha256": observed,
    }


def load_panel_authority(
    authority_root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    paths = authority_paths(authority_root.resolve())
    if sha256_file(paths["contract"]) != PANEL_CONTRACT_SHA256:
        raise ValueError("PW-0331 K4 panel contract mismatch")
    export_path = paths["reference_export"] / "export.json"
    if sha256_file(export_path) != PANEL_EXPORT_SHA256:
        raise ValueError("PW-0331 K4 panel export mismatch")
    panel_contract = json.loads(paths["contract"].read_text())
    implementation = paths["work"] / "tools/export_selected_k4_panel.py"
    declared = panel_contract.get("authority", {}).get("implementation_sha256")
    if declared != PANEL_IMPLEMENTATION_SHA256:
        raise ValueError("PW-0331 K4 panel declared implementation mismatch")
    if sha256_file(implementation) != PANEL_IMPLEMENTATION_SHA256:
        raise ValueError("PW-0331 K4 panel implementation mismatch")
    return _load_authority_modules(paths), {
        "contract_sha256": PANEL_CONTRACT_SHA256,
        "export_sha256": PANEL_EXPORT_SHA256,
        "implementation_sha256": PANEL_IMPLEMENTATION_SHA256,
    }


def _safe_capture_path(root: Path, capture: dict[str, Any]) -> Path:
    path = (root / capture["file"]).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError("PW-0331 capture path escapes corpus root") from error
    if capture.get("dtype") != "BF16_widened_F32":
        raise ValueError("PW-0331 capture dtype mismatch")
    return path


def load_fit_corpus_rows(corpus_manifest: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    if sha256_file(corpus_manifest) != CORPUS_SHA256:
        raise ValueError("PW-0331 corpus manifest mismatch")
    corpus = json.loads(corpus_manifest.read_text())
    layer_rows = [row for row in corpus.get("layers", []) if int(row.get("layer", -1)) == LAYER]
    if len(layer_rows) != 1:
        raise ValueError("PW-0331 layer-4 authority mismatch")
    layer_row = layer_rows[0]
    positions, slots, weights, source_offsets = selected_rows(layer_row, EXPERT)
    partitions = partition_fit_positions(positions)
    local = partitions["fit"]
    fit_positions = positions[local]
    fit_offsets = source_offsets[local]
    if PRIMARY_POSITION in set(map(int, fit_positions)):
        raise ValueError("PW-0331 primary position leaked into fit")
    root = corpus_manifest.parent
    input_capture = layer_row["captures"]["moe_input"]
    source_capture = layer_row["captures"]["expert_down"]
    if (
        input_capture.get("shape") != [ROWS, HIDDEN]
        or source_capture.get("shape") != [1792, HIDDEN]
        or int(input_capture.get("bytes", -1)) != ROWS * HIDDEN * 4
        or int(source_capture.get("bytes", -1)) != 1792 * HIDDEN * 4
    ):
        raise ValueError("PW-0331 frozen capture shape/byte authority mismatch")
    input_path = _safe_capture_path(root, input_capture)
    source_path = _safe_capture_path(root, source_capture)
    fit_input, input_audit = pread_f32_rows(
        input_path,
        fit_positions,
        total_rows=int(input_capture["shape"][0]),
        columns=int(input_capture["shape"][1]),
    )
    fit_source, source_audit = pread_f32_rows(
        source_path,
        fit_offsets,
        total_rows=int(source_capture["shape"][0]),
        columns=int(source_capture["shape"][1]),
    )
    return {
        "fit_input": fit_input,
        "fit_source_bf16": fit_source,
        "fit_weights": weights[local].astype(np.float32),
        "fit_positions": fit_positions.astype(np.int64),
        "fit_source_offsets": fit_offsets.astype(np.int64),
        "fit_slots": slots[local].astype(np.int64),
    }, {
        "corpus_manifest_sha256": CORPUS_SHA256,
        "whole_capture_hashes_from_manifest": {
            "moe_input": input_capture["sha256"],
            "expert_down": source_capture["sha256"],
        },
        "whole_capture_payloads_rescanned": False,
        "input_read": input_audit,
        "source_read": source_audit,
        "split_counts": EXPECTED_COUNTS,
    }


def _read_manifest_array(directory: Path, row: dict[str, Any]) -> np.ndarray:
    path = (directory / row["file"]).resolve()
    try:
        path.relative_to(directory.resolve())
    except ValueError as error:
        raise ValueError("PW-0331 projection payload escapes artifact root") from error
    if path.stat().st_size != int(row["bytes"]) or sha256_file(path) != row["sha256"]:
        raise ValueError(f"PW-0331 projection payload mismatch: {path}")
    dtype = np.dtype(row["dtype"])
    values = np.fromfile(path, dtype=dtype)
    return values.reshape(row["shape"])


def load_zero_correction_k4(
    *,
    expert_root: Path,
    construction_report: Path,
    tlut: np.ndarray,
    modules: dict[str, Any],
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, Any]], dict[str, Any]]:
    if sha256_file(construction_report) != PW0315_EXPERT96_REPORT_SHA256:
        raise ValueError("PW-0331 expert-96 construction report mismatch")
    report = json.loads(construction_report.read_text())
    if (
        report.get("experiment_id") != "PW-0315"
        or report.get("layer") != LAYER
        or report.get("expert") != EXPERT
        or report.get("failure") is not None
        or report.get("semantic", {}).get("gates", {}).get("pass") is not True
    ):
        raise ValueError("PW-0331 expert-96 construction contract mismatch")
    checkpoint = report.get("authority", {}).get("checkpoint", {})
    if (
        checkpoint.get("revision") != CHECKPOINT_REVISION
        or checkpoint.get("receipt_sha256") != CHECKPOINT_RECEIPT_SHA256
        or checkpoint.get("index_sha256") != CHECKPOINT_INDEX_SHA256
        or checkpoint.get("source_shard_sha256_from_receipt")
        != CHECKPOINT_EXPERT96_SHARD_SHA256
        or report.get("authority", {}).get("corpus_sha256") != CORPUS_SHA256
    ):
        raise ValueError("PW-0331 embedded checkpoint/corpus authority mismatch")
    decoded = {}
    serialized = {}
    observations = {}
    for name in ("gate", "up", "down"):
        directory = expert_root / name
        manifest_path = directory / "manifest.json"
        expected_manifest = report["projections"][name]["manifest_sha256"]
        if sha256_file(manifest_path) != expected_manifest:
            raise ValueError(f"PW-0331 {name} manifest mismatch")
        manifest = json.loads(manifest_path.read_text())
        expected_shape = (4096, 2048) if name == "down" else (2048, 4096)
        if (
            manifest.get("schema_version") != 1
            or manifest.get("name") != name
            or (manifest.get("rows"), manifest.get("columns")) != expected_shape
            or manifest.get("rank") != 1
            or set(manifest.get("files", {})) < set(K4_MODEL_ROLES)
        ):
            raise ValueError(f"PW-0331 {name} projection contract mismatch")
        arrays = {
            role: _read_manifest_array(directory, manifest["files"][role])
            for role in K4_MODEL_ROLES
        }
        if not np.array_equal(arrays["row_scale"], np.ones(expected_shape[0], dtype=np.float16)):
            raise ValueError(f"PW-0331 {name} row-scale control mismatch")
        for role in ("correction_left", "correction_right"):
            payload = np.ascontiguousarray(arrays[role].astype("<f2", copy=False)).tobytes()
            expected_zero = ZERO_F16_4096_SHA256 if len(payload) == 4096 else ZERO_F16_8192_SHA256
            if sha256_bytes(payload) != expected_zero or np.any(arrays[role]):
                raise ValueError(f"PW-0331 {name} zero-correction control mismatch")
        candidate = modules["export"]._decode_k4(
            arrays["packed"],
            tlut,
            expected_shape[0],
            expected_shape[1],
            float(arrays["global_scale"].reshape(-1)[0]),
            arrays["left_sign"].astype(np.float32),
            arrays["right_sign"].astype(np.float32),
        )
        require_legacy_framed_array_sha256(
            candidate, manifest["candidate_array_sha256"], f"{name} decoded candidate"
        )
        decoded[name] = candidate
        serialized[name] = {
            "rows": expected_shape[0],
            "columns": expected_shape[1],
            **arrays,
        }
        observations[name] = {
            "manifest_sha256": expected_manifest,
            "candidate_array_sha256": manifest["candidate_array_sha256"],
            "rank": 1,
            "row_scale_identity": True,
            "correction_zero": True,
        }
    observations["checkpoint"] = {
        "revision": CHECKPOINT_REVISION,
        "receipt_sha256": CHECKPOINT_RECEIPT_SHA256,
        "index_sha256": CHECKPOINT_INDEX_SHA256,
        "expert96_shard_sha256_from_receipt": CHECKPOINT_EXPERT96_SHARD_SHA256,
        "source_shard_rescanned": False,
    }
    return decoded, serialized, observations


def load_pw0318_tlut(manifest_path: Path, bundle_path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    if sha256_file(manifest_path) != PW0318_MANIFEST_SHA256:
        raise ValueError("PW-0331 PW-0318 manifest mismatch")
    manifest = json.loads(manifest_path.read_text())
    if (
        manifest.get("schema_version") != 2
        or manifest.get("bundle_sha256") != PW0318_BUNDLE_SHA256
        or int(manifest.get("bundle_bytes", -1)) != bundle_path.stat().st_size
        or sha256_file(bundle_path) != PW0318_BUNDLE_SHA256
    ):
        raise ValueError("PW-0331 PW-0318 bundle mismatch")
    record = manifest["tlut"]
    if int(record["bytes"]) != 4096:
        raise ValueError("PW-0331 TLUT layout mismatch")
    descriptor = os.open(bundle_path, os.O_RDONLY)
    try:
        payload = os.pread(descriptor, int(record["bytes"]), int(record["offset"]))
    finally:
        os.close(descriptor)
    if len(payload) != 4096 or sha256_bytes(payload) != record["sha256"]:
        raise ValueError("PW-0331 TLUT payload mismatch")
    return np.frombuffer(payload, dtype="<f4").reshape(512, 2).copy(), {
        "manifest_sha256": PW0318_MANIFEST_SHA256,
        "bundle_sha256": PW0318_BUNDLE_SHA256,
        "tlut_sha256": record["sha256"],
        "tlut_offset": int(record["offset"]),
        "tlut_bytes": 4096,
    }


def verify_metadata_authorities(pw0315_summary: Path, pw0316_rejection: Path) -> dict[str, Any]:
    if sha256_file(pw0315_summary) != PW0315_SUMMARY_SHA256:
        raise ValueError("PW-0331 PW-0315 summary mismatch")
    if sha256_file(pw0316_rejection) != PW0316_REJECTION_SHA256:
        raise ValueError("PW-0331 PW-0316 rejection mismatch")
    rejection = json.loads(pw0316_rejection.read_text())
    semantic = rejection.get("semantic", {})
    route = semantic.get("route_candidate_vs_source", {}).get("relative_l2")
    final = semantic.get("final_candidate_vs_source", {}).get("relative_l2")
    if route != 0.010988841869031155 or final != 0.0027743952049186665:
        raise ValueError("PW-0331 published PW-0316 scalar mismatch")
    return {
        "pw0315_summary_sha256": PW0315_SUMMARY_SHA256,
        "pw0316_rejection_sha256": PW0316_REJECTION_SHA256,
        "published_position1_route_relative_l2": route,
        "published_position1_final_relative_l2": final,
        "held_out_payloads_opened": False,
    }


def deterministic_tree(root: Path) -> dict[str, Any]:
    files = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name == "construction.json":
            continue
        files.append(
            {
                "path": str(path.relative_to(root)),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return {"files": files, "total_bytes": sum(row["bytes"] for row in files)}


def run(
    *,
    authority_root: Path,
    corpus_manifest: Path,
    pw0315_summary: Path,
    pw0315_expert_root: Path,
    pw0315_construction: Path,
    pw0316_rejection: Path,
    pw0318_manifest: Path,
    pw0318_bundle: Path,
    repo: Path,
    commit: str,
    output: Path,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(output)
    started = time.monotonic()
    execution = verify_execution_authority(repo, commit)
    metadata = verify_metadata_authorities(pw0315_summary, pw0316_rejection)
    safety = HostSafetyMonitor()
    modules, panel_authority = load_panel_authority(authority_root)
    corpus_rows, corpus_authority = load_fit_corpus_rows(corpus_manifest)
    safety.checkpoint("fit_rows_loaded_without_holdouts")
    tlut, tlut_authority = load_pw0318_tlut(pw0318_manifest, pw0318_bundle)
    decoded, serialized, k4_authority = load_zero_correction_k4(
        expert_root=pw0315_expert_root,
        construction_report=pw0315_construction,
        tlut=tlut,
        modules=modules,
    )
    stages = stage_a_candidate_stages(
        corpus_rows["fit_input"],
        serialized,
        tlut,
        modules["panel"].dynamic_input,
    )
    historical = modules["panel"].complete_outputs(corpus_rows["fit_input"], decoded)
    serialized_dense_control = authenticate_serialized_dense_control(
        stages,
        historical,
        load_serialized_dense_control_fixture(repo),
    )
    fit_x = np.asarray(stages["dynamic_hidden_f32"], dtype=np.float32)
    base_raw = np.asarray(stages["down_base_raw_f32"], dtype=np.float32)
    left, right, fit_diagnostics = fit_rank_one(
        fit_x,
        base_raw,
        corpus_rows["fit_source_bf16"],
        corpus_rows["fit_weights"],
    )
    output.mkdir(parents=True)
    left_path = output / "correction-left.f16le"
    right_path = output / "correction-right.f16le"
    atomic_write_new(left_path, left.astype("<f2", copy=False).tobytes(order="C"))
    atomic_write_new(right_path, right.astype("<f2", copy=False).tobytes(order="C"))
    factor_records = {
        "correction_left": {
            "file": left_path.name,
            "dtype": "<f2",
            "shape": [HIDDEN, 1],
            "bytes": left_path.stat().st_size,
            "sha256": sha256_file(left_path),
        },
        "correction_right": {
            "file": right_path.name,
            "dtype": "<f2",
            "shape": [1, INTERMEDIATE],
            "bytes": right_path.stat().st_size,
            "sha256": sha256_file(right_path),
        },
    }
    fit_authority = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "semantic": SEMANTIC,
        "exactness_class": "L3_modified_expert_weights",
        "commit": commit,
        "construction_surface": "fit_rows_only_no_primary_validation_or_pilot_payload_access",
        "execution_authority": execution,
        "metadata_authority": metadata,
        "corpus_authority": corpus_authority,
        "tlut_authority": tlut_authority,
        "panel_authority": panel_authority,
        "k4_authority": k4_authority,
        "serialized_dense_control": serialized_dense_control,
        "fit_positions": corpus_rows["fit_positions"].tolist(),
        "fit_source_offsets": corpus_rows["fit_source_offsets"].tolist(),
        "fit_slots": corpus_rows["fit_slots"].tolist(),
        "array_sha256": {
            "fit_input_f32": array_sha256(corpus_rows["fit_input"]),
            "fit_source_bf16_f32": array_sha256(corpus_rows["fit_source_bf16"]),
            "fit_route_weights_f32": array_sha256(corpus_rows["fit_weights"]),
            "candidate_dynamic_hidden_f32": array_sha256(fit_x),
            "candidate_down_base_raw_f32": array_sha256(base_raw),
            "candidate_down_base_bf16_f32": array_sha256(bf16(base_raw)),
        },
        "fit": fit_diagnostics,
        "factors": factor_records,
        "layout": schema2_layout_ledger(4, 4),
        "numerics": stage_a_numerics_authority(),
        "accepted_tokens": 0,
        "A": 0,
        "U": 0,
        "performance_claim": None,
        "claims_excluded": [
            "held-out fidelity",
            "density-four qualification",
            "density-five qualification",
            "endpoint TPS",
            "target-faithful weights",
        ],
    }
    atomic_write_new(output / "fit-authority.json", canonical_json(fit_authority))
    tree = deterministic_tree(output)
    del (
        decoded,
        serialized,
        stages,
        historical,
        serialized_dense_control,
        fit_x,
        base_raw,
        left,
        right,
        corpus_rows,
        tlut,
    )
    gc.collect()
    safety.release_checkpoint(
        "fit_buffers_released",
        ["decoded K4 weights", "fit-only capture rows", "SVD workspace", "rank-one factors"],
    )
    safety.checkpoint("final_service_health")
    result = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "status": "fit_factors_frozen_without_heldout_access",
        "commit": commit,
        "fit_authority_sha256": sha256_file(output / "fit-authority.json"),
        "deterministic_tree": tree,
        "complete_seconds": time.monotonic() - started,
        "peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "host": {
            "machine": platform.machine(),
            "platform": platform.platform(),
            "total_memory_bytes": int(os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")),
        },
        "process_receipt": {
            "pid": os.getpid(),
            "started_ns": PROCESS_STARTED_NS,
            "nonce": PROCESS_NONCE,
        },
        "safety_snapshots": safety.evidence(),
        "accepted_tokens": 0,
        "A": 0,
        "U": 0,
        "performance_claim": None,
    }
    atomic_write_new(output / "construction.json", canonical_json(result))
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--authority-root", required=True, type=Path)
    parser.add_argument("--corpus-manifest", required=True, type=Path)
    parser.add_argument("--pw0315-summary", required=True, type=Path)
    parser.add_argument("--pw0315-expert-root", required=True, type=Path)
    parser.add_argument("--pw0315-construction", required=True, type=Path)
    parser.add_argument("--pw0316-rejection", required=True, type=Path)
    parser.add_argument("--pw0318-manifest", required=True, type=Path)
    parser.add_argument("--pw0318-bundle", required=True, type=Path)
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    try:
        result = run(**vars(arguments))
        print(json.dumps({"output": str(arguments.output), "status": result["status"]}))
        return 0
    except (
        FileExistsError,
        HostSafetyViolation,
        KeyError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
        subprocess.CalledProcessError,
    ) as error:
        print(json.dumps({"error": str(error)}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
