#!/usr/bin/env python3
"""Run PW-0149's nonuniform INT4 global-Hessian three-expert control."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

try:
    from tools.run_five_bit_global_hessian_three_expert_control import (
        NBitControlConfig,
        _gate as _nbit_gate,
        global_hessian_nbit_fixed_grid,
        physical_ledger as _physical_ledger,
        run as _run_control,
    )
    from tools.run_real_activation_affine_int4_audit import GROUP_SIZE
except ModuleNotFoundError:
    from run_five_bit_global_hessian_three_expert_control import (
        NBitControlConfig,
        _gate as _nbit_gate,
        global_hessian_nbit_fixed_grid,
        physical_ledger as _physical_ledger,
        run as _run_control,
    )
    from run_real_activation_affine_int4_audit import GROUP_SIZE


BITS = 4
LEVELS = 16
LLOYD_ITERATIONS = 8
ROW_BATCH = 64
MAXIMUM_CODE = 15
PACKED_CODE_BYTES = 12_582_912
METADATA_BYTES = 6_291_456
PACKED_BYTES = 18_874_368
PACKED_RATIO = 0.7498169392238223
PW0148_SHA256 = "48d1c28cc589e55002ce5a4b836d62ef172d3ed77106c100b2ad49d708fd1257"

NONUNIFORM_CONFIG = NBitControlConfig(
    experiment="PW-0149",
    bits=BITS,
    packed_code_bytes=PACKED_CODE_BYTES,
    metadata_bytes=METADATA_BYTES,
    metadata_label="f16_codebook_metadata_bytes_per_expert",
    maximum_packed_ratio=0.75,
    candidate_label="nonuniform_int4",
    evidence_class="pw0149_nonuniform_int4_global_hessian_control",
    pass_decision="authorize_all_validation_expert_nonuniform_int4_audit",
    reject_decision="reject_nonuniform_int4_global_hessian_three_expert_control",
    prerequisite_sha256=PW0148_SHA256,
    prerequisite_label="PW-0148 report",
    prerequisite_decision="reject_six_bit_global_hessian_three_expert_control",
    prerequisite_source_key="pw0148_report_sha256",
    prior_candidate_label="six_bit",
)


def reconstruct_codebook_grid(codes: np.ndarray, codebooks: np.ndarray) -> np.ndarray:
    if (
        codes.ndim != 2
        or codes.dtype != np.uint8
        or np.any(codes > MAXIMUM_CODE)
        or codebooks.ndim != 3
        or codebooks.dtype != np.float16
        or codebooks.shape[0] != codes.shape[0]
        or codebooks.shape[2] != LEVELS
        or codes.shape[1] != codebooks.shape[1] * GROUP_SIZE
        or not np.isfinite(codebooks).all()
    ):
        raise ValueError("PW-0149 codebook reconstruction input is invalid")
    rows, columns = codes.shape
    groups = codebooks.shape[1]
    selected = np.take_along_axis(
        codebooks.astype(np.float32)[:, :, None, :],
        codes.reshape(rows, groups, GROUP_SIZE)[..., None],
        axis=3,
    )[..., 0]
    return selected.reshape(rows, columns).astype(np.float16)


def nonuniform_int4_grid(
    weight: np.ndarray, bits: int = BITS
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if (
        bits != BITS
        or weight.ndim != 2
        or weight.shape[1] % GROUP_SIZE
        or not np.isfinite(weight).all()
    ):
        raise ValueError("PW-0149 codebook input is invalid")
    rows, columns = weight.shape
    groups = columns // GROUP_SIZE
    grouped = np.asarray(weight, dtype=np.float32).reshape(rows, groups, GROUP_SIZE)
    codebooks = np.empty((rows, groups, LEVELS), dtype=np.float16)
    codes = np.empty((rows, groups, GROUP_SIZE), dtype=np.uint8)
    for start in range(0, rows, ROW_BATCH):
        end = min(start + ROW_BATCH, rows)
        values = grouped[start:end]
        ordered = np.sort(values, axis=2)
        centers = (
            ordered[:, :, 0:: GROUP_SIZE // LEVELS]
            + ordered[:, :, (GROUP_SIZE // LEVELS - 1) :: GROUP_SIZE // LEVELS]
        ) * np.float32(0.5)
        for _ in range(LLOYD_ITERATIONS):
            assignments = np.argmin(
                np.abs(values[..., None] - centers[:, :, None, :]), axis=3
            ).astype(np.uint8)
            for code in range(LEVELS):
                selected = assignments == code
                counts = selected.sum(axis=2)
                sums = np.where(selected, values, np.float32(0)).sum(
                    axis=2, dtype=np.float32
                )
                centers[:, :, code] = np.where(
                    counts > 0,
                    sums / np.maximum(counts, 1),
                    centers[:, :, code],
                )
        staged = np.sort(centers.astype(np.float16), axis=2)
        final_codes = np.argmin(
            np.abs(values[..., None] - staged.astype(np.float32)[:, :, None, :]),
            axis=3,
        ).astype(np.uint8)
        codebooks[start:end] = staged
        codes[start:end] = final_codes
    flat_codes = codes.reshape(rows, columns)
    quantized = reconstruct_codebook_grid(flat_codes, codebooks)
    return codebooks, np.empty(0, dtype=np.uint8), quantized, flat_codes


def global_hessian_codebook_fixed_grid(
    weight: np.ndarray,
    activations: np.ndarray,
    codebooks: np.ndarray,
    unused: np.ndarray,
    *,
    bits: int = BITS,
) -> tuple[np.ndarray, np.ndarray, dict]:
    if (
        bits != BITS
        or unused.shape != (0,)
        or codebooks.ndim != 3
        or codebooks.dtype != np.float16
        or codebooks.shape != (weight.shape[0], weight.shape[1] // GROUP_SIZE, LEVELS)
        or not np.isfinite(codebooks).all()
    ):
        raise ValueError("PW-0149 global-Hessian codebook metadata is invalid")
    rows = weight.shape[0]
    row_indices = np.arange(rows)

    def quantize_column(values: np.ndarray, original_column: int):
        group = original_column // GROUP_SIZE
        levels = codebooks[:, group, :].astype(np.float64)
        code = np.argmin(np.abs(values[:, None] - levels), axis=1)
        return code, levels[row_indices, code]

    def validate(codes: np.ndarray, quantized: np.ndarray) -> None:
        if not np.array_equal(quantized, reconstruct_codebook_grid(codes, codebooks)):
            raise ValueError("PW-0149 grid membership failed")

    result = global_hessian_nbit_fixed_grid(
        weight,
        activations,
        codebooks,
        unused,
        bits=bits,
        column_quantizer=quantize_column,
        result_validator=validate,
        grid_payloads=(codebooks,),
    )
    result[2]["lloyd_iterations"] = LLOYD_ITERATIONS
    result[2]["levels_per_group"] = LEVELS
    return result


def physical_ledger() -> dict:
    return _physical_ledger(NONUNIFORM_CONFIG)


def _gate(reports: list[dict]) -> dict:
    return _nbit_gate(reports, NONUNIFORM_CONFIG)


def run(
    checkpoint_root: Path,
    verification_path: Path,
    corpus_manifest_path: Path,
    pw0138_path: Path,
    pw0148_path: Path,
    output_path: Path,
    commit: str,
) -> dict:
    return _run_control(
        checkpoint_root,
        verification_path,
        corpus_manifest_path,
        pw0138_path,
        pw0148_path,
        output_path,
        commit,
        NONUNIFORM_CONFIG,
        nonuniform_int4_grid,
        global_hessian_codebook_fixed_grid,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--verification", required=True, type=Path)
    parser.add_argument("--corpus-manifest", required=True, type=Path)
    parser.add_argument("--pw0138", required=True, type=Path)
    parser.add_argument("--pw0148", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--commit", required=True)
    args = parser.parse_args()
    try:
        result = run(
            args.checkpoint,
            args.verification,
            args.corpus_manifest,
            args.pw0138,
            args.pw0148,
            args.output,
            args.commit,
        )
        print(json.dumps({"output": str(args.output), "decision": result["decision"]}))
        return 0
    except (OSError, RuntimeError, ValueError, KeyError, TypeError, json.JSONDecodeError, np.linalg.LinAlgError) as error:
        print(json.dumps({"error": str(error)}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
