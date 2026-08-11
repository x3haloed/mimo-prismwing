#!/usr/bin/env python3
"""Run the predeclared PW-0200 real-expert projection audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess


REFERENCE_MANIFEST_SHA256 = "9c96d85e45832abdccd3be2325db993749579a904469d1862c8f3437cafab86d"
INDEX_SHA256 = "f2e1774c9acf9a62338b68c144e6fc7a66495e59f2e64b3078c1b7ef5a196816"
EXPERTS = (232, 31, 64, 96, 9, 88, 245, 130)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_new(path: Path, payload: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as output:
        output.write(payload)
        output.flush()
        os.fsync(output.fileno())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--index", required=True, type=Path)
    parser.add_argument("--kernel", required=True, type=Path)
    parser.add_argument("--reference-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    if sha256(args.index) != INDEX_SHA256:
        raise ValueError("checkpoint index SHA-256 mismatch")
    reference_manifest = args.reference_root / "manifest.json"
    if sha256(reference_manifest) != REFERENCE_MANIFEST_SHA256:
        raise ValueError("PW-0101 reference manifest SHA-256 mismatch")
    manifest = json.loads(reference_manifest.read_bytes())
    routes = manifest["layer4_routes"]["selected_experts"]
    if tuple(routes) != EXPERTS:
        raise ValueError("PW-0101 selected expert identity mismatch")
    index = json.loads(args.index.read_bytes())["weight_map"]
    args.output.mkdir(parents=True, exist_ok=False)

    reports: list[dict] = []
    for expert in EXPERTS:
        prefix = f"model.layers.4.mlp.experts.{expert}"
        for projection in ("gate", "up", "down"):
            name = f"{prefix}.{projection}_proj.weight"
            scale = name + "_scale_inv"
            shard_name = index[name]
            if index.get(scale) != shard_name:
                raise ValueError(f"weight/scale shard disagreement: {name}")
            input_path = (
                args.reference_root / "layer4_moe_input.f32"
                if projection != "down"
                else args.reference_root / f"expert_{expert}_swiglu.f32"
            )
            reference_path = args.reference_root / f"expert_{expert}_{projection}.f32"
            stem = f"expert-{expert}-{projection}"
            output_path = args.output / f"{stem}.f32"
            report_path = args.output / f"{stem}.json"
            command = [
                str(args.binary),
                "metal-direct-source-bf16-fp8-gemv-audit",
                str(args.checkpoint / shard_name),
                str(args.kernel),
                name,
                scale,
                str(input_path),
                str(reference_path),
                str(output_path),
                str(report_path),
            ]
            subprocess.run(command, check=True, stdout=subprocess.DEVNULL)
            report = json.loads(report_path.read_bytes())
            report["expert"] = expert
            report["projection"] = projection
            reports.append(report)

    total_outputs = sum(report["output_f32"] for report in reports)
    total_mismatches = sum(report["bf16_mismatch_count"] for report in reports)
    summary = {
        "schema_version": 1,
        "semantic": "pw0200_real_layer4_selected_expert_bf16_repair_density",
        "reference_manifest_sha256": REFERENCE_MANIFEST_SHA256,
        "checkpoint_index_sha256": INDEX_SHA256,
        "experts": list(EXPERTS),
        "projection_reports": [
            {
                "expert": report["expert"],
                "projection": report["projection"],
                "parity_gate_passed": report["parity_gate_passed"],
                "bf16_mismatch_count": report["bf16_mismatch_count"],
                "bf16_mismatch_fraction": report["bf16_mismatch_fraction"],
                "relative_l2": report["relative_l2"],
                "maximum_absolute_error": report["maximum_absolute_error"],
                "output_sha256": report["output_sha256"],
                "source_buffer_copy_bytes": report["source_buffer_copy_bytes"],
            }
            for report in reports
        ],
        "total_outputs": total_outputs,
        "total_bf16_mismatches": total_mismatches,
        "aggregate_bf16_mismatch_fraction": total_mismatches / total_outputs,
        "passing_projections": sum(report["parity_gate_passed"] for report in reports),
        "accepted_tokens": 0,
        "A": 0,
        "U": 0,
        "performance_claim": None,
    }
    payload = (json.dumps(summary, sort_keys=True, separators=(",", ":")) + "\n").encode()
    write_new(args.output / "manifest.json", payload)


if __name__ == "__main__":
    main()
