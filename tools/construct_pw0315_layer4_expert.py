#!/usr/bin/env python3
"""Construct one pinned identity for the PW-0315 layer-4 K4 bank."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    import tools.construct_pw0314_layer4_k4 as base
except ModuleNotFoundError:
    import construct_pw0314_layer4_k4 as base


EXPERIMENT_ID = "PW-0315"
EXPERT_AUTHORITIES = {
    64: (
        "model_pp0_ep2_shard0.safetensors",
        "70639d2d3ad4bd80a3b3843632e17a5089baa3b2ac5565e571fb5ad7bafb0be0",
    ),
    96: (
        "model_pp0_ep3_shard0.safetensors",
        "f8c8ab1b22da717ed0360c8248da84d0f9a58af7a89deeb6d4021a67ae98a046",
    ),
    31: (
        "model_pp0_ep0_shard0.safetensors",
        "05586f8488a3540e951e5a5d7b8fd9a96d4046fbafc83ff9b25e851b72b99a50",
    ),
    232: (
        "model_pp0_ep7_shard0.safetensors",
        "1923bd1a8f3ca88ec78a0721cb36089f29528f55900841da3d09da51efaf8c23",
    ),
}
EXPECTED_PLACEMENTS = {64: 181, 96: 174, 31: 168, 232: 166}
EXPERT64_CONTROL = {
    "gate": {
        "candidate_array_sha256": "19032d4a9bc5f6021e2eba8308230b7e5a12b05694dde9d635859b4d8a72dc02",
        "packed_trellis_array_sha256": "410213a7453a71503973d09660a7a4e6ec078835e1c0783616b59ee66905ea7b",
        "manifest_sha256": "d742ae0ba705df0398e69e1edb23055b20b3e6c55943360bc9de9d6ccad9883e",
    },
    "up": {
        "candidate_array_sha256": "63b0f2856a933a0d4cc9241ac81ad93fa92534b67140093c3399b9623e7dabf8",
        "packed_trellis_array_sha256": "90a27b5dba7b042e94ac17e07c9dfc190b74386da9e96f1f62a20b16f684ff73",
        "manifest_sha256": "353b5d86e4ea331420720ff2893d2f6916574517f5022449c8583e302e4ae86e",
    },
    "down": {
        "candidate_array_sha256": "7d223c7eaa84582346d4f89e622d7476f71bc71b9ab22cc731c47e5182e3639e",
        "packed_trellis_array_sha256": "5f113cf27321d97e0591e91b1d6bb7adb8baf83e060fb6250269b058579a3f37",
        "manifest_sha256": "a65770f47c853fd37216a818fe2683dc411401587105887a7969b970fa5fc97f",
    },
}


def configure(expert: int) -> None:
    if expert not in EXPERT_AUTHORITIES:
        raise ValueError(f"expert must be one of {list(EXPERT_AUTHORITIES)}")
    shard, digest = EXPERT_AUTHORITIES[expert]
    original_verify = base.verify_full_checkpoint_install

    def verify(checkpoint_root: Path, receipt_path: Path):
        return original_verify(
            checkpoint_root, receipt_path, layer=base.LAYER, expert=expert
        )

    base.EXPERIMENT_ID = EXPERIMENT_ID
    base.EXPERT = expert
    base.SOURCE_SHARD = shard
    base.SOURCE_SHARD_SHA256 = digest
    base.EXPECTED_PROJECTION_HASHES = EXPERT64_CONTROL if expert == 64 else None
    base.EXPECTED_PLACEMENT_COUNT = EXPECTED_PLACEMENTS[expert]
    base.verify_full_checkpoint_install = verify


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expert", required=True, type=int)
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
        configure(arguments.expert)
        result = base.construct(
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
    except (FileExistsError, KeyError, OSError, RuntimeError, ValueError) as error:
        print(json.dumps({"error": str(error)}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
