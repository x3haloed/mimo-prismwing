#!/usr/bin/env python3
"""Prepare PW-0176's immutable 64K token and sample-position authority."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import time
from typing import Any

from tokenizers import Tokenizer
from transformers import AutoTokenizer

try:
    from tools.host_safety import HostSafetyMonitor
    from tools.million_token_openrouter_reference import (
        MODEL_LOCK_SHA256,
        MODEL_REVISION,
        SEED,
        TARGET_SHA256,
        TOKENIZER_CONFIG_SHA256,
        TOKENIZER_SHA256,
        build_probe,
        canonical_json,
        token_id_sha256,
    )
except ModuleNotFoundError:
    from host_safety import HostSafetyMonitor
    from million_token_openrouter_reference import (
        MODEL_LOCK_SHA256,
        MODEL_REVISION,
        SEED,
        TARGET_SHA256,
        TOKENIZER_CONFIG_SHA256,
        TOKENIZER_SHA256,
        build_probe,
        canonical_json,
        token_id_sha256,
    )


SCHEMA_VERSION = 1
TARGET_TOKENS = 65_536
TOKEN_FILE = "token-ids.u32le"
PW0160_PREPARED_SHA256 = "482b90d87d8b4077a866bef599afca518739f25cea7d1f027c69395840ca743a"
PW0175_ANALYSIS_SHA256 = "e5ac56b7f710285cdeb0088f9fa750748ad74cbc68cd6d4dcb627061209a37ab"
MINFERENCE_FORWARD_SHA256 = "b368e765fcc2021591f7cdf970e4e1a71731ed66f19e97023a13b70a02e6edc2"
CHECKPOINT_VERIFICATION_SHA256 = "9ddc8a99755f04ae2ea3c2484f6dd022d3f3a681b5a72c915ee4de833dbb0d03"
PREDECESSOR_DOCUMENT_SHA256 = {
    "PW-0158": "1a53ba1e36a6173c2c2b17eb648d9ece0bea792de6ec1c0abaaa8220693aea71",
    "PW-0160": "0e9a7ac4fceef3149e256b7e4aca0ccb0c548c51e275ac4cfc31d11722e5413a",
    "PW-0161": "9b627a2addc208a9544ff00567d5647003677abf2e4b45f931a67c7cc9c37f9e",
    "PW-0175": "afe2d11aca1d3841903eb9a11dbcec823ac110e3885d224c8be47ca62318123c",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    with path.open("rb") as handle:
        return json.load(handle)


def atomic_write_new(path: Path, payload: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def require_clean_commit(repo: Path) -> str:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if len(commit) != 40 or dirty:
        raise ValueError("PW-0176 fixture preparation requires a clean full commit")
    return commit


def encode_request(checkpoint: Path, request: dict[str, Any]) -> tuple[str, list[int]]:
    messages = request.get("messages")
    if not isinstance(messages, list) or len(messages) != 1:
        raise ValueError("PW-0176 requires PW-0160's single-message probe")
    source_tokenizer = AutoTokenizer.from_pretrained(
        checkpoint, local_files_only=True, trust_remote_code=True
    )
    rendered = source_tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    if not isinstance(rendered, str):
        raise ValueError("chat template did not produce text")
    tokenizer = Tokenizer.from_file(str(checkpoint / "tokenizer.json"))
    token_ids = tokenizer.encode(rendered, add_special_tokens=False).ids
    decoded = tokenizer.decode(token_ids, skip_special_tokens=False)
    if decoded != rendered or tokenizer.encode(decoded, add_special_tokens=False).ids != token_ids:
        raise ValueError("PW-0176 token payload failed exact decode/re-encode")
    return rendered, token_ids


def sample_positions(prompt_tokens: int = TARGET_TOKENS) -> list[int]:
    if prompt_tokens != TARGET_TOKENS:
        raise ValueError("PW-0176 sample positions are frozen only for 65,536 tokens")
    positions = [63, 127, 255]
    positions.extend(range(4095, 65_536, 4096))
    positions.extend([65_509, 65_515, 65_520, 65_525, 65_530, 65_535])
    positions = sorted(set(positions))
    if positions[:3] != [63, 127, 255] or positions[-1] != 65_535 or len(positions) != 24:
        raise AssertionError("PW-0176 frozen sample schedule drifted")
    return positions


def build_fixture_payload(checkpoint: Path) -> tuple[dict[str, Any], bytes]:
    request, generation = build_probe(
        checkpoint, target_prompt_tokens=TARGET_TOKENS, seed=SEED
    )
    rendered, token_ids = encode_request(checkpoint, request)
    if (
        len(token_ids) != TARGET_TOKENS
        or token_id_sha256(token_ids) != generation["token_ids_sha256"]
        or generation["needle_token_offset"] >= 256
        or generation["question_token_offset"] != 65_509
        or generation["decode_reencode_exact"] is not True
    ):
        raise ValueError("PW-0176 generated token authority failed its edge/hash contract")
    payload = b"".join(struct.pack("<I", token_id) for token_id in token_ids)
    if hashlib.sha256(payload).hexdigest() != generation["token_ids_sha256"]:
        raise AssertionError("PW-0176 little-endian token encoding drifted")
    summary = {
        "seed": SEED,
        "prompt_tokens": TARGET_TOKENS,
        "token_id_encoding": "little-endian u32",
        "token_ids_sha256": generation["token_ids_sha256"],
        "token_payload_bytes": len(payload),
        "rendered_bytes": len(rendered.encode("utf-8")),
        "rendered_sha256": hashlib.sha256(rendered.encode("utf-8")).hexdigest(),
        "decode_reencode_exact": True,
        "needle_code": generation["needle_code"],
        "needle_token_offset": generation["needle_token_offset"],
        "needle_token_end": generation["needle_token_end"],
        "question_token_offset": generation["question_token_offset"],
        "first_256_token_ids": token_ids[:256],
        "last_256_token_ids": token_ids[-256:],
        "sample_positions": sample_positions(),
        "sample_position_bands": {
            "early": [63, 127, 255],
            "interval": list(range(4095, 65_536, 4096)),
            "final_question": [65_509, 65_515, 65_520, 65_525, 65_530, 65_535],
        },
    }
    return summary, payload


def authenticate_predecessors(
    repo: Path,
    checkpoint: Path,
    verification: Path,
    pw0160_prepared: Path,
    pw0175_analysis: Path,
    minference_forward: Path,
) -> dict[str, dict[str, str]]:
    sources = {
        "target": (repo / "TARGET.md", TARGET_SHA256),
        "model_lock": (repo / "spec/model.lock.json", MODEL_LOCK_SHA256),
        "tokenizer": (checkpoint / "tokenizer.json", TOKENIZER_SHA256),
        "tokenizer_config": (
            checkpoint / "tokenizer_config.json",
            TOKENIZER_CONFIG_SHA256,
        ),
        "checkpoint_verification": (
            verification,
            CHECKPOINT_VERIFICATION_SHA256,
        ),
        "pw0158_document": (
            repo / "experiments/PW-0158-million-context-p100-attention-bound.md",
            PREDECESSOR_DOCUMENT_SHA256["PW-0158"],
        ),
        "pw0160_document": (
            repo / "experiments/PW-0160-million-token-hosted-reference.md",
            PREDECESSOR_DOCUMENT_SHA256["PW-0160"],
        ),
        "pw0161_document": (
            repo / "experiments/PW-0161-volta-32gb-complete-envelope.md",
            PREDECESSOR_DOCUMENT_SHA256["PW-0161"],
        ),
        "pw0175_document": (
            repo / "experiments/PW-0175-structured-sparse-prefill-audit.md",
            PREDECESSOR_DOCUMENT_SHA256["PW-0175"],
        ),
        "pw0160_prepared": (pw0160_prepared, PW0160_PREPARED_SHA256),
        "pw0175_analysis": (pw0175_analysis, PW0175_ANALYSIS_SHA256),
        "minference_forward": (minference_forward, MINFERENCE_FORWARD_SHA256),
    }
    result: dict[str, dict[str, str]] = {}
    for name, (path, expected) in sources.items():
        observed = sha256_file(path)
        if observed != expected:
            raise ValueError(f"{name} SHA-256 mismatch")
        result[name] = {"path": str(path), "sha256": observed}
    prepared = read_json(pw0160_prepared)
    analysis = read_json(pw0175_analysis)
    verification_receipt = read_json(verification)
    if (
        prepared.get("experiment") != "PW-0160"
        or prepared.get("state") != "prepared"
        or prepared.get("generation", {}).get("seed") != SEED
        or prepared.get("generation", {}).get("prompt_tokens") != 1_000_000
        or analysis.get("evidence_class")
        != "pw0175_structured_sparse_prefill_continuation_audit"
        or analysis.get("decision")
        != "promote_mimo_specific_minference_style_oracle;reject_released_quest_as_prefill_repair"
        or verification_receipt.get("schema_version") != 1
        or verification_receipt.get("evidence_class")
        != "local_checkpoint_lock_verification"
        or verification_receipt.get("complete") is not True
        or verification_receipt.get("revision") != MODEL_REVISION
        or verification_receipt.get("lock_sha256") != MODEL_LOCK_SHA256
    ):
        raise ValueError("PW-0176 predecessor semantic identity mismatch")
    return result


def prepare(
    repo: Path,
    checkpoint: Path,
    verification: Path,
    pw0160_prepared: Path,
    pw0175_analysis: Path,
    minference_forward: Path,
    output: Path,
) -> Path:
    commit = require_clean_commit(repo)
    safety = HostSafetyMonitor()
    safety.checkpoint("pw0176_fixture_sources_start")
    sources = authenticate_predecessors(
        repo,
        checkpoint,
        verification,
        pw0160_prepared,
        pw0175_analysis,
        minference_forward,
    )
    safety.checkpoint("pw0176_fixture_sources_authenticated")
    generation, token_payload = build_fixture_payload(checkpoint)
    safety.checkpoint("pw0176_fixture_tokens_generated")
    output.mkdir(parents=True, exist_ok=False)
    atomic_write_new(output / TOKEN_FILE, token_payload)
    token_file_sha256 = sha256_file(output / TOKEN_FILE)
    if token_file_sha256 != generation["token_ids_sha256"]:
        raise AssertionError("persisted PW-0176 token payload hash mismatch")
    del token_payload
    safety.release_checkpoint("pw0176_fixture_buffers_released", ["token payload", "tokenizers"])
    safety.checkpoint("pw0176_fixture_final_service_health")
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "experiment": "PW-0176",
        "semantic": "mimo_64k_structured_sparse_oracle_token_authority",
        "revision": MODEL_REVISION,
        "commit": commit,
        "token_file": TOKEN_FILE,
        "token_file_sha256": token_file_sha256,
        "generation": generation,
        "sources": sources,
        "created_at_unix_ns": time.time_ns(),
        "safety_snapshots": safety.evidence(),
        "accepted_tokens": 0,
        "performance_claim": None,
        "endpoint_tps": None,
    }
    atomic_write_new(output / "manifest.json", canonical_json(manifest))
    return output / "manifest.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--verification", type=Path, required=True)
    parser.add_argument("--pw0160-prepared", type=Path, required=True)
    parser.add_argument("--pw0175-analysis", type=Path, required=True)
    parser.add_argument("--minference-forward", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    print(
        prepare(
            arguments.repo.resolve(),
            arguments.checkpoint.resolve(),
            arguments.verification.resolve(),
            arguments.pw0160_prepared.resolve(),
            arguments.pw0175_analysis.resolve(),
            arguments.minference_forward.resolve(),
            arguments.output.resolve(),
        )
    )


if __name__ == "__main__":
    main()
