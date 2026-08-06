#!/usr/bin/env python3
"""Generate the PW-0102 Phase-B DFlash proposal from frozen target states."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
from safetensors import safe_open
import torch
from transformers import DynamicCache, Qwen3Config

try:
    from tools.dflash_semantics import (
        TARGET_LAYER_IDS,
        first_block_position_ids,
        initial_block_ids,
        install_greedy_draft_suffix,
        validate_first_block_cache_lengths,
    )
    from tools.host_safety import HostSafetyMonitor
    from tools.openrouter_reference import atomic_write_new, canonical_json
except ModuleNotFoundError:
    from dflash_semantics import (
        TARGET_LAYER_IDS,
        first_block_position_ids,
        initial_block_ids,
        install_greedy_draft_suffix,
        validate_first_block_cache_lengths,
    )
    from host_safety import HostSafetyMonitor
    from openrouter_reference import atomic_write_new, canonical_json


REVISION = "63651580ca774f8504f676040460aed3e1244ac1"
VERIFICATION_SHA256 = "9ddc8a99755f04ae2ea3c2484f6dd022d3f3a681b5a72c915ee4de833dbb0d03"
PW0091_SHA256 = "87466b59480a5a5b4256c490f1dfe670fe09f28d21d169085ab13bb1b4b7ab59"
ARTIFACT_SHA256 = "e67b0106aa2c26a091f1fef0661a4ccc408389f2bc5d1bab9ed42e46a6e898c6"
DFLASH_REVISION = "1f58446181abcaa01030fdbde835fbd38ae9a2b1"
SGLANG_REVISION = "2fc557254b3aaf539e80266e52a6d1e1f8da9980"
BASE_SHARD = "model_pp0_ep0_shard1.safetensors"
CONTEXT_LENGTH = 27
FIRST_TARGET_TOKEN = 264
CAPTURE_HASHES = {
    0: "140414ab207746639074e8a707f4a896dfc3d4c5628baf009b205797058a7292",
    11: "16a29c75cdacd917bd13d1f4ea84028f582dc53d3cd8a2cc3732690082eaf44b",
    23: "dd41254cf1b54c594996d8a6674583a57c844890255c1395b5410d4300c56c4b",
    35: "8e1537554798b30f752879befad5ab1e0911e31f6234aad46e99d34520407bbe",
    47: "3809f2fb5cc8ff3f543cd2d0362dccd136f2822e8aa844f6080d2effb7e6e300",
}
LAST_LOGITS_SHA256 = "c43be0909487235bddfe6e0de69aa42a98339faf43cd6b77d6ef4b5f1a853cab"
EXPECTED_UNEXPECTED_KEYS = {
    f"layers.{layer}.self_attn.attention_sink_bias" for layer in range(5)
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verified_file(path: Path, record: dict[str, Any]) -> None:
    stat = path.stat()
    if (
        not path.is_file()
        or record.get("status", "verified") != "verified"
        or stat.st_size != record.get("bytes")
        or stat.st_dev != record.get("device")
        or stat.st_ino != record.get("inode")
        or stat.st_mtime_ns != record.get("modified_ns")
    ):
        raise ValueError(f"verified file identity changed: {path}")


def authenticate_inputs(
    checkpoint: Path,
    verification_path: Path,
    prefix: Path,
    artifact_manifest_path: Path,
    dflash_root: Path,
    sglang_lock_path: Path,
    sglang_root: Path,
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    if sha256_file(verification_path) != VERIFICATION_SHA256:
        raise ValueError("checkpoint verification hash mismatch")
    verification = json.loads(verification_path.read_text())
    if verification.get("revision") != REVISION or verification.get("complete") is not True:
        raise ValueError("checkpoint verification identity mismatch")
    record = next(
        (item for item in verification.get("files", []) if item.get("path") == BASE_SHARD),
        None,
    )
    if record is None:
        raise ValueError("checkpoint verification lacks base shard")
    shard = checkpoint / BASE_SHARD
    verified_file(shard, record)

    manifest_path = prefix / "manifest.json"
    if sha256_file(manifest_path) != PW0091_SHA256:
        raise ValueError("PW-0091 manifest hash mismatch")
    prefix_manifest = json.loads(manifest_path.read_text())
    if (
        prefix_manifest.get("revision") != REVISION
        or prefix_manifest.get("semantic") != "mimo_full_prefix_layer_final_rust_trace"
        or prefix_manifest.get("prompt_token_ids") is None
        or len(prefix_manifest["prompt_token_ids"]) != CONTEXT_LENGTH
    ):
        raise ValueError("PW-0091 semantic identity mismatch")
    last_logits_record = prefix_manifest.get("captures", {}).get("last_logits")
    if (
        not isinstance(last_logits_record, dict)
        or last_logits_record.get("shape") != [152576]
        or last_logits_record.get("dtype") != "F32"
        or last_logits_record.get("sha256") != LAST_LOGITS_SHA256
    ):
        raise ValueError("PW-0091 last-logits identity mismatch")
    last_logits_path = prefix / last_logits_record["file"]
    if sha256_file(last_logits_path) != LAST_LOGITS_SHA256:
        raise ValueError("PW-0091 last-logits payload mismatch")
    last_logits = np.fromfile(last_logits_path, dtype="<f4")
    if last_logits.shape != (152576,) or int(last_logits.argmax()) != FIRST_TARGET_TOKEN:
        raise ValueError("PW-0091 first target token mismatch")

    if sha256_file(artifact_manifest_path) != ARTIFACT_SHA256:
        raise ValueError("DFlash artifact manifest hash mismatch")
    artifact = json.loads(artifact_manifest_path.read_text())
    if artifact.get("revision") != DFLASH_REVISION:
        raise ValueError("DFlash artifact revision mismatch")
    draft_record = next(
        item for item in artifact.get("verified_files", [])
        if item.get("path") == "dflash/dflash_draft_model.safetensors"
    )
    verified_file(dflash_root / draft_record["path"], draft_record)

    lock_bytes = sglang_lock_path.read_bytes()
    lock = json.loads(lock_bytes)
    if lock.get("revision") != SGLANG_REVISION:
        raise ValueError("SGLang DFlash source revision mismatch")
    for relative, expected in lock.get("files", {}).items():
        if sha256_file(sglang_root / relative) != expected:
            raise ValueError(f"SGLang DFlash source hash mismatch: {relative}")
    return shard, prefix_manifest, {
        "checkpoint_verification_sha256": VERIFICATION_SHA256,
        "pw0091_manifest_sha256": PW0091_SHA256,
        "artifact_manifest_sha256": ARTIFACT_SHA256,
        "sglang_lock_sha256": hashlib.sha256(lock_bytes).hexdigest(),
    }


def load_capture(prefix: Path, manifest: dict[str, Any], layer: int) -> torch.Tensor:
    name = f"layer_{layer:02}_final"
    record = manifest.get("captures", {}).get(name)
    if (
        not isinstance(record, dict)
        or record.get("shape") != [CONTEXT_LENGTH, 4096]
        or record.get("dtype") != "BF16_widened_F32"
        or record.get("sha256") != CAPTURE_HASHES[layer]
    ):
        raise ValueError(f"PW-0091 capture manifest mismatch: {name}")
    path = prefix / record["file"]
    if sha256_file(path) != record["sha256"]:
        raise ValueError(f"PW-0091 capture payload mismatch: {name}")
    values = np.fromfile(path, dtype="<f4")
    if values.size != CONTEXT_LENGTH * 4096:
        raise ValueError(f"PW-0091 capture size mismatch: {name}")
    tensor = torch.from_numpy(values.reshape(CONTEXT_LENGTH, 4096).copy()).to(torch.bfloat16)
    if not torch.isfinite(tensor).all():
        raise ValueError(f"PW-0091 capture is non-finite: {name}")
    return tensor


def load_published_class(source_path: Path):
    specification = importlib.util.spec_from_file_location("pw0102_pinned_dflash", source_path)
    if specification is None or specification.loader is None:
        raise RuntimeError("cannot import pinned DFlash source")
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module.DFlashDraftModel


def tensor_capture(output: Path, name: str, value: torch.Tensor) -> dict[str, Any]:
    widened = value.detach().float().cpu().contiguous().numpy().astype("<f4", copy=False)
    payload = widened.tobytes()
    atomic_write_new(output / f"{name}.f32", payload)
    return {
        "file": f"{name}.f32",
        "shape": list(value.shape),
        "dtype": "BF16_widened_F32" if value.dtype == torch.bfloat16 else "F32",
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def run(arguments: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    torch.set_num_threads(1)
    arguments.output.mkdir(parents=True, exist_ok=False)
    safety = HostSafetyMonitor()
    arguments._safety = safety
    timings: dict[str, float] = {}
    captures: dict[str, Any] = {}
    layer_states: list[dict[str, Any]] = []

    auth_started = time.monotonic()
    shard, prefix_manifest, identities = authenticate_inputs(
        arguments.checkpoint,
        arguments.verification,
        arguments.prefix,
        arguments.artifact_manifest,
        arguments.dflash_root,
        arguments.sglang_lock,
        arguments.sglang_root,
    )
    timings["authenticate_inputs_ms"] = (time.monotonic() - auth_started) * 1000
    safety.checkpoint("authenticated_inputs")

    hidden_parts = [load_capture(arguments.prefix, prefix_manifest, layer) for layer in TARGET_LAYER_IDS]
    target_hidden = torch.cat(hidden_parts, dim=-1).unsqueeze(0)
    del hidden_parts
    block_ids = initial_block_ids(FIRST_TARGET_TOKEN)
    with safe_open(shard, framework="pt", device="cpu") as source:
        embedding = source.get_slice("model.embed_tokens.weight")
        noise_embedding = torch.cat(
            [embedding[token : token + 1] for token in block_ids[0].tolist()], dim=0
        ).unsqueeze(0)
    if target_hidden.shape != (1, 27, 20480) or noise_embedding.shape != (1, 8, 4096):
        raise ValueError("DFlash frozen input shape mismatch")
    mask_embedding_hash = hashlib.sha256(
        noise_embedding[0, 1].float().contiguous().numpy().astype("<f4", copy=False).tobytes()
    ).hexdigest()
    safety.checkpoint("frozen_inputs_loaded")

    DFlashDraftModel = load_published_class(arguments.dflash_root / "dflash/dflash.py")
    config = Qwen3Config.from_json_file(arguments.dflash_root / "dflash/config.json")
    config._attn_implementation = "eager"
    load_started = time.monotonic()
    model, loading = DFlashDraftModel.from_pretrained(
        arguments.dflash_root / "dflash",
        config=config,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        output_loading_info=True,
    )
    timings["draft_load_ms"] = (time.monotonic() - load_started) * 1000
    missing = set(loading.get("missing_keys", []))
    unexpected = set(loading.get("unexpected_keys", []))
    mismatched = loading.get("mismatched_keys", [])
    errors = loading.get("error_msgs", [])
    if missing or unexpected != EXPECTED_UNEXPECTED_KEYS or mismatched or errors:
        raise ValueError(
            f"DFlash loading mismatch: missing={sorted(missing)}, "
            f"unexpected={sorted(unexpected)}, mismatched={mismatched}, errors={errors}"
        )
    model.eval()
    safety.checkpoint("draft_model_loaded")

    def hook(layer: int):
        def record(_module, _inputs, output_value):
            if output_value.shape != (1, 8, 4096) or output_value.dtype != torch.bfloat16:
                raise ValueError(f"draft layer {layer} output mismatch")
            payload = output_value.detach().float().cpu().contiguous().numpy().astype("<f4", copy=False).tobytes()
            layer_states.append(
                {"layer": layer, "shape": list(output_value.shape), "dtype": "BF16", "sha256": hashlib.sha256(payload).hexdigest()}
            )
            safety.checkpoint(f"draft_layer_{layer}_complete")
        return record

    hooks = [layer.register_forward_hook(hook(index)) for index, layer in enumerate(model.layers)]
    cache = DynamicCache()
    position_ids = first_block_position_ids(CONTEXT_LENGTH)
    forward_started = time.monotonic()
    with torch.inference_mode():
        draft_output = model(
            target_hidden=target_hidden,
            noise_embedding=noise_embedding,
            position_ids=position_ids,
            past_key_values=cache,
            use_cache=True,
            is_causal=False,
        )
    timings["draft_forward_ms"] = (time.monotonic() - forward_started) * 1000
    for handle in hooks:
        handle.remove()
    cache_lengths_before_crop = [cache.get_seq_length(index) for index in range(5)]
    if [item["layer"] for item in layer_states] != list(range(5)):
        raise ValueError("DFlash layer-hook sequence mismatch")
    validate_first_block_cache_lengths(cache_lengths_before_crop, CONTEXT_LENGTH)
    cache.crop(CONTEXT_LENGTH)
    cache_lengths_after_crop = [cache.get_seq_length(index) for index in range(5)]
    if cache_lengths_after_crop != [CONTEXT_LENGTH] * 5:
        raise ValueError("DFlash cache crop mismatch")
    captures["draft_final_hidden"] = tensor_capture(arguments.output, "draft_final_hidden", draft_output)
    safety.checkpoint("draft_hidden_captured")

    draft_suffix = draft_output[:, -7:, :].clone()
    del model, cache, target_hidden, noise_embedding, draft_output, position_ids, DFlashDraftModel
    safety.release_checkpoint(
        "draft_buffers_released",
        ["draft model parameters", "draft K/V cache", "target hidden features", "noise embeddings", "full draft output"],
    )

    lm_started = time.monotonic()
    with safe_open(shard, framework="pt", device="cpu") as source:
        lm_weight = source.get_tensor("lm_head.weight")
    if lm_weight.dtype != torch.bfloat16 or lm_weight.shape != (152576, 4096):
        raise ValueError("base LM-head layout mismatch")
    safety.checkpoint("base_lm_head_loaded")
    with torch.inference_mode():
        logits_bf16 = (draft_suffix.float() @ lm_weight.float().T).to(torch.bfloat16)
        logits = logits_bf16.float()
    timings["base_lm_head_ms"] = (time.monotonic() - lm_started) * 1000
    if logits.shape != (1, 7, 152576) or not torch.isfinite(logits).all():
        raise ValueError("draft logits output mismatch")
    captures["draft_logits"] = tensor_capture(arguments.output, "draft_logits", logits)
    proposed = install_greedy_draft_suffix(block_ids, logits)
    safety.checkpoint("draft_logits_captured_and_sampled")

    proposed_ids = proposed[0].tolist()
    del lm_weight, logits_bf16, logits, draft_suffix, proposed, block_ids
    safety.release_checkpoint(
        "lm_head_buffers_released",
        ["base LM-head weight", "draft suffix", "draft logits", "proposal tensor"],
    )
    safety.checkpoint("phase_b_final_health")
    snapshots = safety.evidence()
    result = {
        "schema_version": 1,
        "evidence_class": "pw0102_frozen_hidden_dflash_proposal",
        "status": "passed",
        "base_revision": REVISION,
        "dflash_revision": DFLASH_REVISION,
        "sglang_semantics_revision": SGLANG_REVISION,
        "identities": identities,
        "context_length": CONTEXT_LENGTH,
        "target_layer_ids": list(TARGET_LAYER_IDS),
        "initial_target_token_id": FIRST_TARGET_TOKEN,
        "mask_token_id": 151675,
        "mask_embedding_authority": "verified_base_model_embed_tokens_row",
        "mask_embedding_widened_f32_sha256": mask_embedding_hash,
        "exported_mask_embedding_used": False,
        "proposed_block_token_ids": proposed_ids,
        "loading_info": {
            "missing_keys": sorted(missing),
            "unexpected_keys": sorted(unexpected),
            "mismatched_keys": mismatched,
            "error_msgs": errors,
        },
        "cache_lengths_before_crop": cache_lengths_before_crop,
        "cache_lengths_after_crop": cache_lengths_after_crop,
        "layer_states": layer_states,
        "captures": captures,
        "timings_ms": timings,
        "physical_io": {
            "process_disk_bytes_read_delta": snapshots[-1]["process_disk_bytes_read"] - snapshots[0]["process_disk_bytes_read"],
            "process_disk_bytes_written_delta": snapshots[-1]["process_disk_bytes_written"] - snapshots[0]["process_disk_bytes_written"],
        },
        "runtime": {
            "python": sys.version,
            "torch": torch.__version__,
            "transformers": __import__("transformers").__version__,
            "numpy": np.__version__,
        },
        "safety": snapshots,
        "wall_ms": (time.monotonic() - started) * 1000,
        "performance_claim": None,
    }
    atomic_write_new(arguments.output / "manifest.json", canonical_json(result))
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--verification", required=True, type=Path)
    parser.add_argument("--prefix", required=True, type=Path)
    parser.add_argument("--artifact-manifest", required=True, type=Path)
    parser.add_argument("--dflash-root", required=True, type=Path)
    parser.add_argument("--sglang-lock", required=True, type=Path)
    parser.add_argument("--sglang-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    try:
        result = run(arguments)
        print(json.dumps({"output": str(arguments.output), "proposal": result["proposed_block_token_ids"]}))
        return 0
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as error:
        safety = getattr(arguments, "_safety", None)
        failure_path = arguments.output / "failure.json"
        if safety is not None and arguments.output.is_dir() and not failure_path.exists():
            failure = {
                "schema_version": 1,
                "evidence_class": "pw0102_frozen_hidden_dflash_proposal_failure",
                "status": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
                "safety": safety.evidence(),
            }
            try:
                atomic_write_new(failure_path, canonical_json(failure))
            except OSError:
                pass
        print(json.dumps({"error": str(error), "output": str(arguments.output)}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
