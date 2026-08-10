#!/usr/bin/env python3
"""Prepare, capture, and verify PW-0160's million-token reference probe."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import struct
import subprocess
import sys
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from tokenizers import Tokenizer
from transformers import AutoTokenizer

try:
    from tools.host_safety import HostSafetyMonitor, HostSafetyViolation
    from tools.openrouter_reference import canonical_json, read_key
except ModuleNotFoundError:
    from host_safety import HostSafetyMonitor, HostSafetyViolation
    from openrouter_reference import canonical_json, read_key


MODEL = "xiaomi/mimo-v2.5"
PROVIDER = "Parasail"
MODEL_REVISION = "63651580ca774f8504f676040460aed3e1244ac1"
TARGET_PROMPT_TOKENS = 1_000_000
MAX_OUTPUT_TOKENS = 16
SEED = "pw0160-million-token-reference-v1"
TARGET_SHA256 = "91fe6e0441bb0a0e1ab0852db60fb575d131b61ff069002c9c333f9b776e4950"
MODEL_LOCK_SHA256 = "df8c74e6f9e1cef154aae5881b9042777653206aaff72855f7b1a1340e0d1050"
TOKENIZER_SHA256 = "633518aad78f9f61bae2ae420d621215754a4424c918b052cd8c22a3b59e99d2"
TOKENIZER_CONFIG_SHA256 = "fd34b805f75a890a5c123d79a2982bbe240b3b6efb156d22401bd619484d9bd2"
MODELS_URL = "https://openrouter.ai/api/v1/models"
ENDPOINTS_URL = "https://openrouter.ai/api/v1/models/xiaomi/mimo-v2.5/endpoints"
CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_CHECKPOINT = Path("/Users/chad/Models/mimo-prismwing/checkpoints/MiMo-V2.5-63651580")
DEFAULT_KEY = Path.home() / ".config/mimo-prismwing/openrouter.key"
SCHEMA_VERSION = 1


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_new(path: Path, value: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(path, flags, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def read_json(path: Path) -> Any:
    with path.open("rb") as handle:
        return json.load(handle)


def git_value(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments], check=True, capture_output=True, text=True
    ).stdout.strip()


def require_clean_commit() -> str:
    commit = git_value("rev-parse", "HEAD")
    if len(commit) != 40:
        raise ValueError("git HEAD is not a full commit")
    if git_value("status", "--porcelain"):
        raise ValueError("PW-0160 execution requires a clean worktree")
    return commit


def authenticate_sources(repo: Path, checkpoint: Path) -> dict[str, Any]:
    paths = {
        "target": repo / "TARGET.md",
        "model_lock": repo / "spec/model.lock.json",
        "tokenizer": checkpoint / "tokenizer.json",
        "tokenizer_config": checkpoint / "tokenizer_config.json",
    }
    expected = {
        "target": TARGET_SHA256,
        "model_lock": MODEL_LOCK_SHA256,
        "tokenizer": TOKENIZER_SHA256,
        "tokenizer_config": TOKENIZER_CONFIG_SHA256,
    }
    observed = {name: sha256_file(path) for name, path in paths.items()}
    if observed != expected:
        raise ValueError(f"PW-0160 source identity mismatch: {observed}")
    lock = read_json(paths["model_lock"])
    if lock.get("revision") != MODEL_REVISION:
        raise ValueError("PW-0160 model revision mismatch")
    lock_files = {row["path"]: row for row in lock.get("files", [])}
    for name in ("tokenizer.json", "tokenizer_config.json"):
        if lock_files.get(name, {}).get("sha256") != expected[name.removesuffix(".json") if name == "tokenizer.json" else "tokenizer_config"]:
            raise ValueError(f"PW-0160 model lock does not bind {name}")
    return {
        name: {"path": str(paths[name]), "sha256": observed[name]}
        for name in paths
    } | {"model_revision": MODEL_REVISION}


def needle_code(seed: str = SEED) -> str:
    return "PW-" + hashlib.sha256(seed.encode("utf-8")).hexdigest()[:12].upper()


def token_id_sha256(token_ids: list[int]) -> str:
    digest = hashlib.sha256()
    for token_id in token_ids:
        if not 0 <= token_id <= 0xFFFFFFFF:
            raise ValueError("token ID outside u32 domain")
        digest.update(struct.pack("<I", token_id))
    return digest.hexdigest()


def token_span_for_text(encoding: Any, rendered: str, text: str) -> tuple[int, int]:
    if rendered.count(text) != 1:
        raise ValueError(f"expected one rendered occurrence of {text!r}")
    character_start = rendered.index(text)
    character_end = character_start + len(text)
    covered = [
        index
        for index, (start, end) in enumerate(encoding.offsets)
        if end > character_start and start < character_end
    ]
    if not covered or covered != list(range(covered[0], covered[-1] + 1)):
        raise ValueError(f"could not map rendered span for {text!r}")
    return covered[0], covered[-1] + 1


def build_probe(
    checkpoint: Path,
    target_prompt_tokens: int = TARGET_PROMPT_TOKENS,
    seed: str = SEED,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if target_prompt_tokens < 512:
        raise ValueError("probe needs at least 512 serialized tokens")
    backend = Tokenizer.from_file(str(checkpoint / "tokenizer.json"))
    source_tokenizer = AutoTokenizer.from_pretrained(
        checkpoint, local_files_only=True, trust_remote_code=True
    )
    code = needle_code(seed)
    prefix = (
        "At the beginning of this record, remember exactly this code: "
        f"{code}. The material between this declaration and the final question "
        "is neutral padding. Do not infer a different code from it."
    )
    suffix = (
        " End of record. What was the exact code declared at the beginning? "
        "Return only that code, with no explanation or punctuation."
    )

    def render(filler_count: int) -> tuple[str, str]:
        content = prefix + " a" * filler_count + suffix
        rendered = source_tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        if not isinstance(rendered, str):
            raise ValueError("chat template did not render text")
        return rendered, content

    empty_rendered, _ = render(0)
    filler_count = target_prompt_tokens - len(
        backend.encode(empty_rendered, add_special_tokens=False).ids
    )
    if filler_count < 0:
        raise ValueError("target is smaller than the fixed probe framing")
    for _ in range(8):
        rendered, content = render(filler_count)
        encoding = backend.encode(rendered, add_special_tokens=False)
        token_ids = encoding.ids
        delta = target_prompt_tokens - len(token_ids)
        if delta == 0:
            break
        filler_count += delta
        if filler_count < 0:
            raise ValueError("filler correction became negative")
    else:
        raise ValueError("could not construct the exact prompt-token count")
    if len(token_ids) != target_prompt_tokens:
        raise ValueError("exact prompt-token count did not converge")
    decoded = backend.decode(token_ids, skip_special_tokens=False)
    roundtrip = backend.encode(decoded, add_special_tokens=False).ids
    if decoded != rendered or roundtrip != token_ids:
        raise ValueError("rendered prefix does not round-trip through the pinned tokenizer")

    code_start, code_end = token_span_for_text(encoding, rendered, code)
    if code_start >= 256:
        raise ValueError("needle does not occur exactly once in the first 256 tokens")
    question = "What was the exact code declared at the beginning?"
    question_start, _ = token_span_for_text(encoding, rendered, question)
    if question_start < target_prompt_tokens - 256:
        raise ValueError("question is not uniquely within the final 256 tokens")
    answer_ids = backend.encode(code, add_special_tokens=False).ids
    if len(answer_ids) > MAX_OUTPUT_TOKENS:
        raise ValueError("derived code cannot fit the declared output budget")

    request = {
        "model": MODEL,
        "messages": [{"role": "user", "content": content}],
        "temperature": 0,
        "max_tokens": MAX_OUTPUT_TOKENS,
        "reasoning": {"enabled": False},
        "logprobs": True,
        "top_logprobs": 20,
        "stream": False,
        "provider": {
            "order": [PROVIDER],
            "allow_fallbacks": False,
            "require_parameters": True,
        },
    }
    request_bytes = canonical_json(request)
    generation = {
        "schema_version": SCHEMA_VERSION,
        "seed": seed,
        "needle_code": code,
        "needle_token_ids": token_ids[code_start:code_end],
        "needle_token_offset": code_start,
        "needle_token_end": code_end,
        "answer_token_ids": answer_ids,
        "question_token_offset": question_start,
        "filler": "one repeated U+0020 plus lowercase a token; artificial viability padding",
        "filler_count": filler_count,
        "prompt_tokens": len(token_ids),
        "token_id_encoding": "little-endian u32",
        "token_ids_sha256": token_id_sha256(token_ids),
        "first_256_token_ids": token_ids[:256],
        "last_256_token_ids": token_ids[-256:],
        "rendered_bytes": len(rendered.encode("utf-8")),
        "rendered_sha256": sha256_bytes(rendered.encode("utf-8")),
        "decode_reencode_exact": True,
        "request_bytes": len(request_bytes),
        "request_sha256": sha256_bytes(request_bytes),
    }
    return request, generation


def fetch(url: str, timeout_seconds: float = 60.0) -> tuple[bytes, dict[str, str]]:
    request = Request(url, headers={"User-Agent": "mimo-prismwing-reference/1"})
    with urlopen(request, timeout=timeout_seconds) as response:
        body = response.read()
        if response.status != 200:
            raise RuntimeError(f"metadata endpoint returned HTTP {response.status}")
        headers = {
            key.lower(): value
            for key, value in response.headers.items()
            if key.lower() in {"content-type", "etag", "last-modified", "x-request-id"}
        }
    json.loads(body)
    return body, headers


def validate_endpoint_metadata(models: Any, endpoints: Any) -> dict[str, Any]:
    model_rows = [row for row in models.get("data", []) if row.get("id") == MODEL]
    if len(model_rows) != 1:
        raise ValueError("OpenRouter models response lacks one exact MiMo row")
    model = model_rows[0]
    if not {"logprobs", "top_logprobs"}.issubset(model.get("supported_parameters", [])):
        raise ValueError("OpenRouter model row lacks required logprob parameters")
    root = endpoints.get("data", {})
    if root.get("id") != MODEL:
        raise ValueError("OpenRouter endpoint response model mismatch")
    rows = [row for row in root.get("endpoints", []) if row.get("provider_name") == PROVIDER]
    if len(rows) != 1:
        raise ValueError("OpenRouter endpoint response lacks one Parasail row")
    row = rows[0]
    if row.get("context_length", 0) < TARGET_PROMPT_TOKENS:
        raise ValueError("Parasail context is below one million tokens")
    if row.get("quantization") != "fp8":
        raise ValueError("Parasail quantization drifted from FP8")
    if not {"logprobs", "top_logprobs"}.issubset(row.get("supported_parameters", [])):
        raise ValueError("Parasail row lacks required logprob parameters")
    return {
        "model_context_length": model.get("context_length"),
        "provider_context_length": row["context_length"],
        "provider_quantization": row["quantization"],
        "provider_tag": row.get("tag"),
        "provider_pricing": row.get("pricing"),
        "provider_status": row.get("status"),
        "provider_supported_parameters": row["supported_parameters"],
    }


def prepare(repo: Path, checkpoint: Path, output: Path) -> Path:
    commit = require_clean_commit()
    safety = HostSafetyMonitor()
    safety.checkpoint("sources-start")
    sources = authenticate_sources(repo, checkpoint)
    safety.checkpoint("sources-authenticated")
    request, generation = build_probe(checkpoint)
    safety.checkpoint("million-token-request-generated")
    models_bytes, models_headers = fetch(MODELS_URL)
    endpoints_bytes, endpoints_headers = fetch(ENDPOINTS_URL)
    endpoint_summary = validate_endpoint_metadata(
        json.loads(models_bytes), json.loads(endpoints_bytes)
    )
    safety.checkpoint("endpoint-metadata-frozen")

    output.mkdir(parents=True, exist_ok=False)
    request_bytes = canonical_json(request)
    request_sha256 = sha256_bytes(request_bytes)
    models_sha256 = sha256_bytes(models_bytes)
    endpoints_sha256 = sha256_bytes(endpoints_bytes)
    atomic_write_new(output / "request.json", request_bytes)
    atomic_write_new(output / "models.json", models_bytes)
    atomic_write_new(output / "endpoints.json", endpoints_bytes)
    del request, request_bytes, models_bytes, endpoints_bytes
    safety.release_checkpoint(
        "prepare-buffers-released",
        ["million-token request", "source tokenizer", "endpoint metadata buffers"],
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "experiment": "PW-0160",
        "state": "prepared",
        "commit": commit,
        "sources": sources,
        "request_file": "request.json",
        "request_sha256": request_sha256,
        "models_file": "models.json",
        "models_sha256": models_sha256,
        "models_headers": models_headers,
        "endpoints_file": "endpoints.json",
        "endpoints_sha256": endpoints_sha256,
        "endpoints_headers": endpoints_headers,
        "endpoint_summary": endpoint_summary,
        "generation": generation,
        "captured_at_unix_ns": time.time_ns(),
        "safety_snapshots": safety.evidence(),
        "accepted_tokens": 0,
        "performance_claim": False,
        "endpoint_tps": None,
    }
    atomic_write_new(output / "prepared-manifest.json", canonical_json(manifest))
    return output / "prepared-manifest.json"


def verify_prepared(output: Path) -> dict[str, Any]:
    manifest = read_json(output / "prepared-manifest.json")
    if manifest.get("schema_version") != SCHEMA_VERSION or manifest.get("state") != "prepared":
        raise ValueError("unknown PW-0160 prepared manifest")
    for stem in ("request", "models", "endpoints"):
        path = output / manifest[f"{stem}_file"]
        if sha256_bytes(path.read_bytes()) != manifest[f"{stem}_sha256"]:
            raise ValueError(f"PW-0160 {stem} hash mismatch")
    request = read_json(output / manifest["request_file"])
    if request.get("model") != MODEL or request.get("provider", {}).get("order") != [PROVIDER]:
        raise ValueError("PW-0160 prepared request identity mismatch")
    generation = manifest.get("generation", {})
    if generation.get("prompt_tokens") != TARGET_PROMPT_TOKENS:
        raise ValueError("PW-0160 prepared prompt count mismatch")
    if sha256_bytes(canonical_json(request)) != generation.get("request_sha256"):
        raise ValueError("PW-0160 generation request hash mismatch")
    validate_endpoint_metadata(
        read_json(output / manifest["models_file"]),
        read_json(output / manifest["endpoints_file"]),
    )
    return manifest


def next_attempt(output: Path) -> tuple[int, Path]:
    existing = sorted(output.glob("attempt-*"))
    if len(existing) >= 3:
        raise ValueError("PW-0160 permits at most three attempts")
    index = len(existing) + 1
    path = output / f"attempt-{index:03d}"
    if path.exists():
        raise ValueError("PW-0160 attempt numbering is not contiguous")
    return index, path


def post_chat(request_bytes: bytes, key: str, timeout_seconds: float) -> tuple[bytes, dict[str, str], int]:
    http_request = Request(
        CHAT_URL,
        data=request_bytes,
        method="POST",
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "X-OpenRouter-Title": "mimo-prismwing-reference",
        },
    )
    started_ns = time.time_ns()
    with urlopen(http_request, timeout=timeout_seconds) as response:
        body = response.read()
        status = response.status
        headers = {
            key.lower(): value
            for key, value in response.headers.items()
            if key.lower() in {"content-type", "x-request-id"}
        }
    return body, headers, time.time_ns() - started_ns


def validate_response(response: Any, generation: dict[str, Any]) -> dict[str, Any]:
    api_error = response.get("error") if isinstance(response, dict) else None
    if isinstance(api_error, dict):
        raise ValueError(
            f"OpenRouter API error {api_error.get('code')}: {api_error.get('message')}"
        )
    if response.get("provider") != PROVIDER:
        raise ValueError("PW-0160 response provider drift")
    if response.get("model") not in {MODEL, "xiaomi/mimo-v2.5-20260422"}:
        raise ValueError("PW-0160 response model drift")
    choices = response.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise ValueError("PW-0160 response must contain exactly one choice")
    choice = choices[0]
    content = choice.get("message", {}).get("content")
    if not isinstance(content, str) or content.strip() != generation["needle_code"]:
        raise ValueError("PW-0160 needle answer mismatch")
    positions = (choice.get("logprobs") or {}).get("content") or []
    if not positions:
        raise ValueError("PW-0160 response lacks token logprobs")
    selected_bytes = bytearray()
    for position in positions:
        token = position.get("token")
        token_bytes = position.get("bytes")
        selected_logprob = position.get("logprob")
        if (
            not isinstance(token, str)
            or not isinstance(token_bytes, list)
            or any(not isinstance(value, int) or not 0 <= value <= 255 for value in token_bytes)
            or not isinstance(selected_logprob, (int, float))
            or not math.isfinite(selected_logprob)
        ):
            raise ValueError("PW-0160 response has malformed selected-token data")
        selected_bytes.extend(token_bytes)
        alternatives = position.get("top_logprobs")
        if not isinstance(alternatives, list) or len(alternatives) < 20:
            raise ValueError("PW-0160 response lacks top-20 at one position")
        for row in alternatives:
            logprob = row.get("logprob")
            if (
                not isinstance(row.get("token"), str)
                or not isinstance(logprob, (int, float))
                or not math.isfinite(logprob)
            ):
                raise ValueError("PW-0160 response has malformed top-logprob data")
    if selected_bytes.decode("utf-8") != content:
        raise ValueError("PW-0160 selected-token bytes do not align with content")
    usage = response.get("usage", {})
    if usage.get("prompt_tokens") != TARGET_PROMPT_TOKENS:
        raise ValueError("PW-0160 provider prompt-token count mismatch")
    details = usage.get("completion_tokens_details", {})
    if details.get("reasoning_tokens", 0) != 0:
        raise ValueError("PW-0160 unexpectedly used reasoning tokens")
    cost = usage.get("cost")
    if not isinstance(cost, (int, float)) or cost < 0 or cost > 0.50:
        raise ValueError("PW-0160 cost is absent or exceeds the experiment ceiling")
    return {
        "needle_exact_after_strip": True,
        "visible_logprob_positions": len(positions),
        "minimum_top_logprobs": min(len(row["top_logprobs"]) for row in positions),
        "prompt_tokens": usage["prompt_tokens"],
        "completion_tokens": usage.get("completion_tokens"),
        "cached_tokens": usage.get("prompt_tokens_details", {}).get("cached_tokens"),
        "reasoning_tokens": details.get("reasoning_tokens", 0),
        "cost_usd": cost,
        "finish_reason": choice.get("finish_reason"),
    }


def summarize_http_error_body(body: bytes) -> dict[str, Any] | None:
    try:
        parsed = json.loads(body)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    error = parsed.get("error") if isinstance(parsed, dict) else None
    if not isinstance(error, dict):
        return None
    metadata = error.get("metadata") if isinstance(error.get("metadata"), dict) else {}
    return {
        "code": error.get("code"),
        "message": error.get("message"),
        "provider_name": metadata.get("provider_name"),
        "is_byok": metadata.get("is_byok"),
        "limit_source": metadata.get("limit_source"),
        "remedy_hint": metadata.get("remedy_hint"),
    }


def capture(output: Path, key_path: Path, timeout_seconds: float) -> Path:
    prepared = verify_prepared(output)
    if require_clean_commit() != prepared.get("commit"):
        raise ValueError("PW-0160 capture commit differs from prepared commit")
    if stat.S_IMODE(key_path.stat().st_mode) & 0o077:
        raise ValueError("OpenRouter key file must be owner-only")
    key = read_key(key_path)
    index, attempt = next_attempt(output)
    attempt.mkdir()
    request_bytes = (output / prepared["request_file"]).read_bytes()
    safety = HostSafetyMonitor()
    safety.checkpoint("paid-request-start")
    status = "failed"
    response_bytes: bytes | None = None
    response_sha256: str | None = None
    response_headers: dict[str, str] = {}
    elapsed_ns: int | None = None
    error: dict[str, Any] | None = None
    error_body_file: str | None = None
    error_body_sha256: str | None = None
    error_body_summary: dict[str, Any] | None = None
    validation: dict[str, Any] | None = None
    try:
        response_bytes, response_headers, elapsed_ns = post_chat(
            request_bytes, key, timeout_seconds
        )
        atomic_write_new(attempt / "response.json", response_bytes)
        response_sha256 = sha256_bytes(response_bytes)
        safety.checkpoint("paid-response-frozen")
        response = json.loads(response_bytes)
        validation = validate_response(response, prepared["generation"])
        del response
        safety.checkpoint("paid-response-validated")
        status = "passed"
    except HTTPError as caught:
        body = caught.read()
        error_body_file = "http-error-body.bin"
        error_body_sha256 = sha256_bytes(body)
        error_body_summary = summarize_http_error_body(body)
        atomic_write_new(attempt / error_body_file, body)
        response_headers = {
            key.lower(): value
            for key, value in caught.headers.items()
            if key.lower() in {"content-type", "x-request-id", "retry-after"}
        }
        error = {"type": "HTTPError", "status": caught.code, "message": str(caught)}
    except (URLError, TimeoutError, OSError, ValueError, json.JSONDecodeError, HostSafetyViolation) as caught:
        error = {"type": type(caught).__name__, "message": str(caught)}
    finally:
        del key
        del request_bytes
        if response_bytes is not None:
            del response_bytes
    try:
        safety.release_checkpoint("attempt-final", ["paid request and response buffers"])
    except HostSafetyViolation as caught:
        status = "failed"
        error = {"type": type(caught).__name__, "message": str(caught)}
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "experiment": "PW-0160",
        "attempt": index,
        "status": status,
        "commit": prepared["commit"],
        "request_sha256": prepared["request_sha256"],
        "response_file": "response.json" if response_sha256 is not None else None,
        "response_sha256": response_sha256,
        "response_headers": response_headers,
        "elapsed_ns": elapsed_ns,
        "validation": validation,
        "error": error,
        "error_body_file": error_body_file,
        "error_body_sha256": error_body_sha256,
        "error_body_summary": error_body_summary,
        "safety_snapshots": safety.evidence(),
        "accepted_tokens": 0,
        "performance_claim": False,
        "endpoint_tps": None,
    }
    atomic_write_new(attempt / "manifest.json", canonical_json(manifest))
    if status != "passed":
        raise RuntimeError(f"PW-0160 attempt {index} failed: {error}")
    return attempt / "manifest.json"


def verify_attempt(output: Path, attempt_name: str) -> dict[str, Any]:
    prepared = verify_prepared(output)
    attempt = output / attempt_name
    manifest = read_json(attempt / "manifest.json")
    if manifest.get("schema_version") != SCHEMA_VERSION or manifest.get("status") != "passed":
        raise ValueError("PW-0160 attempt is not a passing known schema")
    if manifest.get("commit") != prepared.get("commit"):
        raise ValueError("PW-0160 attempt commit mismatch")
    if manifest.get("request_sha256") != prepared.get("request_sha256"):
        raise ValueError("PW-0160 attempt request mismatch")
    response_path = attempt / manifest["response_file"]
    response_bytes = response_path.read_bytes()
    if sha256_bytes(response_bytes) != manifest.get("response_sha256"):
        raise ValueError("PW-0160 response hash mismatch")
    validation = validate_response(json.loads(response_bytes), prepared["generation"])
    if validation != manifest.get("validation"):
        raise ValueError("PW-0160 response validation summary mismatch")
    return {"prepared": prepared, "attempt": manifest}


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser()
    commands = root.add_subparsers(dest="command", required=True)
    prepare_command = commands.add_parser("prepare")
    prepare_command.add_argument("output", type=Path)
    prepare_command.add_argument("--repo", type=Path, default=Path.cwd())
    prepare_command.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    capture_command = commands.add_parser("capture")
    capture_command.add_argument("output", type=Path)
    capture_command.add_argument("--key-file", type=Path, default=DEFAULT_KEY)
    capture_command.add_argument("--timeout-seconds", type=float, default=1200.0)
    verify_command = commands.add_parser("verify")
    verify_command.add_argument("output", type=Path)
    verify_command.add_argument("attempt")
    return root


def main() -> int:
    arguments = parser().parse_args()
    try:
        if arguments.command == "prepare":
            result = prepare(arguments.repo.resolve(), arguments.checkpoint.resolve(), arguments.output)
        elif arguments.command == "capture":
            result = capture(arguments.output, arguments.key_file, arguments.timeout_seconds)
        else:
            result = verify_attempt(arguments.output, arguments.attempt)
        print(result if isinstance(result, Path) else canonical_json(result).decode().strip())
        return 0
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError, subprocess.CalledProcessError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
