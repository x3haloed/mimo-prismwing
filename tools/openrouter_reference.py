#!/usr/bin/env python3
"""Capture and verify immutable OpenRouter reference probes.

The capture path deliberately uses only Python's standard library. Secrets are
read from a mode-0600 file, used only in the Authorization header, and never
written to the request, response, manifest, or console.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


API_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_KEY_PATH = Path.home() / ".config/mimo-prismwing/openrouter.key"
SCHEMA_VERSION = 1


def canonical_json(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def read_json(path: Path) -> Any:
    with path.open("rb") as handle:
        return json.load(handle)


def atomic_write_new(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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


def read_key(path: Path) -> str:
    mode = stat.S_IMODE(path.stat().st_mode)
    if mode & 0o077:
        raise ValueError(f"key file must be owner-only (0600), found {mode:04o}")
    key = path.read_text(encoding="utf-8").strip()
    if not key:
        raise ValueError("key file is empty")
    return key


def validate_capture_request(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("request must be a JSON object")
    if payload.get("model") != "xiaomi/mimo-v2.5":
        raise ValueError("model must be exactly xiaomi/mimo-v2.5")
    provider = payload.get("provider")
    if not isinstance(provider, dict):
        raise ValueError("provider policy is required")
    order = provider.get("order")
    if not isinstance(order, list) or len(order) != 1 or not isinstance(order[0], str):
        raise ValueError("provider.order must pin exactly one provider")
    if provider.get("allow_fallbacks") is not False:
        raise ValueError("provider.allow_fallbacks must be false")
    if provider.get("require_parameters") is not True:
        raise ValueError("provider.require_parameters must be true")
    if payload.get("logprobs") is not True or payload.get("top_logprobs") != 20:
        raise ValueError("logprobs=true and top_logprobs=20 are required")
    if payload.get("stream") is not False:
        raise ValueError("stream must be false for immutable capture")
    return payload


def capture(request_path: Path, output_dir: Path, key_path: Path) -> Path:
    payload = validate_capture_request(read_json(request_path))
    request_bytes = canonical_json(payload)
    key = read_key(key_path)
    http_request = Request(
        API_URL,
        data=request_bytes,
        method="POST",
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "X-OpenRouter-Title": "mimo-prismwing-reference",
        },
    )
    started_ns = time.time_ns()
    try:
        with urlopen(http_request, timeout=300) as response:
            response_bytes = response.read()
            status = response.status
            response_headers = {
                key.lower(): value
                for key, value in response.headers.items()
                if key.lower() in {"content-type", "x-request-id"}
            }
    except HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"OpenRouter HTTP {error.code}: {body}") from error
    except URLError as error:
        raise RuntimeError(f"OpenRouter request failed: {error.reason}") from error
    elapsed_ns = time.time_ns() - started_ns
    if status != 200:
        raise RuntimeError(f"unexpected OpenRouter status {status}")
    response_value = json.loads(response_bytes)
    response_canonical = canonical_json(response_value)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "request_file": "request.json",
        "request_sha256": sha256_bytes(request_bytes),
        "response_file": "response.json",
        "response_sha256": sha256_bytes(response_canonical),
        "http_status": status,
        "response_headers": response_headers,
        "elapsed_ns": elapsed_ns,
        "captured_at_unix_ns": started_ns,
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    try:
        atomic_write_new(output_dir / "request.json", request_bytes)
        atomic_write_new(output_dir / "response.json", response_canonical)
        atomic_write_new(output_dir / "manifest.json", canonical_json(manifest))
    except BaseException:
        for child in output_dir.iterdir():
            child.unlink()
        output_dir.rmdir()
        raise
    return output_dir / "manifest.json"


def verify(output_dir: Path) -> None:
    manifest = read_json(output_dir / "manifest.json")
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unknown capture manifest schema")
    for stem in ("request", "response"):
        filename = manifest.get(f"{stem}_file")
        expected = manifest.get(f"{stem}_sha256")
        if not isinstance(filename, str) or not isinstance(expected, str):
            raise ValueError(f"invalid {stem} manifest entry")
        actual = sha256_bytes(canonical_json(read_json(output_dir / filename)))
        if actual != expected:
            raise ValueError(f"{stem} hash mismatch: expected {expected}, got {actual}")
    validate_capture_request(read_json(output_dir / manifest["request_file"]))
    request = read_json(output_dir / manifest["request_file"])
    response = read_json(output_dir / manifest["response_file"])
    if response.get("model") is None or not isinstance(response.get("choices"), list):
        raise ValueError("response lacks model or choices")
    if response.get("provider") != request["provider"]["order"][0]:
        raise ValueError("response provider does not match pinned provider")
    if not response["choices"]:
        raise ValueError("response contains no choices")
    positions = (response["choices"][0].get("logprobs") or {}).get("content") or []
    if not positions:
        raise ValueError("response contains no token logprobs")
    if any(len(position.get("top_logprobs", [])) != 20 for position in positions):
        raise ValueError("response does not contain top-20 logprobs at every position")


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser()
    commands = root.add_subparsers(dest="command", required=True)
    capture_command = commands.add_parser("capture")
    capture_command.add_argument("request", type=Path)
    capture_command.add_argument("output", type=Path)
    capture_command.add_argument("--key-file", type=Path, default=DEFAULT_KEY_PATH)
    verify_command = commands.add_parser("verify")
    verify_command.add_argument("output", type=Path)
    return root


def main() -> int:
    arguments = parser().parse_args()
    try:
        if arguments.command == "capture":
            manifest = capture(arguments.request, arguments.output, arguments.key_file)
            print(manifest)
        else:
            verify(arguments.output)
            print(f"verified {arguments.output}")
        return 0
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
