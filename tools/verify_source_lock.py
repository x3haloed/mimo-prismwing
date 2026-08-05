#!/usr/bin/env python3
"""Fail closed when a pinned research-source checkout differs from its lock."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_source_lock(lock_path: Path, source_root: Path) -> dict[str, object]:
    lock = json.loads(lock_path.read_text())
    if lock.get("schema_version") != 1:
        raise ValueError("unsupported source-lock schema")
    revision = lock.get("revision")
    files = lock.get("files")
    if not isinstance(revision, str) or len(revision) != 40:
        raise ValueError("source lock has no full revision")
    if not isinstance(files, dict) or not files:
        raise ValueError("source lock has no files")

    actual_revision = subprocess.run(
        ["git", "-C", str(source_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if actual_revision != revision:
        raise ValueError(f"revision mismatch: expected {revision}, got {actual_revision}")

    verified: dict[str, str] = {}
    for relative, expected in sorted(files.items()):
        if not isinstance(relative, str) or not isinstance(expected, str):
            raise ValueError("source lock file identities must be strings")
        path = source_root / relative
        if not path.is_file():
            raise ValueError(f"locked source file missing: {relative}")
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(
                f"source hash mismatch for {relative}: expected {expected}, got {actual}"
            )
        verified[relative] = actual

    return {
        "schema_version": 1,
        "source_root": str(source_root),
        "revision": actual_revision,
        "verified_files": verified,
        "verified_file_count": len(verified),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", required=True, type=Path)
    parser.add_argument("--source-root", required=True, type=Path)
    arguments = parser.parse_args()
    print(json.dumps(verify_source_lock(arguments.lock, arguments.source_root), sort_keys=True))


if __name__ == "__main__":
    main()
