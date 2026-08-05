#!/usr/bin/env python3
"""Create and verify a fail-closed lock for a pinned Hugging Face snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any
from urllib.parse import quote
from urllib.request import urlopen

try:
    from tools.openrouter_reference import atomic_write_new, canonical_json
except ModuleNotFoundError:  # Direct execution places tools/ first on sys.path.
    from openrouter_reference import atomic_write_new, canonical_json


SCHEMA_VERSION = 1


def sha256_file(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_json(url: str) -> Any:
    with urlopen(url, timeout=60) as response:
        return json.load(response)


def create_lock(repository: str, revision: str, checkpoint_dir: Path) -> dict[str, Any]:
    encoded_repository = quote(repository, safe="/")
    encoded_revision = quote(revision, safe="")
    model = fetch_json(
        f"https://huggingface.co/api/models/{encoded_repository}/revision/{encoded_revision}"
    )
    resolved = model.get("sha")
    if resolved != revision:
        raise ValueError(f"revision resolved to {resolved!r}, expected {revision!r}")
    tree = fetch_json(
        f"https://huggingface.co/api/models/{encoded_repository}/tree/{encoded_revision}"
        "?recursive=true&expand=true&limit=100"
    )
    files = []
    for entry in tree:
        if entry.get("type") != "file":
            continue
        relative = entry.get("path")
        size = entry.get("size")
        oid = entry.get("oid")
        if not isinstance(relative, str) or not isinstance(size, int) or not isinstance(oid, str):
            raise ValueError("malformed upstream tree entry")
        local_path = checkpoint_dir / relative
        lfs = entry.get("lfs")
        if isinstance(lfs, dict):
            expected_hash = lfs.get("oid")
            if not isinstance(expected_hash, str) or len(expected_hash) != 64:
                raise ValueError(f"missing LFS SHA-256 for {relative}")
            hash_source = "huggingface_lfs_oid"
        else:
            if not local_path.is_file():
                raise ValueError(f"non-LFS file must be downloaded before locking: {relative}")
            expected_hash = sha256_file(local_path)
            hash_source = "downloaded_content"
        record = {
            "path": relative,
            "bytes": size,
            "git_oid": oid,
            "sha256": expected_hash,
            "sha256_source": hash_source,
        }
        if local_path.is_file() and local_path.stat().st_size == size:
            actual = sha256_file(local_path)
            if actual != expected_hash:
                raise ValueError(f"downloaded file hash mismatch: {relative}")
            record["locally_verified"] = True
        else:
            record["locally_verified"] = False
        files.append(record)
    if not files:
        raise ValueError("upstream tree contained no files")
    files.sort(key=lambda item: item["path"])
    return {
        "schema_version": SCHEMA_VERSION,
        "repository": repository,
        "revision": revision,
        "files": files,
        "file_count": len(files),
        "total_bytes": sum(item["bytes"] for item in files),
    }


def verify_lock(
    lock_path: Path,
    checkpoint_dir: Path,
    require_complete: bool,
) -> dict[str, Any]:
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unknown checkpoint lock schema")
    files = lock.get("files")
    if not isinstance(files, list) or len(files) != lock.get("file_count"):
        raise ValueError("checkpoint lock file count mismatch")
    if sum(item.get("bytes", -1) for item in files) != lock.get("total_bytes"):
        raise ValueError("checkpoint lock byte total mismatch")
    missing = []
    observations = []
    for item in files:
        path = checkpoint_dir / item["path"]
        if not path.is_file():
            missing.append(item["path"])
            observations.append({"path": item["path"], "status": "missing"})
            continue
        if path.stat().st_size != item["bytes"]:
            raise ValueError(f"size mismatch: {item['path']}")
        identity = path.stat()
        actual_sha256 = sha256_file(path)
        if actual_sha256 != item["sha256"]:
            raise ValueError(f"SHA-256 mismatch: {item['path']}")
        observations.append(
            {
                "path": item["path"],
                "status": "verified",
                "bytes": item["bytes"],
                "sha256": actual_sha256,
                "device": identity.st_dev,
                "inode": identity.st_ino,
                "modified_ns": identity.st_mtime_ns,
            }
        )
    if require_complete and missing:
        raise ValueError(f"checkpoint incomplete: {len(missing)} files missing")
    print(f"verified {len(files) - len(missing)}/{len(files)} files; missing={len(missing)}")
    return {
        "schema_version": SCHEMA_VERSION,
        "evidence_class": "local_checkpoint_lock_verification",
        "lock_sha256": sha256_file(lock_path),
        "repository": lock.get("repository"),
        "revision": lock.get("revision"),
        "require_complete": require_complete,
        "complete": not missing,
        "verified_files": len(files) - len(missing),
        "missing_files": missing,
        "files": observations,
    }


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser()
    commands = root.add_subparsers(dest="command", required=True)
    create = commands.add_parser("create")
    create.add_argument("--repository", required=True)
    create.add_argument("--revision", required=True)
    create.add_argument("--checkpoint-dir", required=True, type=Path)
    create.add_argument("--output", required=True, type=Path)
    verify = commands.add_parser("verify")
    verify.add_argument("--lock", required=True, type=Path)
    verify.add_argument("--checkpoint-dir", required=True, type=Path)
    verify.add_argument("--require-complete", action="store_true")
    verify.add_argument("--manifest", type=Path)
    return root


def main() -> int:
    arguments = parser().parse_args()
    try:
        if arguments.command == "create":
            lock = create_lock(arguments.repository, arguments.revision, arguments.checkpoint_dir)
            atomic_write_new(arguments.output, canonical_json(lock))
            print(arguments.output)
        else:
            result = verify_lock(
                arguments.lock, arguments.checkpoint_dir, arguments.require_complete
            )
            if arguments.manifest is not None:
                atomic_write_new(arguments.manifest, canonical_json(result))
                print(arguments.manifest)
        return 0
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
