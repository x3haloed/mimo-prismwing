#!/usr/bin/env python3
"""Generate PW-0156's reproducible 8K text-prefill route fixture."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from tokenizers import Tokenizer


SOURCE_COMMIT = "aca9a6044cd348244028850dbb798178695d6bd8"
SOURCE_PATHS = (
    "TARGET.md",
    "RED_LINES.md",
    "docs/VALIDATION_PROTOCOL.md",
    "LEARNINGS.md",
)
TEMPLATE_PATH = "evals/fixtures/real/pw0112-wide-route-trace.json"
TARGET_TOKENS = 8_000
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def git_blob(path: str) -> bytes:
    return subprocess.run(
        ["git", "show", f"{SOURCE_COMMIT}:{path}"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
    ).stdout


def build_fixture(tokenizer_path: Path) -> dict:
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    source_blobs = {path: git_blob(path) for path in SOURCE_PATHS}
    corpus = b"\n\n".join(
        f"===== {path} =====\n".encode() + source_blobs[path] for path in SOURCE_PATHS
    ).decode("utf-8")
    all_ids = tokenizer.encode(corpus, add_special_tokens=False).ids
    if len(all_ids) < TARGET_TOKENS:
        raise ValueError("pinned corpus does not contain 8,000 tokens")
    token_ids = all_ids[:TARGET_TOKENS]
    prompt = tokenizer.decode(token_ids, skip_special_tokens=False)
    if tokenizer.encode(prompt, add_special_tokens=False).ids != token_ids:
        raise ValueError("8K decoded prompt does not round-trip to the exact token prefix")
    template = json.loads(git_blob(TEMPLATE_PATH))
    template.update(
        {
            "schema_version": 5,
            "semantic": "mimo_v2_5_target_faithful_8k_prefill_route_coverage",
            "prompt_utf8": prompt,
            "expected_prompt_token_ids": token_ids,
            "full_prefix_trace_append_token_ids": None,
            "route_trace_positions": TARGET_TOKENS,
            "hosted_reference": None,
            "corpus_source_commit": SOURCE_COMMIT,
            "corpus_source_paths": list(SOURCE_PATHS),
            "corpus_source_sha256": {
                path: hashlib.sha256(source_blobs[path]).hexdigest()
                for path in SOURCE_PATHS
            },
        }
    )
    return template


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    if arguments.output.exists():
        raise ValueError(f"refusing to overwrite {arguments.output}")
    fixture = build_fixture(arguments.tokenizer)
    arguments.output.write_text(
        json.dumps(fixture, sort_keys=True, separators=(",", ":")) + "\n"
    )
    print(
        json.dumps(
            {
                "output": str(arguments.output),
                "tokens": len(fixture["expected_prompt_token_ids"]),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
