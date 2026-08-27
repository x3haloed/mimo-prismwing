#!/usr/bin/env python3
"""Fail-closed audit CLI for repaired target-bonus native-MTP q8 reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from tools.audit_native_mtp_window_corpus import (
        CATEGORIES,
        audit_target_bonus,
    )
except ModuleNotFoundError:
    from audit_native_mtp_window_corpus import CATEGORIES, audit_target_bonus


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    parser.add_argument("progress", type=Path)
    parser.add_argument("hidden", type=Path)
    parser.add_argument("--category", required=True, choices=sorted(CATEGORIES))
    parser.add_argument("--commit")
    parser.add_argument("--prompt", type=Path)
    args = parser.parse_args()
    try:
        result = audit_target_bonus(
            args.report,
            args.progress,
            args.hidden,
            category=args.category,
            commit=args.commit,
            prompt_path=args.prompt,
        )
    except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError) as error:
        print(json.dumps({"error": str(error)}))
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
