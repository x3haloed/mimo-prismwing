#!/usr/bin/env python3
"""Validate PW-0122 through the shared activation-pilot analyzer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from tools.analyze_rank768_activation_weighted_expert_pilot import (
        PW0122_ANALYSIS_SPEC,
        analyze,
    )
    from tools.openrouter_reference import atomic_write_new, canonical_json
except ModuleNotFoundError:
    from analyze_rank768_activation_weighted_expert_pilot import (
        PW0122_ANALYSIS_SPEC,
        analyze,
    )
    from openrouter_reference import atomic_write_new, canonical_json


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    try:
        result = analyze(arguments.source, PW0122_ANALYSIS_SPEC)
        atomic_write_new(arguments.output, canonical_json(result))
        print(json.dumps({"output": str(arguments.output), "decision": result["decision"]}))
        return 0
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
        print(json.dumps({"error": str(error)}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
