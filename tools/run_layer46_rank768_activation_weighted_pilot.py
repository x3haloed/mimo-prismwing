#!/usr/bin/env python3
"""Run PW-0122 through the shared activation-weighted pilot executor."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from tools.run_rank768_activation_weighted_expert_pilot import PW0122_SPEC, run
except ModuleNotFoundError:
    from run_rank768_activation_weighted_expert_pilot import PW0122_SPEC, run


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--verification", required=True, type=Path)
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--pw0119", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--commit", required=True)
    arguments = parser.parse_args()
    try:
        result = run(
            arguments.checkpoint,
            arguments.verification,
            arguments.corpus,
            arguments.pw0119,
            arguments.output,
            arguments.commit,
            PW0122_SPEC,
        )
        print(json.dumps({"output": str(arguments.output), "decision": result["decision"]}))
        return 0
    except (OSError, RuntimeError, ValueError, KeyError, TypeError, StopIteration, json.JSONDecodeError) as error:
        print(json.dumps({"error": str(error)}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
