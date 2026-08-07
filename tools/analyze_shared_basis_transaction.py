#!/usr/bin/env python3
"""Execute PW-0117's algebra and production-shape operation audit."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

try:
    from tools.openrouter_reference import atomic_write_new, canonical_json
except ModuleNotFoundError:
    from openrouter_reference import atomic_write_new, canonical_json


PW0115_SHA256 = "41cc9b745561a09073902ba65354889d6b87e7d8716aea4db85940cbafc9c67a"
PW0116_SHA256 = "6007e93aa9cc280d20cab3db0f72851ad9f9722e9f225c07c3c1309cc5ef5e08"
MOBE_COMMIT = "7f3501da2a9f7b12d773cb52c454a0be0ceeb185"
CONFIGURATIONS = [(768, 4), (512, 8), (128, 32)]
P, D, K = 2048, 4096, 8


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fixture_audit(fixture: dict) -> dict:
    if fixture.get("schema_version") != 1 or fixture.get("semantic") != "shared_basis_routed_transaction_algebra":
        raise ValueError("PW-0117 fixture identity mismatch")
    a = np.asarray(fixture["A"], dtype=np.float64)
    b = np.asarray(fixture["B"], dtype=np.float64)
    coefficients = np.asarray(fixture["coefficients"], dtype=np.float64)
    common_input = np.asarray(fixture["common_input"], dtype=np.float64)
    down_inputs = np.asarray(fixture["down_inputs"], dtype=np.float64)
    route_weights = np.asarray(fixture["route_weights"], dtype=np.float64)
    combined = np.einsum("em,mrd->erd", coefficients, b)
    matrices = np.einsum("epr,erd->epd", a, combined)
    direct_gate_up = np.einsum("epd,d->ep", matrices, common_input)
    basis_on_input = np.einsum("mrd,d->mr", b, common_input)
    transaction_gate_up = np.einsum(
        "epr,er->ep", a, np.einsum("em,mr->er", coefficients, basis_on_input)
    )
    direct_down = sum(
        route_weights[e] * matrices[e].T @ down_inputs[e] for e in range(a.shape[0])
    )
    reduced_latents = np.einsum(
        "e,em,epr,ep->mr", route_weights, coefficients, a, down_inputs
    )
    transaction_down = np.einsum("mrd,mr->d", b, reduced_latents)
    expected_gate_up = np.asarray(fixture["expected_gate_up"], dtype=np.float64)
    expected_down = np.asarray(fixture["expected_down_mixture"], dtype=np.float64)
    gate_up_error = float(
        max(
            np.max(np.abs(direct_gate_up - transaction_gate_up)),
            np.max(np.abs(direct_gate_up - expected_gate_up)),
        )
    )
    down_error = float(
        max(
            np.max(np.abs(direct_down - transaction_down)),
            np.max(np.abs(direct_down - expected_down)),
        )
    )
    silu = lambda value: value / (1.0 + np.exp(-value))
    nonlinear = {}
    for name, activation, minimum in [
        ("silu", silu, fixture["minimum_silu_noncommutation"]),
        ("tanh", np.tanh, fixture["minimum_tanh_noncommutation"]),
    ]:
        after_combination = np.stack(
            [a[e] @ activation(combined[e]) @ common_input for e in range(a.shape[0])]
        )
        activate_then_combine = np.stack(
            [
                a[e]
                @ np.einsum("m,mrd->rd", coefficients[e], activation(b))
                @ common_input
                for e in range(a.shape[0])
            ]
        )
        maximum_delta = float(np.max(np.abs(after_combination - activate_then_combine)))
        if maximum_delta < minimum:
            raise ValueError(f"PW-0117 {name} counterexample is not material")
        nonlinear[name] = {"maximum_absolute_noncommutation": maximum_delta}
    if gate_up_error > 1e-12 or down_error > 1e-12:
        raise ValueError("PW-0117 transaction identity mismatch")
    return {
        "gate_up_maximum_absolute_error": gate_up_error,
        "down_mixture_maximum_absolute_error": down_error,
        "nonlinear_counterexamples": nonlinear,
    }


def analyze(pw0115_path: Path, pw0116_path: Path, fixture_path: Path) -> dict:
    if sha256_file(pw0115_path) != PW0115_SHA256 or sha256_file(pw0116_path) != PW0116_SHA256:
        raise ValueError("PW-0117 parent evidence hash mismatch")
    pw0115 = json.loads(pw0115_path.read_text())
    pw0116 = json.loads(pw0116_path.read_text())
    if not pw0115.get("all_projection_family_has_physically_eligible_configuration") or not pw0116.get("gates_passed"):
        raise ValueError("PW-0117 parent evidence decision mismatch")
    fixture_bytes = fixture_path.read_bytes()
    fixture = json.loads(fixture_bytes)
    fixture_result = _fixture_audit(fixture)
    source_multiplications = K * P * D
    rows = []
    for rank, bases in CONFIGURATIONS:
        linear = bases * rank * D + K * P * rank + K * bases * rank
        nonlinear_lower_bound = K * bases * rank * D + K * rank * D + K * P * rank
        rows.append(
            {
                "rank": rank,
                "basis_count": bases,
                "source_multiplications_per_projection": source_multiplications,
                "transaction_linear_multiplications_per_projection": linear,
                "transaction_linear_ratio": linear / source_multiplications,
                "published_nonlinear_lower_bound_multiplications_per_projection": nonlinear_lower_bound,
                "published_nonlinear_lower_bound_ratio": nonlinear_lower_bound / source_multiplications,
                "transaction_linear_compute_gate_passed": linear / source_multiplications <= 0.5,
                "published_nonlinear_compute_gate_passed": nonlinear_lower_bound / source_multiplications <= 0.5,
            }
        )
    analysis_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, text=True, capture_output=True
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"], check=True, text=True, capture_output=True
        ).stdout.strip()
    )
    return {
        "schema_version": 1,
        "evidence_class": "pw0117_shared_basis_routed_transaction_algebra",
        "analysis_commit": analysis_commit,
        "analysis_dirty": dirty,
        "mobe_implementation_commit": MOBE_COMMIT,
        "pw0115_manifest_sha256": PW0115_SHA256,
        "pw0116_manifest_sha256": PW0116_SHA256,
        "fixture_sha256": hashlib.sha256(fixture_bytes).hexdigest(),
        "fixture_result": fixture_result,
        "configurations": rows,
        "published_nonlinear_form_rejected_for_transaction_compute": all(
            not row["published_nonlinear_compute_gate_passed"] for row in rows
        ),
        "identity_basis_forms_remain_physically_eligible": all(
            row["transaction_linear_compute_gate_passed"] for row in rows
        ),
        "decision": "continue_identity_basis_only_to_weight_and_activation_fidelity_audit",
        "limitations": "algebra and multiplication counts only; identity-basis form is untrained and has no fidelity, kernel, wall-time, endpoint, accepted-token, or TPS result",
        "performance_claim": None,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pw0115", required=True, type=Path)
    parser.add_argument("--pw0116", required=True, type=Path)
    parser.add_argument("--fixture", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    try:
        result = analyze(arguments.pw0115, arguments.pw0116, arguments.fixture)
        atomic_write_new(arguments.output, canonical_json(result))
        print(json.dumps({"output": str(arguments.output), "decision": result["decision"]}))
        return 0
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
        print(json.dumps({"error": str(error)}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
