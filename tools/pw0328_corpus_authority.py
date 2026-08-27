#!/usr/bin/env python3
"""Fail-closed shared authority for the corrected PW-0328 q8 corpus.

The canonical manifest is an index, not an oracle.  This module authenticates
its 24 bound artifacts and independently reconstructs the repaired transaction
semantics, causal histories, and raw verifier routes before exposing either a
q8-window view or the authorized q1 event stream.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import struct
from typing import Any

try:
    from tools.host_safety import HostSafetyPolicy
except ModuleNotFoundError:
    from host_safety import HostSafetyPolicy


PW0328_MANIFEST_PATH = Path(
    "/Volumes/Elements/mimo-prismwing/evidence/PW-0328/corpus-001/manifest.json"
)
PW0328_MANIFEST_SHA256 = (
    "36e4f10b6f807f766c87ee7078f5f18ea8fc339dd12e4dbc24f1f4ac6e824403"
)
PW0328_CAPTURE_COMMIT = "26d2ea31852c0d63bd022df6d571fd722137c39f"
REVISION = "63651580ca774f8504f676040460aed3e1244ac1"
MODEL_LOCK_SHA256 = "df8c74e6f9e1cef154aae5881b9042777653206aaff72855f7b1a1340e0d1050"
CHECKPOINT_VERIFICATION_SHA256 = (
    "9ddc8a99755f04ae2ea3c2484f6dd022d3f3a681b5a72c915ee4de833dbb0d03"
)
TOKENIZER_SHA256 = "633518aad78f9f61bae2ae420d621215754a4424c918b052cd8c22a3b59e99d2"
TOKENIZER_CONFIG_SHA256 = (
    "fd34b805f75a890a5c123d79a2982bbe240b3b6efb156d22401bd619484d9bd2"
)
CAPTURE_KERNEL_SHA256 = "9bc149eee32ebf28af35929d5fa160edfe9e1767cdcde59a54ec61b7016882ee"

CATEGORIES = ("ordinary", "code", "multilingual", "rare_route")
PROMPTS = {
    "ordinary": (
        "evals/fixtures/requests/pw0208-ordinary.txt",
        "d15e7fad81828b710303ce5e9dc5fd9c2104450108eb627167e6bc2080b9ee5d",
    ),
    "code": (
        "evals/fixtures/requests/pw0208-code.txt",
        "ad2940784d5028baa1dfab4585cb3a5a7fbffa22ca224f455fabc851549daefa",
    ),
    "multilingual": (
        "evals/fixtures/requests/pw0208-multilingual.txt",
        "6ece2dd3189d6b482f3356d344db6228e428db60a7530283eedc39be77d1beca",
    ),
    "rare_route": (
        "evals/fixtures/requests/pw0208-rare-route.txt",
        "5a71638364fff89af264dd3acea1ce31ef92128c3922cc8fb64826e793643373",
    ),
}

MANIFEST_EVIDENCE_CLASS = "pw0328_target_bonus_balanced_q8_causal_corpus"
MANIFEST_SEMANTIC = (
    "first_eight_chronological_target_bonus_q8_windows_per_category_"
    "with_transaction_zero_prefill_and_complete_segmented_target_history"
)
GENERATION_EVIDENCE_CLASS = "pw0208_native_mtp_corrected_window_capture"
GENERATION_SEMANTIC = (
    "mimo_v2_5_pw0208_native_mtp_corrected_verifier_window_capture_"
    "target_bonus_full_match_v1"
)
PREFILL_EVIDENCE_CLASS = "pw0208_native_mtp_corrected_prefill_hidden_capture"
PREFILL_SEMANTIC = "mimo_v2_5_pw0208_corrected_target_layer47_prefill_hidden_capture"
VERIFIER_HIDDEN_SEMANTIC = (
    "consecutive_target_layer_47_final_hidden_before_model_final_norm_for_each_"
    "width_eight_verifier_window_and_row"
)
PREFILL_HIDDEN_SEMANTIC = (
    "target_layer_47_final_hidden_before_model_final_norm_for_each_serialized_prompt_token"
)
TARGET_SELF_PROPOSER = (
    "greedy source-checkpoint proposer using the same retained K/V, deinterleaved "
    "checkpoint-TP QKV layout, and SGLang-directed block-scaled Metal arithmetic"
)

WIDTH = 8
LAYERS = 48
ROUTED_LAYERS = 47
HIDDEN = 4096
VOCAB = 152_576
HIDDEN_ROW_BYTES = HIDDEN * 4
WINDOW_BYTES = WIDTH * HIDDEN_ROW_BYTES
SOURCE_EXPERT_BYTES = 25_171_968

_MANIFEST_KEYS = {
    "accepted_tokens",
    "batch_size",
    "builder_commit",
    "builder_git_dirty",
    "builder_safety_snapshots",
    "concurrency",
    "control",
    "evidence_class",
    "experiment_id",
    "hidden_dtype",
    "performance_claim",
    "prefill_sources",
    "primary_windows",
    "rare_route_evidence",
    "schema_version",
    "semantic",
    "source_expert_bytes",
    "sources",
    "status",
    "verifier_window_shape",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _mapping(value: Any, message: str) -> dict[str, Any]:
    _require(isinstance(value, dict), message)
    return value


def _sequence(value: Any, message: str) -> list[Any]:
    _require(isinstance(value, list), message)
    return value


def _integer(value: Any, message: str, *, minimum: int = 0) -> int:
    _require(type(value) is int and value >= minimum, message)
    return value


def _positive_number(value: Any, message: str) -> float:
    _require(
        type(value) in (int, float)
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) > 0.0,
        message,
    )
    return float(value)


def _tokens(value: Any, message: str, *, allow_empty: bool = False) -> list[int]:
    _require(
        isinstance(value, list)
        and (allow_empty or bool(value))
        and all(type(token) is int and 0 <= token < VOCAB for token in value),
        message,
    )
    return list(value)


def _json(path: Path, label: str) -> dict[str, Any]:
    value = json.loads(path.read_text())
    return _mapping(value, f"{label}: JSON root")


def _authenticate_file(path: Path, expected: str, label: str) -> dict[str, Any]:
    actual = sha256_file(path)
    _require(actual == expected, f"{label}: SHA-256 mismatch")
    return {"path": str(path), "sha256": actual, "bytes": path.stat().st_size}


def _finite_f32(path: Path, expected_bytes: int, label: str) -> None:
    _require(path.stat().st_size == expected_bytes, f"{label}: byte size")
    with path.open("rb") as source:
        while payload := source.read(1024 * 1024):
            _require(len(payload) % 4 == 0, f"{label}: alignment")
            _require(
                all(math.isfinite(value[0]) for value in struct.iter_unpack("<f", payload)),
                f"{label}: non-finite value",
            )


def target_bonus_commit(proposal: list[int], posterior: list[int]) -> dict[str, Any]:
    """Independently apply the repaired full-match target-bonus commit rule."""

    proposal = _tokens(proposal, "proposal tokens")
    posterior = _tokens(posterior, "posterior tokens")
    _require(
        len(proposal) == len(posterior) and len(proposal) >= 2,
        "target-bonus commit requires equal widths of at least two",
    )
    mismatch = next(
        (
            index
            for index in range(len(proposal) - 1)
            if posterior[index] != proposal[index + 1]
        ),
        None,
    )
    if mismatch is not None:
        correction = posterior[mismatch]
        return {
            "authorized": [*proposal[1 : mismatch + 1], correction],
            "retained_proposal_rows": mismatch + 1,
            "next_anchor_token_id": correction,
            "proposal_converged": False,
        }
    target_bonus = posterior[-1]
    return {
        "authorized": [*proposal[1:], target_bonus],
        "retained_proposal_rows": len(proposal),
        "next_anchor_token_id": target_bonus,
        "proposal_converged": True,
    }


def validate_gate8(snapshots: Any, *, label: str) -> dict[str, Any]:
    """Validate recorded Gate 8 snapshots without probing current host state."""

    rows = _sequence(snapshots, f"{label}: safety snapshots")
    _require(bool(rows), f"{label}: missing safety snapshots")
    policy = HostSafetyPolicy()
    first_services = _mapping(rows[0].get("protected_service_pids"), f"{label}: services")
    baseline_names = {
        name for name, pids in first_services.items() if isinstance(pids, list) and pids
    }
    _require(bool(baseline_names), f"{label}: missing protected service baseline")
    for index, raw in enumerate(rows):
        row = _mapping(raw, f"{label}: safety row {index}")
        free = _integer(row.get("system_memory_free_percent"), f"{label}: free memory", minimum=1)
        footprint = _integer(
            row.get("process_physical_footprint_bytes"), f"{label}: footprint", minimum=1
        )
        peak = _integer(
            row.get("process_peak_resident_bytes"), f"{label}: peak RSS", minimum=1
        )
        release = row.get("release_boundary")
        _require(type(release) is bool, f"{label}: release-boundary type")
        _require(
            free >= policy.minimum_system_memory_free_percent
            and footprint <= policy.maximum_process_physical_footprint_bytes
            and peak <= policy.maximum_process_physical_footprint_bytes
            and (
                not release
                or footprint <= policy.maximum_post_release_physical_footprint_bytes
            ),
            f"{label}: memory safety mismatch",
        )
        _require(
            type(row.get("swap_growth_bytes")) is int
            and row["swap_growth_bytes"] == policy.maximum_swap_growth_bytes
            and type(row.get("new_throttled_pages")) is int
            and row["new_throttled_pages"] == policy.maximum_new_throttled_pages,
            f"{label}: swap/throttling safety mismatch",
        )
        services = _mapping(row.get("protected_service_pids"), f"{label}: services")
        _require(
            all(isinstance(services.get(name), list) and services[name] for name in baseline_names),
            f"{label}: protected service disappeared",
        )
    _require(
        any(bool(_mapping(row, f"{label}: safety row").get("release_boundary")) for row in rows),
        f"{label}: missing release boundary",
    )
    return {
        "pass": True,
        "minimum_system_memory_free_percent": min(
            int(row["system_memory_free_percent"]) for row in rows
        ),
        "maximum_process_physical_footprint_bytes": max(
            int(row["process_physical_footprint_bytes"]) for row in rows
        ),
        "maximum_process_peak_resident_bytes": max(
            int(row["process_peak_resident_bytes"]) for row in rows
        ),
        "maximum_swap_growth_bytes": max(int(row["swap_growth_bytes"]) for row in rows),
        "maximum_new_throttled_pages": max(
            int(row["new_throttled_pages"]) for row in rows
        ),
        "release_boundary_present": True,
        "protected_service_names": sorted(baseline_names),
    }


def reconstruct_verification_routes(traces: Any, *, label: str) -> dict[str, Any]:
    """Validate and expose all eight target-verifier route rows."""

    traces = _sequence(traces, f"{label}: verification traces")
    _require(len(traces) == LAYERS, f"{label}: verification layer count")
    identities: set[tuple[int, int]] = set()
    per_layer: list[dict[str, Any]] = []
    all_rows = [{"position": position, "layers": []} for position in range(WIDTH)]
    layer_u: list[float] = []
    for layer, raw_trace in enumerate(traces):
        trace = _mapping(raw_trace, f"{label}: layer {layer}")
        _require(type(trace.get("layer")) is int and trace["layer"] == layer, f"{label}: layer order")
        selected = trace.get("selected_experts_by_position")
        weights = trace.get("route_weights_by_position")
        if layer == 0:
            _require(selected == [] and weights == [], f"{label}: dense layer routes")
            _require(
                math.isclose(float(trace.get("U")), 0.0, abs_tol=1.0e-12),
                f"{label}: dense layer U",
            )
            continue
        selected = _sequence(selected, f"{label}: selected rows")
        weights = _sequence(weights, f"{label}: weight rows")
        _require(
            len(selected) == WIDTH and len(weights) == WIDTH,
            f"{label}: q8 route row count",
        )
        union: set[int] = set()
        layer_rows: list[dict[str, Any]] = []
        for position, (raw_experts, raw_weights) in enumerate(zip(selected, weights, strict=True)):
            experts = _sequence(raw_experts, f"{label}: expert row")
            weight_row = _sequence(raw_weights, f"{label}: weight row")
            _require(
                len(experts) == 8
                and len(set(experts)) == 8
                and all(type(expert) is int and 0 <= expert < 256 for expert in experts),
                f"{label}: expert row identity",
            )
            _require(
                len(weight_row) == 8
                and all(
                    type(weight) in (int, float)
                    and not isinstance(weight, bool)
                    and math.isfinite(float(weight))
                    and float(weight) > 0.0
                    for weight in weight_row
                )
                and abs(math.fsum(map(float, weight_row)) - 1.0) <= 2.0e-5,
                f"{label}: route weight row",
            )
            canonical_experts = sorted(experts)
            union.update(experts)
            identities.update((layer, expert) for expert in experts)
            row = {
                "position": position,
                "experts": list(experts),
                "expert_set": canonical_experts,
                "weights": [float(weight) for weight in weight_row],
            }
            layer_rows.append(row)
            all_rows[position]["layers"].append({"layer": layer, **row})
        derived_u = len(union) / 8.0
        _require(
            math.isclose(float(trace.get("U")), derived_u, abs_tol=1.0e-12),
            f"{label}: layer U",
        )
        layer_u.append(derived_u)
        per_layer.append(
            {
                "layer": layer,
                "rows": layer_rows,
                "union_experts": sorted(union),
                "identities": [
                    {"layer": layer, "expert": expert} for expert in sorted(union)
                ],
                "union_size": len(union),
                "U": derived_u,
            }
        )
    _require(len(per_layer) == ROUTED_LAYERS, f"{label}: routed layer count")
    mean_u = math.fsum(layer_u) / ROUTED_LAYERS
    identity_rows = [
        {"layer": layer, "expert": expert} for layer, expert in sorted(identities)
    ]
    return {
        "U": mean_u,
        "identities": identity_rows,
        "unique_identities": len(identity_rows),
        "unique_source_expert_bytes": len(identity_rows) * SOURCE_EXPERT_BYTES,
        "per_layer_q8": per_layer,
        "all_q8_rows": all_rows,
    }


def validate_proposal_routes(traces: Any, *, label: str) -> None:
    """Authenticate proposal traces while keeping them out of target route views."""

    steps = _sequence(traces, f"{label}: proposal traces")
    _require(len(steps) == WIDTH - 1, f"{label}: proposal step count")
    for step, raw_layers in enumerate(steps):
        layers = _sequence(raw_layers, f"{label}: proposal step {step}")
        _require(len(layers) == LAYERS, f"{label}: proposal layer count")
        for layer, raw_trace in enumerate(layers):
            trace = _mapping(raw_trace, f"{label}: proposal layer")
            _require(type(trace.get("layer")) is int and trace["layer"] == layer, f"{label}: proposal layer order")
            selected = trace.get("selected_experts_by_position")
            weights = trace.get("route_weights_by_position")
            if layer == 0:
                _require(selected == [] and weights == [], f"{label}: proposal dense route")
                _require(
                    math.isclose(float(trace.get("U")), 0.0, abs_tol=1.0e-12),
                    f"{label}: proposal dense U",
                )
                continue
            _require(
                isinstance(selected, list)
                and isinstance(weights, list)
                and len(selected) == len(weights) == 1,
                f"{label}: proposal route row count",
            )
            experts = selected[0]
            weight_row = weights[0]
            _require(
                isinstance(experts, list)
                and len(experts) == 8
                and len(set(experts)) == 8
                and all(type(expert) is int and 0 <= expert < 256 for expert in experts),
                f"{label}: proposal expert row",
            )
            _require(
                isinstance(weight_row, list)
                and len(weight_row) == 8
                and all(
                    type(weight) in (int, float)
                    and not isinstance(weight, bool)
                    and math.isfinite(float(weight))
                    and float(weight) > 0.0
                    for weight in weight_row
                )
                and abs(math.fsum(map(float, weight_row)) - 1.0) <= 2.0e-5,
                f"{label}: proposal weights",
            )
            _require(
                math.isclose(float(trace.get("U")), 8.0, abs_tol=1.0e-12),
                f"{label}: proposal U",
            )


def authorized_q1_rows(all_q8_rows: Any, full_a: int) -> list[dict[str, Any]]:
    """Project exactly the first full-authorized A verifier rows into q1 sets."""

    rows = _sequence(all_q8_rows, "q1 source rows")
    full_a = _integer(full_a, "full verifier A", minimum=1)
    _require(len(rows) == WIDTH and full_a <= WIDTH, "q1 full-A range")
    result: list[dict[str, Any]] = []
    for expected_position, raw in enumerate(rows[:full_a]):
        row = _mapping(raw, "q1 source row")
        _require(row.get("position") == expected_position, "q1 row order")
        layers = _sequence(row.get("layers"), "q1 layers")
        _require(len(layers) == ROUTED_LAYERS, "q1 routed-layer count")
        projected_layers = []
        for expected_layer, raw_layer in enumerate(layers, start=1):
            layer = _mapping(raw_layer, "q1 layer")
            experts = layer.get("expert_set")
            _require(
                layer.get("layer") == expected_layer
                and isinstance(experts, list)
                and len(experts) == 8
                and experts == sorted(set(experts)),
                "q1 layer set identity",
            )
            projected_layers.append({"layer": expected_layer, "experts": list(experts)})
        result.append({"position": expected_position, "layers": projected_layers})
    _require(
        len(result) == full_a
        and [row["position"] for row in result] == list(range(full_a)),
        "q1 suffix/clipping boundary",
    )
    return result


def validate_primary_window_sequence(windows: Any) -> None:
    """Reject category, corpus-index, or transaction reordering."""

    rows = _sequence(windows, "primary windows")
    _require(len(rows) == len(CATEGORIES) * WIDTH, "primary window cardinality")
    for index, raw in enumerate(rows):
        row = _mapping(raw, "primary window")
        _require(
            row.get("corpus_index") == index
            and row.get("category") == CATEGORIES[index // WIDTH]
            and row.get("transaction_index") == index % WIDTH,
            "primary category/window order",
        )


def validate_manifest_window(
    expected: Any, reconstructed: Any, *, index: int
) -> None:
    """Compare a manifest row to replay and keep full A distinct from clipping."""

    expected = _mapping(expected, f"window {index}: manifest")
    reconstructed = _mapping(reconstructed, f"window {index}: replay")
    _require(expected == reconstructed, f"PW-0328 window {index} replay")
    authorized = _tokens(
        reconstructed.get("verifier_authorized_token_ids"),
        f"window {index}: authorized tokens",
    )
    emitted = _tokens(
        reconstructed.get("observable_emitted_token_ids"),
        f"window {index}: observable tokens",
    )
    _require(
        reconstructed.get("A") == len(authorized)
        and reconstructed.get("observable_A") == len(emitted)
        and emitted == authorized[: len(emitted)],
        f"window {index}: full-A versus observable-A authority",
    )


def _transaction(
    raw: Any,
    *,
    index: int,
    terminal: bool,
    progress: Any,
    label: str,
) -> dict[str, Any]:
    transaction = _mapping(raw, f"{label}: transaction")
    _require(
        _integer(transaction.get("index"), f"{label}: transaction index") == index,
        f"{label}: non-contiguous transaction",
    )
    proposal = _tokens(transaction.get("proposal_token_ids"), f"{label}: proposal")
    posterior = _tokens(transaction.get("posterior_token_ids"), f"{label}: posterior")
    _require(len(proposal) == len(posterior) == WIDTH, f"{label}: transaction width")
    committed = target_bonus_commit(proposal, posterior)
    authorized = _tokens(
        transaction.get("verifier_authorized_token_ids"), f"{label}: authorized tokens"
    )
    emitted = _tokens(transaction.get("emitted_token_ids"), f"{label}: emitted tokens")
    _require(authorized == committed["authorized"], f"{label}: target-bonus commit")
    _require(emitted == authorized[: len(emitted)], f"{label}: observable authority prefix")
    verifier_retained = _integer(
        transaction.get("verifier_retained_proposal_rows"),
        f"{label}: verifier retention",
        minimum=1,
    )
    retained = _integer(
        transaction.get("retained_proposal_rows"), f"{label}: retention", minimum=1
    )
    _require(
        verifier_retained == committed["retained_proposal_rows"]
        and retained == len(emitted)
        and transaction.get("proposal_converged") is committed["proposal_converged"],
        f"{label}: convergence/retention",
    )
    if not terminal:
        _require(
            emitted == authorized and retained == verifier_retained,
            f"{label}: clipped nonterminal A",
        )
    validate_proposal_routes(transaction.get("proposal_layer_traces"), label=label)
    routes = reconstruct_verification_routes(
        transaction.get("verification_layer_traces"), label=label
    )
    _require(
        math.isclose(float(transaction.get("U")), routes["U"], abs_tol=1.0e-12),
        f"{label}: transaction U",
    )
    progress = _mapping(progress, f"{label}: progress")
    _require(
        progress.get("phase") == "transaction_complete"
        and type(progress.get("transaction")) is int
        and progress["transaction"] == index
        and type(progress.get("emitted_tokens")) is int
        and progress["emitted_tokens"] == len(emitted)
        and type(progress.get("retained_proposal_rows")) is int
        and progress["retained_proposal_rows"] == retained
        and progress.get("proposal_converged") is committed["proposal_converged"]
        and math.isclose(float(progress.get("U")), routes["U"], abs_tol=1.0e-12),
        f"{label}: progress semantic closure",
    )
    logical = _integer(
        transaction.get("logical_source_bytes"), f"{label}: logical bytes", minimum=1
    )
    physical = _integer(
        transaction.get("process_disk_bytes_read"), f"{label}: physical bytes", minimum=1
    )
    _require(
        type(progress.get("process_disk_bytes_read")) is int
        and progress["process_disk_bytes_read"] == physical,
        f"{label}: progress physical bytes",
    )
    proposal_wall = _positive_number(
        transaction.get("proposal_wall_ms"), f"{label}: proposal wall"
    )
    verification_wall = _positive_number(
        transaction.get("verification_wall_ms"), f"{label}: verification wall"
    )
    _require(
        math.isclose(proposal_wall, float(progress.get("proposal_wall_ms")), abs_tol=1.0e-9)
        and math.isclose(
            verification_wall, float(progress.get("verification_wall_ms")), abs_tol=1.0e-9
        ),
        f"{label}: progress timing closure",
    )
    return {
        "proposal_token_ids": proposal,
        "posterior_token_ids": posterior,
        "verifier_authorized_token_ids": authorized,
        "observable_emitted_token_ids": emitted,
        "proposal_converged": committed["proposal_converged"],
        "next_anchor_token_id": committed["next_anchor_token_id"],
        "A": len(authorized),
        "observable_A": len(emitted),
        "verifier_retained_proposal_rows": verifier_retained,
        "retained_proposal_rows": retained,
        "U": routes["U"],
        "route": {
            "U": routes["U"],
            "unique_identities": routes["unique_identities"],
            "unique_source_expert_bytes": routes["unique_source_expert_bytes"],
            "identities": routes["identities"],
        },
        "route_views": routes,
        "transaction_logical_source_bytes": logical,
        "transaction_process_disk_bytes_read": physical,
        "proposal_wall_ms": proposal_wall,
        "verification_wall_ms": verification_wall,
    }


def _expected_source_paths(
    manifest_path: Path, category: str, prefill_entry: dict[str, Any]
) -> dict[str, Path]:
    generation_root = manifest_path.parent / category
    prefill_report = Path(str(prefill_entry.get("report_file")))
    prefill_hidden = Path(str(prefill_entry.get("hidden_file")))
    _require(
        prefill_report.name == "report.json"
        and prefill_hidden.name == "target-layer47-hidden.f32"
        and prefill_report.parent == prefill_hidden.parent
        and prefill_report.parent.name == category
        and prefill_report.parent.parent.name == "prefill-001",
        f"{category}: prefill path authority",
    )
    return {
        "generation_report": generation_root / "report.json",
        "generation_progress": generation_root / "report.progress.jsonl",
        "generation_hidden": generation_root / "verification-layer47-hidden.f32",
        "prefill_report": prefill_report,
        "prefill_hidden": prefill_hidden,
    }


def _category(
    *,
    category: str,
    source_entry: Any,
    prefill_entry: Any,
    manifest_path: Path,
    repo: Path,
    first_corpus_index: int,
) -> dict[str, Any]:
    source_entry = _mapping(source_entry, f"{category}: source entry")
    prefill_entry = _mapping(prefill_entry, f"{category}: prefill entry")
    _require(
        source_entry.get("category") == prefill_entry.get("category") == category,
        f"{category}: source category/order",
    )
    paths = _expected_source_paths(manifest_path, category, prefill_entry)
    _require(
        Path(str(source_entry.get("report_file"))) == paths["generation_report"]
        and Path(str(source_entry.get("progress_file"))) == paths["generation_progress"]
        and Path(str(source_entry.get("hidden_file"))) == paths["generation_hidden"],
        f"{category}: generation path authority",
    )
    prompt_relative, prompt_sha = PROMPTS[category]
    _require(
        source_entry.get("prompt_file") == prompt_relative
        and source_entry.get("prompt_sha256") == prompt_sha,
        f"{category}: prompt manifest identity",
    )
    artifacts: list[dict[str, Any]] = []
    for kind, field, hash_field in (
        ("generation_report", "generation_report", "report_sha256"),
        ("generation_progress", "generation_progress", "progress_sha256"),
        ("generation_hidden", "generation_hidden", "hidden_sha256"),
    ):
        artifact = _authenticate_file(paths[field], str(source_entry.get(hash_field)), f"{category}: {kind}")
        artifacts.append({"category": category, "kind": kind, **artifact})
    for kind, field, hash_field in (
        ("prefill_report", "prefill_report", "report_sha256"),
        ("prefill_hidden", "prefill_hidden", "hidden_sha256"),
    ):
        artifact = _authenticate_file(paths[field], str(prefill_entry.get(hash_field)), f"{category}: {kind}")
        artifacts.append({"category": category, "kind": kind, **artifact})
    prompt_path = repo / prompt_relative
    prompt_artifact = _authenticate_file(prompt_path, prompt_sha, f"{category}: prompt")
    artifacts.append({"category": category, "kind": "prompt", **prompt_artifact})

    report = _json(paths["generation_report"], f"{category}: generation report")
    _require(
        report.get("schema_version") == 6
        and report.get("evidence_class") == GENERATION_EVIDENCE_CLASS
        and report.get("semantic") == GENERATION_SEMANTIC
        and report.get("revision") == REVISION
        and report.get("commit") == PW0328_CAPTURE_COMMIT
        and report.get("git_dirty") is False,
        f"{category}: generation identity",
    )
    _require(
        report.get("model_lock_sha256") == MODEL_LOCK_SHA256
        and report.get("checkpoint_verification_sha256") == CHECKPOINT_VERIFICATION_SHA256
        and report.get("tokenizer_sha256") == TOKENIZER_SHA256
        and report.get("tokenizer_config_sha256") == TOKENIZER_CONFIG_SHA256
        and report.get("kernel_sha256") == CAPTURE_KERNEL_SHA256,
        f"{category}: model/kernel identity",
    )
    _require(
        report.get("metal_device") == "Apple M1"
        and report.get("batch_size") == 1
        and report.get("concurrency") == 1
        and report.get("verifier_width") == WIDTH
        and report.get("proposer") == TARGET_SELF_PROPOSER
        and isinstance(report.get("cache_state"), str)
        and "cold process start" in report["cache_state"],
        f"{category}: generation context",
    )
    prompt_ids = _tokens(report.get("prompt_token_ids"), f"{category}: prompt tokens")
    generated_ids = _tokens(report.get("generated_token_ids"), f"{category}: generated tokens")
    _require(
        report.get("user_prompt_utf8") == prompt_path.read_text()
        and report.get("requested_output_tokens") == 64
        and report.get("minimum_output_tokens") == 32
        and report.get("accepted_tokens") == 64
        and len(generated_ids) == 64
        and report.get("stop_reason") == "requested_maximum"
        and report.get("route_trace_captured") is True,
        f"{category}: fixed-64 generation contract",
    )
    _require(
        report.get("progress_sha256") == source_entry.get("progress_sha256"),
        f"{category}: embedded progress hash",
    )
    progress = [
        _mapping(json.loads(line), f"{category}: progress row")
        for line in paths["generation_progress"].read_text().splitlines()
        if line.strip()
    ]
    transactions = _sequence(report.get("transactions"), f"{category}: transactions")
    _require(
        len(transactions) >= WIDTH
        and len(progress) == len(transactions) + 1
        and progress[0].get("phase") == "prefill_complete"
        and progress[0].get("prompt_tokens") == len(prompt_ids)
        and progress[0].get("prefill_chunks") == report.get("prefill_chunks")
        and math.isclose(
            float(progress[0].get("prefill_wall_ms")),
            float(report.get("prefill_wall_ms")),
            abs_tol=1.0e-9,
        )
        and progress[0].get("first_anchor_token_id") == generated_ids[0],
        f"{category}: generation progress structure",
    )
    capture = _mapping(report.get("native_mtp_window"), f"{category}: hidden capture")
    _require(
        capture.get("category") == category
        and capture.get("artifact_file") == paths["generation_hidden"].name
        and capture.get("artifact_sha256") == source_entry.get("hidden_sha256")
        and capture.get("windows") == len(transactions)
        and capture.get("shape") == [len(transactions), WIDTH, HIDDEN]
        and capture.get("dtype") == "float32"
        and capture.get("byte_order") == "little_endian"
        and capture.get("semantic") == VERIFIER_HIDDEN_SEMANTIC,
        f"{category}: verifier hidden metadata",
    )
    _finite_f32(
        paths["generation_hidden"],
        len(transactions) * WINDOW_BYTES,
        f"{category}: verifier hidden",
    )

    reconstructed_ids = [generated_ids[0]]
    derived_transactions: list[dict[str, Any]] = []
    for index, raw_transaction in enumerate(transactions):
        derived = _transaction(
            raw_transaction,
            index=index,
            terminal=index + 1 == len(transactions),
            progress=progress[index + 1],
            label=f"{category}: transaction {index}",
        )
        _require(
            derived["proposal_token_ids"][0] == reconstructed_ids[-1],
            f"{category}: transaction anchor chain",
        )
        reconstructed_ids.extend(derived["observable_emitted_token_ids"])
        _require(
            progress[index + 1].get("generated_tokens") == len(reconstructed_ids),
            f"{category}: progress generated-token closure",
        )
        derived_transactions.append(derived)
    _require(reconstructed_ids == generated_ids, f"{category}: generated-token reconstruction")
    report_logical = _integer(
        report.get("logical_source_bytes"), f"{category}: report logical bytes", minimum=1
    )
    report_physical = _integer(
        report.get("process_disk_bytes_read"), f"{category}: report physical bytes", minimum=1
    )
    _require(
        sum(item["transaction_logical_source_bytes"] for item in derived_transactions)
        <= report_logical
        and sum(item["transaction_process_disk_bytes_read"] for item in derived_transactions)
        <= report_physical,
        f"{category}: aggregate byte-ledger order",
    )
    _require(
        math.isclose(
            math.fsum(item["proposal_wall_ms"] for item in derived_transactions),
            float(report.get("proposal_wall_ms")),
            abs_tol=0.01,
        )
        and math.isclose(
            math.fsum(item["verification_wall_ms"] for item in derived_transactions),
            float(report.get("verification_wall_ms")),
            abs_tol=0.01,
        ),
        f"{category}: aggregate timing closure",
    )
    generation_gate = validate_gate8(report.get("safety_snapshots"), label=f"{category}: generation")
    complete_wall = _positive_number(
        report.get("complete_wall_ms"), f"{category}: complete wall"
    )
    _require(
        type(report.get("peak_resident_bytes")) is int
        and report["peak_resident_bytes"]
        == generation_gate["maximum_process_peak_resident_bytes"]
        and complete_wall
        >= math.fsum(
            _positive_number(report.get(field), f"{category}: {field}")
            for field in (
                "preprocessing_wall_ms",
                "prefill_wall_ms",
                "proposal_wall_ms",
                "verification_wall_ms",
            )
        ),
        f"{category}: generation peak/timing closure",
    )
    source_summary = {
        "category": category,
        "capture_commit": PW0328_CAPTURE_COMMIT,
        "prompt_file": prompt_relative,
        "prompt_sha256": prompt_sha,
        "report_file": str(paths["generation_report"]),
        "report_sha256": source_entry["report_sha256"],
        "progress_file": str(paths["generation_progress"]),
        "progress_sha256": source_entry["progress_sha256"],
        "hidden_file": str(paths["generation_hidden"]),
        "hidden_sha256": source_entry["hidden_sha256"],
        "captured_windows": len(transactions),
        "complete_wall_ms": complete_wall,
        "logical_source_bytes": report_logical,
        "process_disk_bytes_read": report_physical,
        "peak_resident_bytes": report["peak_resident_bytes"],
        "gate8": generation_gate,
    }
    _require(source_summary == source_entry, f"{category}: generation manifest summary")

    prefill = _json(paths["prefill_report"], f"{category}: prefill report")
    prefill_prompt_ids = _tokens(prefill.get("prompt_token_ids"), f"{category}: prefill prompt")
    _require(
        prefill.get("schema_version") == 1
        and prefill.get("evidence_class") == PREFILL_EVIDENCE_CLASS
        and prefill.get("semantic") == PREFILL_SEMANTIC
        and prefill.get("revision") == REVISION
        and prefill.get("commit") == PW0328_CAPTURE_COMMIT
        and prefill.get("git_dirty") is False,
        f"{category}: prefill identity",
    )
    _require(
        prefill.get("model_lock_sha256") == MODEL_LOCK_SHA256
        and prefill.get("checkpoint_verification_sha256") == CHECKPOINT_VERIFICATION_SHA256
        and prefill.get("tokenizer_sha256") == TOKENIZER_SHA256
        and prefill.get("tokenizer_config_sha256") == TOKENIZER_CONFIG_SHA256
        and prefill.get("kernel_sha256") == CAPTURE_KERNEL_SHA256,
        f"{category}: prefill model/kernel identity",
    )
    _require(
        prefill.get("user_prompt_utf8") == prompt_path.read_text()
        and prefill.get("serialized_prompt_utf8") == report.get("serialized_prompt_utf8")
        and prefill_prompt_ids == prompt_ids
        and prefill.get("first_anchor_token_id") == generated_ids[0]
        and prefill.get("first_anchor_token_id")
        == derived_transactions[0]["proposal_token_ids"][0]
        and prefill.get("prefill_chunks") == report.get("prefill_chunks"),
        f"{category}: prefill/generation agreement",
    )
    prefill_capture = _mapping(prefill.get("target_hidden"), f"{category}: prefill hidden")
    _require(
        prefill_capture.get("category") == category
        and prefill_capture.get("artifact_file") == paths["prefill_hidden"].name
        and prefill_capture.get("artifact_sha256") == prefill_entry.get("hidden_sha256")
        and prefill_capture.get("shape") == [len(prompt_ids), HIDDEN]
        and prefill_capture.get("dtype") == "float32"
        and prefill_capture.get("byte_order") == "little_endian"
        and prefill_capture.get("semantic") == PREFILL_HIDDEN_SEMANTIC,
        f"{category}: prefill hidden metadata",
    )
    _require(
        prefill.get("metal_device") == "Apple M1"
        and prefill.get("batch_size") == 1
        and prefill.get("concurrency") == 1
        and isinstance(prefill.get("cache_state"), str)
        and "cold process start" in prefill["cache_state"],
        f"{category}: prefill context",
    )
    chunk_traces = _sequence(prefill.get("chunk_layer_traces"), f"{category}: chunks")
    _require(
        len(chunk_traces) == prefill.get("prefill_chunks")
        and all(isinstance(chunk, list) and len(chunk) == LAYERS for chunk in chunk_traces),
        f"{category}: prefill trace shape",
    )
    _finite_f32(
        paths["prefill_hidden"], len(prompt_ids) * HIDDEN_ROW_BYTES, f"{category}: prefill hidden"
    )
    prefill_gate = validate_gate8(prefill.get("safety_snapshots"), label=f"{category}: prefill")
    prefill_logical = _integer(
        _mapping(prefill.get("ledger"), f"{category}: prefill ledger").get("logical_source_bytes"),
        f"{category}: prefill logical bytes",
        minimum=1,
    )
    prefill_physical = _integer(
        prefill.get("process_disk_bytes_read"), f"{category}: prefill physical bytes", minimum=1
    )
    _require(
        type(prefill.get("peak_resident_bytes")) is int
        and prefill["peak_resident_bytes"] == prefill_gate["maximum_process_peak_resident_bytes"]
        and _positive_number(prefill.get("complete_wall_ms"), f"{category}: prefill complete")
        >= _positive_number(prefill.get("prefill_wall_ms"), f"{category}: prefill wall"),
        f"{category}: prefill ledger/timing",
    )
    prefill_summary = {
        "category": category,
        "capture_commit": PW0328_CAPTURE_COMMIT,
        "report_file": str(paths["prefill_report"]),
        "report_sha256": prefill_entry["report_sha256"],
        "hidden_file": str(paths["prefill_hidden"]),
        "hidden_sha256": prefill_entry["hidden_sha256"],
        "hidden_rows": len(prompt_ids),
        "first_anchor_token_id": prefill["first_anchor_token_id"],
        "logical_source_bytes": prefill_logical,
        "process_disk_bytes_read": prefill_physical,
        "complete_wall_ms": prefill["complete_wall_ms"],
        "peak_resident_bytes": prefill["peak_resident_bytes"],
        "gate8": prefill_gate,
    }
    _require(prefill_summary == prefill_entry, f"{category}: prefill manifest summary")

    windows: list[dict[str, Any]] = []
    for transaction_index in range(WIDTH):
        derived = derived_transactions[transaction_index]
        if transaction_index == 0:
            hidden_source = {
                "target_hidden_source": "prefill",
                "target_hidden_source_transaction_index": None,
                "target_hidden_source_row": len(prompt_ids) - 1,
                "target_hidden_file": str(paths["prefill_hidden"]),
                "target_hidden_byte_offset": (len(prompt_ids) - 1) * HIDDEN_ROW_BYTES,
                "target_hidden_byte_length": HIDDEN_ROW_BYTES,
            }
        else:
            previous_retained = derived_transactions[transaction_index - 1]["retained_proposal_rows"]
            row = previous_retained - 1
            _require(0 <= row < WIDTH, f"{category}: preceding hidden row")
            hidden_source = {
                "target_hidden_source": "verifier_transaction",
                "target_hidden_source_transaction_index": transaction_index - 1,
                "target_hidden_source_row": row,
                "target_hidden_file": str(paths["generation_hidden"]),
                "target_hidden_byte_offset": (transaction_index - 1) * WINDOW_BYTES
                + row * HIDDEN_ROW_BYTES,
                "target_hidden_byte_length": HIDDEN_ROW_BYTES,
            }
        generated_history = [generated_ids[0]]
        segments = [
            {
                "source": "prefill",
                "file": str(paths["prefill_hidden"]),
                "byte_offset": 0,
                "byte_length": len(prompt_ids) * HIDDEN_ROW_BYTES,
                "rows": len(prompt_ids),
            }
        ]
        for previous_index in range(transaction_index):
            previous = derived_transactions[previous_index]
            retained = previous["retained_proposal_rows"]
            segments.append(
                {
                    "source": "verifier_transaction",
                    "transaction_index": previous_index,
                    "file": str(paths["generation_hidden"]),
                    "byte_offset": previous_index * WINDOW_BYTES,
                    "byte_length": retained * HIDDEN_ROW_BYTES,
                    "rows": retained,
                }
            )
            generated_history.extend(previous["observable_emitted_token_ids"])
        anchor = derived["proposal_token_ids"][0]
        _require(generated_history[-1] == anchor, f"{category}: MTP history anchor")
        target_input_ids = [*prompt_ids, *generated_history[:-1]]
        _require(
            sum(segment["rows"] for segment in segments) == len(target_input_ids),
            f"{category}: target history length",
        )
        manifest_window = {
            "corpus_index": first_corpus_index + transaction_index,
            "category": category,
            **hidden_source,
            "target_input_token_ids": target_input_ids,
            "target_hidden_rows": len(target_input_ids),
            "target_hidden_segments": segments,
            "mtp_layer0_input_token_ids": [*target_input_ids[1:], anchor],
            "transaction_index": transaction_index,
            "proposal_token_ids": derived["proposal_token_ids"],
            "posterior_token_ids": derived["posterior_token_ids"],
            "verifier_authorized_token_ids": derived["verifier_authorized_token_ids"],
            "observable_emitted_token_ids": derived["observable_emitted_token_ids"],
            "proposal_converged": derived["proposal_converged"],
            "next_anchor_token_id": derived["next_anchor_token_id"],
            "A": derived["A"],
            "observable_A": derived["observable_A"],
            "verifier_retained_proposal_rows": derived["verifier_retained_proposal_rows"],
            "retained_proposal_rows": derived["retained_proposal_rows"],
            "U": derived["U"],
            "A_per_U": derived["A"] / derived["U"],
            "route": derived["route"],
            "transaction_logical_source_bytes": derived["transaction_logical_source_bytes"],
            "transaction_process_disk_bytes_read": derived[
                "transaction_process_disk_bytes_read"
            ],
            "report_logical_source_bytes": report_logical,
            "report_process_disk_bytes_read": report_physical,
            "proposal_wall_ms": derived["proposal_wall_ms"],
            "verification_wall_ms": derived["verification_wall_ms"],
        }
        all_rows = derived["route_views"]["all_q8_rows"]
        authorized_rows = authorized_q1_rows(all_rows, derived["A"])
        windows.append(
            {
                "manifest": manifest_window,
                **manifest_window,
                "all_q8_rows": all_rows,
                "per_layer_q8": derived["route_views"]["per_layer_q8"],
                "authorized_q1_rows": authorized_rows,
            }
        )
    return {
        "source": source_summary,
        "prefill": prefill_summary,
        "windows": windows,
        "artifacts": artifacts,
    }


def _control(windows: list[dict[str, Any]]) -> dict[str, Any]:
    category_metrics: dict[str, Any] = {}
    for category in CATEGORIES:
        selected = [window for window in windows if window["category"] == category]
        _require(len(selected) == WIDTH, f"{category}: primary window cardinality")
        total_a = sum(window["A"] for window in selected)
        total_u = math.fsum(window["U"] for window in selected)
        category_metrics[category] = {
            "windows": WIDTH,
            "sum_A": total_a,
            "sum_observable_A": sum(window["observable_A"] for window in selected),
            "sum_U": total_u,
            "sum_A_per_sum_U": total_a / total_u,
            "unique_identities": len(
                {
                    (identity["layer"], identity["expert"])
                    for window in selected
                    for identity in window["route"]["identities"]
                }
            ),
        }
    total_a = sum(window["A"] for window in windows)
    total_u = math.fsum(window["U"] for window in windows)
    return {
        "windows": len(windows),
        "sum_A": total_a,
        "sum_observable_A": sum(window["observable_A"] for window in windows),
        "sum_U": total_u,
        "sum_A_per_sum_U": total_a / total_u,
        "category_metrics": category_metrics,
    }


def _rare_route_evidence(windows: list[dict[str, Any]]) -> dict[str, Any]:
    control: Counter[tuple[int, int]] = Counter()
    rare: Counter[tuple[int, int]] = Counter()
    for window in windows:
        destination = rare if window["category"] == "rare_route" else control
        destination.update(
            (identity["layer"], identity["expert"])
            for identity in window["route"]["identities"]
        )
    novel = sorted(set(rare) - set(control))
    return {
        "control_categories": [category for category in CATEGORIES if category != "rare_route"],
        "control_unique_identities": len(control),
        "rare_route_unique_identities": len(rare),
        "novel_identities": [
            {"layer": layer, "expert": expert} for layer, expert in novel
        ],
        "novel_identity_count": len(novel),
        "novel_routed_layers": sorted({layer for layer, _expert in novel}),
    }


def authenticate_pw0328_corpus(
    repo: Path,
    manifest_path: Path = PW0328_MANIFEST_PATH,
) -> dict[str, Any]:
    """Authenticate and independently replay the canonical PW-0328 corpus."""

    repo = repo.resolve()
    manifest_path = manifest_path.resolve()
    manifest_hash = sha256_file(manifest_path)
    _require(manifest_hash == PW0328_MANIFEST_SHA256, "PW-0328 manifest SHA-256 mismatch")
    manifest = _json(manifest_path, "PW-0328 manifest")
    _require(set(manifest) == _MANIFEST_KEYS, "PW-0328 manifest schema drift")
    _require(
        manifest.get("schema_version") == 1
        and manifest.get("experiment_id") == "PW-0328"
        and manifest.get("evidence_class") == MANIFEST_EVIDENCE_CLASS
        and manifest.get("semantic") == MANIFEST_SEMANTIC
        and manifest.get("status") == "complete"
        and manifest.get("builder_commit") == PW0328_CAPTURE_COMMIT
        and manifest.get("builder_git_dirty") is False
        and manifest.get("verifier_window_shape") == [WIDTH, HIDDEN]
        and manifest.get("hidden_dtype") == "float32_little_endian"
        and manifest.get("source_expert_bytes") == SOURCE_EXPERT_BYTES
        and manifest.get("batch_size") == 1
        and manifest.get("concurrency") == 1
        and manifest.get("accepted_tokens") == 0
        and manifest.get("performance_claim") is None,
        "PW-0328 manifest identity",
    )
    sources = _sequence(manifest.get("sources"), "PW-0328 generation sources")
    prefill_sources = _sequence(manifest.get("prefill_sources"), "PW-0328 prefill sources")
    _require(
        len(sources) == len(prefill_sources) == len(CATEGORIES),
        "PW-0328 source cardinality",
    )
    reconstructed_sources = []
    reconstructed_prefills = []
    windows: list[dict[str, Any]] = []
    artifacts: list[dict[str, Any]] = []
    for category_index, category in enumerate(CATEGORIES):
        result = _category(
            category=category,
            source_entry=sources[category_index],
            prefill_entry=prefill_sources[category_index],
            manifest_path=manifest_path,
            repo=repo,
            first_corpus_index=category_index * WIDTH,
        )
        reconstructed_sources.append(result["source"])
        reconstructed_prefills.append(result["prefill"])
        windows.extend(result["windows"])
        artifacts.extend(result["artifacts"])
    _require(reconstructed_sources == sources, "PW-0328 generation source order")
    _require(reconstructed_prefills == prefill_sources, "PW-0328 prefill source order")
    manifest_windows = _sequence(manifest.get("primary_windows"), "PW-0328 primary windows")
    validate_primary_window_sequence(manifest_windows)
    validate_primary_window_sequence([window["manifest"] for window in windows])
    for index, (expected, reconstructed) in enumerate(zip(manifest_windows, windows, strict=True)):
        validate_manifest_window(expected, reconstructed["manifest"], index=index)
        _require(
            len(reconstructed["authorized_q1_rows"]) == reconstructed["A"],
            f"PW-0328 window {index} q1 full-A authority",
        )
    control = _control(windows)
    _require(control == manifest.get("control"), "PW-0328 control summary replay")
    rare = _rare_route_evidence(windows)
    _require(rare == manifest.get("rare_route_evidence"), "PW-0328 rare-route replay")
    _require(
        control["sum_A"] == 232
        and control["sum_observable_A"] == 231
        and math.isclose(control["sum_U"], 142.71808510638297, abs_tol=1.0e-12),
        "PW-0328 frozen corpus totals",
    )
    _require(
        len(artifacts) == 24
        and Counter((item["category"], item["kind"]) for item in artifacts)
        == Counter((category, kind) for category in CATEGORIES for kind in (
            "generation_report",
            "generation_progress",
            "generation_hidden",
            "prefill_report",
            "prefill_hidden",
            "prompt",
        )),
        "PW-0328 24-artifact closure",
    )
    q1_events: list[dict[str, Any]] = []
    for window in windows:
        for row in window["authorized_q1_rows"]:
            q1_events.append(
                {
                    "event_index": len(q1_events),
                    "category": window["category"],
                    "corpus_index": window["corpus_index"],
                    "transaction_index": window["transaction_index"],
                    "authorized_token_id": window[
                        "verifier_authorized_token_ids"
                    ][row["position"]],
                    **row,
                }
            )
    _require(len(q1_events) == control["sum_A"] == 232, "PW-0328 q1 event cardinality")
    builder_gate = validate_gate8(
        manifest.get("builder_safety_snapshots"), label="PW-0328 builder"
    )
    return {
        "manifest_file": str(manifest_path),
        "manifest_sha256": manifest_hash,
        "builder_commit": PW0328_CAPTURE_COMMIT,
        "identities": {
            "revision": REVISION,
            "model_lock_sha256": MODEL_LOCK_SHA256,
            "checkpoint_verification_sha256": CHECKPOINT_VERIFICATION_SHA256,
            "tokenizer_sha256": TOKENIZER_SHA256,
            "tokenizer_config_sha256": TOKENIZER_CONFIG_SHA256,
            "capture_kernel_sha256": CAPTURE_KERNEL_SHA256,
        },
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
        "categories": list(CATEGORIES),
        "sources": reconstructed_sources,
        "prefill_sources": reconstructed_prefills,
        "windows": windows,
        "q1_events": q1_events,
        "control": control,
        "rare_route_evidence": rare,
        "builder_gate8": builder_gate,
    }
