#!/usr/bin/env python3
"""Recompute the PW-0324 onboard Prismwing-2 feasibility closure."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import struct
from typing import Any

try:
    from tools.analyze_fp8_symbol_census import greedy_codebook, relative_l2
    from tools.analyze_pw0319_corrected_route_bank import greedy_order, load_rows, sha256_file
    from tools.analyze_pw0320_hybrid_byte_floor import (
        K4_BYTES,
        SOURCE_BYTES,
        STORAGE_BYTES_PER_SECOND,
        oracle_cached_bytes,
    )
    from tools.analyze_pw0322_causal_q64 import route_union
    from tools.host_safety import HostSafetyMonitor, HostSafetyViolation
    from tools.openrouter_reference import atomic_write_new, canonical_json
    from tools.remote_fp8_symbol_census import analyze_row_tile
    from tools.reproduce_pw0311_k4_expert import verify_clean_commit
except ModuleNotFoundError:
    from analyze_fp8_symbol_census import greedy_codebook, relative_l2
    from analyze_pw0319_corrected_route_bank import greedy_order, load_rows, sha256_file
    from analyze_pw0320_hybrid_byte_floor import K4_BYTES, SOURCE_BYTES, STORAGE_BYTES_PER_SECOND, oracle_cached_bytes
    from analyze_pw0322_causal_q64 import route_union
    from host_safety import HostSafetyMonitor, HostSafetyViolation
    from openrouter_reference import atomic_write_new, canonical_json
    from remote_fp8_symbol_census import analyze_row_tile
    from reproduce_pw0311_k4_expert import verify_clean_commit


EXPERIMENT_ID = "PW-0324"
CHECKPOINT_REVISION = "63651580ca774f8504f676040460aed3e1244ac1"
CHECKPOINT_RECEIPT_SHA256 = "9ddc8a99755f04ae2ea3c2484f6dd022d3f3a681b5a72c915ee4de833dbb0d03"
CHECKPOINT_INDEX_SHA256 = "f2e1774c9acf9a62338b68c144e6fc7a66495e59f2e64b3078c1b7ef5a196816"
PW0322_REPORT_SHA256 = "ef893b83105009576771b4dcbd98b4f82320b838e58920f5c32011e9a52acb60"
PW0322_ANALYSIS_SHA256 = "8c824040776c5ca2b9d9f0854d9066c00f7b5495296c86e113729e6e07a6b98d"
PW0320_ANALYSIS_SHA256 = "de6424aa68d0c65f8f9206a53f61475286bde501873cd4f6ee06299c9b37d7a9"
PW0208_CORPUS_SHA256 = "a9bb6bd26bf048a2144133cc0a96023a8af112eae58122b666915149f2993a7b"
PW0208_UPPER_SHA256 = "3aaca59be0e000cac77d5a36b8e3b9d2e2fc5bbb02792c8846dae3da16747f8c"
PW0300_REMOTE_HASHES = {
    "train_census": "abc2ef97a15a286c8ddd71dac0773c5d43eb0e308b8791b9d87b4fee74835c37",
    "holdout_census": "4229b4f893cba004035956f89b92c390502f7464c4c0c3be29799395ad8727a6",
    "analysis": "dd603e2fbd03cdf6600d7a81f716becd2ea809eeaf3b0b46d53a71f613258552",
}
TWO_TPS = 2.0
PANEL = ((4, 96), (24, 22), (46, 28))
PROJECTIONS = ("gate_proj", "up_proj", "down_proj")

# These records cover every surviving onboard mechanism class named in the
# predeclared contract. Their Git content hashes are emitted for auditability.
PORTFOLIO: dict[str, tuple[str, str]] = {
    "source_cache_and_residency": ("experiments/PW-0207-pressure-elastic-resident-working-set.md", "conditional_below_two_tps"),
    "predictive_prefetch": ("experiments/PW-0212-corrected-route-predictive-prefetch-oracle.md", "rejected"),
    "source_stream_transport": ("experiments/PW-0213-uncached-page-aligned-stream-transport.md", "rejected_runtime"),
    "native_mtp": ("experiments/PW-0208-native-mtp-cost-aware-proposer.md", "rejected_cost_gate_lower_latency_retained"),
    "published_bounded_proposers": ("experiments/PW-0173-current-speculator-horizon-audit.md", "rejected_direct"),
    "approximate_mismatch_acceptance": ("experiments/PW-0174-approximate-speculative-suffix-reuse-audit.md", "rejected_direct_and_fidelity_unqualified"),
    "causal_q64": ("experiments/PW-0322-causal-width64-route-capture.md", "rejected_actual_acceptance"),
    "exact_fp8_local_codecs": ("experiments/PW-0300-fp8-symbol-census.md", "rejected_exact_forms"),
    "six_bit_fp8_subset": ("experiments/PW-0301-full-expert-fp8-subset-control.md", "rejected_fidelity"),
    "affine_int4": ("experiments/PW-0129-real-activation-affine-int4-layer-audit.md", "rejected_fidelity"),
    "global_hessian_int4": ("experiments/PW-0139-all-validation-expert-global-hessian-audit.md", "rejected_generalization"),
    "affine_six_bit": ("experiments/PW-0148-six-bit-global-hessian-three-expert-control.md", "rejected_fidelity"),
    "vector_code": ("experiments/PW-0177-coreml-scaled-vector-expert-transaction.md", "rejected_fidelity_and_loading"),
    "subvector_code": ("experiments/PW-0178-input-subvector-code-capacity-oracle.md", "rejected_fidelity"),
    "microscaling_fp4": ("experiments/PW-0182-microscaling-fp4-real-expert.md", "rejected_fidelity"),
    "activation_sparsity": ("experiments/PW-0184-weight-aware-activation-sparsity.md", "rejected_fidelity"),
    "shared_basis": ("experiments/PW-0124-coverage-rebalanced-sharing-control.md", "rejected_generalization"),
    "routed_mixture_compiler": ("experiments/PW-0126-routed-residual-subspace-oracle.md", "rejected_linear_form"),
    "exception_store": ("experiments/PW-0046-expert-bank-exception-store.md", "blocked_on_failed_compiler_prerequisite"),
    "k4_hybrid_bank": ("experiments/PW-0320-corrected-width8-hybrid-byte-floor.md", "rejected_q8_and_q64"),
    "fixed_subset_sparse_attention": ("experiments/PW-0162-global-attention-top20-oracle.md", "rejected_fidelity"),
    "structured_sparse_attention": ("experiments/PW-0176-mimo-64k-structured-sparse-oracle.md", "rejected_fidelity"),
}


def _read_json(path: Path, expected_sha256: str) -> dict[str, Any]:
    if sha256_file(path) != expected_sha256:
        raise ValueError(f"evidence hash mismatch: {path}")
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"evidence root is not an object: {path}")
    return value


def required_uniform_reduction(bytes_moved: int, accepted: int, bandwidth: float, target_tps: float = TWO_TPS) -> dict[str, float | int]:
    if bytes_moved <= 0 or accepted <= 0 or bandwidth <= 0 or target_tps <= 0:
        raise ValueError("invalid reduction inputs")
    allowed = accepted * bandwidth / target_tps
    return {
        "current_bytes": bytes_moved,
        "allowed_bytes": allowed,
        "additional_reduction_factor": bytes_moved / allowed,
        "maximum_remaining_fraction": allowed / bytes_moved,
        "required_reduction_fraction": 1.0 - allowed / bytes_moved,
    }


def _validate_receipt(checkpoint: Path, receipt_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    receipt = _read_json(receipt_path, CHECKPOINT_RECEIPT_SHA256)
    if (
        receipt.get("schema_version") != 1
        or receipt.get("evidence_class") != "local_checkpoint_lock_verification"
        or receipt.get("complete") is not True
        or receipt.get("missing_files") != []
        or receipt.get("revision") != CHECKPOINT_REVISION
    ):
        raise ValueError("checkpoint receipt contract mismatch")
    index_path = checkpoint / "model.safetensors.index.json"
    index = _read_json(index_path, CHECKPOINT_INDEX_SHA256)
    return receipt, index


def _tensor_metadata(shard: Path, name: str) -> tuple[int, dict[str, Any]]:
    with shard.open("rb") as handle:
        header_length = struct.unpack("<Q", handle.read(8))[0]
        if header_length <= 0 or header_length > 32 * 1024 * 1024:
            raise ValueError(f"invalid safetensors header: {shard}")
        header = json.loads(handle.read(header_length))
    metadata = header.get(name)
    if not isinstance(metadata, dict):
        raise ValueError(f"tensor missing from shard: {name}")
    return header_length, metadata


def replicate_pw0300_panel(checkpoint: Path, receipt_path: Path) -> dict[str, Any]:
    receipt, index = _validate_receipt(checkpoint, receipt_path)
    receipt_by_path = {row["path"]: row for row in receipt["files"]}
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        raise ValueError("checkpoint index lacks weight map")
    blocks: list[dict[str, Any]] = []
    tiles: list[dict[str, Any]] = []
    observed_shards: dict[str, dict[str, Any]] = {}
    for layer, expert in PANEL:
        for projection in PROJECTIONS:
            name = f"model.layers.{layer}.mlp.experts.{expert}.{projection}.weight"
            relative = weight_map.get(name)
            if not isinstance(relative, str):
                raise ValueError(f"tensor missing from index: {name}")
            shard = checkpoint / relative
            row = receipt_by_path.get(relative)
            stat = shard.stat()
            if (
                not isinstance(row, dict)
                or row.get("status") != "verified"
                or row.get("bytes") != stat.st_size
                or row.get("inode") != stat.st_ino
                or row.get("modified_ns") != stat.st_mtime_ns
            ):
                raise ValueError(f"checkpoint shard identity mismatch: {relative}")
            observed_shards[relative] = {"bytes": stat.st_size, "sha256": row["sha256"]}
            header_length, metadata = _tensor_metadata(shard, name)
            shape = metadata.get("shape")
            offsets = metadata.get("data_offsets")
            if metadata.get("dtype") != "F8_E4M3" or not isinstance(shape, list) or len(shape) != 2 or not isinstance(offsets, list) or len(offsets) != 2:
                raise ValueError(f"unsupported tensor layout: {name}")
            rows, columns = map(int, shape)
            if rows % 128 or columns % 128:
                raise ValueError(f"unaligned tensor shape: {name}")
            for row_block in (0, rows // 128 - 1):
                length = 128 * columns
                start = 8 + header_length + int(offsets[0]) + row_block * length
                with shard.open("rb") as handle:
                    handle.seek(start)
                    payload = handle.read(length)
                if len(payload) != length:
                    raise ValueError(f"short tensor read: {name}")
                tile_blocks = analyze_row_tile(payload, columns)
                blocks.extend(tile_blocks)
                tiles.append({
                    "tensor": name,
                    "shard": relative,
                    "shape": shape,
                    "row_block": row_block,
                    "bytes": length,
                    "sha256": hashlib.sha256(payload).hexdigest(),
                    "quantization_blocks": len(tile_blocks),
                })
    entropies = [float(block["entropy_bits_per_weight"]) for block in blocks]
    escape = [float(block["exponent_top7_escape_bytes"]) / (128 * 128) for block in blocks]
    counts = [sum(block["symbol_counts"][code] for block in blocks) for code in range(256)]
    codebook = greedy_codebook(counts, 64)
    errors = [relative_l2(block["symbol_counts"], codebook) for block in blocks]
    source_tile_bytes = [int(tile["bytes"]) for tile in tiles]
    six_bit_ratios = [((size * 6 + 7) // 8 + 64) / size for size in source_tile_bytes]
    return {
        "evidence_class": "pw0324_local_replication_of_pw0300_fp8_symbol_mechanisms",
        "checkpoint_revision": CHECKPOINT_REVISION,
        "checkpoint_receipt_sha256": CHECKPOINT_RECEIPT_SHA256,
        "checkpoint_index_sha256": CHECKPOINT_INDEX_SHA256,
        "original_remote_hashes_unavailable_locally": PW0300_REMOTE_HASHES,
        "panel": [{"layer": layer, "expert": expert} for layer, expert in PANEL],
        "tiles": tiles,
        "observed_shards": observed_shards,
        "quantization_blocks": len(blocks),
        "distinct_symbol_range": [min(block["distinct_symbols"] for block in blocks), max(block["distinct_symbols"] for block in blocks)],
        "exact_palette_6bit_blocks": sum(block["exact_palette_bytes"]["6"] is not None for block in blocks),
        "exact_palette_7bit_blocks": sum(block["exact_palette_bytes"]["7"] is not None for block in blocks),
        "entropy_bits_per_weight": {"minimum": min(entropies), "median": statistics.median(entropies), "maximum": max(entropies)},
        "idealized_top7_exponent_escape_ratio": {"minimum": min(escape), "median": statistics.median(escape), "maximum": max(escape)},
        "six_bit_subset_physical_ratio": {"minimum": min(six_bit_ratios), "maximum": max(six_bit_ratios)},
        "global_six_bit_subset_weight_relative_l2": {"minimum": min(errors), "median": statistics.median(errors), "maximum": max(errors)},
        "limitation": "Independent deterministic local panel; it authenticates the PW-0300 mechanisms, not the unavailable original remote JSON payloads or routed-output fidelity.",
    }


def _pw0320_strongest(analysis: dict[str, Any]) -> dict[str, Any]:
    matches = [curve for curve in analysis.get("curves", []) if curve.get("budget") == 2048 and curve.get("oracle_cache_bytes") == 4 * 1024**3]
    if len(matches) != 1 or matches[0].get("continuation_gate_pass") is not False:
        raise ValueError("PW-0320 strongest curve mismatch")
    curve = matches[0]
    windows = curve.get("windows")
    if not isinstance(windows, list) or len(windows) != 32:
        raise ValueError("PW-0320 window contract mismatch")
    return {
        "observed_maximum_optimistic_tps": max(float(row["optimistic_accepted_tps"]) for row in windows),
        "structural_a8_maximum_optimistic_tps": max(8 * STORAGE_BYTES_PER_SECOND / int(row["bytes_after_oracle_cache"]) for row in windows),
        "median_optimistic_tps": float(curve["median_optimistic_tps"]),
        "passing_windows": int(curve["passing_windows"]),
    }


def _strongest_corrected_complete_tps(model: dict[str, Any]) -> dict[str, Any]:
    constants = model.get("constants")
    if not isinstance(constants, dict):
        raise ValueError("throughput model constants missing")
    eligible = []
    for name, row in constants.items():
        if not name.startswith(("pw0211_", "pw0215_", "pw0216_")) or not isinstance(row, dict):
            continue
        value = row.get("candidate_complete_accepted_tps_median")
        if value is None:
            continue
        if row.get("batch_size") != 1 or row.get("concurrency") != 1 or row.get("accepted_tokens_per_complete_request", 0) <= 0:
            raise ValueError(f"incomplete accepted-TPS provenance: {name}")
        eligible.append((float(value), name, row))
    if not eligible:
        raise ValueError("no corrected complete accepted-TPS constants")
    value, name, row = max(eligible)
    return {
        "constant": name,
        "accepted_tps": value,
        "accepted_tokens": int(row["accepted_tokens_per_complete_request"]),
        "batch_size": 1,
        "concurrency": 1,
        "status": row["status"],
        "provenance": row["provenance"],
    }


def _portfolio(repo: Path) -> dict[str, Any]:
    result = {}
    for mechanism, (relative, state) in PORTFOLIO.items():
        path = repo / relative
        if not path.is_file():
            raise ValueError(f"portfolio record missing: {relative}")
        result[mechanism] = {"record": relative, "record_sha256": sha256_file(path), "state": state, "survives_two_tps_closure": False}
    return result


def analyze(*, repo: Path, commit: str, checkpoint: Path, checkpoint_receipt: Path, pw0322_report: Path, pw0322_analysis: Path, pw0320_analysis: Path, pw0208_corpus: Path, pw0208_upper: Path, throughput_model: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(output)
    verify_clean_commit(repo.resolve(), commit)
    report = _read_json(pw0322_report, PW0322_REPORT_SHA256)
    prior_q64 = _read_json(pw0322_analysis, PW0322_ANALYSIS_SHA256)
    width8 = _read_json(pw0320_analysis, PW0320_ANALYSIS_SHA256)
    _read_json(pw0208_corpus, PW0208_CORPUS_SHA256)
    mtp = _read_json(pw0208_upper, PW0208_UPPER_SHA256)
    model = json.loads(throughput_model.read_text())
    if report.get("verifier_width") != 64 or len(report.get("transactions", [])) != 1:
        raise ValueError("PW-0322 raw report contract mismatch")
    transaction = report["transactions"][0]
    actual_a = len(transaction.get("verifier_authorized_token_ids", []))
    identities = route_union(transaction)
    rows, route_sha256, _ = load_rows(pw0208_corpus)
    selected = set(greedy_order(rows, maximum_budget=2048))
    sizes = [K4_BYTES if identity in selected else SOURCE_BYTES for identity in identities]
    moved = oracle_cached_bytes(sizes, 4 * 1024**3)
    q64_tps = actual_a * STORAGE_BYTES_PER_SECOND / moved
    structural_tps = 64 * STORAGE_BYTES_PER_SECOND / moved
    if (
        actual_a != 3
        or len(identities) != 4482
        or route_sha256 != prior_q64["authority"]["corrected_route_sha256"]
        or moved != prior_q64["curves"][-1]["bytes_after_oracle_cache"]
        or not math.isclose(q64_tps, prior_q64["curves"][-1]["actual_a_optimistic_tps"], rel_tol=0, abs_tol=1e-15)
    ):
        raise ValueError("PW-0322 recomputation mismatch")
    q8 = _pw0320_strongest(width8)
    mtp_gain = float(mtp.get("maximum_possible_gain", math.inf))
    if mtp.get("gate_passed") is not False or mtp_gain >= float(mtp.get("required_gain", 0)):
        raise ValueError("PW-0208 upper bound mismatch")
    local_fp8 = replicate_pw0300_panel(checkpoint, checkpoint_receipt)
    if local_fp8["exact_palette_7bit_blocks"] != 0 or local_fp8["idealized_top7_exponent_escape_ratio"]["minimum"] <= 0.75:
        raise ValueError("local PW-0300 replication does not support exact-codec rejection")
    strongest_complete = _strongest_corrected_complete_tps(model)
    reduction = required_uniform_reduction(moved, actual_a, STORAGE_BYTES_PER_SECOND)
    portfolio = _portfolio(repo)
    closure_conditions = {
        "no_measured_corrected_complete_result_reaches_two_tps": strongest_complete["accepted_tps"] < TWO_TPS,
        "q8_structural_storage_ceiling_below_two_tps": q8["structural_a8_maximum_optimistic_tps"] < TWO_TPS,
        "q64_actual_storage_ceiling_below_two_tps": q64_tps < TWO_TPS,
        "q64_requires_less_than_six_percent_of_granted_bytes": reduction["maximum_remaining_fraction"] < 0.06,
        "exact_local_codec_replication_fails_25_percent_gate": local_fp8["idealized_top7_exponent_escape_ratio"]["minimum"] > 0.75 and local_fp8["exact_palette_7bit_blocks"] == 0,
        "native_mtp_cost_gate_fails_even_with_perfect_proposals": mtp_gain < float(mtp["required_gain"]),
        "portfolio_has_no_evidence_backed_survivor": not any(row["survives_two_tps_closure"] for row in portfolio.values()),
        "companion_hardware_excluded": True,
        "no_fast_branch_has_full_capability_and_fidelity_promotion": True,
    }
    frontier_open = [name for name, passed in closure_conditions.items() if not passed]
    safety = HostSafetyMonitor()
    safety.checkpoint("analysis_complete")
    safety.release_checkpoint("analysis_released", ["PW-0322 route union", "PW-0300 local sample tiles", "portfolio records"])
    safety.checkpoint("final_service_health")
    result = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "status": "complete" if not frontier_open else "incomplete",
        "decision": "close_current_onboard_prismwing2_frontier" if not frontier_open else "frontier_open",
        "analysis_commit": commit,
        "scope": {"hardware": "existing 16 GiB Apple M1", "companion_hardware_admissible": False, "correctness_contract_changed": False, "target_50_changed": False, "intermediate_goal": "strictly greater than 2 sustained accepted TPS after prefill"},
        "authorities": {
            "pw0322_report_sha256": PW0322_REPORT_SHA256,
            "pw0322_analysis_sha256": PW0322_ANALYSIS_SHA256,
            "pw0320_analysis_sha256": PW0320_ANALYSIS_SHA256,
            "pw0208_corpus_sha256": PW0208_CORPUS_SHA256,
            "pw0208_upper_bound_sha256": PW0208_UPPER_SHA256,
            "throughput_model_sha256": sha256_file(throughput_model),
            "pw0300_original_remote_hashes_unavailable_locally": PW0300_REMOTE_HASHES,
        },
        "measured_complete_path": strongest_complete,
        "q8_storage_only_upper_bound": q8,
        "q64_storage_only_upper_bound": {"verifier_width": 64, "actual_a": actual_a, "unique_identities": len(identities), "unique_k4_identities": sum(identity in selected for identity in identities), "unique_source_identities": sum(identity not in selected for identity in identities), "perfect_free_cache_bytes": 4 * 1024**3, "bytes_after_cache": moved, "actual_a_optimistic_tps": q64_tps, "structural_a64_optimistic_tps": structural_tps, "proposal_is_independently_qualified": False, "complete_proposal_cost_included": False, "timing_class": "optimistic_storage_only_not_achieved_tps"},
        "q64_required_record_reduction": reduction,
        "native_mtp_upper_bound": {"maximum_possible_gain": mtp_gain, "required_gain": float(mtp["required_gain"]), "gate_passed": False},
        "local_fp8_codec_replication": local_fp8,
        "portfolio": portfolio,
        "closure_conditions": closure_conditions,
        "failed_closure_conditions": frontier_open,
        "limitations": [
            "This is decisive closure of the current evidence-backed onboard architecture portfolio, not a theorem against unknown future algorithms.",
            "The q8/q64 values omit compute and common weights and are upper bounds, not measured endpoint TPS.",
            "The structural q64 A=64 value lacks an independently qualified proposer and omits proposal cost.",
            "No fast candidate has passed the repository's native-modality, long-context, hosted-distribution, and capability gates.",
            "PW-0300's original remote JSON was unavailable; a deterministic local checkpoint panel independently re-runs its exact palette, entropy, escape, and six-bit mechanisms.",
        ],
        "analysis_safety": safety.evidence(),
        "accepted_tokens": 0,
        "performance_claim": None,
        "runtime_default_changed": False,
    }
    output.mkdir(parents=True)
    path = output / "analysis.json"
    atomic_write_new(path, canonical_json(result))
    print(json.dumps({"output": str(path), "decision": result["decision"]}))
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    for name in ("repo", "checkpoint", "checkpoint_receipt", "pw0322_report", "pw0322_analysis", "pw0320_analysis", "pw0208_corpus", "pw0208_upper", "throughput_model", "output"):
        parser.add_argument("--" + name.replace("_", "-"), required=True, type=Path)
    parser.add_argument("--commit", required=True)
    try:
        analyze(**vars(parser.parse_args()))
        return 0
    except (FileExistsError, HostSafetyViolation, KeyError, OSError, RuntimeError, TypeError, ValueError, json.JSONDecodeError) as error:
        print(json.dumps({"error": str(error)}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
