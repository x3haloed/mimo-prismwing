#!/usr/bin/env python3
"""Run PW-0330's frozen cyclic-MTP q32 causal-prefix falsifier."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any, Callable

import numpy as np
import torch

try:
    from tools.analyze_pread_expert_acquisition import analyze as analyze_pw0136
    from tools.analyze_pressure_elastic_residency import (
        fixed_tensor_names,
        resident_allocation_bytes,
        tensor_metadata,
    )
    from tools.analyze_pw0327_target_bonus_q8_pilot import (
        PROMPTS,
        analyze_report as analyze_pw0327_report,
        safety_gate,
    )
    from tools.build_pw0328_target_bonus_corpus import (
        CAPTURE_KERNEL_SHA256,
        HIDDEN,
        REVISION,
        TOKENIZER_CONFIG_SHA256,
        TOKENIZER_SHA256,
        VOCAB,
        finite_f32_payload,
        mtp_history_binding,
        transaction_semantics,
        validate_generation_source,
        validate_prefill_source,
    )
    from tools.generate_native_mtp_first_proposal import (
        authenticate as authenticate_native_mtp,
        expected_mtp_inventory,
        load_target_hidden,
        sha256_file,
    )
    from tools.host_safety import HostSafetyMonitor, HostSafetyViolation
    from tools.native_mtp_reference import (
        NativeMtpLayerResult,
        generate_layer_proposal,
        rotate_mtp_input_ids,
    )
    from tools.openrouter_reference import atomic_write_new, canonical_json
    from tools.reproduce_pw0311_k4_expert import verify_clean_commit
    from tools.run_native_mtp_latency_reference import (
        KNOWN_MTP_MANIFEST_SHA256,
        logits_identity,
    )
except ModuleNotFoundError:
    from analyze_pread_expert_acquisition import analyze as analyze_pw0136
    from analyze_pressure_elastic_residency import (
        fixed_tensor_names,
        resident_allocation_bytes,
        tensor_metadata,
    )
    from analyze_pw0327_target_bonus_q8_pilot import (
        PROMPTS,
        analyze_report as analyze_pw0327_report,
        safety_gate,
    )
    from build_pw0328_target_bonus_corpus import (
        CAPTURE_KERNEL_SHA256,
        HIDDEN,
        REVISION,
        TOKENIZER_CONFIG_SHA256,
        TOKENIZER_SHA256,
        VOCAB,
        finite_f32_payload,
        mtp_history_binding,
        transaction_semantics,
        validate_generation_source,
        validate_prefill_source,
    )
    from generate_native_mtp_first_proposal import (
        authenticate as authenticate_native_mtp,
        expected_mtp_inventory,
        load_target_hidden,
        sha256_file,
    )
    from host_safety import HostSafetyMonitor, HostSafetyViolation
    from native_mtp_reference import (
        NativeMtpLayerResult,
        generate_layer_proposal,
        rotate_mtp_input_ids,
    )
    from openrouter_reference import atomic_write_new, canonical_json
    from reproduce_pw0311_k4_expert import verify_clean_commit
    from run_native_mtp_latency_reference import KNOWN_MTP_MANIFEST_SHA256, logits_identity


EXPERIMENT_ID = "PW-0330"
EVIDENCE_CLASS = "pw0330_cyclic_mtp_q32_prefix_falsifier"
SEMANTIC = "cyclic_mtp_012_v1"
CAPTURE_COMMIT = "26d2ea31852c0d63bd022df6d571fd722137c39f"
CONTRACT_COMMIT = "bce854d098e1ec162bd44eb9306642b16e5d38e2"
CONTRACT_RELATIVE_PATH = "experiments/PW-0330-cyclic-mtp-q32-prefix-falsifier.md"
CONTRACT_SHA256 = "085d4fb0a6bb2a25e3d7482cb885bf4e8a99479270cc0dbd496a768a3d429b9e"
TARGET_SHA256 = "dda459684c194b03491f36e9b66521ff00c400a6cc38d23a567a5a92ef8fb17d"
RED_LINES_SHA256 = "cc261ad9bd67a865715e72cbbadf3b74c3f1f282e17a8ef86ed02c1a92fb8b36"
TARGET_LAYERS = 48

MODEL_LOCK_SHA256 = "df8c74e6f9e1cef154aae5881b9042777653206aaff72855f7b1a1340e0d1050"
CHECKPOINT_RECEIPT_SHA256 = "9ddc8a99755f04ae2ea3c2484f6dd022d3f3a681b5a72c915ee4de833dbb0d03"
MTP_SHA256 = "a0e41a193b2762b0c83e577f83206d0777028de6916408c8c368730c0c9e2143"
SGLANG_LOCK_SHA256 = "8a0db42bedbee1d0c8dbd1d5439c5b7baacf4cf7eb8beb20c011158730fc242b"

PW0327_ANALYSIS = Path(
    "/Users/chad/Models/mimo-prismwing/evidence/PW-0327/analysis-001/analysis.json"
)
PW0327_CODE_REPORT = Path(
    "/Users/chad/Models/mimo-prismwing/evidence/PW-0327/pilot-001/code/report.json"
)
PW0327_CODE_PROGRESS = Path(
    "/Users/chad/Models/mimo-prismwing/evidence/PW-0327/pilot-001/code/report.progress.jsonl"
)
PW0327_ANALYSIS_SHA256 = "a54eeab1d136b938ddebe01a4206d6084bbeb2a2ca6a1395d88edfac337eaeed"
PW0327_CODE_REPORT_SHA256 = "83f9a37ae0da6e12b3289d70d3295539b0e4c67f8aaaa084cbcf0e1ef236910e"
PW0327_CODE_PROGRESS_SHA256 = "df941ef2989ffe3acfc88318ba55171622be5e0ed0c4b68b5152480ab24237cc"
PW0327_PREFIX_COUNTS = [376, 632, 832, 1035, 1236, 1391, 1537, 1757]

PW0328_PREFILL_REPORT = Path(
    "/Volumes/Elements/mimo-prismwing/evidence/PW-0328/prefill-001/code/report.json"
)
PW0328_PREFILL_HIDDEN = Path(
    "/Volumes/Elements/mimo-prismwing/evidence/PW-0328/prefill-001/code/target-layer47-hidden.f32"
)
PW0328_GENERATION_REPORT = Path(
    "/Volumes/Elements/mimo-prismwing/evidence/PW-0328/corpus-001/code/report.json"
)
PW0328_GENERATION_PROGRESS = Path(
    "/Volumes/Elements/mimo-prismwing/evidence/PW-0328/corpus-001/code/report.progress.jsonl"
)
PW0328_GENERATION_HIDDEN = Path(
    "/Volumes/Elements/mimo-prismwing/evidence/PW-0328/corpus-001/code/verification-layer47-hidden.f32"
)
PW0328_PREFILL_REPORT_SHA256 = "c39c7c86ec001c80ab64a2e258b0d7b8a2e96205d1201eec2b67b6d12ae05aa9"
PW0328_PREFILL_HIDDEN_SHA256 = "616ac368c4893517083fef39e58ecc41b85001cdac7ddedf9db66d3ea249b938"
PW0328_GENERATION_REPORT_SHA256 = "e5c896e72654bfdd963bc984293b742b3687d2fac9873444f2c591726e3dd287"
PW0328_GENERATION_PROGRESS_SHA256 = "77cd2af85d2b0f90f1e94de61947c4490589b8b4336c7f1746f83b48ac69df1e"
PW0328_GENERATION_HIDDEN_SHA256 = "31b9941ddd1446184ad1ef8050fda130cfc1aabb4115c1beb15bab943a211c2b"

PW0211_REPORT = Path(
    "/Volumes/Elements/mimo-prismwing/evidence/PW-0211/known-validation-001/report.json"
)
PW0211_REPORT_SHA256 = "395e61eb628c1b9ec3c892d285f5b3d0bc0749b6e5e7bc782cb5671dd299645f"
PW0206_PREFIX = Path(
    "/Volumes/Elements/mimo-prismwing/evidence/PW-0206/corrected-prefix-001/manifest.json"
)
PW0206_DECODE = Path(
    "/Volumes/Elements/mimo-prismwing/evidence/PW-0206/corrected-decode-001.json"
)
PW0206_MTP_MANIFEST = Path(
    "/Volumes/Elements/mimo-prismwing/evidence/PW-0206/mtp-001/manifest.json"
)

Q4_CONTROLS = (
    (
        Path(
            "/Volumes/Elements/mimo-prismwing/evidence/PW-0215/"
            "native-mtp-slice-broadening-001/code-candidate-001.native-mtp/"
            "transaction-000-proposal/report.json"
        ),
        "5b5adb10b9e5fefc016764c2ab97c4c1c62ad29e9d48b88e0829d19cd4205c62",
    ),
    (
        Path(
            "/Volumes/Elements/mimo-prismwing/evidence/PW-0215/"
            "native-mtp-slice-broadening-001/code-candidate-002.native-mtp/"
            "transaction-000-proposal/report.json"
        ),
        "e56b27b2464bfd1155037302b2cf183108b6bb38279e64653f73ecc325868fb5",
    ),
)
Q4_IMPLEMENTATION_COMMIT = "180491db5039d0e72213f3c4bb040ba7165688c3"
Q4_BLOCK = [8420, 374, 264, 4583]
Q4_LOGITS_SHA256 = [
    "41815b0cef9a8123d0dd2507ea7475b9c43ec9ab9e65bf1f8499e6e9f6997675",
    "e8139e7f8478aba044af39986b8c52baa15b6e6fe5882b6ed5d0edceb41e0480",
    "076991e22f2d19606f6dcbe3c1d483b74ca7709e0c5ce0a60542c3da1bdc982e",
]

PW0207_OFFLINE = Path(
    "/Volumes/Elements/mimo-prismwing/evidence/PW-0207/offline-002.json"
)
PW0207_OFFLINE_SHA256 = "1dedbef7c79aa23835d194f52760a1f2c65dcca1481bd6df2d5602615c3fdad6"
PW0207_INDEX_SHA256 = "f2e1774c9acf9a62338b68c144e6fc7a66495e59f2e64b3078c1b7ef5a196816"

PW0136_RAW = Path("/Users/chad/Models/mimo-prismwing/evidence/PW-0136/run-001.json")
PW0136_ANALYSIS = Path(
    "/Users/chad/Models/mimo-prismwing/evidence/PW-0136/analysis-001/manifest.json"
)
PW0136_RAW_SHA256 = "e6ab84cada19c6036ee7b83f318c3920631141b9ea5e882cc88eb9784d0b5a56"
PW0136_ANALYSIS_SHA256 = "7ebf2cde5c4a3f4931d2d705993f822e38af13ea66bc3efc91410296b14e2aab"

S_FIXED = 7_743_236_992
S_FIXED_ALLOCATED = 7_745_470_464
S_MTP_ONLY = 1_189_400_448
S_MTP_LAYER = 396_466_816
S_LM_HEAD = 1_249_902_592
SOURCE_EXPERT_BYTES = 25_171_968
RESIDENCY_BYTES = 12 * 1024**3
EMBEDDING_ROW_BYTES = 4096 * 2
BANDWIDTH_ARTIFACT_BYTES = 201_719_808
BANDWIDTH_EXACT = 3_470_425_919.832775
BANDWIDTH_FAVORABLE = 3_470_448_309.677419


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def strict_hash(path: Path, expected: str, label: str) -> str:
    actual = sha256_file(path)
    require(actual == expected, f"{label} SHA-256 mismatch")
    return actual


def authenticate_execution_contract(repo: Path, commit: str) -> dict[str, str]:
    observed = {
        "contract_sha256": strict_hash(
            repo / CONTRACT_RELATIVE_PATH, CONTRACT_SHA256, "PW-0330 contract"
        ),
        "target_sha256": strict_hash(repo / "TARGET.md", TARGET_SHA256, "TARGET.md"),
        "red_lines_sha256": strict_hash(
            repo / "RED_LINES.md", RED_LINES_SHA256, "RED_LINES.md"
        ),
    }
    ancestry = subprocess.run(
        ["git", "-C", str(repo), "merge-base", "--is-ancestor", CONTRACT_COMMIT, commit],
        check=False,
        capture_output=True,
    )
    require(ancestry.returncode == 0, "PW-0330 execution commit does not descend from contract")
    return {"contract_commit": CONTRACT_COMMIT, **observed}


def input_ids_identity(input_ids: list[int]) -> str:
    require(
        bool(input_ids)
        and all(type(token) is int and 0 <= token < VOCAB for token in input_ids),
        "invalid MTP input token IDs",
    )
    return hashlib.sha256(np.asarray(input_ids, dtype="<i8").tobytes()).hexdigest()


def tensor_identity(value: torch.Tensor) -> str:
    require(value.dtype == torch.bfloat16 and value.ndim == 2, "invalid immutable hidden")
    payload = value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(payload).hexdigest()


def target_token_rank(logits: torch.Tensor, token_id: int) -> int:
    require(
        logits.ndim == 1
        and logits.numel() > 0
        and torch.isfinite(logits).all().item()
        and type(token_id) is int
        and 0 <= token_id < logits.numel(),
        "invalid logits or target token for rank",
    )
    target = logits[token_id]
    ids = torch.arange(logits.numel(), device=logits.device)
    better = int((logits > target).sum().item())
    earlier_ties = int(((logits == target) & (ids < token_id)).sum().item())
    return 1 + better + earlier_ties


def validate_target_spine(transaction: dict[str, Any], *, terminal: bool = False) -> dict[str, Any]:
    semantic = transaction_semantics(transaction, index=0, terminal=terminal)
    require(semantic["proposal_converged"] is True, "target transaction did not converge")
    require(semantic["A"] == 8, "target transaction full A is not eight")
    require(
        semantic["observable_A"] == 8
        and semantic["retained_proposal_rows"] == 8,
        "target transaction is output-clipped",
    )
    require(
        semantic["verifier_retained_proposal_rows"] == 8,
        "target transaction retained fewer than eight verifier rows",
    )
    require(
        semantic["verifier_authorized_token_ids"] == semantic["posterior_token_ids"],
        "target posterior is not the complete verifier-authorized spine",
    )
    return semantic


def prefix_route_metrics(traces: list[dict[str, Any]], accepted_length: int) -> dict[str, Any]:
    require(type(accepted_length) is int and 1 <= accepted_length <= 8, "invalid route prefix A")
    require(
        len(traces) == TARGET_LAYERS,
        "verification trace must contain 48 layers",
    )
    identities: set[tuple[int, int]] = set()
    per_layer = []
    for layer, trace in enumerate(traces):
        require(type(trace.get("layer")) is int and trace["layer"] == layer, "route layer order")
        selected = trace.get("selected_experts_by_position")
        weights = trace.get("route_weights_by_position")
        if layer == 0:
            require(selected == [] and weights == [], "dense layer zero contains routes")
            continue
        require(
            isinstance(selected, list)
            and isinstance(weights, list)
            and len(selected) == 8
            and len(weights) == 8,
            "q8 route row count",
        )
        full_union: set[int] = set()
        prefix_union: set[int] = set()
        for position, (expert_row, weight_row) in enumerate(zip(selected, weights, strict=True)):
            require(
                isinstance(expert_row, list)
                and len(expert_row) == 8
                and len(set(expert_row)) == 8
                and all(type(expert) is int and 0 <= expert < 256 for expert in expert_row),
                "expert route row mismatch",
            )
            require(
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
                "route weight row mismatch",
            )
            full_union.update(expert_row)
            if position < accepted_length:
                prefix_union.update(expert_row)
        require(
            math.isclose(float(trace.get("U")), len(full_union) / 8.0, abs_tol=1.0e-12),
            "route layer U mismatch",
        )
        identities.update((layer, expert) for expert in prefix_union)
        per_layer.append({
            "layer": layer,
            "unique_experts": len(prefix_union),
            "experts": sorted(prefix_union),
        })
    identity_rows = [
        {"layer": layer, "expert": expert} for layer, expert in sorted(identities)
    ]
    return {
        "A": accepted_length,
        "N_A": len(identities),
        "unique_source_expert_bytes": len(identities) * SOURCE_EXPERT_BYTES,
        "identity_sha256": hashlib.sha256(canonical_json(identity_rows)).hexdigest(),
        "identities": identity_rows,
        "per_layer": per_layer,
    }


def storage_ceiling(accepted_length: int, identity_count: int) -> dict[str, Any]:
    require(type(accepted_length) is int and accepted_length >= 1, "invalid ceiling A")
    require(type(identity_count) is int and identity_count >= 0, "invalid ceiling identity count")
    logical = S_FIXED + S_MTP_ONLY + identity_count * SOURCE_EXPERT_BYTES
    miss = max(0, logical - RESIDENCY_BYTES)
    result: dict[str, Any] = {
        "A": accepted_length,
        "N_A": identity_count,
        "logical_bytes_before_joint_residency": logical,
        "joint_residency_bytes": RESIDENCY_BYTES,
        "miss_bytes": miss,
        "unbounded_storage_only_ceiling": miss == 0,
    }
    if miss == 0:
        result.update({
            "candidate_favorable_tps_ceiling": None,
            "raw_exact_tps_ceiling": None,
            "candidate_favorable_at_or_below_one": False,
        })
    else:
        favorable = accepted_length * BANDWIDTH_FAVORABLE / miss
        exact = accepted_length * BANDWIDTH_EXACT / miss
        result.update({
            "candidate_favorable_tps_ceiling": favorable,
            "raw_exact_tps_ceiling": exact,
            "candidate_favorable_at_or_below_one": favorable <= 1.0,
        })
    return result


def disposition(first_mismatch_index: int | None, ceiling: dict[str, Any] | None) -> str:
    if first_mismatch_index is None:
        require(ceiling is None, "prefix exhaustion cannot carry a mismatch ceiling")
        return "prefix_authority_exhausted"
    require(
        type(first_mismatch_index) is int
        and 0 <= first_mismatch_index < 8
        and isinstance(ceiling, dict),
        "invalid mismatch disposition inputs",
    )
    if ceiling["candidate_favorable_at_or_below_one"]:
        return "conditional_hard_storage_rejection"
    return "analytical_only_direct_q32_trace_required"


def mtp_inventory_authority() -> dict[str, Any]:
    item_bytes = {"F8_E4M3": 1, "BF16": 2, "F32": 4}
    per_layer = [0, 0, 0]
    inventory = expected_mtp_inventory()
    require(len(inventory) == 48, "MTP inventory tensor count")
    for name, (dtype, shape) in inventory.items():
        require(dtype in item_bytes and all(type(value) is int and value > 0 for value in shape),
                "MTP inventory tensor identity")
        layer = int(name.split(".")[3])
        require(layer in (0, 1, 2), "MTP inventory layer")
        per_layer[layer] += math.prod(shape) * item_bytes[dtype]
    require(per_layer == [S_MTP_LAYER] * 3 and sum(per_layer) == S_MTP_ONLY,
            "MTP-only source-byte identity")
    return {
        "tensor_count": len(inventory),
        "per_layer_logical_tensor_bytes": per_layer,
        "mtp_only_logical_tensor_bytes": sum(per_layer),
        "lm_head_included": False,
    }


def load_prefill_hidden(
    path: Path,
    rows: int,
    *,
    expected_sha256: str = PW0328_PREFILL_HIDDEN_SHA256,
) -> tuple[torch.Tensor, str]:
    strict_hash(path, expected_sha256, "PW-0328 prefill hidden")
    finite_f32_payload(path, rows * HIDDEN * 4, "PW-0328 prefill hidden")
    values = np.fromfile(path, dtype="<f4")
    require(values.size == rows * HIDDEN and np.isfinite(values).all(), "prefill hidden payload")
    hidden = torch.from_numpy(values.reshape(rows, HIDDEN).copy()).to(torch.bfloat16)
    return hidden, tensor_identity(hidden)


def execute_prefix(
    *,
    target_hidden: torch.Tensor,
    initial_input_ids: list[int],
    target_tokens: list[int],
    run_head: Callable[[int, int, torch.Tensor, list[int], int], dict[str, Any]],
    q4_expected: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    require(len(target_tokens) == 8, "target prefix must contain eight tokens")
    require(len(initial_input_ids) == target_hidden.shape[0], "MTP history/input length")
    initial_hidden_identity = tensor_identity(target_hidden)
    input_ids = list(initial_input_ids)
    head_results = []
    first_mismatch: int | None = None
    for stage_end in (4, 8):
        while len(head_results) < stage_end:
            head = len(head_results)
            layer = head % 3
            call_ids = list(input_ids)
            before_ids = list(call_ids)
            record = run_head(head, layer, target_hidden, call_ids, target_tokens[head])
            require(call_ids == before_ids, "MTP head mutated its input token IDs")
            require(tensor_identity(target_hidden) == initial_hidden_identity,
                    "MTP head mutated immutable target hidden")
            require(record.get("layer") == layer, "MTP head returned wrong layer")
            proposal = record.get("proposal_token_id")
            require(type(proposal) is int and 0 <= proposal < VOCAB, "invalid MTP proposal token")
            require(record.get("input_token_ids") == before_ids, "MTP head input record mismatch")
            require(
                record.get("input_token_ids_sha256") == input_ids_identity(before_ids),
                "MTP head input hash mismatch",
            )
            if q4_expected is not None and head < 3:
                require(len(q4_expected) == 3, "invalid q4 control")
                expected = q4_expected[head]
                require(
                    proposal == expected["proposal_token_id"]
                    and record.get("logits_sha256") == expected["logits_sha256"],
                    f"trained q4 reproduction mismatch at head {head}",
                )
            match = proposal == target_tokens[head]
            head_results.append({
                **record,
                "head_index": head,
                "stage": "heads_0_through_3" if head < 4 else "heads_4_through_7",
                "first_new_scheduler_behavior": head == 3,
                "target_token_id": target_tokens[head],
                "match": match,
                "immutable_hidden_bf16_sha256": initial_hidden_identity,
            })
            if not match:
                first_mismatch = head
                break
            input_ids = rotate_mtp_input_ids(input_ids, proposal)
        if first_mismatch is not None:
            break
    matches = sum(1 for row in head_results if row["match"])
    require(
        matches == (first_mismatch if first_mismatch is not None else len(head_results)),
        "non-prefix match accounting",
    )
    return {
        "head_results": head_results,
        "first_mismatch_index": first_mismatch,
        "authenticated_prefix_matches": matches,
        "evaluated_heads": len(head_results),
        "prefix_authority_exhausted": first_mismatch is None,
        "A": first_mismatch + 1 if first_mismatch is not None else None,
        "final_input_token_ids": input_ids,
    }


def authenticate_planning_authority(repo: Path) -> dict[str, Any]:
    strict_hash(PW0327_ANALYSIS, PW0327_ANALYSIS_SHA256, "PW-0327 analysis")
    strict_hash(PW0327_CODE_REPORT, PW0327_CODE_REPORT_SHA256, "PW-0327 code report")
    strict_hash(PW0327_CODE_PROGRESS, PW0327_CODE_PROGRESS_SHA256, "PW-0327 code progress")
    analysis = json.loads(PW0327_ANALYSIS.read_text())
    require(
        analysis.get("schema_version") == 1
        and analysis.get("experiment_id") == "PW-0327"
        and analysis.get("status") == "complete",
        "PW-0327 analysis identity",
    )
    rows = [row for row in analysis.get("categories", []) if row.get("category") == "code"]
    require(len(rows) == 1, "PW-0327 code summary cardinality")
    code = analyze_pw0327_report(
        category="code",
        report_path=PW0327_CODE_REPORT,
        prompt_path=repo / PROMPTS["code"][0],
        prompt_sha256=PROMPTS["code"][1],
        capture_commit=analysis["capture_commit"],
    )
    require(
        code["report_sha256"] == PW0327_CODE_REPORT_SHA256
        and code["progress_sha256"] == PW0327_CODE_PROGRESS_SHA256
        and code["proposal_token_ids"] == Q4_BLOCK + [8129, 315, 264, 3084]
        and code["posterior_token_ids"] == [374, 264, 4583, 8129, 315, 264, 3084, 2268]
        and code["A"] == 8
        and code["proposal_converged"] is True,
        "PW-0327 planning code authority",
    )
    raw = json.loads(PW0327_CODE_REPORT.read_text())["transactions"][0]
    table = [prefix_route_metrics(raw["verification_layer_traces"], value) for value in range(1, 9)]
    require([row["N_A"] for row in table] == PW0327_PREFIX_COUNTS,
            "PW-0327 planning prefix identity counts")
    return {
        "analysis_file": str(PW0327_ANALYSIS),
        "analysis_sha256": PW0327_ANALYSIS_SHA256,
        "code_report_file": str(PW0327_CODE_REPORT),
        "code_report_sha256": PW0327_CODE_REPORT_SHA256,
        "code_progress_file": str(PW0327_CODE_PROGRESS),
        "code_progress_sha256": PW0327_CODE_PROGRESS_SHA256,
        "anchor_token_id": code["proposal_token_ids"][0],
        "target_token_ids": code["posterior_token_ids"],
        "prefix_identity_counts": PW0327_PREFIX_COUNTS,
        "planning_only": True,
    }


def authenticate_fresh_code(repo: Path) -> dict[str, Any]:
    strict_hash(PW0328_PREFILL_REPORT, PW0328_PREFILL_REPORT_SHA256, "PW-0328 prefill report")
    strict_hash(PW0328_PREFILL_HIDDEN, PW0328_PREFILL_HIDDEN_SHA256, "PW-0328 prefill hidden")
    strict_hash(PW0328_GENERATION_REPORT, PW0328_GENERATION_REPORT_SHA256,
                "PW-0328 generation report")
    strict_hash(PW0328_GENERATION_PROGRESS, PW0328_GENERATION_PROGRESS_SHA256,
                "PW-0328 generation progress")
    strict_hash(PW0328_GENERATION_HIDDEN, PW0328_GENERATION_HIDDEN_SHA256,
                "PW-0328 verifier hidden")
    report, source, windows = validate_generation_source(
        category="code",
        evidence_root=PW0328_GENERATION_REPORT.parent.parent,
        repo=repo,
        capture_commit=CAPTURE_COMMIT,
    )
    prefill = validate_prefill_source(
        category="code",
        prefill_root=PW0328_PREFILL_REPORT.parent.parent,
        generation_report=report,
        prompt_path=repo / PROMPTS["code"][0],
        capture_commit=CAPTURE_COMMIT,
    )
    require(
        len(report["transactions"]) == 9
        and len(windows) == 8
        and source["report_sha256"] == PW0328_GENERATION_REPORT_SHA256
        and source["progress_sha256"] == PW0328_GENERATION_PROGRESS_SHA256
        and source["hidden_sha256"] == PW0328_GENERATION_HIDDEN_SHA256
        and prefill["report_sha256"] == PW0328_PREFILL_REPORT_SHA256
        and prefill["hidden_sha256"] == PW0328_PREFILL_HIDDEN_SHA256,
        "PW-0328 frozen code source closure",
    )
    transaction = report["transactions"][0]
    semantic = validate_target_spine(transaction, terminal=False)
    history = mtp_history_binding(
        report,
        0,
        str(PW0328_PREFILL_HIDDEN),
        str(PW0328_GENERATION_HIDDEN),
    )
    require(
        history["target_hidden_rows"] == prefill["hidden_rows"] == 70
        and len(history["target_hidden_segments"]) == 1
        and history["target_hidden_segments"][0]["source"] == "prefill",
        "PW-0328 transaction-zero history source",
    )
    table = [
        prefix_route_metrics(transaction["verification_layer_traces"], value)
        for value in range(1, 9)
    ]
    return {
        "report": report,
        "transaction": transaction,
        "semantic": semantic,
        "history": history,
        "route_table": table,
        "source": source,
        "prefill": prefill,
    }


def authenticate_q4_controls(initial_input_ids: list[int]) -> dict[str, Any]:
    controls = []
    for report_path, expected_hash in Q4_CONTROLS:
        strict_hash(report_path, expected_hash, "PW-0215 q4 report")
        report = json.loads(report_path.read_text())
        request_path = report_path.parent.parent / "transaction-000-request.json"
        request_hash = sha256_file(request_path)
        request = json.loads(request_path.read_text())
        require(
            report.get("schema_version") == 1
            and report.get("evidence_class") == "pw0211_live_target_cache_native_mtp_q4_proposal"
            and report.get("status") == "passed"
            and report.get("accepted_tokens") == 0
            and report.get("performance_claim") is None
            and report.get("implementation")
            == {"commit": Q4_IMPLEMENTATION_COMMIT, "dirty": False},
            "PW-0215 q4 report identity",
        )
        identities = report["identities"]
        require(
            identities.get("request_sha256") == request_hash
            and identities.get("target_hidden_sha256") == PW0328_PREFILL_HIDDEN_SHA256
            and identities.get("checkpoint_verification_sha256") == CHECKPOINT_RECEIPT_SHA256
            and identities.get("mtp_sha256") == MTP_SHA256
            and identities.get("sglang_mtp_lock_sha256") == SGLANG_LOCK_SHA256,
            "PW-0215 q4 source identity",
        )
        require(
            request.get("schema_version") == 1
            and request.get("semantic") == "pw0211_live_target_cache_native_mtp_request"
            and request.get("transaction_index") == 0
            and request.get("target_hidden_rows") == 70
            and request.get("target_hidden_sha256") == PW0328_PREFILL_HIDDEN_SHA256
            and request.get("anchor_token_id") == Q4_BLOCK[0]
            and request.get("mtp_layer0_input_token_ids") == initial_input_ids,
            "PW-0215 q4 request identity",
        )
        request_hidden = Path(request["target_hidden_file"])
        strict_hash(request_hidden, PW0328_PREFILL_HIDDEN_SHA256, "PW-0215 q4 hidden")
        layer_results = report.get("layer_results")
        require(
            report.get("target_hidden_rows") == 70
            and report.get("anchor_token_id") == Q4_BLOCK[0]
            and report.get("native_mtp_q4_block_token_ids") == Q4_BLOCK
            and isinstance(layer_results, list)
            and len(layer_results) == 3,
            "PW-0215 q4 result identity",
        )
        expected_layers = []
        for layer, row in enumerate(layer_results):
            require(
                row.get("layer") == layer
                and row.get("proposal_token_id") == Q4_BLOCK[layer + 1]
                and row.get("logits_sha256") == Q4_LOGITS_SHA256[layer],
                "PW-0215 q4 layer identity",
            )
            expected_layers.append({
                "layer": layer,
                "proposal_token_id": row["proposal_token_id"],
                "logits_sha256": row["logits_sha256"],
            })
        controls.append({
            "report_file": str(report_path),
            "report_sha256": expected_hash,
            "request_file": str(request_path),
            "request_sha256": request_hash,
            "hidden_file": str(request_hidden),
            "hidden_sha256": PW0328_PREFILL_HIDDEN_SHA256,
            "layer_results": expected_layers,
        })
    require(controls[0]["layer_results"] == controls[1]["layer_results"],
            "PW-0215 q4 controls disagree")
    return {"controls": controls, "expected_layers": controls[0]["layer_results"]}


def authenticate_fixed_spine(checkpoint_root: Path, verification_path: Path) -> dict[str, Any]:
    strict_hash(PW0207_OFFLINE, PW0207_OFFLINE_SHA256, "PW-0207 offline authority")
    strict_hash(verification_path, CHECKPOINT_RECEIPT_SHA256, "checkpoint receipt")
    offline = json.loads(PW0207_OFFLINE.read_text())
    verification = json.loads(verification_path.read_text())
    require(
        offline.get("schema_version") == 1
        and offline.get("evidence_class") == "pw0207_pressure_elastic_offline_residency_falsifier"
        and offline.get("revision") == REVISION
        and offline.get("git_dirty") is False
        and offline.get("performance_claim") is None,
        "PW-0207 offline identity",
    )
    require(
        verification.get("complete") is True and verification.get("revision") == REVISION,
        "checkpoint receipt identity",
    )
    records = {row["path"]: row for row in verification.get("files", [])}
    index_path = checkpoint_root / "model.safetensors.index.json"
    require(
        records.get(index_path.name, {}).get("sha256") == PW0207_INDEX_SHA256
        and sha256_file(index_path) == PW0207_INDEX_SHA256
        and offline.get("identities", {}).get("tensor_index_sha256") == PW0207_INDEX_SHA256,
        "receipt-authenticated tensor index",
    )
    index = json.loads(index_path.read_text())
    weight_map = index.get("weight_map")
    require(isinstance(weight_map, dict), "checkpoint weight map")
    shard_hashes = {
        path: row["sha256"]
        for path, row in records.items()
        if path.endswith(".safetensors") and row.get("status") == "verified"
    }
    names = fixed_tensor_names(weight_map)
    metadata = tensor_metadata(checkpoint_root, weight_map, shard_hashes, set(names))
    objects = offline.get("residency_manifest", {}).get("objects")
    require(isinstance(objects, list), "PW-0207 object authority")
    object_map = {row.get("identity"): row for row in objects}
    require(len(object_map) == len(objects), "PW-0207 duplicate object identity")
    source_bytes = 0
    allocated_bytes = 0
    for name in names:
        row = metadata[name]
        authority = object_map.get(f"tensor:{name}")
        source_bytes += row["bytes"]
        allocated = resident_allocation_bytes(row["bytes"])
        allocated_bytes += allocated
        require(
            isinstance(authority, dict)
            and authority.get("category") == "shared_spine"
            and authority.get("source_bytes") == row["bytes"]
            and authority.get("bytes") == allocated
            and authority.get("tensor_metadata_sha256")
            == hashlib.sha256(canonical_json(row)).hexdigest(),
            f"PW-0207 fixed object mismatch: {name}",
        )
    require(
        len(names) == 381
        and source_bytes == S_FIXED
        and allocated_bytes == S_FIXED_ALLOCATED
        and metadata["lm_head.weight"]["bytes"] == S_LM_HEAD,
        "fixed target-spine byte identity",
    )
    return {
        "offline_file": str(PW0207_OFFLINE),
        "offline_sha256": PW0207_OFFLINE_SHA256,
        "tensor_index_sha256": PW0207_INDEX_SHA256,
        "fixed_object_count": len(names),
        "fixed_logical_source_bytes": source_bytes,
        "fixed_page_aligned_allocation_bytes": allocated_bytes,
        "lm_head_logical_source_bytes": S_LM_HEAD,
        "lm_head_is_one_of_fixed_objects": "lm_head.weight" in names,
    }


def authenticate_bandwidth() -> dict[str, Any]:
    strict_hash(PW0136_RAW, PW0136_RAW_SHA256, "PW-0136 raw")
    strict_hash(PW0136_ANALYSIS, PW0136_ANALYSIS_SHA256, "PW-0136 analysis")
    derived = analyze_pw0136(PW0136_RAW)
    require(
        hashlib.sha256(canonical_json(derived)).hexdigest() == PW0136_ANALYSIS_SHA256,
        "PW-0136 independent analysis reproduction",
    )
    exact_ms = derived["physical_continuation_gate"]["cold_median_ms"]
    exact = BANDWIDTH_ARTIFACT_BYTES / (exact_ms / 1000.0)
    favorable = BANDWIDTH_ARTIFACT_BYTES / (58.125 / 1000.0)
    require(
        exact_ms == 58.125375
        and math.isclose(exact, BANDWIDTH_EXACT, rel_tol=0.0, abs_tol=1.0e-6)
        and math.isclose(favorable, BANDWIDTH_FAVORABLE, rel_tol=0.0, abs_tol=1.0e-6),
        "PW-0136 bandwidth identity",
    )
    return {
        "raw_file": str(PW0136_RAW),
        "raw_sha256": PW0136_RAW_SHA256,
        "analysis_file": str(PW0136_ANALYSIS),
        "analysis_sha256": PW0136_ANALYSIS_SHA256,
        "artifact_bytes": BANDWIDTH_ARTIFACT_BYTES,
        "raw_exact_median_ms": exact_ms,
        "raw_exact_bytes_per_second": exact,
        "rounded_historical_median_ms": 58.125,
        "candidate_favorable_bytes_per_second": favorable,
    }


def reproduce_known_control(
    *,
    checkpoint: Any,
    mtp_path: Path,
    prefix_manifest: dict[str, Any],
    anchor_token: int,
    expected_token: int,
    safety: HostSafetyMonitor,
) -> dict[str, Any]:
    strict_hash(PW0211_REPORT, PW0211_REPORT_SHA256, "PW-0211 known validation")
    prior = json.loads(PW0211_REPORT.read_text())
    require(
        prior.get("schema_version") == 1
        and prior.get("evidence_class") == "pw0211_last_row_native_mtp_known_proposal_validation"
        and prior.get("status") == "passed"
        and prior.get("full_row_logits_bit_identical") is True
        and prior.get("accepted_tokens") == 0
        and prior.get("performance_claim") is None,
        "PW-0211 known report identity",
    )
    strict_hash(PW0206_MTP_MANIFEST, KNOWN_MTP_MANIFEST_SHA256, "PW-0206 MTP manifest")
    manifest = json.loads(PW0206_MTP_MANIFEST.read_text())
    logits_path = PW0206_MTP_MANIFEST.parent / manifest["captures"]["logits"]["file"]
    known = np.fromfile(logits_path, dtype="<f4")
    require(known.shape == (VOCAB,) and np.isfinite(known).all(), "known full-row logits")
    hidden = load_target_hidden(PW0206_PREFIX, prefix_manifest)
    input_ids = [*prefix_manifest["prompt_token_ids"][1:], anchor_token]
    started = time.monotonic()
    result = generate_layer_proposal(
        checkpoint,
        mtp_path,
        0,
        hidden,
        input_ids,
        lambda phase: safety.checkpoint(f"known_{phase}"),
    )
    complete_ms = (time.monotonic() - started) * 1000.0
    exact = bool(torch.equal(result.logits, torch.from_numpy(known.copy())))
    digest = logits_identity(result.logits)
    require(
        exact
        and result.proposal_token_id == prior["target_proposal_token_id"] == expected_token
        and digest == prior["last_row_logits_sha256"]
        == prior["identities"]["known_full_row_logits_sha256"],
        "PW-0211 bit-identical known control reproduction",
    )
    record = {
        "source_report_file": str(PW0211_REPORT),
        "source_report_sha256": PW0211_REPORT_SHA256,
        "known_manifest_file": str(PW0206_MTP_MANIFEST),
        "known_manifest_sha256": KNOWN_MTP_MANIFEST_SHA256,
        "proposal_token_id": result.proposal_token_id,
        "logits_sha256": digest,
        "full_row_logits_bit_identical": exact,
        "timings_ms": {**result.timings_ms, "complete": complete_ms},
    }
    del result, hidden, known
    gc.collect()
    return record


def prepare_output(repo: Path, commit: str, output: Path) -> None:
    verify_clean_commit(repo, commit)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=False)


def run(arguments: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    repo = arguments.repo.resolve()
    prepare_output(repo, arguments.commit, arguments.output)
    safety = HostSafetyMonitor()
    arguments._safety = safety
    torch.set_num_threads(1)

    execution_contract = authenticate_execution_contract(repo, arguments.commit)
    strict_hash(repo / "spec/model.lock.json", MODEL_LOCK_SHA256, "model lock")
    strict_hash(repo / "spec/sglang-mimo-mtp.lock.json", SGLANG_LOCK_SHA256, "SGLang lock")
    planning = authenticate_planning_authority(repo)
    fresh = authenticate_fresh_code(repo)
    history = fresh["history"]
    q4_controls = authenticate_q4_controls(history["mtp_layer0_input_token_ids"])
    mtp_inventory = mtp_inventory_authority()
    fixed = authenticate_fixed_spine(arguments.checkpoint, arguments.verification)
    bandwidth = authenticate_bandwidth()
    (
        checkpoint,
        mtp_path,
        known_prefix,
        native_identities,
        known_anchor,
        known_expected,
        _known_class,
    ) = authenticate_native_mtp(
        arguments.checkpoint,
        arguments.verification,
        PW0206_PREFIX,
        None,
        PW0206_DECODE,
        repo / "spec/sglang-mimo-mtp.lock.json",
        arguments.source_root,
    )
    require(
        native_identities["checkpoint_verification_sha256"] == CHECKPOINT_RECEIPT_SHA256
        and native_identities["mtp_sha256"] == MTP_SHA256
        and native_identities["sglang_mtp_lock_sha256"] == SGLANG_LOCK_SHA256,
        "native MTP authority identity",
    )
    safety.checkpoint("all_authorities_authenticated")
    known_control = reproduce_known_control(
        checkpoint=checkpoint,
        mtp_path=mtp_path,
        prefix_manifest=known_prefix,
        anchor_token=known_anchor,
        expected_token=known_expected,
        safety=safety,
    )
    safety.release_checkpoint("known_control_released", ["PW-0211 known hidden and logits"])

    target_hidden, hidden_bf16_sha256 = load_prefill_hidden(
        PW0328_PREFILL_HIDDEN, history["target_hidden_rows"]
    )
    target_tokens = list(fresh["semantic"]["posterior_token_ids"])
    source_identities = {
        "revision": REVISION,
        "capture_commit": CAPTURE_COMMIT,
        "capture_kernel_sha256": CAPTURE_KERNEL_SHA256,
        "tokenizer_sha256": TOKENIZER_SHA256,
        "tokenizer_config_sha256": TOKENIZER_CONFIG_SHA256,
        "checkpoint_verification_sha256": CHECKPOINT_RECEIPT_SHA256,
        "model_lock_sha256": MODEL_LOCK_SHA256,
        "mtp_sha256": MTP_SHA256,
        "sglang_mtp_lock_sha256": SGLANG_LOCK_SHA256,
        "prefill_report_sha256": PW0328_PREFILL_REPORT_SHA256,
        "prefill_hidden_sha256": PW0328_PREFILL_HIDDEN_SHA256,
        "generation_report_sha256": PW0328_GENERATION_REPORT_SHA256,
        "generation_progress_sha256": PW0328_GENERATION_PROGRESS_SHA256,
        "generation_verifier_hidden_sha256": PW0328_GENERATION_HIDDEN_SHA256,
        "known_control_report_sha256": PW0211_REPORT_SHA256,
        "q4_control_report_sha256": [expected for _path, expected in Q4_CONTROLS],
    }

    def real_head(
        head: int,
        layer: int,
        hidden: torch.Tensor,
        input_ids: list[int],
        target_token: int,
    ) -> dict[str, Any]:
        head_started = time.monotonic()
        before = safety.checkpoint(f"head_{head}_start")
        result: NativeMtpLayerResult = generate_layer_proposal(
            checkpoint,
            mtp_path,
            layer,
            hidden,
            input_ids,
            lambda phase: safety.checkpoint(f"head_{head}_{phase}"),
        )
        after = safety.checkpoint(f"head_{head}_complete")
        record = {
            "layer": layer,
            "input_token_ids": list(input_ids),
            "input_token_ids_sha256": input_ids_identity(input_ids),
            "input_token_ids_hash_encoding": "little_endian_signed_int64",
            "proposal_token_id": result.proposal_token_id,
            "target_token_rank": target_token_rank(result.logits, target_token),
            "logits_sha256": logits_identity(result.logits),
            "top20": [
                {"token_id": token, "logit": value} for token, value in result.top20
            ],
            "timings_ms": {
                **result.timings_ms,
                "complete": (time.monotonic() - head_started) * 1000.0,
            },
            "logical_source_bytes": (
                S_MTP_LAYER + len(input_ids) * EMBEDDING_ROW_BYTES + S_LM_HEAD
            ),
            "process_disk_bytes_read": (
                after.process_disk_bytes_read - before.process_disk_bytes_read
            ),
            "cache_state": (
                "known_control_warmed_mtp_layer_0_and_lm_head"
                if head == 0
                else "first_candidate_use_of_mtp_layer_with_warm_lm_head"
                if head in (1, 2)
                else "same_process_reused_mtp_layer_and_lm_head"
            ),
            "safety_start": before.to_dict(),
            "safety_complete": after.to_dict(),
            "implementation_commit": arguments.commit,
            "source_identities": source_identities,
        }
        del result
        gc.collect()
        return record

    execution = execute_prefix(
        target_hidden=target_hidden,
        initial_input_ids=history["mtp_layer0_input_token_ids"],
        target_tokens=target_tokens,
        run_head=real_head,
        q4_expected=q4_controls["expected_layers"],
    )
    mismatch = execution["first_mismatch_index"]
    selected_route = None
    selected_ceiling = None
    if mismatch is not None:
        selected_route = fresh["route_table"][mismatch]
        selected_ceiling = storage_ceiling(execution["A"], selected_route["N_A"])
    decision = disposition(mismatch, selected_ceiling)
    route_table = []
    for row in fresh["route_table"]:
        ceiling = storage_ceiling(row["A"], row["N_A"])
        route_table.append({
            "A": row["A"],
            "N_A": row["N_A"],
            "unique_source_expert_bytes": row["unique_source_expert_bytes"],
            "identity_sha256": row["identity_sha256"],
            "per_layer_unique_counts": [item["unique_experts"] for item in row["per_layer"]],
            "storage": ceiling,
        })

    del target_hidden
    gc.collect()
    safety.release_checkpoint(
        "proposal_inputs_released",
        ["fresh PW-0328 prefill hidden", "native MTP proposal intermediates"],
    )
    safety.checkpoint("final_service_health")
    snapshots = safety.evidence()
    gate8 = safety_gate(snapshots)
    # The recorded commit must cover the complete capture, not merely startup.
    verify_clean_commit(repo, arguments.commit)
    implementation = {
        "commit": arguments.commit,
        "dirty": False,
        "runner_file": str(repo / "tools/run_pw0330_cyclic_mtp_prefix.py"),
        "runner_sha256": sha256_file(repo / "tools/run_pw0330_cyclic_mtp_prefix.py"),
        "contract_sha256": sha256_file(
            repo / "experiments/PW-0330-cyclic-mtp-q32-prefix-falsifier.md"
        ),
        "target_sha256": sha256_file(repo / "TARGET.md"),
        "red_lines_sha256": sha256_file(repo / "RED_LINES.md"),
    }
    report = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "evidence_class": EVIDENCE_CLASS,
        "semantic": SEMANTIC,
        "exactness_class": "L2_modified_draft_scheduler_with_exact_target_authority",
        "status": "complete",
        "decision": decision,
        "implementation": implementation,
        "authorities": {
            "execution_contract": execution_contract,
            "revision": REVISION,
            "capture_commit": CAPTURE_COMMIT,
            "model_lock_sha256": MODEL_LOCK_SHA256,
            "checkpoint_verification_sha256": CHECKPOINT_RECEIPT_SHA256,
            "mtp_sha256": MTP_SHA256,
            "sglang_mtp_lock_sha256": SGLANG_LOCK_SHA256,
            "planning": planning,
            "fresh_generation": fresh["source"],
            "fresh_prefill": fresh["prefill"],
            "known_control": known_control,
            "q4_controls": q4_controls["controls"],
            "fixed_target_spine": fixed,
            "storage_bandwidth": bandwidth,
        },
        "category": "code",
        "transaction_index": 0,
        "target_authority": {
            "anchor_token_id": fresh["semantic"]["proposal_token_ids"][0],
            "proposal_token_ids": fresh["semantic"]["proposal_token_ids"],
            "posterior_token_ids": target_tokens,
            "proposal_converged": True,
            "A": 8,
            "verifier_retained_proposal_rows": 8,
            "retained_proposal_rows": 8,
        },
        "history": {
            "source": "fresh_pw0328_code_prefill",
            "rows": history["target_hidden_rows"],
            "shape": [history["target_hidden_rows"], HIDDEN],
            "source_dtype": "float32_little_endian",
            "runtime_dtype": "bfloat16",
            "source_f32_sha256": PW0328_PREFILL_HIDDEN_SHA256,
            "runtime_bf16_sha256": hidden_bf16_sha256,
            "initial_mtp_input_token_ids": history["mtp_layer0_input_token_ids"],
            "initial_mtp_input_token_ids_sha256": input_ids_identity(
                history["mtp_layer0_input_token_ids"]
            ),
            "input_token_ids_hash_encoding": "little_endian_signed_int64",
            "verifier_hidden_rows_consumed": 0,
            "draft_block_hidden_consumed": False,
            "future_target_hidden_consumed": False,
            "teacher_forced_target_tokens_consumed": False,
        },
        "scheduler": {
            "name": SEMANTIC,
            "full_q32_layer_use_counts": {"0": 11, "1": 10, "2": 10},
            "stage_end_heads": [4, 8],
            "evaluated_heads": execution["evaluated_heads"],
            "evaluated_layer_sequence": [row["layer"] for row in execution["head_results"]],
            "first_mismatch_index": mismatch,
            "authenticated_prefix_matches": execution["authenticated_prefix_matches"],
            "prefix_authority_exhausted": execution["prefix_authority_exhausted"],
            "A": execution["A"],
            "proposal_token_ids": [
                fresh["semantic"]["proposal_token_ids"][0],
                *[row["proposal_token_id"] for row in execution["head_results"]],
            ],
        },
        "head_results": execution["head_results"],
        "route_prefix_table": route_table,
        "selected_prefix_route": selected_route,
        "selected_storage_ceiling": selected_ceiling,
        "decision_scope": {
            "conditional_on_direct_q32_first_chunk_parity": decision
            == "conditional_hard_storage_rejection",
            "direct_q32_trace_required": decision
            in {
                "conditional_hard_storage_rejection",
                "analytical_only_direct_q32_trace_required",
                "prefix_authority_exhausted",
            },
            "prefix_authority_rows": 8,
            "heads_8_through_30_authorized": False,
        },
        "joint_residency_authority": {
            "fixed_target_logical_source_bytes": S_FIXED,
            "mtp_only_additional_logical_source_bytes": S_MTP_ONLY,
            "source_expert_logical_bytes": SOURCE_EXPERT_BYTES,
            "joint_residency_bytes": RESIDENCY_BYTES,
            "lm_head_logical_source_bytes": S_LM_HEAD,
            "lm_head_already_in_fixed_target_set": True,
            "lm_head_added_again": False,
            "mtp_inventory": mtp_inventory,
            "formula": "max(0,S_fixed+S_mtp_only+N_A*source_expert-R)",
        },
        "planning_crosscheck": {
            "planning_prefix_identity_counts": planning["prefix_identity_counts"],
            "fresh_prefix_identity_counts": [row["N_A"] for row in fresh["route_table"]],
            "equal": planning["prefix_identity_counts"]
            == [row["N_A"] for row in fresh["route_table"]],
            "execution_uses_fresh_values": True,
        },
        "diagnostic_ledgers": {
            "logical_source_bytes": sum(row["logical_source_bytes"] for row in execution["head_results"]),
            "process_disk_bytes_read": sum(
                row["process_disk_bytes_read"] for row in execution["head_results"]
            ),
            "cold_warm_order": (
                "cold process; required PW-0211 layer-0 control precedes candidate heads; "
                "each head records layer/LM-head reuse state"
            ),
        },
        "timings_ms": {"complete": (time.monotonic() - started) * 1000.0},
        "hardware": {"machine": platform.machine(), "platform": platform.platform()},
        "runtime": {"python": sys.version, "torch": torch.__version__, "numpy": np.__version__},
        "safety_gate": gate8,
        "safety_snapshots": snapshots,
        "batch_size": 1,
        "concurrency": 1,
        "accepted_tokens": 0,
        "performance_claim": None,
        "claims_excluded": [
            "actual q32 acceptance or a width-32 verifier result",
            "achieved, sustained, or complete endpoint TPS",
            "native trained q32 semantics",
            "q32 suffix routes or first-chunk parity for an unimplemented q32 kernel",
            "K4 fidelity, bank construction, cache allocation, or K4 bytes",
            "multimodal/full-capability promotion, a runtime default, or companion hardware",
        ],
    }
    atomic_write_new(arguments.output / "report.json", canonical_json(report))
    print(json.dumps({
        "output": str(arguments.output / "report.json"),
        "status": report["status"],
        "decision": decision,
        "evaluated_heads": execution["evaluated_heads"],
    }))
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--verification", required=True, type=Path)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    arguments._safety = None
    try:
        run(arguments)
        return 0
    except (
        FileExistsError,
        HostSafetyViolation,
        KeyError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ) as error:
        if (
            not isinstance(error, FileExistsError)
            and arguments.output.is_dir()
            and not (arguments.output / "failure.json").exists()
        ):
            safety = arguments._safety.evidence() if arguments._safety is not None else []
            atomic_write_new(arguments.output / "failure.json", canonical_json({
                "schema_version": 1,
                "experiment_id": EXPERIMENT_ID,
                "evidence_class": "pw0330_cyclic_mtp_q32_prefix_failure",
                "semantic": SEMANTIC,
                "status": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
                "safety_snapshots": safety,
                "accepted_tokens": 0,
                "performance_claim": None,
            }))
        print(json.dumps({"error": str(error), "output": str(arguments.output)}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
