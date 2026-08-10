#!/usr/bin/env python3
"""Authenticate and adjudicate PW-0175's structured sparse-prefill audit."""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter
from pathlib import Path

try:
    from tools.analyze_owned_epyc_companion_envelope import authenticate_implementation_commit
    from tools.analyze_pw0116_corpus import sha256_file
    from tools.host_safety import HostSafetyMonitor
    from tools.openrouter_reference import atomic_write_new, canonical_json
except ModuleNotFoundError:
    from analyze_owned_epyc_companion_envelope import authenticate_implementation_commit
    from analyze_pw0116_corpus import sha256_file
    from host_safety import HostSafetyMonitor
    from openrouter_reference import atomic_write_new, canonical_json


TARGET_SHA256 = "91fe6e0441bb0a0e1ab0852db60fb575d131b61ff069002c9c333f9b776e4950"
PW0158_SHA256 = "3b5b94cae112bee558ec46566ec09652c58bd434c3f47bebd3e0bc7c533fd315"
PW0161_SHA256 = "fc438d593d8ac99be3cc426496feb830256ffc48c75d58fc8bb9d6b09a2c6c8f"
SOURCE_HASHES = {
    "minference_paper": "65ae8b76b24ef6a8752367e8b6067db5541b6c574591c6c3483ac0524d2c3ef6",
    "minference_archive": "3ca80a6fb543925bdc75be8d8e3341d5897736b383e5c59b1691bf5c808e52a5",
    "minference_experiments_readme": "0fc0eb81903765796e53ac54b0b3295f003bf68947aa81c005c42cc57500a90a",
    "minference_forward": "b368e765fcc2021591f7cdf970e4e1a71731ed66f19e97023a13b70a02e6edc2",
    "minference_glm4_pattern": "6d35602b9a4f161e5d593392b883a2ca6c85bf20356e8612f55a49a47c50aa2c",
    "quest_archive": "07c941cf133bdbd9c87afc711059a5083d577039acc0fca3c8dd3b52b8040f4a",
    "quest_paper": "ea98687fe7176d2ffaedc21634a9ba412fa582d43d530fcd98236c8d69fef5a5",
    "quest_readme": "fa698d2a67ef5f369afcc587d039dec1c7773c5f2270f4c7e5b63b6bf6b04d83",
    "quest_utils": "69fd75256a3985ffe23921e89aa498af17b3fd2225318b7951afd7b0a15c55a6",
}
MINFERENCE_COMMIT = "a4eb395f949ea39e871f9bc586d683390692c6be"
QUEST_COMMIT = "01c1623bf9395009520874e989e29f683203b357"
POSITIONS = 1_000_000
LAST_Q = 64
TTFT_SECONDS = 1_800
P100_COUNT = 2
P100_FP16_FLOPS_EACH = 18.7e12
EPYC_IMPOSSIBLE_FLOPS = 742.4e9
EXPECTED_CEILING = 0.21056139043683178
PAIR_FLOPS = 640
INDEX_PAIR_FLOPS = 384


def _pattern_entries(configuration: object) -> list[tuple[str, int, int, float]]:
    if not isinstance(configuration, list) or len(configuration) != 40:
        raise ValueError("MInference GLM-4 configuration layer count mismatch")
    entries: list[tuple[str, int, int, float]] = []
    for layer_index, layer in enumerate(configuration):
        if not isinstance(layer, dict) or set(layer) != {str(i) for i in range(32)}:
            raise ValueError(f"MInference GLM-4 head map mismatch at layer {layer_index}")
        for head in range(32):
            value = layer[str(head)]
            if (
                not isinstance(value, list)
                or len(value) != 4
                or value[0] != "vertical_and_slash"
                or not isinstance(value[1], int)
                or not isinstance(value[2], int)
                or value[1] <= 0
                or value[2] <= 0
                or not isinstance(value[3], (int, float))
                or not math.isfinite(value[3])
                or not 0.0 <= value[3] <= 1.0
            ):
                raise ValueError(f"unsupported or malformed MInference pattern at {layer_index}:{head}")
            entries.append((value[0], value[1], value[2], float(value[3])))
    return entries


def bounded_causal_pairs(positions: int, retained_keys: int) -> int:
    if positions <= 0 or retained_keys <= 0:
        raise ValueError("pair-bound inputs must be positive")
    retained_keys = min(positions, retained_keys)
    return retained_keys * (retained_keys + 1) // 2 + (positions - retained_keys) * retained_keys


def configuration_work(configuration: object, positions: int = POSITIONS) -> dict:
    entries = _pattern_entries(configuration)
    dense_per_head = positions * (positions + 1) // 2
    selected_pairs = 0
    counts: Counter[str] = Counter()
    for pattern, vertical, slash, _score in entries:
        selected_pairs += bounded_causal_pairs(positions, vertical + slash)
        counts[f"{pattern}:{vertical}:{slash}"] += 1
    dense_pairs = dense_per_head * len(entries)
    index_pairs = min(LAST_Q, positions) * positions * len(entries)
    selected_fraction = selected_pairs / dense_pairs
    index_pair_fraction = index_pairs / dense_pairs
    effective_fraction = (
        selected_pairs * PAIR_FLOPS + index_pairs * INDEX_PAIR_FLOPS
    ) / (dense_pairs * PAIR_FLOPS)
    return {
        "released_configuration": "GLM_4_9B_1M_instruct_kv_out_v32_fit_o_best_pattern.json",
        "source_model": "GLM-4-9B-1M",
        "layers": 40,
        "query_heads_per_layer": 32,
        "head_records": len(entries),
        "positions": positions,
        "pattern_counts": dict(sorted(counts.items())),
        "dense_causal_pairs": dense_pairs,
        "favorable_selected_causal_pairs_upper_bound": selected_pairs,
        "favorable_selected_causal_pair_fraction": selected_fraction,
        "online_index_last_q": LAST_Q,
        "online_index_qk_pairs": index_pairs,
        "online_index_qk_pair_fraction_of_dense_pairs": index_pair_fraction,
        "favorable_effective_global_attention_work_fraction_with_index_qk": effective_fraction,
        "pair_flops_qk_plus_value": PAIR_FLOPS,
        "index_pair_flops_qk_only": INDEX_PAIR_FLOPS,
        "overlap_deduplicated": False,
        "kernel_efficiency_granted": "perfect",
    }


def continuation_ceiling(pw0158: dict, pw0161: dict) -> dict:
    attention = pw0158.get("attention_work_ledger", {})
    global_flops = attention.get("global_attention_flops")
    sliding_flops = attention.get("sliding_window_attention_flops")
    if (
        pw0158.get("evidence_class") != "pw0158_million_context_two_p100_attention_ceiling"
        or global_flops != 184_320_184_320_000_000
        or sliding_flops != 204_459_336_007_680
    ):
        raise ValueError("PW-0158 attention authority mismatch")
    rows = {row.get("id"): row for row in pw0161.get("arithmetic", [])}
    v100 = rows.get("v100_pcie_32gb", {})
    modes = {row.get("mode"): row for row in v100.get("modes", [])}
    direct = modes.get("direct_fp32_control", {})
    total_flops = direct.get("mandatory_matrix_plus_attention_flops")
    epyc = direct.get("granted_concurrent_epyc_flops_per_second")
    if (
        pw0161.get("evidence_class") != "pw0161_volta_32gb_complete_system_envelope"
        or pw0161.get("positions") != POSITIONS
        or total_flops != 214_165_790_024_007_680
        or epyc != EPYC_IMPOSSIBLE_FLOPS
    ):
        raise ValueError("PW-0161 complete arithmetic authority mismatch")
    matrix_flops = total_flops - global_flops - sliding_flops
    granted_rate = P100_COUNT * P100_FP16_FLOPS_EACH + epyc
    granted_total = granted_rate * TTFT_SECONDS
    available_global = granted_total - matrix_flops - sliding_flops
    fraction = available_global / global_flops
    if not math.isclose(fraction, EXPECTED_CEILING, rel_tol=1e-15, abs_tol=0.0):
        raise ValueError("global-attention continuation ceiling did not reproduce")
    return {
        "positions": POSITIONS,
        "ttft_seconds": TTFT_SECONDS,
        "two_p100_advertised_fp16_flops_per_second": P100_COUNT * P100_FP16_FLOPS_EACH,
        "granted_impossible_epyc_flops_per_second": epyc,
        "granted_total_flops": granted_total,
        "mandatory_matrix_flops": matrix_flops,
        "mandatory_sliding_attention_flops": sliding_flops,
        "available_global_attention_flops": available_global,
        "ordinary_global_attention_flops": global_flops,
        "maximum_global_attention_work_fraction": fraction,
    }


def validate_source_semantics(paths: dict[str, Path]) -> dict:
    forward = paths["minference_forward"].read_text(errors="strict")
    minference_readme = paths["minference_experiments_readme"].read_text(errors="strict")
    quest_utils = paths["quest_utils"].read_text(errors="strict")
    quest_readme = paths["quest_readme"].read_text(errors="strict")
    required_forward = (
        "last_q = min(64, q_len)",
        "vertical_topk = torch.topk(vertical, vertical_size, -1).indices",
        "slash = (q_len - 1) - torch.topk(slash, slash_size, -1).indices",
        "return vertical_slash_sparse_attention(q, k, v, vertical_topk, slash)",
    )
    if any(fragment not in forward for fragment in required_forward):
        raise ValueError("MInference released selector semantics mismatch")
    if (
        "Offline Kernel-Aware Sparse Pattern Search" not in minference_readme
        or "--is_search" not in minference_readme
        or "single A100 GPU with 80GB of VRAM" not in minference_readme
        or "--context_window 1_000_000" not in minference_readme
    ):
        raise ValueError("MInference experiment authority mismatch")
    if "current version not support Prefill Optimization" not in quest_utils:
        raise ValueError("Quest prefill scope authority mismatch")
    if "Support GQA models" not in quest_readme or "query-aware sparsity" not in quest_readme:
        raise ValueError("Quest released-scope authority mismatch")
    return {
        "minference": {
            "paper": "MInference 1.0: Accelerating Pre-filling for Long-Context LLMs via Dynamic Sparse Attention",
            "arxiv": "2407.02490",
            "canonical_url": "https://arxiv.org/pdf/2407.02490",
            "source_commit": MINFERENCE_COMMIT,
            "changes_prefill": True,
            "changes_decode": False,
            "additional_training_required": False,
            "model_specific_offline_head_pattern_search_required": True,
            "released_online_selector": "last64_query_vertical_and_slash_index_estimation",
            "released_kernel_substrate": "CUDA_Triton_A100",
            "reported_contexts": "128K_to_1M",
            "reported_models": ["LLaMA-3", "GLM-4", "Yi", "Phi-3", "Qwen2"],
            "reported_fidelity": ["InfiniteBench", "RULER", "PG-19", "Needle-in-a-Haystack"],
            "mimo_configuration_released": False,
            "native_image_audio_video_mixed_slices_reported": False,
            "frozen_openrouter_top20_gate_reported": False,
        },
        "quest": {
            "paper": "Quest: Query-Aware Sparsity for Efficient Long-Context LLM Inference",
            "arxiv": "2406.10774",
            "canonical_url": "https://arxiv.org/pdf/2406.10774",
            "source_commit": QUEST_COMMIT,
            "changes_prefill": False,
            "changes_decode": True,
            "released_prefill_path": "dense_FlashInfer_self_attention",
            "released_sparse_selector": "query_aware_KV_page_selection_for_decode",
            "released_kernel_substrate": "CUDA_FlashInfer",
            "mimo_configuration_released": False,
            "native_image_audio_video_mixed_slices_reported": False,
            "frozen_openrouter_top20_gate_reported": False,
        },
    }


def run(paths: dict[str, Path], output: Path, commit: str) -> dict:
    if output.exists():
        raise ValueError(f"refusing to overwrite {output}")
    authenticate_implementation_commit(commit)
    started = time.perf_counter()
    safety = HostSafetyMonitor()
    expected = {"target": TARGET_SHA256, "pw0158": PW0158_SHA256, "pw0161": PW0161_SHA256, **SOURCE_HASHES}
    for name, digest in expected.items():
        if sha256_file(paths[name]) != digest:
            raise ValueError(f"PW-0175 source hash mismatch: {name}")
    target = paths["target"].read_text(errors="strict")
    if (
        "one 1M-token smoke case" not in target
        or "begin generation within 30 minutes" not in target
        or "mean Jensen-Shannon divergence at most 0.01" not in target
        or "USD $500 total" not in target
    ):
        raise ValueError("PW-0175 TARGET authority mismatch")
    for name in ("minference_paper", "quest_paper"):
        if paths[name].read_bytes()[:5] != b"%PDF-":
            raise ValueError(f"{name} is not a PDF")
    safety.checkpoint("target_predecessors_and_primary_sources_authenticated")
    source_scope = validate_source_semantics(paths)
    configuration = json.loads(paths["minference_glm4_pattern"].read_text())
    work = configuration_work(configuration)
    ceiling = continuation_ceiling(
        json.loads(paths["pw0158"].read_text()), json.loads(paths["pw0161"].read_text())
    )
    structural_pass = (
        work["favorable_effective_global_attention_work_fraction_with_index_qk"]
        < ceiling["maximum_global_attention_work_fraction"]
    )
    if not structural_pass:
        raise ValueError("released MInference configuration misses structural continuation gate")
    safety.checkpoint("released_configuration_arithmetic_adjudicated")
    adjudication = {
        "quest_prefill_continuation": "rejected_released_code_leaves_prefill_dense",
        "quest_decode_mechanism": "outside_pw0175_scope_not_rejected_for_decode",
        "minference_released_configuration_directly_reusable_for_mimo": False,
        "minference_structural_continuation_gate_passes": structural_pass,
        "promoted_followup": (
            "mimo_specific_vertical_slash_online_selector_oracle_with_source_derived_head_patterns_and_full_L3_fidelity_ladder"
        ),
        "runtime_default_promoted": False,
        "fidelity_promoted": False,
        "hardware_purchase_authorized": False,
    }
    manifest = {
        "schema_version": 1,
        "evidence_class": "pw0175_structured_sparse_prefill_continuation_audit",
        "commit": commit,
        "source_hashes": expected,
        "source_scope": source_scope,
        "released_configuration_work": work,
        "continuation_ceiling": ceiling,
        "adjudication": adjudication,
        "decision": "promote_mimo_specific_minference_style_oracle;reject_released_quest_as_prefill_repair",
        "purchase_authorized": False,
        "accepted_tokens": 0,
        "A": 0,
        "U": None,
        "performance_claim": None,
        "endpoint_tps": None,
        "limitations": (
            "primary-source and favorable released-configuration arithmetic audit only; no MiMo selector, "
            "source-state fidelity, Metal or P100 kernel, sustained hardware, native modality, hosted distribution, "
            "one-million-token endpoint, accepted-token timing, or Prismwing TPS"
        ),
    }
    safety.release_checkpoint(
        "source_payloads_released",
        ["papers", "source captures", "released configuration", "predecessor manifests"],
    )
    safety.checkpoint("final_service_health")
    manifest["safety"] = safety.evidence()
    manifest["complete_wall_ms"] = (time.perf_counter() - started) * 1000.0
    atomic_write_new(output, canonical_json(manifest))
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    for name in ("target", "pw0158", "pw0161", *SOURCE_HASHES):
        parser.add_argument(name, type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("commit")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    names = ("target", "pw0158", "pw0161", *SOURCE_HASHES)
    report = run({name: getattr(args, name) for name in names}, args.output, args.commit)
    print(canonical_json(report), end="")


if __name__ == "__main__":
    main()
