#!/usr/bin/env python3
"""Run PW-0159's authenticated 12-GB Ampere system envelope."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import time

try:
    from tools.analyze_million_context_p100_attention_bound import attention_ledger
    from tools.analyze_owned_epyc_companion_envelope import authenticate_implementation_commit
    from tools.analyze_pw0116_corpus import sha256_file
    from tools.host_safety import HostSafetyMonitor
    from tools.openrouter_reference import atomic_write_new, canonical_json
except ModuleNotFoundError:
    from analyze_million_context_p100_attention_bound import attention_ledger
    from analyze_owned_epyc_companion_envelope import authenticate_implementation_commit
    from analyze_pw0116_corpus import sha256_file
    from host_safety import HostSafetyMonitor
    from openrouter_reference import atomic_write_new, canonical_json


REVISION = "63651580ca774f8504f676040460aed3e1244ac1"
TARGET_SHA256 = "91fe6e0441bb0a0e1ab0852db60fb575d131b61ff069002c9c333f9b776e4950"
CONFIG_SHA256 = "292a60e74ae9a6d53422b31b21468ce2111c0ab3f7f7a4f4e9c7cd5133b96587"
PW0127_SHA256 = "6b81023921824906fea94e2bd5756e9a8ac2ab3f98411e1bfe62fe26d125e140"
PW0155_SHA256 = "226603fb2b44e1162a038f51bae47520238150f3b26e39e1cf33c7420b88b064"
PW0157_PREFIX4096_SHA256 = "658d2635e8aee4e97ce5a10d7eb1ac347b722f251b663c163814707d3d3f77cc"
PW0158_SHA256 = "3b5b94cae112bee558ec46566ec09652c58bd434c3f47bebd3e0bc7c533fd315"
THROUGHPUT_MODEL_SHA256 = "a914eb9949ae201d109ca2c107088687bf9f3101b67fd17b0dddd5551300c7ad"
AMPERE_PDF_SHA256 = "d4f8c2335959877ed04223301d50469b95a5f9c20d119175b36e8e776a120412"
RTX_SPECS_HTML_SHA256 = "6a187cbd1052ea0a4228f85788797132f438dc09d8f9a459fb385f9792cbfaa6"
MARKET_SHA256 = "973aa06f18fc5f0fdf145cc99d2fbce84f5a81e56a6d8b79b878214de1ecf3a7"
PW0157_COMMIT = "f677893a12fc5631ddcdecf8fc407b7d1178c3f5"
POSITIONS_8K = 8_000
POSITIONS_1M = 1_000_000
TTFT_8K_SECONDS = 15.0
TTFT_1M_SECONDS = 1_800.0
GRANTED_TENSOR_FLOPS = 123e12
MANDATORY_MACS_PER_TOKEN = 14_820_573_184
EXPERT_BYTES = 25_171_968
HBM_BYTES = 12_000_000_000
THREE_ARENA_BYTES = 2_340_993_024
BF16_8K_KV_BYTES = 209_879_040
STORAGE_LANE_BYTES_PER_SECOND = 3.5e9
GPU_BOARD_WATTS = 350
EPYC_TDP_WATTS = 170
PSU_12V_WATTS = 732


def arithmetic_floor(config: dict, positions: int, peak_flops: float = GRANTED_TENSOR_FLOPS) -> dict:
    if peak_flops <= 0:
        raise ValueError("peak FLOPS must be positive")
    attention = attention_ledger(config, positions)
    matrix_flops = positions * 2 * MANDATORY_MACS_PER_TOKEN
    total = attention["mandatory_attention_flops"] + matrix_flops
    seconds = total / peak_flops
    return {
        "positions": positions,
        "mandatory_matrix_flops": matrix_flops,
        "mandatory_attention_flops": attention["mandatory_attention_flops"],
        "mandatory_matrix_plus_attention_flops": total,
        "granted_dense_bf16_tensor_flops_per_second": peak_flops,
        "matrix_plus_attention_floor_seconds": seconds,
    }


def optimistic_hbm_expert_capacity() -> dict:
    available = HBM_BYTES - THREE_ARENA_BYTES - BF16_8K_KV_BYTES
    if available < EXPERT_BYTES:
        raise ValueError("no complete expert fits in optimistic HBM")
    slots = available // EXPERT_BYTES
    return {
        "aggregate_hbm_decimal_bytes": HBM_BYTES,
        "all_non_routed_tensors_streamed_for_free": True,
        "three_arena_bytes": THREE_ARENA_BYTES,
        "exact_bf16_8k_kv_bytes": BF16_8K_KV_BYTES,
        "available_for_complete_experts_bytes": available,
        "complete_expert_slots": slots,
        "expert_cache_bytes": slots * EXPERT_BYTES,
        "unallocated_tail_bytes": available - slots * EXPERT_BYTES,
    }


def storage_lane_scenarios(distinct_records: int, arithmetic_seconds: float) -> list[dict]:
    if distinct_records <= 0 or arithmetic_seconds <= 0:
        raise ValueError("storage scenario inputs must be positive")
    slots = optimistic_hbm_expert_capacity()["complete_expert_slots"]
    streamed_records = max(0, distinct_records - slots)
    streamed_bytes = streamed_records * EXPERT_BYTES
    rows = []
    for lanes in range(1, 5):
        storage_seconds = streamed_bytes / (lanes * STORAGE_LANE_BYTES_PER_SECOND)
        total = storage_seconds + arithmetic_seconds
        rows.append(
            {
                "lanes": lanes,
                "granted_bytes_per_second_per_lane": STORAGE_LANE_BYTES_PER_SECOND,
                "perfect_foresight_preloaded_expert_slots": slots,
                "minimum_streamed_distinct_records": streamed_records,
                "minimum_streamed_source_expert_bytes": streamed_bytes,
                "storage_only_floor_seconds": storage_seconds,
                "matrix_plus_attention_floor_seconds": arithmetic_seconds,
                "serial_8k_ttft_floor_seconds": total,
                "passes_15_second_gate": total <= TTFT_8K_SECONDS,
            }
        )
    return rows


def cost_ledger(market: dict, minimum_lanes: int) -> dict:
    if minimum_lanes <= 0:
        raise ValueError("minimum lanes must be positive")
    components = {row.get("id"): row for row in market.get("components", [])}
    card = components.get("active_rtx3080_12gb", {})
    sold = components.get("sold_rtx3080_12gb", {})
    drives = components.get("nvme_256gb", {})
    adapters = components.get("single_nvme_adapters", {})
    if (
        market.get("evidence_class") != "dated_search_and_listing_transcription_not_purchase_authority"
        or card.get("active") is not True
        or card.get("seller_feedback_count", 0) <= 0
        or sold.get("active") is not False
        or drives.get("quantity") != minimum_lanes
        or adapters.get("quantity") != minimum_lanes
        or drives.get("identity_bound") is not False
        or drives.get("sustained_read_bound") is not False
    ):
        raise ValueError("market authority mismatch")
    storage_and_adapters = (
        drives["subtotal_usd"]
        + drives["rendered_shipping_usd"]
        + adapters["subtotal_usd"]
        + adapters["shipping_usd"]
    )
    active_before_tax = card["delivered_before_tax_usd"] + storage_and_adapters
    sold_before_unknown_delivery_tax = sold["sold_price_usd"] + storage_and_adapters
    return {
        "cap_usd": 500.0,
        "minimum_storage_lanes": minimum_lanes,
        "active_card_delivered_before_tax_usd": card["delivered_before_tax_usd"],
        "minimum_storage_and_adapters_before_tax_usd": storage_and_adapters,
        "active_named_subtotal_before_tax_usd": active_before_tax,
        "active_named_subtotal_over_cap_before_tax_usd": max(0.0, active_before_tax - 500.0),
        "tax_known": False,
        "captured_active_bom_under_cap": active_before_tax <= 500.0 and card.get("tax_usd") is not None,
        "maximum_delivered_card_price_before_unknown_tax_to_reopen_usd": 500.0 - storage_and_adapters,
        "sold_card_historical_subtotal_before_unknown_delivery_and_tax_usd": sold_before_unknown_delivery_tax,
        "sold_card_is_purchase_authority": False,
        "storage_identity_bound": False,
        "storage_sustained_read_bound": False,
    }


def _authenticate(paths: dict[str, Path]) -> tuple[dict, dict, dict, dict]:
    expected = {
        "target": TARGET_SHA256,
        "config": CONFIG_SHA256,
        "pw0127": PW0127_SHA256,
        "pw0155": PW0155_SHA256,
        "pw0157": PW0157_PREFIX4096_SHA256,
        "pw0158": PW0158_SHA256,
        "throughput_model": THROUGHPUT_MODEL_SHA256,
        "ampere_pdf": AMPERE_PDF_SHA256,
        "rtx_specs_html": RTX_SPECS_HTML_SHA256,
        "market": MARKET_SHA256,
    }
    for name, digest in expected.items():
        if sha256_file(paths[name]) != digest:
            raise ValueError(f"PW-0159 source hash mismatch: {name}")
    if paths["ampere_pdf"].read_bytes()[:5] != b"%PDF-":
        raise ValueError("Ampere authority is not a PDF")
    target = paths["target"].read_text(errors="strict")
    if any(value not in target for value in ("USD $500 total", "begin generation within 30 minutes", "at least 1 accepted TPS")):
        raise ValueError("TARGET authority mismatch")
    specs = paths["rtx_specs_html"].read_text(errors="strict")
    if any(value not in specs for value in ("8960", "12 GB", "1.71", "Ampere")):
        raise ValueError("RTX 3080 specification authority mismatch")
    config = json.loads(paths["config"].read_text())
    pw0127 = json.loads(paths["pw0127"].read_text())
    pw0155 = json.loads(paths["pw0155"].read_text())
    route = json.loads(paths["pw0157"].read_text())
    pw0158 = json.loads(paths["pw0158"].read_text())
    throughput = json.loads(paths["throughput_model"].read_text())
    market = json.loads(paths["market"].read_text())
    if (
        pw0127.get("evidence_class") != "pw0127_r720_cpu_arithmetic_ceiling"
        or pw0127.get("revision") != REVISION
        or pw0127.get("arithmetic_ceiling", {}).get("mandatory_macs_per_token") != MANDATORY_MACS_PER_TOKEN
    ):
        raise ValueError("PW-0127 arithmetic authority mismatch")
    if (
        pw0158.get("evidence_class") != "pw0158_million_context_two_p100_attention_ceiling"
        or pw0158.get("attention_work_ledger", {}).get("mandatory_attention_flops") != 184_524_643_656_007_680
    ):
        raise ValueError("PW-0158 attention authority mismatch")
    coverage = route.get("coverage", {})
    if (
        route.get("semantic") != "mimo_target_faithful_prefill_route_coverage_rust_trace"
        or route.get("revision") != REVISION
        or route.get("commit") != PW0157_COMMIT
        or route.get("traced_prefix_positions") != 4096
        or route.get("corpus_positions") != 8000
        or coverage.get("distinct_layer_expert_records") != 4585
        or coverage.get("source_expert_bytes_per_record") != EXPERT_BYTES
        or route.get("accepted_tokens") != 0
        or route.get("performance_claim") is not None
    ):
        raise ValueError("PW-0157 route authority mismatch")
    if (
        pw0155.get("evidence_class") != "pw0155_owned_epyc_installable_bom_prerequisite"
        or pw0155.get("power", {}).get("psu_combined_12v_watts") != PSU_12V_WATTS
        or pw0155.get("power", {}).get("epyc_7351p_tdp_watts") != EPYC_TDP_WATTS
        or pw0155.get("topology", {}).get("logical_lane_topology_supported") is not True
    ):
        raise ValueError("PW-0155 host authority mismatch")
    turbo = throughput.get("constants", {}).get("atomic_turboquant_mimo_kv_compatibility", {})
    if turbo.get("turbo4_bytes_at_max_context") != 7_708_889_088:
        raise ValueError("PW-0020 throughput-model authority mismatch")
    return config, route, market, throughput


def run(paths: dict[str, Path], output: Path, commit: str) -> dict:
    if output.exists():
        raise ValueError(f"refusing to overwrite {output}")
    authenticate_implementation_commit(commit)
    started = time.perf_counter()
    safety = HostSafetyMonitor()
    config, route, market, throughput = _authenticate(paths)
    safety.checkpoint("source_evidence_authenticated")
    one_million = arithmetic_floor(config, POSITIONS_1M)
    eight_k = arithmetic_floor(config, POSITIONS_8K)
    one_million["ttft_limit_seconds"] = TTFT_1M_SECONDS
    one_million["remaining_complete_prefill_budget_seconds"] = TTFT_1M_SECONDS - one_million["matrix_plus_attention_floor_seconds"]
    one_million["passes_favorable_arithmetic_floor"] = one_million["matrix_plus_attention_floor_seconds"] <= TTFT_1M_SECONDS
    capacity = optimistic_hbm_expert_capacity()
    scenarios = storage_lane_scenarios(route["coverage"]["distinct_layer_expert_records"], eight_k["matrix_plus_attention_floor_seconds"])
    minimum_lanes = next((row["lanes"] for row in scenarios if row["passes_15_second_gate"]), None)
    if minimum_lanes != 3:
        raise ValueError("PW-0159 expected three-lane minimum changed")
    costs = cost_ledger(market, minimum_lanes)
    turbo = throughput["constants"]["atomic_turboquant_mimo_kv_compatibility"]
    safety.checkpoint("arithmetic_storage_cost_and_power_ledgers_complete")
    safety.release_checkpoint("source_reports_released", list(paths))
    safety.checkpoint("final_service_health")
    report = {
        "schema_version": 1,
        "evidence_class": "pw0159_ampere_12gb_complete_system_envelope",
        "revision": REVISION,
        "commit": commit,
        "source_hashes": {f"{name}_sha256": sha256_file(path) for name, path in paths.items()},
        "one_million_arithmetic_ceiling": one_million,
        "eight_k_arithmetic_floor": eight_k,
        "optimistic_8k_hbm_capacity": capacity,
        "eight_k_storage_lane_scenarios": scenarios,
        "minimum_nameplate_storage_lanes": minimum_lanes,
        "cost_ledger": costs,
        "power_ledger": {
            "gpu_board_watts": GPU_BOARD_WATTS,
            "epyc_tdp_watts": EPYC_TDP_WATTS,
            "gpu_plus_cpu_watts": GPU_BOARD_WATTS + EPYC_TDP_WATTS,
            "installed_psu_combined_12v_watts": PSU_12V_WATTS,
            "combined_12v_headroom_before_board_drives_fans_and_transients_watts": PSU_12V_WATTS - GPU_BOARD_WATTS - EPYC_TDP_WATTS,
            "formal_project_wall_cap_watts": 1000,
            "physical_install_proven": False,
        },
        "one_million_kv_modes": {
            "exact_bf16_bytes_at_1000000": 23_065_559_040,
            "exact_bf16_exceeds_12gb_hbm_bytes": 11_065_559_040,
            "exact_bf16_plus_three_arenas_exceeds_12gb_hbm_bytes": 13_406_552_064,
            "pw0020_turbo4_bytes_at_1048576": turbo["turbo4_bytes_at_max_context"],
            "pw0020_turbo4_status": turbo["status"],
            "compressed_kv_exactness": "L3_unqualified_for_accumulated_target_fidelity",
        },
        "decision": "reject_captured_active_rtx3080_12gb_three_lane_bom;retain_only_price_triggered_unproven_ampere_architecture",
        "purchase_authorized": False,
        "gates_passed": False,
        "A": 0,
        "accepted_tokens": 0,
        "performance_claim": None,
        "limitations": [
            "123-TFLOPS dense BF16 Tensor rate is a favorable roofline rather than measured MiMo performance",
            "Ampere numerical topology has not passed source or hosted fidelity gates",
            "three 3.5-GB/s lanes are granted nameplates; the captured drives have ambiguous identity and no sustained-read evidence",
            "the active market subtotal exceeds the cap before unknown tax, cables, or other missing installation parts",
            "exact 1M BF16 KV does not fit 12-GB HBM; compressed KV remains an L3 quality candidate",
            "no CUDA runtime, physical fit, cable, cooling, wall-power, endpoint, or accepted-token result exists",
        ],
        "safety": safety.evidence(),
        "complete_wall_ms": (time.perf_counter() - started) * 1000.0,
        "platform": platform.platform(),
    }
    atomic_write_new(output, canonical_json(report))
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    for name in ("target", "config", "pw0127", "pw0155", "pw0157", "pw0158", "throughput-model", "ampere-pdf", "rtx-specs-html", "market"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--commit", required=True)
    args = parser.parse_args()
    paths = {
        "target": args.target,
        "config": args.config,
        "pw0127": args.pw0127,
        "pw0155": args.pw0155,
        "pw0157": args.pw0157,
        "pw0158": args.pw0158,
        "throughput_model": args.throughput_model,
        "ampere_pdf": args.ampere_pdf,
        "rtx_specs_html": args.rtx_specs_html,
        "market": args.market,
    }
    try:
        result = run(paths, args.output, args.commit)
        print(json.dumps({"output": str(args.output), "decision": result["decision"]}))
        return 0
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
        print(json.dumps({"error": str(error)}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
