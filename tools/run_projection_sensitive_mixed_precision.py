#!/usr/bin/env python3
"""Run PW-0183's projection-sensitive mixed-precision controls."""

from __future__ import annotations

import argparse, gc, json, platform, subprocess, time
from pathlib import Path

import mlx.core as mx
import numpy as np

from tools.generate_real_layer1_expert_oracle import ShardedCheckpoint
from tools.host_safety import HostSafetyMonitor
from tools.openrouter_reference import atomic_write_new, canonical_json
from tools.run_best_rank_real_expert_control import dequant_weight, load_capture, sha256_file
from tools.run_input_subvector_code_capacity_oracle import error_metrics
from tools.run_microscaling_fp4_real_expert import _source_expert, quantize_projection

REVISION = "63651580ca774f8504f676040460aed3e1244ac1"
VERIFICATION_SHA256 = "9ddc8a99755f04ae2ea3c2484f6dd022d3f3a681b5a72c915ee4de833dbb0d03"
CORPUS_SHA256 = "b9df976876d63c1ffbbe0c70507aea8b939a749ce5b1db27cbca0b5d82cf802e"
PW0182_SHA256 = "db62501ba622bb09a18db327c06cc883ab51ec978836f6d0dc703ab72ebbf485"
LAYER, EXPERT = 46, 28
INT4_EXPERT_BYTES = 13_369_344
CANDIDATES = {"affine_3_3_6": (3, 3, 6), "affine_4_4_6": (4, 4, 6), "affine_3_3_8": (3, 3, 8), "affine_4_4_8": (4, 4, 8)}

def projection_configs(bits: tuple[int, int, int]) -> dict[str, dict]:
    return {name: {"mode": "affine", "group_size": 128, "bits": bit} for name, bit in zip(("gate", "up", "down"), bits)}

def mixed_expert(values: np.ndarray, projections: dict, configs: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = mx.array(np.asarray(values, dtype=np.float16))
    from tools.run_microscaling_fp4_real_expert import quantized_linear
    gate = quantized_linear(x, projections["gate"], configs["gate"]); up = quantized_linear(x, projections["up"], configs["up"])
    hidden = mx.sigmoid(gate) * gate * up
    down = quantized_linear(hidden, projections["down"], configs["down"])
    mx.eval(gate, up, down)
    return np.asarray(gate, dtype=np.float32), np.asarray(up, dtype=np.float32), np.asarray(down, dtype=np.float32)

def array_bytes(arrays: tuple) -> int:
    return sum(int(array.nbytes) for array in arrays if array is not None)

def run(checkpoint_root: Path, verification_path: Path, corpus_path: Path, pw0182_path: Path, output_path: Path) -> dict:
    if output_path.exists(): raise ValueError("PW-0183 refuses to overwrite evidence")
    if sha256_file(verification_path) != VERIFICATION_SHA256 or sha256_file(corpus_path) != CORPUS_SHA256: raise ValueError("PW-0183 authority hash mismatch")
    if sha256_file(pw0182_path) != PW0182_SHA256: raise ValueError("PW-0183 prior hash mismatch")
    prior = json.loads(pw0182_path.read_text())
    if prior.get("decision") != "reject_tested_fp4_modes" or prior.get("pilot_holdout_unsealed"): raise ValueError("PW-0183 prior identity mismatch")
    safety = HostSafetyMonitor(); corpus = json.loads(corpus_path.read_text())
    if corpus.get("revision") != REVISION: raise ValueError("PW-0183 corpus revision mismatch")
    authority = next(row for row in corpus["layers"] if row["layer"] == LAYER)
    inputs = load_capture(corpus_path.parent, authority["captures"]["moe_input"]); expected_all = load_capture(corpus_path.parent, authority["captures"]["expert_down"])
    offset = 0
    for schedule in authority["expert_schedule"]:
        if schedule["expert"] == EXPERT: break
        offset += len(schedule["positions"])
    local = [i for i, p in enumerate(schedule["positions"]) if 112 <= p < 168]; positions = [schedule["positions"][i] for i in local]
    if positions != list(range(112, 168)): raise ValueError("PW-0183 validation identity mismatch")
    expected = np.asarray(expected_all[[offset + i for i in local]], dtype=np.float32).copy()
    checkpoint = ShardedCheckpoint(checkpoint_root, verification_path); prefix = f"model.layers.{LAYER}.mlp.experts.{EXPERT}"
    weights = {name: dequant_weight(checkpoint, f"{prefix}.{name}_proj.weight") for name in ("gate", "up", "down")}
    source = _source_expert(weights, np.asarray(inputs[positions])); source_control = error_metrics(source[2], expected)
    if source_control["relative_l2"] != 0: raise ValueError("PW-0183 source control failed")
    reports = {}
    for candidate_name, bits in CANDIDATES.items():
        configs = projection_configs(bits); projections = {name: quantize_projection(weights[name], configs[name]) for name in configs}
        actual = mixed_expert(np.asarray(inputs[positions]), projections, configs)
        validation = {"gate": error_metrics(actual[0], source[0]), "up": error_metrics(actual[1], source[1]), "complete_expert": error_metrics(actual[2], expected)}
        packed_bytes = sum(array_bytes(arrays) for arrays in projections.values()); sample = np.asarray(inputs[positions[0]:positions[0]+1])
        for _ in range(10): mixed_expert(sample, projections, configs)
        timings=[]
        for _ in range(50):
            started=time.perf_counter(); mixed_expert(sample, projections, configs); timings.append((time.perf_counter()-started)*1000)
        physical = packed_bytes <= INT4_EXPERT_BYTES*1.05 and float(np.median(timings)) <= .75
        numerical = validation["complete_expert"]["relative_l2"] <= .02 and validation["complete_expert"]["maximum_row_relative_l2"] <= .05 and validation["gate"]["relative_l2"] <= .02 and validation["up"]["relative_l2"] <= .02
        reports[candidate_name] = {"bits_gate_up_down": list(bits), "packed_expert_bytes": packed_bytes, "to_int4_ratio": packed_bytes/INT4_EXPERT_BYTES, "validation": validation, "warm_median_ms": float(np.median(timings)), "warm_p95_ms": float(np.quantile(timings,.95)), "gates": {"physical_pass": physical, "numerical_pass": numerical}}
        safety.checkpoint(candidate_name+"_complete"); del projections, actual; gc.collect(); mx.clear_cache()
    passing=[name for name,row in reports.items() if row["gates"]["physical_pass"] and row["gates"]["numerical_pass"]]
    report={"schema_version":1,"experiment":"PW-0183","mode":"L3_projection_sensitive_mixed_affine","revision":REVISION,"layer":LAYER,"expert":EXPERT,"pilot_holdout_unsealed":False,"batch_size":1,"concurrency":1,"accepted_tokens":0,"A":0,"U":0,"source_control":source_control,"candidates":reports,"passing_candidates":passing,"decision":"promote_all_validation_and_fused_gather" if passing else "reject_projection_sensitive_mixed_precision","hardware":{"machine":platform.machine(),"platform":platform.platform()},"software":{"mlx":"0.31.2","python":platform.python_version()},"implementation":{"commit":subprocess.run(["git","rev-parse","HEAD"],check=True,capture_output=True,text=True).stdout.strip(),"dirty":bool(subprocess.run(["git","status","--porcelain"],check=True,capture_output=True,text=True).stdout)}}
    del weights,source;gc.collect();mx.clear_cache();safety.release_checkpoint("experiment_released",["source weights","mixed quantized weights","validation activations"]);safety.checkpoint("final_service_health");report["host_safety"]=[x.to_dict() for x in safety.snapshots]
    atomic_write_new(output_path,canonical_json(report));return report

def main():
    p=argparse.ArgumentParser();p.add_argument("--checkpoint-root",type=Path,required=True);p.add_argument("--verification",type=Path,required=True);p.add_argument("--corpus",type=Path,required=True);p.add_argument("--pw0182",type=Path,required=True);p.add_argument("--output",type=Path,required=True);a=p.parse_args();print(json.dumps(run(a.checkpoint_root,a.verification,a.corpus,a.pw0182,a.output),sort_keys=True))
if __name__=="__main__":main()
