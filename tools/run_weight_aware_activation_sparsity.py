#!/usr/bin/env python3
"""Run PW-0184's real-expert activation-sparsity falsification."""

from __future__ import annotations

import argparse, gc, json, platform, subprocess
from pathlib import Path

import numpy as np
import torch

from tools.generate_real_layer1_expert_oracle import ShardedCheckpoint
from tools.host_safety import HostSafetyMonitor
from tools.openrouter_reference import atomic_write_new, canonical_json
from tools.run_best_rank_real_expert_control import dequant_weight, load_capture, sha256_file, source_linear
from tools.run_input_subvector_code_capacity_oracle import error_metrics
from tools.run_microscaling_fp4_real_expert import _source_expert

REVISION = "63651580ca774f8504f676040460aed3e1244ac1"
VERIFICATION_SHA256 = "9ddc8a99755f04ae2ea3c2484f6dd022d3f3a681b5a72c915ee4de833dbb0d03"
CORPUS_SHA256 = "b9df976876d63c1ffbbe0c70507aea8b939a749ce5b1db27cbca0b5d82cf802e"
LAYER, EXPERT = 46, 28
SPARSITIES = (0.25, 0.40, 0.50)


def sparsify_rows(values: torch.Tensor, importance: np.ndarray, sparsity: float) -> torch.Tensor:
    """Keep an exact stable top-k score set per row and zero everything else."""
    array = values.float().numpy()
    score = np.abs(array) * np.asarray(importance, dtype=np.float32)[None, :]
    keep = array.shape[1] - int(round(array.shape[1] * sparsity))
    order = np.argsort(-score, axis=1, kind="stable")[:, :keep]
    result = np.zeros_like(array)
    np.put_along_axis(result, order, np.take_along_axis(array, order, axis=1), axis=1)
    return torch.from_numpy(result).to(torch.bfloat16)


def candidate(weights: dict[str, np.ndarray], values: np.ndarray, sparsity: float, weight_aware: bool):
    x = torch.from_numpy(np.asarray(values, dtype=np.float32).copy()).to(torch.bfloat16)
    if weight_aware:
        input_importance = np.sqrt((np.square(weights["gate"]).mean(0) + np.square(weights["up"]).mean(0)) / 2)
        down_importance = np.sqrt(np.square(weights["down"]).mean(0))
    else:
        input_importance = np.ones(x.shape[1], dtype=np.float32); down_importance = np.ones(weights["down"].shape[1], dtype=np.float32)
    sparse_x = sparsify_rows(x, input_importance, sparsity)
    gate = source_linear(weights["gate"], sparse_x); up = source_linear(weights["up"], sparse_x)
    hidden = (torch.nn.functional.silu(gate) * up).to(torch.bfloat16)
    sparse_hidden = sparsify_rows(hidden, down_importance, sparsity)
    down = source_linear(weights["down"], sparse_hidden)
    return gate.float().numpy(), up.float().numpy(), down.float().numpy()


def run(checkpoint_root: Path, verification_path: Path, corpus_path: Path, output_path: Path) -> dict:
    if output_path.exists(): raise ValueError("PW-0184 refuses to overwrite evidence")
    if sha256_file(verification_path) != VERIFICATION_SHA256 or sha256_file(corpus_path) != CORPUS_SHA256: raise ValueError("PW-0184 authority hash mismatch")
    corpus=json.loads(corpus_path.read_text())
    if corpus.get("revision") != REVISION: raise ValueError("PW-0184 revision mismatch")
    safety=HostSafetyMonitor(); authority=next(row for row in corpus["layers"] if row["layer"] == LAYER)
    inputs=load_capture(corpus_path.parent, authority["captures"]["moe_input"]); expected_all=load_capture(corpus_path.parent, authority["captures"]["expert_down"])
    offset=0
    for schedule in authority["expert_schedule"]:
        if schedule["expert"] == EXPERT: break
        offset += len(schedule["positions"])
    local=[i for i,p in enumerate(schedule["positions"]) if 112 <= p < 168]; positions=[schedule["positions"][i] for i in local]
    if positions != list(range(112,168)): raise ValueError("PW-0184 validation identity mismatch")
    expected=np.asarray(expected_all[[offset+i for i in local]],dtype=np.float32).copy(); checkpoint=ShardedCheckpoint(checkpoint_root,verification_path)
    prefix=f"model.layers.{LAYER}.mlp.experts.{EXPERT}"; weights={name:dequant_weight(checkpoint,f"{prefix}.{name}_proj.weight") for name in ("gate","up","down")}
    source=_source_expert(weights,np.asarray(inputs[positions])); source_control=error_metrics(source[2],expected)
    if source_control["relative_l2"] != 0: raise ValueError("PW-0184 source control failed")
    reports={}
    for rule in ("magnitude","weight_aware"):
        for sparsity in SPARSITIES:
            actual=candidate(weights,np.asarray(inputs[positions]),sparsity,rule=="weight_aware")
            validation={"gate":error_metrics(actual[0],source[0]),"up":error_metrics(actual[1],source[1]),"complete_expert":error_metrics(actual[2],expected)}
            numerical=validation["complete_expert"]["relative_l2"] <= .02 and validation["complete_expert"]["maximum_row_relative_l2"] <= .05 and validation["gate"]["relative_l2"] <= .02 and validation["up"]["relative_l2"] <= .02
            name=f"{rule}_{int(sparsity*100)}"; reports[name]={"rule":rule,"sparsity":sparsity,"source_weight_column_traffic_ratio":1-sparsity,"validation":validation,"numerical_pass":numerical}
            safety.checkpoint(name+"_complete"); del actual; gc.collect()
    passing=[name for name,row in reports.items() if row["sparsity"] >= .25 and row["numerical_pass"]]
    report={"schema_version":1,"experiment":"PW-0184","mode":"L3_shadow_source_weight_activation_sparsity","revision":REVISION,"layer":LAYER,"expert":EXPERT,"pilot_holdout_unsealed":False,"batch_size":1,"concurrency":1,"accepted_tokens":0,"A":0,"U":0,"source_control":source_control,"candidates":reports,"passing_candidates":passing,"decision":"promote_routed_layer_and_locality" if passing else "reject_direct_activation_channel_deletion","implementation":{"commit":subprocess.run(["git","rev-parse","HEAD"],check=True,capture_output=True,text=True).stdout.strip(),"dirty":bool(subprocess.run(["git","status","--porcelain"],check=True,capture_output=True,text=True).stdout)},"hardware":{"machine":platform.machine(),"platform":platform.platform()},"software":{"python":platform.python_version()}}
    del weights,source;gc.collect();safety.release_checkpoint("experiment_released",["source weights","validation activations"]);safety.checkpoint("final_service_health");report["host_safety"]=[x.to_dict() for x in safety.snapshots]
    atomic_write_new(output_path,canonical_json(report));return report


def main():
    p=argparse.ArgumentParser();p.add_argument("--checkpoint-root",type=Path,required=True);p.add_argument("--verification",type=Path,required=True);p.add_argument("--corpus",type=Path,required=True);p.add_argument("--output",type=Path,required=True);a=p.parse_args();print(json.dumps(run(a.checkpoint_root,a.verification,a.corpus,a.output),sort_keys=True))
if __name__ == "__main__": main()
