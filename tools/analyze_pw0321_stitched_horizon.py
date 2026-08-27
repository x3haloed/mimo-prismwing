#!/usr/bin/env python3
"""Select a real wide-route capture horizon from stitched corrected traces."""
from __future__ import annotations
import argparse, json
from collections import defaultdict
from pathlib import Path
from typing import Any
try:
    from tools.analyze_pw0319_corrected_route_bank import load_rows, greedy_order, sha256_file
    from tools.analyze_pw0320_hybrid_byte_floor import K4_BYTES,SOURCE_BYTES,STORAGE_BYTES_PER_SECOND,oracle_cached_bytes
    from tools.host_safety import HostSafetyMonitor,HostSafetyViolation
    from tools.openrouter_reference import atomic_write_new,canonical_json
    from tools.reproduce_pw0311_k4_expert import verify_clean_commit
except ModuleNotFoundError:
    from analyze_pw0319_corrected_route_bank import load_rows, greedy_order, sha256_file
    from analyze_pw0320_hybrid_byte_floor import K4_BYTES,SOURCE_BYTES,STORAGE_BYTES_PER_SECOND,oracle_cached_bytes
    from host_safety import HostSafetyMonitor,HostSafetyViolation
    from openrouter_reference import atomic_write_new,canonical_json
    from reproduce_pw0311_k4_expert import verify_clean_commit

PW0320_SHA256="de6424aa68d0c65f8f9206a53f61475286bde501873cd4f6ee06299c9b37d7a9"
CACHE=4*1024**3
HORIZONS=(16,32,64)

def group_metric(ids:set[tuple[int,int]],selected:set[tuple[int,int]],observed_a:int,horizon:int)->dict[str,Any]:
    sizes=[K4_BYTES if x in selected else SOURCE_BYTES for x in ids]
    moved=oracle_cached_bytes(sizes,CACHE)
    return {"unique_identities":len(ids),"bytes_after_oracle_cache":moved,"observed_a_sum":observed_a,"structural_a":horizon,"observed_sum_optimistic_tps":observed_a*STORAGE_BYTES_PER_SECOND/moved if moved else float('inf'),"structural_optimistic_tps":horizon*STORAGE_BYTES_PER_SECOND/moved if moved else float('inf')}

def analyze(*,corpus_manifest:Path,pw0320_analysis:Path,output:Path,repo:Path,commit:str)->dict[str,Any]:
    if output.exists(): raise FileExistsError(output)
    verify_clean_commit(repo.resolve(),commit)
    if sha256_file(pw0320_analysis)!=PW0320_SHA256: raise ValueError("PW-0320 analysis mismatch")
    manifest=json.loads(corpus_manifest.read_text()); rows,route_sha,_=load_rows(corpus_manifest)
    windows={int(w['corpus_index']):w for w in manifest['primary_windows']}
    by:dict[int,set[tuple[int,int]]]=defaultdict(set)
    for row in rows: by[row.corpus_index].update(row.identities)
    selected=set(greedy_order(rows,maximum_budget=2048)); curves=[]
    for horizon in HORIZONS:
        count=horizon//8; groups=[]
        for base in (0,8,16,24):
            category=windows[base]['category']
            for start in range(base,base+8,count):
                indices=list(range(start,start+count)); ids=set().union(*(by[i] for i in indices)); observed=sum(int(windows[i]['A']) for i in indices)
                row=group_metric(ids,selected,observed,horizon); row.update(category=category,corpus_indices=indices); groups.append(row)
        structural=all(g['structural_optimistic_tps']>=2 for g in groups)
        observed_categories=sum(any(g['observed_sum_optimistic_tps']>=2 for g in groups if g['category']==c) for c in {g['category'] for g in groups})
        curves.append({"horizon":horizon,"groups":groups,"structural_all_groups_pass":structural,"observed_sum_passing_categories":observed_categories,"capture_gate_pass":structural and observed_categories>=2})
    chosen=next((c['horizon'] for c in curves if c['capture_gate_pass']),None)
    safety=HostSafetyMonitor(); safety.checkpoint('analysis_complete'); safety.release_checkpoint('analysis_released',['route rows','stitched unions']); safety.checkpoint('final_service_health')
    report={"schema_version":1,"experiment_id":"PW-0321","status":"complete","decision":"authorize_real_teacher_forced_capture" if chosen else "reject_real_capture_through_q64","selected_horizon":chosen,"commit":commit,"authority":{"pw0320_analysis_sha256":PW0320_SHA256,"corrected_route_sha256":route_sha},"diagnostic":{"bank_identities":2048,"oracle_cache_bytes":CACHE,"cold_storage_bytes_per_second":STORAGE_BYTES_PER_SECOND,"stitched_windows_are_not_causal_wide_transactions":True},"curves":curves,"safety_snapshots":safety.evidence(),"accepted_tokens":0,"performance_claim":None}
    output.mkdir(parents=True); path=output/'analysis.json'; atomic_write_new(path,canonical_json(report)); print(json.dumps({"output":str(path),"decision":report['decision'],"selected_horizon":chosen})); return report

def main()->int:
    p=argparse.ArgumentParser()
    for n in ('corpus_manifest','pw0320_analysis','output','repo'): p.add_argument('--'+n.replace('_','-'),required=True,type=Path)
    p.add_argument('--commit',required=True)
    try: analyze(**vars(p.parse_args())); return 0
    except (FileExistsError,HostSafetyViolation,KeyError,OSError,RuntimeError,TypeError,ValueError,json.JSONDecodeError) as e: print(json.dumps({'error':str(e)})); return 1
if __name__=='__main__': raise SystemExit(main())
