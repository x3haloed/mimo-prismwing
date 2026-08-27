#!/usr/bin/env python3
"""Validate PW-0322 and compute its causal q64 hybrid-storage bound."""
from __future__ import annotations
import argparse, json, math
from pathlib import Path
try:
    from tools.analyze_pw0319_corrected_route_bank import load_rows,greedy_order,sha256_file
    from tools.analyze_pw0320_hybrid_byte_floor import K4_BYTES,SOURCE_BYTES,STORAGE_BYTES_PER_SECOND,oracle_cached_bytes
    from tools.host_safety import HostSafetyMonitor,HostSafetyViolation
    from tools.openrouter_reference import atomic_write_new,canonical_json
    from tools.reproduce_pw0311_k4_expert import verify_clean_commit
except ModuleNotFoundError:
    from analyze_pw0319_corrected_route_bank import load_rows,greedy_order,sha256_file
    from analyze_pw0320_hybrid_byte_floor import K4_BYTES,SOURCE_BYTES,STORAGE_BYTES_PER_SECOND,oracle_cached_bytes
    from host_safety import HostSafetyMonitor,HostSafetyViolation
    from openrouter_reference import atomic_write_new,canonical_json
    from reproduce_pw0311_k4_expert import verify_clean_commit

REPORT_SHA256="ef893b83105009576771b4dcbd98b4f82320b838e58920f5c32011e9a52acb60"
REPORT_COMMIT="be33e94b1dd586fd57243de222ce65b88792444c"
CACHES=(0,2*1024**3,4*1024**3)

def route_union(transaction:dict)->set[tuple[int,int]]:
    traces=transaction['verification_layer_traces']
    if len(traces)!=48 or traces[0]['layer']!=0 or traces[0]['selected_experts_by_position']!=[]: raise ValueError('q64 layer trace authority mismatch')
    identities=set()
    for layer,trace in enumerate(traces[1:],start=1):
        if trace['layer']!=layer or len(trace['selected_experts_by_position'])!=64 or len(trace['route_weights_by_position'])!=64: raise ValueError('q64 routed trace shape mismatch')
        for ids,weights in zip(trace['selected_experts_by_position'],trace['route_weights_by_position']):
            if len(ids)!=8 or len(set(ids))!=8 or len(weights)!=8 or any(not math.isfinite(float(w)) or float(w)<=0 for w in weights) or abs(math.fsum(map(float,weights))-1)>2e-5: raise ValueError('q64 route value mismatch')
            identities.update((layer,int(expert)) for expert in ids)
    return identities

def analyze(*,report:Path,corpus_manifest:Path,output:Path,repo:Path,commit:str)->dict:
    if output.exists(): raise FileExistsError(output)
    verify_clean_commit(repo.resolve(),commit)
    if sha256_file(report)!=REPORT_SHA256: raise ValueError('PW-0322 report mismatch')
    source=json.loads(report.read_text())
    if source.get('commit')!=REPORT_COMMIT or source.get('verifier_width')!=64 or len(source.get('transactions',[]))!=1 or not source.get('route_trace_captured'): raise ValueError('PW-0322 report contract mismatch')
    tx=source['transactions'][0]
    if len(tx['proposal_token_ids'])!=64 or len(tx['posterior_token_ids'])!=64: raise ValueError('q64 token cardinality mismatch')
    accepted=len(tx['verifier_authorized_token_ids'])
    if accepted!=3: raise ValueError('q64 accepted-token authority mismatch')
    identities=route_union(tx)
    rows,route_sha,_=load_rows(corpus_manifest); selected=set(greedy_order(rows,maximum_budget=2048))
    sizes=[K4_BYTES if identity in selected else SOURCE_BYTES for identity in identities]
    curves=[]
    for cache in CACHES:
        moved=oracle_cached_bytes(sizes,cache)
        curves.append({'oracle_cache_bytes':cache,'bytes_after_oracle_cache':moved,'actual_a':accepted,'actual_a_optimistic_tps':accepted*STORAGE_BYTES_PER_SECOND/moved,'structural_a64_optimistic_tps':64*STORAGE_BYTES_PER_SECOND/moved,'required_bytes_per_second_for_two_tps':2*moved/accepted})
    snapshots=source['safety_snapshots']
    if not snapshots or min(s['system_memory_free_percent'] for s in snapshots)<10 or any(s['swap_growth_bytes'] or s['new_throttled_pages'] or any(not pids for pids in s['protected_service_pids'].values()) for s in snapshots): raise ValueError('PW-0322 Gate 8 mismatch')
    safety=HostSafetyMonitor(); safety.checkpoint('analysis_complete'); safety.release_checkpoint('analysis_released',['q64 route union','K4 planner']); safety.checkpoint('final_service_health')
    strongest=curves[-1]
    result={'schema_version':1,'experiment_id':'PW-0322','status':'complete','decision':'authorize_q64_hybrid_runtime' if strongest['actual_a_optimistic_tps']>=2 else 'reject_target_generated_q64_acceptance_branch','analysis_commit':commit,'authority':{'report_sha256':REPORT_SHA256,'report_commit':REPORT_COMMIT,'corrected_route_sha256':route_sha},'capture':{'unique_identities':len(identities),'unique_k4_identities':sum(i in selected for i in identities),'unique_source_identities':sum(i not in selected for i in identities),'actual_a':accepted,'proposal_wall_ms':tx['proposal_wall_ms'],'verification_wall_ms':tx['verification_wall_ms'],'timing_is_diagnostic_only':True},'curves':curves,'capture_safety':{'minimum_free_percent':min(s['system_memory_free_percent'] for s in snapshots),'maximum_peak_resident_bytes':max(s['process_peak_resident_bytes'] for s in snapshots),'maximum_swap_growth_bytes':max(s['swap_growth_bytes'] for s in snapshots),'maximum_new_throttled_pages':max(s['new_throttled_pages'] for s in snapshots)},'analysis_safety':safety.evidence(),'accepted_tokens':0,'performance_claim':None}
    output.mkdir(parents=True); path=output/'analysis.json'; atomic_write_new(path,canonical_json(result)); print(json.dumps({'output':str(path),'decision':result['decision']})); return result

def main()->int:
    p=argparse.ArgumentParser()
    for n in ('report','corpus_manifest','output','repo'): p.add_argument('--'+n.replace('_','-'),required=True,type=Path)
    p.add_argument('--commit',required=True)
    try: analyze(**vars(p.parse_args())); return 0
    except (FileExistsError,HostSafetyViolation,KeyError,OSError,RuntimeError,TypeError,ValueError,json.JSONDecodeError) as e: print(json.dumps({'error':str(e)})); return 1
if __name__=='__main__': raise SystemExit(main())
