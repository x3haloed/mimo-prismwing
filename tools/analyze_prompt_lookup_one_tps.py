#!/usr/bin/env python3
"""Analyze PW-0185's lossless prompt-lookup prerequisite."""
from __future__ import annotations
import argparse, json
from pathlib import Path
from tools.openrouter_reference import atomic_write_new, canonical_json
from tools.run_best_rank_real_expert_control import sha256_file

TRACE_SHA256="584d3a8b1b09b12d4f83908be1fa5471b9fd66373500cc56332213928cd0bc3e"
MISS_SECONDS=1.090015

def propose(history:list[int], nmin:int, nmax:int, q:int)->list[int]:
    for n in range(min(nmax,len(history)),nmin-1,-1):
        key=history[-n:]
        for start in range(len(history)-n-1,-1,-1):
            if history[start:start+n] == key and start+n < len(history): return history[start+n:start+n+q]
    return []

def simulate(prompt:list[int],target:list[int],nmin:int,q:int)->dict:
    position=0; accepted=[]; draft_lengths=[]
    while position < len(target):
        draft=propose(prompt+target[:position],nmin,10,q); matched=0
        while matched < len(draft) and position+matched < len(target) and draft[matched] == target[position+matched]: matched+=1
        count=min(matched+1,len(target)-position);accepted.append(count);draft_lengths.append(len(draft));position+=count
    mean=sum(accepted)/len(accepted)
    return {"minimum_ngram":nmin,"q":q,"passes":len(accepted),"mean_A":mean,"maximum_A":max(accepted),"draft_pass_fraction":sum(x>0 for x in draft_lengths)/len(draft_lengths),"impossible_u1_miss_seconds_per_accepted_token":MISS_SECONDS/mean,"necessary_one_tps_pass":mean>MISS_SECONDS}

def run(trace_path:Path,output_path:Path)->dict:
    if output_path.exists(): raise ValueError("PW-0185 refuses to overwrite evidence")
    if sha256_file(trace_path)!=TRACE_SHA256: raise ValueError("PW-0185 trace hash mismatch")
    trace=json.loads(trace_path.read_text());prompt=trace["prompt_token_ids"];target=trace["teacher_forced_token_ids"]
    if len(prompt)!=87 or len(target)!=137: raise ValueError("PW-0185 trace lengths mismatch")
    candidates=[simulate(prompt,target,nmin,q) for nmin in (1,2,3,4,6,8) for q in (2,4,8,16)]
    best=max(candidates,key=lambda x:x["mean_A"]);report={"schema_version":1,"experiment":"PW-0185","mode":"L2_exact_greedy_prompt_lookup_analysis","trace_sha256":TRACE_SHA256,"prompt_tokens":len(prompt),"target_tokens":len(target),"miss_seconds_per_pass_at_impossible_u1":MISS_SECONDS,"candidates":candidates,"best_candidate":best,"decision":"promote_route_union_audit" if best["necessary_one_tps_pass"] else "reject_prompt_lookup_for_one_tps_on_trace","accepted_tokens":0,"performance_claim":"none"}
    atomic_write_new(output_path,canonical_json(report));return report

def main():
    p=argparse.ArgumentParser();p.add_argument("--trace",type=Path,required=True);p.add_argument("--output",type=Path,required=True);a=p.parse_args();print(json.dumps(run(a.trace,a.output),sort_keys=True))
if __name__=="__main__":main()
