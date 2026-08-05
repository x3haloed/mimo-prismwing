# PW-0023 — MiMo global/SWA GQA attention schedule

- Status: proposed
- Disposition: unexecuted
- Date: 2026-08-04
- Owner: Codex with project owner authorization
- Commit and dirty state: based on `d71f0ac`; contract dirty
- Checkpoint/processor/reference hashes: MiMo revision
  `63651580ca774f8504f676040460aed3e1244ac1`; PW-0020 Atomic source lock
- Hardware, OS, compiler, storage, memory pressure: Apple M1; Macmini9,1;
  16 GiB; macOS 26.4.1 (25E253); Swift/Metal runtime compiler; internal SSD
- Related records: PW-0021, PW-0022

## Hypothesis and mechanism

PW-0022's reduction can be scheduled as one threadgroup per MiMo query head,
mapping 64 Q heads onto four global-attention KV heads or eight sliding-window
KV heads. This should preserve scalar GQA semantics and expose complete
attention-core cost per layer without yet conflating projection or MoE work.

## Contract

Target-faithful head counts, K/V dimensions, GQA mapping, and window lengths;
modified Turbo3/Turbo4 KV representation. The candidate passes only if:

1. global mode uses 64 Q heads, four KV heads, K=192 padded to 256, V=128,
   and maps Q head `h` to KV head `h / 16`;
2. SWA mode uses 64 Q heads, eight KV heads, the same K/V dimensions, exactly
   128 cached tokens, and maps Q head `h` to KV head `h / 8`;
3. Metal agrees with an independent scalar reference for every output element
   at relative L2 at most `3e-4` and maximum absolute error at most `5e-4`, all
   64 head guards remain intact, and outputs are finite;
4. Turbo3 and Turbo4 each pass global contexts 128, 1,024, and 8,192 plus SWA
   context 128;
5. each performance run uses batch one, concurrency one, one accepted token,
   10 warm-ups, and 30 measurements, reporting cold and warm wall/GPU time,
   bytes read, hardware, commit, and warm packed buffers. `A` and `U` are not
   applicable to this attention component.

No performance threshold is predeclared for the first complete GQA schedule.
Results are layer-component diagnostics, not endpoint TPS. Passing promotes
only the schedule to the transformer-layer integration branch; real activation
and model-fidelity gates remain mandatory.

## Baseline and candidate

Baseline is a scalar CPU GQA loop using PW-0021's packed dequantization and
attention oracle. Candidate dispatches 64 PW-0022-style threadgroups in one
Metal command. Global and SWA modes have distinct names and evidence.

Raw evidence will be written under
`/Volumes/Elements/mimo-prismwing/evidence/PW-0023`.

## Isolated attribution

Pending.

## End-to-end result

Out of scope; no endpoint TPS claim is permitted.

## Correctness result

Pending.

## Decision

Pending.
