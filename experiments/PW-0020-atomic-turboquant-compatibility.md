# PW-0020 — Atomic TurboQuant MiMo compatibility audit

- Status: proposed
- Disposition: unexecuted
- Date: 2026-08-04
- Owner: Codex with project owner authorization
- Commit and dirty state: based on `ce7ee40`; contract and source lock dirty
- Checkpoint/processor/reference hashes: MiMo revision
  `63651580ca774f8504f676040460aed3e1244ac1`; Atomic TurboQuant revision
  `074bf826e1b06005a51737d29387e36657f41bf7`; see
  `spec/atomic-turboquant.lock.json`
- Hardware, OS, compiler, storage, memory pressure: Apple M1; Macmini9,1;
  16 GiB; macOS 26.4.1 (25E253); Apple clang 21.0.0; test binaries and
  generated evidence on internal SSD; live checkpoint download excluded
- Related records: PW-0002, PW-0019

## Hypothesis and mechanism

The pinned Atomic llama.cpp fork contains a mechanically compatible low-bit KV
path for MiMo's asymmetric 192-wide K and 128-wide V heads. Its WHT rotation
should preserve unquantized inner products, its packed layouts should materially
reduce the one-million-token global KV footprint, and its Metal source should
contain the exact `dk192_dv128` flash-attention specializations needed here.

This is a source and component compatibility audit, not a fidelity promotion.
The fork implements a practical WHT/Lloyd-Max scheme whose current Turbo4
default is 4-bit PolarQuant; that is not identical to every construction in the
TurboQuant paper.

## Contract

Target-faithful shapes and attention semantics; modified KV representation.
The candidate passes this compatibility audit only if all conditions hold:

1. the source revision and every locked file hash match;
2. Apple Clang compiles the pinned C implementation and its upstream test exits
   zero without non-finite output;
3. a deterministic Prismwing fixture uses K dimension 192 padded to 256, V
   dimension 128, causal softmax, and weighted V reconstruction; the same WHT
   applied to padded Q and K preserves the unquantized dot product within
   `2e-5` relative error;
4. compiled layouts are exactly 34, 50, and 68 bytes per 128 values for
   Turbo2, Turbo3, and Turbo4; all three candidates produce finite deterministic
   score and output diagnostics;
5. source inspection finds both graph-side Q rotation/KV padding and Metal
   flash-attention specializations for `dk192_dv128` for each candidate type;
6. no source assumption silently removes global history, changes the 128-token
   sliding window, drops KV heads, or truncates the 192 logical K dimensions.

Any failure rejects direct reuse until repaired. Passing permits only a minimal
isolated port to advance to accelerated parity and real-activation fidelity
tests. Quantized score/output errors are diagnostics in this audit and cannot
promote a fidelity default. The audit must report source compilation separately
from Metal runtime validation; source presence is not proof that a Metal kernel
builds or runs.

## Baseline and candidate

Baseline is FP32 attention in the deterministic fixture and FP16 for storage
accounting. Candidates are the pinned fork's Turbo2, Turbo3, and Turbo4 row
formats. Commands and raw outputs will be recorded under
`/Volumes/Elements/mimo-prismwing/evidence/PW-0020`.

MiMo storage accounting uses nine persistent global-attention layers, 39
sliding-window layers capped at 128 tokens, four KV heads in global layers,
eight KV heads in sliding layers, K=192, and V=128. Candidate K is charged for
the fork's required padding to 256.

## Isolated attribution

Pending.

## End-to-end result

Out of scope; no endpoint TPS claim is permitted from this audit.

## Correctness result

Pending.

## Decision

Pending.
