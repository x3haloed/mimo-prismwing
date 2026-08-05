# PW-0009 — DFlash checkpoint compatibility audit

- Status: complete
- Disposition: scope-decision
- Date: 2026-08-04
- Owner: Codex with project owner authorization
- Commit and dirty state: based on `3d5da28`; dirty audit implementation
- Checkpoint/processor/reference hashes: base revision
  `63651580ca774f8504f676040460aed3e1244ac1`; DFlash revision
  `1f58446181abcaa01030fdbde835fbd38ae9a2b1`; DFlash lock SHA-256
  `9c8d6a8911524163a7817a19082e32a255a5134d33aaed079132bcc66513ac26`
- Hardware, OS, compiler, storage, memory pressure: Macmini9,1; Apple M1;
  16 GiB; macOS 26.4.1 (25E253); remote HTTP range reads; no material
  memory pressure
- Related records: PW-0001, PW-0002, PW-0008

## Hypothesis and mechanism

The official DFlash draft could be treated as an L2 accelerator for the pinned
MiMo-V2.5 target if exact verification preserves accepted greedy output. Before
assuming its published acceptance, determine whether its bundled target is the
same checkpoint as Prismwing's target and inspect the shipped correction path.

## Contract

Target-faithful audit. Do not silently substitute the bundled DFlash target.
Differing file hashes alone are insufficient because safetensors metadata can
change a file hash without changing tensor payloads. A single unequal payload
sample proves inequality; equal samples do not prove full identity. The shipped
generation algorithm qualifies as L2 only for modes where its acceptance and
correction preserve target output.

## Baseline and candidate

Baseline is `XiaomiMiMo/MiMo-V2.5` at the pinned base revision. Candidate is
`XiaomiMiMo/MiMo-V2.5-DFlash` at its pinned revision. The candidate lock records
39 files totaling 318,647,669,051 bytes, including the 2,936,121,080-byte draft.

Reproduction command:

```sh
python3 tools/remote_safetensors_audit.py \
  --left-lock spec/model.lock.json \
  --right-lock spec/dflash-model.lock.json \
  --path-prefix model_pp \
  --sample-bytes 65536 \
  --output /Volumes/Elements/mimo-prismwing/evidence/PW-0009/target-payload-audit.json
```

## Isolated attribution

The two target indices have the same 73,081 tensor assignments; their only
textual difference is a trailing newline. Each DFlash target shard is 32 bytes
larger because its safetensors header adds `{"format":"pt"}`. After accounting
for that header, all 16 shard payload sizes are exactly equal.

The deterministic remote audit sampled 64 KiB at the start, middle, and end of
each payload. All 48 paired samples differed. This proves that the bundled
DFlash target weights are not byte-identical to the pinned base target; it is
not merely a header repack.

## End-to-end result

No endpoint speed is claimed. The DFlash source proposes blocks of eight,
verifies them with the target, accepts the consecutive matching prefix, and
inserts the target's first mismatch token. Acceptance against Prismwing's
different pinned target remains unmeasured and cannot be imported from the
published bundle.

## Correctness result

For temperature zero, the source uses argmax for both draft and target, so the
accepted greedy sequence is target-preserving under a correct local target
implementation. For positive temperature, it independently samples draft and
target tokens and has no speculative-sampling rejection correction. Therefore
the shipped positive-temperature path is not proven target-distribution
preserving and must not be labeled L2.

Raw evidence:

- External path:
  `/Volumes/Elements/mimo-prismwing/evidence/PW-0009/target-payload-audit.json`
- SHA-256:
  `a42a6ac80d329cea403ee58cc7b017fcbd7fc044544e92fe16d2cc9b2e3a2e14`
- DFlash source SHA-256:
  `da5ab1738b954800950405131f1d1d97c3345f37e32676d511d3a25dfddd9d75`

## Decision

Keep the base checkpoint authoritative. Do not download or substitute the
315-GB DFlash target. Retain the 2.94-GB DFlash draft as a conditional greedy
candidate and measure its actual `A`, `U`, draft time, and verifier time against
the pinned base target after the local target path exists. Positive-temperature
use requires a mathematically correct speculative-sampling algorithm first.

Confidence is high that the two target payloads differ and high for the greedy
source analysis. No claim is made about DFlash acceptance across these targets.
