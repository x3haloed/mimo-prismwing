# PW-0099 — Sparse BF16 boundary repair

- Status: complete
- Disposition: conditional
- Date: 2026-08-06
- Owner: Codex with project owner authorization
- Commit and dirty state: candidate
  `ea7f042fd45e740ab4005aafb2337c3ca7eb5366`; frozen uncorrected control
  `e8a12e6b68bed2e47adffc0674e02dfa921110a7`; clean tree
- Checkpoint/reference hashes: MiMo revision
  `63651580ca774f8504f676040460aed3e1244ac1`; PW-0098 oracle manifest
  `5884217fbc804a7a34bc76534b985eb7e6fe90f5e49e27e6328bda8584607cda`;
  PW-0098 rejected evidence
  `1ffa33d7a7f4d2742e142db65f4267e5ee7f9691c7c6666dbad4e140aa30c3c0`
- Hardware/runtime: Apple M1 shared 16 GiB; bounded Metal plus source-exact
  sparse CPU correction
- Related records: PW-0096, PW-0097, PW-0098

## Hypothesis and mechanism

PW-0098 localizes routed-row numerical failure to expert 182 while seven
controls already pass. Capture expert 182's gate, up, SwiGLU, and down BF16
boundaries independently. Compare the pre-round Metal F32 value to its nearest
BF16 midpoint and derive a conservative uncertainty interval from observed
projection error plus an explicit margin. Recompute only uncertain output rows
from source FP8 weights using the source-exact CPU matrix reduction, replace
their BF16 values, and propagate normally.

The uncertainty predicate must be value-derived and fixed before the final
timed runs. It may not encode expert IDs, row IDs, expected values, oracle
hashes, or route outputs. Expert 182 is the discovery fixture; at least the
seven PW-0098 experts and PW-0097 expert 32 are mandatory holdouts.

## Gates

First add independent gate/up/SwiGLU/down captures and report exactly where
expert 182 first diverges. Preserve raw hashes and reject any unexplained
upstream mismatch. Add tiny fixtures for BF16 midpoint distance, conservative
error intervals, uncertain-row selection, sparse row decode, source-exact
reduction, replacement order, empty/full repair sets, and fail-closed shapes.

On discovery and every holdout expert, repaired final expert output must meet
PW-0097's unchanged gates: relative L2 at most `5e-4`, maximum absolute error
at most `2e-2`, and BF16 identity at least 99%. The complete PW-0098 routed row
must then meet its unchanged output and route gates in two clean processes and
produce byte-identical output. Report the uncertain/recomputed row count and
fraction separately for gate, up, and down; no oracle-derived row selection is
allowed during candidate execution.

Include uncertainty selection, sparse source decode/reduction, and replacement
inside five-warmup/30-measurement timing. Both routed-row medians must remain at
most 100 ms and at least 10x faster than the 3,180 ms CPU attribution. Record
logical bytes, decoded bytes, batch 1, concurrency 1, accepted tokens 0,
`A=0`, `U=8`, commit, cold/warm state, and interleaved uncorrected controls.

Apply the complete Gate 8 contract at compile, warmup, timed-series, and
post-release boundaries. This remains a component experiment; passing only
authorizes complete-token candidate integration and is not accepted TPS.

## Result

Independent stage captures under
`/Users/chad/Models/mimo-prismwing/evidence/PW-0099/reference-001` localize the
first consequential divergence to expert 182's up projection at row 985. The
Metal pre-round value is one F32 ULP from a BF16 midpoint. Rounding it to the
opposite BF16 neighbor changes the subsequent dynamic-FP8 SwiGLU group maximum
and fans out to 1,104 down-projection differences. The oracle manifest hashes
to `75d5a55d954a1fe146c4ec62e0990a9f1b2fcdfe3ac00f80928ee3a148bb4317`.
The decisive diagnostic hashes to
`048e1944960c808585c0672c2561399ed78b17a8cf5c98afc9d58216123207c2`.

The implemented predicate repairs any projection value whose low F32 mantissa
bits place it within four ULPs of a BF16 midpoint. It is applied uniformly to
gate, up, and down projections and contains no expert, row, route, expected
value, or oracle identity. A selected row is decoded from source FP8 and
reduced with the source-exact Accelerate SGEMM path before BF16 replacement.
The complete routed row selects three gate, three up, and nine down rows,
decoding 172,032 weight bytes out of 205,603,840 logical source/I/O bytes per
execution. Its largest resident tensor buffer remains 8,390,656 bytes.

Two initial clean processes produced the same output hash
`77436d4ffc8a112d96f18275fbcc47097a67f2ca18a937c06726b736edc0d2a1`
at 55.404/55.520 ms medians. The final interleaved sequence then ran frozen
uncorrected control A, repaired candidate A, frozen control B, and repaired
candidate B. Controls reproduced the PW-0098 failure at 55.0859/55.6005 ms
with 92.2363% BF16 identity and `9.59021e-4` relative L2. Repaired candidates
ran at 55.1810/55.5958 ms and both achieved:

- `5.25654e-5` relative L2, `2.98023e-8` maximum absolute error, and
  99.9756% BF16 identity;
- exact source route order and bit-exact route weights;
- identical output bytes and repair counts;
- 57.63x/57.20x diagnostic speedup over PW-0096's 3,180 ms attribution.

The candidate report hashes are
`ebb2403f7d88585db10c7ac67b64b0a39bba88c4052b83d8d07d620aca663317`
and `381ea86b44e9090eb46a0f36da126da9b17cc7b59e1275e5543883d80500371f`;
the paired control records hash to
`c919ec259101294fad8f5ee6d1f27e8ec2a6ac8ebce9c2145ee2f85503c8ea36`
and `96a98fc3d36a59080fe11eb613abbfa5c0bda11b5f3d04655455d07d372a70b9`.

PW-0097 expert 32 is an independent holdout. It selected no gate/up and two
down repairs, then passed with `1.78339e-5` relative L2, `2.98023e-8` maximum
error, 99.9756% BF16 identity, and a 6.7227 ms median. Its report hashes to
`e0dfc7a53902ece725641278a5df9817c787e730a9f712d7272a4a739891a842`.
The seven non-discovery PW-0098 experts remain covered inside every routed-row
run and their independently hashed stage captures are validated before a
candidate report can be emitted.

All final runs retained 79% free memory, caused zero swap growth and zero new
throttled pages, and preserved ChatGPT, WindowServer, nxnode, and syncthing.
The largest candidate peak was 257,851,392 bytes and its post-release footprint
was 30,705,600 bytes, comfortably inside Gate 8. Batch is one, concurrency is
one, accepted tokens are zero, `A=0`, and `U=8`. Timings are warm OS cache with
five process-local warmups and 30 measurements; tensor install, sparse repair,
dispatch/wait/readback, and release are all included.
The updated throughput model hashes to
`8743783475e75c6444b316b9864bdc7b3efc8e0c7ef8bb99e44f211f396670dd`.

## Decision

Promote the value-derived sparse BF16 boundary repair conditionally as the
complete-token Metal expert candidate. It repairs a measured source-versus-
Metal reduction-boundary hazard, generalizes to the independent expert-32
holdout, preserves routed-row performance under interleaving, and clears every
component correctness and safety gate.

Do not enable it as an endpoint default or report its component rate as TPS.
The next record must integrate this bounded executor into the real retained-
cache token path, preserve the target-faithful/modified distinction, apply the
complete correctness ladder and Gate 8 phase stops, and measure end to end.
