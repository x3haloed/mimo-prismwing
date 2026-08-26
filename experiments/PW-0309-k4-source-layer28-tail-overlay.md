# PW-0309 — Modified K4/source layer-28-to-logits causal overlay

- Status: planned
- Disposition: pending paired control/candidate target-M1 evidence
- Date: 2026-08-26
- Owner: Codex
- Starting commit: `1c7c370069f77f6177ea6a610fedb628e95945ca`
- Parent experiment: PW-0308

## Question

When the authenticated PW-0424 K4/source candidate replaces only the routed
output of layer 28 at captured position zero, what error reaches layers 29–47,
the final norm, and logits, and what complete-tail cost and Gate 8 behavior are
observed on the 16-GiB M1?

This is an **L3 modified-weights causal slice**. It is not ordinary live decode,
accepted-token throughput, or evidence that the frozen route generalizes.

## New causal authority

The addendum archive SHA-256 is
`9a99427a59cc850766036f0c0bc10000bdaf9347ce85e37e7f28f5c53c1ff2cb`.
Its 25-file durable ledger is
`/Users/chad/Models/mimo-prismwing/evidence/PW-0309/input-ledger.sha256`,
SHA-256 `e14bf1ee001d8938f23208f5709040f4ea573e90a2b2be65c8cfe68f1da86ba4`.

The decisive new input is the 4,096-value layer-28 position-zero
post-attention residual, SHA-256
`e585e851d1b5651717293d4f287f56c804b6435529b58cd85da7c86c2d168ffd`.
The addendum binds it and the existing PW-0424 normalized MoE input to
position zero of the same MRL-0147 capture. Position zero has no prior causal
keys or values, so the continuation through layers 29–47 can use empty
per-layer K/V caches without depending on the omitted 223 later rows.

## Protocol and falsification gates

1. Authenticate the checkpoint installation receipt, model lock, origin
   record, capture provenance, residual, PW-0424 route fixture, K4/source
   bundle and manifest, and the four Metal kernels before execution.
2. Recompute layer-28 post-attention RMSNorm from the residual and source
   checkpoint. It must match every F32 bit of PW-0424 `input_f32`.
3. Execute the K4/source transaction once and require every output F32 bit to
   match PW-0424 `candidate_routed_f32`.
4. Construct paired layer-28 finals with the same BF16 consumer boundary:
   the control uses authenticated `native_source_routed_f32`; the candidate
   uses the Metal K4/source output.
5. Run both branches independently through source-weight layers 29–47, final
   norm, and LM head. Record per-layer hidden error, route-set stability,
   route-weight drift, final-norm/logit error, argmax identity, source-token
   log-probability error, top-20 overlap, and projected top-20 JSD.
6. Preserve the declared external distribution gates: same argmax, source
   chosen-token absolute log-probability error at most 0.08 nat, projected
   top-20 JSD at most 0.01, and source top-20 overlap at least 18.
7. Record complete wall time, candidate K4 wall/GPU time, process disk bytes,
   logical bytes, batch size one, concurrency one, and cold/warm state. These
   are causal-slice diagnostics, not accepted TPS.
8. Apply Gate 8 at every layer boundary: at least 10% system memory free,
   process footprint and peak RSS at most 8 GiB, zero swap growth, zero new
   throttled pages, protected service PID continuity, explicit page release,
   and final footprint at most 4 GiB.

## Open prediction error resolved before execution

Expected: PW-0424 would be schema version 1 and its embedded
`candidate_routed_sha256` and `native_source_routed_sha256` fields would name
the little-endian F32 serialization of the corresponding JSON arrays.

Observed: the authenticated fixture is schema version 2. The two historical
fields match the archived build record but not an F32LE serialization of the
parsed arrays. The one-off PW-0424 assembler that defined those labels was not
preserved. The first clean-commit preflight rejected before Metal/checkpoint
execution and wrote no result.

Resolution: require the historical fields unchanged as construction
authorities, and independently require reproducible parsed-array F32LE hashes:
input `05a9a3e311a775cda46a343ca0828338c78b96d3a4755d098794a291473b63dd`,
candidate `83be648c5918e1eecd962a9f10c6765dd0ebf94e75b0df81e66f0c316f06ba57`,
and source `01396d596c277bba4fffb277a1acc272c6b5ab75d311644a18c25729c47650ae`.
The whole-fixture SHA-256 remains the primary authority and no gate is
weakened.

## Decision rule

- If causal identity gates and the declared distribution gates pass, retain
  the mixed K4/source representation as a downstream-safe frozen-route
  fallback and proceed to a true live-routing embodiment.
- If causal identity fails, reject the addendum or implementation as invalid.
- If downstream distribution gates fail, kill this mixed representation as a
  fidelity fallback even if its local routed error remains below one percent.
- No result from this experiment promotes K4 weights, changes a target-faithful
  default, or counts accepted tokens.

## Claims excluded

- routes other than the frozen layer-28 position-zero route;
- ordinary prompt-to-token execution;
- accepted-token TPS or `A/U`;
- full-bank acquisition/cache behavior;
- multimodal or hosted-reference equivalence;
- 60-minute stability; and
- `TARGET.md` completion.
