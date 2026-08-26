# PW-0310 — Modified K4/source live route gate

- Status: complete
- Disposition: conditional
- Date: 2026-08-26
- Owner: Codex
- Runner commit: `044479a9c5b4585bca3baec3c222632d89a6eb95`
- Parent experiment: PW-0309

## Question

Can the authenticated layer-28 K4/source bundle execute behind a route derived
from the installed source checkpoint and the live normalized hidden state,
rather than replaying router identities and weights from PW-0424?

This is an **L3 modified-weights, route-gated causal slice**. The available
bundle contains executable material for exactly eight expert identities at one
layer. It is not an arbitrary-route bank, ordinary decode, or accepted-token
throughput.

## Hypothesis and mechanism

PW-0309 authenticated the layer-28 input but supplied the archived router IDs
and weights directly to Metal. If the local source router independently
reproduces those values, the runtime can derive routing from live checkpoint
state and use the bundle only when the selected identity set is executable.
Every other route must fail closed; identity substitution remains forbidden.

## Protocol and gates

1. Reuse all PW-0309 content-addressed checkpoint, prompt, capture, residual,
   bundle, fixture, kernel, distribution, and Gate 8 authorities unchanged.
2. Recompute the layer-28 MoE input from the authenticated post-attention
   residual and source RMSNorm weights; require bit identity with PW-0424.
3. Execute the installed source layer-28 router over that recomputed input.
4. Require the live eight expert IDs to equal the archived route in execution
   order and require maximum route-weight absolute error at most `5e-7`.
5. Pass only those live-derived IDs and weights into the K4/source Metal
   transaction. Require its 4,096 F32 outputs to remain bit-identical to the
   authenticated PW-0424 candidate.
6. Repeat PW-0309's paired source-weight layers 29–47, logit distribution
   gates, complete diagnostics, and Gate 8 safety checks.

## Decision rule

- Promote a **route-gated embodiment** only if all live-route, candidate-bit,
  distribution, and safety gates pass.
- Reject the implementation if any authority or live route differs.
- Do not promote weights, a runtime default, TPS, or a throughput constant.
- Even on success, require an arbitrary-expert bank and ordinary endpoint
  evidence before describing K4/source as general live routing.

## Claims excluded

- routes whose identity set is not fully present in the bundle;
- any layer other than layer 28;
- ordinary prompt-to-token execution;
- accepted-token TPS or `A/U`;
- full-bank acquisition, cache behavior, multimodal equivalence, hosted
  equivalence, 60-minute stability, or `TARGET.md` completion.

## Result

Raw-001 ran from the exact clean runner commit on the target 16-GiB Apple M1.
The three formerly external inputs are now preserved under
`/Users/chad/Models/mimo-prismwing/evidence/PW-0310/inputs`; their compact
ledger hashes to
`c4abaa421ba58a02a4cbc5b6fb2d6cc8a15bc051b9a58fadcd5fb5e26f1ea4bc`.
The result hashes to
`3eb547a2de3311ecf35d395885e1d56d405622f71836b575cebc6419dfced3f6`;
the compact raw manifest hashes to
`52120788685cdd90f4bde1becf99957c007b340dc376a111b71070d13d8cfcb9`.

The source checkpoint's live router selected experts
`[114,188,93,199,248,41,117,252]` in the authenticated execution order. Its
maximum route-weight absolute difference from the archived independent capture
was `2.9802322e-8`, below the unchanged `5e-7` gate. Passing only those
live-derived IDs and weights to Metal reproduced every one of the 4,096
candidate F32 output bits. The route-only source read was 4,203,520 logical
bytes and no expert identity substitution occurred.

The downstream result reproduces PW-0309: route sets change at layers 32, 34,
37, 39, 40, 41, 44, 45, 46, and 47; layer-47 hidden relative L2 is
`0.0844552885`, final-norm relative L2 `0.120816266`, and logit relative L2
`0.0596321419`. The named external distribution slice passes with the same
token-284 argmax, `0.005353492`-nat source-token error, 20/20 top-set overlap,
and `0.000493366323` projected top-20 JSD.

The layer-28 K4/source transaction took `19.869875` ms wall and `9.700792` ms
GPU. Cold control and candidate tails took `58,665.486` and `58,343.137` ms;
the paired run took `119,226.933` ms and read `15,306,051,584` physical bytes.
Accepted tokens remain zero, so no endpoint TPS or throughput-model constant
changes.

Gate 8 passes across 46 snapshots: minimum free memory 66%, maximum process
footprint 3,095,054,144 bytes, peak RSS 4,327,604,224 bytes, final footprint
3,095,054,144 bytes, zero swap growth, zero new throttled pages, and stable
protected PID sets.

## Decision

Promote only the fail-closed live route gate for this authenticated eight-expert
identity set. PW-0310 supersedes PW-0309's fixture-supplied-routing limitation:
the bundle can participate in a checkpoint-derived route transaction. It does
not supply arbitrary expert coverage.

Kill construction of a full K4 bank as the next Prismwing 50 performance move.
Even ideal generalization would inherit PW-0308's `351.680083`-ms p90 for 47
routed components—about `2.84` routed tokens/s before attention, sampling,
modalities, or endpoint overhead. That is more than 12 times below the user's
useful `34.3`-TPS milestone and more than 17 times below 50 TPS. Retain the
bundle and live gate as a conditional modified-mode research fixture; reopen
bank construction only if a new representation or executor changes that
measured component ceiling.
