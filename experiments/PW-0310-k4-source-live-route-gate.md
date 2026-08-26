# PW-0310 — Modified K4/source live route gate

- Status: in progress
- Disposition: pending
- Date: 2026-08-26
- Owner: Codex
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

