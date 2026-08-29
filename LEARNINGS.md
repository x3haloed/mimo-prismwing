# Prismwing decision frontier

This is the smallest current state needed to choose the next useful experiment.
It is not the experiment history. Immutable observations, failed attempts,
reversals, commands, and evidence hashes live in [`experiments/`](experiments/);
machine-readable constants and provenance live in
[`spec/throughput-model.json`](spec/throughput-model.json).

Read a branch conclusion as scoped to its named premises. “Rejected” never
means universally impossible, and a component result never implies endpoint
throughput or fidelity.

## Outcome

Build the full local XiaomiMiMo/MiMo-V2.5 system defined by
[`TARGET.md`](TARGET.md), without crossing [`RED_LINES.md`](RED_LINES.md). The
primary completion gate remains a full-capability, near-equivalent native
runtime passing every target gate, including reproducible batch-one decode at
50 median accepted TPS on the 16 GB M1-centered system.

A fully local, full-capability, fidelity-qualified endpoint around 2 accepted
TPS is also a materially valuable delivery tier, but it is not “done” and does
not change any target threshold (PW-0310).

Current position: slow native text paths and useful exact lower-milestone
optimizations exist, but no run passes the full capability, hosted fidelity,
sustained-performance, or 50-TPS gates. The current authenticated onboard
architecture portfolio is analytically closed below 1 TPS; this is not a
universal impossibility result (PW-0333).

## Goal invariants

### Authority and semantic identity

- The pinned open checkpoint and published Xiaomi semantics answer component
  correctness questions. A properly frozen, provider-pinned OpenRouter epoch
  will be the sole external whole-model behavioral authority; PW-0001 proves
  reference-path viability and names only the initial provider. Neither
  authority substitutes for the other, and unavailable whole-model
  official-framework parity remains explicit.
- The source checkpoint is block-scaled FP8. Its census is exact:
  315,683,674,448 tensor bytes; 302,869,118,976 routed-expert bytes; 25,171,968
  bytes per routed expert; and 9,464,659,968 routed bytes selected by one cold
  ordinary token (PW-0002).
- `noaux_tc` selects with bias-corrected sigmoid scores but weights selected
  experts with normalized uncorrected scores. Experts compute
  `down(silu(gate(x)) * up(x))` (PW-0003).
- MiMo QKV is four concatenated tensor-parallel `[Q,K,V]` shards, not global
  `[all Q, all K, all V]`. Evidence produced with the old layout remains valid
  only for that named layout; it is not corrected-route authority
  (PW-0205–PW-0206).
- On full speculative convergence, the target bonus token is output, all `q`
  proposal-input KV rows are retained, and the bonus is the next uncached
  anchor. Bonus-free traces remain historical evidence and must not be repaired
  by adding one to `A` (PW-0326–PW-0328).
- Source batch-shaped arithmetic and one-row decode arithmetic are separate
  numerical authorities. Equality of mathematical weights does not imply bit
  equality across reduction order, backend, batch shape, or device
  (PW-0200–PW-0203, PW-0312–PW-0318, PW-0331).
- Target-faithful and modified modes stay distinct in code, artifacts, reports,
  and claims. Passing a local or distributional slice does not promote L3–L5
  work to L0–L2. Exact candidate execution proves implementation parity only
  with that candidate.
- An approximate draft preserves the target distribution only when its
  correction is mathematically exact and uses the actual proposal
  probabilities required by that algorithm. Verifier authorization alone does
  not establish this for arbitrary speculative methods.
- The rare-route slice is constitutive, not optional: the corrected 32-window
  corpus contains 939 identities absent from the three control categories
  (PW-0328).
- Every full model run on the shared M1 obeys the host-safety gates in
  `TARGET.md`; memory pressure, swap, throttling, release state, and protected
  services are part of correctness, not merely telemetry.

### Causal performance model

Keep these ledgers separate:

```text
SSD bytes/output    = B_e * f_S * U * (1 - h) / A
memory bytes/output = (B_d + B_e * f_M * U) / A
target compute      ~ q / A ordinary-token computations
```

`A` is committed output tokens, `U` is the byte-weighted union of selected
expert sets, `q` is verifier width, and `h` is demand-cache hit rate. The
following distinctions are decision-critical:

- prefetch can hide latency but cannot remove bytes;
- a storage codec helps `f_S` only if expansion precedes execution;
- cached weights avoid SSD traffic but still cross executable memory;
- speculation helps expert traffic only when `A` grows faster than `U`;
- storage-only, kernel-only, warm-block, aggregate, after-prefill, and complete
  request TPS are different claims;
- accepted endpoint tokens, not proposed tokens or component extrapolations,
  are the throughput numerator.
- native multimodal prefill can approach bank-wide expert demand; its traffic
  and time-to-first-token are separate from decode throughput.

The corrected q8 text corpus is the current route/acceptance authority:
across its 32 windows, `sum(A_full)=232`, `sum(A_observable)=231`, and
`sum(U)=142.71808510638297` (PW-0328). It is not authority for multimodal,
long-context, or final held-out behavior.

### Live branch dispositions

| Branch | Decision-relevant conclusion | Authority | Reopens only if |
| --- | --- | --- | --- |
| Corrected source text semantics | The corrected source path emits local `Hello!` exactly like the frozen hosted fixture, and corrected native MTP recovers the first transition. Broad arbitrary-prompt source-faithful parity is not yet established. | PW-0205–PW-0206 | More corrected-layout whole-model and hosted evidence passes. |
| SGLang-directed modified text path | A coherent 47-token endpoint exists, but it is explicitly modified arithmetic and measures only 0.026253 complete-request TPS. It is causal-path evidence, not target completion. | PW-0205 run 009 | Target-faithful arithmetic and hosted gates pass on the same complete path. |
| Native MTP | Exact q4 native MTP gives repeatable complete-path gains across four development categories and two untouched 32-token holdouts. The strongest untouched 32-token complete-request result is 0.0459782 TPS; the same ordinary holdout is 0.0791305 TPS after prefill. The corrected-route perfect-proposer expert-byte gain is at most 1.051643x, so this is a useful lower milestone, not the missing 50-TPS mechanism. | PW-0211, PW-0215–PW-0216, PW-0306, PW-0324, PW-0333 | A new proposer changes measured `A/U` and complete wall on untouched slices. |
| Source-FP8 streaming | Measured bytes and kernels structurally reject published DFlash-8/source-FP8 for 50 TPS. The current complete source-FP8/internal-SSD q8 branch is also below 1 TPS. | PW-0008–PW-0010, PW-0181, PW-0203, PW-0333 | Executable bytes or causal proposal leverage changes materially. |
| Exact cache/residency | Pressure-safe exact residency works. A 4.00 GB, 42-object prefix improves one transaction 5.81%, but misses its 2x gate and exhausts the safe prefix. Even favorable exact-codec plus future-aware token-granularity cache remains below 1 TPS. | PW-0207, PW-0332 | A new lossless representation or schedule changes both capacity and the cold critical cut. |
| Scheduling, prefetch, and transport | Expert-major batching, wider prefill, fused gate/up, predictive prefetch, uncached I/O, lane tuning, and ordinary parallelism yield local or conditional gains but do not cross the current storage-dominated endpoint cut. Several are useful only after the cut changes. | PW-0035–PW-0043, PW-0105–PW-0112, PW-0209–PW-0214, PW-0306–PW-0307 | A changed embodiment makes their bounded enclosing-path effect material. |
| Exact FP8 recoding | Tested palettes and top-exponent escape codecs cannot supply the required byte reduction. Even the impossible zero-escape floor plus perfect cache fails the onboard 1-TPS gates. | PW-0300, PW-0324, PW-0332 | A genuinely different lossless representation or execution schedule is proposed. |
| Conventional low-bit experts | Direct INT4/5/6/7 forms reduce bytes but fail the required real routed-activation fidelity or still miss the physical endpoint bound. Weight cosine or isolated projection error is not sufficient. | PW-0011–PW-0015, PW-0129–PW-0149, PW-0300–PW-0304 | A new form passes identity-local, routed, layer-final, whole-model, and hosted gates at executable size. |
| K4 hybrid experts | K4 can execute directly and some identity-local/cumulative slices pass; byte-neutral rank-one repair is a real fidelity improvement. But errors vary by identity and route, four-K4 density failed one frozen row, and even impossible-best corrected joint residency has fourth-lowest-window storage-only throughput of 0.882741 TPS. Bank construction is therefore closed by the higher-precedence tail bound. | PW-0308–PW-0319, PW-0329, PW-0331 | A smaller executable record or materially better proposer passes the same tail and fidelity gates before bank construction. |
| Wide/cyclic speculation | Stitched q64 acceptance was non-causal. A real q64 transaction accepts only three tokens. Cyclic reuse of the three native MTP heads breaks at the first reused head and is conditionally capped at 0.628115 storage-only TPS. | PW-0321–PW-0322, PW-0330 | A causal wide proposer plus verifier changes the first-mismatch and byte bounds; direct-q32 first-chunk parity alone is not a speed claim. |
| Shared bases / trained residual programs | Tested shared-basis, routed-mixture, low-rank, and recovery-training forms did not meet the combined capacity and held-out fidelity gates. Exact neuron permutation does not make source scale grids freely permutable. | PW-0113–PW-0126 | A new executable topology has a cheap activation-weighted falsifier and a credible full-path byte/compute budget. |
| Changed attention or acceptance | Probability-ranked history pruning, released sparse-attention patterns, and approximate mismatch acceptance failed their named fidelity or semantics gates. | PW-0162, PW-0173–PW-0176 | A distinct target-faithful mechanism or explicitly modified mode passes the appropriate whole-model gates. |
| Companion hardware | Surveyed under-$500 CPU, Volta, Ampere, Blackwell, AMD, and Intel configurations failed capacity, bandwidth, software, power, or complete-system bounds. These scoped rejections do not forbid a new compliant device, and the latest onboard closure imports no companion premise. | PW-0127–PW-0128, PW-0151–PW-0172, PW-0333 | An exact, purchasable BOM with measured end-to-end kernels changes the complete-system bound. |

### Decisive current bounds

- Measured lower milestone: 0.0459782 complete-request TPS on the untouched
  ordinary 32-token holdout; 0.0791305 TPS after prefill. The strongest short
  clean repeat is 0.0958758 TPS. None is a 30-by-512 or 60-minute result.
- Warm diagnostic only: PW-0203 reaches 0.219850 accepted TPS for one dirty
  L3 verifier block. It is not a complete, sustained, or qualifying result.
- K4 impossible-best bound: 1.564790 aggregate storage-only TPS, but 0.882741
  at the required fourth-lowest window (PW-0329).
- Cyclic-q32 named bound: 0.628115 storage-only TPS, conditional on unproven
  first-chunk parity (PW-0330).
- Absolute exact-codec/cache bound: 0.743963 aggregate, 0.589967 token-p10,
  and 0.696227 fourth-lowest-window storage-only TPS (PW-0332).
- Therefore the composed authenticated onboard portfolio is rejected below
  one TPS. These upper bounds omit costs and are not achieved TPS; unknown
  future algorithms and compliant companion hardware are outside the closure
  (PW-0333).

## Prediction errors

These unresolved distinctions can still change the next decision:

Default priority: advance the corrected native endpoint and capability ladder
unless a genuinely new representation, proposer, schedule, or compliant
hardware premise first passes a cheap falsifier.

1. **Corrected source-faithful endpoint breadth.** The corrected QKV source path
   has tiny hosted agreement, but no broad arbitrary-text, multimodal,
   long-context, tool, or full frozen-distribution result.
2. **Qualifying performance.** There are zero authenticated runs of the
   designated 30-by-512-token, 60-minute sustained protocol, and no complete
   full-capability endpoint TPS result.
3. **Native capability.** Image, multi-image, audio, video, mixed modality,
   tool use, and the required context bands have not climbed the complete
   corrected-layout correctness ladder.
4. **External reference durability.** Parasail supplied top-20 logprobs for the
   probed modalities with reasoning disabled in PW-0001, but a final frozen
   epoch, drift canaries, and full scored corpus remain outstanding.
5. **A changed physical premise.** Reopening onboard performance requires a
   new executable representation, a MiMo-specific proposer with materially
   better causal `A/U`, or a different execution schedule that escapes the
   PW-0333 portfolio—not a recombination of its rejected ceilings.
6. **A changed hardware premise.** `TARGET.md` still permits compliant local
   companion hardware, but no surveyed under-$500 system has supplied a
   measured full-path route to the target.
7. **Direct wide-verifier semantics.** Direct-q32 first-chunk parity remains
   unproven. It should be tested only as part of a changed proposer/byte premise;
   proving it alone cannot overturn PW-0330's scoped ceiling.

When evidence resolves one of these items, update this frontier in place:
replace the affected belief, retain the smallest evidence pointer needed to
justify it, and leave experiment chronology in `experiments/`. Do not append a
new diary entry here.
