# PW-0186 — Target-generated Jacobi second iteration

- Status: completed
- Disposition: promoted to a third/convergence iteration; not an endpoint
- Date: 2026-08-10
- Owner: Codex with project owner authorization
- Model/reference: MiMo revision
  `63651580ca774f8504f676040460aed3e1244ac1`; PW-0091; PW-0095; PW-0102
- Prior target authority:
  `cb30738d5a79d7d85587a68b53f876a59101d5ca09bbc7c895daaf501954f4d3`
- Execution mode: target-faithful greedy L2 proposal with exact verification
- Hardware/runtime: source CPU oracle on the existing Apple M1
- Related records: PW-0044, PW-0102, PW-0110 through PW-0112, PW-0173,
  PW-0181, PW-0185

## Question and causal mechanism

Lookahead decoding claims exact multi-token progress without a separate draft
model. PW-0102 already executed a valid first Jacobi target iteration: proposed
block `[264,1773,102092,102092,102092,1773,1773,1773]`, authenticated target
posterior `[13,15,18,481,15,481,15,15]`. Shift that posterior once under the
causal-language-model convention to freeze iteration two as
`[264,13,15,18,481,15,481,15]`.

Run that exact block through the same source checkpoint, authenticated prompt,
prefill parity, K/V construction, all 48 decoder layers, source routing, final
norm, full LM head, and greedy block verifier as PW-0102. Record every target
posterior, accepted prefix `A`, per-layer expert union, mean normalized `U`,
source-byte ledger, cold/warm state, complete wall time, hardware, and Gate 8.
Do not import published Llama acceptance or call a converged Jacobi component
an endpoint.

## Continuation and kill gate

PW-0181's impossible cache leaves 1.090015 seconds of expert-miss acquisition
and 0.714682 seconds of warm MoE work per scalar pass, plus 0.131220 seconds of
attention. Give iteration two every favorable assumption: multiply the first
two terms only by measured `U`, overlap them perfectly, charge attention only
once rather than by row, and make drafting, dense work, LM head, correction,
and rollback free. The optimistic seconds per accepted token are
`(max(1.090015*U,0.714682*U)+0.131220)/A`.

Promote a third-iteration/converged-trajectory audit only if this value is below
one second and `A/U > 1.090015`. Otherwise reject this authenticated Jacobi
chain as the onboard one-TPS mechanism; another initialization or trained
proposer must provide new causal evidence rather than merely more iterations.
Report the real CPU oracle timing diagnostically, zero endpoint TPS, and no
throughput-constant change.

## Result

The authoritative manifest hashes to
`f773fa2859f08b57f851944aa8ba0ef9b502040058580a9344be4ce3ee1e1d1c`.
Iteration two produces posterior `[13,15,13,15,15,15,15,264]`. Exact greedy
verification accepts the anchor plus two draft tokens: `A=3`. Mean normalized
expert union is `U=2.268617`, so `A/U=1.322392` clears the frozen necessary
`1.090015` gate.

Under the deliberately favorable bound, optimistic time is `0.868016` seconds
per accepted token, or `1.152053` TPS before omitted dense work, LM head,
correction, rollback, and real wide-transaction effects. The source CPU oracle
itself takes 274.337 seconds post-prefill and is not a performance candidate.

Promote exactly one authenticated third/convergence iteration. Do not call the
optimistic bound endpoint TPS. Gate 8 passes with 59% minimum free memory,
4,022,747,136-byte maximum peak RSS, 208,292,928-byte maximum physical
footprint, zero swap growth or throttling, and stable protected services.
