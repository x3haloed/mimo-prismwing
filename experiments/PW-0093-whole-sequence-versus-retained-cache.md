# PW-0093 — Whole-sequence versus retained-cache decode

- Status: in progress
- Disposition: unexecuted
- Date: 2026-08-06
- Owner: Codex with project owner authorization
- Commit and dirty state: contract precedes implementation and execution
- Checkpoint/reference hashes: MiMo revision
  `63651580ca774f8504f676040460aed3e1244ac1`; PW-0092 run 001
  `18c3ccde4a8645d9ea46d0091f877eebe256ca2c7d82c34e771f5f4114bb5f25`;
  checkpoint verification
  `9ddc8a99755f04ae2ea3c2484f6dd022d3f3a681b5a72c915ee4de833dbb0d03`
- Hardware/runtime: Apple M1 shared 16 GiB host, verified SSD checkpoint,
  independent PyTorch whole-sequence oracle versus production Rust caches
- Related records: PW-0060, PW-0091, PW-0092

## Hypothesis and mechanism

PW-0092's second token is causally driven by retained per-layer K/V, but its
arithmetic authority and cache authority still live in the same Rust process.
Evaluate the exact 28-token prefix formed by appending PW-0092's accepted token
264 to its frozen 27-token prompt through the independent PyTorch source oracle
as one uncached causal sequence. If K/V positions, RoPE offsets, attention
windows, route state, or cache updates are wrong, its final distribution or
position-27 routes will diverge from the retained-cache endpoint.

Generalize only row count and explicit token identity in the existing oracle;
preserve its default 27-token behavior and frozen semantics. Add a deterministic
fixture test for row/token authority before execution. Do not change production
arithmetic, checkpoint data, prompt serialization, sampling, or thresholds.

## Gates

The 28-token PyTorch oracle must preserve the first 27 route rows exactly from
PW-0092 step one and match route row 28 exactly to PW-0092 step two in all 47
routed layers. Expert order is authoritative. Route-weight comparison uses the
existing source threshold of `5e-7` per expert. The complete 152,576-value final
logit vector must be byte-identical to PW-0092 step two; any mismatch localizes
the retained-cache path rather than being averaged away. All values must be
finite and every identity/hash/shape check must fail closed.

Enforce normative Gate 8 at oracle open and every captured layer/final boundary:
minimum system memory-free 20%, process peak/current at most 8 GiB,
post-release footprint at most 4 GiB, swap growth at most 512 MiB, no new
throttled pages, explicit release relief, and continued health of ChatGPT,
WindowServer, `nxnode`, and Syncthing. Preserve stopped evidence. Run one full
oracle only; a second is justified only if the first passes but comparison is
nondeterministic or ambiguous.

This is a correctness experiment. Its wall time, storage traffic, and memory
are diagnostic and cannot become accepted TPS or promote a performance default.

## Result

Unexecuted.

## Decision

Unexecuted.
