# PW-0069 — Real layer-7 substage localization

- Status: complete
- Disposition: promoted diagnostic; repair not yet selected
- Date: 2026-08-05
- Owner: Codex with project owner authorization
- Commit and dirty state: contract precedes implementation
- Checkpoint/reference hashes: MiMo revision
  `63651580ca774f8504f676040460aed3e1244ac1`; PW-0068 comparison
  `bc380e725d358594d6f73b8ec4e2b87371017eb4e1b7af47d2071ce985363799`
- Hardware/runtime: Apple M1 shared 16 GiB host, verified SSD checkpoint,
  PyTorch 2.13.0 CPU oracle, production Rust trace
- Related records: PW-0064 through PW-0068

## Hypothesis and contract

PW-0068 proves layer 6 is an exact accumulated source boundary and layer 7 is
the first failure. Extend the already-regressed routed-layer diagnostic to
target layer 7 without duplicating its execution or comparison authority.

Rust must causally recompute layers 0–6, then capture production layer 7.
PyTorch must load the hash-verified PW-0060 layer-6 final and independently
derive layer-7 attention, routes, selected experts, expert tensors, scatter,
and residual. Capture the same 21 substages and bind the same source, revision,
checkpoint, prompt, numerical policy, shapes, schedules, and hashes as PW-0065.

Keep all BF16, final-state, exact-expert, and `5e-7` route-weight gates. Name
the first differing substage before changing arithmetic. Retain every
phase-level RSS/free-memory/swap/throttling/release/protected-service stop.
Generated tensors stay external; accepted tokens are zero and no throughput
or hosted threshold can change.

## Result

The oracle completed in 21.757 seconds and the production Rust trace completed
in 119.036 seconds. Incoming state, input normalization, QKV projection, RoPE
query/key, value scaling, and attention sinks are bit-exact. The first actual
arithmetic difference is one of 25,920 centered BF16 attention-score values:
position 22, head 12, source-token index 17 is `-0.2265625` in PyTorch and
`-0.234375` in Rust. That one score changes two BF16 probabilities and 21
attention-output values. The projection amplifies the difference to 292 values;
12 of 110,592 post-attention residual values differ by at most `0.0625`.

The downstream differences match the PW-0068 whole-walk result: expert sets
and order remain exact, while route weights differ by at most
`1.6983320045432793e-6`. This rejects router, expert, scatter, and residual
arithmetic as the first cause. The remaining boundary is the BF16 query/key dot
product used by PyTorch's CPU matrix multiplication versus Rust's scalar F32
sum; its exact accumulation schedule must be isolated before a repair.

The Rust diagnostic peaked at 722,960,384 bytes RSS and 608,182,336 bytes
physical footprint, returned to 125,407,168 bytes after captures, retained at
least 81% system-free memory, grew no swap, observed no throttled pages, and
retained ChatGPT, WindowServer, nxnode, and syncthing. The oracle retained at
least 80% system-free memory with no swap growth or throttling. Evidence hashes:

- Oracle manifest:
  `632b19962663bee4c603cba96ff5f3f65c3f6f72747d0a22e1df0481acd79d55`
- Rust manifest:
  `e945f442c642f6a2ecb65286c1f822d9bed9bed79876864bb4c7b2e24dd18ee8`
- Comparison:
  `01f212e3f16e0f3d5f03899a9608272c770691a1d0ae4777d186635fa792ba98`

## Decision

Promote the localization result, not an arithmetic repair. Build a deterministic
fixture around the single failing query/key pair and use it to identify and
gate PyTorch's aarch64 BF16-dot/F32-accumulation order before changing the
runtime or repeating a full model walk. No throughput constant, hosted
threshold, or target-faithful mode changes.
