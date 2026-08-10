# PW-0157 — pinned PyTorch top-k tie authority

- Status: ready
- Disposition: unexecuted
- Date: 2026-08-09
- Owner: Codex with project owner authorization
- Model/reference: PyTorch `2.13.0`, commit
  `cf30153c4c131c8164ee7798e5022d810682e2cb`; MiMo revision
  `63651580ca774f8504f676040460aed3e1244ac1`
- Related records: PW-0063, PW-0091, PW-0156
- Implementation commit and dirty state: pending

## Hypothesis and changed premise

PW-0156 conservatively rejects every exact eighth/ninth router-score tie because
PyTorch does not promise stable tied indices across implementations. The pinned
local reference is narrower and inspectable: its shipped `TopKImpl.h` uses a
`std::vector<pair<float,int64_t>>`, libc++ `std::nth_element`, the same strict
value comparator, and the same first-eight extraction as Prismwing's existing
C++ bridge. Therefore the bridge may already reproduce the actual pinned CPU
authority at ties; the missing work is proof and explicit evidence accounting,
not a second routing algorithm.

## Exactness and red-line check

This is a target-faithful L0/L1 correctness repair. It changes no logits,
scores, route weights, checkpoint tensor, top-k cardinality, acceptance gate,
or performance threshold. It permits tied selections only when the executable
bridge is proven against the exact pinned PyTorch build and source algorithm.
The native bridge remains the sole runtime selection implementation; PyTorch is
fixture authority, not an inference dependency.

## Contract

1. Generate a hash-pinned fixture with actual `torch.topk(..., k=8,
   sorted=False)` under version `2.13.0`, commit
   `cf30153c4c131c8164ee7798e5022d810682e2cb`, on finite 256-value rows covering
   a two-way boundary tie, a multiway boundary tie, all-equal values, repeated
   plateaus, and signed zero.
2. Store every input as exact F32 bits and every expected unsorted expert index.
   The Rust/C++ bridge must match every index in order, not merely the selected
   set. Preserve PW-0063's untied real fixture as a separate regression.
3. Confirm the installed `TopKImpl.h` source shape and record its SHA-256.
   Fail closed if the fixture's PyTorch version, commit, row width, top-k, or
   source hash changes.
4. Replace PW-0156's blanket tie rejection with use of the already-proven pinned
   selection. Count every boundary-tied row in the endpoint ledger; do not hide
   tie incidence or generalize this authority to another PyTorch build or C++
   standard library.
5. Run the full suite, then repeat PW-0156's original primary 512-position
   prefix from a clean implementation commit. Preserve all Gate-8 stops and
   publish no partial coverage result.
6. If the exact bridge fixture fails, or the real walk changes an already
   untied PW-0091/PW-0112 route, reject this repair and leave PW-0156
   inconclusive. If it passes, supersede only PW-0156's tie-authority blocker
   and evaluate the unchanged 9,003-record storage gate.

## Result

Pending implementation and execution from a clean commit.
