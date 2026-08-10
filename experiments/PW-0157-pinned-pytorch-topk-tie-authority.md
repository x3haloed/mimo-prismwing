# PW-0157 — pinned PyTorch top-k tie authority

- Status: completed
- Disposition: correctness-repair
- Date: 2026-08-09
- Owner: Codex with project owner authorization
- Model/reference: PyTorch `2.13.0`, commit
  `cf30153c4c131c8164ee7798e5022d810682e2cb`; MiMo revision
  `63651580ca774f8504f676040460aed3e1244ac1`
- Related records: PW-0063, PW-0091, PW-0156
- Implementation commits: pinned top-k bridge and bounded walk
  `f677893a12fc5631ddcdecf8fc407b7d1178c3f5`; one-shot K/V release
  `7c0bf18390fcf064258e9486a6ee467d77f0d035`; failure-preserving 8K
  runtime `6368ae80e67e73e008751c5add20780e86b02b0d`; clean analyzer
  `72b970244d8c8513d683d4c4d632493ecf76c10f`

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

The correctness repair passes. The hash-pinned PyTorch 2.13.0/libc++ fixture
and native bridge agree exactly on all five adversarial tied rows, including
unsorted index order. The real walks observed 3, 9, 12, 23, and 63 top-k
boundary-tied rows at prefixes 512, 1,024, 2,048, 4,096, and 8,000. PW-0156's
blanket tie rejection is therefore superseded only for this pinned executable
authority.

The 512-position original control and one-shot K/V-release walk match every
route-semantic field exactly. The bounded runtime then completed all 8,000
positions and touched 4,903 distinct `(layer, expert)` records. After the
impossible grant of 660 perfectly chosen resident records, 4,243 records or
`106,804,660,224` source bytes remain to stream. This is 4,100 records below
the predeclared 9,003-record rejection boundary, so retain only the optimistic
four-lane 8K storage-capacity condition. It is not a measured storage result
and does not reverse PW-0158's complete two-P100 rejection.

The five exact distinct-record observations are 2,980, 3,572, 4,456, 4,585,
and 4,903. The 8K walk took `19,765.815` seconds; this timing is diagnostic CPU
oracle time, not prefill TPS or hardware performance. It read
`132,160,770,048` process bytes and peaked at `3,967,156,224` bytes RSS.
Gate 8 passes with 69% minimum free memory, `3,866,898,752` bytes maximum
physical footprint, zero swap growth or throttling, a
`1,990,219,520`-byte post-release footprint, and stable protected services.

Raw 8K evidence hashes to
`8cfc737df848f4a98cb0774c7367d1e95f5311e051dd5787842712ca6c2fd163`.
The authoritative offline analysis hashes to
`e7df87bb326e543b5b500c698eae1700d2fd204d6b2d2a833736706456955cfc`.
PW-0157 reports zero accepted tokens, no endpoint TPS, no purchase authority,
and no promoted runtime default.
