# PW-0332 — Exact top-seven token-granularity cache oracle

- Status: proposed; unexecuted
- Disposition: unexecuted
- Date: 2026-08-27
- Owner: Codex
- Parent experiments: PW-0136, PW-0207, PW-0212, PW-0300, PW-0324,
  PW-0328
- Exactness: L1 exact-codec storage sensitivity; target work unchanged

## Question

Can the strongest remaining exact source-FP8 codec plus an impossible
future-aware 12-GiB resident cache exceed one accepted token/s on every
corrected PW-0328 category when cache decisions are allowed after every
verifier-authorized row and all non-storage work is free?

This is a byte-floor falsifier for the named token-granularity schedule, not a
decoder or endpoint. It grants exact
top-seven exponent recoding its zero-escape physical floor even though real
blocks have escapes, pins encoded fixed weights for free, chooses the initial
expert cache with future knowledge, charges no decode or prefetch cost, and
permits fractional encoded records. Companion hardware is inadmissible.

## Input freeze

PW-0328 is complete. Its canonical manifest is
`/Volumes/Elements/mimo-prismwing/evidence/PW-0328/corpus-001/manifest.json`,
SHA-256
`36e4f10b6f807f766c87ee7078f5f18ea8fc339dd12e4dbc24f1f4ac6e824403`,
built from clean capture commit
`26d2ea31852c0d63bd022df6d571fd722137c39f`. An independent replay
authenticates all 24 bound artifacts, 32 chronological windows, full `A=232`,
every causal hidden-history binding and q8 route union, byte closure, and Gate
8. Analyzer authorship is now authorized. It must hard-code this hash and must
not accept an older PW-0208 manifest or a caller-asserted replacement.

The completed source must contain exactly eight chronological primary windows
in each of ordinary, code, multilingual, and rare-route. Flatten only the first
full verifier-authorized `A` route rows from each window. Never consume a
mismatch suffix, terminal clipping substitute, proposal route, or rejected
posterior row. Preserve window and category order.

“Token-granularity” describes an impossible-favorable cache schedule over the
already authenticated target rows; it does not assert that a separately built
width-one kernel is bit-identical to the width-eight capture. The oracle may
evict between authorized rows, a freedom broader than the captured q8 runtime.

## Frozen authorities

The executable analyzer must fail closed on:

1. The completed PW-0328 manifest plus every bound generation report, progress
   log, verifier-hidden payload, prefill report, and prefill-hidden payload.
   Reconstruct every chronological authorized route row, `A`, and `U`.
2. PW-0324 canonical analysis
   `/Users/chad/Models/mimo-prismwing/evidence/PW-0324/analysis-002/analysis.json`,
   SHA-256
   `97d4d20a4c709d42429973e867138495756ce9d52d417f98a7edd40b282ccff3`.
   Authenticate its local exact-codec replication, checkpoint index hash, 480
   deterministic FP8 blocks, observed minimum top-seven ratio
   `0.8856201171875`, and limitation that the sample is not a routed full-model
   codec census.
3. PW-0207 `offline-002.json`, SHA-256
   `1dedbef7c79aa23835d194f52760a1f2c65dcca1481bd6df2d5602615c3fdad6`,
   the receipt-authenticated index, and an independent tensor census. Recompute
   381 fixed objects totaling `7,743,236,992` logical source bytes, including
   exactly `3,073,376,256` FP8-code bytes. BF16/F32 bytes are never compressed.
4. PW-0212 canonical prefetch oracle
   `/Volumes/Elements/mimo-prismwing/evidence/PW-0212/corrected-route-prefetch-oracle-001.json`,
   SHA-256
   `2365033116e194b6bac34d2017f644c3499c5fb92a3727f7db9162dce318587f`.
   Its future oracle hides only `1.616833%` of legacy complete wall, but this
   experiment grants perfect future knowledge and zero prefetch cost anew on
   corrected PW-0328 routes rather than importing that stale percentage.
5. PW-0136 raw and validated-analysis SHA-256 values
   `e6ab84cada19c6036ee7b83f318c3920631141b9ea5e882cc88eb9784d0b5a56`
   and `7ebf2cde5c4a3f4931d2d705993f822e38af13ea66bc3efc91410296b14e2aab`.
   Use the slightly faster rounded historical bandwidth
   `3,470,448,309.677419` bytes/s for the hard candidate-favorable ceiling and
   report the exact raw-derived `3,470,425,919.832775` bytes/s separately.
6. Checkpoint revision `63651580ca774f8504f676040460aed3e1244ac1`,
   receipt SHA-256
   `9ddc8a99755f04ae2ea3c2484f6dd022d3f3a681b5a72c915ee4de833dbb0d03`,
   index SHA-256
   `f2e1774c9acf9a62338b68c144e6fc7a66495e59f2e64b3078c1b7ef5a196816`,
   `TARGET.md` SHA-256
   `dda459684c194b03491f36e9b66521ff00c400a6cc38d23a567a5a92ef8fb17d`,
   `RED_LINES.md` SHA-256
   `cc261ad9bd67a865715e72cbbadf3b74c3f1f282e17a8ef86ed02c1a92fb8b36`,
   Apple M1 16 GiB, batch one, concurrency one, and the explicit
   companion-hardware exclusion. The execution commit must descend from the
   clean commit adding this contract and freeze this document's Git blob ID and
   SHA-256.

## Exact codec floor

For one 128-by-128 FP8 block with `N=16,384` codes and `e` escapes, the named
exact top-seven exponent form occupies

```text
encoded_bytes(e) = ceil((7*N + 4*e + 28) / 8)
ratio(e) = encoded_bytes(e) / N
```

The absolute zero-escape floor is therefore
`14,340/16,384 = 0.875244140625`. No lossless implementation of this exact
format can beat it. The PW-0324 observed minimum is
`14,510/16,384 = 0.8856201171875`; it is a separately labeled sensitivity, not
a universal bound on unobserved fixed or routed tensors.

Logical checkpoint bytes never change. Encoded bytes are reported separately;
no encoded ceiling may be relabeled logical traffic or an achieved read.

## Predeclared scenarios

Use `source_expert = 25,171,968` logical bytes and `R = 12 * 2^30` bytes.
Evaluate exactly:

1. `uncompressed`: fixed and expert ratio one.
2. `observed_expert_only`: apply `0.8856201171875` favorably to each complete
   expert record, including its scales, but leave every fixed object unchanged.
3. `absolute_floor_all_fp8`: leave fixed BF16/F32 bytes unchanged, apply
   `0.875244140625` to all `3,073,376,256` fixed FP8-code bytes, and apply the
   same floor favorably to each complete expert record, including its scales.

Scenario three is the hard kill authority. It is strictly more favorable than
the named format: real blocks need escape bits; scales are not FP8 code blocks;
metadata, block boundaries, integer byte lengths for whole expert records,
decoder buffers, alignment,
and the largest-object guard are free. Scenario two may diagnose observed
headroom but cannot reject unobserved tensors.

For each scenario, pin the entire encoded fixed set and reject if it exceeds
`R`. The equal-sized expert capacity is

```text
C = floor((R - encoded_fixed_bytes) / encoded_expert_bytes)
```

Do not let fixed and expert objects each spend `R`. Do not add embedding-row
traffic: omitting it is another favorable grant.

### Fixed-residency dominance for this schedule

The q1 event stream demands every fixed tensor once per authorized row in model
execution order. Each layer-qualified expert identity can be demanded at most
once per row. Therefore, between two consecutive demands for any fixed byte,
any particular expert byte can be demanded at most once: the interval spans at
most one visit to each routed layer.

Take any joint dynamic fixed/expert schedule. Whenever expert storage occupies
space while fixed storage is absent, exchange up to the same number of bytes
back to the absent fixed objects until their next demand. The exchange incurs
no more expert miss bytes than the fixed reload bytes it removes. Repeating the
exchange yields a schedule with the complete fixed set resident and no greater
traffic. Whole-object granularity cannot improve the evict-fixed schedule: a
set of whole experts fitting in an evicted fixed-object byte total saves at most
that same byte total before the fixed objects are needed again. Thus pinning the
fixed set is a traffic-minimizing representative for this token-granularity
stream, not an extra restriction on the hard ceiling.

The proof does not apply to a layer-major or width-eight schedule that reuses a
fixed tensor across multiple rows before another fixed demand. PW-0332 makes no
hard claim about those schedules.

## Batch-set offline Belady oracle

Within each category, map every chronological authorized token row to 47
ordered layer demands, each an unordered set of exactly eight distinct
`(layer, expert)` identities. Reset between categories and grant the complete
initial expert cache for free.

From the universe of identities demanded anywhere in that category, choose
`min(C, distinct identities)` initial identities by earliest next demand, with
canonical `(layer, expert)` order breaking equal-next-use ties. For each layer
demand `D`, count `D - resident` as misses. All identities in `D` must coexist
during the demand. Insert misses and, when capacity is exceeded, evict only
from `resident - D`, choosing farthest next demand first, infinity before
finite, and reverse canonical identity as the deterministic exact tie break.
Update next-use positions only after the whole set is served. Reject
non-distinct route rows, `C < 8`, or any result that depends on iteration order
within `D`.

Implement the oracle twice: an indexed version and an exhaustive tiny-state
dynamic-programming reference. They must agree on deterministic small fixtures.
For the real trace, independently replay the final residency and miss ledger.

## Byte and throughput ceiling

For token event `t`:

```text
logical_moved_t = misses_t * 25,171,968
encoded_moved_t = misses_t * encoded_expert_bytes
storage_wall_t = encoded_moved_t / 3,470,448,309.677419
storage_tps_t = 1 / storage_wall_t
```

Zero-miss events have infinite diagnostic TPS and zero wall; serialize infinity
as an explicit string, never non-standard JSON. Category and corpus aggregate
TPS are token totals divided by summed storage wall. Report nearest-rank p10
over token-event TPS for the corpus and for every category using rank
`ceil(0.10*n)` in ascending order. Also report all 32 original window
aggregates so the fourth-lowest window p10 remains comparable with PW-0329.

Record modeled source `A`, `U`, misses, logical and encoded bytes, capacity,
resident identities, evictions, free initial identities, category resets, and
both bandwidth ceilings. The experiment itself emits `accepted_tokens: 0`,
`A: 0`, `U: 0`, and `performance_claim: null`.

## Frozen gates and dispositions

1. If `absolute_floor_all_fp8` has overall aggregate, any required-category
   aggregate, corpus token p10, any category token p10, or fourth-lowest window
   TPS at or below one, reject the named exact top-seven codec plus 12-GiB
   future-aware token-granularity residency/prefetch/overlap composition on the
   authenticated PW-0328 demand. No decoder is authorized.
2. If every strict gate is above one, retain only an analytical survivor. A
   real decoder, exact encoded checkpoint census, pressure-safe cache manifest,
   target compute, and complete interleaved endpoint remain mandatory.
3. If only the observed scenario fails, record a diagnostic rejection of the
   sampled ratio; do not reject the absolute codec floor.
4. No result changes runtime defaults, fidelity thresholds, TARGET.md, or the
   companion-hardware exclusion.

The hard rejection is decisive for this exact codec and authenticated demand,
not a theorem about every future lossless code or every possible prompt.

## Required fixtures

- reject incomplete/wrong-category manifests, wrong hashes, clipped `A`,
  mismatch suffix rows, proposal routes, duplicate route identities, and
  category/window reordering;
- reproduce the exact escape formula, `14,340` zero-escape bytes, observed
  `14,510` bytes, and the impossibility of a ratio below the floor;
- keep fixed FP8, fixed BF16/F32, expert logical, and expert encoded ledgers
  distinct; prove scenario three is no worse than every other scenario;
- reject common/expert residency double-spend and cover fit-all, `C=8`, and
  `C<8` branches;
- reproduce exact scenario capacities `204`, `230`, and `250`, and prove on a
  tiny q1 event stream that dynamically evicting fixed objects cannot beat the
  fixed-pinned representative;
- prove batch-set order invariance, future-use updates after the set, initial
  free-fill ordering, deterministic ties, category reset, and zero misses;
- compare indexed Belady with exhaustive optimal tiny cases and independently
  replay real misses;
- cover finite/infinite TPS JSON, nearest-rank p10, token/category/window/corpus
  aggregates, exact versus rounded bandwidth, and disposition precedence; and
- fail closed on dirty Git, overwrite, nonfinite evidence, unsafe pressure, or
  evidence-schema drift.

## Claims excluded

- an implemented codec, decoder throughput, physical encoded bytes, or achieved
  endpoint TPS;
- direct width-one target arithmetic or route parity beyond the authenticated
  chronological target rows;
- layer-major or width-eight cache schedules;
- K4, cyclic-MTP q32, a changed proposer, or a modified target;
- multimodal/long-context promotion, a runtime default, or companion hardware.

## Result

Unexecuted.

## Decision

Unexecuted. Freeze the completed PW-0328 manifest hash before analyzer
authorship.
