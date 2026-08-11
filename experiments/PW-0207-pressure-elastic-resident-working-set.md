# PW-0207 — Pressure-elastic resident working set

- Status: complete
- Disposition: conditional — bounded lower milestone promoted; 2x endpoint gate rejected
- Date: 2026-08-10
- Execution mode: L1 target-faithful representation/lifetime change; modified arithmetic controls separate
- Related records: PW-0105 through PW-0111, PW-0181, PW-0205, PW-0206

## Hypothesis and mechanism

PW-0205 emits 47 coherent tokens after about 1.636 TB of process reads while
peaking below 4 GiB. The old 8 GiB ceiling left much of the 16 GiB host
deliberately unused. Gate 8 now permits a declared 12 GiB persistent working
set under a 13 GiB process ceiling.

Hypothesis: choosing resident objects by *measured marginal critical-path stall
avoided per byte*—not tensor frequency—can cut physical reads at least 4x and
complete-path wall time at least 2x on the PW-0205 prompt. Candidate objects
include shared projections/norms, LM-head partitions, and recurrent expert
tiles. The source checkpoint remains authoritative backing storage.

PW-0104 already rejected a 6–8 GiB equal-size expert-only cache on the old
trace: even offline Belady reached only about 60% hit rate at 8 GiB. This record
does not erase that result. Its changed premises are a corrected-QKV route
trace, up to 12 GiB total declared residency, a heterogeneous shared-plus-
expert object set, and selection by attributed stall avoided rather than hits.
If those premises do not change the offline bound, the old rejection stands.

## Contract

Before allocation, emit a residency manifest with object identities, hashes,
exact bytes, reuse distance, expected avoided reads, lifetime, and eviction
order. Total declared persistent residency is at most 12 GiB; process peak is
at most 13 GiB; at least 3 GiB remains outside the process; swap growth and new
throttled pages are zero. Above 8 GiB, sample Darwin pressure state before each
growth step and checkpoint. Warning pressure evicts in declared order (a
runtime without an eviction callback stops); critical pressure stops.

Cold, warm, and pressure-eviction runs are separate. Cache hits count as bytes
avoided only when the control would perform an attributed physical read.

## Cheap falsifier and gates

First replay the corrected PW-0206/PW-0205 route trace through a byte-accurate
offline cache optimizer with measured acquisition latency. Kill implementation
if no legal 12 GiB set predicts either 4x fewer physical bytes or 2x less
attributed acquisition wall.

Then run one real repeated decoder transaction, interleaving control and
candidate. Require exact token/route parity, at least 2x complete-transaction
speedup, and pressure-safe release. Only then run the 32–64-token endpoint.
Endpoint promotion requires at least 2x complete-path TPS, not a cache-hit or
kernel-only number.

## Decision

The first exact route authority is
`/Volumes/Elements/mimo-prismwing/evidence/PW-0207/route-001/report.json`,
SHA-256
`e5c0b93d039ec8d8c6b1f7a0087ec3991ba55df2a1cee7d388f08d6e668d830b`.
Clean commit `871db1ff3c3f654d9184de22c958c877e1402006` reproduces PW-0205
run 009 transaction zero exactly: proposal, posterior, seven committed tokens,
convergence, and `U=4.582446808510638` all match. It captures all 7×48
one-token proposal traces and the 48-layer width-eight verifier trace. Complete
wall is 383.781 seconds, peak RSS is 3,951,427,584 bytes, and no high-residency
allocation occurs.

The clean offline falsifier at commit
`2498dc3b7cdcb1cc7d498ff75428cf4b0cdaf0e0` hashes to
`ee9f71b83ca427bd1a98d166ada77c778c53d81157dfa6ccb071afada54e73eb`.
Its 7,418 object accesses and 171,569,939,456 logical bytes close exactly to
the endpoint ledger. Positive proposal/verification equations attribute
2.180946 ns/byte to the shared spine and 0.450719 ns/byte to routed experts.

The ratio-ranked static 12 GiB set contains 388 shared-spine objects
(7,743,294,336 bytes) and 204 expert bundles (5,135,081,472 bytes), totaling
12,878,375,808 bytes with 6,526,080 bytes unallocated. It predicts
95,872,213,233 physical bytes, only a 1.791485× reduction, so the 4× byte gate
fails. It predicts 56,562.852 ms of attributed acquisition wall versus the
184,510.448 ms control, a 3.262043× speedup, so the alternative 2× wall gate
passes.

That first manifest's 12,878,375,808-byte figure is superseded for allocation
authority, while retained as the exact source-byte total. It did not charge
the 16 KiB physical allocation granularity required by the page-aligned owned
backing. The corrected clean manifest at commit
`e912157e11d66a8cbf4be4b265afca923879c8e4` is
`/Volumes/Elements/mimo-prismwing/evidence/PW-0207/offline-002.json`, SHA-256
`1dedbef7c79aa23835d194f52760a1f2c65dcca1481bd6df2d5602615c3fdad6`.
It retains the same 592 objects and source-byte prediction, but declares
12,882,755,584 resident allocation bytes: 5,137,170,432 for 204 expert bundles
and 7,745,585,152 for 388 shared objects, leaving 2,146,304 bytes unallocated.
The 1.791485× byte and 3.262043× wall predictions are unchanged because the
selected set did not change. Use `offline-002.json` as the implementation
authority; preserve `offline-001.json` only as superseded source-byte evidence.

Authorize the Darwin pressure observer, declared residency authority, warning
eviction callback, and one repeated-transaction implementation. This is an
offline static prediction on one corrected transaction, not endpoint TPS or a
13 GiB default. The experiment remains running until the real interleaved
transaction and pressure-eviction gates pass or reject it.

The first implementation checkpoint is clean commit
`c10cc1e0df23efc69e3e66521e7b57a445bf13d4`. It validates the canonical
592-object offline manifest, owns installed payload lifetimes, observes Darwin
normal/warning/critical events on a dedicated drained dispatch queue, evicts
warning payloads in the manifest's total order, and makes critical pressure a
permanent growth stop. The full suite passed 96 Rust and 364 Python tests. The
normative runtime still uses the 8 GiB ceiling.

The corrected synthetic safety report is
`/Volumes/Elements/mimo-prismwing/evidence/PW-0207/pressure-smoke-002.json`,
SHA-256
`199107b541670a915fba5a17b5ef9cc2c139309e1e81e476692161161867e6a2`.
It retains 8 declared bytes on an injected normal event, releases both owned
payloads in declared order on warning, releases the installed payload and
rejects regrowth on critical, and starts/drains a live Darwin observer. Its
scope is synthetic event injection, not a real OS warning, 12 GiB allocation,
transaction speedup, or TPS.

Preserve the earlier `pressure-smoke-001.json` as rejected evidence: its
SHA-256 is
`758d895c28b253fcc1b0567de53d9cdb4812eef183eac9d675c72a3bcdbf6e52`
and its supplied implementation commit does not match live Git HEAD. That
failure caused commit identity and clean-worktree authentication to become a
mandatory executable gate before the corrected report.

The first real-checkpoint backing pilot is
`/Volumes/Elements/mimo-prismwing/evidence/PW-0207/checkpoint-pilot-001.json`,
SHA-256
`84ed0150b8f20868b4ae4fa143ddf6e7536f08a00e96b2c1f4215a9629f7d942`,
from clean commit `7fae15e8fd5e5f2fb1e6b9d9e91c941a42078a0e`. Selected object
`expert:14:162` copies six authenticated checkpoint tensors totaling
25,171,968 source bytes into its declared 25,182,208-byte page-aligned owned
mapping. Every tensor byte compares exactly. Injected warning pressure removes
the object and returns declared residency to zero through the live observer
lifecycle. Maximum physical footprint is 140,266,368 bytes, minimum free
memory is 70%, and swap growth and new throttled pages are zero.

This promotes the page-aligned backing and single-object loader to the decoder
integration step. It is not a real OS warning, full 12 GiB residency,
interleaved transaction, or TPS result, so none of those gates move.

The first real decoder integration is clean commit
`53483cd1171cb4f4076d83d6310764bfdf4b813b`. It substitutes the owned
page-aligned bytes for all six projections of `expert:14:162` after prefill,
keeps the mapped checkpoint as authoritative fallback for every other tensor,
and separately accounts mapped and resident source bytes. The command refuses
resident evidence unless its supplied commit is exact clean Git `HEAD`.

An A–B–A candidate/control/candidate sequence used the same release binary,
prompt, width-eight transaction, checkpoint, and page-release policy:

| Run | Complete ms | Prefill ms | Transaction ms | Complete TPS | Committed transaction TPS | Resident source bytes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| resident 001 | 392,037.563 | 201,836.115 | 189,434.333 | 0.020406 | 0.036952 | 201,375,744 |
| mapped 002 | 388,941.511 | 199,165.996 | 189,201.042 | 0.020569 | 0.036998 | 0 |
| resident 002 | 387,227.407 | 199,542.027 | 186,885.288 | 0.020660 | 0.037456 | 201,375,744 |

The resident median transaction is 188,159.811 ms, 0.550331% faster than the
intervening mapped control. Preserve this as a lower-milestone signal, not a
promotion: there is only one mapped center run and the result is far below the
2x gate. Resident 001 was 0.123% slower than mapped 002, so the evidence also
retains the order-sensitive neutral result rather than selecting only the
favorable run.

All three reports generate the exact token sequence
`[30092,4145,5610,678,7987,315,279,19745]`, commit seven verifier-authorized
tokens, and have byte-identical proposal/verifier routes and weights after
removing timing/resource metadata. Each candidate replaces eight accesses to
the 25,171,968 source bytes, then injected warning pressure evicts the
25,182,208-byte allocation to zero. Peak RSS stays below 4 GiB, minimum system
free memory is 68%, and swap growth and new throttled pages remain zero.

Raw report and progress hashes, in run order, are:

- resident 001: `c4542b39c21335b1a27a37a6f8edab280a34d16cce7292d5a36a3cb2fca28d5e`,
  `1eb7a5e060a0500cd89d371d5ba5b08b97624c1fcea80f9ab2fad127654c6d9d`;
- mapped 002: `26da53b12edb5dee26e84207e17e33171095b21691635915b393cf71d255b5f3`,
  `52f0afe062325ea726f883ef7f62e66bfa919e1ceff97c230158f5245f071152`;
- resident 002: `72b66220dc38ca2a4e07a58624a64c45f70fabcd4aa7e481aeeb6b5d6ed86683`,
  `23f1418fdd6e1bf27dca14614745c5c9817bc53854161917951b9361938917ec`.

The single-object causal path passes and earns scaling to a bounded declared
set under the unchanged 8 GiB process ceiling. It does not authorize the full
12 GiB offline set, a 13 GiB default, endpoint TPS promotion, or abandonment of
the measured 0.550331% signal.

Clean commit `75bce3a3e63a7307894cbcdba0ec05eda0c72659` adds a ranked
complete-object prefix and resident BF16 Metal binding while preserving the
8 GiB maximum footprint and 4 GiB post-phase footprint. A requested 3 GiB
prefix stopped closed after object eight exceeded the post-phase gate, before
decoder execution. No final report was emitted; its prefill-only progress
artifact hashes to
`e3f237888a85c3555ad027a883ad4900edabde666f57be48691dcb579cebbc84`.
Do not raise the safety default to admit it.

The largest preceding prefix that passed every growth checkpoint contains
seven objects and 1,652,555,776 bytes: LM head, all three layer-zero dense MLP
projections, and attention output projections for layers 0, 1, and 10. Its
same-commit A–B–A transaction sequence is:

| Run | Prefill ms | Transaction ms | Committed transaction TPS | Resident source bytes |
| --- | ---: | ---: | ---: | ---: |
| resident set 001 | 198,054.802 | 183,194.745 | 0.038211 | 4,471,128,064 |
| mapped 003 | 197,717.146 | 188,266.418 | 0.037181 | 0 |
| resident set 002 | 197,817.729 | 189,170.613 | 0.037002 | 4,471,128,064 |

The two-candidate median is 186,182.679 ms or 0.037597 committed TPS, a
1.106803% gain over the center control. Preserve that gain, but do not promote
the set: candidate 002 is 0.480% slower than control, so the sign does not
repeat per run and the result is far below 2x. Candidate 001 reduces measured
transaction reads by 4,472,979,456 bytes, peaks at 5,152,096,256 bytes, keeps
59% system memory free, observes zero swap/throttle growth, and warning-evicts
all seven objects to zero. Candidate 002 passes the same gates and peaks at
5,541,298,176 bytes.

Tokens, acceptance, route identities, and route weights are byte-identical
after excluding timing/resource fields. Raw report/progress hashes are:

- resident set 001: `d86e499fbfb596dae0dba9f4cc1fe67d27fd1661660b5639cc69ddbdd7b141fd`,
  `f9af99f11ccab4c94b657c024da3c138218434161ed43e574927fb93f4715df2`;
- mapped 003: `92f9dbcecaa00fba281ccda9f12e3dc4c70d7e3da04e84cec281fb8a886c37ac`,
  `23bdc8d70ab983d2b3bd18b57e49ed35337fb3ac0822b3ddc45abf7feaf8ed95`;
- resident set 002: `94b602b2b683037713d3d9db20fa2ec74e20ad47f75d7e013c059e4c34250a57`,
  `3440806a7fae783d04a4bc36cc0bb7e765650dc96a248f0d16b8a1f74d9f1ebc`.

The resident ledger accounts only 4,471,128,064 of the 13,220,446,208 source
bytes predicted for eight uses of all seven objects. The causal gap is the
seven one-row proposal LM-head calls, which still use the mapped CPU path;
only the width-eight verifier LM-head call consumes resident BF16. Continue by
making that existing target-faithful Metal path authoritative for one-row
proposal logits, then repeat mapped/resident controls. Separately, scaling
beyond seven objects requires eliminating copy-time double footprint rather
than weakening the 4 GiB gate.

Clean commit `677217ac17bc23a3de2975c9e98b6dea3c491b86` closes the
proposal LM-head gap by using the same target-faithful wide BF16 Metal path for
one-row proposal logits and width-eight verification logits. Two mapped runs
take 170,063.164 and 171,232.623 ms per transaction, median 170,647.894 ms,
versus the immediately preceding 188,266.418 ms mapped control. This is a
repeatable 9.358294% transaction-wall reduction and raises committed TPS
10.324490%, from 0.037181 to 0.041020. Promote the Metal proposal path as a
real lower milestone; it changes execution location but not weights,
arithmetic, tokens, acceptance, or routes.

With that gap closed, two seven-object resident runs account the exact
13,220,446,208 bytes predicted by eight uses of the selected objects. Their
transaction times are 169,396.114 and 172,977.047 ms, median 171,186.580 ms.
Against the 170,647.894 ms mapped median, residency is 0.315672% slower and
committed TPS is 0.314678% lower. Reject this bounded prefix as a performance
default while preserving its exact substitution and pressure-safety mechanism.
The earlier partial-path 1.106803% signal is superseded for promotion, not
erased: once the missing mapped LM-head work is removed, the full causal test
is neutral/slower.

Raw report/progress hashes are:

- mapped Metal 004: `746891226963a2d4c3bcd298bd30a05246c8e69dec436c885e7db94e0f9c936c`,
  `c8bb749dd4dc2de6b7bc90a2d336863f5d4a80c097a4d8a40f712f1523d5fa86`;
- resident Metal 001: `0e84e234ada4a030c11d74468a260e23b0bb05e64c9f3b01f40841f88864592e`,
  `5044bd15cfbc81b9689c97fb85787b5a0e96d6816228700deb69bd5a489e5cda`;
- mapped Metal 005: `d7328fa5277006b913a7a9f0a8c00c6c009991e1b912df4e487804e862ff16c8`,
  `32d4c0885eab4b46da2b3a9819ba75838786ccc50faea670c79bb0f64e256f97`;
- resident Metal 002: `8b326da87426be364f34bbf57098333a7669191411c2156ade1e57044cfba410`,
  `30d546e4d33979e60b062feebd9524f9093f54bab794e08114122c95d4632a87`.

The seven-object runtime branch is rejected for speed, but PW-0207 remains
running only for the distinct loader premise: direct-to-owned loading must
remove copy-time double footprint before a larger ranked prefix can be tested
under the same safety limits. If that cannot admit more than seven objects, the
remaining high-residency pathway is closed.

Clean commit `54ab1f0c1db68b3b925b076b9e4c54bf88fd1150` replaces
mapped-source copying with authenticated offset reads directly into the owned
page-aligned mappings. This reverses the earlier 3 GiB growth rejection without
changing the safety contract: a 30-object prefix totaling 3,196,059,648 bytes
now installs with maximum install footprint 3,329,553,984 bytes, where the old
loader stopped after eight objects.

Two direct-loaded candidate transactions take 162,892.398 and 165,018.637 ms
around a same-commit 170,329.600 ms mapped control. Their 163,955.518 ms median
is 3.742205% faster, raising committed transaction TPS 3.887690% from 0.041097
to 0.042695. Both candidates substitute exactly 25,568,477,184 source bytes;
candidate 001 reduces measured reads by 25,561,776,128 bytes. Exact tokens,
acceptance, routes, and weights remain unchanged. Peak RSS is at most
3,529,293,824 bytes, minimum free system memory is 60%, swap/throttle growth is
zero, and warning pressure evicts all 30 objects to zero in declared order.

Raw report/progress hashes are:

- direct resident 001: `4c9d1f7208f68c4ac94e8f2f37982e5165199ff1798b565ea361e008d899850a`,
  `adb40d63f7acd4d73ec7b5ce0605e6b182bbb52c6b1ab4c3b71277981d7c1eb5`;
- mapped 006: `af0bd4970845c5b063847eeb7ec8cc69deb00820e13d4a6477019af338f338a8`,
  `0ede78937611ba7869f63f36cb627d69ed102cf092ed5aaeaf495aff85fbacdb`;
- direct resident 002: `e4b4f9b855b35bad1547874d64e498d28cbb7bdaa4ec027a8a68a673c48e817a`,
  `e9b542203a9ef974076229ba27fa8e8216b36188076de176ef9c82697b6af32f`.

Promote direct loading and preserve the repeatable 3.887690% TPS gain as a
lower milestone. Do not run the 32–64-token endpoint: the predeclared 2x
transaction gate still fails. The install footprint leaves a narrow, measured
margin under 4 GiB, so one final larger prefix may be falsified under the same
gate; no 12 GiB or 13 GiB path is authorized.

The final clean implementation is
`07f591aaed56fbe9aaacbedb85085a9c1f5b3c8e`. A request capped 256 MiB
below the 4 GiB post-phase ceiling selects 42 complete objects totaling
4,001,366,016 bytes. Two candidates take 160,375.060 and 159,581.309 ms around
a 169,277.593 ms mapped control. Their 159,978.184 ms median reduces
transaction wall 5.493585% and raises committed TPS 5.812923%, from 0.041352
to 0.043756. Both substitute exactly 32,010,928,128 source bytes.

Maximum install footprint is 4,146,854,912 bytes, below the unchanged
4,294,967,296-byte release gate. Maximum run peak is 4,330,487,808 bytes,
below the unchanged 8 GiB peak gate; minimum system free memory is 57%, with
zero swap growth and zero new throttled pages. Warning pressure evicts all 42
objects to zero in declared order. Tokens, acceptance, routes, and weights are
exactly unchanged.

Raw report/progress hashes are:

- resident 42-object 001: `c3daf8a6b21b497eb618ccdfba2be66583eefcbb85343b4a7c9d7528bc0f6c83`,
  `cdbde2eb1deda1b5980e2c6772edc9e9fc9e695aaf9fc5725635568ea11ce583`;
- mapped 007: `72734073fed9d109d158e551a58cde8a1030d0077ba1ed263a87882710f7976d`,
  `32987cc634504c7dad62d13cdf0b86bcd2c2cce597a8bb13253468f448ed5c9e`;
- resident 42-object 002: `7fa83f81f4658b4c11609cf0e41faf82127ce20d217ac1a674cf67fb7ef03560`,
  `189fcc2c5a0c8d685d74b15618b9febd8c215eebf6cfd71aca5ee206b583c4ee`.

Promote this conditional lower milestone and close PW-0207. It improves over
the 30-object result without erasing it, but the 2x transaction gate fails by a
wide margin, so the 32–64-token endpoint and the 12 GiB offline set remain
unauthorized. The 256 MiB reserve plus observed install overhead leaves no
credible meaningfully larger prefix under the same 4 GiB gate. Carry the
10.324490% Metal proposal gain and 5.812923% bounded-residency gain into later
combined architecture measurements; do not describe either as 50 TPS.
