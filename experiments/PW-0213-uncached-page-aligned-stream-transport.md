# PW-0213 — Uncached page-aligned stream transport

- Status: complete
- Disposition: runtime transport rejected; isolated file-backed, overlap, and
  install gains preserved; trailing-drop tuning rejected as redundant
- Date: 2026-08-11
- Execution mode: L1 exact acquisition substitution
- Related records: PW-0108, PW-0110, PW-0207, PW-0212, AN-0002

## Hypothesis and mechanism

The current mapped/cacheable source path may create a reclaimable file-backed
copy of the routed stream in addition to owned or Metal-visible buffers. Under
pressure, that hidden representation can evict useful spine pages or force
revalidation even when nominal SSD bandwidth is adequate.

Hypothesis: page-aligned widened `pread` from a Darwin `F_NOCACHE` descriptor,
with automatic `F_RDAHEAD` disabled and bounded reusable owned buffers, reduces
file-backed growth and exposed acquisition wall without changing source bytes
or arithmetic. Trailing advice and double-buffer overlap are separate steps,
not granted to the first substitution.

## Contract

Authenticate exact checkpoint offsets, lengths, and payload hashes. Widen only
to page boundaries, slice the requested bytes exactly, and charge every widened
physical byte. Preserve demand priority and bounded residency. Record cold and
warm state, `pread` wall, physical reads, logical bytes, file-backed and process
footprint, compressor/swap pressure, page faults where available, Metal wait,
complete layer/verifier wall, and Gate 8.

The deterministic fixture must cover unaligned starts and ends, adjacent and
duplicate ranges, short reads, EOF, and injected I/O failure. The candidate
must reproduce source bytes and final output exactly. Do not combine route
prediction, repacking, compression, or changed arithmetic into this record.

## Cheap falsifier and gates

First substitute one real production-shaped expert object and one complete
routed-layer acquisition. Preserve any smaller repeatable gain, but authorize a
verifier pilot only if the candidate reduces file-backed growth by at least 50%
or cold attributed acquisition wall by at least 10%, while read amplification
stays at or below 1.05x and all safety gates pass.

Stop after one correct negative layer result; do not tune read-ahead distance,
buffer count, and cache advice simultaneously. If the exact substitution
passes, add a two-buffer overlap candidate and only then a measured trailing-
drop distance. Runtime promotion requires an interleaved repeatable complete-
path TPS gain of any positive size; missing 50 TPS is not a kill condition.

## Decision

The deterministic fixture covers unaligned starts and ends, adjacent and
duplicate ranges, short-read retries, EOF, and injected I/O failure. The real
run authenticates all 48 raw-checkpoint tensor offsets and payload hashes for
one source-FP8 routed layer. Each logical range is widened independently to 16
KiB pages and exact logical bytes reconstruct the authenticated streams. Read
amplification is 1.004312x for one expert and 1.003905x for the layer.

In three cold interleaved repetitions, the cacheable control leaves 100% of
probed source pages resident after both scopes. `F_NOCACHE` plus
`F_RDAHEAD=0` leaves 0% resident in every trial: a repeatable 100% reduction
that passes the frozen 50% file-backed continuation gate. Every trial records
at least 95% of logical bytes as physical reads.

The benefit is not acquisition speed. In the final three-way run, complete-
layer medians are 70.219040 ms cacheable, 75.611917 ms sequential uncached,
and 72.191416 ms two-buffer uncached. Two buffers recover 4.523759% against
sequential uncached, but remain 2.808891% slower than cacheable control. One-
expert medians are 9.002251, 9.881750, and 9.023959 ms respectively; two
buffers recover 8.680558% and remain 0.241140% slower than control. Preserve
the overlap gains and the control regressions together.

Gate 8 passes across 21 snapshots with 64% minimum free memory,
11,388,352-byte maximum physical footprint, 11,273,216-byte final footprint,
zero swap growth or throttling, and stable protected services. This is
acquisition evidence only: `A=0`, accepted tokens zero, and no TPS claim.

Raw report:
`/Volumes/Elements/mimo-prismwing/evidence/PW-0213/uncached-stream-transport-004.json`,
SHA-256
`51b2898314ff42ecca0eb7e29802f23346a329244126cd566dea80da9171f17f`,
clean implementation commit `3c63ded58483abf76c74d3bd2b0347658d53307b`.
Validated analysis:
`/Volumes/Elements/mimo-prismwing/evidence/PW-0213/analysis-002/manifest.json`,
SHA-256
`764fba9b12d8bacc5d4d2cd7f1fc57a42323a94bc90717bc41fa948842803fe3`.

The authorized verifier pilot holds the frozen prompt, one declared
`expert:14:162` resident object, q8 proposer/verifier, arithmetic, routes,
warning eviction, and output fixed in a cacheable-control / uncached-candidate /
cacheable-control sequence. All three emit token IDs
`[30092,4145,5610,678,7987,315,279,19745]`, retain seven proposal rows, and
execute exactly 201,375,744 resident source bytes.

The candidate install moves 25,171,968 logical bytes as 25,264,128 widened
bytes (1.003661x) in 9.281750 ms versus a 10.069563 ms cacheable-control
median, a real 7.823701% install-wall reduction. Its complete wall is
403,927.191 ms versus a 406,436.586 ms control median, superficially a
0.621249% accepted-TPS gain. That gain is not causal: the candidate's prefill,
which completes before the transport is installed, is 1.716725% faster. From
the install boundary onward, candidate wall is 196,155.228 ms versus
195,035.447 ms control, regressing accepted TPS by 0.570865%. Proposal wall
regresses 1.007388%; verification wall improves 1.145070%. Preserve every
phase rather than attributing prefill noise to transport.

All verifier runs retain exact output, zero final resident bytes, zero swap or
throttling growth, stable protected services, at least 64% free memory, and
sub-367 MB peak RSS. Raw report hashes are control 1
`74b8d15186aeff9af97393643fe11bf39c11e0d0d2d2da7777493723b538ec0d`,
candidate
`f05c8b8be0794aa1aa1a2e790a96ac3c1a967439e4c0fda1ad8fb6a4eb37eee5`,
and control 2
`be004a835f59c99eb2539ae54dd434ce05748840eb48caffc1dba8e83c88baca`
at clean commit `f096b8e1a5aefa9515b69ce209c11fe96c22ae2d`. Validated
analysis
`/Volumes/Elements/mimo-prismwing/evidence/PW-0213/analysis-003/manifest.json`
hashes to
`e4a3d0696705cc598e16b1568659a28c19a8fe0f6124155e00e882420595b6b7`.

Reject runtime promotion: the only mechanism-causal endpoint interval
regresses, and one candidate cannot satisfy repeatability. Preserve the
isolated 100% file-backed reduction, 4.523759% two-buffer recovery, and
7.823701% install gain as measured lower-level advances. Do not test trailing
drop because direct `mincore` already observes zero source pages after every
uncached read. Reopening requires a different critical cut or a separately
frozen pressure workload, not reinterpretation of the prefill-only gain.
