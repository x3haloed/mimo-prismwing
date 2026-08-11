# PW-0213 — Uncached page-aligned stream transport

- Status: in progress; isolated acquisition stage complete
- Disposition: one bounded verifier pilot authorized by the file-backed gate;
  trailing-drop tuning rejected as redundant
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

Authorize one bounded verifier pilot because the predeclared file-backed gate,
not the transfer-wall gate, passed. Do not promote the transport yet: isolated
transfer wall still regresses against cacheable reads. Do not test trailing
drop under this candidate; direct `mincore` already observes zero resident
source pages after every uncached read, leaving no source cache for advice to
discard. Runtime promotion still requires exact output and a repeatable
positive complete-path TPS result of any size.
