# PW-0213 — Uncached page-aligned stream transport

- Status: proposed
- Disposition: unexecuted
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

Unexecuted. Scheduled after PW-0212 because route lead-time evidence can inform
later `F_RDADVISE`, while the first transport substitution remains predictor-
free and causally isolated.
