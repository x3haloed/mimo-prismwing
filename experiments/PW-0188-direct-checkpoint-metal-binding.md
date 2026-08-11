# PW-0188 — Direct-checkpoint page-offset Metal binding

- Status: completed
- Disposition: promoted to direct-checkpoint FP8 projection integration
- Date: 2026-08-10
- Model/reference: MiMo revision
  `63651580ca774f8504f676040460aed3e1244ac1`
- Execution mode: target-faithful physical-substrate probe
- Hardware/runtime: existing Apple M1 and internal checkpoint storage
- Related records: PW-0106, PW-0108, PW-0111, PW-0187

## Question and causal mechanism

Can Metal consume an expert tensor directly from its original safetensors shard
without copying it or constructing the rejected approximately 303 GB repacked
expert bank? Map the authenticated source shard read-only, round the tensor's
absolute byte interval outward to host VM pages, create one
`newBufferWithBytesNoCopy` over that page-aligned interval, and bind the tensor
at its nonzero intra-page offset.

Run a one-thread Metal kernel that reads the first, middle, and final logical
tensor bytes. Compare them with CPU reads from the exact mapped tensor and
record the page size, base alignment, logical and mapped bytes, offset, source
copy bytes, compile time, cold and warm complete command times, device, source
and tensor hashes, and commit. This is a physical correctness probe, not a
kernel, layer, token, or endpoint performance result.

## Gate

Promote direct-checkpoint wide-verifier integration only if the original real
expert tensor has a nonzero buffer offset; the covering base and length are
page aligned; mapped bytes are no more than logical bytes plus two host pages;
the GPU reproduces every sampled byte exactly; the source-copy ledger is zero;
and the command completes on the existing M1. Fail closed on an out-of-range
cover, unexpected tensor identity, Metal error, or any mismatch.

If the gate passes, replace repacked-artifact dependencies in the wide union
executor with authenticated shard mappings and explicit tensor offsets. Do not
claim endpoint TPS until the complete accepted-token path runs.

## Result

The real layer-46 expert-28 gate tensor starts at nonzero intra-page offset
`14,712` within a 16,384-byte host page. A page-rounded 8,404,992-byte mapping
covers the 8,388,608-byte logical tensor with less than two pages of overhead,
and its base is exactly page aligned. Metal binds that mapping without copying
source bytes and reads logical byte indices `0`, `4,194,304`, and `8,388,607`
as `[232,218,209]`, exactly matching the CPU mapping.

The Apple M1 command completes in 1.266 ms cold and 0.125 ms warm median. These
three sampled reads are not an acquisition or throughput measurement. The
source shard hashes to
`7b92a89c4710b0253a15f1355567bbfc94b57cb8fb8a6dbddca01bacf12d0985`,
the tensor hashes to
`9511b00f1b2b4c536b9614565ab6fa363d5cbbc5d8df25a22d46e9c06f9f00af`,
and the authoritative report hashes to
`9e9bfd44287ab2c74df915d6c242320145387366f34755e7dcdd918f30ae4a7a`.

Promote direct-checkpoint FP8 projection parity using the same page-offset
binding. The approximately 303 GB repacked bank is no longer a prerequisite
for the wide verifier. Zero tokens are accepted and no throughput constant
changes.
