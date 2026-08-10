# PW-0171 — A770 four-lane storage BOM

- Status: completed
- Disposition: rejected
- Date: 2026-08-10
- Owner: Codex with project owner authorization
- Model/reference: MiMo revision
  `63651580ca774f8504f676040460aed3e1244ac1`; PW-0169 and PW-0170 reports
  authenticated at execution
- Hardware candidate: PW-0169's used Intel A770 Limited Edition, four used
  Samsung PM981a 256-GB drives, and one passive-bifurcation quad-M.2 carrier
- Related records: PW-0151, PW-0169, PW-0170; E7
- Implementation commit and dirty state:
  `6d5d1d5ec5ac53f4a0ab50ade6ff9881cf55f5b9`; clean

## Question and changed premise

PW-0170 leaves only a four-independent-lane, wide-speculation A770 envelope.
PW-0169's card costs `$311.71` with rendered shipping, leaving `$188.29` before
tax and every missing part. Determine whether a currently purchasable four-lane
storage set that matches the favorable `3.5 GB/s`-per-lane nameplate can fit
that remainder. This is a procurement falsification, not a storage benchmark.

## Exactness and red-line check

This record changes no model behavior, fidelity gate, or throughput constant.
It inherits PW-0170's analytical acceptance requirements, records zero accepted
tokens and no endpoint TPS, and authorizes no purchase.

## Contract

1. Authenticate TARGET, PW-0169, PW-0170, a direct active quantity-four drive
   listing, a direct active quad-carrier listing, and Samsung's exact 256-GB
   PM981a product specification by SHA-256.
2. Require four PCIe-3-x4, 256-GB drives with a manufacturer 3,500-MB/s
   sequential-read nameplate and a carrier explicitly requiring x4/x4/x4/x4
   bifurcation. Reject stale, sold, quantity-short, or model-ambiguous listings.
3. Compute the most favorable cost lower bound: charge four item prices, only
   one observed drive-order shipping charge, the carrier and its shipping, and
   PW-0169's card-plus-shipping. Leave tax, cables, and additional cooling at
   zero/unknown so exceeding `$500` is decisive for this exact BOM.
4. Keep capacity and nameplate viability separate from procurement. Do not
   convert Samsung's internal sequential result into sustained concurrent host
   bandwidth or endpoint TPS.
5. Reject this exact BOM if its favorable lower bound exceeds `$500`. Retain
   the mechanism only if a cheaper complete BOM appears; do not generalize a
   dated market result into physical impossibility.
6. Apply Gate 8 and report zero accepted tokens, no TPS, and no purchase.

## Result

The authoritative report hashes to
`14549b38ee1daee523fd5a76ca9654cdcf7aa6284c651fb36eccac68908b28d3`.
It authenticates TARGET, PW-0169, PW-0170, Samsung's exact 256-GB PM981a
specification, and direct active listings for four tested used drives and a
quad-M.2 passive-bifurcation carrier.

Capacity is not the failure. Four 256-decimal-GB drives provide
`1,024,000,000,000` bytes, `708,285,946,598` beyond the complete checkpoint.
Samsung specifies 3,500 MB/s sequential read for the exact drive, so the four
nameplates sum to PW-0170's favorable 14-GB/s premise. Neither concurrent
sustained reads nor the owned host's bifurcation path has been measured.

Cost rejects the exact active set before those measurements. The drives cost
`4 * $39.99 = $159.96`; granting only one observed `$8.15` order shipping
charge and the carrier's `$39.99` delivered observation makes storage at least
`$208.10`. Adding PW-0169's `$311.71` card observation gives `$519.81`, already
`$19.81` over the complete cap before sales tax, GPU cables, or additional
cooling. More favorable combined shipping cannot improve this lower bound
because it already charges shipping only once.

Gate 8 passes at 72% minimum free memory, 29,048,832-byte peak RSS,
17,958,400-byte maximum physical footprint, zero swap growth or throttling,
an explicit release boundary, and stable protected services. The report has
zero accepted tokens, no endpoint TPS, and no measured throughput-model
constant changes.

## Decision

Reject the current active A770/four-PM981a/quad-carrier BOM. This supersedes
PW-0169/PW-0170's inference that `$188.29` was credible complete-BOM room; it
does not prove that a cheaper future listing, already-owned storage, or a
different four-lane embodiment cannot fit. Retain the mechanism as a price-
triggered branch only. Authorize no purchase.
