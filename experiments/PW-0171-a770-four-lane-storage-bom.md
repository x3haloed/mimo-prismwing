# PW-0171 — A770 four-lane storage BOM

- Status: ready
- Disposition: unexecuted
- Date: 2026-08-10
- Owner: Codex with project owner authorization
- Model/reference: MiMo revision
  `63651580ca774f8504f676040460aed3e1244ac1`; PW-0169 and PW-0170 reports
  authenticated at execution
- Hardware candidate: PW-0169's used Intel A770 Limited Edition, four used
  Samsung PM981a 256-GB drives, and one passive-bifurcation quad-M.2 carrier
- Related records: PW-0151, PW-0169, PW-0170; E7
- Implementation commit and dirty state: pending

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

Pending clean implementation commit and execution.
