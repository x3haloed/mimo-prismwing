# PW-0170 — single-A770 storage and acceptance envelope

- Status: ready
- Disposition: unexecuted
- Date: 2026-08-10
- Owner: Codex with project owner authorization
- Checkpoint/reference hashes: MiMo revision
  `63651580ca774f8504f676040460aed3e1244ac1`; PW-0112 route trace
  `584d3a8b1b09b12d4f83908be1fa5471b9fd66373500cc56332213928cd0bc3e`;
  PW-0169 preferred A770 report
  `127a898e54f51044bf68bf58f80d071e98b2e10130f2b008a6fe0d313d2d9db3`
- Hardware: analytical pre-purchase envelope for the active used Intel Arc
  A770 Limited Edition 16GB candidate and the owned EPYC host
- Related records: PW-0112, PW-0151 through PW-0155, PW-0167, PW-0169
- Implementation commit and dirty state: pending

## Question and changed premise

PW-0169 establishes that the preferred active A770 has a credible card-only
price, power margin, and exact form, but it does not establish a complete
runtime. Quantify the strongest source-preserving decode envelope after the
card's 16 GB HBM reserves every common source tensor, three bounded layer
arenas, and exact 8K BF16 KV. Apply the remaining whole-expert capacity to the
real prompt-calibrated `q=137` route and determine the storage lanes and
single-transaction acceptance still required.

## Exactness and red-line check

This is target-faithful L1 capacity and placement analysis over unchanged
source bytes and exact routes. It grants the A770 its derived 131-TFLOPS
BF16/F32-accumulate peak and the EPYC its impossible FP32 peak. Those are
upper bounds, not installed performance. No proposed token is accepted, no
nameplate bandwidth is TPS, and no purchase is authorized.

## Contract

1. Authenticate TARGET, config, checkpoint census, PW-0112 route trace,
   PW-0152 proposer prerequisite, PW-0154 cache result, PW-0155 slot topology,
   and PW-0169 exact card/listing report by SHA-256.
2. Reserve all non-routed source tensors, PW-0128's three maximum arenas, and
   exact 8K BF16 KV from 16 decimal GB. Cache only complete 25,171,968-byte
   layer-expert records in the remainder.
3. Select residents causally from the 87-position prompt frequency, then apply
   the frozen set to PW-0112's following 137 positions. Report access and
   union hits; logical hits are not measured avoided I/O.
4. Grant one through four independent 2.5/3.5-GB/s storage lanes. Add storage
   and mandatory matrix time serially. Grant the A770 131 TFLOPS plus the EPYC
   0.7424-TFLOPS impossible peak, but preserve PW-0169's stricter installed
   1M continuation threshold separately.
5. Report the minimum integer `A` for 34.3 and 50 TPS. Width-eight/16 blocks
   fail when their one-transaction structural maximum is below `A`; do not
   compose chained target transactions.
6. A four-lane nameplate survivor is only conditional. It still requires
   measured sustained storage, installed BF16, ReBAR-off/oneAPI stability, a
   base-aligned `q>=137` proposer, exact target correction, physical fit,
   compatible PSU cables, cooling, actual checkout tax, and a complete BOM.
7. Reject the HBM cache as a primary mechanism if it avoids less than 30% of
   union bytes. Reject the card branch if even four 3.5-GB/s lanes need
   `A>137`, or if card plus required parts cannot fit the remaining budget.
8. Apply Gate 8 and report zero accepted tokens, no endpoint TPS, and all
   omitted costs.

## Result

Pending clean execution.
