# PW-0172 — A770 slower four-lane storage BOM

- Status: ready
- Disposition: unexecuted
- Date: 2026-08-10
- Owner: Codex with project owner authorization
- Model/reference: MiMo revision
  `63651580ca774f8504f676040460aed3e1244ac1`; PW-0169 through PW-0171
  authenticated at execution
- Hardware candidate: PW-0169's A770, four used Samsung PM981 256-GB drives,
  and PW-0171's passive-bifurcation quad-M.2 carrier
- Related records: PW-0151, PW-0169 through PW-0171; E7
- Implementation commit and dirty state: pending

## Question and changed premise

PW-0171 rejects the active 3.5-GB/s-per-lane storage set on cost. PW-0170 also
retains a deliberately conservative 2.5-GB/s-per-lane envelope, but at the much
harder `A=113/137` requirement for 50 TPS. Test whether slower surplus drives
change the procurement result without confusing a retail nameplate with
sustained installed bandwidth.

## Exactness and red-line check

This is a cost and device-identity preflight. It changes no model behavior,
fidelity gate, or throughput constant; inherits PW-0170's impossible envelope;
records zero accepted tokens and no TPS; and authorizes no purchase.

## Contract

1. Authenticate TARGET, PW-0169 through PW-0171, the carrier capture, a direct
   active quantity-four-or-greater exact-drive listing, and a direct retail
   specification for the listing's base part number.
2. Require four PCIe-3-x4, 256-GB PM981 drives. Treat the retailer's 2,800-MB/s
   read figure only as a preflight nameplate with 12% margin over PW-0170's
   2.5-GB/s grant. Manufacturer and installed sustained speed remain unproved.
3. Compute item-plus-observed-shipping cost for the card, four drives, and
   carrier. Report the exact remaining allowance and the break-even sales-tax
   rate assuming cables and cooling cost zero; do not silently grant unknown
   tax or missing installation parts.
4. Retain only a pre-tax BOM if it is at most `$500`. A complete BOM requires
   actual checkout plus authenticated original-compatible cables/cooling.
5. Preserve PW-0170's `q=137`, `A=77` at 34.3 TPS, and `A=113` at 50 TPS.
   Hardware affordability does not provide a proposer or measured endpoint.
6. Apply Gate 8. Report zero accepted tokens, no TPS, and no purchase.

## Result

Pending clean implementation commit and execution.
