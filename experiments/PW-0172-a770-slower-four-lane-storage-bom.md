# PW-0172 — A770 slower four-lane storage BOM

- Status: completed
- Disposition: conditional
- Date: 2026-08-10
- Owner: Codex with project owner authorization
- Model/reference: MiMo revision
  `63651580ca774f8504f676040460aed3e1244ac1`; PW-0169 through PW-0171
  authenticated at execution
- Hardware candidate: PW-0169's A770, four used Samsung PM981 256-GB drives,
  and PW-0171's passive-bifurcation quad-M.2 carrier
- Related records: PW-0151, PW-0169 through PW-0171; E7
- Implementation commit and dirty state:
  `e531592119502857171ca4d3b7b007e104e37969`; clean

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

The authoritative report hashes to
`2b38a618c0364ce2c11a7d93b2bf57e357c38d8cc5f3edfc2da954a6795da564`.
It authenticates TARGET, PW-0169 through PW-0171, the existing active carrier,
an active listing with six exact PM981 drives available, and a direct retail
specification for the common `MZVLB256HAHQ` base part.

Four drives cost `$115.96` with free observed shipping. Adding the `$39.99`
carrier and PW-0169's `$311.71` card-plus-rendered-shipping produces a
`$467.66` pre-tax subtotal, leaving `$32.34` for tax, original-compatible GPU
cables, and any additional cooling. On the `$455.95` taxable item subtotal,
the break-even sales-tax rate is only `7.092883%` even if cables and cooling
are free. Therefore this is a pre-tax BOM, not a complete sub-cap BOM.

Capacity is ample at 1.024 decimal TB. The retail page advertises 2,800 MB/s
for the matching base part, 12% above PW-0170's conservative per-drive grant.
That is neither manufacturer authority nor installed sustained bandwidth.
Platform bifurcation, concurrent reads, drive health, and thermal behavior all
remain unmeasured.

Gate 8 passes at 72% minimum free memory, 29,163,520-byte peak RSS,
17,680,000-byte maximum physical footprint, zero swap growth or throttling,
an explicit release boundary, and stable protected services. The report has
zero accepted tokens, no endpoint TPS, and no measured throughput-model
constant changes.

## Decision

Retain only the pre-tax four-by-2.5-GB/s A770 branch. It still requires actual
checkout, authenticated cables/cooling, installed storage and A770 evidence,
and a base-aligned `q=137` proposer achieving at least `A=113` for 50 TPS
(`A=77` for 34.3 TPS). Authorize no purchase. A retailer nameplate and an
affordable cart do not satisfy any performance or fidelity gate.
