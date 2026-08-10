# PW-0169 — active A770 Limited Edition BOM preflight

- Status: ready; not yet executed
- Disposition: pending
- Date: 2026-08-10
- Owner: Codex with project owner authorization
- Checkpoint/reference hashes: MiMo revision
  `63651580ca774f8504f676040460aed3e1244ac1`; PW-0151 and PW-0167 reports
  authenticated at execution
- Hardware candidate: one used Intel Arc A770 Limited Edition 16-GB card in
  the owned EPYC host; analytical pre-purchase preflight only
- Related records: PW-0151, PW-0155, PW-0167, PW-0168; E7
- Implementation commit and dirty state: pending

## Question and changed premise

PW-0168 authenticated a new exact A770 board but left only `$69` before tax and
installation parts. A domestic used Intel Limited Edition listing now matches
PW-0167's official 225-W reference-board premise and shows much larger pre-tax
headroom. Determine whether it is the preferred physical candidate without
mistaking a seller's working-order statement or rendered shipping estimate for
component validation and delivered cost.

## Exactness and red-line check

This record changes no model behavior or acceptance gate. It inherits only the
PW-0167 impossible arithmetic ceiling, reports zero accepted tokens and no TPS,
and authorizes neither purchase nor implementation.

## Contract

1. Authenticate TARGET, PW-0151, PW-0167, the byte-identical PSU photo,
   PW-0167's official Intel product capture, and dated semantic captures of
   Intel dimension/power articles and the direct active listing.
2. Fail closed unless Intel binds Limited Edition A770 to 225-W TBP, required
   8-pin plus 6-pin inputs, and exact 279.9-mm bracket-inclusive length,
   126.36-mm maximum width, and 42-mm bracket-inclusive height.
3. Bind 225-W GPU plus 170-W CPU to the PSU's 732-W combined +12-V label while
   retaining original-compatible cable presence, pinout, rail assignment,
   measured wall power, and thermals as unproved.
4. Record the listing MPN, condition, seller claim, item price, rendered
   shipping destination and price, return policy, and location. A seller claim
   is not a component test. Complete cost requires actual-destination checkout,
   tax, and every required part at or below `$500`.
5. Require measured chassis clearance and airflow. Smaller official dimensions
   improve the candidate but do not prove fit.
6. Preserve PW-0167's ReBAR and oneAPI platform gates.
7. Apply Gate 8. Record zero accepted tokens, no endpoint TPS, and no measured
   throughput-model constant changes.

## Promotion and kill rule

Prefer this listing over PW-0168 only if it remains active, its observed
item-plus-shipping leaves at least `$100` before actual-destination tax and
parts, and official power/fit facts remain favorable. Purchase remains blocked
until physical cable/clearance evidence and non-purchasing checkout total pass;
runtime work remains blocked until the reversible installed component benchmark.

## Result

Pending execution.

## Decision

Pending execution.
