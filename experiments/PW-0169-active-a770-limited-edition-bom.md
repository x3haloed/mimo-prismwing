# PW-0169 — active A770 Limited Edition BOM preflight

- Status: completed
- Disposition: conditional
- Date: 2026-08-10
- Owner: Codex with project owner authorization
- Checkpoint/reference hashes: MiMo revision
  `63651580ca774f8504f676040460aed3e1244ac1`; PW-0151 and PW-0167 reports
  authenticated at execution
- Hardware candidate: one used Intel Arc A770 Limited Edition 16-GB card in
  the owned EPYC host; analytical pre-purchase preflight only
- Related records: PW-0151, PW-0155, PW-0167, PW-0168; E7
- Implementation commit and dirty state:
  `c073a6dfe3890b4ed6103326094e2b6b36f3f8bf`; clean

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

The authoritative report hashes to
`b6c125f1a8cb937b0bb847936e5b251a9d65cb13f7254ca1ae215d60aa450baa`.
It authenticates TARGET, PW-0151, PW-0167, the byte-identical PSU photo,
PW-0167's official Intel product capture, Intel's two dated semantic article
captures, and the direct active listing transcription.

The exact listing identifies MPN `21P01J00BA`, matching Intel's Limited Edition
A770 16-GB board. Intel specifies 225-W TBP, required 8-pin plus 6-pin inputs,
279.9-mm length including the I/O bracket, 126.36-mm maximum width including
the connector and bracket, and 42-mm maximum bracket-inclusive height. The GPU
plus the 170-W EPYC total 395 W by nameplate, leaving 337 W below the NEX750B's
732-W combined +12-V label. Original-compatible cable inventory, pinout, rail
assignment, transients, wall power, clearance, and cooling remain unproved.

The active domestic used listing shows `$300` and `$11.71` shipping to the
renderer-provided `27709` destination, leaving `$188.29` before destination
differences, sales tax, and installation parts. The seller says the card works
and includes original packaging, but accepts no returns; that statement is not
a component test. The actual delivery checkout and complete BOM remain
unauthenticated. This is nevertheless materially better cost and fit evidence
than PW-0168's `$431` pre-tax 300-mm/285-W Photon candidate.

Gate 8 passes at 71% minimum free memory, 32,636,928-byte peak RSS,
19,957,312-byte maximum physical footprint, zero swap growth or throttling, an
explicit release boundary, and stable services. The analyzer reports zero
accepted tokens, no endpoint TPS, and no measured throughput-model constant
changes.

## Decision

Prefer item `168591709192` over PW-0168's Photon listing, but do not purchase
yet. Request physical clearance and original EVGA cable photographs plus a
non-purchasing actual-address checkout total. If those pass while the listing
remains active, purchase is still a user decision and runtime work still
requires the reversible installed oneAPI BF16/PCIe/ReBAR-off/on component
benchmark.
