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
  `2bb8b5c94763c2aee79675f18372d4b05bb86f5f`; clean

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
`127a898e54f51044bf68bf58f80d071e98b2e10130f2b008a6fe0d313d2d9db3`.
It authenticates TARGET, PW-0151, PW-0167, the byte-identical PSU photo,
PW-0167's official Intel product capture, Intel's two dated semantic article
captures, the direct active listing transcription, and all four original
listing images plus their image-bound semantic transcription.

The exact listing identifies MPN `21P01J00BA`, matching Intel's Limited Edition
A770 16-GB board. The photographed box label independently shows the 16-GB
identity and product code; the card photos match the Limited Edition form and
expected inputs. Intel specifies 225-W TBP, required 8-pin plus 6-pin inputs,
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

The installed performance continuation gate is exceptionally narrow. Even
with the EPYC granted its impossible peak and zero time for every omitted
operation, the card must sustain `118.238594` BF16/F32-accumulate TFLOPS,
`90.2585%` of the derived 131-TFLOPS ceiling. Reserving only 30, 60, or 120
seconds for all other work raises that requirement to `91.7979%`, `93.3904%`,
or `96.7460%` of peak. Falling below the zero-overhead threshold rejects the
card; meeting it merely retains the branch.

Gate 8 passes at 71% minimum free memory, 31,195,136-byte peak RSS,
20,219,264-byte maximum physical footprint, zero swap growth or throttling, an
explicit release boundary, and stable services. The analyzer reports zero
accepted tokens, no endpoint TPS, and no measured throughput-model constant
changes. The earlier image-unbound `analysis-001` report remains preserved at
SHA-256
`b6c125f1a8cb937b0bb847936e5b251a9d65cb13f7254ca1ae215d60aa450baa`
and image-bound `analysis-002` at
`d08060c9fa494245069bb61169c48b0b8484c2c2796fa68b34b5cc89c892bfb9`;
both are superseded by `analysis-003`. None authenticates component function.

## Decision

Prefer item `168591709192` over PW-0168's Photon listing, but do not purchase
yet. Request physical clearance and original EVGA cable photographs plus a
non-purchasing actual-address checkout total. If those pass while the listing
remains active, purchase is still a user decision and runtime work still
requires the reversible installed oneAPI BF16/PCIe/ReBAR-off/on component
benchmark. Reject immediately if its source-shape-weighted sustained BF16/F32-
accumulate throughput is below `118.238594` TFLOPS.
