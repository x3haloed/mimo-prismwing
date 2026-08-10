# PW-0168 — active A770 Photon exact-board BOM preflight

- Status: completed
- Disposition: conditional
- Date: 2026-08-10
- Owner: Codex with project owner authorization
- Checkpoint/reference hashes: MiMo revision
  `63651580ca774f8504f676040460aed3e1244ac1`; PW-0151 and PW-0167 reports
  authenticated at execution
- Hardware candidate: one new GUNNIR Intel Arc A770 Photon 16G OC in the owned
  EPYC host; analytical pre-purchase preflight only
- Related records: PW-0151, PW-0155, PW-0167; E7
- Implementation commit and dirty state:
  `4490713479a5734656e41a131113609d74765ba9`; clean

## Question and changed premise

PW-0167 found the first affordable ordinary-dense one-million arithmetic
survivor but had only sold market observations and Intel's 225-W reference-card
nameplate. A direct active listing now offers an exact new 16-GB GUNNIR Photon
board. Determine whether its official dimensions, connectors, 285-W TBP, dated
price, and the photographed PSU leave an installable purchase branch without
mistaking item-plus-shipping for a delivered BOM.

## Exactness and red-line check

This record changes no model behavior, weights, attention, capability, or
acceptance threshold. It inherits only PW-0167's impossible arithmetic ceiling,
reports zero accepted tokens and no TPS, and authorizes neither purchase nor
runtime implementation.

## Contract

1. Authenticate TARGET, PW-0151, PW-0167, the byte-identical PSU photo, the
   official GUNNIR product HTML and specification panel, and dated listing
   transcription by SHA-256.
2. Fail closed unless the official page binds the exact Photon identity to the
   authenticated panel and its transcription records 16 GB, 2x8-pin inputs,
   285-W TBP, and 300x118.5x50-mm dimensions.
3. Replace the 225-W reference-board power premise only for this exact board.
   Bind 285-W GPU plus 170-W CPU to the PSU's 732-W combined +12-V label and
   retain cable presence, pinout, rail assignment, wall power, and thermals as
   unproved installation gates.
4. Record the active item price, shipping, import-fee statement, quantity, and
   seller location. A complete BOM passes only with authenticated destination
   tax and every required installation part at or below the fixed `$500` cap.
5. Require measured chassis length, height, adjacent-slot clearance, and
   airflow before installation; board dimensions alone are not fit evidence.
6. Preserve PW-0167's ReBAR and oneAPI operating-system prerequisites. An
   active listing does not repair the unsupported platform.
7. Apply Gate 8 to the analyzer. Preserve zero accepted tokens, no endpoint
   TPS, and no measured throughput-model constant changes.

## Promotion and kill rule

Promote only to a physical/checkout evidence request if the exact active board
remains below the cap before the still-unknown costs and has positive combined
+12-V nameplate margin. Reject the listing if its authenticated delivered BOM
exceeds `$500`, it cannot fit, original-compatible cabling cannot place both
inputs safely, or the listing ceases to be active. Purchase remains a separate
user decision after the installed component benchmark prerequisite.

## Result

The authoritative report hashes to
`dfd12ca7bb331003e28241e1c5eac49c579eecfa90cb5216fb41edb8a297f6bd`.
It authenticates the exact GUNNIR product page and specification panel, the
dated active listing transcription, PW-0151, PW-0167, TARGET, and the
byte-identical PSU photo.

The official panel resolves two marketplace uncertainties. This exact board
uses two 8-pin inputs, not three, and its TBP is 285 W rather than Intel's
225-W reference-card figure. It measures 300x118.5x50 mm. The GPU plus the
170-W EPYC total 455 W by nameplate, leaving 277 W below the NEX750B's 732-W
combined +12-V label. A candidate cable plan uses VGA1 on +12V2 and VGA3 on
+12V4, but the presence and pinout of original-compatible cables, transients,
wall power, clearance, and cooling remain unproved.

The active new-card observation is `$411` plus `$20` shipping, with four shown
available and import fees stated included. That leaves `$69` before unknown
destination sales tax and any missing installation parts. Therefore the
record reverses only the lack of active inventory; it does not pass the fixed
`$500` complete delivered-BOM gate. The inherited absent native ReBAR and
unsupported owned Debian 13 oneAPI platform gates also remain open.

Gate 8 passes at 70% minimum free memory, 31,162,368-byte peak RSS,
20,252,224-byte maximum physical footprint, zero swap growth or throttling, an
explicit release boundary, and stable services. The analyzer reports zero
accepted tokens, no endpoint TPS, and no measured throughput-model constant
changes. The first invocation failed closed before output because normalized
HTML had discarded the official panel URL; the corrected clean commit binds
the raw HTML attribute and produced the sole report.

## Decision

Retain this exact active card as a conditional candidate. Do not purchase yet.
The next gate requires photographs or measurements proving 300-mm card and
50-mm cooler clearance, photographs authenticating two original EVGA VGA
cables and both connector ends, and a non-purchasing checkout total for the
actual delivery address. If those pass, the branch still requires a reversible
installed oneAPI BF16/PCIe/ReBAR-off/on component benchmark before runtime
implementation or any performance promotion.
