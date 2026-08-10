# PW-0155 — Owned EPYC two-P100 installable-BOM prerequisite

- Status: completed
- Disposition: rejected
- Date: 2026-08-09
- Owner: Codex with project owner authorization
- Checkpoint/reference hashes: MiMo revision
  `63651580ca774f8504f676040460aed3e1244ac1`; PW-0151 analysis
  `d6919e47f0f4495ccac2ad56ebcfe6662b3309aebd3296c6b546a50836829cb1`;
  PW-0154 analysis
  `1b57250d45f1b24e32f43e93a653fc3d00fa061e37cd0df1c6f0fdff551535f2`
- Hardware candidate: two passive P100 PCIe cards in H11SSL-i slots 2 and 4;
  one passive four-drive M.2 carrier in slot 6 under `x4x4x4x4`; photographed
  EVGA NEX750B; analytical pre-purchase prerequisite only
- Related records: PW-0151 through PW-0154; E7
- Implementation commit and dirty state:
  `1d67bcd1ac3795cf76de0f0374898c2777ba6b32`, clean

## Question and changed premise

PW-0154 materially lowers the exact streamed-expert bytes, but still names
two P100s and up to four independent NVMe lanes without proving that those
parts can coexist in the owned host or be delivered inside the complete
`$500` cap. The project owner supplied the PSU label again, byte-identical to
PW-0151's authenticated photo. Close the logical slot and connector topology,
then test one dated component ledger without converting listing text into
installation or performance evidence.

## Exactness and red-line check

This is target-faithful L1 placement and procurement analysis. It changes no
weights, arithmetic, routing, modalities, thresholds, or endpoint semantics.
No hardware is purchased or energized. A component subtotal, interface name,
or supported slot mode is not a delivered BOM, measured bandwidth, safe power
path, or endpoint result.

## Contract

1. Authenticate PW-0151, PW-0154, the official H11SSL manual, the official
   P100 PCIe product brief, and a dated structured market observation by
   SHA-256. Preserve the market transcription's weaker evidence class.
2. Bind slots 2, 4, and 6 as PCIe 3.0 x16 and require the manual's
   `x4x4x4x4` option. Assign two double-width P100s and one single-slot passive
   quad-M.2 carrier without claiming chassis or blower clearance.
3. Bind the P100's 250-W total graphics power, CPU-style 8-pin input, maximum
   240 W/20 A auxiliary input, and NVIDIA dongle part `030-0571-000`. Reject
   direct PCIe-cable insertion or unauthenticated pinout.
4. Bind the photographed PSU's 750-W continuous and 61-A/732-W combined +12-V
   limits. Report GPU-plus-CPU nameplate headroom and whether one P100's
   auxiliary maximum can consume an entire labeled 20-A rail. Do not infer
   simultaneous measured load from TDP or nameplates.
5. Price both cards, four drives, one carrier, two dongles, and two cooling
   kits. A purchase gate requires delivered tax and shipping, exact device
   identity, original-compatible PSU cable inventory, physical fit, pinout,
   airflow, and return/health evidence; an under-cap subtotal is insufficient.
6. Fail the captured storage line if the title and item specifics do not bind
   one exact model. Never treat PCIe/NVMe labeling as sustained cold-read
   evidence.
7. Preserve staged electrical stop conditions: no mixed modular cables; no
   passive-card load without forced airflow; stop on OCP, ECC faults,
   throttling, unsafe temperature, or wall power above either the installed
   PSU envelope or the formal 1,000-W cap.
8. Apply Gate 8. Report zero accepted tokens and no endpoint TPS.

## Promotion and kill rule

Promote only a dated complete delivered BOM at or below `$500` whose exact
parts have a credible physical, electrical, and thermal path. Reject the
captured BOM as purchase authority if any required delivered cost, device
identity, cable/pinout, fit, cooling, or power fact is unknown. A rejection
does not kill the abstract two-P100/quad-NVMe architecture; it names the next
facts required to reopen procurement.

## Result

The authoritative `analysis-001` manifest hashes to
`226603fb2b44e1162a038f51bae47520238150f3b26e39e1cf33c7420b88b064`.
It authenticates PW-0151/PW-0154, the official H11SSL manual, the official
P100 product brief, the byte-identical PSU photo transitively through PW-0151,
and the explicitly weaker dated market transcription.

The logical PCIe layout survives. H11SSL-i slots 2, 4, and 6 are x16 and each
offers `x4x4x4x4`; a candidate can place double-width P100s in slots 2 and 4
and a single-slot passive quad-NVMe carrier in slot 6. That proves lane
availability, not chassis, blower, cable, or obstruction clearance.

The electrical margin is not installation evidence. Two 250-W P100 board
limits plus the EPYC's 170-W TDP total 670 W, leaving only 62 W below the
PSU's 732-W combined +12-V label for the motherboard, memory, four drives,
fans, and transients. NVIDIA specifies up to 240 W/20 A at each P100's CPU
8-pin auxiliary input, exactly the labeled current of one NEX750B rail. The
official `030-0571-000` dongle accepts two PCIe feeds, but the captured cheap
dongles are unbranded and their pinout/construction is not authenticated.

The named cards, four drives, carrier, two dongles, and two cooling kits total
`$403.38`, leaving `$96.62` before tax, destination shipping, missing original
EVGA VGA cables, or fit remedies. This is not a complete delivered BOM. The
drive listing title names `SSDPEKKF256G8L`, while its item specifics name an
unbranded `KBG40ZNT256G`; it has no authenticated delivered identity,
sustained-read result, or return path. Cooling fit and temperature also remain
unproven.

Gate 8 passes across five snapshots with 67% minimum free memory,
45,236,224-byte peak RSS, 18,220,544-byte maximum physical footprint, zero
swap growth or new throttled pages, and stable protected services. The first
clean invocation failed before manifest publication because a binary float
represented `$403.38` as `$403.38000000000005`; the rerun rounds monetary
ledgers to cents. Accepted tokens and endpoint TPS remain zero.

## Decision

Reject the captured dated component list as purchase authority. Retain the
abstract two-P100/quad-NVMe topology as conditional because the motherboard
supports it and the named subtotal is below cap, but do not buy or energize it.
Reopen procurement only after exact SSD identity and delivered cost, original
EVGA cable inventory, authenticated dongles, chassis/cooler clearance, and a
staged rail/power/thermal validation plan are bound. Continue with the cheaper
8K route-coverage/prefill falsification before requesting those physical acts.
