# PW-0166 — affordable Xe2 complete-system envelope

- Status: completed
- Disposition: rejected
- Date: 2026-08-10
- Owner: Codex with project owner authorization
- Checkpoint/reference hashes: MiMo revision
  `63651580ca774f8504f676040460aed3e1244ac1`; config
  `292a60e74ae9a6d53422b31b21468ce2111c0ab3f7f7a4f4e9c7cd5133b96587`;
  authenticated PW-0127, PW-0151, PW-0158, and PW-0161 authorities; official
  Intel B-series product page
  `f823f01910776e04f4ac5b3bb151b960cb857c96630ca9cbead15a86986679c8`;
  oneAPI Xe architecture page
  `ae4b7eaa179b7eabb5383b951f7b6bd8ae27058f727c724a534acc835899881f`;
  Intel AI-datatype support page
  `79c9b9a32ccb7d1869777d85384cd06ddc4b2238218eead74cd03c978a40f3d1`;
  B-series QRG
  `6957f49863018e0226b126f5500a97304ff7cab2a9fe61e75019cc7db51b1d4e`;
  launch release
  `597c943c6a4a7ab6d929a4f47c6731fe45427d1b5715bd7369765f8b3437e934`;
  Intel IGC commit `2eefea9414f2064b2250045305b28a2f73d4f644`, DPAS specification
  `79ba16ab6716e9099aaaf88875d7213c1a2581601aae8fd7e20fcd70d7737170`,
  and Xe2 latency table
  `17502f5b5050ec5538ae3424d09d07a6aea5d32f92b01d71b221bb58f60800c6`
- Hardware candidate: one Intel Arc B580 12-GB card in the owned EPYC host;
  analytical pre-purchase envelope only
- Related records: PW-0127, PW-0151, PW-0158, PW-0161, PW-0164, PW-0165; E7
- Implementation commit and dirty state:
  `924bc76acc6967a61e203dd0869efb0de12dd485`, clean

## Question and changed premise

PW-0164 and PW-0165 reject the current sub-`$500` RTX 5060 Ti and RX 9060 XT
ordinary-dense controls. Intel officially lists B580 at 233 peak INT8 XMX TOPS,
but does not directly publish its BF16 peak. The rate cannot remain unresolved
if Intel's own instruction semantics and compiler scheduling model close it.

The pinned Intel DPAS specification assigns two operations per channel to
BF16 and four to INT8. The pinned compiler's Xe2 scheduling model assigns DPAS
latency by repeat count, not precision, and its occupancy model likewise does
not distinguish BF16 from INT8 at the same execution size and destination
width. Derive the source-oriented BF16/F32-accumulate ceiling as exactly half
of the official INT8 XMX peak, then apply the complete one-million arithmetic
floor.

## Shared construction and compression-depth contract

Capability invariant: preserve every source weight, one million positions,
the pinned nine global and 39 sliding attention layers, native modality paths,
and every TARGET gate. Ordinary dense attention and weights remain the
control.

Authorized embodiment boundary: grant perfect BF16 XMX utilization, the
EPYC's impossible peak concurrently, and zero time for every omitted
operation. This is a source-oriented L1 ceiling, not achieved oneAPI
performance. Do not grant INT8 arithmetic to unchanged BF16 activations or
call the 233-TOPS marketing row BF16.

## Contract

1. Authenticate TARGET, config, PW-0127, PW-0151, PW-0158, PW-0161, all five
   official Intel captures, and both pinned IGC files by SHA-256. Fail closed
   on source or compiler drift.
2. Bind B580 to Xe2-HPG, XMX, 12 GB, 233 peak INT8 TOPS, 190 W, and official
   `$249` launch price without treating launch MSRP as a delivered BOM.
3. Bind BF16 and INT8 XMX support. Require the pinned DPAS semantics to assign
   `OPS_PER_CHAN=2` to 16-bit BF16 and `OPS_PER_CHAN=4` to 8-bit INT8.
4. Require Xe2 DPAS latency to depend on repeat count rather than precision,
   with the same generic occupancy path for same-size BF16/F32-accumulate and
   INT8/int32-accumulate instructions. Derive 116.5-TFLOPS BF16 peak.
5. Add two operations per PW-0127 mandatory MAC at one million positions to
   PW-0158's exact attention work. Grant 116.5 TFLOPS plus the EPYC's
   impossible peak concurrently.
6. Permanently reject ordinary-dense B580 if that arithmetic floor alone
   exceeds 1,800 seconds. Do not generalize to changed attention, changed
   weights, multi-card execution, or a faster Xe2 SKU.
7. Recompute exact 1M BF16 KV, three arenas, and non-routed source tensors
   against 12 decimal GB. Capacity is a streaming requirement, not an
   independent compute proof.
8. Bind 190-W board power and the photographed PSU's 732-W combined +12-V
   ceiling without calling nameplates cable, fit, thermal, or measured-load
   proof.
9. Apply Gate 8. Record zero accepted tokens and no endpoint TPS. Do not
   present the derived ceiling as achieved B580 performance.

## Promotion and kill rule

Permanently reject B580 for ordinary-dense one-million-position execution if
the BF16/Xe2 arithmetic floor exceeds 1,800 seconds under the impossible
concurrent CPU grant. This rejection survives future price changes.

Do not reject changed-attention L3/L4 modes, lower-precision modified weights,
Arc Pro B60/B65/B70, multi-card systems, or future Intel products. Authorize
no purchase or oneAPI runtime from analytical evidence.

## Result

The authenticated analyzer derives a `116.5`-TFLOPS BF16/F32-accumulate
ceiling from Intel's official 233-INT8-TOPS B580 row and the pinned IGC DPAS
semantics and Xe2 scheduler. Mandatory one-million-position matrices plus
ordinary attention total `214,165,790,024,007,680` operations. Even granting
the owned EPYC's impossible `0.7424`-TFLOPS peak concurrently, their lower
bound is `1,826.6923060599893` seconds, already `26.692306059989278` seconds
beyond the complete TTFT gate before every omitted operation, transfer, or
runtime cost.

Exact BF16 1M KV exceeds the card's 12-decimal-GB capacity by
`11,065,559,040` bytes. KV, three maximum layer arenas, and non-routed source
tensors exceed it by `26,221,107,536` bytes, so even a hypothetical arithmetic
survivor would require host or storage streaming. The official `$249` launch
price is not a delivered BOM. The 190-W board plus 170-W CPU leaves 372 W
below the authenticated 732-W combined +12-V PSU label, but nameplates do not
prove cabling, fit, cooling, or measured load.

Permanently reject ordinary-dense source-oriented B580 at one million
positions. Do not authorize purchase or oneAPI implementation. Preserve
changed attention, modified weights, faster Xe2 products, and multi-card
systems as distinct branches. Two earlier fail-closed analyzer invocations
published no output: one exposed an over-literal Xe2 HTML guard, and one was
given a mismatched expanded commit. The corrected clean invocation alone
published the authoritative report at
`/Users/chad/Models/mimo-prismwing/evidence/PW-0166/analysis-001/manifest.json`,
SHA-256
`30908aee4e494aa12c31223ba6b2072684f3c1a954e300d9b55566b978591bce`.

Gate 8 passes with 73% minimum free memory, 36,192,256-byte peak RSS,
19,416,576-byte maximum physical footprint, zero swap growth or throttling,
an explicit source-payload release boundary, and stable protected services.
The experiment reports zero accepted tokens, no endpoint TPS, and no changes
to measured throughput-model constants; `116.5` TFLOPS is a derived candidate
ceiling, not achieved performance.
