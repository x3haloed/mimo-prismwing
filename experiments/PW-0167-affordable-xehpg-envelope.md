# PW-0167 — affordable Xe-HPG complete-system envelope

- Status: completed
- Disposition: conditional
- Date: 2026-08-10
- Owner: Codex with project owner authorization
- Checkpoint/reference hashes: MiMo revision
  `63651580ca774f8504f676040460aed3e1244ac1`; config
  `292a60e74ae9a6d53422b31b21468ce2111c0ab3f7f7a4f4e9c7cd5133b96587`;
  PW-0127, PW-0151, PW-0155, and PW-0158 authorities authenticated at execution
- Hardware candidate: one used Intel Arc A770 16-GB card in the owned EPYC host;
  analytical pre-purchase envelope only
- Related records: PW-0127, PW-0151, PW-0155, PW-0158, PW-0162, PW-0166; E7
- Implementation commit and dirty state:
  `a392e1f6e8aa1e172d0bbad8f7582d83e1bcc32f`; clean

## Question and changed premise

PW-0166 rejects B580 ordinary-dense one-million execution by only 26.6923
seconds. Intel's older A770 exposes more XMX engines, 16 GB, and an official
262-INT8-TOPS dense rate. Intel also publishes Xe-HPG's datatype ratio directly:
4,096 INT8 versus 2,048 FP16/BF16 operations per Xe-core-cycle. Test the derived
131-TFLOPS BF16 ceiling before assuming the newer consumer card is stronger for
this workload.

The owned platform creates two separate risks. Supermicro officially says the
H11SSL-i does not support Resizable BAR, while Intel requires it for optimal Arc
performance and warns that unlisted configurations may have performance or
stability problems. Intel's current oneAPI client-GPU matrix lists Ubuntu, not
the owned Debian 13 installation. These facts cannot be hidden behind an
arithmetic survivor.

## Shared construction and compression-depth contract

Capability invariant: preserve every source weight, one million positions, all
nine global and 39 sliding attention layers, native modalities, and every TARGET
gate. Ordinary dense attention and weights remain the L1 control.

Authorized embodiment boundary: grant perfect sustained BF16 XMX utilization,
the EPYC's impossible peak concurrently, and zero time for every omitted
operation. Permit layer-major or host/storage streaming only as a named capacity
requirement, not as free achieved performance. No firmware modification,
purchase, OS replacement, or undocumented ReBAR workaround is authorized.

## Contract

1. Authenticate TARGET, config, PW-0127, PW-0151, PW-0155, PW-0158, all five
   official Intel/Supermicro captures, the owned-host inventory, and the weaker
   dated sold-market transcription by SHA-256.
2. Bind A770 16 GB to Xe-HPG, 512 XMX engines, 262 peak dense INT8 XMX TOPS,
   225 W, 560 GB/s, PCIe 4.0 x16, and oneAPI support.
3. Require Intel's Xe-HPG architecture to specify 4,096 INT8 and 2,048
   FP16/BF16 XMX operations per Xe-core-cycle. Derive, do not measure, a
   131-TFLOPS source-oriented BF16 ceiling.
4. Add two operations per PW-0127 mandatory MAC at one million positions to
   PW-0158's exact ordinary-attention work. Grant the derived ceiling and
   EPYC's impossible peak concurrently.
5. Retain the arithmetic branch only if that floor is at most 1,800 seconds.
   Do not call omitted softmax, conversion, dispatch, transfer, storage, or
   utilization costs headroom.
6. Recompute exact 1M BF16 KV, three arenas, and non-routed source tensors
   against 16 decimal GB. A layer-major or spill requirement is not an
   independent performance proof.
7. Bind one 225-W card plus the 170-W CPU to the photographed PSU's 732-W
   combined +12-V ceiling without calling nameplates cable, fit, thermal, or
   measured-load proof.
8. Bind H11SSL's absent native ReBAR, Intel's ReBAR guidance, the owned Debian
   13 identity, and Intel's listed client-GPU Linux distributions. Any installed
   branch requires a reversible ReBAR-off/on component benchmark and a supported
   software plan before runtime work.
9. Treat the two sold used-card observations only as proof that sub-cap prices
   have occurred. They are not active inventory or a delivered complete BOM.
10. Apply Gate 8. Record zero accepted tokens and no endpoint TPS.

## Promotion and kill rule

Reject ordinary-dense A770 permanently if even the impossible arithmetic floor
exceeds 1,800 seconds. If it survives, promote only an active-BOM and installed
component prerequisite: exact card/connector/fit evidence, delivered complete
cost at or below `$500`, then measured oneAPI BF16 compute, PCIe acquisition,
and ReBAR-off/on behavior on the owned host. Do not purchase or implement from
this analytical result.

## Result

The authenticated analyzer derives a `131`-TFLOPS BF16/F32-accumulate ceiling
from Intel's official 262-INT8-TOPS A770 row and Xe-HPG's published 4,096:2,048
INT8-to-BF16 operation ratio. Mandatory one-million-position matrices plus
ordinary attention total `214,165,790,024,007,680` operations. Even after
granting the owned EPYC's impossible `0.7424`-TFLOPS peak concurrently, the
resulting floor is `1,625.6405684427161` seconds. It retains `174.35943155728387`
seconds inside the complete TTFT gate, before every omitted operation,
transfer, storage cost, dispatch, and utilization loss. This is an arithmetic
survivor, not achieved performance.

Exact BF16 1M KV exceeds 16 decimal GB by `7,065,559,040` bytes. KV, three
maximum layer arenas, and non-routed source tensors exceed it by
`22,221,107,536` bytes, so a complete implementation requires layer-major or
host/storage streaming. The 225-W board plus 170-W CPU leaves 337 W below the
photographed NEX750B's authenticated 732-W combined +12-V label, but
nameplates do not prove cabling, fit, cooling, rail assignment, or wall load.

The owned H11SSL-i has no supported native Resizable BAR path, Intel requires
Resizable BAR for optimal Arc performance and warns of performance or
stability issues otherwise, and Intel's current client-GPU oneAPI matrix lists
Ubuntu rather than the owned Debian 13 environment. Two dated used A770 sales
at `$245` and `$215.50` prove that sub-cap card prices have occurred; they are
sold listings, not active inventory or a delivered complete BOM.

Retain A770 only as a conditional arithmetic survivor. Before purchase, require
an active complete delivered BOM at or below `$500`, exact connector and fit
evidence, and a reversible installed oneAPI BF16/PCIe/ReBAR-off/on component
benchmark on the owned host. No purchase or runtime implementation is
authorized by this result.

The authoritative report is
`/Users/chad/Models/mimo-prismwing/evidence/PW-0167/analysis-001/manifest.json`,
SHA-256
`0ff6f2cb1017cb6589b8c5705e7adda349fc2637721e3ddc8c695f051dff2c01`.
Gate 8 passes with 70% minimum free memory, 34,635,776-byte peak RSS,
21,087,488-byte maximum physical footprint, zero swap growth or throttling,
an explicit source-payload release boundary, and stable protected services.
The experiment reports zero accepted tokens, no endpoint TPS, and no changes
to measured throughput-model constants; 131 TFLOPS is a derived candidate
ceiling rather than achieved performance. One earlier invocation supplied a
mismatched expanded commit and failed closed before publishing output.
