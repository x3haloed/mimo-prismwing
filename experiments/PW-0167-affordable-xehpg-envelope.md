# PW-0167 — affordable Xe-HPG complete-system envelope

- Status: ready
- Disposition: unexecuted
- Date: 2026-08-10
- Owner: Codex with project owner authorization
- Checkpoint/reference hashes: MiMo revision
  `63651580ca774f8504f676040460aed3e1244ac1`; config
  `292a60e74ae9a6d53422b31b21468ce2111c0ab3f7f7a4f4e9c7cd5133b96587`;
  PW-0127, PW-0151, PW-0155, and PW-0158 authorities authenticated at execution
- Hardware candidate: one used Intel Arc A770 16-GB card in the owned EPYC host;
  analytical pre-purchase envelope only
- Related records: PW-0127, PW-0151, PW-0155, PW-0158, PW-0162, PW-0166; E7
- Implementation commit and dirty state: pending

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

Pending authenticated execution.
