# PW-0163 — MI100 32-GB complete-system envelope

- Status: completed
- Disposition: rejected
- Date: 2026-08-10
- Owner: Codex with project owner authorization
- Checkpoint/reference hashes: MiMo revision
  `63651580ca774f8504f676040460aed3e1244ac1`; config
  `292a60e74ae9a6d53422b31b21468ce2111c0ab3f7f7a4f4e9c7cd5133b96587`;
  PW-0127 arithmetic, PW-0151 owned host, PW-0158 attention, and PW-0161
  complete-envelope authorities to be authenticated at execution;
  official AMD product page `9d0b74dc18ac8afcced3a9efbca17f77f6fd4b148c6823c11a5f479d8f9cbcc6`;
  official AMD brochure `ad383b0c0d2bcb8c719ddcf09ed5a4d7a0afeb901c3b51bb2490fa3e65e6dc2e`;
  official ROCm requirements
  `dead3ad053cde897c83aa33d58f096b1a9b25878abbe82dbcd5206c3f86d3772`;
  dated market transcription
  `bfdc3bcd99685518f810f4d7f5caaa7d6563511cc1f62dfed3f317f2d0bd9022`
- Hardware candidate: one AMD Instinct MI100 32-GB PCIe card in the owned
  H11SSL-i/EPYC host; analytical pre-purchase envelope only
- Related records: PW-0127, PW-0151, PW-0158, PW-0159, PW-0161; E7
- Implementation commit and dirty state:
  `9f43873c04f80b2687261d2a887bbd3bdf2af18a`, clean

## Question and changed premise

PW-0159 and PW-0161 close inexpensive NVIDIA Ampere and 32-GB Volta forms,
but do not close AMD CDNA. MI100 is the strongest conventional 32-GB PCIe
counterexample found in the current search: AMD advertises 92.3-TFLOPS BF16,
184.6-TFLOPS FP16, 32 GB HBM2 at 1.2 TB/s, PCIe 3/4 x16, and 300-W power.

Determine whether its source-oriented BF16 arithmetic can satisfy the complete
one-million-position gate, whether the faster FP16 mode leaves a separately
named L3 branch, whether exact source residency fits, and whether a current
complete installation can remain inside `$500`. Do this before any ROCm
runtime work or purchase request.

## Shared construction and compression-depth contract

Capability invariant: preserve every source weight, all one million positions,
the pinned nine global and 39 sliding attention layers, native modality paths,
and every TARGET gate. Ordinary dense attention remains the control.

Authorized embodiment boundary: treat MI100 BF16 Matrix peak as the most
favorable source-oriented numerical ceiling, without claiming that source FP8
decode or the exact reduction topology can attain it. Treat FP16 peak as L3
because FP16 accumulation cannot silently replace source BF16 behavior. Grant
perfect utilization, concurrent impossible EPYC peak, and zero time for all
non-arithmetic work.

Project constraints: the M1 remains the user-facing host; the owned EPYC host
costs zero; every newly acquired accelerator, storage device, adapter, cable,
cooler, and OS-dependent installation part must fit `$500` delivered including
tax and shipping. Complete measured wall power remains at most 1,000 W, with
the installed NEX750B's tighter 732-W combined +12-V label still binding.

## Contract

1. Authenticate TARGET, config, PW-0127's mandatory matrix MAC ledger,
   PW-0151's EPYC/PSU authority, PW-0158's exact ordinary-attention work,
   PW-0161's complete one-million-position FLOPs, the official AMD product
   page and brochure, current ROCm system requirements, and the dated market
   transcription by SHA-256. Fail closed on revision, arithmetic, memory,
   power, form-factor, software-support, price, or evidence-class drift.
2. Add two operations per PW-0127 MAC at all one million positions to
   PW-0158's exact ordinary-attention work. Divide by each of AMD's direct
   FP32, FP32 Matrix, BF16 Matrix, and FP16 peaks after granting PW-0151's
   impossible EPYC peak concurrently. Report every floor and omitted cost.
3. Permanently reject the source-oriented ordinary-dense MI100 candidate if
   even its favorable 92.3-TFLOPS BF16 Matrix floor exceeds TARGET's complete
   1,800-second 1M TTFT gate. A faster FP16 floor retains at most an L3
   numerical hypothesis pending full accumulated fidelity.
4. Recompute exact BF16 1M KV, three maximum routed-layer arenas, and all
   non-routed source tensors against 32 decimal GB. Also report the impossible
   free-common-streaming expert-slot alternative, without claiming executable
   sharding or allocation.
5. Bind the official 300-W passive, full-height, dual-slot, 10.5-inch PCIe
   form and PCIe 3.0 compatibility to the owned host only as logical
   prerequisites. Report GPU-plus-CPU headroom under 732 W. Do not authorize
   installation without exact auxiliary-cable requirements, chassis clearance,
   forced airflow, staged temperature/ECC/power checks, and measured load.
6. Record that current ROCm 7.1 supports MI100 only on the enumerated Ubuntu,
   RHEL, and SLES releases, not the owned host's current Debian 13. Treat a
   supported OS installation as a deployment prerequisite, not a permanent
   arithmetic rejection or license to overwrite the existing server.
7. Freeze the active used 32-GB listing. Reject its procurement branch if the
   card alone exceeds `$500` before tax; a card-only failure cannot be repaired
   by adding mandatory cooling, cabling, or storage. The transcription remains
   moving-market evidence, not device identity, health, or delivery proof.
8. Apply Gate 8 to the analyzer. Record zero accepted tokens and no endpoint
   TPS. Never present BF16/FP16, HBM, PCIe, price, or power nameplates as
   achieved runtime performance.

## Promotion and kill rule

Reject source-oriented ordinary-dense MI100 execution permanently if its
favorable BF16 arithmetic floor exceeds 1,800 seconds. Reject the captured
procurement branch if its card alone exceeds `$500` before tax.

Retain FP16 only as a price-triggered L3 hypothesis if its ideal arithmetic
passes. Reopening requires an active functional card cheap enough for the
complete delivered cable/cooling/storage/OS ledger, physical installation
evidence, a frozen HIP implementation, and every local, hosted, capability,
modality, long-context, and fidelity gate. Neither arithmetic nor a future
cheap listing promotes an endpoint.

## Result

The authoritative `analysis-001` manifest hashes to
`dcc6a60955e8dfd67a3f1da582b33b332bfa31ece45a70d4606df8e367bcb145`.
It authenticates TARGET, the pinned config and arithmetic authorities, the
owned host, AMD's product page and brochure, ROCm 7.1 requirements, the dated
market transcription, and the clean implementation commit.

MI100 does not rescue source-oriented ordinary dense one-million-position
execution. Granting its complete 92.3-TFLOPS BF16 Matrix peak and the EPYC's
impossible 0.7424-TFLOPS peak concurrently still needs `2,301.8085` seconds,
missing the entire 1M TTFT gate by `501.8085` seconds before softmax, routing,
storage, cache traffic, dispatch, protocol, or any other work. Direct FP32 and
FP32 Matrix are slower. Reject this source-oriented embodiment permanently.

The 184.6-TFLOPS FP16 ceiling reaches an idealized `1,155.5143` seconds and
leaves `644.4857` seconds, but it is only an L3 numerical hypothesis. It does
not prove source-BF16 fidelity, a HIP kernel, or full-path performance.

The 32-decimal-GB capacity ledger matches the prior 32-GB forms: exact 1M BF16
KV, three arenas, and all non-routed tensors total `38,221,107,536` bytes,
exceeding HBM by `6,221,107,536`. Even streaming every non-routed tensor for
free leaves only 261 complete expert slots. The card's 300-W TDP plus the
170-W CPU leaves 262 W below the PSU's 732-W combined +12-V label, but the
passive full-height dual-slot card still lacks authenticated auxiliary cabling,
clearance, airflow, and measured-load evidence.

The captured used, tested-working 32-GB listing is `$999.00` with free
shipping before unknown tax: the card alone exceeds the complete cap by
`$499.00` before cooling, cabling, storage, or OS work. ROCm 7.1 does not list
the owned Debian 13 installation for MI100; a supported Ubuntu/RHEL/SLES
installation is an additional deployment prerequisite, not a permanent
hardware rejection.

Gate 8 passes across five snapshots with 51% minimum free memory,
32,210,944-byte peak RSS, 21,103,936-byte maximum physical footprint, zero
swap growth or new throttled pages, an explicit release boundary, and stable
protected services. The first clean invocation published no manifest because
the operator supplied a mistyped commit hash; the exact-commit rerun passed.
Accepted tokens and endpoint TPS remain zero.

## Decision

Reject source-oriented ordinary-dense MI100 execution on the complete 1M
arithmetic lower bound. Reject the captured procurement branch because the
card alone costs almost twice the complete cap. Retain FP16 only as a future
price-triggered L3 hypothesis requiring a complete delivered BOM, supported OS,
physical installation evidence, measured HIP implementation, and all fidelity,
capability, modality, and long-context gates. Authorize no purchase or runtime
implementation.
