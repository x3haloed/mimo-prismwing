# PW-0134 — Train-only AWQ-style INT4 expert rescaling

- Status: completed
- Disposition: rejected; second-order updates, rotations, or recovery training next
- Date: 2026-08-06
- Owner: Codex with project owner authorization
- Checkpoint/reference hashes: MiMo revision
  `63651580ca774f8504f676040460aed3e1244ac1`; checkpoint verification
  `9ddc8a99755f04ae2ea3c2484f6dd022d3f3a681b5a72c915ee4de833dbb0d03`;
  PW-0116 corpus
  `b9df976876d63c1ffbbe0c70507aea8b939a749ce5b1db27cbca0b5d82cf802e`;
  PW-0129 raw
  `1deb9dd85f0b598f31bc2d8bc1d41bf52cfabcda43de63a2ae5b3fdfad400306`;
  PW-0133 raw
  `a0226e42058a04ea1009a6c00a6b44fdc85728bf36e383166a589b1d3e28b0d8`
- Hardware/runtime: Apple M1 shared 16 GiB; verified internal-SSD checkpoint;
  MLX affine INT4 with train-calibrated exact channel rescaling
- Exactness: explicitly modified L3 weight-only quantization; unchanged
  source-FP8 control
- Related records: PW-0129 through PW-0133

## Question and changed mechanism

PW-0133 shows that retaining the largest diagonal-sensitivity source groups
cannot bridge affine INT4's fidelity gap inside the 60% byte limit. AWQ changes
a different variable: exactly rescale channels before quantization so the same
four-bit grid allocates resolution according to activation salience.

For each expert, use only train positions `0..111` and the official AWQ
activation scale family:

`s(alpha) = mean(abs(x), rows) ** alpha`, normalized by the geometric mean of
its minimum and maximum, with `alpha` swept over `0.00, 0.05, ..., 0.95` and
all scales clamped to at least `1e-4` before normalization.

Search two exact transforms sequentially:

1. **Expert-input transform.** Multiply gate and up weight columns by
   `s_in`; divide the expert input by `s_in`. Select the exponent minimizing
   complete train expert-output squared error with source down projection.
2. **SwiGLU/down transform.** Divide up weight rows by `s_hidden` and multiply
   down weight columns by `s_hidden`. Select the exponent using source BF16
   SwiGLU train inputs and down-projection squared error.

Then compose both transforms, quantize all three transformed projections to
the unchanged MLX affine group-128 INT4 format, and evaluate only validation
positions `112..167`. The exact pre-quantization expert function must be gated
to source-F16 tolerance for every selected exponent; this is especially
important because SiLU is not homogeneous while the multiplicative up branch
is.

An expert absent from train uses layer-pooled input and hidden activation means
and the layer median exponents selected by training-seen validation experts.
Record every exponent, scale-vector hash, train objective, fallback, packed
artifact hash, and validation expert output. No validation input, output, or
route may affect calibration. Positions `168..223` remain sealed.

## Physical ledger and gates

The transformed weights remain the same 13,369,344-byte INT4 payload. Charge
F16 `s_in` and `s_hidden` vectors conservatively even where a runtime artifact
could fold a transform into adjacent packed weights: 12,288 additional bytes
per expert. Record the 4,096 input divides per selected expert and all setup
work separately; calibration time is installation work, not inference.

Pass only if frozen validation reaches all of:

1. aggregate routed-output relative L2 at most 1%;
2. every layer at most 2%;
3. no row above 5%;
4. every transformed scale and packed artifact is finite and deterministic;
5. the complete artifact remains at most 60% of source expert bytes; and
6. added runtime elementwise work remains below 1% of source expert MACs.

A strict pass authorizes a separately frozen holdout evaluation and direct
Metal packed-kernel probe, not a full bank or endpoint. A 2% aggregate, 4%
per-layer, 8% per-row near miss may authorize AWQ plus a separately bounded
exception store. Otherwise reject this AWQ scale family and proceed to
GPTQ-style second-order error propagation, a function-preserving rotation, or
recovery training.

Report zero accepted tokens, `A=0`, no endpoint timing, and no TPS claim.
Apply normative Gate 8 at every expert and layer release boundary.

## Result

The clean run completed in 65,746.525 ms. All PW-0129 validation baselines
reproduced exactly, all scale searches used only positions `0..111`, and
holdout positions `168..223` remained sealed. Every layer improves, but the
result is still a decisive miss:

| Layer | Baseline L2 | AWQ-style L2 | Worst candidate row | Median input alpha | Median hidden alpha |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 4.1919% | 2.5631% | 4.6175% | 0.30 | 0.20 |
| 24 | 11.9174% | 8.3810% | 17.5005% | 0.20 | 0.25 |
| 46 | 15.4606% | 12.6141% | 13.6778% | 0.20 | 0.30 |

Aggregate validation relative L2 falls from 9.7661% to 7.7451%, a 20.69%
relative reduction. The worst layer remains 12.6141% and the worst row
17.5005%, so neither the strict nor near-miss gate is close. Two layer-24
experts use the declared pooled-activation/median-exponent fallback for 15
placements; layers 4 and 46 are fully calibrated and independently reject the
family. All selected exponents are nonzero, and the worst exact-transform
weight reconstruction error is below `2.77e-8`, so the result is not an
identity-control or algebra failure.

The embodiment remains attractive in isolation: 13,381,632 bytes per expert
is 53.1608% of source, including both conservative F16 scale vectors, and
4,096 runtime input divides are only 0.0163% of source expert MACs. Physical
fitness does not compensate for failed routed-output fidelity.

Gate 8 passes across 86 snapshots at 78% minimum free memory,
923,418,624-byte maximum peak RSS, 230,214,656-byte maximum physical footprint,
zero swap growth or new throttled pages, and all protected service names
remaining resident. One of two baseline `nxnode` PIDs exited while the other
remained; this is recorded and does not violate the normative name-level
service-health stop. Raw evidence hashes to
`7d470bd5fa5541424c2b619afb49a2ebf493ce7a11b2498cf281b3d1c6f34490`;
independent analysis hashes to
`8f0da2e109befe20928a1134a178d23343d27afe6c3d60a3e8682d1b5925745c`.

## Decision

Reject the official AWQ activation-mean exponent scale family as adapted to
independently routed MiMo experts. Do not read holdout, compose it with
PW-0133 exceptions, build a packed kernel, or construct a bank. This does not
reject second-order weight-error propagation, rotations, or recovery training;
those change mechanisms that this scalar-exponent rescaling preserves. No
endpoint performance or TPS claim changes.
