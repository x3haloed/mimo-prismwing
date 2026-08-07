# PW-0134 — Train-only AWQ-style INT4 expert rescaling

- Status: planned
- Disposition: unexecuted
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

