# PW-0133 — Train-only INT4 source-FP8 exception store

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
  PW-0132 raw
  `0499a40645452eab646276e1619fb2e94b74439ef4263a71f036fae61fd8a9fe`
- Hardware/runtime: Apple M1 shared 16 GiB; verified internal-SSD checkpoint;
  MLX affine INT4 plus exact source-FP8 group corrections
- Exactness: explicitly modified L3 mixed-precision representation; unchanged
  source-FP8 control
- Related records: PW-0046, PW-0116, PW-0129 through PW-0132

## Compression-depth contract

Capability invariant: preserve the complete declared MiMo near-equivalence and
full-capability target; this pilot may not alter routes, expert topology, or
acceptance thresholds. Embodiment boundary: weights, layouts, quantization,
native kernels, Metal resource mappings, and runtime artifacts may change.
Project constraints remain local inference, at most $500 incremental hardware,
at most 1,000 W, the shared-host safety gate, and explicit L3 labeling.

The baseline affine INT4 expert occupies 53.112% of its source-FP8 record but
fails real routed-output fidelity. PW-0130 through PW-0132 show that fitting
small programs after the complete quantized expert either lacks capacity or
memorizes the pilot. Change the weight representation instead: retain the
INT4 core and restore only the source groups predicted to cause the largest
output error on training activations.

## Frozen mechanism

Use the PW-0129 production affine group-128 INT4 representation. For every
gate, up, and down projection of every expert selected at validation positions
`112..167`:

1. derive the F16-visible source weight and dequantized INT4 weight;
2. calculate squared weight error per row-by-128 group;
3. weight each input channel by its mean squared activation from training
   positions `0..111` for that expert;
4. rank groups independently within each projection by the resulting diagonal
   output-error proxy; and
5. restore the top 1%, 2%, 4%, and 6% of groups to exact source-FP8 values in a
   sparse correction path before the projection's declared output boundary.

Gate/up use real MoE inputs. Down uses source-derived BF16 SwiGLU inputs from
the same training positions. An expert absent from train must remain visible:
gate/up use the layer-pooled training second moment and down falls back to an
unweighted weight-error ranking. No validation input, output, or route may
affect selection. Positions `168..223` remain sealed.

The execution oracle may materialize dense correction matrices to test
fidelity, but the physical ledger must charge the deployable sparse artifact:

- 128 raw source-FP8 weight bytes per selected group;
- one U32 group ordinal per selected group;
- one F32 source scale for each distinct source 128-by-128 block referenced;
- the unchanged complete MLX affine-INT4 payload; and
- one additional correction MAC per selected weight, plus source-FP8 and INT4
  decode work, before a kernel can be promoted.

Dense oracle materialization is diagnostic memory and cannot be reported as
the executable representation or endpoint timing.

## Gates and dispositions

The run must reproduce PW-0129's affine-INT4 validation baseline exactly,
authenticate all source tensors and routes, keep holdout sealed, record every
selection digest and fallback, and enforce Gate 8 at each expert and layer
release boundary.

Select the smallest exception fraction satisfying all of:

1. aggregate validation routed-output relative L2 at most 1%;
2. every layer at most 2%;
3. no validation row above 5%;
4. complete sparse artifact bytes at most 60% of the source layer bank; and
5. correction MACs at most 10% of source expert MACs.

A strict pass authorizes a separately frozen holdout evaluation and then a
sparse Metal kernel probe; it does not authorize a full bank or endpoint. If
no strict pass exists but a candidate reaches 2% aggregate, 4% per layer, and
8% per row within the same physical limits, authorize only a separately named
AWQ/exception composition test. Otherwise reject this diagonal-sensitivity
source-FP8 exception store and proceed to AWQ/GPTQ-style weight calibration,
rotations, or recovery training.

Report zero accepted tokens, `A=0`, no endpoint timing, and no TPS claim.

