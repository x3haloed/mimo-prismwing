# PW-0133 — Train-only INT4 source-FP8 exception store

- Status: completed
- Disposition: rejected; AWQ/GPTQ/rotation or recovery-training branch next
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

## Result

The clean run completed in 31,115.291 ms. Every PW-0129 affine-INT4 validation
baseline reproduced exactly, every selection used only positions `0..111`, and
positions `168..223` remained sealed. The validation curve improves
monotonically but far too slowly:

| Exact group fraction | Aggregate L2 | Worst layer | Worst row | Source-byte ratio | Correction MAC ratio |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0% baseline | 9.7661% | 15.4606% | 17.9215% | 53.1120% | 0% |
| 1% | 9.3098% | 14.9098% | 17.7513% | 54.1753% | 1.0010% |
| 2% | 8.9733% | 14.4888% | 17.5880% | 55.2370% | 2.0004% |
| 4% | 8.5955% | 14.0163% | 17.3097% | 57.3619% | 4.0009% |
| 6% | 8.3871% | 13.7370% | 17.1606% | 59.4868% | 6.0013% |

At 6%, the exception store restores 11,799 row-groups or 1,510,272 raw FP8
weight bytes per expert. The conservative complete artifact is 14,974,008
bytes per expert, already 59.4868% of source. It reduces aggregate error by
only 14.12% relative. Layer-4 validation reaches 2.5346%, layer 24 reaches
9.6881%, and layer 46 reaches 13.7370%. Two layer-24 validation experts lack
training placements; their declared fallback accounts for 15 placements, but
the fully covered layers independently reject the mechanism. A 7% store would
occupy 60.5485% of source and is outside the frozen byte gate.

Gate 8 passes across 47 snapshots at 78% minimum free memory,
847,396,864-byte maximum peak RSS, 253,119,552-byte maximum physical footprint,
zero swap growth or new throttled pages, and stable protected services. Raw
evidence hashes to
`a0226e42058a04ea1009a6c00a6b44fdc85728bf36e383166a589b1d3e28b0d8`;
independent analysis hashes to
`02715ba47566a1269a34ce470e4e04bf6acfd0ebb55c2174b9d329d00300b350`.

## Decision

Reject diagonal train-activation-weighted selection of exact source-FP8
row-groups over the fixed affine INT4 core. Do not build its sparse Metal
kernel, full bank, or holdout evaluation. The result does not reject
weight-domain calibration broadly: AWQ changes the quantization grid by exact
channel rescaling, GPTQ propagates second-order error into unquantized weights,
and rotations change outlier geometry. Those are distinct mechanisms. No
endpoint performance or TPS claim changes.
