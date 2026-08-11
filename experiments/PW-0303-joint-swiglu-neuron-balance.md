# PW-0303 — Joint SwiGLU neuron-balance control

- Status: complete
- Disposition: rejected at the frozen source-F16 symmetry gate
- Date: 2026-08-11
- Owner: Thimble with project-owner authorization
- Contract commit: uncommitted pre-execution record
- Checkpoint: XiaomiMiMo/MiMo-V2.5 revision
  `63651580ca774f8504f676040460aed3e1244ac1`
- Query authority: PW-0116 corpus
  `b9df976876d63c1ffbbe0c70507aea8b939a749ce5b1db27cbca0b5d82cf802e`
- Exactness: L3 quantized weights after an exact pre-quantization symmetry
- Related records: PW-0113, PW-0134, PW-0148, PW-0301, PW-0302

## Question

PW-0302 rejects treating expert rows as independent vector-query records.
Test a whole-expert degree of freedom that independent matrix quantization
cannot see. For every SwiGLU hidden neuron `i`, the transformation

`up_i <- up_i / s_i` and `down_:,i <- down_:,i * s_i`

is exactly function-preserving for positive finite `s_i`; gate is unchanged.
Can scales selected from the joint up-row/down-column geometry reshape the
six-bit quantization problem enough to improve complete expert outputs?

PW-0134 does not answer this question. It selected the official AWQ activation-
mean exponent family at four bits. This experiment selects a weight-geometry
balance family and scores six-bit complete-expert behavior. It does not repeat
exact neuron permutation (PW-0113), global-Hessian assignment (PW-0148), or
independent row sketches (PW-0302).

## Frozen candidate and controls

Use the three PW-0138 experts and frozen PW-0116 partitions. For each neuron,
compute source-F32 RMS values `u_i` for its complete up row and `d_i` for its
complete down column. For alpha in `{0.0, 0.1, ..., 1.0}`, define

`s_i(alpha) = clip((u_i / max(d_i, 1e-12)) ** (alpha / 2), 1/16, 16)`.

Normalize every 128-neuron source scale block by its geometric-mean `s` so the
transform cannot obtain a free blockwise magnitude shift. Search one shared
alpha per expert using train positions only and complete source-semantics
expert-output squared error after quantization. Ties choose the smaller alpha.

Controls:

1. source FP8 with dynamic FP8 inputs and BF16 boundaries;
2. unchanged affine6 group-128 RTN with F16 scale/bias;
3. the immutable PW-0148 global-Hessian six-bit result; and
4. an identity `alpha=0` round-trip through the candidate implementation.

Quantize the transformed gate/up/down tensors with the unchanged affine6 RTN
grid first. The candidate may fold `s_i` into installed up/down weights, but
charge a conservative F16 scale vector until byte-identical folding and kernel
semantics are demonstrated.

## Gates

Before any alpha search, prove the unquantized transformation reproduces every
train and validation expert output within the source-F16 tolerance used by
PW-0134. Fail on non-finite values, zero norms, scale clipping not recorded,
partition leakage, source mismatch, or identity-control drift.

Expand beyond the three experts only if all three:

1. improve validation relative L2 over affine6 RTN by at least 20%;
2. improve rather than regress against PW-0148's global-Hessian six-bit result;
3. reach at most 2% complete-expert relative L2 and 5% maximum-row relative L2;
4. improve train as well as validation;
5. add no runtime MACs and at most 0.1% bytes after proven scale folding; and
6. pass the declared memory, hash, and release gates.

Failure rejects only this RMS-balance family. A pass authorizes a separately
frozen shared-grid/metadata-removal test needed to reach 75.1% source bytes; it
does not itself authorize a bank, decoder, kernel, holdout, endpoint, or
throughput claim.

## Result

The cheap layer-4/expert-96 gate rejected the family before the affine6 alpha
search. The original NumPy approximation was retained as an honest pre-install
diagnostic: it missed the captured source by `0.0031973763` train relative L2
and `0.0092632836` validation relative L2 and therefore stopped.

With the exact PyTorch dynamic-FP8/BF16 execution path, captured-source parity
passed comfortably: `0.0002907024` train and `0.0000957342` validation relative
L2. Identity alpha `0.0` reproduced the exact source execution byte-for-byte.
The first nonidentity candidate, alpha `0.1`, nevertheless changed unquantized
source-semantics outputs by `0.0049012884` train and `0.0137005385` validation
relative L2. Both exceed the frozen `0.002` source-F16 threshold. No quantized
search and no later-expert work were permitted.

This does not contradict the real-arithmetic SwiGLU symmetry. Non-power-of-two
rescaling crosses BF16 rounding boundaries in the up projection and changes the
dynamic-FP8 grouping/scales before the down projection. The operational source
graph is therefore not invariant to the declared continuous RMS scale family.

Evidence:

- `PW-0303-layer04-expert096.json`
  SHA-256 `815cd3cacc9d745c76c6ec9a3c9d57d0885035c5d82c5a46afcd56aafec8992e`
- source expert archive SHA-256
  `d2a51359c38e754c30d84cc97acc0bca8b7bc8f06a0cb95c6ef851030e740f74`
- Torch wheel `torch-2.13.0+cpu-cp313-cp313-manylinux_2_28_x86_64.whl`,
  version `2.13.0+cpu`, 191,815,667 bytes, SHA-256
  `3fbf9c9d1f3c10c2d59d04aca426dee9ccc6ceb32d255c61e93acc3b4f75fae6`

The wheel was installed without sudo into isolated `/tmp/pw0160-venv`. It was
the only large toolchain download; required pure-Python runtime dependencies
were fetched sequentially after the deliberate `--no-deps` import exposed
their absence.

The first exact artifact was generated as PW-0161 before discovery that the
checkout lagged an upstream history already using that number. It is preserved
as `PW-0303-pre-renumber-layer04-expert096.json` with SHA-256
`51a218c6a3e12c0db813e515ff62b8e0e0d979cf22ae1455643decdb273e7a50`.
A second artifact generated as PW-0214 before the active upstream stream
claimed PW-0211 through PW-0216 is preserved as
`PW-0303-pre-pw0300-layer04-expert096.json`, SHA-256
`fa346cf31319c8fcd18f7320ea5841e480e547a3c22b74cdcdd3a1606caa73e6`.
The canonical artifact above changes only the experiment identity to PW-0303.
