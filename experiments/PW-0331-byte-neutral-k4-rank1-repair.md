# PW-0331 — Byte-neutral K4 down-rank-one repair

- Status: proposed; unexecuted
- Disposition: unexecuted
- Date: 2026-08-27
- Owner: Codex
- Parent experiments: PW-0116, PW-0315, PW-0316, PW-0318, PW-0329
- Exactness: L3 modified expert weights; source target and thresholds unchanged

## Question

Can filling only the already allocated and executed down-projection rank-one
slot for layer-4 expert 96 make PW-0316's unseen four-K4/four-source row pass
the unchanged strict routed and layer-final gates, without changing K4 logical
bytes, schema-2 stride, loader, kernel, or runtime operations?

This is the cheapest fidelity falsifier for the density-four/five points exposed
by PW-0329. It is not a bank build, endpoint, or general K4 qualification.
Companion hardware is inadmissible.

## Newly discovered constraint

Every authenticated schema-2 K4 projection declares rank one and already stores
both correction factors. Every factor is currently all-zero F16. The loader
requires rank one and the Metal path already reads and evaluates the correction
before finishing the projection. Replacing the down-factor contents therefore
adds no bytes and no operations relative to the measured K4 record.

This matters at one TPS: adding precision bytes consumes the already narrow
ordinary/rare storage margin, whereas a correction inside the existing
`12,877,824`-byte stride can improve fidelity without worsening the byte bound.
The observable invariant is byte-neutral fidelity improvement; rank-one repair
is the first mechanism tested against it.

## Frozen authorities

The implementation must fail closed on:

1. Clean parent planning commit `57572aaa017cedec191167b6147ee422ff7e21ba`,
   `TARGET.md` SHA-256
   `dda459684c194b03491f36e9b66521ff00c400a6cc38d23a567a5a92ef8fb17d`,
   and `RED_LINES.md` SHA-256
   `cc261ad9bd67a865715e72cbbadf3b74c3f1f282e17a8ef86ed02c1a92fb8b36`.
   The clean execution commit must descend from the clean commit that adds this
   contract; the runner must freeze this document's Git blob ID and SHA-256.
2. Checkpoint revision `63651580ca774f8504f676040460aed3e1244ac1`,
   receipt SHA-256
   `9ddc8a99755f04ae2ea3c2484f6dd022d3f3a681b5a72c915ee4de833dbb0d03`,
   index SHA-256
   `f2e1774c9acf9a62338b68c144e6fc7a66495e59f2e64b3078c1b7ef5a196816`,
   and expert-96 shard SHA-256
   `f8c8ab1b22da717ed0360c8248da84d0f9a58af7a89deeb6d4021a67ae98a046`.
3. PW-0116 corpus manifest
   `/Volumes/Elements/mimo-prismwing/cold-assets/internal-ssd-migration-2026-08-26/Users/chad/Models/mimo-prismwing/evidence/PW-0116/corpus-001/manifest.json`,
   SHA-256
   `b9df976876d63c1ffbbe0c70507aea8b939a749ce5b1db27cbca0b5d82cf802e`,
   including its layer-4 input, exact source-expert output, routed,
   post-attention, and layer-final payload hashes.
4. PW-0315 summary
   `/Volumes/Elements/mimo-prismwing/evidence/PW-0315/summary.json`, SHA-256
   `07b3d3793a6750a030eb5b7e12a0add1b603d48758a85e6f45b44504e404d0e8`;
   expert-96 run-001 construction SHA-256
   `eb033a9a60f304a4c54069484247b31801a203c70142417afe3055161b940609`;
   candidate output SHA-256
   `b83fffef12db82614bceabeaac9be153b41060728d14692de595f0a0c67c5e56`;
   and down manifest SHA-256
   `4ba7c570ecc82c80a09a20aa4ee6aa015d12566dedc69d00ab39b236f86cf53f`.
5. PW-0316 canonical rejection
   `/Volumes/Elements/mimo-prismwing/evidence/PW-0316/build-003/rejection.json`,
   SHA-256
   `7e5560cf2cdc2abdec8ec1a17af0462f69fa7204f8ba528808ce1f046d0e6ff4`.
   Authenticate, without opening held-out payloads, the published
   zero-correction position-1 routed relative L2 `0.010988841869031155` and
   layer-final relative L2 `0.0027743952049186665` before fitting. Recompute
   them only after the factor hashes are frozen.
6. PW-0318 summary, schema-2 manifest, bundle, and fixture SHA-256 values
   `a91af31bdea45749c9ae9d5d679260bcbcd8284c238479938206a7e7e0b5eb2f`,
   `ca2cd8005c3c8f712fabd0b2fc88183d740bd6613efa065cdd4b25738c4924c3`,
   `e87a0af2aba57f46b6a2f394d70e530533d04c18aa61650afbc8528a4b8bdc35`,
   and `0189a8c15299410537cd43f934c4aefbda1c160e7c9f6920790cabfd812a6706`.
7. Zero-F16 payload SHA-256 values
   `ad7facb2586fc6e966c004d7d1d16b024f5805ff7cb47c7a85dabd8b48892ca7`
   for 4,096 bytes and
   `9f1dcbc35c350d6027f98be0f5c8b43b42ca52b7604459c0c42be3aa88913d47`
   for 8,192 bytes. Gate and up remain exactly these zero controls.
8. Unchanged implementation hashes: loader
   `fd863e53514afa0eecaf06ce0a43d7fef93ad88d4d22e465f447b707da81c9cb`,
   Metal host `01f4837d71da370b25e64ccaa9ebce8b4262fcb2999728707551b739fd771c8b`,
   K4 kernel `50c835699e7f80403d8127bdbe19e572acbf89774144f3bc079cd3a9c68b58c8`,
   and mixed reducer
   `d20446229683edb5855e6e2b9cf1aadc0183f5d10b976fe52f165cb03384ac84`.
   Any drift requires a separately named semantic/kernel experiment.

## Leakage-free fit

Expert 96 has 174 authenticated placements: 108 fit, one unseen primary at
position 1, 56 validation, and nine pilot holdouts. The factor-construction
process may read only placements `p < 112` excluding position 1. Validation is
`[112,168)` and pilot is `[168,224)`. Freeze factor bytes and hashes before a
separate analysis process may open or recompute any position-1, validation, or
pilot target metric.

For the fit set, decode the immutable K4 gate/up/down. Let `X` be the 108-by-
2,048 candidate post-SwiGLU dynamic-FP8 hidden input to down, `B_raw` the
unrounded serialized-K4 down GEMV output on that input, `Y_bf16` the exact
source BF16 down output, and `w` the frozen route weights. In F64 fitting
algebra:

```text
Xw = diag(w) X
Ew = diag(w) (Y_bf16 - BF16(B_raw))
Ew = U Sigma V^T
z  = U[:,0] Sigma[0]
r  = Xw^T pinv(Xw Xw^T, rcond=1e-12) z
l  = V[0,:]
```

Canonicalize sign by making the lowest-index maximum-absolute element of `l`
nonnegative. Serialize once as little-endian F16: down `correction_left=l`
with shape `[4096,1]` and 8,192 bytes; down `correction_right=r` with shape
`[1,2048]` and 4,096 bytes. Reject nonfinite data, wrong shapes, an all-zero
product, sign ambiguity, or fresh-process hash drift. Do not fit row scale or
gate/up factors.

Authoritative execution is the current order:
`BF16(B_raw + (X r) l^T)` before route accumulation. The scalar serialized-F16
slow reference must use F32 inputs and accumulators in the kernel's ascending
column/index order with IEEE-754 fused multiply-add contraction matching the
Metal compilation, then the existing BF16 consumer. This reference, not
continuous SVD loss or an already-rounded candidate output, is the correctness
authority.

## Byte and runtime invariants

The record remains exactly:

```text
packed states        12,582,912
sign payloads            18,432
global scales                12
row scales               16,384
rank-one factors         36,864
logical total        12,654,604 bytes
schema-2 stride      12,877,824 bytes
```

Only 12,288 already allocated down-factor bytes change contents. First build an
independent zero-factor four-K4/four-source layout control; no such PW-0316
bundle was emitted, and PW-0318's three-K4/five-source offsets are not a valid
comparison. The control and corrected bundles must have identical offsets and
exact size `16,384 + 4*12,877,824 + 4*25,214,976 = 152,387,584` bytes. Loader
and Metal perform the same operations as before; timing is diagnostic and no
accepted-TPS claim follows.

## Correctness ladder and frozen gates

### Stage A — slow-reference density four

1. Authenticate exact source replay and all published zero-correction
   PW-0315/PW-0316 hashes and scalars without exposing held-out payloads to the
   construction process.
2. Build the factors twice in fresh processes from one clean pushed commit.
   Require byte-identical factors and a timing-free deterministic tree.
3. In a separate held-out analysis process, require two distinct sliced gate
   sets: corrected expert-96 identity-local substitution route/final metrics
   from PW-0315, and corrected cumulative `[96,64,232,31]` four-expert
   route/final metrics. For both sets, overall, fit, validation, and pilot
   relative L2 must each be strictly below `0.01`, and every partition maximum
   row must be strictly below `0.05`. Separately require cumulative unseen
   position-1 routed and final relative L2 strictly below `0.01`. The thresholds
   remain exclusive.
4. The diagnostic along the current error direction requires at least
   `15.23576677%` attenuation of expert 96's present contribution, but the
   serialized slow-reference gates are authoritative.

Any Stage-A failure rejects this down-only rank-one embodiment. Do not tune
gate/up, increase rank, inspect held-out targets, or proceed to density five.

### Stage B — unchanged schema-2 Metal

Only after Stage A passes, build the corrected schema-2 bundle, require Rust
readback parity, then run one initial call, 20 warmups, and 100 timed Apple-M1
calls. All 4,096 routed F32 elements must match the independent slow fixture
bit-for-bit. Require unchanged bytes/offsets, batch one,
concurrency one, and every Gate 8 boundary.

### Stage C — disposition toward density five

Stage A+B can qualify only the frozen density-four row. If they pass and
PW-0329 later says density five is necessary for one TPS, open a separate
predeclared five-of-eight experiment. Freeze the first PW-0329-selected
remaining `(layer=4, expert)` identity that actually appears in the frozen
position-1 route, construct and qualify it independently, and then run a
separately frozen five-K4/three-source row. Do not select a convenient fifth
expert after observing this result.

## Required fixtures

- exact 108/1/56/9 split, position-1 exclusion, and held-out-target mutation that
  leaves factor hashes unchanged;
- tiny known rank-one recovery, orientation, sign canonicalization, F16
  serialization, and scalar application;
- reject wrong authority, rank, shape, dtype, nonfinite, zero-product, tamper,
  or implementation hash;
- zero factors reproduce PW-0315/PW-0316 bits;
- factor-content changes preserve logical bytes, offsets, and stride;
- exact exclusive `0.01` and `0.05` gates and stage precedence;
- deterministic two-run evidence tree and exact Rust/Metal readback; and
- failure at any Stage-A gate prevents Metal and density-five work.

## Evidence and reporting

Keep large factor, fixture, candidate, bundle, and Metal artifacts outside Git.
Commit schemas, hashes, small fixtures, and the final disposition. The evidence
tree must bind fit positions/weights/inputs/targets/base, numerical library
versions, factor files, corrected manifest semantic `m1-native-k4-r1-down-v1`,
candidate output, sliced metrics, position-1 fixture, schema-2 build/readback,
Metal reports, hardware, cold/warm state, complete walls, bytes, and Gate 8.

Both construction and execution reports set `accepted_tokens: 0`, `A: 0`,
`U: 0`, and `performance_claim: null`.

## Claims excluded

- L1 or source-exact weights; this remains explicit L3;
- identities, layers, routes, or density beyond the frozen expert/row;
- density five or higher, a complete bank, cache, or checkpoint;
- endpoint or sustained accepted-token TPS, target-faithful labeling, hosted
  parity, multimodal/long-context capability, or runtime-default promotion;
- companion hardware.

## Result

Unexecuted.

## Decision

Unexecuted. Stage A is the only authorized first action.
