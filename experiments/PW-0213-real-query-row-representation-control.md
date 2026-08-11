# PW-0213 — Real-query row-representation control

- Status: complete
- Disposition: rejected at bounded early-gate probe
- Date: 2026-08-11
- Owner: Thimble with project-owner authorization
- Contract commit: uncommitted pre-execution record
- Checkpoint: XiaomiMiMo/MiMo-V2.5 revision
  `63651580ca774f8504f676040460aed3e1244ac1`
- Query authority: PW-0116 corpus
  `b9df976876d63c1ffbbe0c70507aea8b939a749ce5b1db27cbca0b5d82cf802e`
- Exactness: L3 modified weight representations
- Related records: PW-0020, PW-0116, PW-0138, PW-0141, PW-0148,
  PW-0211, PW-0212

## Question

PW-0212 rejects a greedy tile-local FP8 subset because parameter-space error
is strongly nonuniform across experts and projections. Test the more relevant
object directly: can a row be stored as a compact query structure that
estimates `w dot x` on real routed queries more accurately than conventional
six-bit weight quantization at an executable byte and operation budget?

This is not a generic matrix reconstruction contest. It is an asymmetric
streaming query experiment: stored rows are compressed offline; real query
vectors remain available at runtime; shared query transforms may be amortized
across every row of one projection.

## Frozen data and partitions

Use the same three experts and placements as PW-0138:

- layer 4 / expert 96: 109 train, 56 validation placements;
- layer 24 / expert 22: 26 train, 56 validation placements; and
- layer 46 / expert 28: 100 train, 56 validation placements.

Positions `0..111` may fit train-dependent representations. Positions
`112..167` are the sole numerical comparison set. Positions `168..223` remain
untouched. Gate/up queries are the captured real MoE inputs. Down queries are
source-derived BF16 SwiGLU activations formed independently for train and
validation. Source projection outputs are the immutable reference.

The raw PW-0116 corpus must hash to its recorded authority before any fitting.
Do not recreate a synthetic query distribution when the corpus is absent.

## Frozen candidates

Evaluate all candidates on identical source rows and validation queries:

1. **Affine6 RTN control.** Independent 128-value row groups, six-bit codes,
   and F16 scale/bias, matching PW-0148's byte semantics.
2. **FP8-subset6 control.** The rejected PW-0212 tile-local greedy 64-source-
   value codebook, retained to connect weight error to dot-product error.
3. **TurboQuant-MSE6.** One deterministic seeded randomized Walsh-Hadamard
   rotation shared by a projection, followed by a frozen six-bit scalar
   quantizer in rotated coordinates. Transform the query with the identical
   orthogonal map before the packed dot product.
4. **TurboQuant-PROD6.** Five-bit rotated scalar reconstruction plus a one-bit
   structured QJL sign sketch of its residual, including every required norm,
   seed, codebook, and alignment byte. Report the base and correction terms
   separately so bias reduction cannot hide variance.
5. **Block-covariance6.** For each 128-channel block, fit one orthogonal basis
   from train queries only, shared across all rows of the projection, rotate
   both stored row blocks and runtime query blocks, then apply the same frozen
   six-bit scalar rule. Charge the bases once at their declared layer/bank
   sharing scope and charge their query-transform MACs.

Dense unstructured Gaussian JL or a full 4096-by-4096 covariance basis is not
an executable candidate: either requires an uncharged dense query transform
or a large stored matrix. It may appear only as a labeled oracle bound.

## Byte and operation ledgers

Report, without amortization and at declared full-bank amortization:

- code/index, norm, scale/bias, codebook, basis, seed, offset, and alignment
  bytes;
- bytes per source row, projection, expert, and complete 12,032-expert bank;
- packed dot-product operations per output, shared query-transform operations,
  residual-correction operations, and total operations per projection;
- arithmetic intensity and the number of independent streamed payloads.

The equal-byte comparison uses at most 75.1% of source FP8 weight bytes after
all row-local metadata. The affine6 control's 78.125% ledger is reported as a
quality control but is not silently treated as equal size. The equal-operation
comparison must either remain within 10% of affine6's projection operations or
reduce a candidate's sketch dimension until it does. TurboQuant-PROD6 may also
be reported at equal bytes with higher operations, but cannot win the joint
gate from that point alone.

## Measurements and continuation gate

For every expert/projection/candidate report validation:

- dot-product relative L2 over the complete output matrix;
- mean signed error, normalized bias, and error variance;
- median, p95, and maximum row-relative L2;
- cosine similarity and top-error query/row identities; and
- deterministic seed-to-seed range for five predeclared seeds where random
  transforms participate.

The seed panel is `{16001, 16002, 16003, 16004, 16005}`. Seed 16001 was the
bounded early-gate implementation probe; the other four values were frozen
immediately after that probe and before panel expansion. Preserve this timing
rather than representing the entire panel as predeclared before first contact.

A representation survives only if it:

1. fits the equal-byte ledger;
2. fits the equal-operation ledger;
3. improves validation dot-product relative L2 over affine6 RTN in all nine
   projections and by at least 15% in aggregate;
4. has no projection above 2% relative L2 and no validation row above 5%;
5. does not trade lower L2 for larger absolute normalized bias; and
6. passes every identity, partition, finiteness, byte, seed, and memory gate.

Failure rejects only the named row representation and frozen fitting rule. A
pass authorizes a separately frozen complete-expert SwiGLU/routed-output test;
it does not authorize a bank, decoder, Metal kernel, accumulated model,
holdout access, endpoint claim, or throughput promotion.

## Pre-execution boundary (resolved)

The Linux resident host initially had the pinned PW-0212 source-weight census but not the
132,120,576-byte raw PW-0116 corpus. Its recorded path is on the prior Mac:
`/Users/chad/Models/mimo-prismwing/evidence/PW-0116/corpus-001/`. Harness and
fixture work proceeded locally while a hash-preserving transfer was arranged;
the result below records its successful resolution.

## Execution and result

The complete PW-0116 directory was transferred from the prior Mac with the
dedicated migration identity. Its manifest reproduced SHA-256
`b9df976876d63c1ffbbe0c70507aea8b939a749ce5b1db27cbca0b5d82cf802e`,
and the repository's independent analyzer accepted every payload hash,
partition, reconstruction, and authority gate. The three source experts were
then materialized through sequential pinned range reads; their local source
archive hashes to
`d2a51359610d5851b8769cc2ceaaf31813042cb1a5a7c3758781e506f83a0f74`.

The frozen cheapest failure boundary was layer 4 / expert 96 / gate, the same
projection on which PW-0212 showed its clearest nonuniform regression. Its 56
real validation queries give affine6 RTN relative L2 `0.00636152`.

Across seeds 16001–16005:

- TurboQuant-MSE6 ranges from `0.00729956` to `0.00837045`, median
  `0.00772214`, a 21.39% regression from affine6.
- TurboQuant-PROD6 structured-QJL ranges from `0.00913397` to `0.01042873`,
  median `0.01000513`, a 57.28% regression. It reduces normalized signed bias
  to `[-0.0000488, 0.0001576]`, but pays for that with substantially higher
  variance and total error.
- The post-contact block-covariance/shared-64-level-grid variant reaches
  `0.03632544`, a 470.99% regression with normalized bias `0.00940896`.

Two covariance diagnostics are preserved but excluded from candidate evidence.
The first accidentally applied covariance whitening rather than the frozen
orthogonal basis; the second combined an orthogonal block basis with a
sphere-global grid that falsely assumed equal energy across blocks. Their
failures led to the correctly named shared-grid variant above and are not used
to strengthen its result.

Validated analysis hashes to
`5d9cd7b8b27ebb752c8c1f536288f18e2de0119f4a5e56d23b6ba0fab979fdb6`.
The canonical shared-grid probe hashes to
`c48e1af7536b648ec86ad138e9c4fb94f04873e9762ce396e1ef4fba3b5c69c3`.

## Decision

Reject the tested data-oblivious TurboQuant-MSE6, structured TurboQuant-PROD6,
and block-covariance/shared-grid6 row representations before nine-projection
expansion. Each fails the required improvement over affine6 at the first
frozen hard projection. Bias correction alone is not fidelity, and no decoder,
bank, Metal kernel, holdout access, or endpoint work follows.

This does not reject whole-expert SwiGLU reparameterization or a representation
that jointly changes the three expert projections. Those do not preserve the
row-query abstraction and require a separate contract.
