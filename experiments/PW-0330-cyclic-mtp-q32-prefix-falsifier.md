# PW-0330 — Cyclic MiMo-MTP q32 prefix falsifier

- Status: complete
- Disposition: conditional
- Date: 2026-08-27
- Owner: Codex
- Parent experiments: PW-0136, PW-0207, PW-0211, PW-0327, PW-0328

## Question

On corrected code transaction zero, how far does the cheapest causal wide MiMo
draft survive when the pinned three non-chain MTP layers are reused cyclically,
and does the authenticated target prefix alone put that source-FP8 q32 schedule
at or below one accepted token/s before a q32 verifier is built?

This is an L2 draft falsifier. The target verifier remains exact and sole output
authority. Reusing the three trained heads beyond q4 is a distinctly named
modified scheduler, not native trained q32 behavior and not a model change.
Companion hardware is inadmissible.

## Prediction and cheap ordering

The q32 schedule needs 31 serial MTP forwards and a real width-32 target pass.
The first three draft heads use the pinned q4 chain; the fourth is the first new
reuse of layer zero. Test only those four heads first. If any head mismatches,
later heads cannot increase acceptance and the exact target route prefix gives
the complete causal byte floor for the resulting `A <= 4` transaction.

The planning authority is PW-0327 code transaction zero: anchor `8420`, target
tokens `[374, 264, 4583, 8129, 315, 264, 3084, 2268]`, and prefix identity
counts for `A=1..8` of `[376, 632, 832, 1035, 1236, 1391, 1537, 1757]`.
PW-0328 is capturing a fresh same-kernel history and must replace these planning
values as execution authority. No result may silently retain PW-0327 values if
the new target tokens, routes, or identities differ.

## Frozen scheduler

Name the proposal semantic `cyclic_mtp_012_v1`.

```text
H  = exact committed target layer-47 hidden history [rows, 4096]
x0 = committed target input IDs shifted left, with the current anchor appended

for j = 0..30:
    layer_j = j mod 3
    p_j = argmax(MTP[layer_j](H, x_j))
    x_(j+1) = x_j[1:] + [p_j]

proposal = [anchor, p_0, ..., p_30]
```

`H` is immutable. Only the token IDs rotate. The full schedule uses layer zero
11 times and layers one and two 10 times each. MiMo is non-chain in the pinned
SGLang worker: do not append draft block hidden, consume future target hidden,
teacher-force a target token, branch, or substitute a target-model proposal.
Each would be a different experiment.

If the first mismatch is zero-based draft index `j`, exact greedy verification
authorizes `A=j+1`: the `j` matching drafts, if any, plus the target correction.
If all 31 drafts match, the repaired full-match transaction authorizes `A=32`,
including the target bonus. This proposal-only experiment accepts zero tokens.

## Authorities and input freeze

The executable record must fail closed on all of the following:

1. PW-0327 canonical analysis SHA-256
   `a54eeab1d136b938ddebe01a4206d6084bbeb2a2ca6a1395d88edfac337eaeed`,
   code report SHA-256
   `83f9a37ae0da6e12b3289d70d3295539b0e4c67f8aaaa084cbcf0e1ef236910e`,
   and progress SHA-256
   `df941ef2989ffe3acfc88318ba55171622be5e0ed0c4b68b5152480ab24237cc`.
   These are planning and cross-check authorities, not the proposal history.
2. Fresh PW-0328 code prefill report
   `/Volumes/Elements/mimo-prismwing/evidence/PW-0328/prefill-001/code/report.json`,
   SHA-256
   `c39c7c86ec001c80ab64a2e258b0d7b8a2e96205d1201eec2b67b6d12ae05aa9`,
   and 70-by-4,096 little-endian finite F32 hidden payload SHA-256
   `616ac368c4893517083fef39e58ecc41b85001cdac7ddedf9db66d3ea249b938`.
3. The fresh PW-0328 code generation report, progress log, and verifier-hidden
   payload from clean capture commit
   `26d2ea31852c0d63bd022df6d571fd722137c39f`:
   `/Volumes/Elements/mimo-prismwing/evidence/PW-0328/corpus-001/code/report.json`
   SHA-256
   `e5c896e72654bfdd963bc984293b742b3687d2fac9873444f2c591726e3dd287`,
   `/Volumes/Elements/mimo-prismwing/evidence/PW-0328/corpus-001/code/report.progress.jsonl`
   SHA-256
   `77cd2af85d2b0f90f1e94de61947c4490589b8b4336c7f1746f83b48ac69df1e`,
   and
   `/Volumes/Elements/mimo-prismwing/evidence/PW-0328/corpus-001/code/verification-layer47-hidden.f32`
   SHA-256
   `31b9941ddd1446184ad1ef8050fda130cfc1aabb4115c1beb15bab943a211c2b`.
   The independent corpus auditor passes nine chronological windows, eight
   primary windows, transaction-zero full convergence, primary
   `A=[8,8,2,8,8,8,8,8]`, and `A/U=1.5256751084371063`. Require the
   repaired target-bonus semantic, transaction zero, exact prompt/prefill
   agreement, all route rows, byte-ledger closure, and Gate 8. Transaction zero
   must be a full match with `proposal_converged=true`, full `A=8`, and eight
   verifier-retained proposal rows. Otherwise its target-prefix authority ends
   at the first correction and this experiment stops unexecuted; rejected-branch
   posterior rows or routes may never seed a cyclic proposal comparison.
4. Model lock SHA-256
   `df8c74e6f9e1cef154aae5881b9042777653206aaff72855f7b1a1340e0d1050`,
   checkpoint receipt SHA-256
   `9ddc8a99755f04ae2ea3c2484f6dd022d3f3a681b5a72c915ee4de833dbb0d03`,
   kernel SHA-256
   `9bc149eee32ebf28af35929d5fa160edfe9e1767cdcde59a54ec61b7016882ee`,
   and exact prompt, tokenizer, revision, hardware, batch-one, and
   concurrency-one identities from the two PW-0328 reports.
5. MTP payload SHA-256
   `a0e41a193b2762b0c83e577f83206d0777028de6916408c8c368730c0c9e2143`,
   SGLang source lock SHA-256
   `8a0db42bedbee1d0c8dbd1d5439c5b7baacf4cf7eb8beb20c011158730fc242b`,
   and its pinned MiMo non-chain semantic.
6. PW-0211 known last-row/full-row validation SHA-256
   `395e61eb628c1b9ec3c892d285f5b3d0bc0749b6e5e7bc782cb5671dd299645f`.
   Reproduce this bit-identical logits control before the new history.
7. PW-0215's two clean code transaction-zero q4 proposal reports at
   `/Volumes/Elements/mimo-prismwing/evidence/PW-0215/native-mtp-slice-broadening-001/`,
   SHA-256
   `5b5adb10b9e5fefc016764c2ab97c4c1c62ad29e9d48b88e0829d19cd4205c62`
   and
   `e56b27b2464bfd1155037302b2cf183108b6bb38279e64653f73ecc325868fb5`.
   Both bind the same fresh hidden SHA-256 as item 2 and produce exact q4 block
   `[8420, 374, 264, 4583]`; independently reproduce it before layer reuse.
8. PW-0207 `offline-002.json` SHA-256
   `1dedbef7c79aa23835d194f52760a1f2c65dcca1481bd6df2d5602615c3fdad6`
   and the receipt-authenticated index. Independently reconstruct 381 fixed
   target objects and `7,743,236,992` logical source bytes.
9. PW-0136 raw SHA-256
   `e6ab84cada19c6036ee7b83f318c3920631141b9ea5e882cc88eb9784d0b5a56`
   and analysis SHA-256
   `7ebf2cde5c4a3f4931d2d705993f822e38af13ea66bc3efc91410296b14e2aab`.
   Use its slightly faster rounded historical bandwidth
   `3,470,448,309.677419` bytes/s for the hard candidate-favorable ceiling and
   also report the exact raw-derived `3,470,425,919.832775` bytes/s.

Historical PW-0208 prefill hidden is inadmissible because its kernel differs.
PW-0327 has no prefill hidden payload and cannot seed the proposer.

## Staged execution

1. Add fixtures, authenticate every authority, and reproduce PW-0211's known
   layer-zero bit-identical control.
2. Run heads zero, one, and two, then the first reused layer-zero head. Stop at
   the first mismatch. Record a positive match only against the fresh PW-0328
   target posterior.
3. If all four match, continue through head seven, still stopping immediately
   on mismatch. The q8 target authority is exhausted after target token seven.
4. If all eight heads match, report `prefix_authority_exhausted`; do not infer
   q32 acceptance or stitch later q8 transactions. Predeclare a direct q32
   target capture before evaluating heads eight through 30.

Each head records immutable hidden hash, complete rotated input IDs and hash,
layer index, proposal token, logits hash, top 20, target token and rank, match,
stage and complete wall, logical source bytes, process disk bytes, cold/warm
state, safety snapshots, clean commit, and all source identities. The report
records `accepted_tokens: 0` and `performance_claim: null`.

## Source-FP8 joint-residency byte floor

For authenticated target prefix length `A`, independently reconstruct the
number `N_A` of distinct layer-qualified expert records across the first `A`
verifier rows. Use the strongest physically admissible joint residency,
`R = 12 GiB`, and charge:

```text
S_fixed = 7,743,236,992
S_mtp_only = 1,189,400,448
source_expert = 25,171,968
B_favorable = 3,470,448,309.677419 bytes/s
M_A = max(0, S_fixed + S_mtp_only + N_A * source_expert - R)
TPS_ceiling_A = A * B_favorable / M_A
```

This is more favorable than an implementation: it grants perfect fractional
future-aware residency and free time sharing; omits embedding rows, alignment,
largest-object slack, every unaccepted suffix route, 31 MTP weight scans,
target shared-weight scans, DRAM work, compute, attention, routing,
synchronization, correction, rollback, sampling, and endpoint overhead.
The LM head is already in `S_fixed` and must not be double-counted. MTP weights
are additional and may not spend the same 12 GiB as fixed or expert objects.

As a planning cross-check only, PW-0327 code gives source-FP8 ceilings for
`A=1..8` of `[0.629572, 0.580516, 0.612763, 0.628115, 0.638883,
0.670360, 0.699344, 0.689352]` TPS. Execution must recompute them from PW-0328.

## Frozen gates and dispositions

- Any first-eight mismatch with a recomputed `TPS_ceiling_A <= 1` rejects
  `cyclic_mtp_012_v1` plus a q8-chunked, prefix-bit-identical source-FP8 q32
  verifier on the required code slice before that verifier is implemented.
  The proposal runner reports this as a conditional hard storage rejection:
  the direct-q32 follow-up must still prove real first-chunk parity before
  closing the combined verifier. This is not an achieved measurement or a
  theorem about unknown future proposers. A different q32 arithmetic must
  qualify its target prefix and routes directly; width-dependent token drift is
  not an assumed rescue.
- A mismatch whose ceiling remains above one is analytical only and requires a
  direct q32 verifier trace because the divergent suffix routes are unknown.
- Eight matching heads authorize only a direct-q32 capture contract. They do
  not authorize heads nine through 31, implementation, or throughput.
- CPU proposal wall is diagnostic. A future endpoint must satisfy strict
  complete post-prefill wall `< A` seconds at one TPS and include all omitted
  work.

## Required fixtures

- reject stale/wrong-kernel hidden and every wrong hash, shape, category,
  transaction, commit, model, or route authority;
- reject a non-converged target-self transaction, clipped `A`, or fewer than
  eight retained rows as an eight-token target spine;
- preserve immutable hidden and exact shifted/rotated token IDs;
- reproduce the trained q4 layer order and freeze the fourth reused-layer
  token as the first new scheduler behavior;
- cover mismatches at heads zero, three, and seven and the eight-match
  exhausted-prefix disposition;
- prove synthetically that an arbitrary proposal suffix cannot change earlier
  causal target inputs; defer real bit-identical first-chunk hidden, router
  scores, expert order, route weights, and posterior parity to the conditional
  direct-q32 follow-up;
- reconstruct `N_A` independently from eight-expert route rows;
- reproduce the byte table, joint-residency no-double-spend rule, zero-miss
  branch, exact-versus-rounded bandwidth, and disposition precedence; and
- fail closed on safety, dirty Git, overwrite, or evidence-schema mismatch.

## Claims excluded

- actual q32 acceptance, achieved or endpoint TPS, or a sustained result;
- a native trained q32 semantic, q32 suffix routes, or prefix parity for an
  unimplemented width-32 target kernel;
- K4 fidelity, bank construction, cache allocation, or any K4 byte claim;
- multimodal/full-capability promotion, a runtime default, or companion
  hardware.

## Result

Executed once from clean commit
`1bb8645775b014aa2a59ac1c80d5edd48b05ea90` on Apple M1 16 GiB, batch one,
concurrency one; the process exited zero:

```text
python3 tools/run_pw0330_cyclic_mtp_prefix.py \
  --repo /Volumes/Elements/mimo-prismwing/worktrees/pw0330-run \
  --commit 1bb8645775b014aa2a59ac1c80d5edd48b05ea90 \
  --checkpoint /Users/chad/Models/mimo-prismwing/checkpoints/MiMo-V2.5-63651580 \
  --verification /Volumes/Elements/mimo-prismwing/cold-assets/internal-ssd-migration-2026-08-26/Users/chad/Models/mimo-prismwing/evidence/PW-0049/checkpoint-verification.json \
  --source-root /Volumes/Elements/mimo-prismwing/research-sources/sglang-dflash \
  --output /Volumes/Elements/mimo-prismwing/evidence/PW-0330/run-001
```

The canonical raw report is
`/Volumes/Elements/mimo-prismwing/evidence/PW-0330/run-001/report.json`,
SHA-256
`fbb454f6992ba8e21ade89aff416a494d14625dc126b769f420a861ed6414674`.
The runner SHA-256 is
`298acfa4424991e860939b46e818063da658c6dfd32b7526fe53764cd36d1746`.

Every authority and both q4 controls authenticated. Fresh PW-0328 routes
reproduced the planning prefix counts exactly. Heads zero through two matched
the target with bit-identical logits and proposal tokens `[374,264,4583]`.
The first cyclic reuse, head three/layer zero, proposed token `13`; the exact
target token was `8129`, rank two. The frozen correction rule therefore fixes
the conditional direct-q32 transaction at `A=4` if its first chunk is proven
prefix-identical, and later cyclic heads cannot increase that acceptance.

The first four verifier rows contain `N_A=1,035` distinct layer-qualified
source-FP8 experts. Fixed target bytes (`7,743,236,992`), additional MTP bytes
(`1,189,400,448`), and those expert bytes total `34,985,624,320`. After the
candidate-favorable perfect 12-GiB joint-residency grant, the unavoidable miss
floor is `22,100,722,432` bytes. The rounded favorable PW-0136 bandwidth gives
an optimistic storage-only ceiling of `0.6281149080724167` accepted token/s;
the exact raw-derived bandwidth gives `0.6281108557443151`.

The complete runner took `15,830.87141699798` ms, including authority checks
and the required known control. That wall is not endpoint TPS and does not
include a q32 verifier. Gate 8 passed with at least 66% free memory, zero swap
growth, zero new throttling, and a maximum recorded process peak resident size
of `4,152,442,880` bytes. This proposal-only run accepted zero endpoint tokens
and changes no achieved TPS or runtime default.

## Decision

Conditionally reject `cyclic_mtp_012_v1` combined with a q8-chunked,
prefix-bit-identical source-FP8 q32 verifier on the required code slice. Its
authenticated `A=4` prefix is already below one TPS under an impossible-best
storage model, before proposer, verifier compute, common-weight scans, or
endpoint work.

This kills the named composition, not every future wide proposer and not an
implemented direct-q32 target path. A direct q32 follow-up would still have to
prove first-chunk hidden, routing, and posterior parity before upgrading the
conditional rejection to a combined-verifier closure. Do not execute heads
four through 30 from this prefix authority. Preserve K4 and exact-codec
branches as separate remaining work.
