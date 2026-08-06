# PW-0097 — BF16-staged Metal one-row expert

- Status: complete
- Disposition: conditional
- Date: 2026-08-06
- Owner: Codex with project owner authorization
- Commit and dirty state: contract `e7123ce`; executable implementation
  `ad6e1fb404d9d832c3abceb6423df3cb529d7c07`; clean tree for the two
  disposition runs
- Checkpoint/reference hashes: MiMo revision
  `63651580ca774f8504f676040460aed3e1244ac1`; PW-0034 input
  `ad261a98cc64c34277a40168f45654cabb1c1059e88771c3c71092ae6ffee5ba`;
  checkpoint verification
  `9ddc8a99755f04ae2ea3c2484f6dd022d3f3a681b5a72c915ee4de833dbb0d03`
- Hardware/runtime: Apple M1 shared 16 GiB host; Rust-owned Metal plus CPU
  source staging; real layer-43 expert-32 source tensors
- Related records: PW-0033 through PW-0036, PW-0095, PW-0096

## Hypothesis and mechanism

The existing Metal source-FP8 projection kernel can replace per-matrix F32
expansion for a production-shaped one-row expert while preserving the source
execution contract around it. Quantize/dequantize each input activation group
exactly as the endpoint, project gate and up directly from source FP8, round
both to BF16, apply the source SwiGLU staging, round its result to BF16, project
down directly from source FP8, and round the final expert output to BF16.

Compile one reusable Metal executor. Each timed execution must still install
the real expert's weight/scale views into bounded shared Metal buffers; do not
hide a 25 MB per-expert copy in setup or retain hundreds of experts. Use the
frozen PW-0034 one-row input and a newly generated independent PyTorch oracle
with the exact dynamic-FP8/BF16 source boundaries.

## Gates

Add deterministic tests for dynamic activation bytes/dequantization, BF16
gate/up/SwiGLU/down staging, shape/layout rejection, non-finite rejection, and
create-new output. Gate/up/down tensor names, shapes, source bytes, scale grids,
input hash, checkpoint revision, and reference hash fail closed.

All 4,096 outputs must be finite. Against independent PyTorch, relative L2 must
be at most `5e-4`, maximum absolute error at most `2e-2`, BF16 equality at least
99%, and the topological error must not exceed the existing target-faithful
component gates. Two clean candidate processes must produce byte-identical
output.

After one untimed compile and five warmups, run 30 complete expert executions
including dynamic activation staging, real tensor-to-Metal buffer installation,
gate/up dispatch, CPU BF16/SwiGLU staging, down dispatch, completion waits, and
readback. Median must be at most 100 ms in both processes and improve at least
10x over PW-0096's 3,180 ms average routed layer divided by eight experts
(`397.5 ms/expert`). Record cold first execution, p10/median/p90, bytes, RSS,
batch 1, concurrency 1, accepted tokens 0, `A=0`, `U=1`, device, compiler,
commit, and cache state.

Enforce Gate 8 before compilation, after warmups, after the timed series, and
after buffer release. This component cannot become accepted TPS or a
performance default. Endpoint integration requires a later real routed-layer
and complete-token experiment.

## Result

The independent PyTorch oracle completed in 7,862.449 ms. Its manifest is
`/Users/chad/Models/mimo-prismwing/evidence/PW-0097/reference-001/manifest.json`
with SHA-256
`7cd469837a30beda871daa5b010c1819aee459d22a5ad6a278062002284e995b`;
its widened-BF16 output hashes to
`bcd5926faf53cee9c4d96e489263f19c2a15a0b0dbe3e0544dfc53cb42c9b5bc`.
The oracle peaked at 353,665,024 resident bytes, retained 78% free memory,
caused no swap growth or throttling, and preserved every protected service.

The first candidate attempt failed closed before writing output: relative L2
was `0.0140689`, maximum absolute error `1.4305e-6`, and BF16 equality only
11.5967%. Diagnosis found that candidate SwiGLU rounded only the product,
whereas the source rounds SiLU to BF16 before multiplying by `up`. Commit
`0770da1` repaired that semantic boundary and added a fixture; no gate changed.

Two clean processes at the pinned final commit then passed every gate. Raw
reports and hashes are:

- `candidate-004/report.json`:
  `5a374f78a62fe85cb6d6787592e12d55ba2e5aec1e095c1bf021642486e4fbe0`
- `candidate-005/report.json`:
  `0c6b5a69b617b7eaabcb6fd7b8248526419b4b933c2e4c5d3c6c223f714c4dd4`

Both outputs are byte-identical with SHA-256
`144845edebafc3b9cd8c045fc29ce3f385d4200d61451454c7ec837b5f1bca06`.
Against the independent oracle, all 4,096 values are finite, relative L2 is
`1.78340e-5`, maximum absolute error is `2.98023e-8`, and 4,094/4,096 widened
BF16 values are identical (99.9512%). The two 30-measurement timing series are:

| Process | Cold ms | p10 ms | Median ms | p90 ms | Speedup vs 397.5 ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| 004 | 7.3225 | 6.3625 | 6.7615 | 6.9213 | 58.7887x |
| 005 | 7.5583 | 6.3788 | 6.7661 | 6.9324 | 58.7489x |

The mean median is 6.763792 ms. Every timed execution includes exact dynamic
activation staging before both linear stages, 25,204,736 logical source/I/O
bytes, real tensor-to-shared-Metal installation, three dispatches and waits,
BF16 boundaries, CPU SwiGLU, readback, and buffer destruction. No retained
tensor buffer exceeds 8,390,656 bytes. The full-file identity pass warms the
OS source cache before timing; pipeline and LUT remain application-resident.
The updated throughput model hashes to
`e5a4b7fae200f80c5fcf5b2360e827e21eb541a11e64e60f68ba3b05a5930d89`.

Both runs remained at 78% free memory, had zero swap growth and new throttled
pages, preserved ChatGPT, WindowServer, nxnode, and syncthing, and completed
buffer-release checks. Peak process RSS was at most 67,780,608 bytes and the
post-release footprint at most 24,561,216 bytes. The source identities are the
verified PW-0049 down-shard hash
`fd89388271eac237e06ace68a832156357b42f85820856afee24da7bb36d9dcc`
and gate/up artifact hash
`ca02748075edd889014c1e5beb4a2ce2abd96c1a2adebe5bd3faf278aa724276`.
The apparent difference from the earlier repository model-lock size/hash was
checked: both platter and SSD copies match the frozen PW-0049 verified file
identity byte-for-byte.

Environment: Apple M1, 16 GiB shared memory, macOS 26.6 (25G72), Rust 1.96.0,
LLVM 22.1.2, batch 1, concurrency 1, accepted tokens 0, `A=0`, `U=1`.
Runtime Metal source compilation works; the optional standalone Xcode Metal
toolchain is not installed and was not required.

## Decision

Promote the BF16-staged, bounded-buffer Metal expert executor as the explicit
target-faithful candidate for a real routed-layer integration experiment. It
is conditional because only one production-shaped expert and warm OS-cache
state have been measured. Do not promote it as the endpoint default or report
its 6.76 ms component timing as TPS.

Next, execute the actual eight routed experts for one retained-cache layer
using checkpoint-index tensor authority, exact route weights, bounded serial
buffer lifetime, interleaved CPU control/candidate runs, and the same Gate 8
contract. Only then integrate the candidate across the complete one-token
endpoint and measure accepted output end to end.
