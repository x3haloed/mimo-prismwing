# PW-0020 — Atomic TurboQuant MiMo compatibility audit

- Status: complete
- Disposition: rejected
- Date: 2026-08-04
- Owner: Codex with project owner authorization
- Commit and dirty state: contract committed as `a67695d`; audit tools dirty
- Checkpoint/processor/reference hashes: MiMo revision
  `63651580ca774f8504f676040460aed3e1244ac1`; Atomic TurboQuant revision
  `074bf826e1b06005a51737d29387e36657f41bf7`; see
  `spec/atomic-turboquant.lock.json`
- Hardware, OS, compiler, storage, memory pressure: Apple M1; Macmini9,1;
  16 GiB; macOS 26.4.1 (25E253); Apple clang 21.0.0; test binaries and
  generated evidence on internal SSD; live checkpoint download excluded
- Related records: PW-0002, PW-0019

## Hypothesis and mechanism

The pinned Atomic llama.cpp fork contains a mechanically compatible low-bit KV
path for MiMo's asymmetric 192-wide K and 128-wide V heads. Its WHT rotation
should preserve unquantized inner products, its packed layouts should materially
reduce the one-million-token global KV footprint, and its Metal source should
contain the exact `dk192_dv128` flash-attention specializations needed here.

This is a source and component compatibility audit, not a fidelity promotion.
The fork implements a practical WHT/Lloyd-Max scheme whose current Turbo4
default is 4-bit PolarQuant; that is not identical to every construction in the
TurboQuant paper.

## Contract

Target-faithful shapes and attention semantics; modified KV representation.
The candidate passes this compatibility audit only if all conditions hold:

1. the source revision and every locked file hash match;
2. Apple Clang compiles the pinned C implementation and its upstream test exits
   zero without non-finite output;
3. a deterministic Prismwing fixture uses K dimension 192 padded to 256, V
   dimension 128, causal softmax, and weighted V reconstruction; the same WHT
   applied to padded Q and K preserves the unquantized dot product within
   `2e-5` relative error;
4. compiled layouts are exactly 34, 50, and 68 bytes per 128 values for
   Turbo2, Turbo3, and Turbo4; all three candidates produce finite deterministic
   score and output diagnostics;
5. source inspection finds both graph-side Q rotation/KV padding and Metal
   flash-attention specializations for `dk192_dv128` for each candidate type;
6. no source assumption silently removes global history, changes the 128-token
   sliding window, drops KV heads, or truncates the 192 logical K dimensions.

Any failure rejects direct reuse until repaired. Passing permits only a minimal
isolated port to advance to accelerated parity and real-activation fidelity
tests. Quantized score/output errors are diagnostics in this audit and cannot
promote a fidelity default. The audit must report source compilation separately
from Metal runtime validation; source presence is not proof that a Metal kernel
builds or runs.

## Baseline and candidate

Baseline is FP32 attention in the deterministic fixture and FP16 for storage
accounting. Candidates are the pinned fork's Turbo2, Turbo3, and Turbo4 row
formats. Commands and raw outputs will be recorded under
`/Volumes/Elements/mimo-prismwing/evidence/PW-0020`.

The executable commands are:

```sh
ATOMIC=/Volumes/Elements/mimo-prismwing/research-sources/atomic-llama-cpp-turboquant
python3 tools/verify_source_lock.py \
  --lock spec/atomic-turboquant.lock.json \
  --source-root /Volumes/Elements/mimo-prismwing/research-sources/atomic-llama-cpp-turboquant

clang -std=c11 -O2 -I$ATOMIC/ggml/include -I$ATOMIC/ggml/src \
  $ATOMIC/tests/test-turbo-quant.c $ATOMIC/ggml/src/ggml-turbo-quant.c \
  -o /tmp/prismwing-test-turbo-quant -lm
/tmp/prismwing-test-turbo-quant

clang -std=c11 -O2 -I$ATOMIC/ggml/include -I$ATOMIC/ggml/src \
  tools/atomic_turboquant_attention_audit.c \
  -o /tmp/prismwing-atomic-turboquant-audit -lm
/tmp/prismwing-atomic-turboquant-audit
```

Here `$ATOMIC` denotes the locked external source root above; it is not an
ambient credential or mutable model path. The source-dispatch trace records the
exact `sed`/`rg` excerpts used to follow cache allocation through pipeline-name
selection and enumerate the available specializations.

MiMo storage accounting uses nine persistent global-attention layers, 39
sliding-window layers capped at 128 tokens, four KV heads in global layers,
eight KV heads in sliding layers, K=192, and V=128. Candidate K is charged for
the fork's required padding to 256.

## Isolated attribution

The locked C source compiles with Apple Clang. Its upstream test exits zero:

| Case | MSE | Cosine |
| --- | ---: | ---: |
| Turbo3 basis vector | approximately zero | 1.000000 |
| Turbo3 sinusoid | 1.345263 | 0.986448 |
| Turbo4 cosine | 0.286009 | 0.988753 |

The stronger deterministic fixture confirms the compiled layouts and the
fork's 192-to-256 K padding tax:

| Format | Bytes / 128 | 1M-context KV bytes | GiB | FP16 compression |
| --- | ---: | ---: | ---: | ---: |
| FP16 | — | 24,184,750,080 | 22.524 | 1.000× |
| Turbo2 | 34 | 3,854,444,544 | 3.590 | 6.275× |
| Turbo3 | 50 | 5,668,300,800 | 5.279 | 4.267× |
| Turbo4 | 68 | 7,708,889,088 | 7.179 | 3.137× |

Exact WHT of padded Q and K changes fixture scores by only
`1.0533791806e-7` relative L2, passing the `2e-5` invariant. All packed outputs
are deterministic and finite. Quantized diagnostics, which are not promotion
gates here, are:

| Format | Score relative L2 | Output relative L2 | Output cosine |
| --- | ---: | ---: | ---: |
| Turbo2 | 0.2833 | 0.5220 | 0.8850 |
| Turbo3 | 0.1341 | 0.2273 | 0.9740 |
| Turbo4 | 0.09960 | 0.2304 | 0.9796 |

These synthetic errors reinforce that storage compatibility cannot establish
fidelity.

## End-to-end result

Out of scope; no endpoint TPS claim is permitted from this audit.

## Correctness result

Conditions 1–4 pass. Condition 5 fails after following the source's actual
shape path rather than accepting specialization names at face value:

1. `llama-kv-cache.cpp` pads MiMo K heads from 192 to 256 for Turbo types.
2. `get_k()` exposes the padded 256-wide head and the graph pads/rotates Q to
   match it.
3. `ggml-metal-device.cpp` builds the Metal pipeline name from K and V tensor
   `ne[0]`, so MiMo requests `dk256_dv128`.
4. The Metal source instantiates `dk192_dv128` and `dk256_dv256` for each of
   Turbo2/3/4, but no `dk256_dv128` specialization.

The advertised `dk192_dv128` kernels therefore do not match the fork's own
required padded MiMo dispatch. Direct execution would fail to resolve the
pipeline rather than silently truncate K; this also prevents condition 6 from
passing as a runnable path.

An independent Metal source compile was attempted but the installed Xcode
lacks the optional Metal Toolchain component. That limitation is recorded
separately and is not the rejection cause: the dispatch specialization is
absent in source.

Evidence hashes are in
`/Volumes/Elements/mimo-prismwing/evidence/PW-0020/SHA256SUMS`; that manifest's
SHA-256 is
`7ab9349045c4d2b604962745c5ca9b1b501d04da9da2326c60939cd8832e93ed`.

## Decision

Reject the Atomic fork as a directly reusable MiMo runtime. Preserve its
locked WHT, packing, dequantization, and kernel code as research input.

The next cheapest falsification is a minimal Prismwing-owned attention path
that adds the real `dk256_dv128` specialization and tests it against the
committed 192/128 fixture. Turbo4 is the first quality-oriented candidate;
Turbo3 remains the compact branch. Neither advances past deterministic
component status until accelerated parity and real-activation attention/logit
tests pass.
