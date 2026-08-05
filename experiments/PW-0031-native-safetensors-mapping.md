# PW-0031 — Native fail-closed safetensors mapping

- Status: complete
- Disposition: production
- Date: 2026-08-04
- Owner: Codex with project owner authorization
- Commit and dirty state: contract committed as `e6e2cb1`; implementation dirty
- Checkpoint/processor/reference hashes: MiMo revision
  `63651580ca774f8504f676040460aed3e1244ac1`; locked complete MTP SHA-256
  `a0e41a193b2762b0c83e577f83206d0777028de6916408c8c368730c0c9e2143`
- Hardware, OS, compiler, storage, memory pressure: Apple M1; Macmini9,1;
  16 GiB; macOS 26.4.1 (25E253); Rust release build; source read-only on
  external platter
- Related records: PW-0001, PW-0013, PW-0030

## Hypothesis and mechanism

The Rust executable can become a real inference foundation by memory-mapping
safetensors, validating the complete layout once, and exposing immutable typed
tensor byte views without Python, full-file copies, or trusting caller-supplied
shapes. This should accept the complete locked MTP file and later accept EP0
without changing the authority for offsets and dtypes.

## Contract

Add a native `MappedSafetensors` authority and `inspect-tensor` CLI. Pass only
if:

1. header length, JSON schema, tensor names, known dtype, dimensions, element
   count, dtype byte width, data length, payload bounds, offset ordering, and
   non-overlap are checked with overflow-safe arithmetic before any view is
   returned;
2. unknown dtypes, zero/invalid header lengths, malformed metadata, duplicate
   or overlapping ranges, shape/byte mismatches, truncated payloads, and
   missing tensors fail closed in deterministic unit fixtures;
3. the mapped file and tensor views borrow immutable bytes from one read-only
   mapping; the API does not infer shapes from file length or expose mutable
   checkpoint storage;
4. a release build opens the complete locked MTP file and inspects layer-zero
   BF16 norm, FP8 fused QKV, F32 QKV scales, and FP8 MLP gate. Native dtype,
   shape, offsets, byte counts, and tensor SHA-256 must match an independent
   Python raw-range oracle exactly;
5. report cold and warm whole-command wall time, source storage, hardware,
   commit, and bytes hashed. Timing is loader-component evidence only and
   cannot become endpoint TPS or a throughput-model default.

Passing promotes the mapper as the sole native safetensors layout authority.
It does not establish a forward pass, model fidelity, or endpoint throughput.

## Baseline and candidate

Baseline is independent Python raw-header/range parsing. Candidate is the Rust
read-only memory map and validated tensor view over the same locked MTP file.

Raw evidence will be written under
`/Volumes/Elements/mimo-prismwing/evidence/PW-0031`.

## Isolated attribution

The release binary maps the complete `1,189,405,960`-byte MTP source read-only,
validates all tensor metadata, and hashes these production tensors directly
from immutable mapped views:

| Tensor | Dtype / shape | Data offsets | Bytes | SHA-256 |
|---|---|---|---:|---|
| input norm | BF16 `[4096]` | `[67325440,67333632]` | 8,192 | `f7afaa98fdf20d5d6f15612735b09befe39b4833e02b8237757091ea3b3dd847` |
| fused QKV | FP8 `[14848,4096]` | `[604295040,665112448]` | 60,817,408 | `808601ba5811ae1d424d8ecfd49e5295709692214f6d96b8a48aaf314d9154ae` |
| QKV scales | F32 `[116,32]` | `[49152,64000]` | 14,848 | `d77b12adc3a62f09b6a7ab15f630f492957c4913f5d1afaa28727735d6aa4c7d` |
| MLP gate | FP8 `[16384,4096]` | `[470077312,537186176]` | 67,108,864 | `7a65820d833aef349e059e8a7af3335056fc340fb6f3e1211839b97554f21864` |

An independent Python parser reads only the raw eight-byte header prefix,
header JSON, and declared byte ranges. It matches native dtype, shape, offsets,
byte count, and SHA-256 exactly for all four tensors.

The complete release command that maps and validates the file, locates and
hashes the 60,817,408-byte QKV tensor, and serializes JSON takes 0.18 seconds on
the first recorded application process and 0.17 seconds repeated. The source
is on the external platter, but the OS page cache was already warm from the
correctness pass; these numbers are not cold-storage latency or throughput.

## End-to-end result

Out of scope; no endpoint TPS claim is permitted.

## Correctness result

All five conditions pass with one explicitly recorded cache-state limitation
on timing. Unit fixtures reject duplicate JSON keys, unknown dtypes,
shape/dtype byte mismatches, overlaps, truncation, zero dimensions, and missing
tensors. Header and tensor arithmetic is checked, the mapping is read-only,
and tensor slices borrow the mapping rather than copying or exposing mutation.

Raw evidence is under `/Volumes/Elements/mimo-prismwing/evidence/PW-0031`.
Its `SHA256SUMS` manifest hashes to
`957d2a51a67091f48b3e5a1e7df9a4d7c3a0628906fdb4f0e2737011e3a61d9a`.

## Decision

Promote `MappedSafetensors` as the sole native source-checkpoint layout
authority. Future native tensor decode, projection, layer, and modality paths
must obtain metadata and immutable bytes through this validated view rather
than reparsing offsets or trusting caller shapes.

This advances executable foundation only. The 0.17–0.18 second commands are
hash-heavy loader diagnostics over warm OS cache, not forward-pass performance
or TPS. EP0 and complete text decode remain pending.
