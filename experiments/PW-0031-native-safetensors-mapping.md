# PW-0031 — Native fail-closed safetensors mapping

- Status: proposed
- Disposition: unexecuted
- Date: 2026-08-04
- Owner: Codex with project owner authorization
- Commit and dirty state: based on `b355ec2`; contract dirty
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

Pending.

## End-to-end result

Out of scope; no endpoint TPS claim is permitted.

## Correctness result

Pending.

## Decision

Pending.
