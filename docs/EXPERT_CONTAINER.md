# PWEXPRT1 expert container

`PWEXPRT1` is Prismwing's lossless, fail-closed tensor container for bounded
expert streaming. It is a storage and identity format, not a quantization
format. Source bytes are copied without numerical conversion.

## Binary layout

| Offset | Bytes | Meaning |
| ---: | ---: | --- |
| 0 | 8 | ASCII magic `PWEXPRT1` |
| 8 | 8 | little-endian u64 JSON-header length |
| 16 | variable | UTF-8 JSON header |
| next 64-byte boundary | variable | aligned tensor payloads |

Every tensor payload begins at a 64-byte-aligned offset relative to the payload
region. Padding bytes are zero. The file ends at the last tensor byte; trailing
bytes fail verification.

## Header schema version 1

The JSON header records:

- source safetensors filename, complete file size, and complete SHA-256;
- required payload alignment;
- for each tensor, its name, dtype, shape, source data-offset pair,
  payload-relative offset, byte count, and payload SHA-256.

Tensor names are sorted and unique. Unknown magic or schema, malformed source
metadata, duplicate names, unaligned/overlapping/out-of-range payloads,
unexpected trailing bytes, and any hash mismatch fail closed.

## Creation and verification

```sh
cargo run --release -- repack SOURCE.safetensors OUTPUT.pwexpert TENSOR [TENSOR ...]
cargo run --release -- verify-container OUTPUT.pwexpert
```

Creation refuses to overwrite an existing target. It writes and verifies a
same-directory temporary file, then uses a no-clobber hard link to publish the
artifact atomically. Model weights and generated containers remain outside
Git; experiment records commit their identities and representative schemas.
