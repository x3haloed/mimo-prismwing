# Evaluation assets and evidence

This directory will hold manifests and tooling, not private data or model
weights.

Planned layout:

```text
evals/
  fixtures/
    manifest.jsonl
    objects/                 # redistributable content-addressed assets only
  reference/
    <epoch-id>/
      manifest.json
      requests/
      responses/
  runs/
    <run-id>/
      manifest.json
      probabilities/
      capabilities/
      performance/
  reports/
    <run-id>.json
    <run-id>.md
```

Large raw evidence should normally live in content-addressed external storage.
The repository records hashes, acquisition instructions, schemas, and small
redistributable fixtures. `.gitignore` excludes local raw result directories by
default.

## Fixture rules

- Stable ID, slice, scorer, tags, license, and SHA-256 are mandatory.
- Media must be redistributable or represented only by a hash plus private
  acquisition instructions.
- No private user conversations or media.
- At least 20% of each final slice remains hidden until the target and runtime
  mode are frozen.
- A fixture never contains a precomputed expert route, cached answer, or
  runtime-specific hint available only to the local implementation.

## Evidence rules

- Preserve raw provider JSON before normalization.
- Remove credentials but retain model, provider, routing, parameter, timing,
  usage, and logprob metadata.
- Record failures and incomplete responses.
- Never combine different OpenRouter reference epochs.
- Every report must be derivable offline from hashes and raw evidence.

