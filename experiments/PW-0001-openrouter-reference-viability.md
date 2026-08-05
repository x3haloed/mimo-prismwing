# PW-0001 — OpenRouter reference viability

- Status: complete
- Disposition: production
- Date: 2026-08-04
- Owner: Codex with project owner authorization
- Commit and dirty state: implementation commit pending; repository dirty
  during initial capture
- Checkpoint/processor/reference hashes: checkpoint revision discovery pending
- Hardware, OS, compiler, storage, memory pressure: Macmini9,1; 16 GiB; macOS
  26.4.1 (25E253); external evidence on `/Volumes/Elements`
- Related records: none

## Hypothesis and mechanism

At least one individually pinned OpenRouter provider exposes immutable raw
responses with `logprobs` and `top_logprobs=20` for every required input
modality. If true, those responses can serve as the project's only external
whole-model reference while source-derived oracles cover component semantics.

## Contract

This experiment does not alter the model. It tests the reference evidence
boundary. Requests pin exactly one provider, disable fallbacks, require every
parameter, retain raw JSON, and never contain private media. A modality passes
only if the returned response contains usable token-level top-20 logprobs and
identifies the actual provider. Text success does not imply modality success.

Success: text, image, multi-image, audio, video, and mixed probes all return
the required evidence from one stable provider.

Kill: no provider can expose the required logprobs for a required modality. In
that case the existing distributional gate is not executable and the project
must stop for an explicit scope/target decision rather than waive it silently.

## Baseline and candidate

Endpoint discovery on 2026-08-04 found that Parasail advertises `logprobs` and
`top_logprobs` for `xiaomi/mimo-v2.5`; other discovered endpoints did not
advertise both. The first candidate is therefore Parasail with fallbacks
disabled.

The first text request allowed reasoning to consume its eight-token completion
budget. It returned no visible content or logprobs. The second request disabled
reasoning explicitly and allowed 32 completion tokens. Exact commands:

```sh
python3 tools/openrouter_reference.py capture \
  evals/fixtures/requests/text-smoke.json \
  /Volumes/Elements/mimo-prismwing/evidence/PW-0001/text-smoke-002
python3 tools/openrouter_reference.py verify \
  /Volumes/Elements/mimo-prismwing/evidence/PW-0001/text-smoke-002
```

## Isolated attribution

The successful request used 34 prompt tokens and four completion tokens, cost
USD 0.00000408 according to the response, and completed in approximately one
second. These are reference-path diagnostics, not runtime performance claims.

## End-to-end result

The second request returned `prismwing`, identified Parasail as the provider,
and supplied exactly 20 alternative logprobs for each of three visible token
positions. Its capture verifies offline. The first capture is now correctly
rejected by the stricter verifier because it contains no token logprobs.

## Correctness result

External raw-evidence manifest hashes:

- failed text capture `text-smoke-001`:
  `78abb4cef5e4ff3b1e75e1e6b70c31e64237029ef45f5704e4457d1ade25da69`
- passing text capture `text-smoke-002`:
  `62319a8d3db94d2c86baab45db97046d9636d4c733ba004abc215c8ea3aa189d`

The repository verifier passes the second and fails the first for the expected
reason. Three deterministic unit tests cover policy validation, immutable
write behavior, and offline tamper detection.

## Decision

Text, single-image, multi-image, audio, video, and mixed image/audio requests
all returned from pinned provider Parasail with 20 alternatives at every
visible token position. The synthetic tasks also produced semantically correct
answers; the image cases reached their deliberately small output limit only
after correctly identifying the supplied colors.

Additional external raw-evidence manifest hashes:

- image: `ed78d4c973668c7775f83b66ca35de1b64e977527f7fc3ad9ffb77160f0c4845`
- multi-image: `b80b99894804d70edcf1e8126635b5a2dd34358ebf429d94c238f4707c2a485e`
- audio: `446967a10c0cc260c76096f6d0f5b6b54a9e61801d3361aafff93aba80cab372`
- video: `d950008fbd855adf3286aea8c41c048a99269675f9fe1b9259e3dae6b5f5db4f`
- mixed image/audio:
  `39bf48021e94f57580f9e92f5dc478d601818e1b49c6909db8ff938479643c5e`

Promote Parasail as the initial reference provider and the fail-closed capture
tool as the reference acquisition path. This is a viability result, not a
frozen final epoch: endpoint metadata, a larger canary set, fixture licenses,
and final reference parameters must still be frozen before quality evaluation.
