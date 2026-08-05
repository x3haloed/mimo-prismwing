# PW-0001 — OpenRouter reference viability

- Status: running
- Disposition: unexecuted
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

Text reference viability passes conditionally. Reasoning mode must be disabled
explicitly for scored completions. The experiment remains running until every
required modality is probed; no conclusion transfers from text to media.
