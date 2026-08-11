# PW-0216 — Native MTP untouched longer-output holdout

- Status: proposed; prompts frozen before execution
- Disposition: unexecuted
- Date: 2026-08-11
- Execution mode: L2 target-distribution-preserving native MTP draft under exact verifier-only commit
- Hardware: Apple M1 Mac mini, 16 GiB, batch one, concurrency one, internal APFS checkpoint
- Related records: PW-0208, PW-0211, PW-0215

## Question

PW-0211/PW-0215 show repeatable positive complete-path gains on four frozen
seven-token category requests. Those prompts were visible during mechanism
development and do not establish sustained proposal acceptance, correction,
rollback, or performance over a longer generated sequence. Freeze a new prompt
panel before first use and test 32 verifier-authorized output tokens.

## Frozen panel and contract

The panel contains ordinary, code, multilingual, and rare-route requests under
`evals/fixtures/requests/pw0216-holdout-*.txt`. Their byte hashes are recorded
below; prompt text, output length, and order may not be changed after seeing
proposals.

- ordinary: `1cc0561c9f1cd1782a8babcc1c667b048ebe8716c54f0211eadd4c8794ee685c`
- code: `900d31fa20fa3b839a0e49e6f3b1bfe2bb43ae8b00cbd8bbd3ceea64af0d794a`
- multilingual: `b723fa20f9e06962e1339ef59474ce5ca31c8d49c189f76e4ca955b6166c63f4`
- rare-route: `30c69b3a1f54bd3cd0c22435fb2db82e33a73b7369f0b3b24274ea1111b022dc`

Begin with ordinary. Run cold
candidate-control-candidate from clean processes on the same prompt, exact
checkpoint receipt, kernel, commit, 32-token request, batch size one, and
concurrency one.

The candidate is the authenticated three-layer native MiMo MTP q4 proposer
against live target state. The control is the existing same-model q8 proposer.
The verifier remains the only commit authority. Record every transaction's
proposal, posterior, committed tokens `A`, route union `U`, rollback, proposal
and verification wall, along with complete prefill, logical bytes, process
reads, residency, and safety state.

Preserve any repeatable positive complete-request TPS gain. A category passes
only when both candidates emit byte-identical output to the control and the
candidate median complete accepted TPS is higher. A negative category does not
erase positive categories; it remains a slice-specific control and may motivate
a causal scheduler only under a new frozen contract. Continue to the remaining
holdout categories only after ordinary completes safely. General text-default
promotion requires all four holdouts; native modalities and the full TARGET.md
fidelity/latency/throughput gates remain separate.

## Result

Unexecuted. No prompt has supplied proposals, routes, acceptance, or output at
the commit that introduces this record.
