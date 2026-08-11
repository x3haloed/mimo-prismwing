# PW-0216 — Native MTP untouched longer-output holdout

- Status: in progress; ordinary holdout complete
- Disposition: conditional positive
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

The ordinary prompt was first executed only after clean commit
`42605d3e507bd3c9c2209bd3d3cf0bf209b9138e` froze the panel and contract.
Cold candidate-control-candidate processes emit identical 32-token output,
ending with ``---\n\n### **``. The exact token IDs and decoded bytes are retained
in every report.

Both candidates execute eleven q4 transactions. Their committed-token sequence
is `[3,3,3,3,3,3,3,3,2,2,3]`: nine transactions accept all three draft tokens,
and two retain two before exact correction. The q8 control executes five
transactions and commits `[7,7,7,7,3]`. This establishes sustained proposal,
verification, correction, rollback, and retained target state across the
32-token endpoint rather than extrapolating a single window.

| Measure | Candidate 1 | q8 control | Candidate 2 |
| --- | ---: | ---: | ---: |
| Prefill wall ms | 284,578.780 | 282,256.991 | 296,319.312 |
| Proposal wall ms | 102,265.099 | 729,626.466 | 124,701.010 |
| Verification wall ms | 285,781.837 | 185,894.434 | 296,042.360 |
| Complete wall ms | 673,753.110 | 1,198,644.198 | 718,212.368 |
| Logical source bytes | 829,222,868,224 | 1,257,287,161,856 | 829,222,868,224 |
| Process disk bytes read | 791,578,136,576 | 1,258,691,862,528 | 794,205,954,048 |
| Conservative peak resident bytes | 4,569,169,920 | 429,604,864 | 4,570,955,776 |

Candidate complete walls differ by 6.388%. Their 695,982.739-ms median is
`0.0459782` accepted TPS versus the control's `0.0266968`, a repeatable
`1.722233x` complete accepted-TPS gain. Post-prefill proposal plus verification
falls from 915,520.900 ms to a 404,395.153-ms candidate median, a `2.263926x`
wall gain. Candidate logical traffic falls 34.05%; process reads fall 37.01%.
Every run records zero swap growth, zero newly throttled pages, and no
protected-service loss. Minimum free memory remains at least 51%.

Report hashes are:

- candidate 1: `ca16c4fb99c4fefd53cedc73be02f1370c600a508d707bde586700f2b9cf91fc`
- control: `e64a8a625086be05c853d16f1fd60ea4644239f579fb4ebf0b6097928eefbec3`
- candidate 2: `86f23cc61d790aedb9373ac41aa9852435acd55466a736bdafb0b9edd5c5e20b`

Progress hashes are respectively
`9112b3710cf47462ef76b5f300e0c96b7d2b9e685e348a6e392bbd245c1b72b3`,
`76932ae1c439e67b2b7df9f6c66ee455976b06c5460354be118bb309b7f807c6`,
and `74b5ea11b441530bccf930174adda60605d8cce48777cc62b41746c538fb260d`.

Promote the untouched ordinary 32-token path as a conditional lower milestone.
It strengthens, but does not replace, PW-0215's category evidence. Continue to
the already-frozen code holdout next; multilingual and rare-route follow.
