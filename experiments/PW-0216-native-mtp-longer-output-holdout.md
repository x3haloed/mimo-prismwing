# PW-0216 — Native MTP untouched longer-output holdout

- Status: in progress; ordinary and code holdouts complete
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
It strengthens, but does not replace, PW-0215's category evidence.

## Code holdout result

The already-frozen code prompt was then run from clean commit
`16974043d1c45bbb591d114543cd69be18a48d73`. Cold
candidate-control-candidate processes emit identical token IDs and decoded
bytes. The 32-token output begins `Here is a design for a safe, zero-copy Rust
API` and ends with ``We will use a `struct``.

Both candidates execute thirteen q4 transactions and repeat the committed-token
sequence `[3,3,2,3,1,3,3,3,2,3,2,2,1]`. This exercises full acceptance,
partial correction, and an immediate first-token rejection. The q8 control
executes five transactions and commits `[6,7,7,7,4]` proposal rows plus exact
corrections where required. Candidate `U` ranges from 5.329787 to 6.015957;
control `U` ranges from 4.404255 to 4.781915. The complete per-transaction
values remain in all three content-hashed reports.

| Measure | Candidate 1 | q8 control | Candidate 2 |
| --- | ---: | ---: | ---: |
| Prefill wall ms | 336,947.595 | 332,498.912 | 328,903.514 |
| Proposal wall ms | 123,099.164 | 719,677.691 | 122,347.640 |
| Verification wall ms | 342,521.662 | 184,311.785 | 333,509.458 |
| Complete wall ms | 803,642.999 | 1,237,243.109 | 785,845.432 |
| Logical source bytes | 968,898,887,296 | 1,316,315,365,376 | 968,898,887,296 |
| Process disk bytes read | 922,988,752,896 | 1,317,759,832,064 | 924,428,574,720 |
| Conservative peak resident bytes | 4,553,572,352 | 421,068,800 | 4,535,746,560 |

Candidate complete walls differ by 2.239%. Their 794,744.216-ms median is
`0.0402645` accepted TPS versus the control's `0.0258640`, a repeatable
`1.556782x` complete accepted-TPS gain. Post-prefill wall falls from
903,989.477 ms to a 460,738.962-ms candidate median, a `1.962043x` wall gain.
Candidate logical traffic falls 26.39%; median measured process reads fall
29.90%. Every run records zero swap growth, zero newly throttled pages, no
protected-service loss, and at least 53% system memory free.

Report hashes are:

- candidate 1: `bd5a5a4002b28e875ca45eee232ee461b0943391284d823b0641e7640e40fcb5`
- control: `353a14b6f7dddc7f5009876697f639999151b47539da24126237f4be253c17dc`
- candidate 2: `1cd4987260b97943aaceebdda30cf4429e3aa6a6235bb60e84a41968fd16d6f6`

Progress hashes are respectively
`6a4db785b3db4351989ec899aac0e489ef941097f39a4216c41095045fde716c`,
`7b0796452ce0cbb600c5a14e19ee79f17f7c1bbf95bf011e182b83b4f834a8cd`,
and `8423e94cd248a05f70c4802c2b92ec68feb44bb930fde3e3ba4908a28794fd4d`.

Promote the untouched code 32-token path as a second conditional lower
milestone. Its smaller gain than ordinary is preserved rather than hidden.
General text-default promotion remains withheld; continue to the frozen
multilingual holdout next, then rare-route.
