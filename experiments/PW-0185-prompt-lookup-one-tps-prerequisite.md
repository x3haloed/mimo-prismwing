# PW-0185 — Prompt-lookup one-TPS prerequisite

- Status: completed
- Disposition: rejected on the authenticated text trace
- Date: 2026-08-10
- Authority: PW-0112 route manifest
  `584d3a8b1b09b12d4f83908be1fa5471b9fd66373500cc56332213928cd0bc3e`
- Execution mode: exact greedy L2 proposal trace analysis
- Related records: PW-0104, PW-0110, PW-0112, PW-0181

## Contract

Test lossless prompt-lookup speculation on PW-0112's authenticated 87-token
prompt and 137-token target suffix. At each pass, search the existing history
for the most recent earlier occurrence of the longest suffix between frozen
minimum lengths `{1,2,3,4,6,8}` and maximum length 10. Propose up to
`q={2,4,8,16}` following tokens, accept the consecutive target-matching draft
prefix, then grant the exact target bonus token. Advance only by committed
tokens. Add fixtures for most-recent/longest matching and target correction.

PW-0181's ideal miss-acquisition term is 1.090015 seconds/token. Even granting
impossible `U=1` and making attention, dense work, lookup, correction, and all
other endpoint work free, prompt lookup needs mean accepted `A>1.090015` to
leave any path above one TPS. Reject the family on this trace if no frozen
configuration clears that necessary bound. This is a trace prerequisite, not
an endpoint measurement or a general rejection of Jacobi/lookahead decoding.

## Result

The authoritative report hashes to
`be1709777b2b2402c0419c917f010389fcffa0dc1fa9169ac6a72ec26a20a2d6`.
The strongest configuration is the unsafe minimum-one-token lookup with
`q=4`: it commits 137 tokens in 126 passes, mean `A=1.087302`, with drafts on
37.30% of passes and maximum `A=4`. Even granting impossible `U=1`, its miss
term is 1.002496 seconds per accepted token before attention or any other work.

Minimum n-gram two falls to `A=1.037879`; minimum four or higher accepts no
draft token. Reject prompt lookup as the one-TPS mechanism on this trace. This
does not reject Jacobi/lookahead trajectories, whose candidates are generated
by target-model iteration rather than repetition in the prompt/history. Zero
executed tokens and no endpoint TPS or throughput constant follow.
