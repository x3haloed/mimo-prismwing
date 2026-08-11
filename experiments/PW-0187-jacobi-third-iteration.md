# PW-0187 — Target-generated Jacobi third iteration

- Status: completed
- Disposition: promoted to direct-checkpoint wide Metal integration; not an endpoint
- Date: 2026-08-10
- Model/reference: MiMo revision
  `63651580ca774f8504f676040460aed3e1244ac1`; PW-0186 manifest
  `f773fa2859f08b57f851944aa8ba0ef9b502040058580a9344be4ce3ee1e1d1c`
- Execution mode: target-faithful greedy L2 proposal with exact verification
- Hardware/runtime: source CPU oracle on the existing Apple M1
- Related records: PW-0102, PW-0112, PW-0181, PW-0186

## Contract

Shift PW-0186's authenticated posterior once to freeze the third Jacobi block
as `[264,13,15,13,15,15,15,15]`. Execute it through the unchanged complete
PW-0102/PW-0186 source target verifier. Authenticate every prior authority,
reproduce prefill hidden states and the first-token distribution, preserve
source weights, routes, arithmetic, greedy correction, and Gate 8, and record
posterior, `A`, `U`, bytes, and complete wall.

Use the same favorable physical expression:
`(max(1.090015*U,0.714682*U)+0.131220)/A`. A value below one second and
`A/U>1.090015` promotes a production-shaped wide Metal verifier. Failure kills
this Jacobi chain for one TPS. If the accepted prefix reaches the seven-token
draft maximum, treat the window as converged; otherwise do not run a fourth
source walk unless the third result improves `A/U` over PW-0186.

This is not endpoint TPS. Report zero endpoint tokens and do not change a
measured throughput constant until the complete accelerated path runs.

## Result

The authoritative manifest hashes to
`a1066fafa979b923f9c2f5d259ff85b2f3d5aa2e77400e8b7075a48f3fa67950`.
Iteration three produces posterior `[13,15,13,15,481,13,15,15]`. Exact greedy
verification accepts the anchor plus four draft tokens: `A=5`. Mean normalized
expert union is `U=2.050532`, so `A/U=2.438392` improves materially over
PW-0186 and clears the frozen `1.090015` continuation gate.

Under the deliberately favorable bound, optimistic time is `0.473266` seconds
per accepted token, or `2.112976` TPS before omitted dense work, LM head,
correction, rollback, and a real wide Metal transaction. The source CPU oracle
takes 254.565 seconds post-prefill and 1,131.715 seconds complete; neither is a
performance candidate.

Promote the authenticated block to a production-shaped wide Metal verifier.
The immediate physical prerequisite is copy-free binding of page-rounded
regions from the original checkpoint shards; do not construct the roughly
303 GB repacked full-bank artifact. Gate 8 passes with 60% minimum free memory,
3,942,252,544-byte maximum peak RSS, zero swap growth or throttling, and stable
protected services. This experiment reports no endpoint TPS and changes no
measured throughput constant.
