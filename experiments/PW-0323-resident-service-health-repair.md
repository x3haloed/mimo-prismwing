# PW-0323 — Resident-service health semantics repair

- Status: complete
- Disposition: correctness-repair
- Date: 2026-08-26
- Owner: Codex
- Trigger: two PW-0322 Gate 8 stops after healthy supervised `nxnode` PID replacement

## Prediction error

The Rust Gate 8 monitor requires every baseline PID of a protected named
service to survive. The normative Python monitor requires the baseline service
name to remain present. NoMachine supervises and replaces `nxnode`; two q64
attempts stopped while a new `nxnode` PID was already healthy. PID continuity
therefore measures worker lifetime, not resident-service health.

## Contract

Align Rust with the existing normative Python rule. A service present at
baseline remains protected by name and must have at least one current PID at
every checkpoint. A replacement PID passes and is recorded. An empty/missing
current PID set fails closed. Services absent at baseline remain outside the
run-specific requirement. Do not alter memory, swap, throttling, release, or
service-name policy.

Unit fixtures must cover unchanged PID, replaced PID, actual disappearance,
and baseline absence. Resume PW-0322 only after the repaired monitor passes the
complete Rust suite from a clean pushed commit. Preserve both stopped attempts.

## Result and decision

The Rust rule now matches the established Python normative monitor: every
service present at baseline must remain present by name, while its recorded PID
set may be replaced. The fixture accepts unchanged and replacement PIDs,
rejects an empty `nxnode` set, and ignores services absent at baseline. All 114
Rust library tests pass. Memory, footprint, release, swap, throttling, service
names, and snapshot recording are unchanged.

Promote this as a Gate 8 correctness repair and resume PW-0322 from a fresh
process. The two stopped q64 attempts remain under the evidence directory; no
model transaction completed and no performance evidence was produced.
