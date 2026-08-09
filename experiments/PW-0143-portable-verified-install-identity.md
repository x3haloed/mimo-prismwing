# PW-0143 — Portable verified-install runtime identity

- Status: planned
- Disposition: unexecuted correctness repair
- Date: 2026-08-09
- Owner: Codex with project owner authorization
- Checkpoint/reference hashes: MiMo revision
  `63651580ca774f8504f676040460aed3e1244ac1`; checkpoint verification
  `9ddc8a99755f04ae2ea3c2484f6dd022d3f3a681b5a72c915ee4de833dbb0d03`
- Hardware/runtime: Apple M1 shared 16 GiB; APFS Data volume remounted with a
  different process-visible device number
- Exactness: L1 installation-authority repair; model bytes and hashes unchanged
- Related records: PW-0002, PW-0049, PW-0142

## Question and observed failure

PW-0142 failed before weight access because every authenticated checkpoint file
retained its path, byte count, inode, nanosecond mtime, and receipt SHA-256, but
macOS changed the APFS Data volume's `st_dev` from `16777233` to `16777231`
across a remount or reboot. A device number is a mount-session locator, not a
durable installation identity. Treating it as immutable makes a valid receipt
non-portable across ordinary host restarts.

## Frozen repair

Create one runtime identity authority shared by checkpoint consumers. A file
passes only when its receipt record is verified and contains a valid SHA-256,
and its current path is a regular file with the exact recorded byte count,
inode, and nanosecond mtime. Preserve receipt and current device numbers as
diagnostics, but do not make equality a gate. Callers must continue to
authenticate the complete receipt by its pinned SHA-256 before trusting these
observations.

This does not regenerate the receipt, rescan tensor payloads, or accept changed
bytes, sizes, inode identities, mtimes, revisions, layouts, or receipt hashes.
It removes only the transient mount-session field from the durable identity
predicate.

## Gate

Promote as a correctness repair only if:

1. a fixture with identical size/inode/mtime and a changed recorded device
   passes while reporting the drift;
2. changed size, inode, mtime, status, missing SHA-256, and non-file paths fail;
3. the real checkpoint index passes against the pinned receipt and reports only
   the known device drift;
4. checkpoint, DFlash proposal, base-layer, and PW-0142 focused tests pass; and
5. PW-0142 crosses its checkpoint preflight without altering receipt or model
   hashes.

This record makes no token, performance, fidelity, or endpoint claim.

