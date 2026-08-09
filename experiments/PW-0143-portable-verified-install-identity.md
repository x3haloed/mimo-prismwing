# PW-0143 — Portable verified-install runtime identity

- Status: completed
- Disposition: correctness-repair
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

## Result

The centralized runtime identity predicate now requires a verified receipt
record with a valid SHA-256 plus exact current size, inode, and nanosecond
mtime. It records receipt/current device numbers and their difference without
making the transient device number a durable gate. Missing or non-regular
paths and every protected metadata mutation still fail closed.

All 39 real checkpoint files pass and report the same isolated drift:
`recorded_device=16777233`, `current_device=16777231`. The checkpoint index and
the first real PW-0142 expert shard open through the unchanged receipt, and
PW-0142 completes its full authenticated run. Twenty-three focused checkpoint,
DFlash proposal, base-layer, remote-extraction, and PW-0142 tests pass. The
complete repository gate also passes 63 Rust and 205 Python tests, Clippy, both
real Metal FP8 fixtures, the real Metal INT4 fixture, and the MLX C++ smoke. The
receipt retains SHA-256
`9ddc8a99755f04ae2ea3c2484f6dd022d3f3a681b5a72c915ee4de833dbb0d03`;
no model file or model lock changed.

This repair removes one duplicated identity implementation by routing the
affected checkpoint consumers through `validate_verified_install_file`. It
does not claim resilience against an attacker who can replace local bytes
while forging inode and nanosecond mtime; the pinned receipt hash and original
full SHA-256 installation verification remain the authority within the local
host threat envelope. No accepted tokens or performance constants change.
