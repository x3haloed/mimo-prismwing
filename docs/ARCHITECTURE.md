# Initial architecture contract

## Shared contract

Prismwing carries native MiMo-V2.5 inputs through one locally authoritative
runtime to accepted output tokens. It either passes every target gate on the
declared system or preserves enough evidence to show decisively that no tested
admissible embodiment can do so within the fixed constraints.

The initial evidence horizon is the 16 GiB M1 Mac mini, the pinned public
checkpoint and source semantics, the pinned OpenRouter provider, and generated
or redistributable fixtures. The deepest currently authorized implementation
boundary is native CPU, Metal, storage, and network execution on locally owned
consumer hardware. Modified weights or topology remain a separate named mode.

## Candidate shapes

1. **Swift runtime plus Python research plane.** Rejected as the initial shape:
   Swift has excellent Metal integration, but its model/container ecosystem and
   portable low-level tooling are narrower for this project.
2. **C++ runtime plus Python research plane.** Viable, but manual ownership and
   error propagation add risk at adversarial model-artifact boundaries.
3. **Rust runtime plus Metal kernels and Python research plane.** Selected.
   Rust owns validated artifacts, scheduling, state, I/O, and runtime control;
   MSL owns GPU kernels; Python owns reference acquisition, fixture generation,
   analysis, and report recomputation.

PW-0012 adds one bounded implementation option without changing runtime
authority: MLX's pinned C++ API can supply optimized native Metal primitives
behind the Rust-controlled runtime. A compiled C++ smoke test now executes
affine-INT4 quantized matmul directly through `libmlx`, with no Python on that
path. MLX remains a replaceable kernel substrate; it does not own model
semantics, artifact validation, scheduling, or accepted-token state.

This creates two language boundaries but only one runtime authority. Python
does not decide production inference semantics or execute accepted tokens.

## Risk-bearing slices

The construction order follows the least-proven boundaries:

1. Pinned hosted request to immutable, offline-verifiable token evidence.
2. Pinned checkpoint to complete tensor census and byte budget.
3. Seeded tensor through readable reference semantics to checked output.
4. Sampled real tensor through the same semantic path.
5. Accelerated kernel parity, then complete text decode.
6. Native image, audio, video, and mixed preprocessing and projection.
7. Full-path performance experiments and only then promoted optimization.

## Surviving authorities

- `TARGET.md` is the human-readable acceptance authority.
- `spec/acceptance.yaml` is its machine-readable mirror.
- `spec/model.lock.json` identifies every upstream checkpoint file.
- Rust runtime code owns accepted local inference state transitions.
- A pinned C++/MLX kernel bridge may execute validated array operations but
  cannot weaken Rust-side artifact or state-transition checks.
- Frozen raw evidence owns measurements; Markdown reports interpret it but do
  not override it.

Every unknown schema, revision, tensor, shape, layout, provider, or runtime mode
fails closed. Compatibility paths require an explicit external reason and
removal condition.
