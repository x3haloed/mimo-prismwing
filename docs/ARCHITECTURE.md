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
- `docs/EXPERT_CONTAINER.md` defines the lossless runtime tensor container;
  generated containers remain external and are identified by experiment hash.
- `MappedSafetensors` is the sole native authority for source-checkpoint
  metadata, bounds, and immutable tensor byte views. Kernels and schedulers do
  not reparse headers or accept caller-inferred source shapes.
- Rust runtime code owns accepted local inference state transitions.
- A pinned C++/MLX kernel bridge may execute validated array operations but
  cannot weaken Rust-side artifact or state-transition checks.
- The readable Rust mapped-FP8 GEMV is the source-projection correctness
  reference. Accelerated Metal or MLX paths must consume the same validated
  metadata contract and pass against this reference before promotion.
- The promoted Metal FP8 projection is dispatched by Rust only after that
  shared mapped-tensor validator establishes dtype, dimensions, scale grid,
  finite encodings, and input shape. MSL owns parallel arithmetic, not source
  layout inference or artifact authority.
- The native source-FP8 expert path composes three such validated projections
  with an f64-fixtured Metal SwiGLU under one Rust-owned command sequence.
  Expert batching, routing, and weighted reduction remain separate authorities
  that must be made real before this primitive becomes a routed layer.
- The promoted batch-eight expert schedule assigns one output row per Metal
  threadgroup and applies each decoded source-FP8 weight to eight positions.
  A flattened position-row schedule is retained only as a correctness control;
  it is not an executable default because it rereads weights per position.
- The fixture-scheduled heterogeneous MoE path owns explicit per-tensor source
  authorities, gathers uneven expert batches, executes padded shared-weight
  kernels, and performs route-weighted scatter-add in Metal. Frozen route IDs
  and weights remain its independent correctness oracle.
- The promoted native layer-43 router validates the exact F32 projection and
  correction bias, shares projection weights across eight positions in Metal,
  and owns noaux-tc selection and normalization in Rust. It emits a canonical,
  hash-bound route artifact matching the frozen oracle.
- The promoted dynamic MoE path dispatches that router inside every measured
  request, derives gather/weight/position/scatter buffers from its decisions,
  and executes the heterogeneous source-FP8 expert union. Frozen routes are
  parity oracle only. This closes the layer-43 routed-MLP causal path for one
  exact input fixture, not a complete decoder layer or representative decode.
- Frozen raw evidence owns measurements; Markdown reports interpret it but do
  not override it.

Every unknown schema, revision, tensor, shape, layout, provider, or runtime mode
fails closed. Compatibility paths require an explicit external reason and
removal condition.
