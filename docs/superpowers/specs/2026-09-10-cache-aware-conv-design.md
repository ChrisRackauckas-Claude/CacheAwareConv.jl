# CacheAwareConv.jl design

Date: 2026-09-10. Status: approved in discussion, implementation pending.

## Goal

A pure-Julia, CPU, direct (non-FFT, non-im2col) N-dimensional convolution whose
blocking is derived from the machine's L1/L2/L3 cache sizes (CPUSummary.jl) and
whose inner kernel is a hand-written SIMD.jl register-tiled microkernel. It must

- match the semantics of the common deep-learning `conv` (NNlib): spatial
  1D/2D/3D, layout `(W, [H, [D]], C_in, N)`, weights `(k..., C_in ÷ groups, C_out)`,
  `stride`, `pad` (Int, N-tuple, or 2N-tuple lo/hi), `dilation`, `groups`,
  and `flipped` (`false` = true convolution, `true` = cross-correlation);
- not depend on NNlib anywhere (tests may use it as an oracle);
- run at near-FMA-peak for compute-bound shapes and at full DRAM bandwidth for
  memory-bound shapes (few channels, small kernels, GB-scale inputs);
- expose a pre-allocation phase (`ConvPlan`) that owns every scratch buffer so
  `conv!` allocates nothing;
- specialise on element type: Float32/Float64 (native SIMD), Float16 (compute in
  Float32), ComplexF32/ComplexF64 (3-multiply split-plane kernel), everything
  else (generic blocked scalar path, e.g. Dual numbers, BigFloat);
- support 32-bit Julia (`Int` indexing, vector width from HostCPUFeatures);
- provide gradients (`∇conv_data!`, `∇conv_filter!`) implemented with the same
  cache-aware machinery, wired into EnzymeCore and ChainRulesCore extensions;
- provide Lux.jl and Flux.jl layer extensions that use this package's own API;
- ship as a SciML-standard repository (SciMLTesting, Runic, reusable workflows,
  Documenter docs, Aqua/JET/ExplicitImports QA).

Out of scope: GPU arrays, FFT/Winograd algorithms, LoopVectorization.

## Public API

```julia
struct ConvPlan{T, N, ...}      # T = eltype of x/w/y, N = ndims(x) = spatial + 2
ConvPlan(::Type{T}, xsize::Dims{N}, wsize::Dims{N};
         stride = 1, pad = 0, dilation = 1, groups = 1, flipped = false,
         nthreads = Threads.nthreads())
plan_conv(x::AbstractArray, w::AbstractArray; kwargs...)   # sizes/eltype from arrays

output_size(p::ConvPlan)          # full size of y, (spatial_out..., C_out, N)
input_size(p), kernel_size(p), stride(p), padding(p), dilation(p), groups(p), flipped(p)

conv!(y, x, w, p::ConvPlan; bias = nothing, σ = identity)   # y = σ.(conv(x, w) .+ bias)
conv(x, w, p::ConvPlan; bias, σ)                            # allocating
conv(x, w; stride, pad, dilation, groups, flipped, bias, σ) # plans internally

∇conv_data!(x̄, ȳ, w, p)     # x̄ = ∂/∂x, same shape as x
∇conv_filter!(w̄, x, ȳ, p)   # w̄ = ∂/∂w, same shape as w
∇conv_data(ȳ, w, p), ∇conv_filter(x, ȳ, p)

reference_conv!(y, x, w, p)  # naive triple loop, exported for testing/oracle use
```

`conv!` accepts `alpha`/`beta`-free semantics: it overwrites `y`. Gradient
functions also overwrite. Accumulation for AD is done by the rules with a
temporary owned by the plan (`p.grad_scratch`), never by the user.

All arrays must be `StridedArray`s with unit stride in the first dimension
(`Array`, contiguous `SubArray`, `reshape`). Others get a clear `ArgumentError`.

Padding is zero padding. Output size per spatial dim `i`:
`(I_i + lo_i + hi_i - d_i*(k_i-1) - 1) ÷ s_i + 1`. Index convention
(1-based): `x_idx = (o-1)*s - lo + 1 + (k-1)*d`, with `k` reversed when
`flipped == false`. Groups: input channels are split into `groups` contiguous
blocks; output channels likewise; group `g` uses weight slice `w[..., :, g-block]`.

## Algorithm

### Data layout inside the plan

For each work item (one batch index `n`, one block of output rows), the input
tile is **packed** into a per-thread buffer `Xp` that is

- zero-padded on all spatial borders (so the microkernel never bounds-checks),
- **stride-phase deinterleaved**: for stride `s` along W, the padded row is
  split into `s` phases so that for a fixed `(kw mod s)` the inputs needed by
  consecutive output columns are contiguous. Along H/D the phase split is a
  row-selection, not a data reorder. Result: every load in the kernel is a
  contiguous unaligned vector load, for any stride;
- dilation is handled by the kernel's address arithmetic (offset `kw*d` within a
  phase), no data reorder;
- for Float16: packed as Float32; for Complex: packed as three planes
  `re`, `im`, `re+im` (Gauss 3-multiply trick).

Weights are packed once per `conv!` call into `Wp` with layout
`(NR, k_w, k_h, k_d, Kc, co_block, ci_block)` so the kernel reads `NR`
contiguous scalars per `(k, ci)` step and broadcasts each. Flipping is applied
during packing. Complex weights pack as `(wr, wi, wr+wi)` triples.

### Microkernel (SIMD.jl, `@generated` per `(T, MR, NR, V)`)

Accumulator tile: `MR` vectors of width `V` along output W × `NR` output
channels. For each `(ci, kd, kh, kw)`: `MR` vector loads of `Xp`, `NR` scalar
broadcasts of `Wp`, `MR*NR` `muladd`s. Chosen from `register_count()`:

| registers | real `(MR, NR)` | complex `(MR, NR)` |
|-----------|-----------------|--------------------|
| 32 (AVX-512) | (4, 6) | (2, 4) |
| 16 (AVX2, NEON) | (2, 4) | (1, 4) |
| 8 (SSE on i686) | (1, 4) | (1, 2) |

Spike results on Zen 4 (AVX-512): real Float32 kernel 97 GFLOPS single core
(~88% of peak), no spills; complex 3-multiply kernel 16 GCMAC/s vs 12.5 for the
4-FMA form, so 3-multiply is used.

Tail handling: the last partial vector along W is a masked load/store
(`Vec{V,Bool}` mask); the last partial `NR` block of output channels uses a
smaller `NR` instantiation (1..NR-1) rather than masking; rows never need masks.

The epilogue (bias add, activation, Float16 down-convert, complex recombine
`re = a1 - a2`, `im = a3 - a1 - a2`) is applied at store time after the last
input-channel block, while the tile is in L1.

### Blocking (the cache-aware part)

Let `sz` be the packed element size (4 for Float16/Float32, 8 for Float64;
complex counts 3 planes). Let `K = prod(k)`.

- **L1 ← weight panel.** `Kc` = number of input channels per block, the largest
  value with `K * Kc * NR * sz ≤ L1/2`, clamped to `[1, C_in/groups]`. The
  panel is reused across every output column and row of the tile.
- **L2 ← packed input tile.** `Hb` = number of output rows per work item, the
  largest with `Kc * Wp_padded * ((Hb-1)*s_h + (k_h-1)*d_h + 1) * sz ≤ L2/2`
  (3D: rows × depth slabs analogously), clamped to `[1, H_out]`. The tile is
  reused across every output-channel block.
- **L3 (per core share) ← packed weights for the co_block loop.** `Nc` = output
  channels per block with `K * Kc * Nc * sz ≤ L3_per_core/2`; for the sizes
  seen in practice this is all of `C_out`, and the loop degenerates.
- When `Kc < C_in/groups`, output tiles are accumulated through memory: the
  first block stores, later blocks load-add-store. Bias/activation epilogue then
  runs only on the last block.

Loop nest per work item `(n, ho_block)`:

```
for ci_block                      # Kc channels
  pack Xp[ci_block, rows of ho_block] (per-thread buffer)
  for co_block (Nc), for co_tile (NR) in co_block
     for ho in ho_block, for wo_tile (MR*V) in W_out
        microkernel(Xp, Wp[co_tile, ci_block], y[wo_tile, ho, co_tile, n])
```

Threading: work items `(n, ho_block)` are distributed with `Threads.@spawn`
over `min(nthreads, nitems)` tasks in contiguous chunks; each task owns one
packing buffer from `plan.buffers[tid]`. Single-threaded when `nthreads == 1`.
Deterministic: identical results regardless of thread count.

Memory-bound regime: each input element is read from DRAM once per
`ci_block` (once in the common `Kc == C_in` case), packed inside L2, and each
output element written once. A GB-scale benchmark (`benchmark/streaming.jl`)
reports achieved GB/s against a `copyto!` baseline.

### Gradients

- `∇conv_data!`: transposed conv. Weights are flipped and channel-transposed
  during packing (no user-visible copy); `ȳ` is packed with zero-stuffing for
  `stride > 1` and transposed padding `(k-1)*d - lo`. Reuses the forward
  microkernel and blocking. Zero-stuffing wastes `prod(s)` flops; a later
  phase-decomposition optimisation is noted in the docs as future work.
- `∇conv_filter!`: a dedicated reduction microkernel. For a fixed spatial
  kernel offset `k`, `w̄[k, ci, co] = Σ_{wo, ho, n} Xp[wo + k, ci] * ȳ[wo, co]`.
  Register tile: `MRc` input channels × `NRc` output channels, vectors along
  `wo`, one horizontal `sum` per accumulator after the full `(wo, ho, n)`
  reduction. Loads: `MRc + NRc` vectors per `MRc*NRc` FMAs, no broadcasts. Same
  packed `Xp` (input tile in L2), `ȳ` rows streamed. Accumulation across work
  items (rows, batch) happens in a per-thread `w̄` partial that is reduced at
  the end (deterministic order).

### Element-type specialisation

| eltype | pack type | kernel |
|--------|-----------|--------|
| Float32, Float64 | same | real SIMD kernel |
| Float16 | Float32 | real SIMD kernel, convert on store |
| ComplexF32/ComplexF64 | 3 real planes | 3-multiply kernel |
| other `Number` | same | generic blocked path: same loop nest and packing, scalar `muladd` inner loop, `@simd` where legal |

`T` for `conv!(y, x, w)` is `promote_type(eltype(x), eltype(w))` and must equal
`eltype(y)`; mixed-precision inputs raise an `ArgumentError` pointing at
`convert`.

### 32-bit

All sizes/offsets are `Int`. Buffer lengths are checked against `typemax(Int)`
at plan time. `V = pick_vector_width(T)` and `register_count()` come from
HostCPUFeatures. CPUSummary's generic fallback (32 KiB L1, 64 KiB L2) is used
on i686; sizes are read as `Int(cache_size(Val(i)))` with a 0 guard (unknown
level → fall back to the next-smaller known level × 8). CI runs an
`arch = "x86"` lane; locally tested with juliaup `release~x86`.

## Extensions

- **CacheAwareConvEnzymeCoreExt** (weakdep EnzymeCore): `EnzymeRules.forward`,
  `augmented_primal`, `reverse` for `conv!`, `∇conv_data!`, `∇conv_filter!`
  following NNlib's pattern: shadow of `y` computed with the same kernels,
  gradients accumulated into `x.dval`/`w.dval` via `plan.grad_scratch` then
  `.+=`, `dy .= 0` after use, width > 1 handled, `overwritten(config)` used
  to decide whether `x`/`w` are copied. The plan is annotated `Const` (its
  buffers are scratch; rules never read them across passes).
- **CacheAwareConvChainRulesCoreExt** (weakdep ChainRulesCore): `rrule` for the
  out-of-place `conv(x, w, p; bias, σ)` when `σ === identity` (bias handled
  in the rule); `@non_differentiable` for `ConvPlan`/`plan_conv`. For
  `σ ≠ identity` the rule computes `z = conv .+ bias` and lets AD handle `σ.`
  via a second `rrule_via_ad` call. Gives Zygote support for the Flux layer.
- **CacheAwareConvLuxExt** (weakdep Lux): `CacheAwareConv.LuxConv <: AbstractLuxLayer`
  with Lux's `Conv` constructor signature (`k, in=>out, σ; stride, pad (incl. SamePad), dilation, groups, use_bias, init_weight, init_bias, cross_correlation`).
  Parameters `(; weight, bias)`, state holds nothing (plans are cached in a
  `Dict{Tuple, ConvPlan}` on the layer keyed by input size/eltype, guarded by a
  lock). Forward: `conv!` with the fused bias/σ epilogue.
- **CacheAwareConvFluxExt** (weakdep Flux): `CacheAwareConv.FluxConv` built with
  `Flux.@layer`, same constructor as Flux's `Conv` (`bias = true/false/array`,
  `SamePad`), `trainable = (weight, bias)`, forward via `conv`. Show methods
  mirror Flux's.

Layers are named `LuxConv`/`FluxConv` to avoid shadowing `Lux.Conv`/`Flux.Conv`
when both are loaded.

## Repository

```
Project.toml            deps: CPUSummary, HostCPUFeatures, SIMD, Static, SciMLPublic, PrecompileTools
                        weakdeps: EnzymeCore, ChainRulesCore, Lux, Flux
src/CacheAwareConv.jl   module, exports, includes
src/cache_params.jl     cache sizes, vector width, register tiles per T
src/plan.jl             ConvPlan, size arithmetic, buffer allocation
src/pack.jl             input/weight packing (real, f16, complex, zero-stuffed)
src/kernel.jl           @generated forward microkernels (real, complex), masked tails
src/kernel_grad.jl      ∇filter reduction microkernel
src/conv.jl             blocked loop nest, threading, conv!/conv
src/grad.jl             ∇conv_data!, ∇conv_filter!
src/generic.jl          scalar fallback for arbitrary Number
src/reference.jl        reference_conv!, reference gradients
src/precompile.jl       PrecompileTools workload (Float32/Float64 2D)
ext/                    the four extensions
test/runtests.jl        using SciMLTesting; run_tests()
test/test_groups.toml   Core (lts,1,pre × 3 OS; 32-bit x86 lane), QA, Enzyme, Layers
test/*.jl               plan, correctness (vs reference & NNlib, all combos), types, threading, gradients, chainrules
test/enzyme/            forward + reverse Enzyme checks vs reference gradients
test/layers/            Lux + Flux layer tests vs Lux.Conv/Flux.Conv
test/qa/                run_qa (Aqua, JET, ExplicitImports, API docs)
docs/                   Documenter: index, API, design (blocking), benchmarks, AD, layers
benchmark/              microkernel GFLOPS, conv vs NNlib, streaming GB/s
.github/workflows/      Tests, Downgrade, Documentation, DocPreviewCleanup, FormatCheck (Runic),
                        RunicSuggestions, SpellCheck, TagBot, DependabotAutoMerge; dependabot.yml
```

Testing strategy: every correctness test compares against `reference_conv!`
with `rtol` scaled to eltype, over a grid of `(spatial dims 1/2/3) × stride
{1,2,3} × pad {0, sym, asym} × dilation {1,2} × groups {1, 2, C} × flipped ×
sizes hitting every tail case (W < V, W = MR*V ± 1, C_out not multiple of NR,
C_in not multiple of Kc)`. NNlib is an additional oracle in the Core group.
Gradients are checked against the reference gradients and against finite
differences. Thread count 1 vs many must give bitwise-identical output.

GitHub-side setup (repo creation, secrets, transfer) is handled by the user.
