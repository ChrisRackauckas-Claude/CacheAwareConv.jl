# CacheAwareConv.jl: cache-blocked direct convolution for the CPU

CacheAwareConv.jl is a pure-Julia implementation of the N-dimensional
convolution used in deep learning (`(spatial..., channels, batch)` layout with
`stride`, `pad`, `dilation`, `groups`, and both true-convolution and
cross-correlation kernels). It is a *direct* convolution: no FFT, no im2col.
Performance comes from two things:

- **Cache-aware blocking.** The sizes of the L1, L2 and L3 caches (queried
  through [CPUSummary.jl](https://github.com/JuliaSIMD/CPUSummary.jl)) decide
  how many input channels form a weight panel (kept in L1), how large an input
  tile is packed at a time (kept in L2), and how many output channels are
  swept per packed tile (weights kept in L3). See [Design](@ref design).
- **A hand-written SIMD microkernel.** Register tiles of `MR` vectors along
  the output width times `NR` output channels are accumulated with fused
  multiply-adds written with [SIMD.jl](https://github.com/eschnett/SIMD.jl);
  the tile shape is chosen from the machine's vector width and register count.

Everything the computation needs (packed buffers, packed weights, gradient
scratch) is allocated once in a [`ConvPlan`](@ref); `conv!` itself allocates
nothing. Work is split across Julia tasks when `Threads.nthreads() > 1`, with
results independent of the thread count.

## Installation

```julia
using Pkg
Pkg.add("CacheAwareConv")
```

## Quick start

```@example quick
using CacheAwareConv

x = randn(Float32, 32, 32, 3, 8)      # width, height, channels, batch
w = randn(Float32, 3, 3, 3, 16)       # kernel, kernel, in channels, out channels

plan = plan_conv(x, w; pad = 1)      # blocking + buffers for these sizes
y = conv(x, w, plan)                 # or conv!(y, x, w, plan) to reuse `y`
size(y)
```

The semantics are those of NNlib's `conv`: `flipped = false` (the default)
performs a true convolution with the kernel reversed, `flipped = true` a
cross-correlation.

## Features

- 1D, 2D and 3D spatial convolutions with arbitrary stride, asymmetric or
  negative padding, dilation and channel groups.
- Element-type specialisation: `Float32` and `Float64` run the SIMD kernel
  directly; `Float16` is packed to and accumulated in `Float32`; `ComplexF32`
  and `ComplexF64` use a three-real-multiplication split-plane kernel; any other
  `Number` (dual numbers, `BigFloat`, ...) uses the same blocking with a
  scalar kernel.
- Gradients: [`∇conv_data!`](@ref) reuses the forward machinery as a
  transposed convolution; [`∇conv_filter!`](@ref) has its own reduction
  microkernel over the same packed tiles.
- Automatic differentiation through package extensions for
  [EnzymeCore](https://github.com/EnzymeAD/Enzyme.jl) (forward and reverse
  rules) and [ChainRulesCore](https://github.com/JuliaDiff/ChainRulesCore.jl)
  (so Zygote works).
- Drop-in layers for [Lux.jl](https://github.com/LuxDL/Lux.jl) and
  [Flux.jl](https://github.com/FluxML/Flux.jl) through package extensions.
- Optional package extensions: `kernel = :lv` runs a
  [LoopVectorization.jl](https://github.com/JuliaSIMD/LoopVectorization.jl)
  `@turbo` microkernel, and `executor = :polyester` schedules work with
  [Polyester.jl](https://github.com/JuliaSIMD/Polyester.jl) `@batch`
  (see [`ConvPlan`](@ref)).
- Works on 32-bit Julia.

## Contributing

- Please refer to the
  [SciML ColPrac: Contributor's Guide on Collaborative Practices for Community Packages](https://github.com/SciML/ColPrac/blob/master/README.md)
  for guidance on PRs, issues, and other matters relating to contributing to SciML.
- See the [SciML Style Guide](https://github.com/SciML/SciMLStyle) for common coding practices and other style decisions.
- There are a few community forums:
    - The #diffeq-bridged and #sciml-bridged channels in the
      [Julia Slack](https://julialang.org/slack/)
    - The #diffeq-bridged and #sciml-bridged channels in the
      [Julia Zulip](https://julialang.zulipchat.com/#narrow/stream/279055-sciml-bridged)
    - On the [Julia Discourse forums](https://discourse.julialang.org)
    - See also [SciML Community page](https://sciml.ai/community/)

## Reproducibility

```@raw html
<details><summary>The documentation of this SciML package was built using these direct dependencies,</summary>
```

```@example
using Pkg # hide
Pkg.status() # hide
```

```@raw html
</details>
```

```@raw html
<details><summary>and using this machine and Julia version.</summary>
```

```@example
using InteractiveUtils # hide
versioninfo() # hide
```

```@raw html
</details>
```

```@raw html
<details><summary>A more complete overview of all dependencies and their versions is also provided.</summary>
```

```@example
using Pkg # hide
Pkg.status(; mode = PKGMODE_MANIFEST) # hide
```

```@raw html
</details>
```

```@eval
using TOML
using Markdown
version = TOML.parse(read("../../Project.toml", String))["version"]
name = TOML.parse(read("../../Project.toml", String))["name"]
link_manifest = "https://github.com/SciML/" * name * ".jl/tree/gh-pages/v" * version *
                "/assets/Manifest.toml"
link_project = "https://github.com/SciML/" * name * ".jl/tree/gh-pages/v" * version *
               "/assets/Project.toml"
Markdown.parse("""You can also download the
[manifest]($link_manifest)
file and the
[project]($link_project)
file.
""")
```
