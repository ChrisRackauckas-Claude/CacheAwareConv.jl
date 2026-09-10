# CacheAwareConv.jl

[![Join the chat at https://julialang.zulipchat.com #sciml-bridged](https://img.shields.io/static/v1?label=Zulip&message=chat&color=9558b2&labelColor=389826)](https://julialang.zulipchat.com/#narrow/stream/279055-sciml-bridged)
[![Global Docs](https://img.shields.io/badge/docs-SciML-blue.svg)](https://docs.sciml.ai/CacheAwareConv/stable/)

[![codecov](https://codecov.io/gh/SciML/CacheAwareConv.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/SciML/CacheAwareConv.jl)
[![Build Status](https://github.com/SciML/CacheAwareConv.jl/workflows/Tests/badge.svg)](https://github.com/SciML/CacheAwareConv.jl/actions?query=workflow%3ATests)

[![ColPrac: Contributor's Guide on Collaborative Practices for Community Packages](https://img.shields.io/badge/ColPrac-Contributor%27s%20Guide-blueviolet)](https://github.com/SciML/ColPrac)
[![SciML Code Style](https://img.shields.io/static/v1?label=code%20style&message=SciML&color=9558b2&labelColor=389826)](https://github.com/SciML/SciMLStyle)

Fast, pure-Julia, cache-blocked **direct convolution** for the CPU, with a
hand-written SIMD microkernel. Supports the deep-learning convolution
semantics (N-d spatial, `stride`, `pad`, `dilation`, `groups`, convolution or
cross-correlation), `Float16`/`Float32`/`Float64`/complex element types, exact
gradients, Enzyme and ChainRules (Zygote) rules, and Lux/Flux layers.

```julia
using CacheAwareConv

x = randn(Float32, 224, 224, 3, 8)
w = randn(Float32, 7, 7, 3, 64)
plan = plan_conv(x, w; pad = 3, stride = 2)   # cache blocking + buffers, once
y = conv(x, w, plan)                          # or conv!(y, x, w, plan): zero allocations

ȳ = randn(Float32, size(y))
x̄ = ∇conv_data(ȳ, w, plan)
w̄ = ∇conv_filter(x, ȳ, plan)
```

How it works, in one paragraph: the input tile for each work item is packed
into a zero-padded, stride-deinterleaved buffer sized to fit half of L2; the
weights for a block of `Kc` input channels are packed to fit half of L1; the
microkernel accumulates `MR` vectors × `NR` output channels in registers with
fused multiply-adds; the block sizes are derived from
[CPUSummary.jl](https://github.com/JuliaSIMD/CPUSummary.jl)'s cache sizes and
[HostCPUFeatures.jl](https://github.com/JuliaSIMD/HostCPUFeatures.jl)'s
register count. See the
[documentation](https://docs.sciml.ai/CacheAwareConv/stable/) for the design,
benchmarks, and the Lux/Flux/Enzyme integration.
