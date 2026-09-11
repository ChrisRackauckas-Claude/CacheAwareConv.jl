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

## Benchmarks

One AMD EPYC 9354 (Zen 4, AVX-512), Julia 1.12, `Float32` unless noted,
against NNlib's im2col + BLAS `conv!` (which uses all cores through the
BLAS). Run `julia --project=benchmark -t 16 benchmark/benchmarks.jl`.

| shape | ours, 1 thread | ours, 16 threads | NNlib | speedup (16 vs NNlib) |
|---|---|---|---|---|
| 56×56, 64→64, 3×3, batch 8 | 85 GFLOPS | 958 GFLOPS | 122 GFLOPS | 7.8× |
| 224×224, 3→64, 7×7, stride 2, batch 8 | 94 | 361 | 161 | 2.2× |
| 28×28, 128→128, 3×3, batch 16 | 85 | 1121 | 399 | 2.8× |
| 7×7, 512→512, 3×3, batch 16 | 72 | 660 | 382 | 1.7× |
| 1D 65536, 16→32, k=9, batch 4 | 98 | 569 | 98 | 5.8× |
| 3D 32³, 8→16, k=3, batch 2 | 89 | 503 | 39 | 13× |
| 4096² image, 1→1, 3×3 (stencil) | 48 | 274 | 2.1 | 129× |
| Float64 56×56, 64→64, 3×3, batch 8 | 49 | 554 | 99 | 5.6× |
| ComplexF32 56×56, 16→16, 3×3, batch 4 | 18 (real-FLOP equiv.) | 101 | 17 | 5.9× |

One thread runs at roughly 80% of the core's FMA peak on compute-bound
shapes; 2D single-channel stencils reach 77–94% of peak. See the
[benchmark docs](https://docs.sciml.ai/CacheAwareConv/stable/benchmarks/)
for the stencil roofline table and streaming-bandwidth results.
