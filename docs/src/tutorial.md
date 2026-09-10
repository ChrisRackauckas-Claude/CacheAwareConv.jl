# Tutorial

## Planning once, running many times

A [`ConvPlan`](@ref) holds everything derived from the array sizes and the
element type: the output size, the cache blocking, tap offsets, and all
scratch buffers. Build it once and reuse it for every call with the same
shapes:

```@example tut
using CacheAwareConv

x = randn(Float32, 64, 64, 16, 4)
w = randn(Float32, 3, 3, 16, 32)
plan = ConvPlan(Float32, size(x), size(w); pad = 1, stride = 2)
```

The printed summary shows the derived parameters: the compute type and vector
width `V`, the register tile `MR × NR`, the input-channel block `Kc`, the
output-channel block `Nc`, the spatial output `tile` packed per work item, and
the number of tasks.

```@example tut
y = similar(x, Float32, CacheAwareConv.output_size(plan))
conv!(y, x, w, plan)
@allocated conv!(y, x, w, plan)   # 0 when single-threaded
```

`plan_conv(x, w; kwargs...)` is a shorthand that reads sizes and element type
from the arrays, and `conv(x, w; kwargs...)` plans and runs in one go.

## Keyword arguments

| keyword | meaning |
|---|---|
| `stride` | integer or one per spatial dimension |
| `pad` | integer, one per spatial dimension (symmetric), or `(lo_1, hi_1, lo_2, hi_2, …)`; negative values crop |
| `dilation` | integer or one per spatial dimension |
| `groups` | channel groups; `size(w, N-1) == size(x, N-1) ÷ groups` |
| `flipped` | `false` (default): true convolution, kernel reversed; `true`: cross-correlation |
| `nthreads` | maximum number of tasks (default `Threads.nthreads()`) |
| `cache` | a [`CacheInfo`](@ref) overriding the detected cache sizes |
| `gradients` | allocate the gradient buffers up front |

## Bias and activation

`conv!` and `conv` accept `bias` (one value per output channel) and an
activation `σ`, which are applied to each output tile while it is still in
cache:

```@example tut
b = randn(Float32, 32)
y2 = conv(x, w, plan; bias = b, σ = tanh)
y2 ≈ tanh.(conv(x, w, plan) .+ reshape(b, 1, 1, :, 1))
```

The fused activation has no differentiation rules; for training use
`σ = identity` (or [`CacheAwareConv.conv_bias`](@ref)) and broadcast the
activation afterwards, as the Lux and Flux layers do.

## Element types

```@example tut
for T in (Float16, Float64, ComplexF32)
    xt = T.(x[:, :, 1:4, 1:1]); wt = T.(w[:, :, 1:4, 1:8])
    println(T, " => ", plan_conv(xt, wt))
end
```

`Float16` inputs are packed into `Float32` buffers and accumulated in
`Float32`; complex inputs are split into three real planes per element
(`re`, `im`, `re + im`) so that each complex multiply-add costs three real
fused multiply-adds instead of four. Types without a SIMD kernel (dual numbers,
`BigFloat`, …) take a generic scalar path with the same blocking.

## Gradients

```@example tut
ȳ = randn(Float32, size(y))
x̄ = ∇conv_data(ȳ, w, plan)
w̄ = ∇conv_filter(x, ȳ, plan)
size(x̄), size(w̄)
```

The in-place forms accept `accumulate = true` to add into the destination,
which is what the Enzyme rules use.

## Checking against the reference

[`reference_conv!`](@ref) is a direct translation of the definition and is the
oracle for the test suite. It is exported so you can check your own
configurations:

```@example tut
yr = similar(y)
reference_conv!(yr, x, w, CacheAwareConv.geometry(plan))
y ≈ yr
```
