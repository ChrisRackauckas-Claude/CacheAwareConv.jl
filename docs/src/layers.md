# Lux and Flux layers

Two package extensions provide convolution layers that use this package
internally and mirror the constructors of `Lux.Conv` and `Flux.Conv`. They
are named `LuxConv` and `FluxConv` so they do not shadow the framework layers.
Both cache a [`ConvPlan`](@ref) per input shape inside the layer, apply the
bias inside the convolution epilogue, and apply the activation as a broadcast
so that Zygote and Enzyme can differentiate it.

## Lux

```julia
using Lux, CacheAwareConv, Random

layer = LuxConv((3, 3), 3 => 16, relu; pad = 1, stride = 2)
ps, st = Lux.setup(Random.default_rng(), layer)
y, st = layer(x, ps, st)
```

Keyword arguments: `stride`, `pad` (including `SamePad()` from either
package), `dilation`, `groups`, `use_bias`, `init_weight`, `init_bias`,
`cross_correlation`. Parameters are `(; weight, bias)` with
`size(weight) == (k..., in ÷ groups, out)`.

## Flux

```julia
using Flux, CacheAwareConv

model = Chain(FluxConv((3, 3), 1 => 8, relu; pad = SamePad()), Flux.flatten, Dense(8 * 28 * 28 => 10))
```

Keyword arguments: `init`, `stride`, `pad`, `dilation`, `groups`, `bias`
(`true`, `false`, or a vector). A `FluxConv(weight, bias, σ; ...)` constructor
wraps existing arrays. The layer is a `Flux.@layer` with `weight` and `bias`
trainable.

```@docs
LuxConv
FluxConv
```
