# Automatic differentiation

The convolution is linear in both `x` and `w`, so both gradients are
themselves convolutions and are computed with the same cache-aware machinery
([`∇conv_data!`](@ref), [`∇conv_filter!`](@ref)). Two package extensions
expose them to AD systems.

## ChainRulesCore (Zygote and friends)

Loading ChainRulesCore activates an `rrule` for
[`CacheAwareConv.conv_bias`](@ref)`(x, w, bias, plan)` that returns tangents
for `x`, `w` and `bias`, and one for `conv(x, w, plan; bias, σ)` (keyword
arguments carry no tangents in ChainRules, so use `conv_bias` when the bias is
trained). The activation is differentiated by the calling AD through
`rrule_via_ad`.

```julia
using CacheAwareConv, Zygote
plan = plan_conv(x, w; pad = 1)
loss(x, w, b) = sum(abs2, CacheAwareConv.conv_bias(x, w, b, plan))
gx, gw, gb = Zygote.gradient(loss, x, w, b)
```

## Enzyme

Loading EnzymeCore (which Enzyme does) activates forward and reverse rules for
the in-place positional core [`CacheAwareConv.conv_core!`](@ref)`(y, x, w, bias,
plan, accumulate)`. Both `conv!` with `σ = identity` and `conv_bias` call it,
so differentiating through those works in either mode, including batched
(vector) modes:

```julia
using CacheAwareConv, Enzyme
plan = plan_conv(x, w; pad = 1)
f(x, w) = sum(abs2, conv(x, w, plan))
dx, dw = Enzyme.gradient(Reverse, f, x, w)
```

The reverse rule accumulates `∇conv_data!` and `∇conv_filter!` into the
shadows (`accumulate = true`), sums the output gradient into the bias shadow,
and zeroes the output shadow afterwards. Plans are marked inactive
(`EnzymeRules.inactive_type`): they contain only geometry and scratch buffers.
Because a plan's buffers are written during a call, closures that capture a
plan must be passed to `autodiff` as `Const(f)`.

The fused-activation path (`conv!(...; σ = tanh)`) has no rules; Enzyme would
attempt to differentiate the SIMD kernel itself. Apply the activation as a
separate broadcast instead.

## Why gradients call another cache-aware convolution

`∂/∂x` of a convolution is a convolution of the output gradient with the
flipped, channel-transposed weights; for stride `s` the output gradient is
first zero-stuffed by `s`. The plan for this transposed convolution is built
lazily and cached in the forward plan, so repeated backward passes allocate
nothing. `∂/∂w` is a correlation between the input and the output gradient
with tiny spatial output (the kernel size) and a huge reduction (all
positions and the batch); a forward-style kernel would waste its lanes there,
so it has a dedicated reduction kernel instead. See [Design](@ref design).
