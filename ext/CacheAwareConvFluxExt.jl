module CacheAwareConvFluxExt

using CacheAwareConv
using CacheAwareConv: PlanCache, get_plan!, conv_bias, SamePad, calc_padding, _expand
using Flux
using Flux: @layer
const Functors = Flux.Functors

Functors.@leaf PlanCache

# Accept Flux's own SamePad marker as well as ours.
CacheAwareConv.calc_padding(::Flux.SamePad, k, dilation, stride) = calc_padding(SamePad(), k, dilation, stride)

"""
    FluxConv(k::NTuple{N, Integer}, in => out, σ = identity;
             stride = 1, pad = 0, dilation = 1, groups = 1, bias = true,
             init = Flux.glorot_uniform)
    FluxConv(weight::AbstractArray, bias = true, σ = identity; stride, pad, dilation, groups)

A Flux convolution layer backed by [`CacheAwareConv.conv`](@ref). The
constructors mirror `Flux.Conv` (including `SamePad()` for `pad`): `weight`
has size `(k..., in ÷ groups, out)`, `bias` is a vector of length `out` or
`false`. Performs a true convolution (kernel flipped), like `Flux.Conv`.

The convolution plan for each input shape is cached inside the layer.
"""
struct FluxConv{N, M, F, A, V}
    σ::F
    weight::A
    bias::V
    stride::NTuple{N, Int}
    pad::NTuple{M, Int}
    dilation::NTuple{N, Int}
    groups::Int
    plans::PlanCache
end

@layer FluxConv trainable = (weight, bias)

function CacheAwareConv.FluxConv(
        w::AbstractArray{T, N}, b = true, σ = identity;
        stride = 1, pad = 0, dilation = 1, groups::Integer = 1
    ) where {T, N}
    S = N - 2
    size(w, N) % groups == 0 || throw(DimensionMismatch("Output channel dimension must be divisible by groups."))
    st = _expand(Val(S), stride)
    dl = _expand(Val(S), dilation)
    pd = calc_padding(pad, ntuple(i -> size(w, i), Val(S)), dl, st)
    bias = Flux.create_bias(w, b, size(w, N))
    return FluxConv(σ, w, bias, st, pd, dl, Int(groups), PlanCache())
end

function CacheAwareConv.FluxConv(
        k::NTuple{S, <:Integer}, ch::Pair{<:Integer, <:Integer}, σ = identity;
        init = Flux.glorot_uniform, stride = 1, pad = 0, dilation = 1, groups::Integer = 1, bias = true
    ) where {S}
    weight = Flux.convfilter(k, ch; init, groups)
    return CacheAwareConv.FluxConv(weight, bias, σ; stride, pad, dilation, groups)
end

_channels_in(l::FluxConv) = size(l.weight, ndims(l.weight) - 1) * l.groups
_channels_out(l::FluxConv) = size(l.weight, ndims(l.weight))

function (c::FluxConv)(x::AbstractArray)
    ndims(x) == ndims(c.weight) || throw(DimensionMismatch("layer $c expects ndims(input) == $(ndims(c.weight)), got $(summary(x))"))
    size(x, ndims(x) - 1) == _channels_in(c) || throw(DimensionMismatch("layer $c expects size(input, $(ndims(x) - 1)) == $(_channels_in(c)), got $(summary(x))"))
    w = c.weight
    xT = eltype(x) === eltype(w) ? x : convert(AbstractArray{eltype(w)}, x)
    p = get_plan!(c.plans, xT, w; stride = c.stride, pad = c.pad, dilation = c.dilation, groups = c.groups, flipped = false)
    bias = c.bias === false ? nothing : c.bias
    z = conv_bias(xT, w, bias, p)
    return c.σ === identity ? z : c.σ.(z)
end

function Base.show(io::IO, l::FluxConv)
    print(io, "FluxConv(", size(l.weight)[1:(ndims(l.weight) - 2)])
    print(io, ", ", _channels_in(l), " => ", _channels_out(l))
    l.σ == identity || print(io, ", ", l.σ)
    all(==(0), l.pad) || print(io, ", pad=", l.pad)
    all(==(1), l.stride) || print(io, ", stride=", l.stride)
    all(==(1), l.dilation) || print(io, ", dilation=", l.dilation)
    l.groups == 1 || print(io, ", groups=", l.groups)
    l.bias === false && print(io, ", bias=false")
    return print(io, ")")
end

end
