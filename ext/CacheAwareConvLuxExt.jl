module CacheAwareConvLuxExt

using CacheAwareConv
using CacheAwareConv: PlanCache, get_plan!, conv_bias, SamePad, calc_padding, _expand
using Lux
using Lux: AbstractLuxLayer
const AbstractRNG = Lux.Random.AbstractRNG

# Accept Lux's own SamePad marker as well as ours.
CacheAwareConv.calc_padding(::Lux.SamePad, k, dilation, stride) = calc_padding(SamePad(), k, dilation, stride)

"""
    LuxConv(k::NTuple{N, Integer}, in_chs => out_chs, activation = identity;
            stride = 1, pad = 0, dilation = 1, groups = 1, use_bias = true,
            init_weight = Lux.kaiming_uniform, init_bias = Lux.zeros32,
            cross_correlation = false)

A Lux convolution layer backed by [`CacheAwareConv.conv`](@ref). The
constructor mirrors `Lux.Conv` (including `SamePad()` for `pad`) and the
parameters have the same shapes: `weight` of size `(k..., in_chs ÷ groups, out_chs)`
and, when `use_bias`, `bias` of length `out_chs`. Input layout is
`(spatial..., in_chs, batch)`.

The convolution plan for each input shape is cached inside the layer.
"""
struct LuxConv{S, F, IW, IB} <: AbstractLuxLayer
    activation::F
    in_chs::Int
    out_chs::Int
    kernel_size::NTuple{S, Int}
    stride::NTuple{S, Int}
    pad::NTuple{S2, Int} where {S2}
    dilation::NTuple{S, Int}
    groups::Int
    use_bias::Bool
    cross_correlation::Bool
    init_weight::IW
    init_bias::IB
    plans::PlanCache
end

function CacheAwareConv.LuxConv(
        k::NTuple{S, <:Integer}, ch::Pair{<:Integer, <:Integer}, activation = identity;
        stride = 1, pad = 0, dilation = 1, groups::Integer = 1, use_bias::Bool = true,
        init_weight = Lux.kaiming_uniform, init_bias = Lux.zeros32, cross_correlation::Bool = false
    ) where {S}
    st = _expand(Val(S), stride)
    dl = _expand(Val(S), dilation)
    pd = calc_padding(pad, map(Int, k), dl, st)
    ch[1] % groups == 0 || throw(DimensionMismatch("Input channel dimension must be divisible by groups."))
    ch[2] % groups == 0 || throw(DimensionMismatch("Output channel dimension must be divisible by groups."))
    return LuxConv(
        activation, Int(ch[1]), Int(ch[2]), map(Int, k), st, pd, dl, Int(groups), use_bias,
        cross_correlation, init_weight, init_bias, PlanCache()
    )
end

function Lux.initialparameters(rng::AbstractRNG, c::LuxConv)
    weight = c.init_weight(rng, c.kernel_size..., c.in_chs ÷ c.groups, c.out_chs)
    c.use_bias || return (; weight)
    return (; weight, bias = c.init_bias(rng, c.out_chs))
end
Lux.initialstates(::AbstractRNG, ::LuxConv) = NamedTuple()
function Lux.parameterlength(c::LuxConv)
    return prod(c.kernel_size) * (c.in_chs ÷ c.groups) * c.out_chs + (c.use_bias ? c.out_chs : 0)
end
Lux.statelength(::LuxConv) = 0

function (c::LuxConv)(x::AbstractArray, ps, st::NamedTuple)
    w = ps.weight
    xT = eltype(x) === eltype(w) ? x : convert(AbstractArray{eltype(w)}, x)
    p = get_plan!(c.plans, xT, w; stride = c.stride, pad = c.pad, dilation = c.dilation, groups = c.groups, flipped = c.cross_correlation)
    bias = c.use_bias ? ps.bias : nothing
    z = conv_bias(xT, w, bias, p)
    y = c.activation === identity ? z : c.activation.(z)
    return y, st
end

function Base.show(io::IO, l::LuxConv)
    print(io, "LuxConv(", l.kernel_size, ", ", l.in_chs, " => ", l.out_chs)
    l.activation == identity || print(io, ", ", l.activation)
    all(==(0), l.pad) || print(io, ", pad=", l.pad)
    all(==(1), l.stride) || print(io, ", stride=", l.stride)
    all(==(1), l.dilation) || print(io, ", dilation=", l.dilation)
    l.groups == 1 || print(io, ", groups=", l.groups)
    l.use_bias || print(io, ", use_bias=false")
    l.cross_correlation && print(io, ", cross_correlation=true")
    return print(io, ")")
end

end
