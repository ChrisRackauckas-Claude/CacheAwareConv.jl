"""
    ConvGeometry{N, S, P}

Immutable description of a convolution: the sizes of the input `x`
`(I_1, …, I_S, C_in, B)`, the weights `w` `(K_1, …, K_S, C_in ÷ groups, C_out)`,
and the resulting output `y` `(O_1, …, O_S, C_out, B)`, together with the
spatial `stride`, zero `pad` (stored as `(lo_1, hi_1, …, lo_S, hi_S)`),
`dilation`, channel `groups`, and whether the kernel is `flipped`
(`false` = true convolution, `true` = cross-correlation).

`N = S + 2` is the number of array dimensions. Padding entries may be
negative, which crops the input.

Index convention (1-based) for the spatial dimension `i`:

    x_index = (o - 1) * stride_i - lo_i + 1 + (k′ - 1) * dilation_i

with `k′ = k` when `flipped` and `k′ = K_i - k + 1` otherwise.
"""
struct ConvGeometry{N, S, P}
    xsize::NTuple{N, Int}
    wsize::NTuple{N, Int}
    ysize::NTuple{N, Int}
    stride::NTuple{S, Int}
    pad::NTuple{P, Int}          # P == 2S entries: (lo_1, hi_1, ..., lo_S, hi_S)
    dilation::NTuple{S, Int}
    groups::Int
    flipped::Bool
end

_expand(::Val{S}, v::Integer) where {S} = ntuple(_ -> Int(v), Val(S))
function _expand(::Val{S}, v::NTuple{S, <:Integer}) where {S}
    return map(Int, v)
end
function _expand(::Val{S}, v::Tuple) where {S}
    throw(ArgumentError("expected an Integer or an NTuple{$S}, got a tuple of length $(length(v))"))
end

_expand_pad(::Val{S}, p::Integer) where {S} = ntuple(_ -> Int(p), Val(2S))
function _expand_pad(::Val{S}, p::NTuple{S, <:Integer}) where {S}
    return ntuple(i -> Int(p[(i + 1) ÷ 2]), Val(2S))
end
function _expand_pad(::Val{S}, p::NTuple{S2, <:Integer}) where {S, S2}
    S2 == 2S && return map(Int, p)
    throw(ArgumentError("pad must be an Integer, an NTuple{$S} (symmetric), or an NTuple{$(2S)} (lo/hi per dim); got length $S2"))
end

"""
    conv_output_size(I, K, stride, lo, hi, dilation)

Output extent of one spatial dimension.
"""
function conv_output_size(I::Int, K::Int, s::Int, lo::Int, hi::Int, d::Int)
    return (I + lo + hi - d * (K - 1) - 1) ÷ s + 1
end

"""
    ConvGeometry(xsize, wsize; stride = 1, pad = 0, dilation = 1, groups = 1, flipped = false)

Validate a convolution configuration and compute its output size.
"""
function ConvGeometry(
        xsize::Dims{N}, wsize::Dims{N};
        stride = 1, pad = 0, dilation = 1, groups::Integer = 1, flipped::Bool = false
    ) where {N}
    N >= 3 || throw(ArgumentError("arrays must have at least 3 dimensions (spatial..., channels, batch); got $N"))
    S = N - 2
    st = _expand(Val(S), stride)
    pd = _expand_pad(Val(S), pad)
    dl = _expand(Val(S), dilation)
    G = Int(groups)
    all(>(0), st) || throw(ArgumentError("stride must be positive, got $st"))
    all(>(0), dl) || throw(ArgumentError("dilation must be positive, got $dl"))
    G > 0 || throw(ArgumentError("groups must be positive, got $G"))
    cin = xsize[N - 1]
    cout = wsize[N]
    cin % G == 0 || throw(DimensionMismatch("input channels ($cin) must be divisible by groups ($G)"))
    cout % G == 0 || throw(DimensionMismatch("output channels ($cout) must be divisible by groups ($G)"))
    wsize[N - 1] == cin ÷ G || throw(
        DimensionMismatch(
            "weight size $(wsize) is incompatible with input channels $cin and groups $G; expected size(w, $(N - 1)) == $(cin ÷ G)"
        )
    )
    ospatial = ntuple(Val(S)) do i
        o = conv_output_size(xsize[i], wsize[i], st[i], pd[2i - 1], pd[2i], dl[i])
        o >= 1 || throw(
            DimensionMismatch(
                "spatial dimension $i: input $(xsize[i]) with pad ($(pd[2i - 1]), $(pd[2i])), kernel $(wsize[i]), dilation $(dl[i]) yields non-positive output extent $o"
            )
        )
        o
    end
    ysize = (ospatial..., cout, xsize[N])
    return ConvGeometry{N, S, 2S}(map(Int, xsize), map(Int, wsize), ysize, st, pd, dl, G, flipped)
end

spatial_dims(::ConvGeometry{N, S}) where {N, S} = S
input_size(g::ConvGeometry) = g.xsize
kernel_size(g::ConvGeometry) = g.wsize
output_size(g::ConvGeometry) = g.ysize
channels_in(g::ConvGeometry{N}) where {N} = g.xsize[N - 1]
channels_out(g::ConvGeometry{N}) where {N} = g.wsize[N]
batch_size(g::ConvGeometry{N}) where {N} = g.xsize[N]
groups(g::ConvGeometry) = g.groups
flipped(g::ConvGeometry) = g.flipped
stride(g::ConvGeometry) = g.stride
padding(g::ConvGeometry) = g.pad
dilation(g::ConvGeometry) = g.dilation
pad_lo(g::ConvGeometry{N, S}) where {N, S} = ntuple(i -> g.pad[2i - 1], Val(S))
pad_hi(g::ConvGeometry{N, S}) where {N, S} = ntuple(i -> g.pad[2i], Val(S))

"""
    kernel_index(g, k, i)

Map the weight index `k` along spatial dimension `i` to the effective tap
offset index `k′` (see [`ConvGeometry`](@ref)).
"""
kernel_index(g::ConvGeometry, k::Int, i::Int) = g.flipped ? k : g.wsize[i] - k + 1

function check_conv_args(g::ConvGeometry{N}, y, x, w) where {N}
    size(y) == g.ysize || throw(DimensionMismatch("output has size $(size(y)), expected $(g.ysize)"))
    size(x) == g.xsize || throw(DimensionMismatch("input has size $(size(x)), expected $(g.xsize)"))
    size(w) == g.wsize || throw(DimensionMismatch("weights have size $(size(w)), expected $(g.wsize)"))
    return nothing
end
