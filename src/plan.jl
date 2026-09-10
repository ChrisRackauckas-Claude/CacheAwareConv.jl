# ConvPlan: everything computed once per (eltype, sizes, hyper-parameters):
# geometry, cache-derived blocking, tap offsets, and all scratch buffers.

"""
    ConvPlan(T, xsize, wsize; stride = 1, pad = 0, dilation = 1, groups = 1,
             flipped = false, nthreads = Threads.nthreads(), cache = cache_info())

Pre-plan a convolution of arrays with element type `T`: input of size `xsize`
`(spatial..., C_in, batch)` and weights of size `wsize`
`(kernel..., C_in ÷ groups, C_out)`. The plan owns every scratch buffer, so
[`conv!`](@ref) with a plan does not allocate.

Keyword arguments follow the deep-learning convention: `stride`, `pad`
(an integer, one value per spatial dimension, or `(lo, hi)` pairs per
dimension), `dilation`, channel `groups`, and `flipped` (`false` performs a
true convolution with the kernel reversed, `true` a cross-correlation).
`nthreads` is the maximum number of tasks used; `cache` overrides the cache
sizes the blocking is derived from. The buffers needed by [`∇conv_data!`](@ref)
and [`∇conv_filter!`](@ref) are allocated on first use unless
`gradients = true`, which allocates them up front.

See also [`plan_conv`](@ref), [`output_size`](@ref).
"""
struct ConvPlan{T, Tc, N, S, P, V, MR, NR, NP, SIMD}
    geom::ConvGeometry{N, S, P}
    cache::CacheInfo
    Kc::Int                       # input channels per block (weight panel in L1)
    Nc::Int                       # output channels per block (packed weights in L3)
    tile::NTuple{S, Int}          # output block per work item (packed input in L2)
    nblocks::NTuple{S, Int}
    Lp::Int                       # packed phase-segment length along dim 1
    W1::Int                       # stride_1 * Lp
    R::NTuple{S, Int}             # packed extents per spatial dim (R[1] == W1)
    xci_stride::Int               # elements per packed channel
    xplane_stride::Int            # elements per packed plane (Kc channels)
    Lpy::Int                      # buffered-output row length (multiple of V)
    taps::Vector{Int}             # per-tap offsets into the packed tile
    Wp::Vector{Tc}                # packed weights
    xbufs::Vector{Vector{Tc}}     # per-task packed input tiles
    ybufs::Vector{Vector{Tc}}     # per-task output accumulation tiles (empty when direct)
    direct::Bool                  # kernel writes `y` directly
    nthreads::Int
    grad::Base.RefValue{Any}      # lazily built gradient state (see grad.jl)
    lock::ReentrantLock
end

const SIMD_MIN_ITEMS_PER_THREAD = 1

_sz(::Type{Tc}) where {Tc} = isbitstype(Tc) ? sizeof(Tc) : 2 * sizeof(Int)

function _packed_extents(g::ConvGeometry{N, S}, tile::NTuple{S, Int}, V::Int) where {N, S}
    Lp = cld(tile[1], V) * V + max_tap_shift(g)
    W1 = g.stride[1] * Lp
    R = ntuple(Val(S)) do i
        i == 1 ? W1 : (tile[i] - 1) * g.stride[i] + (g.wsize[i] - 1) * g.dilation[i] + 1
    end
    return Lp, W1, R
end

function _tile_bytes(g::ConvGeometry, Kc::Int, NP::Int, V::Int, tile, sz::Int)
    _, _, R = _packed_extents(g, tile, V)
    return NP * Kc * prod(R) * sz
end

# Largest t in lo:hi with pred(t) true, assuming pred is monotone (true then false).
function _largest_true(pred, lo::Int, hi::Int)
    pred(lo) || return lo - 1
    while lo < hi
        mid = (lo + hi + 1) ÷ 2
        if pred(mid)
            lo = mid
        else
            hi = mid - 1
        end
    end
    return lo
end

"""
    choose_blocking(g, Tc, NP, V, MR, NR, cache, nthreads) -> (Kc, Nc, tile)

Derive the cache blocking:
- `Kc` input channels so the weight panel `K * Kc * NR * NP` fits half of L1;
- an output `tile` so the packed input tile for `Kc` channels fits half of
  L2, shrinking the last spatial dimensions first, then the width in
  multiples of `MR*V`, then halving `Kc` if even a single tile row is too big;
- `Nc` output channels so the packed weights of one `(co block, ci block)`
  fit half of the per-core L3.
Finally the tile is split further if that is needed to give every thread at
least one work item.
"""
function choose_blocking(g::ConvGeometry{N, S}, ::Type{Tc}, NP::Int, V::Int, MR::Int, NR::Int, cache::CacheInfo, nthreads::Int) where {N, S, Tc}
    sz = _sz(Tc)
    G = g.groups
    cin_g = channels_in(g) ÷ G
    cout_g = channels_out(g) ÷ G
    K = prod(ntuple(i -> g.wsize[i], Val(S)))
    O = ntuple(i -> g.ysize[i], Val(S))
    Kc = clamp(fld(cache.l1 ÷ 2, K * NR * NP * sz), 1, cin_g)
    budget = cache.l2 ÷ 2
    fits(tile, kc) = _tile_bytes(g, kc, NP, V, tile, sz) <= budget
    tile = O
    minw = min(O[1], MR * V)
    while true
        tile = O
        fits(tile, Kc) && break
        # shrink trailing spatial dims
        found = false
        for i in S:-1:2
            t1 = Base.setindex(tile, 1, i)
            if fits(t1, Kc)
                ti = _largest_true(t -> fits(Base.setindex(tile, t, i), Kc), 1, O[i])
                tile = Base.setindex(tile, ti, i)
                found = true
                break
            else
                tile = t1
            end
        end
        found && break
        # shrink width in multiples of MR*V
        nunits = cld(O[1], MR * V)
        if fits(Base.setindex(tile, minw, 1), Kc)
            u = _largest_true(u -> fits(Base.setindex(tile, min(O[1], u * MR * V), 1), Kc), 1, nunits)
            tile = Base.setindex(tile, min(O[1], u * MR * V), 1)
            break
        end
        if Kc == 1
            tile = Base.setindex(tile, minw, 1)
            break
        end
        Kc = max(1, Kc ÷ 2)
    end
    # Ensure enough work items for the threads (split trailing dims, then width).
    nitems(t) = batch_size(g) * prod(map(cld, O, t))
    for i in S:-1:2
        nitems(tile) >= nthreads && break
        while nitems(tile) < nthreads && tile[i] > 1
            tile = Base.setindex(tile, cld(tile[i], 2), i)
        end
    end
    if nitems(tile) < nthreads
        while nitems(tile) < nthreads && tile[1] > MR * V
            units = cld(tile[1], MR * V)
            tile = Base.setindex(tile, cld(units, 2) * MR * V, 1)
        end
    end
    tile = map(min, tile, O)
    Nc = clamp(fld(cache.l3 ÷ 2, K * Kc * NP * sz), NR, cout_g)
    Nc = min(cout_g, cld(Nc, NR) * NR)
    return Kc, Nc, tile
end

function ConvPlan(
        ::Type{T}, xsize::Dims{N}, wsize::Dims{N};
        stride = 1, pad = 0, dilation = 1, groups::Integer = 1, flipped::Bool = false,
        nthreads::Integer = Threads.nthreads(), cache::CacheInfo = cache_info(), gradients::Bool = false
    ) where {T, N}
    g = ConvGeometry(xsize, wsize; stride, pad, dilation, groups, flipped)
    p = ConvPlan(T, g; nthreads, cache)
    gradients && grad_state(p)
    return p
end

function ConvPlan(::Type{T}, g::ConvGeometry{N, S, P}; nthreads::Integer = Threads.nthreads(), cache::CacheInfo = cache_info()) where {T, N, S, P}
    T <: Number || throw(ArgumentError("element type must be a Number, got $T"))
    Tc = compute_type(T)
    NP = nplanes(T)
    SIMD = simd_type(Tc)
    V = vector_width(Tc)
    MR, NR = register_tile(Tc, NP)
    nt = max(1, Int(nthreads))
    Kc, Nc, tile = choose_blocking(g, Tc, NP, V, MR, NR, cache, nt)
    Lp, W1, R = _packed_extents(g, tile, V)
    xci_stride = prod(R)
    xplane_stride = xci_stride * Kc
    xbuf_len = xplane_stride * NP
    G = g.groups
    cout_g = channels_out(g) ÷ G
    Lpy = cld(tile[1], V) * V
    direct = SIMD && (Tc === T) && NP == 1
    ybuf_len = direct ? 0 : Lpy * prod(ntuple(i -> tile[i + 1], Val(S - 1))) * cout_g * NP
    nitems = batch_size(g) * prod(map(cld, ntuple(i -> g.ysize[i], Val(S)), tile))
    ntasks = max(1, min(nt, nitems))
    # Overflow guard for 32-bit platforms.
    for len in (xbuf_len, ybuf_len, prod(g.ysize), prod(g.xsize))
        len <= typemax(Int) ÷ 4 || throw(ArgumentError("problem too large for this platform's Int"))
    end
    taps = tap_offsets(g, Lp, W1, R)
    Wp = Vector{Tc}(undef, packed_weight_length(g, NP))
    xbufs = [Vector{Tc}(undef, xbuf_len) for _ in 1:ntasks]
    ybufs = [Vector{Tc}(undef, ybuf_len) for _ in 1:(direct ? 0 : ntasks)]
    return ConvPlan{T, Tc, N, S, P, V, MR, NR, NP, SIMD}(
        g, cache, Kc, Nc, tile, map(cld, ntuple(i -> g.ysize[i], Val(S)), tile), Lp, W1, R,
        xci_stride, xplane_stride, Lpy, taps, Wp, xbufs, ybufs, direct, ntasks, Ref{Any}(nothing), ReentrantLock()
    )
end

"""
    plan_conv(x, w; kwargs...)

Build a [`ConvPlan`](@ref) for the arrays `x` and `w` (sizes and element type
are taken from them). Keyword arguments are those of `ConvPlan`.
"""
function plan_conv(x::AbstractArray{T, N}, w::AbstractArray{T, N}; kwargs...) where {T <: Number, N}
    return ConvPlan(T, size(x), size(w); kwargs...)
end
function plan_conv(x::AbstractArray{<:Number, N}, w::AbstractArray{<:Number, N}; kwargs...) where {N}
    throw(ArgumentError("x and w must have the same element type; got $(eltype(x)) and $(eltype(w)). Convert one of them first."))
end

geometry(p::ConvPlan) = p.geom
for f in (:output_size, :input_size, :kernel_size, :channels_in, :channels_out, :batch_size, :groups, :flipped, :stride, :padding, :dilation, :spatial_dims)
    @eval $f(p::ConvPlan) = $f(p.geom)
end
Base.eltype(::ConvPlan{T}) where {T} = T
compute_type(::ConvPlan{T, Tc}) where {T, Tc} = Tc
vector_width(::ConvPlan{T, Tc, N, S, P, V}) where {T, Tc, N, S, P, V} = V
register_tile(::ConvPlan{T, Tc, N, S, P, V, MR, NR}) where {T, Tc, N, S, P, V, MR, NR} = (MR, NR)
nplanes(::ConvPlan{T, Tc, N, S, P, V, MR, NR, NP}) where {T, Tc, N, S, P, V, MR, NR, NP} = NP

function Base.show(io::IO, p::ConvPlan{T, Tc, N, S, P, V, MR, NR, NP, SIMD}) where {T, Tc, N, S, P, V, MR, NR, NP, SIMD}
    g = p.geom
    print(io, "ConvPlan{", T, "}(", g.xsize, " ⋆ ", g.wsize, " → ", g.ysize)
    print(io, "; stride=", g.stride, ", pad=", g.pad, ", dilation=", g.dilation, ", groups=", g.groups, ", flipped=", g.flipped)
    print(io, ") [", SIMD ? "SIMD " : "scalar ", Tc, " V=", V, " MR=", MR, " NR=", NR, " NP=", NP)
    print(io, " Kc=", p.Kc, " Nc=", p.Nc, " tile=", p.tile, " tasks=", p.nthreads, "]")
    return nothing
end

"""
    allocated_bytes(p::ConvPlan)

Total size of the scratch buffers owned by the plan.
"""
function allocated_bytes(p::ConvPlan{T, Tc}) where {T, Tc}
    n = length(p.Wp) + sum(length, p.xbufs; init = 0) + sum(length, p.ybufs; init = 0)
    return n * _sz(Tc)
end
