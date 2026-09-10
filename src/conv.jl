# The blocked driver: work-item decomposition, threading, and the loop nest
# around packing and the microkernel.

"""
    conv!(y, x, w, plan::ConvPlan; bias = nothing, σ = identity, accumulate = false)

Compute `y = σ.(conv(x, w) .+ bias)` in place using the pre-planned blocking.
`bias`, if given, is a vector with one entry per output channel. With
`accumulate = true` the result is added to the existing contents of `y`
instead (`y .+= conv(x, w) .+ bias`; `σ` must then be `identity`). All three
arrays must have the plan's element type and sizes. Returns `y`.

The activation `σ` is fused into the tile epilogue while the output tile is
still in cache. This fused path has no automatic-differentiation rules: for
training, use `σ = identity` (or [`conv_bias`](@ref)) and apply the activation
as a separate broadcast, which is what the Lux and Flux layers do.
"""
function conv!(
        y::AbstractArray{T, N}, x::AbstractArray{T, N}, w::AbstractArray{T, N}, p::ConvPlan{T, Tc, N};
        bias = nothing, σ::F = identity, accumulate::Bool = false
    ) where {T, Tc, N, F}
    check_conv_args(p.geom, y, x, w)
    _check_bias(bias, channels_out(p.geom))
    _check_strided(y, "y")
    _check_strided(x, "x")
    accumulate && σ !== identity && throw(ArgumentError("accumulate = true requires σ = identity"))
    if σ === identity
        # Positional core: this is what the AD rules attach to.
        return conv_core!(y, x, w, bias, p, accumulate)
    end
    S = N - 2
    iv = InputView(x, ntuple(_ -> 1, Val(S)))
    _conv_impl!(y, iv, w, p, bias, σ, false, accumulate)
    return y
end

function conv!(y::AbstractArray, x::AbstractArray, w::AbstractArray, p::ConvPlan{T}; kwargs...) where {T}
    throw(ArgumentError("conv!: all arrays must have element type $T (the plan's); got y::$(eltype(y)), x::$(eltype(x)), w::$(eltype(w)). Convert the arrays or build the plan with the desired type."))
end

"""
    conv(x, w, plan::ConvPlan; bias = nothing, σ = identity)
    conv(x, w; stride = 1, pad = 0, dilation = 1, groups = 1, flipped = false, bias = nothing, σ = identity, kwargs...)

Allocating convolution. The second form builds a plan (see [`ConvPlan`](@ref))
on the fly; reuse a plan when calling repeatedly with the same sizes.
"""
function conv(x::AbstractArray{T, N}, w::AbstractArray{T, N}, p::ConvPlan{T, Tc, N}; kwargs...) where {T <: Number, Tc, N}
    y = similar(x, T, output_size(p))
    return conv!(y, x, w, p; kwargs...)
end
function conv(x::AbstractArray{T, N}, w::AbstractArray{T, N}; bias = nothing, σ = identity, kwargs...) where {T <: Number, N}
    p = plan_conv(x, w; kwargs...)
    return conv(x, w, p; bias, σ)
end
function conv(x::AbstractArray{<:Number, N}, w::AbstractArray{<:Number, N}; kwargs...) where {N}
    T = promote_type(eltype(x), eltype(w))
    return conv(convert(AbstractArray{T}, x), convert(AbstractArray{T}, w); kwargs...)
end

_check_bias(::Nothing, ::Int) = nothing
function _check_bias(b::AbstractVector, cout::Int)
    length(b) == cout || throw(DimensionMismatch("bias has length $(length(b)), expected $cout (one per output channel)"))
    return nothing
end
function _check_bias(b, ::Int)
    throw(ArgumentError("bias must be `nothing` or an AbstractVector, got $(typeof(b))"))
end

function _check_strided(a::AbstractArray, name)
    a isa StridedArray || throw(ArgumentError("$name must be a strided array (Array, or a contiguous view/reshape); got $(typeof(a))"))
    Base.has_offset_axes(a) && throw(ArgumentError("$name must have 1-based axes"))
    strides(a)[1] == 1 || throw(ArgumentError("$name must have unit stride along its first dimension; got strides $(strides(a))"))
    return nothing
end

"""
    run_tasks(f, nitems, ntasks)

Call `f(task, items)` for contiguous chunks of `1:nitems` on `ntasks` tasks
(inline when `ntasks == 1`). Each task index owns its own scratch buffers.
"""
function run_tasks(f::F, nitems::Int, ntasks::Int) where {F}
    nt = max(1, min(ntasks, nitems))
    if nt == 1
        f(1, 1:nitems)
        return nothing
    end
    chunk = cld(nitems, nt)
    @sync for t in 1:nt
        lo = (t - 1) * chunk + 1
        hi = min(nitems, t * chunk)
        lo <= hi || continue
        Threads.@spawn f(t, lo:hi)
    end
    return nothing
end

# Decode work item `item` (1-based) into batch index and 0-based tile origin.
@inline function decode_item(p::ConvPlan{T, Tc, N, S}, item::Int) where {T, Tc, N, S}
    nb = p.nblocks
    # Mixed-radix decode of item-1 with radices nb[1], nb[2], ...; the last
    # "digit" is the batch index.
    divs = ntuple(Val(S)) do d
        q = 1
        for j in 1:(d - 1)
            q *= nb[j]
        end
        q
    end
    idx = item - 1
    o = ntuple(d -> ((idx ÷ divs[d]) % nb[d]) * p.tile[d], Val(S))
    b = idx ÷ (divs[S] * nb[S]) + 1
    return b, o
end

@inline function _packed_rowstrides(p::ConvPlan{T, Tc, N, S}) where {T, Tc, N, S}
    return ntuple(Val(S - 1)) do i
        st = p.W1
        for j in 2:i
            st *= p.R[j]
        end
        st
    end
end

function _conv_impl!(
        y::AbstractArray{T, N}, iv::InputView, w::AbstractArray{<:Number, N},
        p::ConvPlan{T, Tc, N, S, P, V, MR, NR, NP, SIMD}, bias, σ::F, conjw::Bool, accumulate::Bool
    ) where {T, Tc, N, S, P, V, MR, NR, NP, SIMD, F}
    g = p.geom
    pack_weights!(p.Wp, w, g, p.Kc, NR, Val(NP), conjw)
    nitems = batch_size(g) * prod(p.nblocks)
    run_tasks(nitems, p.nthreads) do task, items
        for item in items
            _work_item!(y, iv, p, bias, σ, accumulate, task, item)
        end
    end
    return y
end

function _work_item!(
        y::AbstractArray{T, N}, iv::InputView, p::ConvPlan{T, Tc, N, S, P, V, MR, NR, NP, SIMD},
        bias, σ::F, accumulate::Bool, task::Int, item::Int
    ) where {T, Tc, N, S, P, V, MR, NR, NP, SIMD, F}
    g = p.geom
    G = g.groups
    cin_g = channels_in(g) ÷ G
    cout_g = channels_out(g) ÷ G
    K = length(p.taps)
    Kc = p.Kc
    ncb = cld(cin_g, Kc)
    ntiles = cld(cout_g, NR)
    tiles_per_block = max(1, cld(p.Nc, NR))
    b, origin = decode_item(p, item)
    O = ntuple(i -> g.ysize[i], Val(S))
    te = ntuple(i -> min(p.tile[i], O[i] - origin[i]), Val(S))   # clipped tile extents
    Xp = p.xbufs[task]
    rowstr = _packed_rowstrides(p)
    rows = CartesianIndices(ntuple(i -> te[i + 1], Val(S - 1)))
    ystr = strides(y)
    # Buffered-output layout: [Lpy, tile[2:end]..., cout_g, NP]
    yb_rowstr = ntuple(Val(S - 1)) do i
        st = p.Lpy
        for j in 2:i
            st *= p.tile[j]
        end
        st
    end
    yb_costride = p.Lpy * prod(ntuple(i -> p.tile[i + 1], Val(S - 1)))
    yb_planestride = yb_costride * cout_g
    Yb = p.direct ? Xp : p.ybufs[task]   # Xp is a placeholder of the right type when direct
    ydest = p.direct ? y : Yb
    for grp in 1:G
        for cb in 1:ncb
            ci0 = (grp - 1) * cin_g + (cb - 1) * Kc
            kc = min(Kc, cin_g - (cb - 1) * Kc)
            pack_input_tile!(Xp, iv, g, b, ci0, kc, origin, p.Lp, p.W1, p.R, p.xci_stride, p.xplane_stride, Val(NP), false)
            acc = cb > 1 || (accumulate && p.direct)
            ct = 1
            while ct <= ntiles
                ct_end = min(ntiles, ct + tiles_per_block - 1)
                for t in ct:ct_end
                    nr = min(NR, cout_g - (t - 1) * NR)
                    wbase = weight_block_offset(g, Kc, NR, NP, grp, t, cb)
                    co0 = (grp - 1) * cout_g + (t - 1) * NR
                    for r in rows
                        xrow = 0
                        for i in 1:(S - 1)
                            xrow += (r[i] - 1) * g.stride[i + 1] * rowstr[i]
                        end
                        if p.direct
                            ybase = origin[1] * ystr[1] + co0 * ystr[N - 1] + (b - 1) * ystr[N]
                            for i in 1:(S - 1)
                                ybase += (origin[i + 1] + r[i] - 1) * ystr[i + 1]
                            end
                            ycs = ystr[N - 1]
                            yps = 0
                        else
                            ybase = (t - 1) * NR * yb_costride
                            for i in 1:(S - 1)
                                ybase += (r[i] - 1) * yb_rowstr[i]
                            end
                            ycs = yb_costride
                            yps = yb_planestride
                        end
                        wo = 0
                        while wo < te[1]
                            remaining = te[1] - wo
                            mr = min(MR, cld(remaining, V))
                            lanes = remaining - (mr - 1) * V
                            masked = SIMD && lanes < V
                            mask = lane_mask(Val(V), lanes)
                            dispatch_microkernel!(
                                Val(SIMD), Val(V), Val(MR), Val(NR), Val(NP), acc, mr, nr, masked, mask,
                                ydest, ybase + wo, ycs, yps,
                                Xp, xrow + wo, p.xci_stride, p.xplane_stride,
                                p.Wp, wbase, p.taps, K, kc
                            )
                            wo += mr * V
                        end
                    end
                end
                ct = ct_end + 1
            end
        end
        # Epilogue for this group's slice of the tile.
        if p.direct
            if bias !== nothing || σ !== identity
                _epilogue_direct!(y, g, b, origin, te, grp, cout_g, bias, σ)
            end
        else
            _finalize_buffered!(y, Yb, p, b, origin, te, grp, cout_g, yb_rowstr, yb_costride, yb_planestride, bias, σ, accumulate)
        end
    end
    return nothing
end

@inline _bias_at(::Nothing, ::Type{T}, co) where {T} = zero(T)
@inline _bias_at(b::AbstractVector, ::Type{T}, co) where {T} = @inbounds T(b[co])

function _epilogue_direct!(y::AbstractArray{T, N}, g::ConvGeometry{N, S}, b::Int, origin, te, grp::Int, cout_g::Int, bias, σ::F) where {T, N, S, F}
    region = CartesianIndices(ntuple(i -> (origin[i] + 1):(origin[i] + te[i]), Val(S)))
    @inbounds for co_l in 1:cout_g
        co = (grp - 1) * cout_g + co_l
        bv = _bias_at(bias, T, co)
        for I in region
            y[Tuple(I)..., co, b] = σ(y[Tuple(I)..., co, b] + bv)
        end
    end
    return nothing
end

@inline _recombine(::Type{T}, Yb, i, pstride, ::Val{1}) where {T} = @inbounds T(Yb[i])
@inline function _recombine(::Type{T}, Yb, i, pstride, ::Val{3}) where {T}
    @inbounds begin
        a1 = Yb[i]
        a2 = Yb[i + pstride]
        a3 = Yb[i + 2pstride]
    end
    return T(complex(a1 - a2, a3 - a1 - a2))
end

function _finalize_buffered!(
        y::AbstractArray{T, N}, Yb::Vector{Tc}, p::ConvPlan{T, Tc, N, S, P, V, MR, NR, NP},
        b::Int, origin, te, grp::Int, cout_g::Int, yb_rowstr, yb_costride::Int, yb_planestride::Int, bias, σ::F, accumulate::Bool
    ) where {T, Tc, N, S, P, V, MR, NR, NP, F}
    rows = CartesianIndices(ntuple(i -> te[i + 1], Val(S - 1)))
    @inbounds for co_l in 1:cout_g
        co = (grp - 1) * cout_g + co_l
        bv = _bias_at(bias, T, co)
        cbase = (co_l - 1) * yb_costride
        for r in rows
            rbase = cbase
            for i in 1:(S - 1)
                rbase += (r[i] - 1) * yb_rowstr[i]
            end
            for wo in 1:te[1]
                v = _recombine(T, Yb, rbase + wo, yb_planestride, Val(NP))
                I = (origin[1] + wo, ntuple(i -> origin[i + 1] + r[i], Val(S - 1))..., co, b)
                if accumulate
                    y[I...] += v + bv
                else
                    y[I...] = σ(v + bv)
                end
            end
        end
    end
    return nothing
end
