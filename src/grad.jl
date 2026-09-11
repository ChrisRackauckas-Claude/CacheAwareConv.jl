# Gradients of the convolution with respect to its input (a transposed
# convolution, reusing the forward machinery) and its weights (a dedicated
# reduction kernel over the same packed tiles).

"""
    TransposedWeights(w, g)

Lazy view of the weights `w` of geometry `g` as the weights of the transposed
convolution: `wt[k..., co_l, ci] = w[k..., ci_l, co]` where the channel roles
within each group are exchanged. Conjugation is applied during packing.
"""
struct TransposedWeights{T, N, A <: AbstractArray{T, N}, S, P} <: AbstractArray{T, N}
    w::A
    g::ConvGeometry{N, S, P}
end
function Base.size(t::TransposedWeights{T, N}) where {T, N}
    g = t.g
    return (ntuple(i -> g.wsize[i], Val(N - 2))..., channels_out(g) ÷ g.groups, channels_in(g))
end
Base.IndexStyle(::Type{<:TransposedWeights}) = IndexCartesian()
@inline function Base.getindex(t::TransposedWeights{T, N}, I::Vararg{Int, N}) where {T, N}
    g = t.g
    G = g.groups
    cin_g = channels_in(g) ÷ G
    cout_g = channels_out(g) ÷ G
    co_l = I[N - 1]
    ci = I[N]
    grp = (ci - 1) ÷ cin_g
    ci_l = ci - grp * cin_g
    co = grp * cout_g + co_l
    return @inbounds t.w[ntuple(i -> I[i], Val(N - 2))..., ci_l, co]
end

"""
    transposed_geometry(g) -> ConvGeometry

Geometry of the stride-1 convolution that maps the stride-stuffed output
gradient to the input gradient.
"""
function transposed_geometry(g::ConvGeometry{N, S}) where {N, S}
    O = ntuple(i -> g.ysize[i], Val(S))
    I = ntuple(i -> g.xsize[i], Val(S))
    lo = pad_lo(g)
    stuffed = ntuple(i -> (O[i] - 1) * g.stride[i] + 1, Val(S))
    xsize_t = (stuffed..., channels_out(g), batch_size(g))
    wsize_t = (ntuple(i -> g.wsize[i], Val(S))..., channels_out(g) ÷ g.groups, channels_in(g))
    pad_t = ntuple(Val(2S)) do j
        i = (j + 1) ÷ 2
        if isodd(j)
            (g.wsize[i] - 1) * g.dilation[i] - lo[i]
        else
            I[i] - stuffed[i] + lo[i]
        end
    end
    return ConvGeometry(xsize_t, wsize_t; stride = 1, pad = pad_t, dilation = g.dilation, groups = g.groups, flipped = !g.flipped)
end

"""
    GradState

Buffers for the gradient computations, created lazily per plan.
"""
struct GradState{Tc, MRc, NRc, DP}
    data_plan::DP                     # ConvPlan for the transposed convolution (same type as the parent)
    ybufs::Vector{Vector{Tc}}         # packed ȳ tiles, one per task
    wpartials::Vector{Vector{Tc}}     # per-task partial weight gradients (NP planes)
end

# The transposed plan shares every type parameter with the parent except the
# stencil descriptor, so the state's type is not known from the parent's type.
# Callers extract the concretely typed fields they need (see ∇conv_filter!).
function grad_state(p::ConvPlan{T, Tc, N, S, P, V, MR, NR, NP}) where {T, Tc, N, S, P, V, MR, NR, NP}
    st = p.grad.state
    st === nothing || return st
    return Base.@lock p.lock begin
        st2 = p.grad.state
        if st2 === nothing
            g = p.geom
            gt = transposed_geometry(g)
            data_plan = ConvPlan(T, gt; nthreads = p.nthreads, cache = p.cache, kernel = p.kernel, executor = p.executor)
            cout_g = channels_out(g) ÷ g.groups
            _, ycostride = _ybuf_geometry(p)
            ylen = ycostride * cout_g * NP + V     # slack for flat-mode reads past the end
            ybufs = [zeros(Tc, ylen) for _ in 1:p.nthreads]
            wpartials = [zeros(Tc, NP * prod(g.wsize)) for _ in 1:p.nthreads]
            MRc, NRc = filter_register_tile(Tc, NP)
            st2 = GradState{Tc, MRc, NRc, typeof(data_plan)}(data_plan, ybufs, wpartials)
            p.grad.state = st2
        end
        st2
    end
end

"""
    ∇conv_data!(x̄, ȳ, w, plan::ConvPlan; accumulate = false)

Gradient of `conv(x, w)` with respect to `x`, given the output gradient `ȳ`;
writes (or with `accumulate = true`, adds) into `x̄`. For complex data this is
the adjoint map, i.e. the weights are conjugated.
"""
function ∇conv_data!(x̄::AbstractArray{T, N}, ȳ::AbstractArray{T, N}, w::AbstractArray{T, N}, p::ConvPlan{T, Tc, N}; accumulate::Bool = false) where {T, Tc, N}
    check_conv_args(p.geom, ȳ, x̄, w)
    _check_strided(x̄, "x̄")
    _check_strided(ȳ, "ȳ")
    st = grad_state(p)
    _∇conv_data_impl!(x̄, ȳ, w, p, st.data_plan, accumulate)
    return x̄
end

# Function barrier: the transposed plan's stencil type parameter is not
# known statically from the parent plan's type.
function _∇conv_data_impl!(x̄, ȳ, w, p::ConvPlan, pt::ConvPlan, accumulate::Bool)
    @assert output_size(pt) == size(x̄) && input_size(pt)[end - 1] == size(ȳ)[end - 1]
    iv = InputView(ȳ, p.geom.stride)
    _conv_impl!(x̄, iv, TransposedWeights(w, p.geom), pt, nothing, identity, true, accumulate)
    return x̄
end

"""
    ∇conv_filter!(w̄, x, ȳ, plan::ConvPlan; accumulate = false)

Gradient of `conv(x, w)` with respect to `w`, given the output gradient `ȳ`;
writes (or with `accumulate = true`, adds) into `w̄`. For complex data the
input is conjugated.
"""
function ∇conv_filter!(w̄::AbstractArray{T, N}, x::AbstractArray{T, N}, ȳ::AbstractArray{T, N}, p::ConvPlan{T, Tc, N}; accumulate::Bool = false) where {T, Tc, N}
    check_conv_args(p.geom, ȳ, x, w̄)
    _check_strided(x, "x")
    _check_strided(ȳ, "ȳ")
    st = grad_state(p)
    # One dynamic dispatch (the state's type is not known statically); the
    # implementation below is fully typed.
    _∇conv_filter_impl!(w̄, x, ȳ, p, st, accumulate)
    return w̄
end

function _∇conv_filter_impl!(
        w̄::AbstractArray{T, N}, x::AbstractArray{T, N}, ȳ::AbstractArray{T, N},
        p::ConvPlan{T, Tc, N}, st::GradState{Tc, MRc, NRc}, accumulate::Bool
    ) where {T, Tc, N, MRc, NRc}
    S = N - 2
    iv = InputView(x, ntuple(_ -> 1, Val(S)))
    nitems = batch_size(p.geom) * prod(p.nblocks)
    ntasks = max(1, min(p.nthreads, nitems))
    ybufs = st.ybufs
    wpartials = st.wpartials
    for t in 1:ntasks
        fill!(wpartials[t], zero(Tc))
    end
    run_tasks(nitems, p.nthreads, p.executor) do task, items
        for item in items
            _filter_work_item!(ybufs, wpartials, Val(MRc), Val(NRc), iv, ȳ, p, task, item)
        end
    end
    _reduce_partials!(w̄, wpartials, ntasks, p, accumulate)
    return w̄
end

"""
    ∇conv_data(ȳ, w, plan)

Allocating version of [`∇conv_data!`](@ref): the gradient of `conv(x, w)`
with respect to `x`.
"""
∇conv_data(ȳ::AbstractArray{T, N}, w::AbstractArray{T, N}, p::ConvPlan{T, Tc, N}) where {T, Tc, N} =
    ∇conv_data!(similar(ȳ, T, input_size(p)), ȳ, w, p)

"""
    ∇conv_filter(x, ȳ, plan)

Allocating version of [`∇conv_filter!`](@ref): the gradient of `conv(x, w)`
with respect to `w`.
"""
∇conv_filter(x::AbstractArray{T, N}, ȳ::AbstractArray{T, N}, p::ConvPlan{T, Tc, N}) where {T, Tc, N} =
    ∇conv_filter!(similar(x, T, kernel_size(p)), x, ȳ, p)

# Column-major strides of an array with the given size.
@inline function _dims_strides(dims::NTuple{N, Int}) where {N}
    return ntuple(Val(N)) do i
        s = 1
        for j in 1:(i - 1)
            s *= dims[j]
        end
        s
    end
end

"""
    pack_output_tile!(Yp, ȳ, b, origin, te, rowstr, costride, cout_g, grp, ::Val{NP})

Pack the output-gradient tile for group `grp` into the per-task buffer with
row strides `rowstr` and channel stride `costride` (the geometry returned by
`_ybuf_geometry`), zero everywhere outside the valid region so that lanes past
the tile edge contribute nothing.
"""
function pack_output_tile!(
        Yp::Vector{Tc}, ȳ::AbstractArray{T, N}, b::Int, origin::NTuple{S, Int},
        te::NTuple{S, Int}, rowstr, costride::Int, cout_g::Int, grp::Int, ::Val{NP}
    ) where {Tc, T, N, S, NP}
    rows = CartesianIndices(ntuple(i -> te[i + 1], Val(S - 1)))
    planestride = costride * cout_g
    fill!(Yp, zero(Tc))
    @inbounds for co_l in 1:cout_g
        co = (grp - 1) * cout_g + co_l
        cbase = (co_l - 1) * costride
        for r in rows
            rbase = cbase
            for i in 1:(S - 1)
                rbase += (r[i] - 1) * rowstr[i]
            end
            for wo in 1:te[1]
                v = ȳ[origin[1] + wo, ntuple(i -> origin[i + 1] + r[i], Val(S - 1))..., co, b]
                _store_planes!(Yp, rbase + wo, planestride, v, Val(NP))
            end
        end
    end
    return Yp
end

function _filter_work_item!(
        ybufs::Vector{Vector{Tc}}, wpartials::Vector{Vector{Tc}}, ::Val{MRc}, ::Val{NRc}, iv::InputView, ȳ::AbstractArray{T, N},
        p::ConvPlan{T, Tc, N, S, P, V, MR, NR, NP, SIMD}, task::Int, item::Int
    ) where {Tc, MRc, NRc, T, N, S, P, V, MR, NR, NP, SIMD}
    g = p.geom
    G = g.groups
    cin_g = channels_in(g) ÷ G
    cout_g = channels_out(g) ÷ G
    K = length(p.taps)
    Kc = p.Kc
    ncb = cld(cin_g, Kc)
    b, origin = decode_item(p, item)
    O = ntuple(i -> g.ysize[i], Val(S))
    te = ntuple(i -> min(p.tile[i], O[i] - origin[i]), Val(S))
    Xp = p.xbufs[task]
    Yp = ybufs[task]
    out = wpartials[task]
    nvec = cld(te[1], V)
    # Packed x geometry.
    xrowstr = _packed_rowstrides(p)                 # per output-row-dim stride (before × stride)
    # Packed ȳ geometry (row mode: [Lpy, tile[2:end]...]; flat mode: packed-tile geometry).
    yrowstr, y_costride = _ybuf_geometry(p)
    y_planestride = y_costride * cout_g
    # Flat mode: the whole tile is one vector of length F.
    flatlen = 1
    for i in 1:S
        flatlen += (te[i] - 1) * (i == 1 ? 1 : xrowstr[i - 1])
    end
    # Partial gradient layout: natural w layout, NP planes.
    wlen = prod(g.wsize)
    wstr = _dims_strides(g.wsize)
    kspatial = CartesianIndices(ntuple(i -> g.wsize[i], Val(S)))
    # Rows beyond the first row dimension are iterated outside the kernel.
    outer = CartesianIndices(ntuple(i -> te[i + 2], Val(max(S - 2, 0))))
    nrows = S >= 2 ? te[2] : 1
    x_row_stride = S >= 2 ? g.stride[2] * xrowstr[1] : 0
    y_row_stride = S >= 2 ? yrowstr[1] : 0
    # Row chunking so that the x and ȳ rows of one register tile stay in L1.
    sz = _sz(Tc)
    chunk = clamp(fld(p.cache.l1 ÷ 2 ÷ sz, NP * (MRc * p.Lp + NRc * p.Lpy)), 1, nrows)
    chunk = max(chunk, min(nrows, cld(32, max(nvec, 1))))
    # Flat mode chunks the vector range instead of rows.
    fchunk = max(V, (clamp(fld(p.cache.l1 ÷ 2 ÷ sz, NP * (MRc + NRc)), V, max(flatlen, V)) ÷ V) * V)
    for grp in 1:G
        pack_output_tile!(Yp, ȳ, b, origin, te, yrowstr, y_costride, cout_g, grp, Val(NP))
        for cb in 1:ncb
            ci0 = (grp - 1) * cin_g + (cb - 1) * Kc
            kc = min(Kc, cin_g - (cb - 1) * Kc)
            pack_input_tile!(Xp, iv, g, b, ci0, kc, origin, p.Lp, p.W1, p.R, p.xci_stride, p.xplane_stride, Val(NP), true)
            for cit in 1:cld(kc, MRc)
                ci_l0 = (cit - 1) * MRc
                mrc = min(MRc, kc - ci_l0)
                for cot in 1:cld(cout_g, NRc)
                    co0 = (cot - 1) * NRc
                    nrc = min(NRc, cout_g - co0)
                    if p.flat
                        f0 = 0
                        while f0 < flatlen
                            nv = min(cld(flatlen - f0, V), fchunk ÷ V)
                            for (t, k) in enumerate(kspatial)
                                obase = 0
                                for i in 1:S
                                    obase += (k[i] - 1) * wstr[i]
                                end
                                obase += ((cb - 1) * Kc + ci_l0) * wstr[N - 1] + ((grp - 1) * cout_g + co0) * wstr[N]
                                dispatch_filter_microkernel!(
                                    Val(SIMD), Val(V), Val(MRc), Val(NRc), Val(NP), mrc, nrc,
                                    out, obase, wstr[N - 1], wstr[N], wlen,
                                    Xp, ci_l0 * p.xci_stride + f0 + p.taps[t], p.xci_stride, p.xplane_stride, 0,
                                    Yp, co0 * y_costride + f0, y_costride, y_planestride, 0,
                                    1, nv
                                )
                            end
                            f0 += nv * V
                        end
                        continue
                    end
                    for od in outer
                        xouter = 0
                        youter = 0
                        for i in 1:(S - 2)
                            xouter += (od[i] - 1) * g.stride[i + 2] * xrowstr[i + 1]
                            youter += (od[i] - 1) * yrowstr[i + 1]
                        end
                        r0 = 0
                        while r0 < nrows
                            nr_ = min(chunk, nrows - r0)
                            for (t, k) in enumerate(kspatial)
                                obase = 0
                                for i in 1:S
                                    obase += (k[i] - 1) * wstr[i]
                                end
                                obase += ((cb - 1) * Kc + ci_l0) * wstr[N - 1] + ((grp - 1) * cout_g + co0) * wstr[N]
                                xbase = ci_l0 * p.xci_stride + xouter + r0 * x_row_stride + p.taps[t]
                                ybase = co0 * y_costride + youter + r0 * y_row_stride
                                dispatch_filter_microkernel!(
                                    Val(SIMD), Val(V), Val(MRc), Val(NRc), Val(NP), mrc, nrc,
                                    out, obase, wstr[N - 1], wstr[N], wlen,
                                    Xp, xbase, p.xci_stride, p.xplane_stride, x_row_stride,
                                    Yp, ybase, y_costride, y_planestride, y_row_stride,
                                    nr_, nvec
                                )
                            end
                            r0 += nr_
                        end
                    end
                end
            end
        end
    end
    return nothing
end

function _reduce_partials!(w̄::AbstractArray{T, N}, partials::Vector{Vector{Tc}}, ntasks::Int, p::ConvPlan{T, Tc, N, S, P, V, MR, NR, NP}, accumulate::Bool) where {T, Tc, N, S, P, V, MR, NR, NP}
    wlen = length(w̄)
    acc = partials[1]
    @inbounds for t in 2:ntasks
        pt = partials[t]
        for i in eachindex(acc)
            acc[i] += pt[i]
        end
    end
    @inbounds for i in 1:wlen
        v = _recombine(T, acc, i, wlen, Val(NP))
        if accumulate
            w̄[i] += v
        else
            w̄[i] = v
        end
    end
    return w̄
end
