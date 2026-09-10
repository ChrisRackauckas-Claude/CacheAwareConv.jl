# The register-tiled microkernel.
#
# One call computes an `MR*V × NR` block of outputs (V lanes along the output
# width, NR output channels) for `NP` planes, summing over `Kc` input channels
# and `K` kernel taps. Inputs come from the packed tile, weights from the packed
# weight block, and results are stored to the destination (either the output
# array itself or a per-thread accumulation buffer), accumulating onto existing
# values when `ACC`. The last of the `MR` vectors is masked when `MASKED`
# (partial tile along the width).
#
# For complex data the three planes (re, im, re+im) are multiplied plane-wise
# (Gauss' 3-multiplication trick); `finalize_tile!` recombines them.
#
# All offsets passed to the drivers are 0-based element offsets; the SIMD
# kernel converts them to byte pointers, the scalar kernel to 1-based indices.

_acc(i, j, p) = Symbol("acc_", i, "_", j, "_", p)
_xv(i, p) = Symbol("x_", i, "_", p)

@generated function conv_microkernel!(
        ::Type{Vec{V, Tc}}, ::Val{MR}, ::Val{NR}, ::Val{NP}, ::Val{ACC}, ::Val{MASKED},
        yptr::Ptr{Tc}, y_co_stride::Int, y_plane_stride::Int,
        xptr::Ptr{Tc}, x_ci_stride::Int, x_plane_stride::Int,
        wptr::Ptr{Tc}, taps::Ptr{Int}, K::Int, Kc::Int, mask::Vec{V, Bool}, nrep::Int
    ) where {V, Tc, MR, NR, NP, ACC, MASKED}
    VT = Vec{V, Tc}
    sz = sizeof(Tc)
    yoff(i, j, p) = :(yr + (($(j - 1)) * y_co_stride + $(p - 1) * y_plane_stride + $((i - 1) * V)) * $sz)
    init = Expr[]
    for p in 1:NP, j in 1:NR, i in 1:MR
        if ACC
            if MASKED && i == MR
                push!(init, :($(_acc(i, j, p)) = vload($VT, $(yoff(i, j, p)), mask)))
            else
                push!(init, :($(_acc(i, j, p)) = vload($VT, $(yoff(i, j, p)))))
            end
        else
            push!(init, :($(_acc(i, j, p)) = zero($VT)))
        end
    end
    loads = Expr[]
    for p in 1:NP, i in 1:MR
        push!(loads, :($(_xv(i, p)) = vload($VT, xk + ($((p - 1)) * x_plane_stride + $((i - 1) * V)) * $sz)))
    end
    fmas = Expr[]
    for j in 1:NR, p in 1:NP
        push!(fmas, :(wv = $VT(unsafe_load(wk, $((p - 1) * NR + j)))))
        for i in 1:MR
            push!(fmas, :($(_acc(i, j, p)) = muladd($(_xv(i, p)), wv, $(_acc(i, j, p)))))
        end
    end
    stores = Expr[]
    for p in 1:NP, j in 1:NR, i in 1:MR
        if MASKED && i == MR
            push!(stores, :(vstore($(_acc(i, j, p)), $(yoff(i, j, p)), mask)))
        else
            push!(stores, :(vstore($(_acc(i, j, p)), $(yoff(i, j, p)))))
        end
    end
    # `nrep` consecutive tiles along the width per call: amortises the call,
    # tap-offset loads and accumulator setup when `K * Kc` is small.
    return quote
        @inbounds for rep in 0:(nrep - 1)
            yr = yptr + rep * $(MR * V * sz)
            xr = xptr + rep * $(MR * V * sz)
            $(init...)
            for ci in 0:(Kc - 1)
                xc = xr + ci * x_ci_stride * $sz
                wc = wptr + ci * K * $(NR * NP) * $sz
                for t in 0:(K - 1)
                    xk = xc + unsafe_load(taps, t + 1) * $sz
                    wk = wc + t * $(NR * NP) * $sz
                    $(loads...)
                    $(fmas...)
                end
            end
            $(stores...)
        end
        return nothing
    end
end

# Stencil variant: a single input channel, a single output channel, one plane,
# and `K` known at compile time so that the `K` weight broadcasts and tap
# offsets are loaded once per call and stay in registers across all `nrep`
# tiles. Used when `Kc == 1 && NR == 1 && NP == 1 && K <= MAX_HOISTED_TAPS`;
# for more than `HOISTED_FULL_MR_TAPS` taps the tile is narrowed so that the
# weights, accumulators and one input vector still fit the register file.
const MAX_HOISTED_TAPS = 27
const HOISTED_FULL_MR_TAPS = 12

"""
    hoisted_mr(MR, K)

Width (in vectors) of the hoisted stencil kernel's tile for `K` taps.
"""
hoisted_mr(MR::Int, K::Int) = K <= HOISTED_FULL_MR_TAPS ? MR : max(1, min(MR, 4))

@generated function conv_microkernel_hoisted!(
        ::Type{Vec{V, Tc}}, ::Val{MR}, ::Val{K}, ::Val{ACC},
        yptr::Ptr{Tc}, xptr::Ptr{Tc}, wptr::Ptr{Tc}, taps::Ptr{Int}, nrep::Int
    ) where {V, Tc, MR, K, ACC}
    VT = Vec{V, Tc}
    sz = sizeof(Tc)
    wl = [:($(Symbol("w_", t)) = $VT(unsafe_load(wptr, $t))) for t in 1:K]
    tl = [:($(Symbol("o_", t)) = unsafe_load(taps, $t) * $sz) for t in 1:K]
    init = [ACC ? :($(_acc(i, 1, 1)) = vload($VT, yr + $((i - 1) * V * sz))) : :($(_acc(i, 1, 1)) = zero($VT)) for i in 1:MR]
    body = Expr[]
    for t in 1:K, i in 1:MR
        push!(body, :($(_acc(i, 1, 1)) = muladd(vload($VT, xr + $(Symbol("o_", t)) + $((i - 1) * V * sz)), $(Symbol("w_", t)), $(_acc(i, 1, 1)))))
    end
    stores = [:(vstore($(_acc(i, 1, 1)), yr + $((i - 1) * V * sz))) for i in 1:MR]
    return quote
        $(wl...)
        $(tl...)
        @inbounds for rep in 0:(nrep - 1)
            yr = yptr + rep * $(MR * V * sz)
            xr = xptr + rep * $(MR * V * sz)
            $(init...)
            $(body...)
            $(stores...)
        end
        return nothing
    end
end

@generated function dispatch_hoisted!(
        ::Val{V}, ::Val{MR}, acc::Bool, K::Int,
        y::AbstractArray{Tc}, ybase::Int, Xp::Vector{Tc}, xbase::Int, Wp::Vector{Tc}, wbase::Int, taps::Vector{Int}, nrep::Int
    ) where {Tc, V, MR}
    ex = :(error("unreachable"))
    for k in MAX_HOISTED_TAPS:-1:1
        mrk = hoisted_mr(MR, k)
        ex = :(
            if K == $k
                if acc
                    conv_microkernel_hoisted!(Vec{$V, Tc}, Val($mrk), Val($k), Val(true), yptr, xptr, wptr, tptr, nrep)
                else
                    conv_microkernel_hoisted!(Vec{$V, Tc}, Val($mrk), Val($k), Val(false), yptr, xptr, wptr, tptr, nrep)
                end
            else
                $ex
            end
        )
    end
    return quote
        sz = sizeof(Tc)
        yptr = pointer(y) + ybase * sz
        xptr = pointer(Xp) + xbase * sz
        wptr = pointer(Wp) + wbase * sz
        tptr = pointer(taps)
        GC.@preserve y Xp Wp taps begin
            $ex
        end
        return nothing
    end
end

# Scalar fallback with the same contract, for element types SIMD.jl cannot
# vectorise (BigFloat, dual numbers, ...). `V == 1`, so the tile is MR × NR
# scalars, and buffers are addressed as arrays with 1-based indices.
@generated function conv_microkernel_scalar!(
        ::Val{MR}, ::Val{NR}, ::Val{NP}, ::Val{ACC},
        y::AbstractArray{Tc}, ybase::Int, y_co_stride::Int, y_plane_stride::Int,
        Xp::Vector{Tc}, xbase::Int, x_ci_stride::Int, x_plane_stride::Int,
        Wp::Vector{Tc}, wbase::Int, taps::Vector{Int}, K::Int, Kc::Int, nrep::Int
    ) where {Tc, MR, NR, NP, ACC}
    yidx(i, j, p) = :(yb + ($(j - 1)) * y_co_stride + $(p - 1) * y_plane_stride + $(i))
    init = Expr[]
    for p in 1:NP, j in 1:NR, i in 1:MR
        if ACC
            push!(init, :($(_acc(i, j, p)) = y[$(yidx(i, j, p))]))
        else
            push!(init, :($(_acc(i, j, p)) = zero(Tc)))
        end
    end
    loads = Expr[]
    for p in 1:NP, i in 1:MR
        push!(loads, :($(_xv(i, p)) = Xp[xk + $((p - 1)) * x_plane_stride + $(i)]))
    end
    fmas = Expr[]
    for j in 1:NR, p in 1:NP
        push!(fmas, :(wv = Wp[wk + $((p - 1) * NR + j)]))
        for i in 1:MR
            push!(fmas, :($(_acc(i, j, p)) = muladd($(_xv(i, p)), wv, $(_acc(i, j, p)))))
        end
    end
    stores = Expr[]
    for p in 1:NP, j in 1:NR, i in 1:MR
        push!(stores, :(y[$(yidx(i, j, p))] = $(_acc(i, j, p))))
    end
    return quote
        @inbounds for rep in 0:(nrep - 1)
            yb = ybase + rep * $MR
            xb = xbase + rep * $MR
            $(init...)
            for ci in 0:(Kc - 1)
                xc = xb + ci * x_ci_stride
                wc = wbase + ci * K * $(NR * NP)
                for t in 0:(K - 1)
                    xk = xc + taps[t + 1]
                    wk = wc + t * $(NR * NP)
                    $(loads...)
                    $(fmas...)
                end
            end
            $(stores...)
        end
        return nothing
    end
end

"""
    dispatch_microkernel!(::Val{SIMD}, ::Val{V}, ::Val{MR}, ::Val{NR}, ::Val{NP},
                          acc, mr, nr, masked, mask,
                          y, ybase, y_co_stride, y_plane_stride,
                          Xp, xbase, x_ci_stride, x_plane_stride,
                          Wp, wbase, taps, K, Kc, nrep)

Run the microkernel instantiation for `nrep` consecutive tiles of `mr ≤ MR`
vectors and `nr ≤ NR` channels (a partial tile is always a single repetition).
`ybase`, `xbase`, `wbase` are 0-based element offsets into `y`, `Xp`, `Wp`.
"""
@generated function dispatch_microkernel!(
        ::Val{SIMD}, ::Val{V}, ::Val{MR}, ::Val{NR}, ::Val{NP},
        acc::Bool, mr::Int, nr::Int, masked::Bool, mask::Vec{V, Bool},
        y::AbstractArray{Tc}, ybase::Int, y_co_stride::Int, y_plane_stride::Int,
        Xp::Vector{Tc}, xbase::Int, x_ci_stride::Int, x_plane_stride::Int,
        Wp::Vector{Tc}, wbase::Int, taps::Vector{Int}, K::Int, Kc::Int, nrep::Int
    ) where {SIMD, Tc, V, MR, NR, NP}
    function call(mrv, nrv, accv, maskedv)
        if SIMD
            return :(
                conv_microkernel!(
                    Vec{$V, Tc}, Val($mrv), Val($nrv), Val($NP), Val($accv), Val($maskedv),
                    yptr, y_co_stride, y_plane_stride, xptr, x_ci_stride, x_plane_stride,
                    wptr, tptr, K, Kc, mask, nrep
                )
            )
        else
            return :(
                conv_microkernel_scalar!(
                    Val($mrv), Val($nrv), Val($NP), Val($accv),
                    y, ybase, y_co_stride, y_plane_stride, Xp, xbase, x_ci_stride, x_plane_stride,
                    Wp, wbase, taps, K, Kc, nrep
                )
            )
        end
    end
    function chain(var, vals, body)
        ex = :(error("unreachable"))
        for v in reverse(vals)
            ex = :(
                if $var == $v
                    $(body(v))
                else
                    $ex
                end
            )
        end
        return ex
    end
    body = chain(
        :mr, 1:MR, mrv -> chain(
            :nr, 1:NR, nrv -> quote
                if acc
                    if masked
                        $(call(mrv, nrv, true, true))
                    else
                        $(call(mrv, nrv, true, false))
                    end
                else
                    if masked
                        $(call(mrv, nrv, false, true))
                    else
                        $(call(mrv, nrv, false, false))
                    end
                end
            end
        )
    )
    setup = if SIMD
        quote
            sz = sizeof(Tc)
            yptr = pointer(y) + ybase * sz
            xptr = pointer(Xp) + xbase * sz
            wptr = pointer(Wp) + wbase * sz
            tptr = pointer(taps)
        end
    else
        :(nothing)
    end
    return quote
        $setup
        GC.@preserve y Xp Wp taps begin
            $body
        end
        return nothing
    end
end

"""
    lane_mask(::Val{V}, n) -> Vec{V, Bool}

Mask with the first `n` lanes true.
"""
@inline function lane_mask(::Val{V}, n::Int) where {V}
    return Vec{V, Int32}(ntuple(i -> Int32(i - 1), Val(V))) < Vec{V, Int32}(Int32(n))
end
