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
        wptr::Ptr{Tc}, taps::Ptr{Int}, K::Int, Kc::Int, mask::Vec{V, Bool}
    ) where {V, Tc, MR, NR, NP, ACC, MASKED}
    VT = Vec{V, Tc}
    sz = sizeof(Tc)
    yoff(i, j, p) = :(yptr + (($(j - 1)) * y_co_stride + $(p - 1) * y_plane_stride + $((i - 1) * V)) * $sz)
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
    return quote
        $(init...)
        @inbounds for ci in 0:(Kc - 1)
            xc = xptr + ci * x_ci_stride * $sz
            wc = wptr + ci * K * $(NR * NP) * $sz
            for t in 0:(K - 1)
                xk = xc + unsafe_load(taps, t + 1) * $sz
                wk = wc + t * $(NR * NP) * $sz
                $(loads...)
                $(fmas...)
            end
        end
        $(stores...)
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
        Wp::Vector{Tc}, wbase::Int, taps::Vector{Int}, K::Int, Kc::Int
    ) where {Tc, MR, NR, NP, ACC}
    yidx(i, j, p) = :(ybase + ($(j - 1)) * y_co_stride + $(p - 1) * y_plane_stride + $(i))
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
        $(init...)
        @inbounds for ci in 0:(Kc - 1)
            xc = xbase + ci * x_ci_stride
            wc = wbase + ci * K * $(NR * NP)
            for t in 0:(K - 1)
                xk = xc + taps[t + 1]
                wk = wc + t * $(NR * NP)
                $(loads...)
                $(fmas...)
            end
        end
        $(stores...)
        return nothing
    end
end

"""
    dispatch_microkernel!(::Val{SIMD}, ::Val{V}, ::Val{MR}, ::Val{NR}, ::Val{NP},
                          acc, mr, nr, masked, mask,
                          y, ybase, y_co_stride, y_plane_stride,
                          Xp, xbase, x_ci_stride, x_plane_stride,
                          Wp, wbase, taps, K, Kc)

Run the microkernel instantiation for a possibly-partial tile of `mr ≤ MR`
vectors and `nr ≤ NR` channels. `ybase`, `xbase`, `wbase` are 0-based element
offsets into `y`, `Xp`, `Wp`.
"""
@generated function dispatch_microkernel!(
        ::Val{SIMD}, ::Val{V}, ::Val{MR}, ::Val{NR}, ::Val{NP},
        acc::Bool, mr::Int, nr::Int, masked::Bool, mask::Vec{V, Bool},
        y::AbstractArray{Tc}, ybase::Int, y_co_stride::Int, y_plane_stride::Int,
        Xp::Vector{Tc}, xbase::Int, x_ci_stride::Int, x_plane_stride::Int,
        Wp::Vector{Tc}, wbase::Int, taps::Vector{Int}, K::Int, Kc::Int
    ) where {SIMD, Tc, V, MR, NR, NP}
    function call(mrv, nrv, accv, maskedv)
        if SIMD
            return :(
                conv_microkernel!(
                    Vec{$V, Tc}, Val($mrv), Val($nrv), Val($NP), Val($accv), Val($maskedv),
                    yptr, y_co_stride, y_plane_stride, xptr, x_ci_stride, x_plane_stride,
                    wptr, tptr, K, Kc, mask
                )
            )
        else
            return :(
                conv_microkernel_scalar!(
                    Val($mrv), Val($nrv), Val($NP), Val($accv),
                    y, ybase, y_co_stride, y_plane_stride, Xp, xbase, x_ci_stride, x_plane_stride,
                    Wp, wbase, taps, K, Kc
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
