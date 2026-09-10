# Reduction microkernel for the weight gradient.
#
# For one kernel tap, an `MRc × NRc` tile of (input channel, output channel)
# pairs, and `nrows` rows of `nvec` vectors each, it accumulates
#     acc[i, j, p] += Σ_rows Σ_lanes Xp[i-th channel, tap-shifted row] * Ȳ[j-th channel, row]
# plane-wise, then adds the horizontal sums into `out`. Loads are shared: one
# vector of x per input channel and one of ȳ per output channel feed
# `MRc*NRc` FMAs, with no broadcasts.

_yv(j, p) = Symbol("y_", j, "_", p)

@generated function filter_microkernel!(
        ::Type{Vec{V, Tc}}, ::Val{MRc}, ::Val{NRc}, ::Val{NP},
        out::Ptr{Tc}, out_ci_stride::Int, out_co_stride::Int, out_plane_stride::Int,
        xptr::Ptr{Tc}, x_ci_stride::Int, x_plane_stride::Int, x_row_stride::Int,
        yptr::Ptr{Tc}, y_co_stride::Int, y_plane_stride::Int, y_row_stride::Int,
        nrows::Int, nvec::Int
    ) where {V, Tc, MRc, NRc, NP}
    VT = Vec{V, Tc}
    sz = sizeof(Tc)
    init = [:($(_acc(i, j, p)) = zero($VT)) for p in 1:NP for j in 1:NRc for i in 1:MRc]
    xloads = [:($(_xv(i, p)) = vload($VT, xv + ($((i - 1)) * x_ci_stride + $((p - 1)) * x_plane_stride) * $sz)) for p in 1:NP for i in 1:MRc]
    yloads = [:($(_yv(j, p)) = vload($VT, yv + ($((j - 1)) * y_co_stride + $((p - 1)) * y_plane_stride) * $sz)) for p in 1:NP for j in 1:NRc]
    fmas = [:($(_acc(i, j, p)) = muladd($(_xv(i, p)), $(_yv(j, p)), $(_acc(i, j, p)))) for p in 1:NP for j in 1:NRc for i in 1:MRc]
    stores = Expr[]
    for p in 1:NP, j in 1:NRc, i in 1:MRc
        o = :(out + ($((i - 1)) * out_ci_stride + $((j - 1)) * out_co_stride + $((p - 1)) * out_plane_stride) * $sz)
        push!(stores, :(unsafe_store!($o, unsafe_load($o) + sum($(_acc(i, j, p))))))
    end
    return quote
        $(init...)
        @inbounds for r in 0:(nrows - 1)
            xr = xptr + r * x_row_stride * $sz
            yr = yptr + r * y_row_stride * $sz
            for v in 0:(nvec - 1)
                xv = xr + v * $(V * sz)
                yv = yr + v * $(V * sz)
                $(xloads...)
                $(yloads...)
                $(fmas...)
            end
        end
        $(stores...)
        return nothing
    end
end

@generated function filter_microkernel_scalar!(
        ::Val{MRc}, ::Val{NRc}, ::Val{NP},
        out::Vector{Tc}, obase::Int, out_ci_stride::Int, out_co_stride::Int, out_plane_stride::Int,
        Xp::Vector{Tc}, xbase::Int, x_ci_stride::Int, x_plane_stride::Int, x_row_stride::Int,
        Yp::Vector{Tc}, ybase::Int, y_co_stride::Int, y_plane_stride::Int, y_row_stride::Int,
        nrows::Int, nvec::Int
    ) where {Tc, MRc, NRc, NP}
    init = [:($(_acc(i, j, p)) = zero(Tc)) for p in 1:NP for j in 1:NRc for i in 1:MRc]
    xloads = [:($(_xv(i, p)) = Xp[xv + $((i - 1)) * x_ci_stride + $((p - 1)) * x_plane_stride]) for p in 1:NP for i in 1:MRc]
    yloads = [:($(_yv(j, p)) = Yp[yv + $((j - 1)) * y_co_stride + $((p - 1)) * y_plane_stride]) for p in 1:NP for j in 1:NRc]
    fmas = [:($(_acc(i, j, p)) = muladd($(_xv(i, p)), $(_yv(j, p)), $(_acc(i, j, p)))) for p in 1:NP for j in 1:NRc for i in 1:MRc]
    stores = Expr[]
    for p in 1:NP, j in 1:NRc, i in 1:MRc
        o = :(obase + $((i - 1)) * out_ci_stride + $((j - 1)) * out_co_stride + $((p - 1)) * out_plane_stride + 1)
        push!(stores, :(out[$o] += $(_acc(i, j, p))))
    end
    return quote
        $(init...)
        @inbounds for r in 0:(nrows - 1)
            xr = xbase + r * x_row_stride
            yr = ybase + r * y_row_stride
            for v in 1:nvec
                xv = xr + v
                yv = yr + v
                $(xloads...)
                $(yloads...)
                $(fmas...)
            end
        end
        $(stores...)
        return nothing
    end
end

@generated function dispatch_filter_microkernel!(
        ::Val{SIMD}, ::Val{V}, ::Val{MRc}, ::Val{NRc}, ::Val{NP}, mrc::Int, nrc::Int,
        out::Vector{Tc}, obase::Int, out_ci_stride::Int, out_co_stride::Int, out_plane_stride::Int,
        Xp::Vector{Tc}, xbase::Int, x_ci_stride::Int, x_plane_stride::Int, x_row_stride::Int,
        Yp::Vector{Tc}, ybase::Int, y_co_stride::Int, y_plane_stride::Int, y_row_stride::Int,
        nrows::Int, nvec::Int
    ) where {SIMD, Tc, V, MRc, NRc, NP}
    function call(m, n)
        if SIMD
            return :(
                filter_microkernel!(
                    Vec{$V, Tc}, Val($m), Val($n), Val($NP),
                    optr, out_ci_stride, out_co_stride, out_plane_stride,
                    xptr, x_ci_stride, x_plane_stride, x_row_stride,
                    yptr, y_co_stride, y_plane_stride, y_row_stride, nrows, nvec
                )
            )
        else
            return :(
                filter_microkernel_scalar!(
                    Val($m), Val($n), Val($NP),
                    out, obase, out_ci_stride, out_co_stride, out_plane_stride,
                    Xp, xbase, x_ci_stride, x_plane_stride, x_row_stride,
                    Yp, ybase, y_co_stride, y_plane_stride, y_row_stride, nrows, nvec
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
    body = chain(:mrc, 1:MRc, m -> chain(:nrc, 1:NRc, n -> call(m, n)))
    setup = if SIMD
        quote
            sz = sizeof(Tc)
            optr = pointer(out) + obase * sz
            xptr = pointer(Xp) + xbase * sz
            yptr = pointer(Yp) + ybase * sz
        end
    else
        :(nothing)
    end
    return quote
        $setup
        GC.@preserve out Xp Yp begin
            $body
        end
        return nothing
    end
end

"""
    filter_register_tile(Tc, nplanes) -> (MRc, NRc)

Register tile of the weight-gradient kernel: `MRc` input channels × `NRc`
output channels, each a vector along the width, with `nplanes` planes.
"""
function filter_register_tile(::Type{Tc}, np::Int) where {Tc}
    simd_type(Tc) || return (4, 4)
    rc = Int(known(register_count()))::Int
    if np == 1
        rc >= 32 && return (4, 5)
        rc >= 16 && return (2, 4)
        return (1, 3)
    else
        rc >= 32 && return (1, 4)
        rc >= 16 && return (1, 2)
        return (1, 1)
    end
end
