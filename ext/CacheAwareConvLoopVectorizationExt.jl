module CacheAwareConvLoopVectorizationExt

using CacheAwareConv
using CacheAwareConv: ConvPlan
using LoopVectorization: @turbo

# @turbo replacement for the register-tiled microkernel, reached through
# `_tile_row!` when the plan was built with `kernel = :lv`. One call computes
# `len` consecutive output positions for `nr` output channels of one plane:
# LoopVectorization picks the vector width, register tile and tail handling
# itself, so there is no (MR, NR, V) dispatch and no masking. `wpn` is the
# packed-weight stride per tap (`nr * NP`); the `j` term folds the +1 of
# 1-based indexing, as does `l`. Loop-invariant strides must be hoisted into
# locals (`xc`, `wc`): LoopVectorization's index-expression parser only
# accepts affine expressions in the loop variables.
function _lv_plane!(
        ::Val{ACC}, nr::Int, len::Int,
        y::AbstractArray{Tc}, yo::Int, ycs::Int,
        Xp::Vector{Tc}, xo::Int, xcis::Int,
        Wp::Vector{Tc}, wo::Int, wpn::Int,
        taps::Vector{Int}, K::Int, kc::Int
    ) where {Tc, ACC}
    Kwpn = K * wpn
    if ACC
        @turbo for j in 1:nr, l in 1:len
            s = y[yo + (j - 1) * ycs + l]
            for ci in 0:(kc - 1)
                xc = xo + ci * xcis
                wc = wo + ci * Kwpn
                for t in 1:K
                    s += Xp[xc + taps[t] + l] * Wp[wc + (t - 1) * wpn + j]
                end
            end
            y[yo + (j - 1) * ycs + l] = s
        end
    else
        @turbo for j in 1:nr, l in 1:len
            s = zero(Tc)
            for ci in 0:(kc - 1)
                xc = xo + ci * xcis
                wc = wo + ci * Kwpn
                for t in 1:K
                    s += Xp[xc + taps[t] + l] * Wp[wc + (t - 1) * wpn + j]
                end
            end
            y[yo + (j - 1) * ycs + l] = s
        end
    end
    return nothing
end

function CacheAwareConv.lv_tile_row!(
        ::Val{NR}, ::Val{NP}, acc::Bool, nr::Int, len::Int,
        ydest::AbstractArray{Tc}, ybase::Int, ycs::Int, yps::Int,
        Xp::Vector{Tc}, xbase::Int, p::ConvPlan, wbase::Int, K::Int, kc::Int
    ) where {NR, NP, Tc}
    # LoopVectorization miscompiles `y[i]` linear indexing on N-d arrays —
    # give it a one-dimensional view of the destination.
    yd = ydest isa AbstractVector ? ydest : view(ydest, :)
    wpn = nr * NP
    for pp in 1:NP
        xo = xbase + (pp - 1) * p.xplane_stride
        yo = ybase + (pp - 1) * yps
        wo = wbase + (pp - 1) * nr
        _lv_plane!(Val(acc), nr, len, yd, yo, ycs, Xp, xo, p.xci_stride, p.Wp, wo, wpn, p.taps, K, kc)
    end
    return nothing
end

end
