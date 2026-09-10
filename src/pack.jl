# Packing of input tiles and weights into the contiguous, zero-padded,
# stride-phase-deinterleaved layout read by the microkernel.
#
# Input tile layout (one work item, `Kc` channels, `NP` planes), all 0-based:
#   Xp[j + ph*Lp + W1*(r_2 + R_2*(r_3 + ...)) + xci_stride*ci + xplane_stride*p]
# where along the first spatial dimension the padded coordinate `q = j*s + ph`
# (`s` = stride, `ph` = phase) and `r_i` are padded coordinates along the other
# spatial dimensions relative to the tile's first input row. Positions outside
# the input are zero, and `j` runs to `Lp`, which is padded to a multiple of the
# vector width plus the maximum tap offset so the kernel never reads past the
# segment.

"""
    InputView(x, stuff)

Describes the array feeding a convolution: `x` itself, optionally *zero
stuffed* along each spatial dimension by `stuff` (element `i` of the stuffed
array is `x[(i-1)/stuff + 1]` when divisible and zero otherwise). Stuffing by
the forward stride turns the data-gradient into a stride-1 convolution.
"""
struct InputView{A <: AbstractArray, S}
    x::A
    stuff::NTuple{S, Int}
end

# Write plane `p` (1-based) of value `v` into `dst[i]`.
@inline plane_value(v::Real, ::Val{1}) = v
@inline plane_value(v::Complex, ::Val{1}) = real(v)
@inline plane_value(v::Complex, ::Val{2}) = imag(v)
@inline plane_value(v::Complex, ::Val{3}) = real(v) + imag(v)

"""
    pack_row!(dst, doff, Lp, s, ph, src, soff, sstride, I, lo, stuff, xplane_stride, ::Val{NP}, ::Type{Tc}, conj)

Fill one phase segment of one packed row: for `j in 0:Lp-1` the padded
coordinate `q = j*s + ph` maps to source index `q - lo` (0-based) in a
zero-stuffed source of logical length `(I-1)*stuff + 1`. `src[soff + m*sstride]`
is element `m` (0-based) of the source row; `conj` conjugates complex values.
"""
@inline function pack_row!(
        dst::Vector{Tc}, doff::Int, Lp::Int, s::Int, ph::Int,
        src::AbstractArray, soff::Int, sstride::Int, I::Int, lo::Int, stuff::Int,
        xplane_stride::Int, ::Val{NP}, conjugate::Bool
    ) where {Tc, NP}
    # Zero the whole segment on every plane, then scatter the valid entries.
    @inbounds for p in 0:(NP - 1)
        base = doff + p * xplane_stride
        for j in 1:Lp
            dst[base + j] = zero(Tc)
        end
    end
    # Valid j satisfy 0 <= q - lo <= (I-1)*stuff and (q - lo) % stuff == 0,
    # with q = j*s + ph.
    Lstuffed = (I - 1) * stuff + 1
    jlo = max(0, cld(lo - ph, s))
    jhi = min(Lp - 1, fld(lo + Lstuffed - 1 - ph, s))
    jlo > jhi && return nothing
    if stuff == 1
        @inbounds for j in jlo:jhi
            m = j * s + ph - lo
            v = src[soff + m * sstride + 1]
            v = conjugate ? conj(v) : v
            _store_planes!(dst, doff + j + 1, xplane_stride, v, Val(NP))
        end
    else
        @inbounds for j in jlo:jhi
            t = j * s + ph - lo
            r = t % stuff
            r == 0 || continue
            m = t ÷ stuff
            v = src[soff + m * sstride + 1]
            v = conjugate ? conj(v) : v
            _store_planes!(dst, doff + j + 1, xplane_stride, v, Val(NP))
        end
    end
    return nothing
end

@inline function _store_planes!(dst::Vector{Tc}, i::Int, pstride::Int, v, ::Val{1}) where {Tc}
    @inbounds dst[i] = convert(Tc, plane_value(v, Val(1)))
    return nothing
end
@inline function _store_planes!(dst::Vector{Tc}, i::Int, pstride::Int, v, ::Val{3}) where {Tc}
    @inbounds begin
        dst[i] = convert(Tc, plane_value(v, Val(1)))
        dst[i + pstride] = convert(Tc, plane_value(v, Val(2)))
        dst[i + 2pstride] = convert(Tc, plane_value(v, Val(3)))
    end
    return nothing
end

"""
    pack_weights!(Wp, w, g, Kc, NR, ::Val{NP}, ::Type{Tc}, conjugate)

Pack the weight array into the layout read by the microkernel. For every
group, output-channel tile (`NR` channels, the last possibly fewer) and
input-channel block (`Kc` channels, the last possibly fewer), the block is a
contiguous run over `(ci_local, tap)` of `NP * nr` values:
`plane_1[1:nr], plane_2[1:nr], ...`. Taps are enumerated in the same order as
[`tap_offsets`](@ref); kernel flipping is applied here.
"""
function pack_weights!(
        Wp::Vector{Tc}, w::AbstractArray{<:Number, N}, g::ConvGeometry{N, S},
        Kc::Int, NR::Int, ::Val{NP}, conjugate::Bool
    ) where {Tc, N, S, NP}
    G = g.groups
    cin_g = channels_in(g) ÷ G
    cout_g = channels_out(g) ÷ G
    K = prod(ntuple(i -> g.wsize[i], Val(S)))
    kspatial = CartesianIndices(ntuple(i -> g.wsize[i], Val(S)))
    ntiles = cld(cout_g, NR)
    nblocks = cld(cin_g, Kc)
    @inbounds for grp in 1:G, ct in 1:ntiles
        co0 = (ct - 1) * NR
        nr = min(NR, cout_g - co0)
        for cb in 1:nblocks
            ci0 = (cb - 1) * Kc
            kc = min(Kc, cin_g - ci0)
            off = weight_block_offset(g, Kc, NR, NP, grp, ct, cb)
            idx = off
            for ci_l in 1:kc
                ci = ci0 + ci_l
                for (t, k) in enumerate(kspatial)
                    for p in 1:NP, j in 1:nr
                        co = (grp - 1) * cout_g + co0 + j
                        v = w[Tuple(k)..., ci, co]
                        v = conjugate ? conj(v) : v
                        Wp[idx + (p - 1) * nr + j] = convert(Tc, plane_value(v, Val(p)))
                    end
                    idx += NP * nr
                end
            end
        end
    end
    return Wp
end

"""
    weight_block_offset(g, Kc, NR, NP, grp, ct, cb)

Offset (0-based) into the packed weight buffer of the block for group `grp`,
output tile `ct` and input-channel block `cb`.
"""
@inline function weight_block_offset(g::ConvGeometry{N, S}, Kc::Int, NR::Int, NP::Int, grp::Int, ct::Int, cb::Int) where {N, S}
    G = g.groups
    cin_g = channels_in(g) ÷ G
    cout_g = channels_out(g) ÷ G
    K = prod(ntuple(i -> g.wsize[i], Val(S)))
    nr = min(NR, cout_g - (ct - 1) * NR)
    return NP * K * ((grp - 1) * cin_g * cout_g + (ct - 1) * NR * cin_g + (cb - 1) * Kc * nr)
end

packed_weight_length(g::ConvGeometry, NP::Int) = NP * prod(g.wsize)

"""
    tap_offsets(g, Lp, W1, R) -> Vector{Int}

For every kernel tap (enumerated in `CartesianIndices` order of the spatial
kernel dims), the 0-based offset into the packed tile of the input element
feeding output position 0 of the tile, relative to the tile base for the
channel. Along dimension 1 the tap `off = k′*d` splits into phase `off % s`
and shift `off ÷ s`; along the others it is `k′*d` rows.
"""
function tap_offsets(g::ConvGeometry{N, S}, Lp::Int, W1::Int, R::NTuple{S, Int}) where {N, S}
    kspatial = CartesianIndices(ntuple(i -> g.wsize[i], Val(S)))
    taps = Vector{Int}(undef, length(kspatial))
    for (t, k) in enumerate(kspatial)
        off1 = (kernel_index(g, k[1], 1) - 1) * g.dilation[1]
        ph = off1 % g.stride[1]
        q = off1 ÷ g.stride[1]
        o = q + ph * Lp
        rowstride = W1
        for i in 2:S
            r = (kernel_index(g, k[i], i) - 1) * g.dilation[i]
            o += r * rowstride
            rowstride *= R[i]
        end
        taps[t] = o
    end
    return taps
end

"""
    max_tap_shift(g)

Largest `k′*d ÷ s` along the first spatial dimension over all taps.
"""
max_tap_shift(g::ConvGeometry) = ((g.wsize[1] - 1) * g.dilation[1]) ÷ g.stride[1]

"""
    pack_input_tile!(Xp, view, g, b, ci0, kc, tile_origin, tile, Lp, W1, R, xci_stride, xplane_stride, ::Val{NP}, conjugate)

Pack the input region needed for output tile at `tile_origin` (0-based per
spatial dim), for input channels `ci0+1:ci0+kc` of batch `b`.
"""
function pack_input_tile!(
        Xp::Vector{Tc}, iv::InputView{<:AbstractArray{<:Number, N}, S}, g::ConvGeometry{N, S},
        b::Int, ci0::Int, kc::Int, tile_origin::NTuple{S, Int}, Lp::Int, W1::Int,
        R::NTuple{S, Int}, xci_stride::Int, xplane_stride::Int, ::Val{NP}, conjugate::Bool
    ) where {Tc, N, S, NP}
    x = iv.x
    lo = pad_lo(g)
    s = g.stride
    xs = strides(x)
    # Row-space of the tile: dims 2..S have R[i] padded rows starting at
    # padded coordinate tile_origin[i]*s_i.
    rows = CartesianIndices(ntuple(i -> R[i + 1], Val(S - 1)))
    rowstrides = ntuple(Val(S - 1)) do i
        st = W1
        for j in 2:i
            st *= R[j]
        end
        st
    end
    I1 = size(x, 1)                # unstuffed source extent
    stuff1 = iv.stuff[1]
    # Effective (stuffed) padded coordinate lo for dim 1 is lo[1] itself.
    lo1 = lo[1] - tile_origin[1] * s[1]   # shift so local output 0 maps to padded coordinate 0
    @inbounds for ci_l in 1:kc
        ci = ci0 + ci_l
        cbase = (ci_l - 1) * xci_stride
        for r in rows
            # padded coordinate along dim i (0-based), and the corresponding source index
            inside = true
            soff = (ci - 1) * xs[N - 1] + (b - 1) * xs[N]
            for i in 1:(S - 1)
                q = tile_origin[i + 1] * s[i + 1] + (r[i] - 1) - lo[i + 1]   # 0-based stuffed source coordinate
                stf = iv.stuff[i + 1]
                if q < 0 || q % stf != 0
                    inside = false
                    break
                end
                m = q ÷ stf
                if m >= size(x, i + 1)
                    inside = false
                    break
                end
                soff += m * xs[i + 1]
            end
            rbase = cbase
            for i in 1:(S - 1)
                rbase += (r[i] - 1) * rowstrides[i]
            end
            for ph in 0:(s[1] - 1)
                doff = rbase + ph * Lp
                if inside
                    pack_row!(Xp, doff, Lp, s[1], ph, x, soff, xs[1], I1, lo1, stuff1, xplane_stride, Val(NP), conjugate)
                else
                    for p in 0:(NP - 1), j in 1:Lp
                        Xp[doff + p * xplane_stride + j] = zero(Tc)
                    end
                end
            end
        end
    end
    return Xp
end
