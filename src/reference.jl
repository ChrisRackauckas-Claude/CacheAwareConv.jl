# Naive reference implementations. These are deliberately simple (a direct
# translation of the definition) and serve as the oracle for every test of the
# fast path. They are exported so downstream packages can use them in tests.

"""
    reference_conv!(y, x, w, g::ConvGeometry)

Compute the convolution described by `g` with plain nested loops, writing into
`y`. Slow; intended as a correctness oracle.
"""
function reference_conv!(y::AbstractArray{<:Number, N}, x::AbstractArray{<:Number, N}, w::AbstractArray{<:Number, N}, g::ConvGeometry{N, S}) where {N, S}
    check_conv_args(g, y, x, w)
    T = eltype(y)
    fill!(y, zero(T))
    G = g.groups
    cin_g = channels_in(g) ÷ G
    cout_g = channels_out(g) ÷ G
    lo = pad_lo(g)
    ospatial = CartesianIndices(ntuple(i -> g.ysize[i], Val(S)))
    kspatial = CartesianIndices(ntuple(i -> g.wsize[i], Val(S)))
    @inbounds for b in 1:batch_size(g), grp in 1:G, co_l in 1:cout_g
        co = (grp - 1) * cout_g + co_l
        for o in ospatial
            acc = zero(T)
            for ci_l in 1:cin_g
                ci = (grp - 1) * cin_g + ci_l
                for k in kspatial
                    xi = ntuple(Val(S)) do i
                        (o[i] - 1) * g.stride[i] - lo[i] + 1 + (kernel_index(g, k[i], i) - 1) * g.dilation[i]
                    end
                    inside = true
                    for i in 1:S
                        inside &= 1 <= xi[i] <= g.xsize[i]
                    end
                    inside || continue
                    acc = muladd(T(x[xi..., ci, b]), T(w[Tuple(k)..., ci_l, co]), acc)
                end
            end
            y[Tuple(o)..., co, b] = acc
        end
    end
    return y
end

"""
    reference_∇conv_data!(x̄, ȳ, w, g::ConvGeometry)

Gradient of the convolution with respect to its input, by plain loops.
For complex numbers this is the adjoint (conjugate transpose) map, matching
the convention used by ChainRules and Enzyme.
"""
function reference_∇conv_data!(x̄::AbstractArray{<:Number, N}, ȳ::AbstractArray{<:Number, N}, w::AbstractArray{<:Number, N}, g::ConvGeometry{N, S}) where {N, S}
    check_conv_args(g, ȳ, x̄, w)
    T = eltype(x̄)
    fill!(x̄, zero(T))
    G = g.groups
    cin_g = channels_in(g) ÷ G
    cout_g = channels_out(g) ÷ G
    lo = pad_lo(g)
    ospatial = CartesianIndices(ntuple(i -> g.ysize[i], Val(S)))
    kspatial = CartesianIndices(ntuple(i -> g.wsize[i], Val(S)))
    @inbounds for b in 1:batch_size(g), grp in 1:G, co_l in 1:cout_g
        co = (grp - 1) * cout_g + co_l
        for o in ospatial
            ȳv = T(ȳ[Tuple(o)..., co, b])
            for ci_l in 1:cin_g
                ci = (grp - 1) * cin_g + ci_l
                for k in kspatial
                    xi = ntuple(Val(S)) do i
                        (o[i] - 1) * g.stride[i] - lo[i] + 1 + (kernel_index(g, k[i], i) - 1) * g.dilation[i]
                    end
                    inside = true
                    for i in 1:S
                        inside &= 1 <= xi[i] <= g.xsize[i]
                    end
                    inside || continue
                    x̄[xi..., ci, b] = muladd(ȳv, conj(T(w[Tuple(k)..., ci_l, co])), x̄[xi..., ci, b])
                end
            end
        end
    end
    return x̄
end

"""
    reference_∇conv_filter!(w̄, x, ȳ, g::ConvGeometry)

Gradient of the convolution with respect to its weights, by plain loops.
"""
function reference_∇conv_filter!(w̄::AbstractArray{<:Number, N}, x::AbstractArray{<:Number, N}, ȳ::AbstractArray{<:Number, N}, g::ConvGeometry{N, S}) where {N, S}
    check_conv_args(g, ȳ, x, w̄)
    T = eltype(w̄)
    fill!(w̄, zero(T))
    G = g.groups
    cin_g = channels_in(g) ÷ G
    cout_g = channels_out(g) ÷ G
    lo = pad_lo(g)
    ospatial = CartesianIndices(ntuple(i -> g.ysize[i], Val(S)))
    kspatial = CartesianIndices(ntuple(i -> g.wsize[i], Val(S)))
    @inbounds for b in 1:batch_size(g), grp in 1:G, co_l in 1:cout_g
        co = (grp - 1) * cout_g + co_l
        for o in ospatial
            ȳv = T(ȳ[Tuple(o)..., co, b])
            for ci_l in 1:cin_g
                ci = (grp - 1) * cin_g + ci_l
                for k in kspatial
                    xi = ntuple(Val(S)) do i
                        (o[i] - 1) * g.stride[i] - lo[i] + 1 + (kernel_index(g, k[i], i) - 1) * g.dilation[i]
                    end
                    inside = true
                    for i in 1:S
                        inside &= 1 <= xi[i] <= g.xsize[i]
                    end
                    inside || continue
                    w̄[Tuple(k)..., ci_l, co] = muladd(conj(T(x[xi..., ci, b])), ȳv, w̄[Tuple(k)..., ci_l, co])
                end
            end
        end
    end
    return w̄
end
