# Positional, AD-friendly entry points and a small plan cache used by the
# layer extensions.

"""
    conv_core!(y, x, w, bias, plan::ConvPlan, accumulate::Bool = false)

Positional form of `conv!(y, x, w, plan; bias, accumulate)` without an
activation. This is the function the EnzymeCore extension attaches rules to;
`bias` is `nothing` or a vector with one entry per output channel.
"""
function conv_core!(y::AbstractArray{T, N}, x::AbstractArray{T, N}, w::AbstractArray{T, N}, bias, p::ConvPlan{T, Tc, N}, accumulate::Bool = false) where {T, Tc, N}
    check_conv_args(p.geom, y, x, w)
    _check_bias(bias, channels_out(p.geom))
    _check_strided(y, "y")
    _check_strided(x, "x")
    iv = InputView(x, ntuple(_ -> 1, Val(N - 2)))
    _conv_impl!(y, iv, w, p, bias, identity, false, accumulate)
    return y
end

"""
    conv_bias(x, w, bias, plan::ConvPlan)

Allocating `conv(x, w) .+ bias` (`bias` may be `nothing`). Positional so that
the ChainRulesCore extension can return a tangent for `bias`; the Lux and Flux
layers call this and apply the activation as a separate broadcast.
"""
function conv_bias(x::AbstractArray{T, N}, w::AbstractArray{T, N}, bias, p::ConvPlan{T, Tc, N}) where {T, Tc, N}
    y = similar(x, T, output_size(p))
    return conv_core!(y, x, w, bias, p, false)
end

"""
    bias_gradient!(b̄, ȳ; accumulate = false)

Sum the output gradient over all dimensions except the channel dimension.
"""
function bias_gradient!(b̄::AbstractVector, ȳ::AbstractArray{T, N}; accumulate::Bool = false) where {T, N}
    C = size(ȳ, N - 1)
    length(b̄) == C || throw(DimensionMismatch("bias gradient has length $(length(b̄)), expected $C"))
    accumulate || fill!(b̄, zero(eltype(b̄)))
    inner = prod(ntuple(i -> size(ȳ, i), Val(N - 2)))
    ȳr = reshape(ȳ, inner, C, size(ȳ, N))
    @inbounds for b in axes(ȳr, 3), c in 1:C
        s = zero(T)
        @simd for i in 1:inner
            s += ȳr[i, c, b]
        end
        b̄[c] += s
    end
    return b̄
end

"""
    PlanCache()

Thread-safe cache of [`ConvPlan`](@ref)s keyed by element type and input
size, used by the Lux and Flux layers so that the plan (and its buffers) is
built once per input shape.
"""
struct PlanCache
    plans::Dict{Any, Any}
    lock::ReentrantLock
end
PlanCache() = PlanCache(Dict{Any, Any}(), ReentrantLock())

"""
    get_plan!(cache::PlanCache, x, w; kwargs...) -> ConvPlan

Return the cached plan for `eltype(x)`, `size(x)`, `size(w)` and the
convolution keyword arguments, building it on first use.
"""
function get_plan!(cache::PlanCache, x::AbstractArray{T, N}, w::AbstractArray{T, N}; kwargs...) where {T, N}
    key = (T, size(x), size(w), values(kwargs))
    return Base.@lock cache.lock begin
        get!(cache.plans, key) do
            plan_conv(x, w; kwargs...)
        end
    end::ConvPlan{T}
end
Base.empty!(c::PlanCache) = (Base.@lock c.lock empty!(c.plans); c)
Base.length(c::PlanCache) = Base.@lock c.lock length(c.plans)
Base.show(io::IO, c::PlanCache) = print(io, "PlanCache(", length(c), " plans)")

# Padding helpers shared by the layer extensions (same conventions as Lux/Flux).
"""
    SamePad()

Padding marker for the layer constructors: pad so that the output has the
same spatial size as the input when `stride == 1` (extra padding goes to the
low side when the total is odd, as in Lux and Flux).
"""
struct SamePad end

"""
    calc_padding(pad, k, dilation, stride) -> NTuple{2S, Int}

Expand a layer's `pad` argument (integer, per-dimension tuple, `(lo, hi)`
tuple, or [`SamePad`](@ref)) to explicit `(lo_1, hi_1, …)` padding for
kernel size `k`.
"""
function calc_padding(pad, k::NTuple{S, Int}, dilation::NTuple{S, Int}, stride::NTuple{S, Int}) where {S}
    return _expand_pad(Val(S), pad)
end
function calc_padding(::SamePad, k::NTuple{S, Int}, dilation::NTuple{S, Int}, stride::NTuple{S, Int}) where {S}
    k_eff = ntuple(i -> k[i] + (k[i] - 1) * (dilation[i] - 1), Val(S))
    return ntuple(Val(2S)) do j
        i = (j + 1) ÷ 2
        amt = k_eff[i] - 1
        isodd(j) ? cld(amt, 2) : fld(amt, 2)
    end
end

"""
    LuxConv(k, in_chs => out_chs, activation = identity; kwargs...)

Lux.jl layer using this package's convolution. Available after `using Lux`;
see the Lux extension for the full signature.
"""
function LuxConv end

"""
    FluxConv(k, in => out, σ = identity; kwargs...)
    FluxConv(weight, bias = true, σ = identity; kwargs...)

Flux.jl layer using this package's convolution. Available after `using Flux`;
see the Flux extension for the full signature.
"""
function FluxConv end
