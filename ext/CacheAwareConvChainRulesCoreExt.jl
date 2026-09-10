module CacheAwareConvChainRulesCoreExt

using CacheAwareConv
using CacheAwareConv: ConvPlan, conv_bias, ∇conv_data, ∇conv_filter, bias_gradient!, plan_conv, output_size
using ChainRulesCore: ChainRulesCore, NoTangent, ZeroTangent, @thunk, unthunk, @non_differentiable, ProjectTo,
    RuleConfig, HasReverseMode, rrule_via_ad

@non_differentiable ConvPlan(::Any...)
@non_differentiable plan_conv(::Any...)
@non_differentiable CacheAwareConv.get_plan!(::Any...)
@non_differentiable CacheAwareConv.calc_padding(::Any...)

_tangent_array(ȳ, ::Type{T}, sz) where {T} = convert(Array{T}, reshape(unthunk(ȳ), sz))

function ChainRulesCore.rrule(::typeof(conv_bias), x::AbstractArray{T, N}, w::AbstractArray{T, N}, bias, p::ConvPlan{T, Tc, N}) where {T, Tc, N}
    y = conv_bias(x, w, bias, p)
    px = ProjectTo(x)
    pw = ProjectTo(w)
    pb = bias === nothing ? nothing : ProjectTo(bias)
    ysz = size(y)
    function conv_bias_pullback(ȳ)
        ȳa = _tangent_array(ȳ, T, ysz)
        x̄ = @thunk px(∇conv_data(ȳa, w, p))
        w̄ = @thunk pw(∇conv_filter(x, ȳa, p))
        b̄ = if bias === nothing
            NoTangent()
        else
            @thunk pb(bias_gradient!(similar(bias, T), ȳa))
        end
        return NoTangent(), x̄, w̄, b̄, NoTangent()
    end
    return y, conv_bias_pullback
end

# `conv(x, w, plan; bias, σ)`: keyword arguments carry no tangent in
# ChainRules, so the bias gradient is dropped here; use `conv_bias` to train a
# bias. The activation is handled by differentiating the broadcast with the
# caller's AD.
function ChainRulesCore.rrule(
        config::RuleConfig{>:HasReverseMode}, ::typeof(conv), x::AbstractArray{T, N}, w::AbstractArray{T, N}, p::ConvPlan{T, Tc, N};
        bias = nothing, σ = identity
    ) where {T, Tc, N}
    z, back = ChainRulesCore.rrule(conv_bias, x, w, bias, p)
    if σ === identity
        function conv_pullback_id(ȳ)
            _, x̄, w̄, _, _ = back(ȳ)
            return NoTangent(), x̄, w̄, NoTangent()
        end
        return z, conv_pullback_id
    end
    y, σback = rrule_via_ad(config, z -> σ.(z), z)
    function conv_pullback(ȳ)
        z̄ = σback(ȳ)[2]
        _, x̄, w̄, _, _ = back(z̄)
        return NoTangent(), x̄, w̄, NoTangent()
    end
    return y, conv_pullback
end

end
