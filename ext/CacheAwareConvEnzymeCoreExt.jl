module CacheAwareConvEnzymeCoreExt

using CacheAwareConv
using CacheAwareConv: ConvPlan, conv_core!, ∇conv_data!, ∇conv_filter!, bias_gradient!
using EnzymeCore
using EnzymeCore: EnzymeRules, Const, Duplicated, DuplicatedNoNeed, BatchDuplicated, BatchDuplicatedNoNeed, Annotation

# Plans hold only scratch buffers and geometry: never differentiate them.
EnzymeRules.inactive_type(::Type{<:ConvPlan}) = true
EnzymeRules.inactive_type(::Type{<:CacheAwareConv.PlanCache}) = true
EnzymeRules.inactive(::typeof(CacheAwareConv.get_plan!), args...) = nothing
EnzymeRules.inactive(::typeof(CacheAwareConv.plan_conv), args...) = nothing
EnzymeRules.inactive(::typeof(CacheAwareConv.calc_padding), args...) = nothing

const AccArg = Union{Const{Bool}, Nothing}
_acc(::Nothing) = false
_acc(a::Const{Bool}) = a.val

_shadows(a::Const, n) = ntuple(_ -> nothing, n)
_shadows(a::Union{Duplicated, DuplicatedNoNeed}, n) = (a.dval,)
_shadows(a::Union{BatchDuplicated, BatchDuplicatedNoNeed}, n) = a.dval

_isconst(::Const) = true
_isconst(::Annotation) = false
_isconst(::Nothing) = true

# ---------------------------------------------------------------------------
# Forward mode: ẏ = conv(ẋ, w) + conv(x, ẇ) + ḃ
function EnzymeRules.forward(
        config::EnzymeRules.FwdConfig, func::Const{typeof(conv_core!)}, ::Type{RT},
        y::Annotation{<:AbstractArray{T, N}}, x::Annotation{<:AbstractArray{T, N}},
        w::Annotation{<:AbstractArray{T, N}}, bias::Annotation, p::Const{<:ConvPlan{T}},
        accumulate::Vararg{Const{Bool}}
    ) where {RT, T, N}
    acc = isempty(accumulate) ? false : accumulate[1].val
    width = EnzymeRules.width(config)
    if !_isconst(y)
        dys = _shadows(y, width)
        dxs = _isconst(x) ? ntuple(_ -> nothing, width) : _shadows(x, width)
        dws = _isconst(w) ? ntuple(_ -> nothing, width) : _shadows(w, width)
        dbs = (bias isa Const || bias.val === nothing) ? ntuple(_ -> nothing, width) : _shadows(bias, width)
        for i in 1:width
            dy = dys[i]
            acc || fill!(dy, zero(T))
            dx = dxs[i]
            dw = dws[i]
            db = dbs[i]
            if dx !== nothing
                conv_core!(dy, dx, w.val, db, p.val, true)
                db = nothing
            end
            if dw !== nothing
                conv_core!(dy, x.val, dw, nothing, p.val, true)
            end
            if db !== nothing
                dy .+= reshape(db, ntuple(_ -> 1, N - 2)..., :, 1)
            end
        end
    end
    conv_core!(y.val, x.val, w.val, bias isa Const ? bias.val : bias.val, p.val, acc)
    if EnzymeRules.needs_primal(config) && EnzymeRules.needs_shadow(config)
        return width == 1 ? Duplicated(y.val, y.dval) : BatchDuplicated(y.val, y.dval)
    elseif EnzymeRules.needs_shadow(config)
        return y.dval
    elseif EnzymeRules.needs_primal(config)
        return y.val
    else
        return nothing
    end
end

# ---------------------------------------------------------------------------
# Reverse mode
function EnzymeRules.augmented_primal(
        config::EnzymeRules.RevConfig, func::Const{typeof(conv_core!)}, ::Type{RT},
        y::Annotation{<:AbstractArray{T, N}}, x::Annotation{<:AbstractArray{T, N}},
        w::Annotation{<:AbstractArray{T, N}}, bias::Annotation, p::Const{<:ConvPlan{T}},
        accumulate::Vararg{Const{Bool}}
    ) where {RT, T, N}
    acc = isempty(accumulate) ? false : accumulate[1].val
    conv_core!(y.val, x.val, w.val, bias.val, p.val, acc)
    primal = EnzymeRules.needs_primal(config) ? y.val : nothing
    shadow = EnzymeRules.needs_shadow(config) ? y.dval : nothing
    ow = EnzymeRules.overwritten(config)
    # Cache x if it may be overwritten before the reverse pass and w is active.
    cache_x = (ow[3] && !_isconst(w) && !_isconst(y)) ? copy(x.val) : nothing
    cache_w = (ow[4] && !_isconst(x) && !_isconst(y)) ? copy(w.val) : nothing
    return EnzymeRules.AugmentedReturn(primal, shadow, (cache_x, cache_w))
end

function EnzymeRules.reverse(
        config::EnzymeRules.RevConfig, func::Const{typeof(conv_core!)}, ::Type{RT}, cache,
        y::Annotation{<:AbstractArray{T, N}}, x::Annotation{<:AbstractArray{T, N}},
        w::Annotation{<:AbstractArray{T, N}}, bias::Annotation, p::Const{<:ConvPlan{T}},
        accumulate::Vararg{Const{Bool}}
    ) where {RT, T, N}
    acc = isempty(accumulate) ? false : accumulate[1].val
    cache_x, cache_w = cache
    xv = cache_x === nothing ? x.val : cache_x
    wv = cache_w === nothing ? w.val : cache_w
    if !_isconst(y)
        width = EnzymeRules.width(config)
        dys = _shadows(y, width)
        dxs = _isconst(x) ? ntuple(_ -> nothing, width) : _shadows(x, width)
        dws = _isconst(w) ? ntuple(_ -> nothing, width) : _shadows(w, width)
        dbs = (bias isa Const || bias.val === nothing) ? ntuple(_ -> nothing, width) : _shadows(bias, width)
        for i in 1:width
            dy = dys[i]
            dy === y.val && continue          # runtime-inactive shadow
            dx = dxs[i]
            dw = dws[i]
            db = dbs[i]
            if dx !== nothing && dx !== x.val
                ∇conv_data!(dx, dy, wv, p.val; accumulate = true)
            end
            if dw !== nothing && dw !== w.val
                ∇conv_filter!(dw, xv, dy, p.val; accumulate = true)
            end
            if db !== nothing && db !== bias.val
                bias_gradient!(db, dy; accumulate = true)
            end
            # y was overwritten by the primal unless accumulating into it.
            acc || fill!(dy, zero(T))
        end
    end
    n = 5 + length(accumulate)
    return ntuple(_ -> nothing, n)
end

# ---------------------------------------------------------------------------
# The gradient functions are linear maps too, so their derivatives are again
# cache-aware convolutions:
#   ∇conv_data!(x̄, ȳ, w, p):    ∂x̄/∂ȳ ⋅ v = ∇conv_data(v, w),  ∂x̄/∂w ⋅ u = ∇conv_data(ȳ, u)
#      reverse: dȳ += conv(dx̄, w),            dw += ∇conv_filter(dx̄, ȳ)
#   ∇conv_filter!(w̄, x, ȳ, p):  ∂w̄/∂x ⋅ v = ∇conv_filter(v, ȳ), ∂w̄/∂ȳ ⋅ u = ∇conv_filter(x, u)
#      reverse: dx += ∇conv_data(ȳ, dw̄),      dȳ += conv(x, dw̄)

_accum_kw(kwargs) = get(kwargs, :accumulate, false)::Bool

function EnzymeRules.forward(
        config::EnzymeRules.FwdConfig, func::Const{typeof(∇conv_data!)}, ::Type{RT},
        x̄::Annotation{<:AbstractArray{T, N}}, ȳ::Annotation{<:AbstractArray{T, N}},
        w::Annotation{<:AbstractArray{T, N}}, p::Const{<:ConvPlan{T}}; kwargs...
    ) where {RT, T, N}
    acc = _accum_kw(kwargs)
    width = EnzymeRules.width(config)
    if !_isconst(x̄)
        dxs = _shadows(x̄, width)
        dys = _isconst(ȳ) ? ntuple(_ -> nothing, width) : _shadows(ȳ, width)
        dws = _isconst(w) ? ntuple(_ -> nothing, width) : _shadows(w, width)
        for i in 1:width
            dx = dxs[i]
            acc || fill!(dx, zero(T))
            dys[i] === nothing || ∇conv_data!(dx, dys[i], w.val, p.val; accumulate = true)
            dws[i] === nothing || ∇conv_data!(dx, ȳ.val, dws[i], p.val; accumulate = true)
        end
    end
    ∇conv_data!(x̄.val, ȳ.val, w.val, p.val; accumulate = acc)
    return _forward_return(config, x̄)
end

function EnzymeRules.forward(
        config::EnzymeRules.FwdConfig, func::Const{typeof(∇conv_filter!)}, ::Type{RT},
        w̄::Annotation{<:AbstractArray{T, N}}, x::Annotation{<:AbstractArray{T, N}},
        ȳ::Annotation{<:AbstractArray{T, N}}, p::Const{<:ConvPlan{T}}; kwargs...
    ) where {RT, T, N}
    acc = _accum_kw(kwargs)
    width = EnzymeRules.width(config)
    if !_isconst(w̄)
        dws = _shadows(w̄, width)
        dxs = _isconst(x) ? ntuple(_ -> nothing, width) : _shadows(x, width)
        dys = _isconst(ȳ) ? ntuple(_ -> nothing, width) : _shadows(ȳ, width)
        for i in 1:width
            dw = dws[i]
            acc || fill!(dw, zero(T))
            dxs[i] === nothing || ∇conv_filter!(dw, dxs[i], ȳ.val, p.val; accumulate = true)
            dys[i] === nothing || ∇conv_filter!(dw, x.val, dys[i], p.val; accumulate = true)
        end
    end
    ∇conv_filter!(w̄.val, x.val, ȳ.val, p.val; accumulate = acc)
    return _forward_return(config, w̄)
end

function _forward_return(config, y)
    width = EnzymeRules.width(config)
    if EnzymeRules.needs_primal(config) && EnzymeRules.needs_shadow(config)
        return width == 1 ? Duplicated(y.val, y.dval) : BatchDuplicated(y.val, y.dval)
    elseif EnzymeRules.needs_shadow(config)
        return y.dval
    elseif EnzymeRules.needs_primal(config)
        return y.val
    else
        return nothing
    end
end

for (f, argA, argB) in ((:∇conv_data!, :ȳ, :w), (:∇conv_filter!, :x, :ȳ))
    @eval begin
        function EnzymeRules.augmented_primal(
                config::EnzymeRules.RevConfig, func::Const{typeof($f)}, ::Type{RT},
                out::Annotation{<:AbstractArray{T, N}}, a::Annotation{<:AbstractArray{T, N}},
                b::Annotation{<:AbstractArray{T, N}}, p::Const{<:ConvPlan{T}}; kwargs...
            ) where {RT, T, N}
            $f(out.val, a.val, b.val, p.val; kwargs...)
            primal = EnzymeRules.needs_primal(config) ? out.val : nothing
            shadow = EnzymeRules.needs_shadow(config) ? out.dval : nothing
            ow = EnzymeRules.overwritten(config)
            cache_a = (ow[3] && !_isconst(b) && !_isconst(out)) ? copy(a.val) : nothing
            cache_b = (ow[4] && !_isconst(a) && !_isconst(out)) ? copy(b.val) : nothing
            return EnzymeRules.AugmentedReturn(primal, shadow, (cache_a, cache_b))
        end
    end
end

function EnzymeRules.reverse(
        config::EnzymeRules.RevConfig, func::Const{typeof(∇conv_data!)}, ::Type{RT}, cache,
        x̄::Annotation{<:AbstractArray{T, N}}, ȳ::Annotation{<:AbstractArray{T, N}},
        w::Annotation{<:AbstractArray{T, N}}, p::Const{<:ConvPlan{T}}; kwargs...
    ) where {RT, T, N}
    acc = _accum_kw(kwargs)
    cache_y, cache_w = cache
    yv = cache_y === nothing ? ȳ.val : cache_y
    wv = cache_w === nothing ? w.val : cache_w
    if !_isconst(x̄)
        width = EnzymeRules.width(config)
        dxs = _shadows(x̄, width)
        dys = _isconst(ȳ) ? ntuple(_ -> nothing, width) : _shadows(ȳ, width)
        dws = _isconst(w) ? ntuple(_ -> nothing, width) : _shadows(w, width)
        for i in 1:width
            dx = dxs[i]
            dx === x̄.val && continue
            if dys[i] !== nothing && dys[i] !== ȳ.val
                conv_core!(dys[i], dx, wv, nothing, p.val, true)
            end
            if dws[i] !== nothing && dws[i] !== w.val
                ∇conv_filter!(dws[i], dx, yv, p.val; accumulate = true)
            end
            acc || fill!(dx, zero(T))
        end
    end
    return (nothing, nothing, nothing, nothing)
end

function EnzymeRules.reverse(
        config::EnzymeRules.RevConfig, func::Const{typeof(∇conv_filter!)}, ::Type{RT}, cache,
        w̄::Annotation{<:AbstractArray{T, N}}, x::Annotation{<:AbstractArray{T, N}},
        ȳ::Annotation{<:AbstractArray{T, N}}, p::Const{<:ConvPlan{T}}; kwargs...
    ) where {RT, T, N}
    acc = _accum_kw(kwargs)
    cache_x, cache_y = cache
    xv = cache_x === nothing ? x.val : cache_x
    yv = cache_y === nothing ? ȳ.val : cache_y
    if !_isconst(w̄)
        width = EnzymeRules.width(config)
        dws = _shadows(w̄, width)
        dxs = _isconst(x) ? ntuple(_ -> nothing, width) : _shadows(x, width)
        dys = _isconst(ȳ) ? ntuple(_ -> nothing, width) : _shadows(ȳ, width)
        for i in 1:width
            dw = dws[i]
            dw === w̄.val && continue
            if dxs[i] !== nothing && dxs[i] !== x.val
                ∇conv_data!(dxs[i], yv, dw, p.val; accumulate = true)
            end
            if dys[i] !== nothing && dys[i] !== ȳ.val
                conv_core!(dys[i], xv, dw, nothing, p.val, true)
            end
            acc || fill!(dw, zero(T))
        end
    end
    return (nothing, nothing, nothing, nothing)
end

end
