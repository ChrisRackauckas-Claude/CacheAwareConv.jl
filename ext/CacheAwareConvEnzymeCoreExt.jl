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

end
