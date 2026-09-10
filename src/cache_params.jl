# Hardware parameters: cache sizes (CPUSummary), SIMD width and register count
# (HostCPUFeatures), and the element-type-specific choices derived from them.

"""
    CacheInfo(; l1, l2, l3, linesize)

Per-core cache sizes in bytes used to derive the blocking of a
[`ConvPlan`](@ref). The default, [`cache_info`](@ref), queries CPUSummary.jl;
pass an explicit `CacheInfo` to a plan to override it (useful for testing the
blocking logic or tuning by hand).
"""
struct CacheInfo
    l1::Int
    l2::Int
    l3::Int
    linesize::Int
    function CacheInfo(; l1::Integer, l2::Integer, l3::Integer, linesize::Integer = 64)
        l1 > 0 || throw(ArgumentError("l1 must be positive"))
        l2 >= l1 || throw(ArgumentError("l2 must be at least l1"))
        l3 >= l2 || throw(ArgumentError("l3 must be at least l2"))
        return new(Int(l1), Int(l2), Int(l3), Int(linesize))
    end
end

"""
    cache_info()

Per-core L1/L2/L3 sizes as reported by CPUSummary.jl, with conservative
fallbacks for levels the platform does not report (a missing level is taken to
be eight times the previous one).
"""
function cache_info()
    l1 = Int(known(cache_size(Val(1))))::Int
    l2 = Int(known(cache_size(Val(2))))::Int
    l3 = Int(known(cache_size(Val(3))))::Int
    l1 <= 0 && (l1 = 32768)
    l2 <= l1 && (l2 = 8 * l1)
    l3 <= l2 && (l3 = 8 * l2)
    line = Int(known(cache_linesize()))::Int
    line <= 0 && (line = 64)
    return CacheInfo(; l1, l2, l3, linesize = line)
end

"""
    compute_type(T)

Element type used inside the packed buffers and the microkernel for arrays of
element type `T`. `Float16` is computed in `Float32`; complex types are split
into real planes of `compute_type(real(T))`; everything else is unchanged.
"""
compute_type(::Type{T}) where {T} = T
compute_type(::Type{Float16}) = Float32
compute_type(::Type{Complex{T}}) where {T} = compute_type(T)

"""
    nplanes(T)

Number of real planes an element of type `T` is packed into: 1 for real types,
3 for complex types (`re`, `im`, `re + im` for the 3-multiplication kernel).
"""
nplanes(::Type{T}) where {T} = 1
nplanes(::Type{<:Complex}) = 3

"""
    simd_type(Tc)

Whether the compute type `Tc` gets the SIMD.jl microkernel (`Float32`,
`Float64`) rather than the generic scalar kernel.
"""
simd_type(::Type{Tc}) where {Tc} = false
simd_type(::Type{Float32}) = true
simd_type(::Type{Float64}) = true

"""
    vector_width(Tc)

SIMD lanes per vector for compute type `Tc` on this machine (1 for the scalar path).
"""
function vector_width(::Type{Tc}) where {Tc}
    simd_type(Tc) || return 1
    return max(1, Int(known(pick_vector_width(Tc)))::Int)
end

"""
    register_tile(Tc, nplanes) -> (MR, NR)

Register-tile shape of the microkernel: `MR` vectors along the output width
times `NR` output channels, chosen from the architecture's vector register
count so that `nplanes * MR * NR` accumulators plus `nplanes * MR` input vectors
and one broadcast fit without spilling.
"""
function register_tile(::Type{Tc}, np::Int) where {Tc}
    simd_type(Tc) || return (4, 4)
    rc = Int(known(register_count()))::Int
    if np == 1
        rc >= 32 && return (4, 6)
        rc >= 16 && return (2, 4)
        return (1, 4)
    else
        rc >= 32 && return (2, 4)
        rc >= 16 && return (1, 3)
        return (1, 2)
    end
end
