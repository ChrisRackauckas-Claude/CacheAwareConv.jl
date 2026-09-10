using CacheAwareConv, Test, Random
using CacheAwareConv: CacheInfo, output_size, geometry

const RTOL = Dict(
    Float16 => 2.0e-2, Float32 => 1.0e-4, Float64 => 1.0e-10,
    ComplexF32 => 1.0e-4, ComplexF64 => 1.0e-10, BigFloat => 1.0e-20,
)

randarr(rng, ::Type{T}, dims) where {T} = randn(rng, T, dims)
randarr(rng, ::Type{Float16}, dims) = Float16.(randn(rng, Float32, dims))
randarr(rng, ::Type{BigFloat}, dims) = BigFloat.(randn(rng, Float64, dims))

# Small caches force Kc < C_in and tile splits along every dimension.
const SMALL_CACHE = CacheInfo(; l1 = 512, l2 = 4096, l3 = 8192)

function check_case(rng, T, spatial, k, cin, cout; kwargs...)
    x = randarr(rng, T, (spatial..., cin, 2))
    w = randarr(rng, T, (k..., cin ÷ get(kwargs, :groups, 1), cout))
    cache = get(kwargs, :cache, CacheAwareConv.cache_info())
    plankw = (; (k => v for (k, v) in kwargs if k in (:stride, :pad, :dilation, :groups, :flipped))...)
    p = ConvPlan(T, size(x), size(w); plankw..., cache, nthreads = get(kwargs, :nthreads, Threads.nthreads()))
    y = conv(x, w, p)
    yr = similar(y)
    reference_conv!(yr, x, w, geometry(p))
    rtol = RTOL[T]
    @test isapprox(y, yr; rtol, atol = rtol)
    return p
end

@testset "1D/2D/3D grid vs reference" begin
    rng = MersenneTwister(0)
    for S in 1:3, stride in (1, 2, 3), dil in (1, 2), flipped in (false, true), groups in (1, 2)
        spatial = ntuple(i -> rand(rng, 5:11), S)
        k = ntuple(i -> rand(rng, 1:3), S)
        for pad in (0, 1, ntuple(i -> rand(rng, 0:2), 2S))
            g = try
                ConvGeometry((spatial..., 2groups, 2), (k..., 2, 3groups); stride, pad, dilation = dil, groups, flipped)
            catch e
                e isa DimensionMismatch && continue
                rethrow()
            end
            check_case(rng, Float32, spatial, k, 2groups, 3groups; stride, pad, dilation = dil, groups, flipped)
        end
    end
end

@testset "width tails and channel tails ($T)" for T in (Float32, Float64)
    rng = MersenneTwister(1)
    p0 = ConvPlan(T, (64, 4, 1, 1), (3, 3, 1, 1))
    V = CacheAwareConv.vector_width(p0)
    MR, NR = CacheAwareConv.register_tile(p0)
    for W in unique((1, 2, V - 1, V, V + 1, MR * V - 1, MR * V, MR * V + 1, 2MR * V + 3)),
            cout in unique((1, NR - 1, NR, NR + 1, 2NR + 1)), cin in (1, 3)

        check_case(rng, T, (W, 5), (3, 3), cin, cout; pad = 1)
    end
end

@testset "forced blocking (tiny caches)" begin
    rng = MersenneTwister(2)
    for T in (Float32, Float64, Float16, ComplexF32)
        p = check_case(rng, T, (37, 13), (3, 3), 12, 7; pad = (1, 2, 0, 1), stride = (2, 1), cache = SMALL_CACHE)
        @test p.Kc < 12                 # channel blocks force accumulation through y
        @test prod(p.nblocks) > 1       # spatial tiles were split
        check_case(rng, T, (29, 9, 6), (3, 2, 2), 6, 5; pad = 1, dilation = (1, 2, 1), cache = SMALL_CACHE)
        check_case(rng, T, (50,), (5,), 8, 9; stride = 3, pad = 2, cache = SMALL_CACHE)
    end
end

@testset "element types" begin
    rng = MersenneTwister(3)
    for T in (Float16, Float64, ComplexF32, ComplexF64)
        check_case(rng, T, (20, 9), (3, 3), 4, 6; pad = 1, stride = 2)
        check_case(rng, T, (17,), (4,), 3, 5; flipped = true, groups = 1)
    end
    # Generic scalar path
    p = check_case(rng, BigFloat, (9, 6), (3, 2), 2, 3; pad = 1)
    @test !(p isa ConvPlan{BigFloat, BigFloat, 4, 2, 4, 1, 4, 4, 1, true})
    @test CacheAwareConv.vector_width(p) == 1
end

@testset "single-channel stencil path (row-blocked kernel)" begin
    rng = MersenneTwister(7)
    small = CacheInfo(; l1 = 2048, l2 = 16384, l3 = 65536)
    # 2D/3D, strides (phases), dilation, flip, asymmetric pad, tiny tiles, row tails, column tails
    for (T, spatial, k, kw) in (
            (Float64, (37, 11), (3, 3), (; pad = 1)),
            (Float64, (40, 9), (5, 5), (; pad = 2, stride = (2, 1))),
            (Float32, (70, 13), (3, 3), (; pad = (1, 0, 2, 1), dilation = (2, 1), flipped = true)),
            (Float64, (33, 9, 5), (3, 3, 3), (; pad = 1)),
            (Float64, (33, 9, 5), (3, 3, 2), (; pad = 1, stride = (1, 1, 2), dilation = (1, 2, 1))),
            (Float64, (64, 17), (3, 3), (; pad = 1, cache = small)),
            (Float64, (20, 7), (7, 7), (; pad = 3)),
            (Float64, (130, 30), (3, 3), (; pad = 1)),
            (Float64, (17, 4), (3, 3), (; pad = 1)),
            (Float32, (129, 31), (9, 9), (; pad = 4)),
        )
        p = check_case(rng, T, spatial, k, 1, 1; kw...)
        @test typeof(p).parameters[end] !== nothing      # stencil descriptor present
    end
    # row stride > 1 falls back to the general kernel and stays correct
    p = check_case(rng, Float64, (30, 20), (3, 3), 1, 1; pad = 1, stride = (1, 2))
    @test typeof(p).parameters[end] === nothing
    # results independent of thread count
    x = randn(rng, Float64, 200, 60, 1, 2)
    w = randn(rng, Float64, 3, 3, 1, 1)
    y1 = conv(x, w, plan_conv(x, w; pad = 1, nthreads = 1))
    y8 = conv(x, w, plan_conv(x, w; pad = 1, nthreads = 8))
    @test y1 == y8
end

@testset "negative padding (cropping)" begin
    rng = MersenneTwister(4)
    check_case(rng, Float64, (12, 10), (3, 3), 2, 2; pad = (-1, -2, 0, -1))
end

@testset "bias and activation epilogue" begin
    rng = MersenneTwister(5)
    for T in (Float32, Float16, ComplexF64)
        x = randarr(rng, T, (13, 7, 3, 2))
        w = randarr(rng, T, (3, 3, 3, 5))
        bias = randarr(rng, T, 5)
        p = plan_conv(x, w; pad = 1)
        σ = T <: Complex ? (z -> z * z) : (v -> v * v + one(v))
        y = conv(x, w, p; bias, σ)
        yr = similar(y)
        reference_conv!(yr, x, w, geometry(p))
        yr .= σ.(yr .+ reshape(bias, 1, 1, :, 1))
        rtol = RTOL[T] * 10
        @test isapprox(y, yr; rtol, atol = rtol)
    end
end

@testset "in-place conv! reuses the plan without allocating" begin
    x = randn(Float32, 32, 32, 8, 2)
    w = randn(Float32, 3, 3, 8, 16)
    p = plan_conv(x, w; pad = 1, nthreads = 1)
    y = similar(x, Float32, output_size(p))
    conv!(y, x, w, p)
    @test (@allocated conv!(y, x, w, p)) == 0
end

@testset "views and reshapes" begin
    rng = MersenneTwister(6)
    xbig = randn(rng, Float32, 20, 20, 3, 4)
    x = view(xbig, :, :, :, 2:3)
    w = randn(rng, Float32, 3, 3, 3, 4)
    p = plan_conv(x, w)
    y = conv(x, w, p)
    yr = similar(y)
    reference_conv!(yr, x, w, geometry(p))
    @test y ≈ yr
    bad = view(xbig, 1:2:20, :, :, 2:3)
    @test_throws ArgumentError conv(bad, w)
end

@testset "argument validation" begin
    @test_throws DimensionMismatch ConvGeometry((8, 8, 3, 1), (3, 3, 2, 4))
    @test_throws DimensionMismatch ConvGeometry((8, 8, 4, 1), (3, 3, 2, 3); groups = 2)
    @test_throws ArgumentError ConvGeometry((8, 8, 3, 1), (3, 3, 3, 4); stride = 0)
    @test_throws ArgumentError ConvGeometry((8, 8, 3, 1), (3, 3, 3, 4); pad = (1, 2, 3))
    @test_throws DimensionMismatch ConvGeometry((2, 2, 3, 1), (3, 3, 3, 4))
    x = randn(Float32, 8, 8, 3, 1)
    w = randn(Float32, 3, 3, 3, 4)
    p = plan_conv(x, w)
    @test_throws ArgumentError conv!(zeros(Float64, output_size(p)), x, w, p)
    @test_throws DimensionMismatch conv!(zeros(Float32, 5, 5, 4, 1), x, w, p)
    @test_throws DimensionMismatch conv(x, w, p; bias = zeros(Float32, 3))
    @test_throws ArgumentError plan_conv(x, Float64.(w))
    @test conv(x, Float64.(w)) ≈ conv(Float64.(x), Float64.(w))
end

@testset "show" begin
    p = ConvPlan(Float32, (8, 8, 3, 1), (3, 3, 3, 4))
    s = sprint(show, p)
    @test occursin("ConvPlan{Float32}", s)
    @test occursin("Kc=", s)
end
