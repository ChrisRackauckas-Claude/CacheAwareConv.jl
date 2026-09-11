using CacheAwareConv, Test, Random
using LoopVectorization, Polyester
using CacheAwareConv: geometry, output_size

const RTOL = Dict(
    Float16 => 2.0e-2, Float32 => 1.0e-4, Float64 => 1.0e-10,
    ComplexF32 => 1.0e-4, ComplexF64 => 1.0e-10,
)

randarr(rng, ::Type{T}, dims) where {T} = randn(rng, T, dims)
randarr(rng, ::Type{Float16}, dims) = Float16.(randn(rng, Float32, dims))

function check_kernel(rng, T, spatial, k, cin, cout; kernel, kwargs...)
    x = randarr(rng, T, (spatial..., cin, 2))
    w = randarr(rng, T, (k..., cin ÷ get(kwargs, :groups, 1), cout))
    p = ConvPlan(T, size(x), size(w); kwargs..., kernel)
    @test p.kernel === CacheAwareConv._kernel_enum(kernel)
    y = conv(x, w, p)
    yr = similar(y)
    reference_conv!(yr, x, w, geometry(p))
    rtol = RTOL[T]
    @test isapprox(y, yr; rtol, atol = rtol)
    return x, w, y, p
end

@testset "kernel = :lv vs reference" begin
    rng = MersenneTwister(11)
    # 1D/2D/3D, strided, padded, dilated, grouped, flipped; hits direct and
    # buffered stores, partial channel tiles and width tails.
    for (spatial, k, kw) in (
            ((37,), (3,), (;)),
            ((33, 27), (3, 3), (; pad = 1)),
            ((21, 19), (2, 4), (; stride = (2, 1), pad = (0, 1, 0, 2), flipped = true)),
            ((17, 15, 9), (3, 2, 2), (; pad = 1, dilation = (1, 2, 1))),
            ((14, 12), (5, 5), (; pad = 2)),          # large K with Kc blocking
            ((9, 8), (3, 3), (; pad = 1, flat = true)),
            ((9, 8), (3, 3), (; pad = 1, flat = false)),
            ((64,), (9,), (; stride = 2, pad = 4)),
        )
        check_kernel(rng, Float32, spatial, k, 4, 7; kernel = :lv, kw...)
        check_kernel(rng, Float64, spatial, k, 3, 5; kernel = :lv, kw...)
    end
    check_kernel(rng, Float32, (26, 24), (3, 3), 4, 8; kernel = :lv, groups = 2, pad = 1)
    # single channel (would take the stencil path under :simd)
    check_kernel(rng, Float32, (40, 40), (3, 3), 1, 1; kernel = :lv, pad = 1)
    # Float16: buffered output with down-conversion; Complex: NP = 3 planes
    check_kernel(rng, Float16, (23, 21), (3, 3), 4, 6; kernel = :lv, pad = 1)
    check_kernel(rng, ComplexF32, (23, 21), (3, 3), 4, 6; kernel = :lv, pad = 1)
end

@testset "kernel = :lv accumulate and gradients" begin
    rng = MersenneTwister(12)
    x = randn(rng, Float32, 29, 25, 4, 2)
    w = randn(rng, Float32, 3, 3, 4, 6)
    p_lv = plan_conv(x, w; pad = 1, kernel = :lv)
    p_sd = plan_conv(x, w; pad = 1)
    # accumulate: sums onto existing y
    y0 = randn(rng, Float32, output_size(p_lv))
    ya, yb = copy(y0), copy(y0)
    conv!(ya, x, w, p_lv; accumulate = true)
    conv!(yb, x, w, p_sd; accumulate = true)
    @test ya ≈ yb rtol = 1.0e-4
    # ∇conv_data runs the transposed convolution through the LV kernel
    ȳ = randn(rng, Float32, size(ya))
    @test ∇conv_data(ȳ, w, p_lv) ≈ ∇conv_data(ȳ, w, p_sd) rtol = 1.0e-4
    @test ∇conv_filter(x, ȳ, p_lv) ≈ ∇conv_filter(x, ȳ, p_sd) rtol = 1.0e-4
end

@testset "kernel = :scalar forces the scalar kernel" begin
    rng = MersenneTwister(13)
    check_kernel(rng, Float32, (33, 27), (3, 3), 4, 7; kernel = :scalar, pad = 1)
    check_kernel(rng, Float32, (21, 19), (2, 4), 4, 7; kernel = :scalar, stride = 2, pad = (0, 1, 0, 2))
    check_kernel(rng, Float64, (17, 15, 9), (3, 2, 2), 2, 3; kernel = :scalar, pad = 1)
end

@testset "executor = :polyester is bitwise identical to :spawn" begin
    rng = MersenneTwister(14)
    x = randn(rng, Float32, 37, 29, 6, 3)
    w = randn(rng, Float32, 3, 3, 6, 10)
    ps = plan_conv(x, w; pad = 1, nthreads = 4)
    pp = plan_conv(x, w; pad = 1, nthreads = 4, executor = :polyester)
    @test pp.executor === CacheAwareConv.ExecPolyester
    @test conv(x, w, pp) == conv(x, w, ps)
    ȳ = randn(rng, Float32, output_size(ps))
    @test ∇conv_data(ȳ, w, pp) == ∇conv_data(ȳ, w, ps)
    @test ∇conv_filter(x, ȳ, pp) ≈ ∇conv_filter(x, ȳ, ps)
    # polyester also schedules weight packing and combines with kernel = :lv
    plv = plan_conv(x, w; pad = 1, nthreads = 4, kernel = :lv, executor = :polyester)
    @test conv(x, w, plv) ≈ conv(x, w, ps) rtol = 1.0e-4
end

@testset "kernel/executor validation" begin
    x = randn(Float32, 9, 7, 2, 1)
    w = randn(Float32, 3, 3, 2, 4)
    @test_throws ArgumentError plan_conv(x, w; kernel = :bogus)
    @test_throws ArgumentError plan_conv(x, w; kernel = :turbo)
    @test_throws ArgumentError plan_conv(x, w; executor = :bogus)
    @test_throws ArgumentError plan_conv(x, w; executor = 42)
    # enum values are accepted equivalently to the Symbols
    pe = plan_conv(x, w; kernel = CacheAwareConv.KernelLV, executor = CacheAwareConv.ExecPolyester)
    @test pe.kernel === CacheAwareConv.KernelLV
    @test pe.executor === CacheAwareConv.ExecPolyester
    xb = randn(BigFloat, 9, 2, 1)
    wb = randn(BigFloat, 3, 2, 4)
    @test_throws ArgumentError plan_conv(xb, wb; kernel = :lv)
    @test_throws ArgumentError plan_conv(xb, wb; kernel = :simd)
    @test plan_conv(xb, wb).kernel === CacheAwareConv.KernelScalar
end
