using CacheAwareConv, Test, Random
using CacheAwareConv: output_size

@testset "results are independent of the number of tasks" begin
    rng = MersenneTwister(60)
    x = randn(rng, Float32, 37, 29, 6, 3)
    w = randn(rng, Float32, 3, 3, 6, 10)
    ȳ = nothing
    ref = nothing
    refx = nothing
    refw = nothing
    for nthreads in (1, 2, 5, 64)
        p = plan_conv(x, w; pad = 1, nthreads)
        @test p.nthreads <= nthreads
        y = conv(x, w, p)
        if ȳ === nothing
            ȳ = randn(rng, Float32, size(y))
            ref = y
            refx = ∇conv_data(ȳ, w, p)
            refw = ∇conv_filter(x, ȳ, p)
        else
            @test y == ref                       # bitwise identical: each output written once
            @test ∇conv_data(ȳ, w, p) == refx
            @test ∇conv_filter(x, ȳ, p) ≈ refw    # per-task partials reduced in fixed order
        end
    end
end

@testset "more tasks than natural work items splits the tile" begin
    p = ConvPlan(Float32, (64, 64, 3, 1), (3, 3, 3, 8); pad = 1, nthreads = 8)
    @test p.nthreads == 8
    @test prod(p.nblocks) >= 8
    x = randn(Float32, 64, 64, 3, 1)
    w = randn(Float32, 3, 3, 3, 8)
    y = conv(x, w, p)
    yr = similar(y)
    reference_conv!(yr, x, w, p.geom)
    @test y ≈ yr
end
