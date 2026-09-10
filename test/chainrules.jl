using CacheAwareConv, Test, Random, Zygote, ChainRulesCore, FiniteDifferences
using CacheAwareConv: conv_bias, output_size, geometry

@testset "Zygote through conv_bias (rrule)" begin
    rng = MersenneTwister(20)
    for (T, rtol) in ((Float64, 1.0e-7), (Float32, 1.0e-3)), stride in (1, 2), groups in (1, 2)
        x = randn(rng, T, 11, 9, 2groups, 2)
        w = randn(rng, T, 3, 3, 2, 3groups)
        b = randn(rng, T, 3groups)
        p = plan_conv(x, w; stride, pad = 1, groups)
        loss(x, w, b) = sum(abs2, conv_bias(x, w, b, p))
        gx, gw, gb = Zygote.gradient(loss, x, w, b)
        ȳ = 2 .* conv_bias(x, w, b, p)
        gx_ref = similar(x)
        reference_∇conv_data!(gx_ref, ȳ, w, geometry(p))
        gw_ref = similar(w)
        reference_∇conv_filter!(gw_ref, x, ȳ, geometry(p))
        gb_ref = vec(sum(ȳ; dims = (1, 2, 4)))
        @test isapprox(gx, gx_ref; rtol)
        @test isapprox(gw, gw_ref; rtol)
        @test isapprox(gb, gb_ref; rtol)
        if T === Float64
            fd = FiniteDifferences.central_fdm(5, 1)
            gx_fd, gw_fd, gb_fd = FiniteDifferences.grad(fd, loss, x, w, b)
            @test isapprox(gx, gx_fd; rtol = 1.0e-6)
            @test isapprox(gw, gw_fd; rtol = 1.0e-6)
            @test isapprox(gb, gb_fd; rtol = 1.0e-6)
        end
        # no bias, and the keyword front-end
        g2 = Zygote.gradient((x, w) -> sum(conv_bias(x, w, nothing, p)), x, w)
        @test size(g2[1]) == size(x) && size(g2[2]) == size(w)
        g3 = Zygote.gradient((x, w) -> sum(abs2, conv(x, w, p)), x, w)
        @test isapprox(g3[2], 2 .* (similar(w) |> w̄ -> reference_∇conv_filter!(w̄, x, conv(x, w, p), geometry(p))); rtol)
    end
end

@testset "rrule for the activation path" begin
    rng = MersenneTwister(21)
    x = randn(rng, 8, 8, 2, 1)
    w = randn(rng, 3, 3, 2, 2)
    p = plan_conv(x, w; pad = 1)
    f(x, w) = sum(tanh.(conv_bias(x, w, nothing, p)))
    g = Zygote.gradient(f, x, w)
    fd = FiniteDifferences.central_fdm(5, 1)
    gfd = FiniteDifferences.grad(fd, f, x, w)
    @test isapprox(g[1], gfd[1]; rtol = 1.0e-6)
    @test isapprox(g[2], gfd[2]; rtol = 1.0e-6)
end
