using CacheAwareConv, Test, Random, Lux, Zygote, Enzyme
using CacheAwareConv: SamePad

@testset "LuxConv matches Lux.Conv" begin
    rng = MersenneTwister(40)
    for (k, ch, kw) in (
            ((3, 3), 3 => 4, (; pad = 1)),
            ((3, 3), 4 => 6, (; pad = SamePad(), stride = 2, groups = 2)),
            ((5,), 2 => 3, (; pad = 2, dilation = 2)),
            ((3, 3), 3 => 4, (; pad = 1, cross_correlation = true, use_bias = false)),
            ((2, 3, 2), 2 => 2, (; pad = (1, 0, 1))),
        )
        ours = LuxConv(k, ch, tanh; kw...)
        kw_lux = (; (k => (v isa SamePad ? Lux.SamePad() : v) for (k, v) in pairs(kw))...)
        theirs = Lux.Conv(k, ch, tanh; kw_lux...)
        ps, st = Lux.setup(MersenneTwister(1), ours)
        ps2, st2 = Lux.setup(MersenneTwister(1), theirs)
        @test size(ps.weight) == size(ps2.weight)
        @test haskey(ps, :bias) == haskey(ps2, :bias)
        @test Lux.parameterlength(ours) == Lux.parameterlength(theirs)
        # Use identical parameters
        ps_shared = haskey(ps2, :bias) ? (; weight = ps2.weight, bias = ps2.bias) : (; weight = ps2.weight)
        x = randn(rng, Float32, ntuple(i -> 9 + i, length(k))..., ch[1], 2)
        y, _ = ours(x, ps_shared, st)
        y2, _ = theirs(x, ps_shared, st2)
        @test size(y) == size(y2)
        @test y ≈ y2 rtol = 1.0e-4
        @test length(ours.plans) == 1
        ours(x, ps_shared, st)
        @test length(ours.plans) == 1
        # Zygote gradients agree
        g1 = Zygote.gradient(ps -> sum(abs2, first(ours(x, ps, st))), ps_shared)[1]
        g2 = Zygote.gradient(ps -> sum(abs2, first(theirs(x, ps, st2))), ps_shared)[1]
        @test g1.weight ≈ g2.weight rtol = 1.0e-3
        haskey(g1, :bias) && @test g1.bias ≈ g2.bias rtol = 1.0e-3
    end
    l = LuxConv((3, 3), 3 => 4, relu; pad = 1, stride = 2, groups = 1)
    @test occursin("LuxConv((3, 3), 3 => 4, relu", sprint(show, l))
    @test_throws DimensionMismatch LuxConv((3, 3), 3 => 4; groups = 2)
end

@testset "LuxConv with Enzyme" begin
    rng = MersenneTwister(41)
    layer = LuxConv((3, 3), 2 => 3, tanh; pad = 1)
    ps, st = Lux.setup(rng, layer)
    x = randn(rng, Float32, 8, 8, 2, 2)
    loss(ps, x) = sum(abs2, first(layer(x, ps, st)))
    dps = Enzyme.make_zero(ps)
    Enzyme.autodiff(Reverse, loss, Active, Duplicated(ps, dps), Const(x))
    g = Zygote.gradient(ps -> loss(ps, x), ps)[1]
    @test dps.weight ≈ g.weight rtol = 1.0e-3
    @test dps.bias ≈ g.bias rtol = 1.0e-3
end
