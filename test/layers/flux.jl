using CacheAwareConv, Test, Random, Flux, Zygote
using CacheAwareConv: SamePad

@testset "FluxConv matches Flux.Conv" begin
    rng = MersenneTwister(50)
    for (k, ch, kw) in (
            ((3, 3), 3 => 4, (; pad = 1)),
            ((3, 3), 4 => 6, (; pad = SamePad(), stride = 2, groups = 2)),
            ((5,), 2 => 3, (; pad = 2, dilation = 2)),
            ((3, 3), 3 => 4, (; pad = 1, bias = false)),
            ((2, 3, 2), 2 => 2, (; pad = (1, 0, 1))),
        )
        kw_flux = (; (k => (v isa SamePad ? Flux.SamePad() : v) for (k, v) in pairs(kw))...)
        theirs = Flux.Conv(k, ch, tanh; kw_flux...)
        ours = FluxConv(theirs.weight, theirs.bias, tanh; stride = theirs.stride, pad = theirs.pad, dilation = theirs.dilation, groups = theirs.groups)
        x = randn(rng, Float32, ntuple(i -> 9 + i, length(k))..., ch[1], 2)
        @test ours(x) ≈ theirs(x) rtol = 1.0e-4
        @test length(ours.plans) == 1
        g1 = Zygote.gradient(m -> sum(abs2, m(x)), ours)[1]
        g2 = Zygote.gradient(m -> sum(abs2, m(x)), theirs)[1]
        @test g1.weight ≈ g2.weight rtol = 1.0e-3
        if theirs.bias !== false
            @test g1.bias ≈ g2.bias rtol = 1.0e-3
        end
        @test Flux.trainables(ours) == Flux.trainables(theirs)
    end
    l = FluxConv((3, 3), 3 => 4, relu; pad = 1, stride = 2)
    @test occursin("FluxConv((3, 3), 3 => 4, relu", sprint(show, l))
    @test size(l.weight) == (3, 3, 3, 4)
    # Float64 input is converted to the weight eltype
    y = l(randn(rng, 10, 10, 3, 1))
    @test eltype(y) == Float32
    # Training step works
    model = Flux.Chain(FluxConv((3, 3), 1 => 2, relu; pad = 1), Flux.flatten, Flux.Dense(2 * 64 => 1))
    opt = Flux.setup(Flux.Adam(), model)
    xb = randn(rng, Float32, 8, 8, 1, 4)
    yb = randn(rng, Float32, 1, 4)
    l0 = Flux.mse(model(xb), yb)
    for _ in 1:20
        gs = Zygote.gradient(m -> Flux.mse(m(xb), yb), model)[1]
        Flux.update!(opt, model, gs)
    end
    @test Flux.mse(model(xb), yb) < l0
end
