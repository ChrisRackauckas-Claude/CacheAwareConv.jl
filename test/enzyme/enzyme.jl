using CacheAwareConv, Test, Random, Enzyme
using CacheAwareConv: conv_core!, conv_bias, output_size, geometry

function reference_grads(x, w, b, ȳ, p)
    gx = similar(x)
    reference_∇conv_data!(gx, ȳ, w, geometry(p))
    gw = similar(w)
    reference_∇conv_filter!(gw, x, ȳ, geometry(p))
    N = ndims(ȳ)
    gb = b === nothing ? nothing : vec(sum(ȳ; dims = (ntuple(identity, N - 2)..., N)))
    return gx, gw, gb
end

@testset "Enzyme reverse: conv_core! ($T)" for (T, rtol) in ((Float64, 1.0e-8), (Float32, 1.0e-3))
    rng = MersenneTwister(30)
    for stride in (1, 2), groups in (1, 2)
        x = randn(rng, T, 12, 9, 2groups, 2)
        w = randn(rng, T, 3, 3, 2, 3groups)
        b = randn(rng, T, 3groups)
        p = plan_conv(x, w; stride, pad = 1, groups)
        y = zeros(T, output_size(p))
        function loss!(y, x, w, b, p)
            conv_core!(y, x, w, b, p)
            return sum(abs2, y)
        end
        dx = zero(x)
        dw = zero(w)
        db = zero(b)
        dy = zero(y)
        Enzyme.autodiff(Reverse, loss!, Active, Duplicated(y, dy), Duplicated(x, dx), Duplicated(w, dw), Duplicated(b, db), Const(p))
        ȳ = 2 .* conv_bias(x, w, b, p)
        gx, gw, gb = reference_grads(x, w, b, ȳ, p)
        @test isapprox(dx, gx; rtol)
        @test isapprox(dw, gw; rtol)
        @test isapprox(db, gb; rtol)
        # Const x, active w only; bias = nothing
        dw2 = zero(w)
        y2 = zeros(T, output_size(p))
        loss2!(y, x, w, p) = (conv_core!(y, x, w, nothing, p); sum(abs2, y))
        Enzyme.autodiff(Reverse, loss2!, Active, Duplicated(y2, zero(y2)), Const(x), Duplicated(w, dw2), Const(p))
        ȳ2 = 2 .* conv_bias(x, w, nothing, p)
        @test isapprox(dw2, reference_grads(x, w, nothing, ȳ2, p)[2]; rtol)
    end
end

@testset "Enzyme reverse through the allocating conv (Float64)" begin
    rng = MersenneTwister(31)
    x = randn(rng, 10, 10, 3, 2)
    w = randn(rng, 3, 3, 3, 4)
    p = plan_conv(x, w; pad = 1)
    f(x, w) = sum(abs2, conv(x, w, p))
    dx, dw = Enzyme.gradient(Reverse, f, x, w)
    ȳ = 2 .* conv(x, w, p)
    gx, gw, _ = reference_grads(x, w, nothing, ȳ, p)
    @test dx ≈ gx
    @test dw ≈ gw
    # activation applied by broadcast after the conv
    g(x, w) = sum(tanh.(conv_bias(x, w, nothing, p)))
    dx2, dw2 = Enzyme.gradient(Reverse, g, x, w)
    z = conv_bias(x, w, nothing, p)
    ȳ2 = 1 .- tanh.(z) .^ 2
    gx2, gw2, _ = reference_grads(x, w, nothing, ȳ2, p)
    @test dx2 ≈ gx2
    @test dw2 ≈ gw2
end

@testset "Enzyme forward: conv_core!" begin
    rng = MersenneTwister(32)
    x = randn(rng, 10, 8, 2, 1)
    w = randn(rng, 3, 3, 2, 3)
    b = randn(rng, 3)
    p = plan_conv(x, w; pad = 1)
    ẋ = randn(rng, size(x))
    ẇ = randn(rng, size(w))
    ḃ = randn(rng, size(b))
    y = zeros(output_size(p))
    ẏ = zeros(output_size(p))
    Enzyme.autodiff(Forward, conv_core!, Duplicated(y, ẏ), Duplicated(x, ẋ), Duplicated(w, ẇ), Duplicated(b, ḃ), Const(p))
    expected = conv_bias(ẋ, w, ḃ, p) .+ conv_bias(x, ẇ, nothing, p)
    @test y ≈ conv_bias(x, w, b, p)
    @test ẏ ≈ expected
    # only w active
    ẏ2 = zeros(output_size(p))
    Enzyme.autodiff(Forward, conv_core!, Duplicated(copy(y), ẏ2), Const(x), Duplicated(w, ẇ), Const(b), Const(p))
    @test ẏ2 ≈ conv_bias(x, ẇ, nothing, p)
    # through the allocating function
    # (b is captured as a constant here, so its tangent does not appear)
    ẏ3 = Enzyme.autodiff(Forward, Const((x, w) -> conv_bias(x, w, b, p)), Duplicated(x, ẋ), Duplicated(w, ẇ))[1]
    @test ẏ3 ≈ conv_bias(ẋ, w, nothing, p) .+ conv_bias(x, ẇ, nothing, p)
end

@testset "Enzyme reverse: gradient functions are themselves differentiable" begin
    # ∇conv_data is linear in ȳ; differentiate a loss of it w.r.t. ȳ.
    rng = MersenneTwister(33)
    x = randn(rng, 9, 9, 2, 1)
    w = randn(rng, 3, 3, 2, 2)
    p = plan_conv(x, w; pad = 1)
    ȳ = randn(rng, output_size(p))
    f(ȳ) = sum(abs2, ∇conv_data(ȳ, w, p))
    dȳ = Enzyme.gradient(Reverse, f, ȳ)[1]
    # d/dȳ ‖Aᵀȳ‖² = 2 A Aᵀ ȳ, with A the forward conv
    @test dȳ ≈ 2 .* conv(∇conv_data(ȳ, w, p), w, p)
end
