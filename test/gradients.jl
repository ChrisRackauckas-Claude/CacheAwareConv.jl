using CacheAwareConv, Test, Random
using CacheAwareConv: CacheInfo, output_size, geometry

const GRTOL = Dict(Float16 => 3.0e-2, Float32 => 1.0e-4, Float64 => 1.0e-10, ComplexF32 => 1.0e-4, ComplexF64 => 1.0e-10, BigFloat => 1.0e-20)
grandarr(rng, ::Type{T}, dims) where {T} = randn(rng, T, dims)
grandarr(rng, ::Type{Float16}, dims) = Float16.(randn(rng, Float32, dims))
grandarr(rng, ::Type{BigFloat}, dims) = BigFloat.(randn(rng, Float64, dims))

function check_grads(rng, T, spatial, k, cin, cout; cache = CacheAwareConv.cache_info(), nthreads = Threads.nthreads(), kwargs...)
    groups = get(kwargs, :groups, 1)
    x = grandarr(rng, T, (spatial..., cin, 2))
    w = grandarr(rng, T, (k..., cin ÷ groups, cout))
    p = ConvPlan(T, size(x), size(w); cache, nthreads, kwargs...)
    ȳ = grandarr(rng, T, output_size(p))
    rtol = GRTOL[T]
    x̄ = ∇conv_data(ȳ, w, p)
    x̄r = similar(x)
    reference_∇conv_data!(x̄r, ȳ, w, geometry(p))
    @test isapprox(x̄, x̄r; rtol, atol = rtol)
    w̄ = ∇conv_filter(x, ȳ, p)
    w̄r = similar(w)
    reference_∇conv_filter!(w̄r, x, ȳ, geometry(p))
    @test isapprox(w̄, w̄r; rtol, atol = rtol * 10)
    # accumulate
    x̄2 = copy(x̄)
    ∇conv_data!(x̄2, ȳ, w, p; accumulate = true)
    @test isapprox(x̄2, 2 .* x̄r; rtol, atol = rtol)
    w̄2 = copy(w̄)
    ∇conv_filter!(w̄2, x, ȳ, p; accumulate = true)
    @test isapprox(w̄2, 2 .* w̄r; rtol, atol = rtol * 10)
    return p
end

@testset "gradient grid vs reference" begin
    rng = MersenneTwister(10)
    for S in 1:3, stride in (1, 2, 3), dil in (1, 2), flipped in (false, true), groups in (1, 2)
        spatial = ntuple(i -> rand(rng, 5:10), S)
        k = ntuple(i -> rand(rng, 1:3), S)
        for pad in (0, 1, ntuple(i -> rand(rng, 0:2), 2S))
            try
                ConvGeometry((spatial..., 2groups, 2), (k..., 2, 3groups); stride, pad, dilation = dil, groups, flipped)
            catch e
                e isa DimensionMismatch && continue
                rethrow()
            end
            check_grads(rng, Float64, spatial, k, 2groups, 3groups; stride, pad, dilation = dil, groups, flipped)
        end
    end
end

@testset "gradient element types and blocking" begin
    rng = MersenneTwister(11)
    small = CacheInfo(; l1 = 512, l2 = 4096, l3 = 8192)
    for T in (Float32, Float16, ComplexF32, ComplexF64)
        check_grads(rng, T, (23, 9), (3, 3), 10, 7; pad = 1, stride = 2)
        check_grads(rng, T, (23, 9), (3, 3), 10, 7; pad = (1, 0, 2, 1), cache = small)
        check_grads(rng, T, (40,), (5,), 6, 6; pad = 2, dilation = 2, groups = 2)
    end
    check_grads(rng, BigFloat, (9, 6), (3, 2), 2, 3; pad = 1)
    check_grads(rng, Float32, (70, 3), (3, 3), 9, 13; pad = 1)   # width and channel tails
end

@testset "gradient buffers are allocated once" begin
    x = randn(Float32, 24, 24, 8, 2)
    w = randn(Float32, 3, 3, 8, 16)
    p = plan_conv(x, w; pad = 1, nthreads = 1)
    ȳ = randn(Float32, output_size(p))
    x̄ = similar(x)
    w̄ = similar(w)
    ∇conv_data!(x̄, ȳ, w, p)
    ∇conv_filter!(w̄, x, ȳ, p)
    @test (@allocated ∇conv_data!(x̄, ȳ, w, p)) == 0
    @test (@allocated ∇conv_filter!(w̄, x, ȳ, p)) == 0
    p2 = plan_conv(x, w; pad = 1, gradients = true)
    @test p2.grad.state !== nothing
end
