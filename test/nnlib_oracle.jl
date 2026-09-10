# Cross-check the semantics (not just the reference implementation) against NNlib.
using CacheAwareConv, Test, Random
import NNlib

@testset "conv and gradients agree with NNlib" begin
    rng = MersenneTwister(70)
    for S in 1:3, stride in (1, 2), dil in (1, 2), flipped in (false, true), groups in (1, 2)
        spatial = ntuple(i -> rand(rng, 6:10), S)
        k = ntuple(i -> rand(rng, 1:3), S)
        for pad in (0, ntuple(i -> rand(rng, 0:2), 2S))
            x = randn(rng, Float64, spatial..., 2groups, 2)
            w = randn(rng, Float64, k..., 2, 3groups)
            cd = try
                NNlib.DenseConvDims(x, w; stride, padding = pad, dilation = dil, groups, flipkernel = flipped)
            catch e
                e isa DimensionMismatch && continue
                rethrow()
            end
            p = plan_conv(x, w; stride, pad, dilation = dil, groups, flipped)
            y = conv(x, w, p)
            @test y ≈ NNlib.conv(x, w, cd)
            ȳ = randn(rng, size(y))
            @test ∇conv_data(ȳ, w, p) ≈ NNlib.∇conv_data(ȳ, w, cd)
            @test ∇conv_filter(x, ȳ, p) ≈ NNlib.∇conv_filter(x, ȳ, cd)
        end
    end
end
