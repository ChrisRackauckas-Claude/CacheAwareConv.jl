# Compare against NNlib's im2col + BLAS convolution on representative shapes.
# Run with e.g. `julia --project=benchmark -t 16 benchmark/benchmarks.jl`.
using CacheAwareConv, NNlib, BenchmarkTools, Printf
using CacheAwareConv: conv!, plan_conv, output_size, geometry

function bench(name, T, xs, ws; pad = 0, stride = 1, threads = Threads.nthreads())
    x = randn(T, xs)
    w = randn(T, ws)
    p = plan_conv(x, w; pad, stride, nthreads = threads)
    y = similar(x, T, output_size(p))
    conv!(y, x, w, p)
    t = @belapsed conv!($y, $x, $w, $p)
    g = geometry(p)
    flops = 2 * prod(g.ysize) * prod(ntuple(i -> ws[i], length(ws) - 2)) * (xs[end - 1] ÷ g.groups)
    cd = NNlib.DenseConvDims(x, w; padding = pad, stride)
    yn = NNlib.conv(x, w, cd)
    tn = @belapsed NNlib.conv!($yn, $x, $w, $cd)
    err = maximum(abs.(y .- yn)) / maximum(abs.(yn))
    @printf(
        "%-30s %-10s thr=%2d  ours %8.3f ms (%7.1f GFLOPS)   NNlib %8.3f ms (%7.1f GFLOPS)  speedup %5.2fx  relerr %.1e  [Kc=%d tile=%s]\n",
        name, T, threads, t * 1.0e3, flops / t / 1.0e9, tn * 1.0e3, flops / tn / 1.0e9, tn / t, err, p.Kc, p.tile
    )
    return nothing
end

for thr in unique((1, Threads.nthreads()))
    bench("resnet 56x56 64->64 3x3 b8", Float32, (56, 56, 64, 8), (3, 3, 64, 64); pad = 1, threads = thr)
    bench("resnet 224x224 3->64 7x7 s2", Float32, (224, 224, 3, 8), (7, 7, 3, 64); pad = 3, stride = 2, threads = thr)
    bench("28x28 128->128 3x3 b16", Float32, (28, 28, 128, 16), (3, 3, 128, 128); pad = 1, threads = thr)
    bench("7x7 512->512 3x3 b16", Float32, (7, 7, 512, 16), (3, 3, 512, 512); pad = 1, threads = thr)
    bench("1D 65536 x 16->32 k9 b4", Float32, (65536, 16, 4), (9, 16, 32); pad = 4, threads = thr)
    bench("3D 32^3 8->16 k3 b2", Float32, (32, 32, 32, 8, 2), (3, 3, 3, 8, 16); pad = 1, threads = thr)
    bench("image filter 4096^2 1->1 3x3", Float32, (4096, 4096, 1, 1), (3, 3, 1, 1); pad = 1, threads = thr)
    bench("F64 56x56 64->64 3x3 b8", Float64, (56, 56, 64, 8), (3, 3, 64, 64); pad = 1, threads = thr)
    bench("ComplexF32 56x56 16->16 3x3 b4", ComplexF32, (56, 56, 16, 4), (3, 3, 16, 16); pad = 1, threads = thr)
end
