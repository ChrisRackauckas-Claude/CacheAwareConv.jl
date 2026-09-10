# Memory-bound regime: a single-channel 3x3 filter over a multi-GB image.
# Reports achieved bandwidth (input read + output written) against copyto!.
# Run with e.g. `julia --project=benchmark -t 16 benchmark/streaming.jl`.
using CacheAwareConv, BenchmarkTools, Printf
using CacheAwareConv: conv!, plan_conv, output_size

function streaming(T, n; threads = Threads.nthreads())
    x = randn(T, n, n, 1, 1)
    w = randn(T, 3, 3, 1, 1)
    p = plan_conv(x, w; pad = 1, nthreads = threads)
    y = similar(x, T, output_size(p))
    conv!(y, x, w, p)
    t = @belapsed conv!($y, $x, $w, $p) samples = 5 evals = 1
    bytes = sizeof(x) + sizeof(y)
    yc = similar(x)
    tc = @belapsed copyto!($yc, $x) samples = 5 evals = 1
    @printf(
        "%s %dx%d (%.2f GB in+out) thr=%2d: conv %.1f ms = %.1f GB/s; copyto! %.1f ms = %.1f GB/s; ratio %.2f  [tile=%s]\n",
        T, n, n, bytes / 1.0e9, threads, t * 1.0e3, bytes / t / 1.0e9, tc * 1.0e3, 2sizeof(x) / tc / 1.0e9, (bytes / t) / (2sizeof(x) / tc), p.tile
    )
    return nothing
end

for thr in unique((1, Threads.nthreads()))
    streaming(Float32, 16384; threads = thr)     # 2 GB in + out
    streaming(Float64, 16384; threads = thr)     # 4 GB in + out
end
