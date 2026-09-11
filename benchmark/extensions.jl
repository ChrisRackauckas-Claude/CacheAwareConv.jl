# kernel = :lv (LoopVectorization @turbo) and executor = :polyester vs the
# defaults, on representative shapes. Single-thread for the kernel comparison,
# threaded for the executor comparison.
using CacheAwareConv, BenchmarkTools, Printf
using LoopVectorization, Polyester
using CacheAwareConv: output_size

function gflops(p, x, w, t)
    S = CacheAwareConv.spatial_dims(p)
    flops = 2 * prod(output_size(p)) * prod(CacheAwareConv.kernel_size(p)[1:S]) * (CacheAwareConv.channels_in(p) ÷ CacheAwareConv.groups(p))
    return flops / t / 1.0e9
end

function bench_kernel(name, x, w; kwargs...)
    println("== $name  (nthreads=1) ==")
    for kern in (:simd, :lv, :scalar)
        p = plan_conv(x, w; kernel = kern, nthreads = 1, kwargs...)
        y = conv(x, w, p)
        t = @belapsed conv!($y, $x, $w, $p)
        @printf("  %-7s %9.3f ms   %7.1f GFLOPS\n", kern, t * 1.0e3, gflops(p, x, w, t))
    end
    return nothing
end

function bench_executor(name, x, w; kwargs...)
    nthreads = Threads.nthreads()
    println("== $name  (nthreads=$nthreads) ==")
    for exec in (:spawn, :polyester)
        for kern in (:simd, :lv)
            p = plan_conv(x, w; kernel = kern, executor = exec, nthreads, kwargs...)
            y = conv(x, w, p)
            t = @belapsed conv!($y, $x, $w, $p)
            @printf("  %-9s %-7s %9.3f ms   %7.1f GFLOPS\n", exec, kern, t * 1.0e3, gflops(p, x, w, t))
        end
    end
    return nothing
end

# Compute-bound: deep channel counts, 3x3 kernels (the NN layer regime)
x1 = randn(Float32, 56, 56, 64, 1);   w1 = randn(Float32, 3, 3, 64, 64)
x2 = randn(Float32, 28, 28, 128, 4);  w2 = randn(Float32, 3, 3, 128, 128)
# Stencil regime: single channel (row-blocked kernel under :simd)
x3 = randn(Float32, 1024, 1024, 1, 1); w3 = randn(Float32, 5, 5, 1, 1)
# Wide shallow: memory-heavy
x4 = randn(Float32, 256, 256, 3, 8);  w4 = randn(Float32, 7, 7, 3, 16)
# Small conv — launch overhead dominates (where @batch vs @spawn differs)
x5 = randn(Float32, 14, 14, 8, 8);    w5 = randn(Float32, 3, 3, 8, 8)

bench_kernel("56×56×64→64 3×3 pad=1", x1, w1; pad = 1)
bench_kernel("28×28×128→128 3×3 pad=1 batch=4", x2, w2; pad = 1)
bench_kernel("1024×1024 1→1 5×5 pad=2 (stencil)", x3, w3; pad = 2)
bench_kernel("256×256 3→16 7×7 pad=3 batch=8", x4, w4; pad = 3)

bench_executor("56×56×64→64 3×3 pad=1", x1, w1; pad = 1)
bench_executor("28×28×128→128 3×3 pad=1 batch=4", x2, w2; pad = 1)
bench_executor("14×14×8→8 3×3 batch=8 (tiny)", x5, w5; pad = 1)
