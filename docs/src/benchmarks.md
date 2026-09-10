# Benchmarks

Measured on one AMD EPYC 9354 (Zen 4, AVX-512, 32 KiB L1d, 1 MiB L2,
32 MiB L3 per 32 cores), Julia 1.12, `Float32`, compared with NNlib's
im2col + BLAS path (which uses all threads through the BLAS). Scripts are in
`benchmark/`.

| shape | ours, 1 thread | ours, 16 threads | NNlib | speedup (16 vs NNlib) |
|---|---|---|---|---|
| 56×56, 64→64, 3×3, batch 8 | 83 GFLOPS | 852 GFLOPS | 131 GFLOPS | 6.5× |
| 224×224, 3→64, 7×7, stride 2, batch 8 | 95 | 652 | 156 | 4.2× |
| 28×28, 128→128, 3×3, batch 16 | 85 | 583 | 329 | 1.8× |
| 1D 65536, 16→32, k=9, batch 4 | 88 | 636 | 100 | 6.4× |
| 3D 32³, 8→16, k=3, batch 2 | 86 | 493 | 38 | 13× |
| 4096² image, 1→1, 3×3 (memory bound) | 13 | 159 | 2.6 | 62× |

Single-core throughput is roughly 80% of the core's FMA peak for
compute-bound shapes. The microkernel alone reaches about 88%; the rest is
packing and partial tiles. The memory-bound image filter runs at streaming
bandwidth once enough threads are used (see `benchmark/streaming.jl`, which
compares against `copyto!`).
