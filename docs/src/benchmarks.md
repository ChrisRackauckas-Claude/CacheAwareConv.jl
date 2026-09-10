# Benchmarks

Measured on one AMD EPYC 9354 (Zen 4, AVX-512, 32 KiB L1d, 1 MiB L2,
32 MiB L3 per 32 cores), Julia 1.12, `Float32`, compared with NNlib's
im2col + BLAS path (which uses all threads through the BLAS). Scripts are in
`benchmark/`.

| shape | ours, 1 thread | ours, 16 threads | NNlib | speedup (16 vs NNlib) |
|---|---|---|---|---|
| 56×56, 64→64, 3×3, batch 8 | 83 GFLOPS | 950 GFLOPS | 121 GFLOPS | 7.9× |
| 224×224, 3→64, 7×7, stride 2, batch 8 | 94 | 515 | 153 | 3.4× |
| 28×28, 128→128, 3×3, batch 16 | 84 | 1088 | 316 | 3.4× |
| 7×7, 512→512, 3×3, batch 16 (flat-row mode) | 72 | 665 | 379 | 1.8× |
| 1D 65536, 16→32, k=9, batch 4 | 86 | 497 | 115 | 4.3× |
| 3D 32³, 8→16, k=3, batch 2 | 86 | 488 | 37 | 13× |
| 4096² image, 1→1, 3×3 (memory bound) | 13 | 151 | 2.2 | 69× |
| Float64 56×56, 64→64, 3×3, batch 8 | 48 | 538 | 101 | 5.3× |
| ComplexF32 56×56, 16→16, 3×3, batch 4 | 17 (real-FLOP equiv.) | 96 | 19 | 5.1× |

Single-core throughput is roughly 80% of the core's FMA peak for
compute-bound shapes. The microkernel alone reaches about 88%; the rest is
packing and partial tiles. The memory-bound image filter runs at streaming
bandwidth once enough threads are used: `benchmark/streaming.jl` on a
16384² `Float32` image (2 GB in + out) reaches 54 GB/s with 16 threads
against 36 GB/s for a single-threaded `copyto!` of the same data; one thread
alone is limited to 4 GB/s by the per-tap kernel-call overhead of the
single-channel case.

Multi-threaded numbers on this shared machine varied by up to 1.8× between
runs depending on other load; the table shows an idle run.
