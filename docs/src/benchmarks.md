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

## Stencils (PDE-style) against the roofline

Single input and output channel, `Float64`, 2²⁴ points (1D) or a 4096² grid
(2D), no padding. The roofline is `max(flops / FMA peak, bytes / copy bandwidth)`
with both peaks measured in the same process: 45 GFLOPS per core for the dense
microkernel, 34 GB/s single-thread and 145 GB/s 16-thread `copyto!`, and
577 GFLOPS aggregate for 16 unpinned threads running the dense kernel
concurrently (this machine has 32 cores with SMT and lower all-core clocks).

| stencil | 1 thread | % of roofline | 16 threads | % of roofline (577 GFLOPS / 145 GB/s) |
|---|---|---|---|---|
| 1D k=3 | 24.6 GB/s | 73% (bandwidth) | 116 GB/s | 80% |
| 1D k=7 | 19.8 GB/s | 59% (bandwidth) | 121 GB/s | 83% |
| 1D k=11 | 21 GFLOPS | 48% (FMA) | 126 GFLOPS | 63% (bandwidth) |
| 1D k=19 | 25 GFLOPS | 55% (FMA) | 162 GFLOPS | 28% (FMA) |
| 1D k=51 | 27 GFLOPS | 61% (FMA) | 166 GFLOPS | 29% (FMA) |
| 2D 3×3 | 15.1 GB/s | 45% (bandwidth) | 95 GB/s | 65% |
| 2D 5×5 | 25 GFLOPS | 55% (FMA) | 161 GFLOPS | 28% (FMA) |
| 2D 9×9 | 28 GFLOPS | 62% (FMA) | 269 GFLOPS | 47% (FMA) |
| 2D 13×13 | 28 GFLOPS | 62% (FMA) | 406 GFLOPS | 70% (FMA) |

A hand-written `@inbounds @simd` stencil loop over the same data runs 3–8×
slower than `conv!` on one thread.

Single-thread, the FMA-bound stencils sit at a flat 55–62% of peak from
k = 11 up to 13×13: the single-channel kernel issues one input vector load per
fused multiply-add (`MR × 1` tile with hoisted weights), so it is bound by the
load ports, not the FMA units. Reusing loaded vectors across neighbouring taps
with lane shifts would lift this; it is the main remaining single-core
optimisation for stencils.
