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
microkernel, 34 GB/s single-thread and 121–145 GB/s 16-thread `copyto!`, and
577 GFLOPS aggregate for 16 unpinned threads running the dense kernel
concurrently (this machine has 32 cores with SMT and lower all-core clocks).

| stencil | 1 thread | % of roofline | 16 threads | % of roofline (577 GFLOPS / 121 GB/s) |
|---|---|---|---|---|
| 1D k=3 | 24.4 GB/s | 72% (bandwidth) | 134 GB/s | 100% |
| 1D k=7 | 20.0 GB/s | 59% (bandwidth) | 119 GB/s | 98% |
| 1D k=11 | 21 GFLOPS | 47% (FMA) | 126 GFLOPS | 76% (bandwidth) |
| 1D k=19…51 | 25–27 GFLOPS | 55–61% (FMA) | 155–163 GFLOPS | 23–54% (FMA) |
| 2D 3×3 | 19.2 GB/s | 57% (bandwidth) | 108 GB/s | 89% |
| 2D 5×5 | 34 GFLOPS | 77% (FMA) | 205 GFLOPS | 36% (FMA) |
| 2D 7×7 | 37 GFLOPS | 82% (FMA) | 225 GFLOPS | 39% (FMA) |
| 2D 9×9 | 42 GFLOPS | 94% (FMA) | 262 GFLOPS | 45% (FMA) |
| 2D 11×11 | 35 GFLOPS | 77% (FMA) | | |
| 2D 13×13 | 42 GFLOPS | 93% (FMA) | | |

A hand-written `@inbounds @simd` stencil loop over the same data runs 3–8×
slower than `conv!` on one thread.

Two-dimensional (and higher) single-channel stencils use the row-blocked
kernel (`MRH` output rows × `MRW` vectors per register tile): an input row is
loaded once and feeds every output row that overlaps it, so the loads per
fused multiply-add drop by roughly the kernel height. That is what moves the
2D stencils from ~60% to 77–94% of the FMA peak. One-dimensional stencils
have no such reuse; with one 512-bit load per FMA they stay at 55–61% on this
core, where a cache-line-straddling load and a 512-bit lane shift (`valignq`)
both cost two cycles, so neither lane shifts nor 256-bit vectors help (both
were measured). The 16-thread FMA-bound numbers are limited by the driver
under load rather than by the kernel, which reaches 577 GFLOPS aggregate on
its own; that is the remaining stencil optimisation.
