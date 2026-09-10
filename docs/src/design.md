# [Design: cache-aware blocking](@id design)

A direct convolution is closer to a BLAS-2 operation than to a matrix
multiplication: every input element is combined with only `K × C_out`
weights (with `K` the number of kernel taps), so the reuse that can be
extracted per input element is bounded by the kernel, not by the problem
size. The blocking therefore streams the input through the cache hierarchy
once while keeping the weights resident, rather than the three-level
Goto/BLIS panel scheme of a GEMM.

## Packed input tiles

Each *work item* is one batch index and one spatial block of the output.
Before any arithmetic, the input region it needs (with the halo of
`(K_i − 1)·dilation_i` extra rows/columns) is copied into a per-task buffer
that is

- **zero padded** on all borders, so the kernel never checks bounds,
- **stride-phase deinterleaved** along the first spatial dimension: for
  stride `s` the padded row is split into `s` interleaved phases, so that the
  inputs feeding consecutive output columns for a fixed tap are contiguous
  and can be read with plain vector loads for any stride,
- converted to the compute type (`Float32` for `Float16`, split planes for
  complex numbers).

Dilation is pure address arithmetic on this layout. Every kernel tap is then
a constant offset into the packed tile, precomputed once per plan
(`ConvPlan.taps`).

## The microkernel

The register tile is `MR` vectors of `V` lanes along the output width times
`NR` output channels (times three planes for complex data). For each input
channel and tap the kernel issues `MR` vector loads of the packed tile,
`NR` scalar broadcasts of the packed weights, and `MR·NR` fused
multiply-adds. `MR` and `NR` are picked from `HostCPUFeatures.register_count()`:

| vector registers | real `(MR, NR)` | complex `(MR, NR)` |
|---|---|---|
| 32 (AVX-512) | (4, 6) | (2, 4) |
| 16 (AVX2, NEON) | (2, 4) | (1, 3) |
| 8 (SSE) | (1, 4) | (1, 2) |

Weights are repacked so that the `NR` values a tap needs are adjacent, with
kernel flipping applied during packing. Partial tiles at the right edge use a
masked load/store for the last vector; partial output-channel tiles use a
smaller `NR` instantiation. On a Zen 4 core the Float32 kernel alone runs at
about 88% of the fused-multiply-add peak.

For complex numbers the three planes `re`, `im`, `re + im` of the input and
`wr`, `wi`, `wr + wi` of the weights are multiplied plane-wise (Gauss'
trick); `re = a₁ − a₂` and `im = a₃ − a₁ − a₂` are recombined once per tile.
This was measured about 30% faster than the four-multiply form because it
needs three loads per three FMAs and fewer accumulators per useful product.

## Narrow outputs: flat-row mode

When the output width is not a multiple of the vector width, the last vector
of every row wastes lanes; for the 7×7 and 14×14 feature maps common late in a
CNN that costs more than half the throughput. For unit strides the packed
tile has a constant row pitch, so the kernel can instead treat the whole tile
(all rows, including the halo columns between them) as one long vector and
store into a per-task output buffer with the same geometry. The halo positions
produce garbage that the final copy skips. The plan picks this mode when it
wastes at least 15% fewer lanes than row-by-row traversal (shown as `flat` in
the plan summary; `flat = true/false` forces it). The weight-gradient kernel
uses the same trick, with the packed output-gradient tile zero everywhere
outside the valid region.

## Single-channel stencils

With one input and one output channel the `MR × NR` tile degenerates (`NR = 1`)
and every fused multiply-add needs its own input vector load, which is the
binding constraint on AVX-512 cores that double-pump 512-bit loads. For
`C_in = C_out = 1` with unit row stride, the plan therefore switches to a
row-blocked kernel: `MRH` output rows × `MRW` vectors per register tile, with
the `K` weights held in registers. Taps are grouped by their runtime base
(stride phase along the width, taps along the third and higher spatial
dimensions) and, within a group, described by compile-time (row offset, lane
shift) pairs that become a type parameter of the plan. Each input row of a
group is loaded once per lane shift and feeds every output row that overlaps
it, dividing loads per FMA by roughly the kernel height. Partial row blocks
and the column tail fall back to the general kernel. The gradient with
respect to the input is a stencil too and takes the same path through the
transposed plan.

## Choosing the blocks from the cache sizes

With `sz` the size of a packed element and `K = ∏ kernel size`:

1. **L1 ← weight panel.** `Kc` input channels per block, the largest with
   `K · Kc · NR · planes · sz ≤ L1 / 2`. The panel is reused across every
   output column and row of a tile.
2. **L2 ← packed input tile.** The spatial output block per work item is the
   largest for which the packed tile (`Kc` channels with halo) fits in
   `L2 / 2`; trailing spatial dimensions shrink first, then the width in
   multiples of `MR · V`, and if even one tile row is too large `Kc` is halved.
   The tile is reused across every output-channel tile.
3. **L3 ← packed weights.** `Nc` output channels per sweep of the packed
   tile, so that `K · Kc · Nc · planes · sz ≤ L3_per_core / 2`.
4. The tile is split further if that is needed to give every task at least
   one work item.

When `Kc < C_in`, output tiles are accumulated through memory: the first
channel block stores, later blocks load-add-store. The bias/activation
epilogue runs after the last block, while the tile is in L1.

Memory traffic is therefore one read of the input per channel block (once in
the common case), one write of the output, and the weights. Packing copies
happen tile by tile inside L2 and add no DRAM traffic, so memory-bound shapes
(few channels, small kernels, large images) run at streaming bandwidth once
enough tasks are used.

## Gradients

- The input gradient is a transposed convolution: the output gradient is
  zero-stuffed by the forward stride (done in the packing step, not in
  memory), the padding is transposed, and the weights are viewed with the
  channel roles exchanged, flipped, and conjugated. It reuses the forward
  kernel and blocking.
- The weight gradient is a reduction over positions and batch for every
  (tap, input channel, output channel) triple. It uses the same packed input
  tile plus a packed output-gradient tile and a second register-tiled kernel:
  `MRc` input channels × `NRc` output channels, each a vector along the width,
  reduced horizontally once per tile. Partial gradients are accumulated per
  task and reduced in a fixed order, so results do not depend on the thread
  count.

## Threading

Work items `(batch, spatial block)` are distributed over
`min(nthreads, items)` tasks in contiguous chunks with `Threads.@spawn`; each
task owns its packing buffers. Every output element is written by exactly
one task, so the result is bitwise identical for any thread count.

## Limits and future work

- Direct convolution is the right algorithm up to roughly 7×7 kernels. For
  much larger kernels an FFT-based method needs far fewer operations; the
  blocking here keeps the direct kernel at full FMA rate for any kernel size,
  but does not change the operation count.
- Flat-row mode requires unit strides; strided narrow outputs still waste
  lanes in the last vector of each row.
- The stride-`s` data gradient zero-stuffs the output gradient and therefore
  performs `∏ s` times the necessary multiply-adds; a phase decomposition
  would remove this.
