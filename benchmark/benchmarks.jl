using CacheAwareConv, BenchmarkTools
using StableRNGs

const SUITE = BenchmarkGroup()
const rng = StableRNG(123)

# NCHW tensors: (W, H, Cin, N) / filter (w, h, Cin, Cout)
x = rand(rng, 32, 32, 3, 4)
w = rand(rng, 3, 3, 3, 8)
x_big = rand(rng, 64, 64, 8, 2)
w_big = rand(rng, 5, 5, 8, 16)

plan = plan_conv(x, w; pad = 1)
plan_big = plan_conv(x_big, w_big; pad = 2)
y = similar(x, 32, 32, 8, 4)
dx = similar(x)
dw = similar(w)

# =============================================================================
# Convolution
# =============================================================================

SUITE["conv"] = BenchmarkGroup()

SUITE["conv"]["plan"] = @benchmarkable plan_conv($x, $w; pad = 1)
SUITE["conv"]["forward"] = @benchmarkable conv($x, $w, $plan)
SUITE["conv"]["forward!"] = @benchmarkable conv!($y, $x, $w, $plan)
SUITE["conv"]["forward_big"] = @benchmarkable conv($x_big, $w_big, $plan_big)

# =============================================================================
# Gradients
# =============================================================================

SUITE["grad"] = BenchmarkGroup()

SUITE["grad"]["data"] = @benchmarkable ∇conv_data!($dx, $y, $w, $plan)
SUITE["grad"]["filter"] = @benchmarkable ∇conv_filter!($dw, $x, $y, $plan)
SUITE["grad"]["data_alloc"] = @benchmarkable ∇conv_data($y, $w, $plan)
SUITE["grad"]["filter_alloc"] = @benchmarkable ∇conv_filter($x, $y, $plan)
