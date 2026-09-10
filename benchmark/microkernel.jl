# Peak throughput of the register-tiled microkernel alone.
using CacheAwareConv, BenchmarkTools, Printf, SIMD
using CacheAwareConv: conv_microkernel!, vector_width, register_tile, nplanes, compute_type

function kernel_peak(::Type{T}) where {T}
    Tc = compute_type(T)
    NP = nplanes(T)
    V = vector_width(Tc)
    MR, NR = register_tile(Tc, NP)
    K = 9
    Kc = 64
    Lp = MR * V + K
    Xp = rand(Tc, NP * Kc * Lp)
    Wp = rand(Tc, NP * NR * K * Kc)
    Y = zeros(Tc, NP * MR * V * NR)
    taps = collect(0:(K - 1))
    mask = CacheAwareConv.lane_mask(Val(V), V)
    f() = GC.@preserve Xp Wp Y taps conv_microkernel!(
        Vec{V, Tc}, Val(MR), Val(NR), Val(NP), Val(false), Val(false),
        pointer(Y), MR * V, MR * V * NR, pointer(Xp), Lp, Kc * Lp, pointer(Wp), pointer(taps), K, Kc, mask
    )
    f()
    t = @belapsed $f()
    macs = MR * V * NR * K * Kc
    @printf("%-10s V=%2d MR=%d NR=%d NP=%d: %6.1f G real-FMA/s (%6.1f real GFLOPS equivalent)\n", T, V, MR, NR, NP, NP * macs / t / 1.0e9, 2NP * macs / t / 1.0e9)
    return nothing
end

foreach(kernel_peak, (Float32, Float64, Float16, ComplexF32, ComplexF64))
