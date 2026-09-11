module CacheAwareConv

using SciMLPublic: @public
using SIMD: Vec, vload, vstore
using Static: known
using CPUSummary: cache_size, cache_linesize
using HostCPUFeatures: pick_vector_width, register_count

"""
    ConvKernel

Microkernel selection for [`ConvPlan`](@ref): `KernelAuto` (SIMD when the
compute type supports it, scalar otherwise), `KernelSIMD`, `KernelScalar`,
or `KernelLV` (the LoopVectorization.jl extension; requires
`using LoopVectorization`).
"""
@enum ConvKernel::UInt8 KernelAuto KernelSIMD KernelScalar KernelLV

"""
    ConvExecutor

Task scheduler for [`ConvPlan`](@ref): `ExecSpawn` (`Threads.@spawn`) or
`ExecPolyester` (the Polyester.jl extension; requires `using Polyester`).
"""
@enum ConvExecutor::UInt8 ExecSpawn ExecPolyester

include("geometry.jl")
include("cache_params.jl")
include("pack.jl")
include("kernel.jl")
include("plan.jl")
include("conv.jl")
include("kernel_grad.jl")
include("grad.jl")
include("api.jl")
include("reference.jl")
include("precompile.jl")

export ConvPlan, ConvGeometry, CacheInfo, plan_conv, conv, conv!, ∇conv_data, ∇conv_data!, ∇conv_filter, ∇conv_filter!
export reference_conv!, reference_∇conv_data!, reference_∇conv_filter!
@public output_size, input_size, kernel_size, channels_in, channels_out, batch_size, groups, flipped, stride, padding, dilation, spatial_dims
@public cache_info, compute_type, vector_width, register_tile, nplanes, allocated_bytes, geometry
@public conv_core!, conv_bias, bias_gradient!, PlanCache, get_plan!, SamePad, calc_padding
@public ConvKernel, ConvExecutor
export LuxConv, FluxConv

end
