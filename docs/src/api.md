# API

## Planning and running

```@docs
ConvPlan
plan_conv
conv
conv!
CacheAwareConv.conv_core!
CacheAwareConv.conv_bias
CacheAwareConv.output_size
CacheAwareConv.input_size
CacheAwareConv.kernel_size
CacheAwareConv.channels_in
CacheAwareConv.channels_out
CacheAwareConv.batch_size
CacheAwareConv.groups
CacheAwareConv.flipped
CacheAwareConv.stride
CacheAwareConv.padding
CacheAwareConv.dilation
CacheAwareConv.spatial_dims
CacheAwareConv.geometry
CacheAwareConv.allocated_bytes
CacheAwareConv.ConvKernel
CacheAwareConv.ConvExecutor
```

## Geometry

```@docs
ConvGeometry
CacheAwareConv.SamePad
CacheAwareConv.calc_padding
```

## Gradients

```@docs
∇conv_data!
∇conv_filter!
∇conv_data
∇conv_filter
CacheAwareConv.bias_gradient!
```

## Hardware parameters

```@docs
CacheInfo
CacheAwareConv.cache_info
CacheAwareConv.compute_type
CacheAwareConv.vector_width
CacheAwareConv.register_tile
CacheAwareConv.nplanes
```

## Layer support

```@docs
CacheAwareConv.PlanCache
CacheAwareConv.get_plan!
```

## Reference implementations

```@docs
reference_conv!
reference_∇conv_data!
reference_∇conv_filter!
```
