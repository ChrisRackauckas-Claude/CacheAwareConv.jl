module CacheAwareConv

using SciMLPublic: @public

include("geometry.jl")
include("reference.jl")

export ConvGeometry, reference_conv!, reference_∇conv_data!, reference_∇conv_filter!
@public output_size, input_size, kernel_size, channels_in, channels_out, batch_size, groups, flipped, stride, padding, dilation, spatial_dims

end
