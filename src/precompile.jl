using PrecompileTools: @setup_workload, @compile_workload

@setup_workload begin
    x32 = randn(Float32, 8, 8, 2, 1)
    w32 = randn(Float32, 3, 3, 2, 3)
    x64 = randn(Float64, 8, 8, 2, 1)
    w64 = randn(Float64, 3, 3, 2, 3)
    @compile_workload begin
        for (x, w) in ((x32, w32), (x64, w64))
            p = plan_conv(x, w; pad = 1, nthreads = 1)
            y = conv(x, w, p)
            ∇conv_data(y, w, p)
            ∇conv_filter(x, y, p)
        end
    end
end
