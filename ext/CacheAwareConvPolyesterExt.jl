module CacheAwareConvPolyesterExt

using CacheAwareConv
using Polyester: @batch

# Same contiguous-chunk contract as `run_tasks`: iteration `t` owns scratch
# buffer `t`, so batching over the task index keeps buffer exclusivity and the
# deterministic reduction order.
function CacheAwareConv.run_tasks_polyester(f::F, nitems::Int, nt::Int) where {F}
    chunk = cld(nitems, nt)
    @batch for t in 1:nt
        lo = (t - 1) * chunk + 1
        hi = min(nitems, t * chunk)
        lo <= hi && f(t, lo:hi)
    end
    return nothing
end

end
