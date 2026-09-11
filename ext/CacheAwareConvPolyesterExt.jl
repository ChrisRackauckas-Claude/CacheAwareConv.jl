module CacheAwareConvPolyesterExt

using CacheAwareConv
using Polyester: @batch

# Parallelise over the work items themselves and let Polyester do the
# partitioning: the scratch-buffer index is the executing thread's id, so
# plans allocate `Threads.nthreads()` buffers under this executor (see
# `nscratch`). `minbatch` bounds the number of participating threads by `nt`.
function CacheAwareConv.run_tasks_polyester(f::F, nitems::Int, nt::Int) where {F}
    @batch minbatch = cld(nitems, nt) for item in 1:nitems
        f(Threads.threadid(), item:item)
    end
    return nothing
end

end
