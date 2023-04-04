#!/usr/bin/env python3
# Accera Tensor Sequential Input Caching MatMul GPU sample: generator
import accera as acc
import hello_matmul_gpu_generator
import matmul_utils as utils

def tensorize_cache_matul_plan(target: acc.Target):
    mma_shape = acc.MMAShape.M16xN16xK4_B1

    # Create the matmul schedule
    schedule, A, B, C = hello_matmul_gpu_generator.create_matmul_schedule()
    i, ii, j, jj, k = schedule.get_indices()

    # Additional k-split required for caching
    kk = schedule.split(k, 256)

    # Create additional splits for tensorization
    tensor_splits = target.tensor_core_info.compute_tensor_splits(mma_shape)
    iii, jjj, kkk = schedule.tile({
        ii: tensor_splits[0],
        jj: tensor_splits[1],
        kk: tensor_splits[2]
    })

    block_indices = (i, j)
    warp_indices = (ii, jj)
    tensor_indices = (iii, jjj, kkk)

    # Set the dimension order
    schedule.reorder(*block_indices, k, kk, *warp_indices, *tensor_indices)

    # Create the GPU plan
    plan = schedule.create_plan(target)

    # Bind dimensions to a grid of execution units
    plan.bind({
        i: target.GridUnit.BLOCK_Y,
        j: target.GridUnit.BLOCK_X,
        ii: target.GridUnit.WARP_Y,
        jj: target.GridUnit.WARP_X
    })

    # Tensorize the plan
    plan.tensorize(indices=tensor_indices, mma_shape=mma_shape)
    return plan, A, B, C, kk, k

# Create the tensorized plan
target = acc.Target(acc.Target.Model.AMD_MI100)
plan, A, B, C, in_cache_idx, _ = tensorize_cache_matul_plan(target)

# Add input caching
plan.cache(A, index=in_cache_idx, location=target.MemorySpace.SHARED)
plan.cache(B, index=in_cache_idx, location=target.MemorySpace.SHARED)

utils.add_function_build_pkg(plan, A, B, C, "tensor_input_cache_matmul_gpu")
