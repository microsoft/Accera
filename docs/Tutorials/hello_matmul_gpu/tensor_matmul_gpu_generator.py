#!/usr/bin/env python3
# Accera Tensor MatMul GPU sample: generator
import accera as acc
import hello_matmul_gpu_generator
import matmul_utils as utils

def create_matmul_plan_from_schedule(schedule: acc.Schedule, tensor_splits: tuple[int]):
    i, ii, j, jj, k = schedule.get_indices()

    # Create additional splits for tensorization
    iii, jjj, kk = schedule.tile({
        ii: tensor_splits[0],
        jj: tensor_splits[1],
        k: tensor_splits[2]
    })

    block_indices = (i, j)
    warp_indices = (ii, jj)
    tensor_indices = (iii, jjj, kk)

    # Set the dimension order
    schedule.reorder(*block_indices, k, *warp_indices, *tensor_indices)

    # Create the GPU plan
    plan = schedule.create_plan(target)

    # Bind dimensions to a grid of execution units
    plan.bind({
        i: target.GridUnit.BLOCK_Y,
        j: target.GridUnit.BLOCK_X,
        ii: target.GridUnit.WARP_Y,
        jj: target.GridUnit.WARP_X
    })

    return plan, tensor_indices

def create_basic_tensor_matmul_plan(target: acc.Target, mma_shape: acc.MMAShape, block_x: int = 32, block_y: int = 32):
    schedule, A, B, C = hello_matmul_gpu_generator.create_matmul_schedule(block_x, block_y)
    tensor_splits = target.tensor_core_info.compute_tensor_splits(mma_shape)
    plan, tensor_indices = create_matmul_plan_from_schedule(schedule, tensor_splits)
    return plan, A, B, C, tensor_indices

target = acc.Target(acc.Target.Model.AMD_MI100)
mma_shape = acc.MMAShape.M16xN16xK4_B1
plan, A, B, C, tensor_indices = create_basic_tensor_matmul_plan(target, mma_shape)

# Tensorize the plan
plan.tensorize(indices=tensor_indices, mma_shape=mma_shape)

utils.add_function_build_pkg(plan, A, B, C, "tensor_matmul_gpu")
