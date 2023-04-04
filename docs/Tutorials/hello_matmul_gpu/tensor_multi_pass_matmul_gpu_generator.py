#!/usr/bin/env python3
# Accera Multi-pass Tensor MatMul GPU (with pass fusion)
import accera as acc
import hello_matmul_gpu_generator
import tensor_matmul_gpu_generator

def create_multipass_tensor_matmul_plan(target: acc.Target, mma_shape: acc.MMAShape, num_passes: int):
    schedule, A, B, C = hello_matmul_gpu_generator.create_matmul_schedule()
    tensor_splits = target.tensor_core_info.compute_tensor_splits(mma_shape, num_passes)
    plan, tensor_indices = tensor_matmul_gpu_generator.create_matmul_plan_from_schedule(schedule, tensor_splits)
    return plan, A, B, C, tensor_indices

target = acc.Target(acc.Target.Model.AMD_MI100)
mma_shape = acc.MMAShape.M16xN16xK4_B1
num_passes = 8
package = acc.Package()

# total passes = 8, fused passes = ALL (default: fuse all passes)
plan, A, B, C, tensor_indices = create_multipass_tensor_matmul_plan(target, mma_shape, num_passes)
plan.tensorize(indices=tensor_indices, mma_shape=mma_shape, num_total_passes=num_passes)
package.add(plan, args=(A, B, C), base_name="tensor_multipass_full_fusion_matmul_gpu")

# total passes = 8, fused passes = 2
plan, A, B, C, tensor_indices = create_multipass_tensor_matmul_plan(target, mma_shape, num_passes)
plan.tensorize(indices=tensor_indices, mma_shape=mma_shape, num_total_passes=num_passes, num_fused_passes=2)
package.add(plan, args=(A, B, C), base_name="tensor_multipass_partial_fusion_matmul_gpu")

package.build("tensor_multipass_matmul_gpu", format=acc.Package.Format.HAT_SOURCE)
