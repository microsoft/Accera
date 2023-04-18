#!/usr/bin/env python3
# Accera Multi-Pass Tensor MatMul GPU sample: generator
import accera as acc
import tensor_matmul_gpu_generator

mma_shape = acc.MMAShape.M64xN64xK1_B4
target = acc.Target(acc.Target.Model.AMD_MI100)
package = acc.Package()

# Pass order
plan, A, B, C, tensor_indices = tensor_matmul_gpu_generator.create_basic_tensor_matmul_plan(target, mma_shape, 64, 64)
plan.tensorize(indices=tensor_indices, mma_shape=mma_shape, scheduling_policy=acc.MMASchedulingPolicy.PASS_ORDER)
package.add(plan, args=(A, B, C), base_name="tensor_pass_order_matmul_gpu")

# Block order
plan, A, B, C, tensor_indices = tensor_matmul_gpu_generator.create_basic_tensor_matmul_plan(target, mma_shape, 64, 64)
plan.tensorize(indices=tensor_indices, mma_shape=mma_shape, scheduling_policy=acc.MMASchedulingPolicy.BLOCK_ORDER)
package.add(plan, args=(A, B, C), base_name="tensor_block_order_matmul_gpu")

package.build("tensor_sched_policy_matmul_gpu", format=acc.Package.Format.HAT_SOURCE)