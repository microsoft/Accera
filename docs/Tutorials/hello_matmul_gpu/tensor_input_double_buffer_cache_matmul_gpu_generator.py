#!/usr/bin/env python3
# Accera Tensor Double Buffer Input Caching MatMul GPU sample: generator
import accera as acc
import tensor_input_cache_matmul_gpu_generator
import matmul_utils as utils

def create_tensor_input_caching_matmul_plan(target: acc.Target):
    # Create the tensorized plan
    plan, A, B, C, in_cache_idx, out_cache_idx = tensor_input_cache_matmul_gpu_generator.tensorize_cache_matul_plan(target)

    # Add input caching
    plan.cache(A, index=in_cache_idx, location=target.MemorySpace.SHARED, double_buffer=True, double_buffer_location=target.MemorySpace.PRIVATE)
    plan.cache(B, index=in_cache_idx, location=target.MemorySpace.SHARED, double_buffer=True, double_buffer_location=target.MemorySpace.PRIVATE)

    return plan, A, B, C, out_cache_idx

target = acc.Target(acc.Target.Model.AMD_MI100)
plan, A, B, C, _ = create_tensor_input_caching_matmul_plan(target)
utils.add_function_build_pkg(plan, A, B, C, "tensor_input_double_buffer_cache_matmul_gpu")
