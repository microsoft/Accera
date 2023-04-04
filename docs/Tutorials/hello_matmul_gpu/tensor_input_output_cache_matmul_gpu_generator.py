#!/usr/bin/env python3
# Accera Input Output Caching MatMul GPU sample: generator
import accera as acc
import tensor_input_double_buffer_cache_matmul_gpu_generator
import matmul_utils as utils

target = acc.Target(acc.Target.Model.AMD_MI100)
plan, A, B, C, out_cache_idx = tensor_input_double_buffer_cache_matmul_gpu_generator.create_tensor_input_caching_matmul_plan(target)

# Add output caching
plan.cache(C, index=out_cache_idx, location=target.MemorySpace.MMA_FRAGMENT)

utils.add_function_build_pkg(plan, A, B, C, "tensor_input_output_cache_matmul_gpu")
