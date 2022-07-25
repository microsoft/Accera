#!/usr/bin/env python3
# Accera Optimized MatMul sample: generator

import accera as acc

# Define our matrix sizes. These represent an arbitraily chosen layer in a
# Resnet-50 model.
M = 784
N = 512
K = 128

A = acc.Array(role=acc.Array.Role.INPUT, element_type=acc.ScalarType.float32, shape=(M, K))
B = acc.Array(role=acc.Array.Role.INPUT, element_type=acc.ScalarType.float32, shape=(K, N))
C = acc.Array(role=acc.Array.Role.INPUT_OUTPUT, element_type=acc.ScalarType.float32, shape=(M, N))

# Define the loop nest
nest = acc.Nest(shape=(M, N, K))

# Get the loop nest indices
i, j, k = nest.get_indices()

# Define the loop nest logic
@nest.iteration_logic
def _():
    C[i, j] += A[i, k] * B[k, j]

schedule = nest.create_schedule()

# Define constants used in the schedule and plan. The values are
# either hardware target characteristics or can be found through auto-tuning.
tile_size_i = 6
tile_size_j = 128
tile_size_k = 128
inner_dim_unroll = 4
num_rows_in_kernel = 6

# Create a CPU target which will define the hardware target characteristics
target = acc.Target(category=acc.Target.Category.CPU)

# Transform the iteration space
ii = schedule.split(i, tile_size_i)
jj = schedule.split(j, tile_size_j)
kk = schedule.split(k, tile_size_k)

kkk = schedule.split(kk, inner_dim_unroll)
iii = schedule.split(ii, num_rows_in_kernel)
jjj = schedule.split(jj, (target.vector_bytes // 4) * 2) # There are 2 vfma execution units, each holding (target.vector_bytes // 4) 32-bit float elements
jjjj = schedule.split(jjj, target.vector_bytes // 4) # Each SIMD register holds (target.vector_bytes // 4) 32-bit float elements

schedule.reorder(j, k, i, jj, ii, kk, kkk, iii, jjj, jjjj)

plan = schedule.create_plan(target)

# Add caching
# Cache the B array by prefetching and packing the memory footprint along slices of the jj dimension.
plan.cache(B, jj)
# Cache the C array along slices of jj dimension. Since the C array is the output, its footprint is
# the size of the kernel. If the kernel is small enough, Accera will use registers for this
# accumulation before writing these values back to C.
plan.cache(C, ii)

# Kernelize the inner dimensions, which applies unroll and vectorize transformations
plan.kernelize(unroll_indices=[jjj, iii, kkk], vectorize_indices=jjjj)

# Create a package and add a function to the package based on the plan
package = acc.Package()
package.add(plan, args=(A, B, C), base_name="optimized_matmul_py")

# Build a statically-linked HAT package to be consumed by the C++ runner
package.build(name="optimized_matmul", format=acc.Package.Format.HAT_STATIC)
