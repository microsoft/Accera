#!/usr/bin/env python3
# Accera Hello MatMul GPU sample: generator
import accera as acc

# Define our matrix sizes
M = 1024
N = 512
K = 256

# Define the arguments we want to take for the MatMul function
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

# Create the schedule from the nest
schedule = nest.create_schedule()

# Define constants
block_x = 16
block_y = 16
block_z = 1

# Transform the schedule
ii = schedule.split(i, block_x)
jj = schedule.split(j, block_y)

# Set the dimension order
schedule.reorder(i, j, ii, jj, k)

# Create the GPU plan
target = acc.Target(category=acc.Target.Category.GPU, runtime=acc.Target.Runtime.VULKAN)
plan = schedule.create_plan(target)

# Bind dimensions to a grid of execution units
plan.bind({
    i: target.GridUnit.BLOCK_X,
    j: target.GridUnit.BLOCK_Y,
    ii: target.GridUnit.THREAD_X,
    jj: target.GridUnit.THREAD_Y
})

# Create a package and add a function to the package based on the plan
package = acc.Package()
package.add(plan, args=(A, B, C), base_name="hello_matmul_gpu")

# Build a statically-linked HAT package to be consumed by the C++ runner
# Change format=acc.Package.Format.HAT_STATIC to format=acc.Package.Format.MLIR_STATIC to also generate MLIR to _tmp/hello_matmul_gpu
package.build("hello_matmul_gpu", format=acc.Package.Format.HAT_STATIC)
