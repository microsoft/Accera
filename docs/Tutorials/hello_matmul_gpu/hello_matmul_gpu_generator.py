#!/usr/bin/env python3
# Accera Hello MatMul GPU sample: generator
import accera as acc

def create_matmul_schedule(block_x: int = 32, block_y: int = 32):
    # Define our matrix sizes
    M = 2048
    N = 1024
    K = 2048

    # Define the arguments we want to take for the MatMul function
    A = acc.Array(role=acc.Role.INPUT, element_type=acc.ScalarType.float32, shape=(M, K))
    B = acc.Array(role=acc.Role.INPUT, element_type=acc.ScalarType.float32, shape=(K, N))
    C = acc.Array(role=acc.Role.INPUT_OUTPUT, element_type=acc.ScalarType.float32, shape=(M, N))

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

    # Transform the schedule
    schedule.split(i, block_x)
    schedule.split(j, block_y)

    return schedule, A, B, C

# Create the matmul schedule
schedule, A, B, C = create_matmul_schedule()

# Set the dimension order
i, ii, j, jj, k = schedule.get_indices()
schedule.reorder(i, j, ii, jj, k)

# Create the GPU plan
target = acc.Target(acc.Target.Model.AMD_MI100)
plan = schedule.create_plan(target)

# Bind dimensions to a grid of execution units
plan.bind({
    i: target.GridUnit.BLOCK_Y,
    j: target.GridUnit.BLOCK_X,
    ii: target.GridUnit.THREAD_Y,
    jj: target.GridUnit.THREAD_X
})

package = acc.Package()
package.add(plan, args=(A, B, C), base_name="hello_matmul_gpu")
package.build("hello_matmul_gpu", format=acc.Package.Format.HAT_SOURCE)
