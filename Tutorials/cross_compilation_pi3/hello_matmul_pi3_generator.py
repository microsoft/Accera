#!/usr/bin/env python3
# Cross compilation for pi3 sample: Accera Hello MatMul generator
import accera as acc

# Define our matrix sizes
M = 128
N = 256
K = 256

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

# Split the k loop into blocks of 4
kk = schedule.split(k, 4)

# Create a plan, specify the target to be a Raspberry Pi 3
pi3 = acc.Target(acc.Target.Model.RASPBERRY_PI_3B)
plan = schedule.create_plan(pi3)

# Then unroll kk
plan.unroll(kk)

# Create a package and add a function to the package based on the plan
package = acc.Package()
package.add(plan, args=(A, B, C), base_name="hello_matmul_pi3_py")

# Build the HAT package
package.build(name="hello_matmul_pi3", format=acc.Package.Format.HAT_STATIC, platform=acc.Package.Platform.RASPBIAN)
