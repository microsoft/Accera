[//]: # (Project: Accera)
[//]: # (Version: v1.2.10)

## Optimized MatMul

Optimizing MatMul is highly dependent on the target platform. The code in the example below is optimized specifically for an Intel Xeon E5-2673 v3 CPU. However, it should work equally well on CPUs with similar hardware characteristics, such as an AMD Epyc 7551.

By the end of this tutorial, you will learn how to:
* Implement a performant Matrix Multiplication (MatMul) function targetting AVX2 FMA3 CPUs like Intel Haswell or the AMD Epyc families.
* Produce a [HAT](https://github.com/microsoft/hat) package containing the optimized MatMul function.
* Call the function from C or C++ code.

### Prerequisites

* You should have Accera installed. If not, you can find the instructions in [here](../Install/README.md).
* You are familiar with writing Python and C++.
* You know about [SIMD](https://en.wikipedia.org/wiki/SIMD) instructions and registers.
* You have completed the [Hello_MatMul](Hello_MatMul.md) tutorial.

### Review: the naive MatMul algorithm

As in the [Hello_MatMul](Hello_MatMul.md) tutorial, we'll consider the example of multiplying matrices A and B and adding the result into matrix C. In NumPy syntax, this can be expressed as:

```
C += A @ B
```

A naive algorithm for matrix multiplication typically contains 3 nested for-loops. Expressed in Python, this will look like:

```
for i in range(M):
    for j in range(N):
        for k in range(K):
            C[i, j] += A[i, k] * B[k, j]
```

#### Accera Python DSL

We will walk through how to specify an optimized Matrix Multiplication (MatMul) using Accera. This tutorial assumes the following:

* Specific matrix sizes. Inputs A and B are 784 x 128 and 128 x 512 matrices, respectively. The output C is a 784 x 512 matrix. These can represent a mid-level layer in a Resnet-50 model. The A matrix contains the activation values from the previous layer, and the B matrix contains the weights of the neural network layer.
* Row-major layout of the matrix elements.
* The target hardware is capable of AVX2 FMA3 instructions, such as the Intel Xeon E5-2673 v3 or the AMD Epyc 7551.

Create an empty file called `optimized_matmul_generator.py`. Import dependent modules:
```python
import accera as acc
```

Define some matrix sizes, where A's shape is M by K, B's is K by N, and C's, M by N.
```python
# Define our matrix sizes
M = 784
N = 512
K = 128
```

Declare arrays `A`, `B`, and `C`. These are our input and input/output matrices and hold 32-bit floating-point elements.
```python
A = acc.Array(role=acc.Array.Role.INPUT, element_type=acc.ScalarType.float32, shape=(M, K))
B = acc.Array(role=acc.Array.Role.INPUT, element_type=acc.ScalarType.float32, shape=(K, N))
C = acc.Array(role=acc.Array.Role.INPUT_OUTPUT, element_type=acc.ScalarType.float32, shape=(M, N))
```

Use the `Nest` class to define a 3-layered nested for-loop and get the indices:
```python
# Define the loop nest
nest = acc.Nest(shape=(M, N, K))

# Get the loop nest indices
i, j, k = nest.get_indices()
```

Next, we define the logic for every iteration of the loop nest:
```python
# Define the loop nest logic
@nest.iteration_logic
def _():
    C[i, j] += A[i, k] * B[k, j]
```

We have finished defining the logic of MatMul. Let's now define the schedule which controls execution logic. To do this, we first create the schedule from the nest:
```python
schedule = nest.create_schedule()
```

We will transform the iteration space and change the plan according to some predefined constants to execute this efficiently on our chosen hardware target. Values of these constants come either from hardware target characteristics and the shapes of the arrays or can be found through auto-tuning. These will be explained in detail in a subsequent tutorial. For now, define:
```python
tile_size_i = 6
tile_size_j = 256
tile_size_k = 128
inner_dim_unroll = 4
num_rows_in_kernel = 6
```

We create a CPU target that defines constants for the SIMD vector sizes and the number of vector execution units to use the hardware characteristics.
```python
target = acc.Target(category=acc.Target.Category.CPU)
```

Transform the iteration space to specify the tiling behavior:
```python
ii = schedule.split(i, tile_size_i)
jj = schedule.split(j, tile_size_j)
kk = schedule.split(k, tile_size_k)
```

Next, let's split the iteration space to match the kernel characteristics:
```python
kkk = schedule.split(kk, inner_dim_unroll)
iii = schedule.split(ii, num_rows_in_kernel)
jjj = schedule.split(jj, (target.vector_bytes // 4) * 2) # There are 2 vfma execution units, each holding (target.vector_bytes // 4) 32-bit float elements
jjjj = schedule.split(jjj, target.vector_bytes // 4) # Each SIMD register holds (target.vector_bytes // 4) 32-bit float elements
```

Accera will handle the encountered boundary conditions for each of these splits and do appropriate optimizations such as loop un-switching to ensure that efficient code gets generated in those cases.

Set the order to traverse the iteration space. We start with the outer indices that control the tiling, then move to the innermost indices that are used in the kernel:
```python
schedule.reorder(j, k, i, jj, kk, ii, kkk, iii, jjj, jjjj)
```

Create a plan from the schedule and the current target. The plan allows us to control specific execution behavior on the hardware target, such as vectorization and caching, which are essential for high performance:
```python
plan = schedule.create_plan(target)
```

Add caching. We use an input cache for array `B` that exceeds our threshold. The `B` cache will be packed according to the access pattern specified by the schedule. We use an input/output cache for array `C`. See [caching](../Manual/06%20Plans%20-%20Caching.md) for more information:
```python
# Cache the B array by prefetching and packing the memory footprint along slices of the jj dimension.
plan.cache(B, jj)
# Cache the C array along slices of jj dimension. Since the C array is the output, its footprint is
# the size of the kernel. If the kernel is small enough, Accera will use registers for this
# accumulation before writing these values back to C.
plan.cache(C, jj)
```

Kernelize the inner dimensions, which applies unroll and vectorize transformations allowing use of SIMD registers:
```python
plan.kernelize(unroll_indices=[jjj, iii, kkk], vectorize_indices=jjjj)
```

Use the plan to add a function named `optimized_matmul_py` to a HAT package.
```python
# Create a package and add a function to the package based on the plan
package = acc.Package()
package.add(plan, args=(A, B, C), base_name="optimized_matmul_py")
```

Finally, we build the HAT package:
```python
# Build a statically-linked HAT package to be consumed by the C++ runner
package.build(name="optimized_matmul", format=acc.Package.Format.HAT_STATIC)
```

By now, you should have all the code necessary to generate an optimized Accera MatMul function. You can find the complete Python script [here](optimized_matmul/optimized_matmul_generator.py).

#### Generate HAT package

Next, we run the generator script to produce a HAT package.

##### Windows/MacOS

```shell
python optimized_matmul_generator.py
```

##### Ubuntu

```shell
python3 optimized_matmul_generator.py
```

The generator script produces a HAT file (`optimized_matmul.hat`). Examine this file, and you will see that it contains the exported function with the following meta-data:

```toml
[functions.optimized_matmul_py_4a6286d9]
name = 'optimized_matmul_py_4a6286d9'
description = ''
calling_convention = "cdecl"
arguments = [
    {name = '', description = '', logical_type = "affine_array", declared_type = 'float*', element_type = 'float', usage = "input_output", shape = [ 784, 128 ], affine_map = [ 128, 1 ], affine_offset = 0},
    {name = '', description = '', logical_type = "affine_array", declared_type = 'float*', element_type = 'float', usage = "input_output", shape = [ 128, 512 ], affine_map = [ 512, 1 ], affine_offset = 0},
    {name = '', description = '', logical_type = "affine_array", declared_type = 'float*', element_type = 'float', usage = "input_output", shape = [ 784, 512 ], affine_map = [ 512, 1 ], affine_offset = 0}
]
return = {name = '', description = '', logical_type = "void", declared_type = 'void', element_type = 'void', usage = "output"}
```

The C declaration from the header is:
```cpp
void optimized_matmul_py_4a6286d9(float*, float*, float*);
```

Accera automatically appends a unique identifier to the function implementation, such as `optimized_matmul_py_4a6286d9` to support auto-tuning. This name is re-generated every time the HAT package is rebuilt. To make it easier for client code to use the function, Accera also provides a fixed-name alias, `optimized_matmul_py`, for the same function.

To see how Accera generates code for the iteration space transformations and the plan, you can change the `format=HAT` to `format=MLIR`, which will output MLIR for each lowering phase. Stepping through the progression of lowerings, you can see how Accera moves from simple representation of the [Accera DSL](optimized_matmul/mlir/1_Canonicalizer.mlir), to the final [optimized assembly](optimized_matmul/mlir/optimized_matmul_llvm.mlir).

Compare this to previous tutorial, whose naive DSL is given [here](hello_matmul/mlir/1_Canonicalizer.mlir), and final assembly can be viewed [here](hello_matmul/mlir/hello_matmul_llvm.mlir).

#### Runner code

Let's see how to call our MatMul implementation from the HAT package.

Create a file called `optimized_matmul_runner.cpp` with the code below. You can also find it [here](optimized_matmul/optimized_matmul_runner.cpp).

```cpp
#include <stdio.h>
#include <algorithm>

// Include the HAT file that declares our MatMul function
#include "optimized_matmul.hat"

#define M 784
#define N 512
#define K 128

int main(int argc, const char** argv)
{
    // Prepare our matrices (using the heap for large matrices)
    float* A = new float[M*K];
    float* B = new float[K*N];
    float* C = new float[M*N];

    // Fill with data
    std::fill_n(A, M*K, 2.0f);
    std::fill_n(B, K*N, 3.0f);
    std::fill_n(C, M*N, 0.42f);

    printf("Calling MatMul M=%d, K=%d, N=%d\n", M, K, N);
    optimized_matmul_py(A, B, C);

    printf("Result (first 10 elements): ");
    for (int i = 0; i < 10; ++i)
    {
        printf("%f ", C[i]);
    }
    printf("\n");

    delete[] A;
    delete[] B;
    delete[] C;
    return 0;
}
```

The above code creates the matrices `A`, `B`, and `C` and calls the function `optimized_matmul_py` to perform MatMul.

Now that we have the code, let's compile and link it with the HAT package to create an executable. Save the file to your working directory, in the same location as `optimized_matmul_generator.py` and the generated `*.hat` and object files.

#### Build and run

##### Windows

We will need the 64-bit Visual C++ tools to link against the generated 64-bit .obj. From an __"x64 Native Tools Command Prompt"__:

```
cl.exe optimized_matmul_runner.cpp *.lib
optimized_matmul_runner.exe
```

##### MacOS

```
clang++ optimized_matmul_runner.cpp *.a -o optimized_matmul_runner
./optimized_matmul_runner
```

##### Ubuntu

```
g++ optimized_matmul_runner.cpp *.a -o optimized_matmul_runner
./optimized_matmul_runner
```

The output should look like:

```
Calling MatMul M=784, K=128, N=512
Result (first 10 elements): 768.419983 768.419983 768.419983 768.419983 768.419983 768.419983 768.419983 768.419983 768.419983 768.419983
```

You can now experiment with the generated MatMul function with your own inputs.
