[//]: # (Project: Accera)
[//]: # (Version: v1.2.3)

## Hello MatMul GPU

This tutorial will teach you how to implement a simple Matrix Multiplication (MatMul) function for execution on a GPU. We will use the Accera's Domain Specific Language (DSL) to produce a [HAT](https://github.com/microsoft/hat) package containing the MatMul function that can be called from the host to launch the MatMul function on the GPU.

### Prerequisites

* You should have Accera installed. If not, you can find the instructions in [here](../Install/README.md).
* Be familiar with writing Python and C++ code.
* Be familiar with basic GPU programming and concepts.
* You have completed the [Hello_MatMul](Hello_MatMul.md) tutorial.
* You have installed the [Vulkan SDK and runtime](https://vulkan.lunarg.com/sdk/home).

### Review: the naive MatMul algorithm

As in the [Hello_MatMul](Hello_MatMul.md) tutorial, we'll consider the example of multiplying matrices A and B and adding the result into matrix C. In NumPy syntax, this can be expressed as:

```
C += A @ B
```

A naive algorithm for matrix multiplication typically contains 3 nested for-loops. Expressed in Python, this will look like this:

```
for i in range(M):
    for j in range(N):
        for k in range(K):
            C[i, j] += A[i, k] * B[k, j]
```

#### Accera Python DSL

We will now walk through a basic Matrix Multiplication (MatMul) using Accera. Additionally, we will direct Accera to execute this MatMul function on the default GPU.

Create an empty file called `hello_matmul_gpu_generator.py`. Import dependent modules:

```python
import accera as acc
```

Define some matrix sizes. A will be M by K, B will be K by N, and C will be M by N.

```python
# Define our matrix sizes
M = 1024
N = 512
K = 256
```

Declare our arrays `A`, `B` and `C`. These are our input and input/output matrices and hold 32-bit floating point elements.

```python
A = acc.Array(role=acc.Array.Role.INPUT, element_type=acc.ScalarType.float32, shape=(M, K))
B = acc.Array(role=acc.Array.Role.INPUT, element_type=acc.ScalarType.float32, shape=(K, N))
C = acc.Array(role=acc.Array.Role.INPUT_OUTPUT, element_type=acc.ScalarType.float32, shape=(M, N))
```

Use the `Nest` class to define our 3-layered nested for loop and get the indices:
```python
# Define the loop nest
nest = acc.Nest(shape=(M, N, K))

# Get the loop nest indices
i, j, k = nest.get_indices()
```

Next we define the logic of each iteration of the loop nest:
```python
# Define the loop nest logic
@nest.iteration_logic
def _():
    C[i, j] += A[i, k] * B[k, j]
```

We have finished defining the logic of MatMul. Notice how up to this point, this is identical to what we did for the [CPU example](Hello_MatMul.md). Next, define the schedule which controls how the logic is executed. To do this, we first create the schedule from the nest:

```python
schedule = nest.create_schedule()
```

In order to execute this efficiently on our chosen hardware target, we will transform the iteration space and change the plan according to some predefined constants. The values of these constants can come either from hardware target characteristics and the shapes of the arrays, or can be found through auto-tuning. These will be explained in more detail in a subsequent tutorial. For now, define:
```python
block_x = 16
block_y = 16
```

Transform the iteration space to specify the thread block behavior. See (GPU blocks)[TODO:markdown...] section to learning more about optimizing block sizes on GPU:
```python
ii = schedule.split(i, block_x)
jj = schedule.split(j, block_y)
```

Set the order to traverse the iteration space. Note that on the precise order of execution on GPU targets will be unknown due to the parallel nature of the hardware. Nevertheless, setting the order here is important, since the coarse grain parallelization (e.g. grid) should precede the more fine grained (e.g. warps/wavefronts):
```python
schedule.reorder(i, j, ii, jj, k)
```

Create a plan from the schedule. The plan allows us to control specific execution behavior on the hardware target, such grid launch dimensions and thread blocks sizes, which are essential for high performance:
```python
target = acc.Target(category=acc.Target.Category.GPU, runtime=acc.Target.Runtime.VULKAN)
plan = schedule.create_plan(target)
```

Bind dimensions of the schedule to execution units on the GPU. Use the outer dimensions _i_, _j_ to be the block indices _x_,_y_ in the grid, and the _ii_ and _jj_ dimensions to be the thread indices _x_,_y_ in the block:
```python
plan.bind({
    i: target.GridUnit.BLOCK_X,
    j: target.GridUnit.BLOCK_Y,
    ii: target.GridUnit.THREAD_X,
    jj: target.GridUnit.THREAD_Y
})
```

Use the plan to add a callable function named `hello_matmul_gpu` to a HAT package.

```python
# Create a package and add a function to the package based on the plan
package = acc.Package()
package.add(plan, args=(A, B, C), base_name="hello_matmul_gpu")
```

Finally, we build the HAT package:
```python
# Build a statically-linked HAT package to be consumed by the C++ runner
package.build(name="hello_matmul_gpu", format=acc.Package.Format.HAT_STATIC)
```

By now, you have all the code necessary to generate an Accera MatMul function that runs on the GPU. You can also find the complete Python script [here](hello_matmul_gpu/hello_matmul_gpu_generator.py).

#### Generate HAT package

Next, we run the generator script to produce a HAT package.

##### Windows/MacOS

```shell
python hello_matmul_gpu_generator.py
```

##### Ubuntu

```shell
python3 hello_matmul_gpu_generator.py
```

After this runs, you should see a header file `hello_matmul_gpu.hat` and some object files (such as `hello_matmul_gpu.obj` or `hello_matmul_gpu.o`). The build process also generates a supporting module, `AcceraGPUUtilities.hat` and its object file, for GPU initialization and uninitialization. In Accera, we call these files the "HAT package".

#### Runner code

We will now walk through how to call our MatMul implementation from the HAT package.

Create a file called `hello_matmul_gpu_runner.cpp` with the code below. You can also find it [here](hello_matmul_gpu/hello_matmul_gpu_runner.cpp).

```cpp
#include <stdio.h>
#include <algorithm>

// Include the HAT file that declares GPU initialization/uninitialization functions
#include "AcceraGPUUtilities.hat"

// Include the HAT file that declares our MatMul function
#include "hello_matmul_gpu.hat"

#define M 1024
#define N 512
#define K 256

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

    // Initialize the GPU
    AcceraGPUInitialize();

    printf("Calling MatMul M=%d, K=%d, N=%d\n", M, K, N);
    hello_matmul_gpu(A, B, C);

    printf("Result (first 10 elements): ");
    for (int i = 0; i < 10; ++i)
    {
        printf("%f ", C[i]);
    }
    printf("\n");

    // Uninitialize the GPU
    AcceraGPUDeInitialize();

    delete[] A;
    delete[] B;
    delete[] C;
    return 0;
}
```

The code above creates the `A`, `B`, and `C` matrices, and calls the function `hello_matmul_gpu` to perform MatMul.

Now that we have written the code, we will compile and link it with the HAT package to create an executable. Save the file to your working directory, in the same location as `hello_matmul_gpu_generator.py` and the generated `*.hat` and object files.

#### Build and run

Accera includes a shared library that wraps the Vulkan APIs (`acc-vulkan-runtime-wrappers.so`, `acc-vulkan-runtime-wrappers.dll`, or `acc-vulkan-runtime-wrappers.dylib`). We will need to provide the path to this shared library when building and running the executable.

Find the installed path to the "accera" package:

##### Windows/MacOS
```shell
pip show accera
```

##### Ubuntu
```shell
pip3 show accera
```


From the output above, find the `Location` entry, for example:
```
Location: /usr/local/lib/python3.8/dist-packages
```

Note down this path, we will be using it below.

##### Windows

We will need the 64-bit Visual C++ tools to link against the generated 64-bit .obj. From an __"x64 Native Tools Command Prompt"__:

Set the `ACCERA_PATH` environment variable to the full install path of the "accera" package (derived from `pip show accera` to locate `acc-vulkan-runtime-wrappers.dll`):

```shell
set ACCERA_PATH=<Location_path>\accera
```

Set the `PATH` environment variable to allow the runner to locate `acc-vulkan-runtime-wrappers.dll`:

```shell
set PATH=%PATH%;%ACCERA_PATH%
```

Now build and run:

```shell
cl.exe hello_matmul_gpu_runner.cpp *.lib %ACCERA_PATH%/*.lib
hello_matmul_gpu_runner.exe
```

##### MacOS

Set the `ACCERA_PATH` environment variable to the full install path of the "accera" package (derived from `pip show accera` to locate `acc-vulkan-runtime-wrappers.dylib`):

```shell
export ACCERA_PATH=<Location_path>/accera
```

Now build and run:

```shell
clang++ hello_matmul_gpu_runner.cpp *.a $ACCERA_PATH/*.dylib -o hello_matmul_gpu_runner
DYLD_LIBRARY_PATH=$ACCERA_PATH ./hello_matmul_gpu_runner
```

##### Ubuntu

Set the `ACCERA_PATH` environment variable to the full install path of the "accera" package (derived from `pip3 show accera` to locate `acc-vulkan-runtime-wrappers.so`):

```shell
export ACCERA_PATH=<Location_path>/accera
```

Now build and run:

```shell
g++ hello_matmul_gpu_runner.cpp *.a $ACCERA_PATH/*.so -o hello_matmul_gpu_runner
LD_LIBRARY_PATH=$ACCERA_PATH ./hello_matmul_gpu_runner
```

The output should look like:

```
Calling MatMul M=1024, K=256, N=512
Result (first 10 elements): 1536.419922 1536.419922 1536.419922 1536.419922 1536.419922 1536.419922 1536.419922 1536.419922 1536.419922 1536.419922
```

You can now experiment with the generated MatMul function with your own inputs.
