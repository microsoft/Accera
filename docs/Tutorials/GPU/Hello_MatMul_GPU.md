[//]: # (Project: Accera)
[//]: # (Version: v1.2)

# Hello MatMul on GPU

In this tutorial, you will learn how to implement a simple Matrix Multiplication (MatMul) function for execution on a GPU. We will use the Accera's Domain Specific Language (DSL) to produce a [HAT](https://github.com/microsoft/hat) package containing the MatMul function that can be called from the host to launch the MatMul function on the GPU.

## Prerequisites

* You should have Accera installed. If not, you can find the instructions in [here](../../Install/README.md).
* Be familiar with writing Python and C++ code.
* Be familiar with basic GPU programming and concepts.
* You have completed the [Hello_MatMul](../Hello_MatMul.md) tutorial.

## Review: the naive MatMul algorithm

As in the [Hello_MatMul](../Hello_MatMul.md) tutorial, we'll consider the example of multiplying matrices A and B and adding the result into matrix C. In NumPy syntax, this can be expressed as:

```
C += A @ B
```

A naive algorithm for matrix multiplication typically contains 3 nested for-loops. Expressed in Python, this can look like:

```
for i in range(M):
    for j in range(N):
        for k in range(K):
            C[i, j] += A[i, k] * B[k, j]
```

### Accera Python DSL

We will now walk through a basic Matrix Multiplication (MatMul) using Accera. Additionally, we will direct Accera to execute this MatMul function on the default GPU.

Create an empty file called `hello_matmul_gpu_generator.py`. Import dependent modules:

```python
import accera as acc
```

Define some matrix sizes, where A's shape is M by K, B's is K by N, and C's, M by N.

```python
# Define our matrix sizes
M = 2048
N = 1024
K = 2048
```

Declare arrays `A`, `B`, and `C`. These are our input and input/output matrices and hold 32-bit floating-point elements.

```python
A = acc.Array(role=acc.Role.INPUT, element_type=acc.ScalarType.float32, shape=(M, K))
B = acc.Array(role=acc.Role.INPUT, element_type=acc.ScalarType.float32, shape=(K, N))
C = acc.Array(role=acc.Role.INPUT_OUTPUT, element_type=acc.ScalarType.float32, shape=(M, N))
```

Use the `Nest` class to define our 3-layered nested for-loop and get the indices:
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

We have finished defining the logic of MatMul. Notice how, up to this point, it is identical to what we did for the [CPU example](../Hello_MatMul.md). Let's now define the schedule to control the execution logic. To do this, we first create the schedule from the nest:

```python
schedule = nest.create_schedule()
```

We will transform the iteration space and change the plan according to some predefined constants to execute this efficiently on our chosen hardware target. The values of these constants can come either from hardware target characteristics and the shapes of the arrays or can be found through auto-tuning. These will be explained in detail in a subsequent tutorial. For now, define:

```python
block_x = 32
block_y = 32
```

Transform the iteration space to specify the thread block behavior. See (GPU blocks)[TODO:markdown...] section to learning more about optimizing block sizes on GPU:
```python
ii = schedule.split(i, block_x)
jj = schedule.split(j, block_y)
```

Set the order to traverse the iteration space. Note that the precise order of execution on GPU targets will be unknown due to the parallel nature of the hardware. Nevertheless, setting the order here is important since the coarse grain parallelization (e.g., grid) should precede the more fine-grained (e.g., warps/wavefronts):
```python
schedule.reorder(i, j, ii, jj, k)
```

Create a plan from the schedule. The plan allows us to control specific execution behavior on the hardware target (AMD MI100 in this example). The same schedule can be retargetted for a different platform like an NVIDIA GPU (acc.Target.Model.NVIDIA_RTX_A6000):
```python
target = acc.Target(acc.Target.Model.AMD_MI100)
plan = schedule.create_plan(target)
```

Bind dimensions of the schedule to execution units on the GPU. Use the outer dimensions _i_, _j_ to be the block indices _y_,_x_ in the grid, and the _ii_ and _jj_ dimensions to be the thread indices _y_,_x_ in the block:
```python
plan.bind({
    i: target.GridUnit.BLOCK_Y,
    j: target.GridUnit.BLOCK_X,
    ii: target.GridUnit.THREAD_Y,
    jj: target.GridUnit.THREAD_X
})
```

Use the plan to add a callable function named `hello_matmul_gpu` to a HAT package.

```python
# Create a package and add a function to the package based on the plan
package = acc.Package()
package.add(plan, args=(A, B, C), base_name="hello_matmul_gpu")
```

Finally, we build the HAT package, using the HAT_SOURCE format to produce code for the GPU:
```python
package.build(name="hello_matmul_gpu", format=acc.Package.Format.HAT_SOURCE)
```

By now, you have all the code necessary to generate an Accera MatMul function that runs on the GPU. You can find the complete Python script [here](../hello_matmul_gpu/hello_matmul_gpu_generator.py).

### Generate HAT package

Next, we run the generator script to produce a HAT package.

```shell
python hello_matmul_gpu_generator.py
```

After this script runs, you should see a header file `hello_matmul_gpu.hat` and a source file (such as `hello_matmul_gpu.cu`). In Accera, we call these files the "HAT package".

The `.cu` source file contains C++ functions required to launch the kernel on the GPU (the source code below is only shown as an example, the actual generated code might be different based on optimizations, GPU target, cosmetic changes etc. with future Accera releases). Note the HIP compiler intrinsics in the generated code below since we used the AMD GPU target, similar target specific code will be emitted if the plan is created for a different GPU target:

#### Host launcher

```c
#if !defined(__HIP_DEVICE_COMPILE__)
void hello_matmul_gpu_f77287579284bbac_impl_2389286605904206643(float *arg0, float *arg1, float *arg2) {
    hello_matmul_gpu_f77287579284bbac__gpu__<<<dim3(32, 64, 1), dim3(32, 32, 1), 0>>>(arg0, arg1, arg2);
    return;
}


#endif // !defined(__HIP_DEVICE_COMPILE__)
#if !defined(__HIP_DEVICE_COMPILE__)
extern "C" __host__ void hello_matmul_gpu_f77287579284bbac(float *arg0, float *arg1, float *arg2) {
    hello_matmul_gpu_f77287579284bbac_impl_2389286605904206643(arg0, arg1, arg2);
    return;
}
```

#### GPU kernel
```c
extern "C" __global__  __launch_bounds__(1024) void hello_matmul_gpu_f77287579284bbac__gpu__(float *arg0, float *arg1, float *arg2) {
    // Calculate threadid offsets and other locals
    extern __shared__ char sharedMemBaseAddr[];
    int32_t var0 = __builtin_amdgcn_workitem_id_x();
    int32_t var1 = __builtin_amdgcn_workitem_id_y();
    int32_t var2 = __builtin_amdgcn_workgroup_id_x();
    int32_t var3 = __builtin_amdgcn_workgroup_id_y();
    int32_t var4 = var3 * 32;
    int32_t var5 = var1 + var4;
    int32_t var6 = var2 * 32;
    int32_t var7 = var0 + var6;

    // Main K-loop
    for (int32_t idx8 = 0; idx8 < 2048; idx8 += 1) {
        // Matrix multiplication
        const auto arg0_offset0 = var5 * 2048 + idx8 * 1;
        float var9 = ((float*)arg0)[arg0_offset0];
        const auto arg1_offset1 = idx8 * 1024 + var7 * 1;
        float var10 = ((float*)arg1)[arg1_offset1];
        float var11 = var9 * var10;
        const auto arg2_offset2 = var5 * 1024 + var7 * 1;
        float var12 = ((float*)arg2)[arg2_offset2];
        float var13 = var12 + var11;

        // Store the result to global memory
        const auto arg2_offset3 = var5 * 1024 + var7 * 1;
        ((float*)arg2)[arg2_offset3] = var13;
    }
}
```

### Execution and Benchmarking GPU kernels using `hatlib` (recommended)
`hatlib` provides a convenient way to benchmark GPU kernels generated by Accera using Python. To benchmark the HAT package on an AMD MI100 system, install these dependencies:
* [ROCm (>= 5.1)](https://docs.amd.com/category/ROCm_v5.1)
* [hatlib (>= 0.0.38)](https://pypi.org/project/hatlib/0.0.38/)

Run the following command:
```shell
python3 -m hatlib.benchmark_hat_package <path to hello_matmul_gpu.hat> --cpp --min_time_in_sec 10 --time_in_ms
```

This will compile the generated GPU source code and execute it on the device with the provided benchmarking parameters. More details about the hatlib benchmarking tool can be found [here](https://github.com/microsoft/hat/tree/main/hatlib).

The above invocation of the hatlib tool will output the time in milliseconds (~13.7 ms) to execute the GPU kernel on an AMD MI100 system:
```shell
                       function_name        mean  median_of_means  mean_of_small_means  robust_mean  min_of_means
0  hello_matmul_gpu_f77287579284bbac 13.69571655      13.70083130          13.68229834  13.69727234   13.65689209
```

### Execution using standalone C++ runner (not recommended)
Since the Accera generated GPU kernel is source code, it can be compiled using the HIP compiler into a standalone C++ runner. Here's an example that calls the host launcher function from C++. An example of such a runner code is shown below:

```cpp
// hello_matmul_gpu_runner.cpp
#include <stdio.h>
#include <algorithm>

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

    float* dev_A;
    float* dev_B;
    float* dev_C;
    hipMalloc(&dev_A, M * K * sizeof(float));
    hipMalloc(&dev_B, K * N * sizeof(float));
    hipMalloc(&dev_C, M * N * sizeof(float));
    hipMemcpy(dev_A, A, M * K * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(dev_B, B, K * N * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(dev_C, C, M * N * sizeof(float), hipMemcpyHostToDevice);

    printf("Calling MatMul M=%d, K=%d, N=%d\n", M, K, N);
    hello_matmul_gpu_bbe110463fdb1f6b(A, B, C);

    hipDeviceSynchronize();
    hipMemcpy(C, dev_C, M * N * sizeof(float), hipMemcpyDeviceToHost);

    printf("Result (first 10 elements): ");
    for (int i = 0; i < 10; ++i)
    {
        printf("%f ", C[i]);
    }
    printf("\n");

    delete[] A;
    delete[] B;
    delete[] C;
    hipFree(dev_A);
    hipFree(dev_B);
    hipFree(dev_C);
    return 0;
}
```

The above code creates the `A`, `B`, and `C` matrices and calls the function `hello_matmul_gpu` to perform MatMul.

Now that we have the code, compile and link it with the HAT package to create an executable. The compilation/execution steps are left as an exercise for the reader. For more details, you can refer to the HIP compiler documentation [here](https://github.com/ROCm-Developer-Tools/HIP).

The output should look like this:

```
Calling MatMul M=1024, K=256, N=512
Result (first 10 elements): 1536.419922 1536.419922 1536.419922 1536.419922 1536.419922 1536.419922 1536.419922 1536.419922 1536.419922 1536.419922
```

You can now experiment with the generated MatMul function with your own inputs.
