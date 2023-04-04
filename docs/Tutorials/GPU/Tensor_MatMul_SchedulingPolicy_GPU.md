[//]: # (Project: Accera)
[//]: # (Version: v1.2)

# Tensor MatMul on GPU: Scheduling Policy experiments

In this tutorial, you will learn how to use different scheduling policies to control GPU register usage while using tensor cores for matrix multiplication.

## Prerequisites
* You have completed the [Tensor_MatMul_GPU](Tensor_MatMul_GPU.md) tutorial.

## Background
Some MMA shapes perform matrix multiplication over multiple invocations of the same MMA instruction with different arguments. Conventionally, MMA shape of MmxNnxKk_Bb performs matmul of A {__m__ x __k__} and B {__k__ x __n__} to produce the result of size {__m__ x __n__} containing __b__ chunks or blocks. For example, M64xN64xK1_B4 produces result matrix of size 64x64 in 4 blocks of size 16x64 each where the first 16 rows of the result form block 1, the next 16 rows form block 2 and so on.

For MMA shapes where __b__ is greater than 1, Accera allocates registers for output data based on the `scheduling_policy` parameter passed to the `plan.tensorize` call (conversely, when __b__ is 1, this parameter will not affect the emitted GPU code). Currently, Accera supports 2 different values of `scheduling_policy`, as mentioned in [MMASchedulingPolicy.md](../../Reference/enumerations/MMASchedulingPolicy.md), which expose different register usage vs. memory I/O tradeoffs as explained in detail below.

## Block order
In this mode blocks are computed sequentially, which means allocation for registers required for accumulator/output data is done for a single block. This mode can be enable by setting the `scheduling_policy` parameter as shown below:

```python
plan.tensorize(indices=tensor_indices, mma_shape=acc.MMAShape.M64xN64xK1_B4, scheduling_policy=acc.MMASchedulingPolicy.BLOCK_ORDER)
```

The generated source code looks something like this (note the accumulator fragment `mmaMatrix_24` being reused for each block currently being computed):

```c
extern "C" __global__  __launch_bounds__(64) void tensor_block_order_matmul_gpu_a0f5a6cd5453d086__gpu__(float *arg0, float *arg1, float *arg2) {
    // Calculate threadid offsets and other locals
    /*...*/

    // Main K-loop
    for (int32_t idx20 = 0; idx20 < 512; idx20 += 1) {
        // Declare matrix fragments for A, B and C
        /*...*/

        // Load matrix A data
        rocwmma::load_matrix_sync<2048>(var0, mmaMatrix_22, arg0 + ...);

        // Load matrix B data
        rocwmma::load_matrix_sync<1024>(var0, mmaMatrix_23, arg1 + ...);

        // Matrix multiplication yieling block 0
        rocwmma::load_matrix_sync<0, rocwmma::layout_t::mem_row_major, 1024>(var0, mmaMatrix_24, arg2 + ...);
        rocwmma::mma_sync<2, 0, 0>(mmaMatrix_24, mmaMatrix_22, mmaMatrix_23, mmaMatrix_24);
        rocwmma::store_matrix_sync<0, rocwmma::layout_t::mem_row_major, 1024>(var0, arg2 + ..., mmaMatrix_24);

        // Matrix multiplication yieling block 1
        rocwmma::load_matrix_sync<0, rocwmma::layout_t::mem_row_major, 1024>(var0, mmaMatrix_24, arg2 + ...);
        rocwmma::mma_sync<2, 1, 0>(mmaMatrix_24, mmaMatrix_22, mmaMatrix_23, mmaMatrix_24);
        rocwmma::store_matrix_sync<0, rocwmma::layout_t::mem_row_major, 1024>(var0, arg2 + ..., mmaMatrix_24);

        // Matrix multiplication yieling block 2
        rocwmma::load_matrix_sync<0, rocwmma::layout_t::mem_row_major, 1024>(var0, mmaMatrix_24, arg2 + ...);
        rocwmma::mma_sync<2, 2, 0>(mmaMatrix_24, mmaMatrix_22, mmaMatrix_23, mmaMatrix_24);
        rocwmma::store_matrix_sync<0, rocwmma::layout_t::mem_row_major, 1024>(var0, arg2 + ..., mmaMatrix_24);

        // Matrix multiplication yieling block 3
        rocwmma::load_matrix_sync<0, rocwmma::layout_t::mem_row_major, 1024>(var0, mmaMatrix_24, arg2 + ...);
        rocwmma::mma_sync<2, 3, 0>(mmaMatrix_24, mmaMatrix_22, mmaMatrix_23, mmaMatrix_24);
        rocwmma::store_matrix_sync<0, rocwmma::layout_t::mem_row_major, 1024>(var0, arg2 + ..., mmaMatrix_24);
    }
}
```

## Pass order
This operation is applied on each element of the tensor fragment data after matrix multiplication is done. Similar to `prologue_op`, we can also set the `epilogue_op` argument to achieve this. Here is an example of how alpha scaling of matmul result can be done with the following code:

```python
plan.tensorize(indices=tensor_indices, mma_shape=acc.MMAShape.M64xN64xK1_B4, scheduling_policy=acc.MMASchedulingPolicy.PASS_ORDER)
```

The generated source code shows how accumulator data for all the 4 blocks is loaded before the compute phase.:

```c
extern "C" __global__  __launch_bounds__(64) void tensor_pass_order_matmul_gpu_0d6383ac17fdfc9c__gpu__(float *arg0, float *arg1, float *arg2) {
    // Calculate threadid offsets and other locals
    /*...*/

    // Main K-loop
    for (int32_t idx20 = 0; idx20 < 512; idx20 += 1) {
        // Declare matrix fragments for A, B and C
        /*...*/

        // Load matric C data for block 0
        rocwmma::fragment<rocwmma::accumulator, 64, 64, 4, 4, 1, float> mmaMatrix_24;
        rocwmma::load_matrix_sync<0, rocwmma::layout_t::mem_row_major, 1024>(var0, mmaMatrix_24, arg2 + ...);

        // Load matric C data for block 1
        rocwmma::fragment<rocwmma::accumulator, 64, 64, 4, 4, 1, float> mmaMatrix_25;
        rocwmma::load_matrix_sync<0, rocwmma::layout_t::mem_row_major, 1024>(var0, mmaMatrix_25, arg2 + ...);

        // Load matric C data for block 2
        rocwmma::fragment<rocwmma::accumulator, 64, 64, 4, 4, 1, float> mmaMatrix_26;
        rocwmma::load_matrix_sync<0, rocwmma::layout_t::mem_row_major, 1024>(var0, mmaMatrix_26, arg2 + ...);

        // Load matric C data for block 3
        rocwmma::fragment<rocwmma::accumulator, 64, 64, 4, 4, 1, float> mmaMatrix_27;
        rocwmma::load_matrix_sync<0, rocwmma::layout_t::mem_row_major, 1024>(var0, mmaMatrix_27, arg2 + ...);
        
        // Load matric A data
        rocwmma::load_matrix_sync<2048>(var0, mmaMatrix_22, arg0 + ...);

        // Load matric B data
        rocwmma::load_matrix_sync<1024>(var0, mmaMatrix_23, arg1 + ...);

        // Compute matrix multiplication for blocks [0, 4)
        rocwmma::mma_sync<2, 0, 0>(mmaMatrix_24, mmaMatrix_22, mmaMatrix_23, mmaMatrix_24);
        rocwmma::mma_sync<2, 1, 0>(mmaMatrix_25, mmaMatrix_22, mmaMatrix_23, mmaMatrix_25);
        rocwmma::mma_sync<2, 2, 0>(mmaMatrix_26, mmaMatrix_22, mmaMatrix_23, mmaMatrix_26);
        rocwmma::mma_sync<2, 3, 0>(mmaMatrix_27, mmaMatrix_22, mmaMatrix_23, mmaMatrix_27);

        // Store C data for blocks [0, 4)
        rocwmma::store_matrix_sync<0, rocwmma::layout_t::mem_row_major, 1024>(var0, arg2 + ..., mmaMatrix_24);
        rocwmma::store_matrix_sync<0, rocwmma::layout_t::mem_row_major, 1024>(var0, arg2 + ..., mmaMatrix_25);
        rocwmma::store_matrix_sync<0, rocwmma::layout_t::mem_row_major, 1024>(var0, arg2 + ..., mmaMatrix_26);
        rocwmma::store_matrix_sync<0, rocwmma::layout_t::mem_row_major, 1024>(var0, arg2 + ..., mmaMatrix_27);
    }
}
```

## Benchmarking using `hatlib`
The complete python script with both scheduling policies can be found [here](../hello_matmul_gpu/tensor_sched_policy_matmul_gpu_generator.py). Benchmarking this `.hat` package on a AMD MI100 system outputs the following, which shows how using pass-order scheduling can yield slightly better performance:

```shell
                                    function_name        mean  median_of_means  mean_of_small_means  robust_mean  min_of_means
0  tensor_block_order_matmul_gpu_a0f5a6cd5453d086 29.03946851      29.04570557          29.02557666  29.04261027   29.00701904
1   tensor_pass_order_matmul_gpu_0d6383ac17fdfc9c 26.11595337      26.12025391          26.09279639  26.11305623   26.07321045
```

## Exercises for the reader
1. Try to tensorize a schedule with multiple tensor passes and see how different scheduling policies affect performance.
2. Now, try to fuse some of these passes and see if that changes anything.