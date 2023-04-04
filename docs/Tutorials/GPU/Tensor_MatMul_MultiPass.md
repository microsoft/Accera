[//]: # (Project: Accera)
[//]: # (Version: v1.2)

# Tensor MatMul on GPU: Multi-Pass

In this tutorial, we will see how to direct Accera compiler to generate multi-pass tensorization code and how we can use pass fusion to control input data (matrices `A` and `B`) register allocation.

## Prerequisites
* You have completed the [Tensor_MatMul_GPU](Tensor_MatMul_GPU.md) tutorial.

## Background
Multi-pass tensorization allows for unrolling of tensor loops in the K-dimension as shown in the examples below. Fusion of passes allows for better control of registers required for input data (`A`/`B`). For example, if the total number of passes to be executed is 8 and number of fused passes is 2, this means there are 4 pass groups of 2 fused passes each. This generates code which does the following 4 times (once per pass group):
- Allocate registers for input data required for 2 passes
- Load input data for 2 passes
- Compute matmul for 2 passes
- Store result for 2 passes

In effect, increasing the number of fused passes leads to increased register usage for input data and more densely packed memory I/O and compute instructions. Similarly, reducing the number of fused passes implies less register pressure and more interleaving among the memory I/O and the compute instructions. The number of passes to be fused can be controlled by setting the `num_fused_passes` parameter in the `plan.tensorize` function call.

## Full Pass Fusion
We are going to change the call to `compute_tensor_splits` to pass the `num_total_passes` argument and also set it on the `plan.tensorize` call. Not setting the `num_fused_passes` argument on the `plan.tensorize` leaves it in its default setting which is to fuse all the passes:

```python
num_passes = 8
tensor_splits = target.tensor_core_info.compute_tensor_splits(mma_shape, num_passes)
...
plan.tensorize(indices=tensor_indices, mma_shape=mma_shape, num_total_passes=num_passes)
```

The generated source code allocates registers for input data for all the 8 passes and loads them all before performing matrix multiplication as shown below:

```c
extern "C" __global__  __launch_bounds__(256) void tensor_multipass_full_fusion_matmul_gpu_9d020531735492ef__gpu__(float *arg0, float *arg1, float *arg2) {
    // Calculate threadid offsets and other locals
    /*...*/

    // Main K-loop
    for (int32_t idx14 = 0; idx14 < 64; idx14 += 1) {
        int32_t var15 = idx14 * 32;

        // Declare matrix fragments for A (8 passes)
        rocwmma::fragment<rocwmma::matrix_a, 16, 16, 4, 1, 1, float, rocwmma::row_major> mmaMatrix_16;
        rocwmma::fragment<rocwmma::matrix_a, 16, 16, 4, 1, 1, float, rocwmma::row_major> mmaMatrix_17;
        rocwmma::fragment<rocwmma::matrix_a, 16, 16, 4, 1, 1, float, rocwmma::row_major> mmaMatrix_18;
        rocwmma::fragment<rocwmma::matrix_a, 16, 16, 4, 1, 1, float, rocwmma::row_major> mmaMatrix_19;
        rocwmma::fragment<rocwmma::matrix_a, 16, 16, 4, 1, 1, float, rocwmma::row_major> mmaMatrix_20;
        rocwmma::fragment<rocwmma::matrix_a, 16, 16, 4, 1, 1, float, rocwmma::row_major> mmaMatrix_21;
        rocwmma::fragment<rocwmma::matrix_a, 16, 16, 4, 1, 1, float, rocwmma::row_major> mmaMatrix_22;
        rocwmma::fragment<rocwmma::matrix_a, 16, 16, 4, 1, 1, float, rocwmma::row_major> mmaMatrix_23;

        // Declare matrix fragments for B (8 passes)
        rocwmma::fragment<rocwmma::matrix_b, 16, 16, 4, 1, 1, float, rocwmma::row_major> mmaMatrix_24;
        rocwmma::fragment<rocwmma::matrix_b, 16, 16, 4, 1, 1, float, rocwmma::row_major> mmaMatrix_25;
        rocwmma::fragment<rocwmma::matrix_b, 16, 16, 4, 1, 1, float, rocwmma::row_major> mmaMatrix_26;
        rocwmma::fragment<rocwmma::matrix_b, 16, 16, 4, 1, 1, float, rocwmma::row_major> mmaMatrix_27;
        rocwmma::fragment<rocwmma::matrix_b, 16, 16, 4, 1, 1, float, rocwmma::row_major> mmaMatrix_28;
        rocwmma::fragment<rocwmma::matrix_b, 16, 16, 4, 1, 1, float, rocwmma::row_major> mmaMatrix_29;
        rocwmma::fragment<rocwmma::matrix_b, 16, 16, 4, 1, 1, float, rocwmma::row_major> mmaMatrix_30;
        rocwmma::fragment<rocwmma::matrix_b, 16, 16, 4, 1, 1, float, rocwmma::row_major> mmaMatrix_31;

        // Declare matrix fragment for C
        rocwmma::fragment<rocwmma::accumulator, 16, 16, 4, 1, 1, float> mmaMatrix_32;

        // Load matrix fragment for C
        rocwmma::load_matrix_sync<0, rocwmma::layout_t::mem_row_major, 1024>(var7, mmaMatrix_32, arg2 + ...);

        // Load matrix fragments for A (8 passes)
        rocwmma::load_matrix_sync<2048>(var7, mmaMatrix_16, arg0 + ...);
        int32_t var33 = var15 + 4;
        rocwmma::load_matrix_sync<2048>(var7, mmaMatrix_17, arg0 + ...);
        int32_t var34 = var15 + 8;
        rocwmma::load_matrix_sync<2048>(var7, mmaMatrix_18, arg0 + ...);
        int32_t var35 = var15 + 12;
        rocwmma::load_matrix_sync<2048>(var7, mmaMatrix_19, arg0 + ...);
        int32_t var36 = var15 + 16;
        rocwmma::load_matrix_sync<2048>(var7, mmaMatrix_20, arg0 + ...);
        int32_t var37 = var15 + 20;
        rocwmma::load_matrix_sync<2048>(var7, mmaMatrix_21, arg0 + ...);
        int32_t var38 = var15 + 24;
        rocwmma::load_matrix_sync<2048>(var7, mmaMatrix_22, arg0 + ...);
        int32_t var39 = var15 + 28;
        rocwmma::load_matrix_sync<2048>(var7, mmaMatrix_23, arg0 + ...);

        // Load matrix fragments for B (8 passes)
        rocwmma::load_matrix_sync<1024>(var7, mmaMatrix_24, arg1 + ...);
        rocwmma::load_matrix_sync<1024>(var7, mmaMatrix_25, arg1 + ...);
        rocwmma::load_matrix_sync<1024>(var7, mmaMatrix_26, arg1 + ...);
        rocwmma::load_matrix_sync<1024>(var7, mmaMatrix_27, arg1 + ...);
        rocwmma::load_matrix_sync<1024>(var7, mmaMatrix_28, arg1 + ...);
        rocwmma::load_matrix_sync<1024>(var7, mmaMatrix_29, arg1 + ...);
        rocwmma::load_matrix_sync<1024>(var7, mmaMatrix_30, arg1 + ...);
        rocwmma::load_matrix_sync<1024>(var7, mmaMatrix_31, arg1 + ...);

        // Matrix multiplication for Pass Group 0 (8 passes)
        rocwmma::mma_sync<0, 0, 0>(mmaMatrix_32, mmaMatrix_16, mmaMatrix_24, mmaMatrix_32);
        rocwmma::mma_sync<0, 0, 0>(mmaMatrix_32, mmaMatrix_17, mmaMatrix_25, mmaMatrix_32);
        rocwmma::mma_sync<0, 0, 0>(mmaMatrix_32, mmaMatrix_18, mmaMatrix_26, mmaMatrix_32);
        rocwmma::mma_sync<0, 0, 0>(mmaMatrix_32, mmaMatrix_19, mmaMatrix_27, mmaMatrix_32);
        rocwmma::mma_sync<0, 0, 0>(mmaMatrix_32, mmaMatrix_20, mmaMatrix_28, mmaMatrix_32);
        rocwmma::mma_sync<0, 0, 0>(mmaMatrix_32, mmaMatrix_21, mmaMatrix_29, mmaMatrix_32);
        rocwmma::mma_sync<0, 0, 0>(mmaMatrix_32, mmaMatrix_22, mmaMatrix_30, mmaMatrix_32);
        rocwmma::mma_sync<0, 0, 0>(mmaMatrix_32, mmaMatrix_23, mmaMatrix_31, mmaMatrix_32);

        // Store matrix fragment for C
        rocwmma::store_matrix_sync<0, rocwmma::layout_t::mem_row_major, 1024>(var7, arg2 + ..., mmaMatrix_32);
    }
}
```

## Partial Pass Fusion
With partial pass fusion, we will explicitly set the `num_fused_passes` argument to 2 on the `plan.tensorize` call as shown below:

```python
plan.tensorize(indices=tensor_indices, mma_shape=mma_shape, num_total_passes=num_passes, num_fused_passes=2)
```

By doing this we are effectively reducing the register requirement for input data by a factor of 4 since matrix multiplication is performed for only 2 passes at a time:

```c
extern "C" __global__  __launch_bounds__(256) void tensor_multipass_partial_fusion_matmul_gpu_0fcbcea0448782e8__gpu__(float *arg0, float *arg1, float *arg2) {
    // Calculate threadid offsets and other locals
    /*...*/

    // Main K-loop
    for (int32_t idx14 = 0; idx14 < 64; idx14 += 1) {
        int32_t var15 = idx14 * 32;

        // Declare matrix fragments for A (2 passes)
        rocwmma::fragment<rocwmma::matrix_a, 16, 16, 4, 1, 1, float, rocwmma::row_major> mmaMatrix_16;
        rocwmma::fragment<rocwmma::matrix_a, 16, 16, 4, 1, 1, float, rocwmma::row_major> mmaMatrix_17;

        // Declare matrix fragments for B (2 passes)
        rocwmma::fragment<rocwmma::matrix_b, 16, 16, 4, 1, 1, float, rocwmma::row_major> mmaMatrix_18;
        rocwmma::fragment<rocwmma::matrix_b, 16, 16, 4, 1, 1, float, rocwmma::row_major> mmaMatrix_19;

        // Declare matrix fragment for C
        rocwmma::fragment<rocwmma::accumulator, 16, 16, 4, 1, 1, float> mmaMatrix_20;

        // Load matrix fragment for C
        rocwmma::load_matrix_sync<0, rocwmma::layout_t::mem_row_major, 1024>(var7, mmaMatrix_20, arg2 + ...);

        // Matrix multiplication for Pass Group 0 (pass 0, 1)
        rocwmma::load_matrix_sync<2048>(var7, mmaMatrix_16, arg0 + ...);
        int32_t var21 = var15 + 4;
        rocwmma::load_matrix_sync<2048>(var7, mmaMatrix_17, arg0 + ...);
        rocwmma::load_matrix_sync<1024>(var7, mmaMatrix_18, arg1 + ...);
        rocwmma::load_matrix_sync<1024>(var7, mmaMatrix_19, arg1 + ...);
        rocwmma::mma_sync<0, 0, 0>(mmaMatrix_20, mmaMatrix_16, mmaMatrix_18, mmaMatrix_20);
        rocwmma::mma_sync<0, 0, 0>(mmaMatrix_20, mmaMatrix_17, mmaMatrix_19, mmaMatrix_20);

        // Matrix multiplication for Pass Group 1 (pass 2, 3)
        int32_t var22 = var15 + 8;
        rocwmma::load_matrix_sync<2048>(var7, mmaMatrix_16, arg0 + ...);
        int32_t var23 = var15 + 12;
        rocwmma::load_matrix_sync<2048>(var7, mmaMatrix_17, arg0 + ...);
        rocwmma::load_matrix_sync<1024>(var7, mmaMatrix_18, arg1 + ...);
        rocwmma::load_matrix_sync<1024>(var7, mmaMatrix_19, arg1 + ...);
        rocwmma::mma_sync<0, 0, 0>(mmaMatrix_20, mmaMatrix_16, mmaMatrix_18, mmaMatrix_20);
        rocwmma::mma_sync<0, 0, 0>(mmaMatrix_20, mmaMatrix_17, mmaMatrix_19, mmaMatrix_20);

        // Matrix multiplication for Pass Group 2 (pass 4, 5)
        int32_t var24 = var15 + 16;
        rocwmma::load_matrix_sync<2048>(var7, mmaMatrix_16, arg0 + ...);
        int32_t var25 = var15 + 20;
        rocwmma::load_matrix_sync<2048>(var7, mmaMatrix_17, arg0 + ...);
        rocwmma::load_matrix_sync<1024>(var7, mmaMatrix_18, arg1 + ...);
        rocwmma::load_matrix_sync<1024>(var7, mmaMatrix_19, arg1 + ...);
        rocwmma::mma_sync<0, 0, 0>(mmaMatrix_20, mmaMatrix_16, mmaMatrix_18, mmaMatrix_20);
        rocwmma::mma_sync<0, 0, 0>(mmaMatrix_20, mmaMatrix_17, mmaMatrix_19, mmaMatrix_20);

        // Matrix multiplication for Pass Group 3 (pass 6, 7)
        int32_t var26 = var15 + 24;
        rocwmma::load_matrix_sync<2048>(var7, mmaMatrix_16, arg0 + ...);
        int32_t var27 = var15 + 28;
        rocwmma::load_matrix_sync<2048>(var7, mmaMatrix_17, arg0 + ...);
        rocwmma::load_matrix_sync<1024>(var7, mmaMatrix_18, arg1 + ...);
        rocwmma::load_matrix_sync<1024>(var7, mmaMatrix_19, arg1 + ...);
        rocwmma::mma_sync<0, 0, 0>(mmaMatrix_20, mmaMatrix_16, mmaMatrix_18, mmaMatrix_20);
        rocwmma::mma_sync<0, 0, 0>(mmaMatrix_20, mmaMatrix_17, mmaMatrix_19, mmaMatrix_20);

        // Store matrix fragment for C
        rocwmma::store_matrix_sync<0, rocwmma::layout_t::mem_row_major, 1024>(var7, arg2 + var10 * 1024 + var13 * 1, mmaMatrix_20);
    }
}
```
