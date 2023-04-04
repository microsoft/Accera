[//]: # (Project: Accera)
[//]: # (Version: v1.2)

# Tensor MatMul on GPU: Basic Tutorial

In this tutorial, you will learn how to implement a Matrix Multiplication (MatMul) function that uses specialized matrix multiplication hardware on the GPU.

## Prerequisites
* You have familiarized yourself with the concepts presented in the Tensorization manual [page](../../Manual/11%20Plans%20-%20GPU%20Tensorization.md).
* You have completed the [Hello_MatMul_GPU](Hello_MatMul_GPU.md) tutorial.

## Accera Implementation
To _tensorize_ a nest we need to create additional splits such that the innermost 3 dimensions are of shape {__m__, __n__, __p__ * __k__} for MMA shape __MmxNnxKk_Bb__, where __p__ is the number of passes. For example, when using MMA shape `M16xN16xK4_B1`, for a single pass we need to split the innermost dimensions to be of shape {16, 16, 4} for them to be _tensorizable_. The split factors for a specific MMA shape and the given number of passes can be calculated by the use of the helper function `compute_tensor_splits` on the `TensorCoreInformation` class as shown below:
```python
mma_shape = acc.MMAShape.M16xN16xK4_B1
tensor_splits = target.tensor_core_info.compute_tensor_splits(mma_shape) # num_total_passes = 1
iii, jjj, kk = schedule.tile({
    ii: tensor_splits[0],
    jj: tensor_splits[1],
    k: tensor_splits[2]
})
```

Type compatibility for the different values of `MMAShape` is documented [here](../../Reference/enumerations/MMAShape.md).

Now we would want to schedule the block-level iteration space as the outermost iteration, followed by warp-level iteration space and finally the tensorized iteration space:
```python
block_indices = (i, j)
warp_indices = (ii, jj)
tensor_indices = (iii, jjj, kk)

schedule.reorder(*block_indices, k, *warp_indices, *tensor_indices)
```

Since tensor core primitives operate at a warp level rather than at the thread level, we bind the schedule dimensions to the corresponding warp indices:
```python
plan.bind({
    i: target.GridUnit.BLOCK_Y,
    j: target.GridUnit.BLOCK_X,
    ii: target.GridUnit.WARP_Y,
    jj: target.GridUnit.WARP_X
})
```

After having done all the necessary splits and setting their order, we need to call [`tensorize(...)`](../../Reference/classes/Plan/tensorize.md) on the `plan` object. This will cause the lowering logic to produce warp-level matrix multiplication primitives which can utilize specialized tensor cores on the GPU:
```python
plan.tensorize(indices=tensor_indices, mma_shape=mma_shape)
```

By now, you have all the code necessary to generate an Accera MatMul function that runs on tensor cores on the GPU. You can find the complete Python script [here](../hello_matmul_gpu/tensor_matmul_gpu_generator.py).

### Generate HAT package

Next, we run the generator script to produce a HAT package.

```shell
python tensor_matmul_gpu_generator.py
```

The `.cu` source file now contains a launcher with different grid parameters based on the splits created for tensorization:

#### Host launcher

```c
#if !defined(__HIP_DEVICE_COMPILE__)
void tensor_matmul_gpu_cb6af7a31162e75c_impl_2389286605904206643(float *arg0, float *arg1, float *arg2) {
    tensor_matmul_gpu_cb6af7a31162e75c__gpu__<<<dim3(32, 64, 1), dim3(128, 2, 1), 0>>>(arg0, arg1, arg2);
    return;
}


#endif // !defined(__HIP_DEVICE_COMPILE__)
#if !defined(__HIP_DEVICE_COMPILE__)
extern "C" __host__ void tensor_matmul_gpu_cb6af7a31162e75c(float *arg0, float *arg1, float *arg2) {
    tensor_matmul_gpu_cb6af7a31162e75c_impl_2389286605904206643(arg0, arg1, arg2);
    return;
}
```

#### GPU kernel
The GPU kernel now contains warp-level primitives for loading, multiplying and storing matrices using tensor cores:
```c
extern "C" __global__  __launch_bounds__(256) void tensor_matmul_gpu_cb6af7a31162e75c__gpu__(float *arg0, float *arg1, float *arg2) {
    // Calculate threadid offsets and other locals
    /*...*/

    // Main K-loop
    for (int32_t idx14 = 0; idx14 < 512; idx14 += 1) {
        int32_t var15 = idx14 * 4;

        // Declare matrix fragments for A, B and C
        rocwmma::fragment<rocwmma::matrix_a, 16, 16, 4, 1, 1, float, rocwmma::row_major> mmaMatrix_16;
        rocwmma::fragment<rocwmma::matrix_b, 16, 16, 4, 1, 1, float, rocwmma::row_major> mmaMatrix_17;
        rocwmma::fragment<rocwmma::accumulator, 16, 16, 4, 1, 1, float> mmaMatrix_18;

        // Load matrix fragments from global memory
        rocwmma::load_matrix_sync<0, rocwmma::layout_t::mem_row_major, 1024>(var7, mmaMatrix_18, arg2 + ...);
        rocwmma::load_matrix_sync<2048>(var7, mmaMatrix_16, arg0 + ...);
        rocwmma::load_matrix_sync<1024>(var7, mmaMatrix_17, arg1 + ...);

        // Compute matrix multiplication
        rocwmma::mma_sync<0, 0, 0>(mmaMatrix_18, mmaMatrix_16, mmaMatrix_17, mmaMatrix_18);

        // Store result into global memory
        rocwmma::store_matrix_sync<0, rocwmma::layout_t::mem_row_major, 1024>(var7, arg2 + ..., mmaMatrix_18);
    }
}
```

### Benchmarking results using `hatlib`
Running the following command on machine with an AMD MI100 GPU:
```shell
python -m hatlib.benchmark_hat_package <path to tensor_matmul_gpu.hat> --cpp --min_time_in_sec 10 --time_in_ms
```
produces the following output. Note that compared to the non-tensorized implementation presented in [Hello_MatMul_GPU.md](Hello_MatMul_GPU.md#execution-and-benchmarking-gpu-kernels-using-hatlib-recommended), this implementation takes only __4.3 ms__ which is roughly __3x__ faster:
```shell
                        function_name       mean  median_of_means  mean_of_small_means  robust_mean  min_of_means
0  tensor_matmul_gpu_cb6af7a31162e75c 4.33855977       4.33607330           4.33025297   4.33672341    4.32588257
```