[//]: # (Project: Accera)
[//]: # (Version: v1.2)

# Tensor MatMul on GPU: Caching

In this tutorial, you will learn how to implement a Matrix Multiplication (MatMul) function that uses specialized matrix multiplication hardware on the GPU while caching data to minimize expensive data traffic to/from global memory.

## Prerequisites
* You have completed the [Tensor_MatMul_GPU](Tensor_MatMul_GPU.md) tutorial.

## Input data caching
Since accessing the same data repeatedly from the global memory can be expensive, we will use the shared memory which is available much closer to the compute units to cache the input data to achieve much faster data accesses.

Since for input data caching we use shared memory, we need to be careful how much of the global data we cache since shared memory is comparatively much smaller in size. For this reason, we introduce an additional split in the K-loop and we use this newly created loop index `kk` for caching:
```python
kk = schedule.split(k, 256)
```

### Sequential caching
In this approach, each thread block starts caching the next tile of input data only after the computation on the current tile is complete. This is the most simple form of shared memory caching which does not involve any overlapped execution of data copy and computation pipelines. This can be achieved by adding the following lines of DSL code:

```python
plan.cache(A, index=kk, location=target.MemorySpace.SHARED)
plan.cache(B, index=kk, location=target.MemorySpace.SHARED)
```

The complete python script with caching of input data into shared memory can be found [here](../hello_matmul_gpu/tensor_input_cache_matmul_gpu_generator.py).

This generates the following kernel code, note the barriers in the generated code to see how caching of the next tile waits for the computation of the current tile to finish:
```c
extern "C" __global__  __launch_bounds__(256) void tensor_input_cache_matmul_gpu_866f5763c1d8d520__gpu__(float *arg0, float *arg1, float *arg2) {
    // Calculate threadid offsets and other locals
    /*...*/

    // Declare shared memory caches for A and B
    __shared__ float var8[32][256];
    __shared__ float var9[256][32];

    // k-loop
    for (int32_t idx16 = 0; idx16 < 8; idx16 += 1) {
        int32_t var17 = idx16 * 256;

        // Wait for compute on previously cached items to finish
        __builtin_amdgcn_s_barrier();

        // Cache current tile of A into shared memory
        block_copy<CopyMode::Striped, /*SRC_ROW_MAJOR*/ 1, /*DST_ROW_MAJOR*/ 1, /*STRIDE*/ 1, /*WPT*/ 32, /*TILE_R,C*/32, 256, /*BLOCK_DIM_X,Y,Z*/ 128, 2, 1, MemSpace::None, MemSpace::Shared, float>(var11, (float*)arg0, var7, var17, affine_map_func_0_i0, (float*)var8);

        // Cache current tile of B into shared memory
        block_copy<CopyMode::Striped, /*SRC_ROW_MAJOR*/ 1, /*DST_ROW_MAJOR*/ 1, /*STRIDE*/ 1, /*WPT*/ 32, /*TILE_R,C*/256, 32, /*BLOCK_DIM_X,Y,Z*/ 128, 2, 1, MemSpace::None, MemSpace::Shared, float>(var11, (float*)arg1, var17, var5, affine_map_func_1_i0, (float*)var9);

        // Wait for input caching to finish
        __builtin_amdgcn_s_barrier();

        // kk-loop
        for (int32_t idx18 = 0; idx18 < 64; idx18 += 1) {
            int32_t var19 = idx18 * 4;

            // Declare matrix fragments for A, B and C
            /*...*/

            // Load C from global memory
            rocwmma::load_matrix_sync<0, rocwmma::layout_t::mem_row_major, 1024>(var11, mmaMatrix_22, arg2 + ...);

            // Load A and B from shared memory cache
            rocwmma::load_matrix_sync<256>(var11, mmaMatrix_20, &var8[var12][var19]);
            rocwmma::load_matrix_sync<32>(var11, mmaMatrix_21, &var9[var19][var14]);

            // Compute matrix multiplication
            rocwmma::mma_sync<0, 0, 0>(mmaMatrix_22, mmaMatrix_20, mmaMatrix_21, mmaMatrix_22);

            // Store result into global memory
            rocwmma::store_matrix_sync<0, rocwmma::layout_t::mem_row_major, 1024>(var11, arg2 + ..., mmaMatrix_22);
        }
    }
}
```

#### Benchmarking results using `hatlib`
Similar to the previous experiments we can use hatlib to benchmark this kernel using the following command:
```shell
python3 -m hatlib.benchmark_hat_package <path to tensor_input_cache_matmul_gpu.hat> --cpp --min_time_in_sec 10 --time_in_ms
```

This produces the following output which shows that sequential caching reduces the runtime to __~3 ms__ which is __~30%__ faster than the non-cached version presented in [Tensor_MatMul_GPU.md](Tensor_MatMul_GPU.md#benchmarking-results-using-hatlib):

```shell
                                    function_name       mean  median_of_means  mean_of_small_means  robust_mean  min_of_means
0  tensor_input_cache_matmul_gpu_866f5763c1d8d520 3.02507486       3.02532532           3.02407842   3.02519751    3.02233856
```

### Overlapped caching (a.k.a. Double Buffering)
In this approach, each thread block prefetches the next tile into registers while the current tile is being computed. This overlapped execution of data copy and compute typically achieves better performance by utilizing different hardware pipelines more efficiently. Using Accera DSL, this can be done by setting the `double_buffer` flag in the [`plan.cache`](../../Reference/classes/Plan/cache.md) call:

```python
plan.cache(A, index=kk, location=target.MemorySpace.SHARED, double_buffer=True, double_buffer_location=target.MemorySpace.PRIVATE)
plan.cache(B, index=kk, location=target.MemorySpace.SHARED, double_buffer=True, double_buffer_location=target.MemorySpace.PRIVATE)
```

The complete python script with caching of input data using double buffering can be found [here](../hello_matmul_gpu/tensor_input_double_buffer_cache_matmul_gpu_generator.py).

The generated kernel code looks something like this, note how the prefetch of the next tile and the computation of the current tile happen without synchronization to achieve _global memory latency hiding_:
```c
extern "C" __global__  __launch_bounds__(256) void tensor_input_double_buffer_cache_matmul_gpu_ce60189b3e52267d__gpu__(float *arg0, float *arg1, float *arg2) {
    // Calculate threadid offsets and other locals
    /*...*/

    // Declare register caches for prefetching input data of A and B
    float var8[32][1];
    float var9[32][1];

    // Declare shared memory caches for A and B
    __shared__ float var10[32][256];
    __shared__ float var11[256][32];

    // Cache tile 0 of A from global memory to shared memory
    block_copy<CopyMode::Striped, /*SRC_ROW_MAJOR*/ 1, /*DST_ROW_MAJOR*/ 1, /*STRIDE*/ 1, /*WPT*/ 32, /*TILE_R,C*/32, 256, /*BLOCK_DIM_X,Y,Z*/ 128, 2, 1, MemSpace::None, MemSpace::Shared, float>(var13, (float*)arg0, var7, 0, affine_map_func_0_i0, (float*)var10);

    // Cache tile 0 of B from global memory to shared memory
    block_copy<CopyMode::Striped, /*SRC_ROW_MAJOR*/ 1, /*DST_ROW_MAJOR*/ 1, /*STRIDE*/ 1, /*WPT*/ 32, /*TILE_R,C*/256, 32, /*BLOCK_DIM_X,Y,Z*/ 128, 2, 1, MemSpace::None, MemSpace::Shared, float>(var13, (float*)arg1, 0, var5, affine_map_func_1_i0, (float*)var11);

    // Wait for tile 0 data to finish copying
    __builtin_amdgcn_s_barrier();
    
    // k-loop (Current tile)
    for (int32_t idx18 = 0; idx18 < 7; idx18 += 1) {
        // Prefetch next tile of A from global memory to registers
        block_copy<CopyMode::Striped, /*SRC_ROW_MAJOR*/ 1, /*DST_ROW_MAJOR*/ 1, /*STRIDE*/ 1, /*WPT*/ 32, /*TILE_R,C*/32, 256, /*BLOCK_DIM_X,Y,Z*/ 128, 2, 1, MemSpace::None, MemSpace::Private, float>(
        var13, (float*)arg0, var7, var20, affine_map_func_0_i0, (float*)var9);

        // Prefetch next tile of B from global memory to registers
        block_copy<CopyMode::Striped, /*SRC_ROW_MAJOR*/ 1, /*DST_ROW_MAJOR*/ 1, /*STRIDE*/ 1, /*WPT*/ 32, /*TILE_R,C*/256, 32, /*BLOCK_DIM_X,Y,Z*/ 128, 2, 1, MemSpace::None, MemSpace::Private, float>(
        var13, (float*)arg1, var20, var5, affine_map_func_1_i0, (float*)var8);

        // kk-loop
        for (int32_t idx24 = 0; idx24 < 64; idx24 += 1) {
            // Declare matrix fragments for A, B and C
            /*...*/

            // Perform matmul on the current tile from shared memory
            rocwmma::load_matrix_sync<0, rocwmma::layout_t::mem_row_major, 1024>(var13, mmaMatrix_28, arg2 + ...);
            rocwmma::load_matrix_sync<256>(var13, mmaMatrix_26, &var10[var14][var25]);
            rocwmma::load_matrix_sync<32>(var13, mmaMatrix_27, &var11[var25][var16]);
            rocwmma::mma_sync<0, 0, 0>(mmaMatrix_28, mmaMatrix_26, mmaMatrix_27, mmaMatrix_28);
            rocwmma::store_matrix_sync<0, rocwmma::layout_t::mem_row_major, 1024>(var13, arg2 + ..., mmaMatrix_28);
        }

        // Wait for matmul on current tile to finish
        __builtin_amdgcn_s_barrier();

        // Copy prefetched data of A from registers to shared memory
        block_copy<CopyMode::Striped, /*SRC_ROW_MAJOR*/ 1, /*DST_ROW_MAJOR*/ 1, /*STRIDE*/ 1, /*WPT*/ 32, /*TILE_R,C*/256, 32, /*BLOCK_DIM_X,Y,Z*/ 128, 2, 1, MemSpace::Private, MemSpace::Shared, float>(var13, (float*)var11, 0, 0, affine_map_func_4_i0, (float*)var8);

        // Copy prefetched data of B from registers to shared memory
        block_copy<CopyMode::Striped, /*SRC_ROW_MAJOR*/ 1, /*DST_ROW_MAJOR*/ 1, /*STRIDE*/ 1, /*WPT*/ 32, /*TILE_R,C*/32, 256, /*BLOCK_DIM_X,Y,Z*/ 128, 2, 1, MemSpace::Private, MemSpace::Shared, float>(var13, (float*)var10, 0, 0, affine_map_func_3_i0, (float*)var9);

        // Wait for copy to finish before starting next tile
        __builtin_amdgcn_s_barrier();
    }

    // Last tile (loop peeling)
    for (int32_t idx19 = 0; idx19 < 64; idx19 += 1) {
        // Declare matrix fragments for A, B and C
        /*...*/

        // Perform matmul on the last tile from shared memory
        rocwmma::load_matrix_sync<0, rocwmma::layout_t::mem_row_major, 1024>(var13, mmaMatrix_23, arg2 + ...);
        rocwmma::load_matrix_sync<256>(var13, mmaMatrix_21, &var10[var14][var20]);
        rocwmma::load_matrix_sync<32>(var13, mmaMatrix_22, &var11[var20][var16]);
        rocwmma::mma_sync<0, 0, 0>(mmaMatrix_23, mmaMatrix_21, mmaMatrix_22, mmaMatrix_23);
        rocwmma::store_matrix_sync<0, rocwmma::layout_t::mem_row_major, 1024>(var13, arg2 + ..., mmaMatrix_23);
    }
}
```

#### Benchmarking results using `hatlib`
Benchmarking the above kernel with hatlib shows that double-buffer caching further reduces the runtime to __~1.45 ms__ which is __~66%__ faster than the non-cached version presented in [Tensor_MatMul_GPU.md](Tensor_MatMul_GPU.md#benchmarking-results-using-hatlib):

```shell
                                       function_name       mean  median_of_means  mean_of_small_means  robust_mean  min_of_means
0  tensor_input_double_buffer_cache_matmul_gpu_ce... 1.45501032       1.45495605           1.45370367   1.45489838    1.45116257
```


## Output data caching
Similar to input caching, the result data can also be cached to prevent unnecessary global memory accesses. Here we will see how we can accumulate the result in registers before copying it to global memory. This is can be done by adding:

```python
plan.cache(C, index=k, location=target.MemorySpace.MMA_FRAGMENT)
```

The complete python script with both input and output caching can be found [here](../hello_matmul_gpu/tensor_input_output_cache_matmul_gpu_generator.py).

```c
extern "C" __global__  __launch_bounds__(256) void tensor_input_output_cache_matmul_gpu_1b4d39ede237d688__gpu__(float *arg0, float *arg1, float *arg2) {
    // Calculate threadid offsets and other locals
    /*...*/

    // Declare register caches for prefetching input data of A and B
    float var8[32][1];
    float var9[32][1];

    // Declare shared memory caches for A and B
    __shared__ float var10[32][256];
    __shared__ float var11[256][32];

    // Declare fragment cache (registers) for output, C
    rocwmma::fragment<rocwmma::accumulator, 16, 16, 4, 1, 1, float> mmaMatrix_12;

    // Fill output cache with data from global memory
    rocwmma::load_matrix_sync<0, rocwmma::layout_t::mem_row_major, 1024>(var18, mmaMatrix_12, arg2 + ...);

    // Cache tile 0 of A and B from global memory to shared memory
    /*...*/

    // Wait for tile 0 data to finish copying
    __builtin_amdgcn_s_barrier();

    // k-loop (Current tile)
    for (int32_t idx19 = 0; idx19 < 7; idx19 += 1) {
        // Prefetch next tile of A and B from global memory to registers
        /*...*/

        // kk-loop
        for (int32_t idx25 = 0; idx25 < 64; idx25 += 1) {
            // Declare matrix fragments for A and B
            /*...*/

            // Load A and B from shared memory cache
            /*...*/

            // Compute matrix multiplication and accumulate in fragment cache
            rocwmma::mma_sync<0, 0, 0>(mmaMatrix_12, mmaMatrix_27, mmaMatrix_28, mmaMatrix_12);
        }

        // Wait for matmul on current tile to finish
        __builtin_amdgcn_s_barrier();

        // Copy prefetched data of A and B from registers to shared memory
        /*...*/

        // Wait for copy to finish before starting next tile
        __builtin_amdgcn_s_barrier();
    }

    // Last tile (loop peeling)
    for (int32_t idx20 = 0; idx20 < 64; idx20 += 1) {
        // Declare matrix fragments for A and B
        /*...*/

        // Load A and B from shared memory cache
        /*...*/

        // Compute matrix multiplication and accumulate in fragment cache
        rocwmma::mma_sync<0, 0, 0>(mmaMatrix_12, mmaMatrix_22, mmaMatrix_23, mmaMatrix_12);
    }

    // Store result into global memory ONCE!
    rocwmma::store_matrix_sync<0, rocwmma::layout_t::mem_row_major, 1024>(var18, arg2 + ..., mmaMatrix_12);
}
```

#### Benchmarking results using `hatlib`
Benchmarking the above kernel with hatlib shows that double-buffer caching combined with output caching further reduces the runtime to __~1.34 ms__ which is an overall __~69%__ improvement compared to the non-cached version presented in [Tensor_MatMul_GPU.md](Tensor_MatMul_GPU.md#benchmarking-results-using-hatlib):

```shell
                                       function_name       mean  median_of_means  mean_of_small_means  robust_mean  min_of_means
0  tensor_input_output_cache_matmul_gpu_1b4d39ede... 1.33967323       1.33956711           1.33841698   1.33953149    1.33726944
```