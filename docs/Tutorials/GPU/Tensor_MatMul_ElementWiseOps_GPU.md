[//]: # (Project: Accera)
[//]: # (Version: v1.2)

# Tensor MatMul on GPU: Fused Element-wise Operations

In this tutorial, you will learn how to implement a composite op which performs Matrix Multiplication (MatMul) on GPU tensor cores along with element-wise operations on the result data in an optimized way.

## Prerequisites
* You have completed the [Tensor_MatMul_GPU](Tensor_MatMul_GPU.md) tutorial.

## Accera implementation
Often times we want to perform element-wise operation (auxiliary) on the data before/after the main operation. For example, for non-unit alpha (α != 1) in GEMM, we want to scale each element of the matrix multiplication result by a constant value. For these scenarios, we enable the use of prologue and epilogue ops on tensor data for pre/post processing of data held in tensor fragments. Note that when tensorization is not used, this can be simply achieved by fusion.

### Prologue Tensor Operation
This operation is applied on each element of the tensor fragment data before matrix multiplication is done. Here is an example where we use the `SET` operation to zero-initialize the result fragment before matmul is performed:

```python
plan.tensorize(indices=tensor_indices, mma_shape=mma_shape, prologue_op=acc.MMAFragmentOp.SET, prologue_arg=0.0)
```

The complete python script with 0-init of result data can be found [here](../hello_matmul_gpu/tensor_zero_init_matmul_gpu_generator.py). The generated source code looks something like this (note the `fill_fragment` call below instead of `load_matrix_sync`):

```c
extern "C" __global__  __launch_bounds__(256) void tensor_zero_init_matmul_gpu_d6f8bdcfb87f53cc__gpu__(float *arg0, float *arg1, float *arg2) {
    // Calculate threadid offsets and other locals
    /*...*/

    // Main K-loop
    for (int32_t idx14 = 0; idx14 < 512; idx14 += 1) {
        // Declare matrix fragments for A, B and C
        /*...*/

        // Tensor Prologue: Fill result fragment with 0s
        rocwmma::fill_fragment(mmaMatrix_18, float{});

        // Matrix multiplication
        rocwmma::load_matrix_sync<2048>(var7, mmaMatrix_16, arg0 + ...);
        rocwmma::load_matrix_sync<1024>(var7, mmaMatrix_17, arg1 + ...);
        rocwmma::mma_sync<0, 0, 0>(mmaMatrix_18, mmaMatrix_16, mmaMatrix_17, mmaMatrix_18);

        // Store result into global memory
        rocwmma::store_matrix_sync<0, rocwmma::layout_t::mem_row_major, 1024>(var7, arg2 + ..., mmaMatrix_18);
    }
}
```

### Epilogue Tensor Operation
This operation is applied on each element of the tensor fragment data after matrix multiplication is done. Similar to `prologue_op`, we can also set the `epilogue_op` argument to achieve this. Here is an example of how alpha scaling of matmul result can be done with the following code:

```python
plan.tensorize(indices=tensor_indices, mma_shape=mma_shape, prologue_op=acc.MMAFragmentOp.SET, prologue_arg=0.0, epilogue_op=acc.MMAFragmentOp.SCALE, epilogue_arg=5.0)
```

The complete python script with alpha-scaling of result data can be found [here](../hello_matmul_gpu/tensor_alpha_scaling_matmul_gpu_generator.py). The generated source code looks something like this:

```c
extern "C" __global__  __launch_bounds__(64) void tensor_alpha_scaling_matmul_gpu_e5c7114024bfca18__gpu__(float *arg0, float *arg1, float *arg2) {
    // Calculate threadid offsets and other locals
    /*...*/

    // Main K-loop
    for (int32_t idx14 = 0; idx14 < 512; idx14 += 1) {
        // Declare matrix fragments for A, B and C
        /*...*/

        // Tensor Prologue: Fill result fragment with 0s
        rocwmma::fill_fragment(mmaMatrix_18, float{});

        // Matrix multiplication
        rocwmma::load_matrix_sync<2048>(var7, mmaMatrix_16, arg0 + ...);
        rocwmma::load_matrix_sync<1024>(var7, mmaMatrix_17, arg1 + ...);
        rocwmma::mma_sync<0, 0, 0>(mmaMatrix_18, mmaMatrix_16, mmaMatrix_17, mmaMatrix_18);

        // Tensor Epilogue: Alpha scaling of matmul result
        {
            float* mmaMatrix_18_data = reinterpret_cast<float*>(&mmaMatrix_18);
            for (int i = 0; i < sizeof(mmaMatrix_18) / sizeof(float); ++i) {
                mmaMatrix_18_data[i] *= float{5.000000e+00};
            }
        }

        // Store result into global memory
        rocwmma::store_matrix_sync<0, rocwmma::layout_t::mem_row_major, 1024>(var7, arg2 + ..., mmaMatrix_18);
    }
}
```

### General Matrix Multiply (GEMM) using Prologue and Epilogue Ops
The general matric multiplication problem can be formulated as below, where `A`, `B`, and `C` are matrices and `α` and `β` are scalar constants:
```
C = α.(A @ B) + β.C
```

GEMM with some of the different combinations of `α` and `β` can be computed using prologue and epilogue ops along with tensorization as summarized in the table below:

 | | β == 0 | β == 1 | β != 1
--- | --- | --- | ---
__α==1__ | __C = A @ B__: `prologue_op` = MMAFragmentOp.SET, `prologue_arg` = 0 | __C += A @ B__: _no need to set prologue and epilogue args_ | __C = A @ B + β.C__: `prologue_op` = MMAFragmentOp.SCALE, `prologue_arg` = β
__α!=1__ | __C = α.(A @ B)__: `prologue_op` = MMAFragmentOp.SET, `prologue_arg` = 0, `epilogue_op` = MMAFragmentOp.SCALE, `epilogue_arg` = α | __C += α.(A @ B)__: _use fusion_ | __C = α.(A @ B) + β.C__: _use fusion_
