[//]: # (Project: Accera)
[//]: # (Version: v1.2)

# Section 11: Plans - GPU Tensorization
In this section we will look more closely at how we can utilize tensor cores on supported GPUs to accelerate matrix multiplication operations.

## Related Concepts
Since tensor cores on the GPU can perform matrix multiplication of some standard shapes, we need to first familiarize ourselves with some of the associated terminology:
- __MMA shape__ - the smallest tensorizable matrix multiplication shape. In other words, nest of this shape or its multiple can be executed on tensor cores. Accera supports [MMA shapes](../Reference/enumerations/MMAShape.md) in the form of __MmxNnxKk_Bb__ which performs matrix multiplication of shape {__m__, __n__, __k__}, i.e., `C` += `A` x `B`, where matrix `A` is of shape {__m__, __k__}, matrix `B` is of shape {__k__, __n__} and the result matrix `C` is of shape {__m__, __n__}. The MMA shape can be specified by setting the `mma_shape` parameter in the `plan.tensorize` function call.
- __Tensor pass__ - A single tensor pass refers to a single unit of tensor operation. For example, a single pass of the MMA shape `M16xN16xK4_B1` performs matrix multiplication of shape {16, 16, 4}, whereas 4 passes of the same MMA shape performs a matmul of shape {16, 16, 16} in 4 iterations (passes) where each pass performs a matmul of shape {16, 16, 4}. The number of passes can be controlled by setting the `num_total_passes` parameter in the `plan.tensorize` function call.

## Tuning parameters
- __Pass fusing/grouping__ - A group of passes can be fused together to control allocation of registers required for input data (`A` and `B` matrices) and memory I/O density during tensor matmul. This is explained in more detail in the [Multi-Pass Tensorized MatMul with Pass Fusion](../Tutorials/GPU/Tensor_MatMul_MultiPass.md) tutorial.
- __Scheduling policy__ - This parameter can be used to tune register usage for accumulator data (`C` matrix) for multi-block tensor shapes. This is explained in more detail in [Tensor MatMul on GPU: Scheduling Policy experiments](../Tutorials/GPU/Tensor_MatMul_SchedulingPolicy_GPU.md) tutorial.
- __Prologue/Epilogue Ops__ - These parameters can be set to perform element-wise ops before and after matmul operations on tensor cores in an optimized way. Examples of this usage is presented in the [Tensor MatMul on GPU: Fused Element-wise Operations](../Tutorials/GPU/Tensor_MatMul_ElementWiseOps_GPU.md) tutorial.

<div style="page-break-after: always;"></div>
