[//]: # (Project: Accera)
[//]: # (Version: v1.2)

# Accera Tutorials

|Tutorial|Description|
|--|--|
|[Hello Matrix Multiplication](Hello_MatMul.md)|Start here if you are completely new to Accera and would like to learn more about the workflow|
|[Optimized Matrix Multiplication](Optimized_MatMul.md)|Once you understand the basics, we'll look at how to optimize matrix multiplication for a specific hardware target |
|[Cross Compilation for Raspberry Pi 3](Pi3_Cross_Compilation.md)|After you know how to generate code for the host target, we'll look at how to generate code for other targets|
|[[GPU] Hello Matrix Multiplication](GPU/Hello_MatMul_GPU.md)| We'll look at how to apply the basic concepts for GPU targets |
|[[GPU] Tensorized Matrix Multiplication](GPU/Tensor_MatMul_GPU.md)| Explains the basic usage of Tensor cores on GPU |
|[[GPU] Multi-Pass Tensorized MatMul with Pass Fusion](GPU/Tensor_MatMul_MultiPass.md)| Shows how pass fusion can be used to control register usage of input data |
|[[GPU] Tensorized MatMul with Caching](GPU/Tensor_MatMul_Caching_GPU.md)| Explores shared memory and register caching techniques on GPU |
|[[GPU] Tensorized MatMul with Element-wise Op fusion](GPU/Tensor_MatMul_ElementWiseOps_GPU.md)| Enhanced Matmul with element-wise pre/post matmul OP fusion |
|[[GPU] Multi-Block Tensorized MatMul with different Scheduling Policies](GPU/Tensor_MatMul_SchedulingPolicy_GPU.md)| Explains tradeoffs between register usage and memory I/O, and their performance impact |