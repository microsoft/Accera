[//]: # (Project: Accera)
[//]: # (Version: v1.2.3)

# Accera v1.2.3 Reference
## `accera.MMAShape`

The following table shows the matrix multiplication parameters associated with the different enum values, for different data types for a single pass. So for example a single pass of the `M32xN32xK2_B1` operation would take input matrices of dimensions [32x2] (A) and [2x32] (B) to produce a matrix multiplication result of dimensions [32x32] (C). These operations can then be composed together to perform matrix multiplication of larger matrices.

More information about the corresponding Matrix Arithmetic Instructions (MAI) can be found [here](https://developer.amd.com/wp-content/resources/CDNA1_Shader_ISA_14December2020.pdf).

type | MFMA type | M, N, K | Inout type | Output Type |
---|---|---|---|---|
`accera.MMAShape.M64xN64xK1_B4` | V_MFMA_F32_16x16x1F32 | 64, 64, 1 | ScalarType.float32 | ScalarType.float32|
`accera.MMAShape.M64xM64xK1_B2` | V_MFMA_F32_32x32x1F32 | 64, 64, 1 | ScalarType.float32 | ScalarType.float32|
`accera.MMAShape.M32xN32xK2_B1` | V_MFMA_F32_32x32x2F32 | 32, 32, 2 | ScalarType.float32 | ScalarType.float32|
`accera.MMAShape.M16xN16xK4_B1` | V_MFMA_F32_16x16x4F32 | 16, 16, 4 | ScalarType.float32 | ScalarType.float32|
`accera.MMAShape.M64xN64xK4_B4` | V_MFMA_F32_16x16x4F16 | 64, 64, 4 | ScalarType.float16 | ScalarType.float16 / ScalarType.float32|
`accera.MMAShape.M64xM64xK4_B2` | V_MFMA_F32_32x32x4F16 | 64, 64, 4 | ScalarType.float16 | ScalarType.float16 / ScalarType.float32|
`accera.MMAShape.M32xN32xK8_B1` | V_MFMA_F32_32x32x8F16 | 32, 32, 8 |ScalarType.float16 | ScalarType.float16 / ScalarType.float32|
`accera.MMAShape.M16xN16xK16_B1` | V_MFMA_F32_16x16x16F16 | 16, 16, 16 | ScalarType.float16 | ScalarType.float16 / ScalarType.float32|

<div style="page-break-after: always;"></div>
