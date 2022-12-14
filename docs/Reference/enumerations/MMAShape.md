[//]: # (Project: Accera)
[//]: # (Version: v1.2.13)

# Accera v1.2.13 Reference
## `accera.MMAShape`

The following table shows the matrix multiplication parameters associated with the different enum values, for different data types for a single pass. So for example a single pass of the `M32xN32xK2_B1` operation would take input matrices of dimensions [32x2] (A) and [2x32] (B) to produce a matrix multiplication result of dimensions [32x32] (C). These operations can then be composed together to perform matrix multiplication of larger matrices.

More information about the corresponding Matrix Arithmetic Instructions (MAI) can be found [here](https://developer.amd.com/wp-content/resources/CDNA1_Shader_ISA_14December2020.pdf).

<style>
table, td {
   border: 1px solid black;
}
th {
   border: 2px solid black;
   background-color:grey;
}
</style>

<table>
    <caption>Supported MMA shapes and their compatible types for AMD targets</caption>
    <tr>
        <th>accera.MMAShape</th>
        <th>MFMA Instruction</th>
        <th>M, N, K</th>
        <th>Input Type (ScalarType)</th>
        <th>Output Type (ScalarType)</th>
        <th>Compute Type (C++)</th>
    </tr>
    <tr>
        <td>M64xN64xK1_B4</td>
        <td>V_MFMA_F32_16x16x1F32</td>
        <td rowspan="2" style="text-align:center;vertical-align:middle;">64, 64, 1</td>
        <td rowspan="4" style="text-align:center;vertical-align:middle;">float32</td>
        <td rowspan="4" style="text-align:center;vertical-align:middle;">float32</td>
        <td rowspan="9" style="text-align:center;vertical-align:middle;">float</td>
    </tr>
    <tr>
        <td>M64xN64xK1_B2</td>
        <td>V_MFMA_F32_32x32x1F32</td>
    </tr>
    <tr>
        <td>M32xN32xK2_B1</td>
        <td>V_MFMA_F32_32x32x2F32</td>
        <td style="text-align:center;">32, 32, 2</td>
    </tr>
    <tr>
        <td>M16xN16xK4_B1</td>
        <td>V_MFMA_F32_16x16x4F32</td>
        <td style="text-align:center">16, 16, 4</td>
    </tr>
    <tr>
        <td style="vertical-align:middle;">M64xN64xK2_B4</td>
        <td style="vertical-align:middle;">V_MFMA_F32_16X16X2BF16</td>
        <td rowspan="2" style="text-align:center;vertical-align:middle;">64, 64, 2</td>
        <td rowspan="4" style="text-align:center;vertical-align:middle;">bfloat16</td>
        <td style="text-align:center;vertical-align:middle;">bfloat16/float32</td>
    </tr>
    <tr>
        <td style="vertical-align:middle;">M64xN64xK2_B2</td>
        <td style="vertical-align:middle;">V_MFMA_F32_32X32X2BF16</td>
        <td style="text-align:center;vertical-align:middle;">bfloat16/float32</td>
    </tr>
    <tr>
        <td style="vertical-align:middle;">M32xN32xK4_B1</td>
        <td style="vertical-align:middle;">V_MFMA_F32_32X32X4BF16</td>
        <td style="text-align:center;vertical-align:middle;">32, 32, 4</td>
        <td style="text-align:center;vertical-align:middle;">bfloat16/float32</td>
    </tr>
    <tr>
        <td style="vertical-align:middle;">M16xN16xK8_B1</td>
        <td style="vertical-align:middle;">V_MFMA_F32_16X16X8BF16</td>
        <td style="text-align:center;vertical-align:middle;">16, 16, 8</td>
        <td style="text-align:center;vertical-align:middle;">bfloat16/float32</td>
    </tr>
    <tr>
        <td rowspan="2" style="vertical-align:middle;">M64xN64xK4_B4</td>
        <td style="vertical-align:middle;">V_MFMA_F32_16x16x4F16</td>
        <td rowspan="4" style="text-align:center;vertical-align:middle;">64, 64, 4</td>
        <td style="text-align:center;vertical-align:middle;">float16</td>
        <td style="text-align:center;vertical-align:middle;">float16/32</td>
    </tr>
    <tr>
        <td style="vertical-align:middle;">V_MFMA_I32_16X16X4I8</td>
        <td style="text-align:center;vertical-align:middle;">int8</td>
        <td style="text-align:center;vertical-align:middle;">int8/16/32</td>
        <td style="text-align:center;vertical-align:middle;">int</td>
    </tr>
    <tr>
        <td rowspan="2" style="vertical-align:middle;">M64xN64xK4_B2</td>
        <td style="vertical-align:middle;">V_MFMA_F32_32x32x4F16</td>
        <td style="text-align:center;vertical-align:middle;">float16</td>
        <td style="text-align:center;vertical-align:middle;">float16/32</td>
        <td style="text-align:center;vertical-align:middle;">float</td>
    </tr>
    <tr>
        <td style="vertical-align:middle;">V_MFMA_I32_32X32X4I8</td>
        <td style="text-align:center;vertical-align:middle;">int8</td>
        <td style="text-align:center;vertical-align:middle;">int8/16/32</td>
        <td style="text-align:center;vertical-align:middle;">int</td>
    </tr>
    <tr>
        <td rowspan="2" style="vertical-align:middle;">M32xN32xK8_B1</td>
        <td style="vertical-align:middle;">V_MFMA_F32_32x32x8F16</td>
        <td rowspan="2" style="text-align:center;vertical-align:middle;">32, 32, 8</td>
        <td style="text-align:center;vertical-align:middle;">float16</td>
        <td style="text-align:center;vertical-align:middle;">float16/32</td>
        <td style="text-align:center;vertical-align:middle;">float</td>
    </tr>
    <tr>
        <td style="vertical-align:middle;">V_MFMA_I32_32X32X8I8</td>
        <td style="text-align:center;vertical-align:middle;">int8</td>
        <td style="text-align:center;vertical-align:middle;">int8/16/32</td>
        <td style="text-align:center;vertical-align:middle;">int</td>
    </tr>
    <tr>
        <td rowspan="2" style="vertical-align:middle;">M16xN16xK16_B1</td>
        <td style="vertical-align:middle;">V_MFMA_F32_16x16x16F16</td>
        <td rowspan="2" style="text-align:center;vertical-align:middle;">16, 16, 16</td>
        <td style="text-align:center;vertical-align:middle;">float16</td>
        <td style="text-align:center;vertical-align:middle;">float16/32</td>
        <td style="text-align:center;vertical-align:middle;">float</td>
    </tr>
    <tr>
        <td style="vertical-align:middle;">V_MFMA_I32_16X16X16I8</td>
        <td style="text-align:center;vertical-align:middle;">int8</td>
        <td style="text-align:center;vertical-align:middle;">int8/16/32</td>
        <td style="text-align:center;vertical-align:middle;">int</td>
    </tr>
</table>

<table>
    <caption>Supported MMA shapes and their compatible types for Nvidia targets</caption>
    <tr>
        <th>accera.MMAShape</th>
        <th>M, N, K</th>
        <th>Input Type (ScalarType)</th>
        <th>Output Type (ScalarType)</th>
        <th>Compute Type (C++)</th>
    </tr>
    <tr>
        <td style="vertical-align:middle;">M16xN16xK8_B1</td>
        <td style="text-align:center;vertical-align:middle;">16, 16, 8</td>
        <td style="text-align:center;vertical-align:middle;">float32</td>
        <td style="text-align:center;vertical-align:middle;">float32</td>
        <td style="text-align:center;vertical-align:middle;">tf32<sup>*</sup></td>
    </tr>
    <tr>
        <td rowspan="3" style="vertical-align:middle;">M16xN16xK16_B1</td>
        <td rowspan="3" style="text-align:center;vertical-align:middle;">16, 16, 16</td>
        <td style="text-align:center;vertical-align:middle;">float16</td>
        <td style="text-align:center;vertical-align:middle;">float16/32</td>
        <td rowspan="2" style="text-align:center;vertical-align:middle;">float</td>
    </tr>
    <tr>
        <td style="text-align:center;vertical-align:middle;">bfloat16</td>
        <td style="text-align:center;vertical-align:middle;">float32</td>
    </tr>
    <tr>
        <td style="text-align:center;vertical-align:middle;">u/int8</td>
        <td style="text-align:center;vertical-align:middle;">int32</td>
        <td style="text-align:center;vertical-align:middle;">int</td>
    </tr>
    <tr>
        <td rowspan="3" style="vertical-align:middle;">M32xN8xK16_B1</td>
        <td rowspan="3" style="text-align:center;vertical-align:middle;">32, 8, 16</td>
        <td style="text-align:center;vertical-align:middle;">float16</td>
        <td style="text-align:center;vertical-align:middle;">float16/32</td>
        <td rowspan="2" style="text-align:center;vertical-align:middle;">float</td>
    </tr>
    <tr>
        <td style="text-align:center;vertical-align:middle;">bfloat16</td>
        <td style="text-align:center;vertical-align:middle;">float32</td>
    </tr>
    <tr>
        <td style="text-align:center;vertical-align:middle;">u/int8</td>
        <td style="text-align:center;vertical-align:middle;">int32</td>
        <td style="text-align:center;vertical-align:middle;">int</td>
    </tr>
    <tr>
        <td rowspan="3" style="vertical-align:middle;">M8xN32xK16_B1</td>
        <td rowspan="3" style="text-align:center;vertical-align:middle;">8, 32, 16</td>
        <td style="text-align:center;vertical-align:middle;">float16</td>
        <td style="text-align:center;vertical-align:middle;">float16/32</td>
        <td rowspan="2" style="text-align:center;vertical-align:middle;">float</td>
    </tr>
    <tr>
        <td style="text-align:center;vertical-align:middle;">bfloat16</td>
        <td style="text-align:center;vertical-align:middle;">float32</td>
    </tr>
    <tr>
        <td style="text-align:center;vertical-align:middle;">u/int8</td>
        <td style="text-align:center;vertical-align:middle;">int32</td>
        <td style="text-align:center;vertical-align:middle;">int</td>
    </tr>
</table>

<div style="page-break-after: always;"></div>

<a name="m">*</a>TensorFloat-32 is a floating-point type introduced in the Nvidia Ampere architecture for accelerating FP32 performance. Information about this can be found [here](https://blogs.nvidia.com/blog/2020/05/14/tensorfloat-32-precision-format/) and in more detail in the [architecture whitepaper](https://images.nvidia.com/aem-dam/en-zz/Solutions/data-center/nvidia-ampere-architecture-whitepaper.pdf). In this mode, multiplication is performed in TF32 precision and accumulation happens in FP32 precision.
