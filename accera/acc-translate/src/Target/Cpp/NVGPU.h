////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
//  Authors: Abdul Dakkak
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef DIALECT_ARGO_NVGPU_H_
#define DIALECT_ARGO_NVGPU_H_

#include <mlir/IR/BuiltinTypes.h>

namespace mlir
{
class MLIRContext;
class AffineMap;
namespace cpp_printer
{
    struct LayoutParam;

    // Memory space
    enum NVGPUMemorySpace : unsigned
    {
        Generic = 0,
        Global = 1,
        Reserved = 2,
        Shared = 3,
        Constant = 4,
        Local = 5,
        Register = 6,
        TensorFragmentReg = 7, // a virtual format of registers for MMA
    };

    int64_t getVectorWidthFromElementBit(int64_t bit);

    // MMA attribute
    constexpr const char* ArgoNVGPUMMAALayoutAttributeName = "alayout";
    constexpr const char* ArgoNVGPUMMABLayoutAttributeName = "blayout";

    // MMA register iteration map
    constexpr const char* kArgoNVGPUMMARegM8N8k4f16RowALayoutName =
        "nv_mma_reg_m8n8k4f16_row_a";
    constexpr const char* kArgoNVGPUMMARegM8N8k4f16ColBLayoutName =
        "nv_mma_reg_m8n8k4f16_col_b";
    constexpr const char* kArgoNVGPUMMARegM8N8k4f16f32CLayoutName =
        "nv_mma_reg_m8n8k4f16_f32_c";

    // Shared memory iteration map
    constexpr const char* kArgoNVGPUSharedRowLayoutName = "nv_shared_row";
    constexpr const char* kArgoNVGPUSharedColLayoutName = "nv_shared_col";

    // Private memory iteration map
    constexpr const char* kArgoNVGPUPrivateRowLayoutName = "nv_private_row";

    // MMA shared memory layout map
    constexpr const char* kArgoNVGPUMMAShared4x64sxf16ColBLayoutName =
        "nv_mma_shared_4x64sxf16_col_b";
    constexpr const char* kArgoNVGPUMMAShared64x32sxf16ColBLayoutName =
        "nv_mma_shared_64x32sxf16_col_b";
    constexpr const char* kArgoNVGPUMMAShared4x64sxf16RowBLayoutName =
        "nv_mma_shared_4x64sxf16_row_b";
    constexpr const char* kArgoNVGPUMMAShared64sx4xf16RowALayoutName =
        "nv_mma_shared_64sx4xf16_row_a";
    constexpr const char* kArgoNVGPUMMAShared32sx64xf16RowALayoutName =
        "nv_mma_shared_32sx64xf16_row_a";
    constexpr const char* kArgoNVGPUMMAShared64sx4xf16ColALayoutName =
        "nv_mma_shared_64sx4xf16_col_a";
    constexpr const char* kArgoNVGPUMMAShared32sx32xf16RowALayoutName =
        "nv_mma_shared_32sx32xf16_row_a";
    constexpr const char* kArgoNVGPUMMAShared32x32sxf16ColBLayoutName =
        "nv_mma_shared_32x32sxf16_col_b";

    // layout map for register tiling
    constexpr const char* kArgoNVGPURegTiling4x4x8LayoutName =
        "nv_reg_tiling_4x4x8";

    constexpr const char* kArgoNVGPURegTilingReduceRowLayoutName =
        "nv_reg_tiling_reduce_row";

    constexpr const char* kArgoNVGPURegTilingReduceColLayoutName =
        "nv_reg_tiling_reduce_col";

    constexpr const char* kArgoNVGPURegTilingTestDiv32Div32Mod8LayoutName =
        "test_div32div32mod8";

    void registerNVGPUDictionaries();
} // namespace cpp_printer
} // namespace mlir

#endif // DIALECT_ARGO_NVGPU_H_
