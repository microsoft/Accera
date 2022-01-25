////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
//  Authors: Abdul Dakkak
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef GPU_DIALECT_CPP_PRINTER_H_
#define GPU_DIALECT_CPP_PRINTER_H_

#include <mlir/Dialect/GPU/GPUDialect.h>

#include "CppPrinter.h"

namespace mlir
{
namespace cpp_printer
{

    struct GpuDialectCppPrinter : public DialectCppPrinter
    {
        GpuDialectCppPrinter(CppPrinter* printer_) :
            DialectCppPrinter(printer_) {}

        std::string getName() override { return "GPU"; }

        /// print Operation from GPU Dialect
        LogicalResult printDialectOperation(Operation* op, bool* skipped, bool* consumed) override;

        LogicalResult printBarrierOp(gpu::BarrierOp barrierOp);

        LogicalResult printGridDimOp(gpu::GridDimOp gridDimOp);

        LogicalResult printBlockDimOp(gpu::BlockDimOp blockDimOp);

        LogicalResult printBlockIdOp(gpu::BlockIdOp bidOp);

        LogicalResult printThreadIdOp(gpu::ThreadIdOp tidOp);

        LogicalResult printVectorTypeArrayDecl(VectorType vecType,
                                               StringRef vecVar) override;

        LogicalResult printGpuFPVectorType(VectorType vecType, StringRef vecVar);
    };

} // namespace cpp_printer
} // namespace mlir

#endif
