////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
//  Authors: Abdul Dakkak, Kern Handa
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef GPU_DIALECT_CPP_PRINTER_H_
#define GPU_DIALECT_CPP_PRINTER_H_

#include <llvm/ADT/SmallVector.h>
#include <mlir/Dialect/GPU/GPUDialect.h>
#include <mlir/Support/LogicalResult.h>

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

        LogicalResult runPrePrintingPasses(Operation*) override;

        LogicalResult printHeaderFiles() override;

        LogicalResult printDeclarations() override;

        /// print Operation from GPU Dialect
        LogicalResult printDialectOperation(Operation* op, bool* skipped, bool* consumed) override;

        LogicalResult printVectorTypeArrayDecl(VectorType vecType,
                                               StringRef vecVar) override;

        /// print the function delcaration for the given GPUFuncOp.
        /// A trailing semicolon will be generated if trailingSemiColon is true.
        LogicalResult printFunctionDeclaration(gpu::GPUFuncOp funcOp, bool trailingSemiColon);

        LogicalResult printOp(gpu::BarrierOp);
        LogicalResult printOp(gpu::BlockDimOp);
        LogicalResult printOp(gpu::BlockIdOp);
        LogicalResult printOp(gpu::GPUFuncOp);
        LogicalResult printOp(gpu::GPUModuleOp);
        LogicalResult printOp(gpu::GridDimOp);
        LogicalResult printOp(gpu::LaunchFuncOp);
        LogicalResult printOp(gpu::ModuleEndOp);
        LogicalResult printOp(gpu::ReturnOp);
        LogicalResult printOp(gpu::ThreadIdOp);
        LogicalResult printOp(gpu::SubgroupMmaConstantMatrixOp);
        LogicalResult printOp(gpu::SubgroupMmaLoadMatrixOp);
        LogicalResult printOp(gpu::SubgroupMmaComputeOp);
        LogicalResult printOp(gpu::SubgroupMmaStoreMatrixOp);

        LogicalResult printGpuFPVectorType(VectorType vecType, StringRef vecVar);

        LogicalResult printGPUIndexType();

    private:
        std::string getWmmaNamespace();
        std::string getFragmentEnum(const gpu::MMAMatrixType& mmaMatrix);
        LogicalResult printFragmentType(const gpu::MMAMatrixType& mmaMatrix, int m, int n, int k, bool row_major);
        LogicalResult printAccType(const gpu::MMAMatrixType& mmaMatrix);

        llvm::SmallVector<gpu::GPUModuleOp> _gpuModuleOps;
    };

} // namespace cpp_printer
} // namespace mlir

#endif
