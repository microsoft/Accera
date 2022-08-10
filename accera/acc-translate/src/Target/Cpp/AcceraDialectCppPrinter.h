////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef ARGO_DIALECT_CPP_PRINTER_H_
#define ARGO_DIALECT_CPP_PRINTER_H_

// #include "CppPrinter.h"
// #include "mlir/Dialect/Argo/IR/ArgoOps.h"

#include <ir/include/argo/ArgoOps.h>
#include <ir/include/value/ValueDialect.h>

#include "CppPrinter.h"

namespace mlir
{
namespace cpp_printer
{

    struct AcceraDialectCppPrinter : public DialectCppPrinter
    {
        enum class MMAKernelKind
        {
            m8n8k4RowColfp32,
            InvalidKernel
        };

        AcceraDialectCppPrinter(CppPrinter* printer) :
            DialectCppPrinter(printer) {}

        std::string getName() override { return "Accera"; }

        LogicalResult printOp(accera::ir::value::MMAFillSyncOp op);
        LogicalResult printOp(accera::ir::value::MMALoadSyncOp op);
        LogicalResult printOp(accera::ir::value::MMAComputeSyncOp op);
        LogicalResult printOp(accera::ir::value::MMAStoreSyncOp op);
        LogicalResult printOp(accera::ir::value::CallOp op);
        LogicalResult printOp(accera::ir::value::ReturnOp op);

        LogicalResult printDialectOperation(Operation* op, bool* skipped, bool* consumed) override;

        LogicalResult printIntrinsicCallOp(Operation* callOp, Operation* defFuncOp, bool* consumed) override;

        /// print out host function that launches the kernel
        LogicalResult printHostLaunchFunc();

        LogicalResult printPrologue() override;

        LogicalResult printEpilogue() override;

        LogicalResult runPrePrintingPasses(Operation* op) override;

        llvm::SmallVector<StringRef, 0> MMAKernelNames;

        llvm::SmallVector<FuncOp, 1> CudaKernels;
    };

} // namespace cpp_printer
} // namespace mlir

#endif
