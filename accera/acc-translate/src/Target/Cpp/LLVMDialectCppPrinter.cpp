////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
//  Authors: Abdul Dakkak
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "LLVMDialectCppPrinter.h"
#include <mlir/Support/LogicalResult.h>

using namespace mlir;

namespace mlir
{
namespace cpp_printer
{
    LogicalResult LLVMDialectCppPrinter::printFenceOp(LLVM::FenceOp op)
    {
        if (op.getSyncscope() == "agent" && op.getOrdering() == LLVM::AtomicOrdering::seq_cst)
        {
            os << "__threadfence()";
            return success();
        }
        return failure();
    }

    LogicalResult LLVMDialectCppPrinter::printDialectOperation(Operation* op,
                                                               bool* /*skipped*/,
                                                               bool* consumed)
    {
        *consumed = true;

        if (auto fenceOp = dyn_cast<LLVM::FenceOp>(op))
            return printFenceOp(fenceOp);

        *consumed = false;
        return success();
    }

} // namespace cpp_printer
} // namespace mlir