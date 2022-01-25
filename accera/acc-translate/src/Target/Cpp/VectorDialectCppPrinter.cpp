////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
//  Authors: Abdul Dakkak
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "VectorDialectCppPrinter.h"

using namespace mlir;

namespace mlir
{
namespace cpp_printer
{

    LogicalResult VectorDialectCppPrinter::printExtractElementOp(vector::ExtractElementOp op)
    {
        auto result = op.getResult();
        auto idx = state.nameState.getOrCreateName(
            result, SSANameState::SSANameKind::Variable);

        RETURN_IF_FAILED(printer->printType(result.getType()));
        os << " " << idx;
        os << " = ";
        os << state.nameState.getName(op.vector()) << "[";
        os << state.nameState.getName(op.position()) << "]";
        return success();
    }

    LogicalResult VectorDialectCppPrinter::printInsertElementOp(vector::InsertElementOp op)
    {
        os << state.nameState.getName(op.dest()) << "[";
        os << state.nameState.getName(op.position()) << "]";
        os << " = ";
        os << state.nameState.getName(op.source());
        return success();
    }

    LogicalResult VectorDialectCppPrinter::printDialectOperation(Operation* op,
                                                                 bool* /*skipped*/,
                                                                 bool* consumed)
    {
        *consumed = true;

        if (auto extractElementOp = dyn_cast<mlir::vector::ExtractElementOp>(op))
            return printExtractElementOp(extractElementOp);
        if (auto insertElementOp = dyn_cast<mlir::vector::InsertElementOp>(op))
            return printInsertElementOp(insertElementOp);

        *consumed = false;
        return success();
    }

} // namespace cpp_printer
} // namespace mlir