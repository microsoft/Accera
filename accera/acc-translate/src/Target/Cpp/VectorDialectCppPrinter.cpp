////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
//  Authors: Abdul Dakkak
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "VectorDialectCppPrinter.h"
#include "AffineDialectCppPrinter.h"
#include <mlir/Dialect/Vector/IR/VectorOps.h>

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
        auto result = op.getResult();
        auto idx = state.nameState.getOrCreateName(
            result, SSANameState::SSANameKind::Variable);

        os << state.nameState.getName(op.dest()) << "[";
        os << state.nameState.getName(op.position()) << "]";
        os << " = ";
        os << state.nameState.getName(op.source());
        os << ";\n";

        RETURN_IF_FAILED(printer->printType(result.getType()));
        os << " " << idx;
        os << " = ";
        os << state.nameState.getName(op.dest());

        return success();
    }

    LogicalResult VectorDialectCppPrinter::printLoadOp(vector::LoadOp op)
    {
        return printer->printMemRefLoadOrStore(true, op.base(), op.getMemRefType(), op.indices(), op.getResult());
    }

    LogicalResult VectorDialectCppPrinter::printStoreOp(vector::StoreOp op)
    {
        return printer->printMemRefLoadOrStore(false, op.base(), op.getMemRefType(), op.indices(), op.valueToStore());
    }
    
    LogicalResult VectorDialectCppPrinter::printBroadcastOp(vector::BroadcastOp op)
    {
        auto vecTy = op.getVectorType();
        if (vecTy.getRank() != 1) {
            os << "[[ only rank 1 vector is supported ]]";
            return failure();
        }

        auto vec = op.vector();
        auto source = op.source();

        
        auto idx = state.nameState.getOrCreateName(
            vec, SSANameState::SSANameKind::Variable);

        RETURN_IF_FAILED(printer->printType(vecTy));
        os << " " << idx;
        os << "{";
        for (int i = 0; i < vecTy.getNumElements(); i++) {
            os << " ";
            os << state.nameState.getName(source);
            if (i != vecTy.getNumElements() - 1) {
                os << ",";
            }
        }
        os << "}";

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
        if (auto loadOp = dyn_cast<mlir::vector::LoadOp>(op))
            return printLoadOp(loadOp);
        if (auto storeOp = dyn_cast<mlir::vector::StoreOp>(op))
            return printStoreOp(storeOp);
        if (auto broadcastOp = dyn_cast<mlir::vector::BroadcastOp>(op))
            return printBroadcastOp(broadcastOp);


        *consumed = false;
        return success();
    }

} // namespace cpp_printer
} // namespace mlir