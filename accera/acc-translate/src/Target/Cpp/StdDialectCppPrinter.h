////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
//  Authors: Abdul Dakkak, Kern Handa
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef STD_DIALECT_CPP_PRINTER_H_
#define STD_DIALECT_CPP_PRINTER_H_

#include <mlir/Dialect/Math/IR/Math.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>

#include "CppPrinter.h"

namespace mlir
{
namespace cpp_printer
{

    struct StdDialectCppPrinter : public DialectCppPrinter
    {
        StdDialectCppPrinter(CppPrinter* printer_) :
            DialectCppPrinter(printer_) {}

        std::string getName() override { return "Std"; }

        LogicalResult printHeaderFiles() override;

        LogicalResult printPrologue() override;

        /// print Operation from StandardOps Dialect
        LogicalResult printDialectOperation(Operation* op, bool* skipped, bool* consumed) override;

        /// print binary ops such as '+', '-', '*', etc
        LogicalResult printBinaryOp(Operation* binOp);

        /// print a CastOp where the dst type is an Integer whose signed-ness
        /// is determined by the argument isSigned
        LogicalResult printCastToIntegerOp(Operation* op, bool isSigned);

        /// adds an alias in the alias table
        LogicalResult printIndexCastOp(arith::IndexCastOp op);

        /// print a ``simple'' CastOp such as arith::IndexCastOp and arith::TruncIOp
        /// that can be converted into a cast expression without worrying
        /// about the signed-ness of the operands
        LogicalResult printSimpleCastOp(Operation* op);

        /// print AllocOp
        LogicalResult printAllocOp(memref::AllocOp allocOp);

        /// print AllocaOp
        LogicalResult printAllocaOp(memref::AllocaOp allocaOp);

        /// print CallOp
        LogicalResult printCallOp(CallOp constOp);

        /// print arith::ConstantOp
        LogicalResult printConstantOp(arith::ConstantOp constOp);

        /// print DeallocOp
        LogicalResult printDeallocOp(memref::DeallocOp deallocOp, bool* skipped);

        /// print DimOp
        LogicalResult printDimOp(memref::DimOp dimOp);

        /// print ExpOp
        LogicalResult printExpOp(math::ExpOp expOp);

        /// print LoadOp
        LogicalResult printLoadOp(memref::LoadOp loadOp);

        /// print MemRefCastOp
        LogicalResult printMemRefCastOp(memref::CastOp memRefCastOp);

        LogicalResult printMemRefTransposeOp(memref::TransposeOp memRefCastOp);

        /// print ReturnOp
        LogicalResult printReturnOp(ReturnOp returnOp);

        /// print SelectOp as ternary operator
        LogicalResult printSelectOp(SelectOp selectOp);

        /// print GetGlobalOp as a call to the global variable
        LogicalResult printGetGlobalOp(memref::GetGlobalOp getGlobalOp);

        /// print StoreOp
        LogicalResult printStoreOp(memref::StoreOp storeOp);

        /// print ReinterpretCastOp
        LogicalResult printReinterpretCastOp(memref::ReinterpretCastOp reinterpretCastop);

        LogicalResult printMaxFOp(arith::MaxFOp maxfOp);
    };

} // namespace cpp_printer
} // namespace mlir

#endif
