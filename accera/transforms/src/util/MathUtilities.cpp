////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "util/MathUtilities.h"

#include <mlir/Dialect/Arithmetic/IR/Arithmetic.h>
#include <mlir/Dialect/Vector/IR/VectorOps.h>

namespace accera::transforms
{
mlir::Value SaturateValue(mlir::PatternRewriter& rewriter, mlir::Value value, int64_t bitWidth, bool isSigned)
{
    auto loc = value.getLoc();

    int64_t minVal = isSigned ? -(1 << (bitWidth - 1)) : 0;
    int64_t maxVal = isSigned ? (1 << (bitWidth - 1)) - 1 : (1 << bitWidth) - 1;
    auto resultType = value.getType();
    auto resultBits = resultType.getIntOrFloatBitWidth();

    mlir::Value minConst = rewriter.create<mlir::arith::ConstantIntOp>(loc, minVal, resultBits);
    mlir::Value maxConst = rewriter.create<mlir::arith::ConstantIntOp>(loc, maxVal, resultBits);

    if (auto vectorType = resultType.dyn_cast<mlir::VectorType>())
    {
        minConst = rewriter.create<mlir::vector::SplatOp>(loc, minConst, vectorType);
        maxConst = rewriter.create<mlir::vector::SplatOp>(loc, maxConst, vectorType);
    }
    auto maxCmp = rewriter.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::sgt, value, minConst);
    auto temp = rewriter.create<mlir::arith::SelectOp>(loc, maxCmp, value, minConst);
    auto minCmp = rewriter.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::slt, temp, maxConst);
    auto result = rewriter.create<mlir::arith::SelectOp>(loc, minCmp, temp, maxConst);

    return result;
}
} // namespace accera::transforms
