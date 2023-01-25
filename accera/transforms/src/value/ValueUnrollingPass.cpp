////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "AcceraPasses.h"

#include <ir/include/IRUtil.h>
#include <ir/include/value/ValueDialect.h>
#include <ir/include/nest/LoopNestAttributes.h>

#include <mlir/IR/Location.h>
#include <mlir/Dialect/Affine/Analysis/LoopAnalysis.h>
#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Dialect/Affine/LoopUtils.h>

#include <memory>

using namespace mlir;

namespace
{

struct ValueUnrollingPass : public accera::transforms::ValueUnrollingBase<ValueUnrollingPass>
{
    void runOnOperation() final
    {
        auto module = getOperation();

        module.walk([&](AffineForOp op) {
            if (op->getAttrOfType<UnitAttr>("accv_unrolled"))
            {
                auto tripCount = mlir::getConstantTripCount(op);
                if (tripCount && *tripCount >= 1)
                    (void)mlir::loopUnrollFull(op);
            }
            else if (auto jammed = op->getAttrOfType<IntegerAttr>("accv_unroll_jam"))
            {
                (void)mlir::loopUnrollJamByFactor(op, (uint64_t)jammed.getInt());
            }
            else
            {
                (void)mlir::promoteIfSingleIteration(op);
            }
        });
    }
};

} // namespace

namespace accera::transforms::value
{
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createValueUnrollingPass()
{
    return std::make_unique<ValueUnrollingPass>();
}
} // namespace accera::transforms::value
