////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
//  Authors: Abdul Dakkak
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "AcceraPasses.h"

#include "util/RangeValueUtilities.h"

#include <ir/include/IRUtil.h>

#include <llvm/IR/GlobalValue.h>
#include <mlir/Analysis/DataFlowAnalysis.h>
#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/Linalg/IR/LinalgOps.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/SCF.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/Value.h>
#include <mlir/Interfaces/ControlFlowInterfaces.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Transforms/FoldUtils.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>
#include <mlir/Transforms/Passes.h>

#include <llvm/ADT/APInt.h>
#include <llvm/ADT/TypeSwitch.h>
#include <llvm/IR/ConstantRange.h>
#include <llvm/Support/Debug.h>
#include <llvm/Support/raw_os_ostream.h>

#include <algorithm>

#define DEBUG_TYPE "value-optimize"

using namespace mlir;

using namespace accera::ir;
using namespace accera::transforms;
using namespace accera::ir::value;
using namespace accera::ir::util;

using llvm::CmpInst;
using llvm::ConstantRange;
using llvm::Instruction;

namespace
{
struct RangeValueOptimizePass : public ConvertRangeValueOptimizeBase<RangeValueOptimizePass>
{
    void runOnOperation() final
    {
        rangeValue = &getAnalysis<RangeValueAnalysis>();

        // now we use them to classify the comparison operation
        auto ctx = &getContext();
        OpBuilder builder(ctx);
        Type i1Ty = builder.getI1Type();
        getOperation()->walk([&](CmpIOp op) {
            auto classification = classifyCmpIOp(op);
            if (classification != CmpIOpClassification::Unknown)
            {
                builder.setInsertionPoint(op);
                Value val = builder.create<ConstantOp>(op->getLoc(), i1Ty, builder.getBoolAttr(classification == CmpIOpClassification::AlwaysTrue));
                op.replaceAllUsesWith(val);
                op.erase();
            }
        });
    }

private:
    enum CmpIOpClassification : int
    {
        Unknown,
        AlwaysFalse,
        AlwaysTrue
    };

    CmpIOpClassification classifyCmpIOp(CmpIOp op)
    {
        auto predicate = op.getPredicate();
        auto lhs = op.lhs();
        auto rhs = op.rhs();
        if (!rangeValue->hasRange(lhs) || !rangeValue->hasRange(rhs))
        {
            return CmpIOpClassification::Unknown;
        }
        auto lhsRange = rangeValue->getRange(lhs);
        auto rhsRange = rangeValue->getRange(rhs);
        if (lhsRange.isFullSet() || rhsRange.isFullSet())
        {
            return CmpIOpClassification::Unknown;
        }

        switch (predicate)
        {
        case CmpIPredicate::slt:
            if (lhsRange.icmp(CmpInst::Predicate::ICMP_SLT, rhsRange))
            {
                return CmpIOpClassification::AlwaysTrue;
            }
            else if (lhsRange.icmp(CmpInst::Predicate::ICMP_SGE, rhsRange))
            {
                return CmpIOpClassification::AlwaysFalse;
            }
            break;
        case CmpIPredicate::sle:
            if (lhsRange.icmp(CmpInst::Predicate::ICMP_SLE, rhsRange))
            {
                return CmpIOpClassification::AlwaysTrue;
            }
            else if (lhsRange.icmp(CmpInst::Predicate::ICMP_SGT, rhsRange))
            {
                return CmpIOpClassification::AlwaysFalse;
            }
            break;
        case CmpIPredicate::sgt:
            if (lhsRange.icmp(CmpInst::Predicate::ICMP_SGT, rhsRange))
            {
                return CmpIOpClassification::AlwaysTrue;
            }
            else if (lhsRange.icmp(CmpInst::Predicate::ICMP_SLE, rhsRange))
            {
                return CmpIOpClassification::AlwaysFalse;
            }
            break;
        case CmpIPredicate::sge:
            if (lhsRange.icmp(CmpInst::Predicate::ICMP_SGE, rhsRange))
            {
                return CmpIOpClassification::AlwaysTrue;
            }
            else if (lhsRange.icmp(CmpInst::Predicate::ICMP_SLT, rhsRange))
            {
                return CmpIOpClassification::AlwaysFalse;
            }
            break;
        default:
            break;
        }

        return CmpIOpClassification::Unknown;
    }
    RangeValueAnalysis* rangeValue = nullptr;
};

} // namespace

namespace accera::transforms::value
{

std::unique_ptr<mlir::Pass> createRangeValueOptimizePass()
{
    return std::make_unique<RangeValueOptimizePass>();
}

} // namespace accera::transforms::value
