////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
//  Authors: Abdul Dakkak, Mason Remy
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "AcceraPasses.h"

#include "util/RangeValueUtilities.h"

#include <ir/include/IRUtil.h>

#include <llvm/IR/GlobalValue.h>
#include <mlir/Analysis/DataFlowAnalysis.h>
#include <mlir/Dialect/Affine/Analysis/LoopAnalysis.h>
#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/Affine/LoopUtils.h>
#include <mlir/Dialect/Arithmetic/IR/Arithmetic.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/Linalg/IR/Linalg.h>
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
#include <optional>

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

enum class CmpIOpClassification : int
{
    Unknown,
    AlwaysFalse,
    AlwaysTrue
};

// TODO : de-dupe with value-to-std
static arith::CmpIPredicate CmpOpPredicateToCmpIPredicate(accera::ir::value::CmpOpPredicate pred)
{
#define MAP_PREDICATE(v1, v2)                   \
    case accera::ir::value::CmpOpPredicate::v1: \
        return arith::CmpIPredicate::v2

    switch (pred)
    {
        MAP_PREDICATE(EQ, eq);
        MAP_PREDICATE(GE, sge);
        MAP_PREDICATE(GT, sgt);
        MAP_PREDICATE(LE, sle);
        MAP_PREDICATE(LT, slt);
        MAP_PREDICATE(NE, ne);
    default:
        assert(false);
    }

#undef MAP_PREDICATE
}

CmpIOpClassification classifyCmpIOp(RangeValueAnalysis& rangeValue, arith::CmpIOp op)
{
    auto predicate = op.getPredicate();
    auto lhs = op.getLhs();
    auto rhs = op.getRhs();
    if (!rangeValue.hasRange(lhs) || !rangeValue.hasRange(rhs))
    {
        return CmpIOpClassification::Unknown;
    }
    auto lhsRange = rangeValue.getRange(lhs);
    auto rhsRange = rangeValue.getRange(rhs);
    if (lhsRange.isFullSet() || rhsRange.isFullSet())
    {
        return CmpIOpClassification::Unknown;
    }

    switch (predicate)
    {
    case arith::CmpIPredicate::slt:
        if (lhsRange.icmp(CmpInst::Predicate::ICMP_SLT, rhsRange))
        {
            return CmpIOpClassification::AlwaysTrue;
        }
        else if (lhsRange.icmp(CmpInst::Predicate::ICMP_SGE, rhsRange))
        {
            return CmpIOpClassification::AlwaysFalse;
        }
        break;
    case arith::CmpIPredicate::sle:
        if (lhsRange.icmp(CmpInst::Predicate::ICMP_SLE, rhsRange))
        {
            return CmpIOpClassification::AlwaysTrue;
        }
        else if (lhsRange.icmp(CmpInst::Predicate::ICMP_SGT, rhsRange))
        {
            return CmpIOpClassification::AlwaysFalse;
        }
        break;
    case arith::CmpIPredicate::sgt:
        if (lhsRange.icmp(CmpInst::Predicate::ICMP_SGT, rhsRange))
        {
            return CmpIOpClassification::AlwaysTrue;
        }
        else if (lhsRange.icmp(CmpInst::Predicate::ICMP_SLE, rhsRange))
        {
            return CmpIOpClassification::AlwaysFalse;
        }
        break;
    case arith::CmpIPredicate::sge:
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

std::optional<bool> GetConstantCmpIOpResult(arith::CmpIOp cmpIOp)
{
    RangeValueAnalysis rangeValueAnalysis(cmpIOp);
    auto classification = classifyCmpIOp(rangeValueAnalysis, cmpIOp);
    if (classification != CmpIOpClassification::Unknown)
    {
        return classification == CmpIOpClassification::AlwaysTrue;
    }
    return std::nullopt;
}

LogicalResult RewriteConstantCmpIOpCommon(PatternRewriter& rewriter, arith::CmpIOp cmpIOp, mlir::Operation* opToReplace = nullptr)
{
    if (!opToReplace)
    {
        opToReplace = cmpIOp;
    }

    auto constantCmpIOpResultOpt = GetConstantCmpIOpResult(cmpIOp);

    if (constantCmpIOpResultOpt.has_value())
    {
        Type i1Ty = rewriter.getI1Type();
        rewriter.replaceOpWithNewOp<arith::ConstantOp>(opToReplace, i1Ty, rewriter.getBoolAttr(*constantCmpIOpResultOpt));
        return mlir::success();
    }
    return mlir::failure();
}

struct ConstantCmpIOpRewrite : public mlir::OpRewritePattern<arith::CmpIOp>
{
    using OpRewritePattern::OpRewritePattern;
    LogicalResult matchAndRewrite(arith::CmpIOp op, PatternRewriter& rewriter) const final
    {
        return RewriteConstantCmpIOpCommon(rewriter, op);
    }
};

struct ConstantAcceraCmpOpRewrite : public mlir::OpRewritePattern<accera::ir::value::CmpOp>
{
    using OpRewritePattern::OpRewritePattern;
    LogicalResult matchAndRewrite(accera::ir::value::CmpOp op, PatternRewriter& rewriter) const final
    {
        std::stack<mlir::Operation*> tempOps;
        TempOpCleanupGuard guard(&tempOps, rewriter);

        // TODO : de-dupe with value-to-std conversion
        auto lhs = op.lhs();
        auto rhs = op.rhs();

        auto pred = op.getPredicate();
        if (util::GetElementType(lhs.getType()).isa<FloatType>())
        {
            // Doesn't support CmpFOp classification currently
            return failure();
        }
        auto stdCmpIOp = rewriter.create<arith::CmpIOp>(op.getLoc(), CmpOpPredicateToCmpIPredicate(pred), lhs, rhs);
        tempOps.push(stdCmpIOp.getOperation());

        return RewriteConstantCmpIOpCommon(rewriter, stdCmpIOp, op);
    }
};

struct ConstantAcceraMaxMinOpRewrite : public mlir::OpRewritePattern<BinOp>
{
    using OpRewritePattern::OpRewritePattern;
    LogicalResult matchAndRewrite(BinOp op, PatternRewriter& rewriter) const final
    {
        // If the Bin op is a max or a min, then check if it is always equal to one of its operands
        // i.e. if we have z = max(x, y), and x <= y always, then replace max(x, y) with y
        // To do this, check:
        //      (x <= y), and
        //      (x >= y)
        // If the former is always true, then replace max(x, y) with y, min(x, y) with x
        // If the latter is always true, then replace max(x, y) with x, min(x, y) with y
        // If neither are always true, then don't replace the max or min op
        // We have to check both to handle the case where a '<' or '>' check doesn't capture that the point where they are equal doesn't change which operand is the replacement value of the max/min and to avoid an operand ordering bias

        auto predicate = op.getPredicate();
        if (predicate != BinaryOpPredicate::MAX && predicate != BinaryOpPredicate::MIN)
        {
            return failure();
        }
        std::stack<mlir::Operation*> tempOps;
        TempOpCleanupGuard guard(&tempOps, rewriter);

        auto lhs = op.lhs();
        auto rhs = op.rhs();

        if (util::GetElementType(lhs.getType()).isa<FloatType>())
        {
            // Doesn't support CmpFOp classification currently
            return failure();
        }
        auto LEQCmpIOp = rewriter.create<arith::CmpIOp>(op.getLoc(), arith::CmpIPredicate::sle, lhs, rhs);
        tempOps.push(LEQCmpIOp.getOperation());
        auto LEQconstantResultOpt = GetConstantCmpIOpResult(LEQCmpIOp);

        if (LEQconstantResultOpt.has_value() && *LEQconstantResultOpt)
        {
            if (predicate == BinaryOpPredicate::MAX)
            {
                rewriter.replaceOp(op, mlir::ValueRange{ rhs });
            }
            else
            {
                rewriter.replaceOp(op, mlir::ValueRange{ lhs });
            }
            return success();
        }

        auto GEQCmpIOp = rewriter.create<arith::CmpIOp>(op.getLoc(), arith::CmpIPredicate::sge, lhs, rhs);
        tempOps.push(GEQCmpIOp.getOperation());
        auto GEQconstantResultOpt = GetConstantCmpIOpResult(GEQCmpIOp);

        if (GEQconstantResultOpt.has_value() && *GEQconstantResultOpt)
        {
            if (predicate == BinaryOpPredicate::MAX)
            {
                rewriter.replaceOp(op, mlir::ValueRange{ lhs });
            }
            else
            {
                rewriter.replaceOp(op, mlir::ValueRange{ rhs });
            }
            return success();
        }
        return failure();
    }
};


struct RangeValueOptimizePass : public ConvertRangeValueOptimizeBase<RangeValueOptimizePass>
{
    void runOnOperation() final
    {
        auto context = &getContext();
        auto operation = getOperation();

        mlir::GreedyRewriteConfig topDownConfig; // Handle outer simplifications first as they will resolve to constants need for inner simplifications
        topDownConfig.useTopDownTraversal = true;

        mlir::RewritePatternSet patterns(context);
        accera::transforms::value::populateRangeValueOptimizePatterns(patterns);
        util::FillCanonicalPatternsRecursively(operation, patterns);
        (void)applyPatternsAndFoldGreedily(operation, std::move(patterns), topDownConfig);
    }
};

} // namespace

namespace accera::transforms::value
{

void populateRangeValueOptimizePatterns(mlir::RewritePatternSet& patterns)
{
    patterns.insert<ConstantCmpIOpRewrite,
                    ConstantAcceraCmpOpRewrite,
                    ConstantAcceraMaxMinOpRewrite>(patterns.getContext());
}

std::unique_ptr<mlir::Pass> createRangeValueOptimizePass()
{
    return std::make_unique<RangeValueOptimizePass>();
}

} // namespace accera::transforms::value
