////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "nest/LoopNestPasses.h"
#include "util/MathUtilities.h"

#include <ir/include/IRUtil.h>
#include <ir/include/exec/ExecutionPlanOps.h>
#include <ir/include/nest/LoopIndexInfo.h>
#include <ir/include/nest/LoopNestBuilder.h>
#include <ir/include/nest/LoopNestOps.h>
#include <ir/include/nest/Util.h>
#include <ir/include/value/ValueDialect.h>

#include <utilities/include/Exception.h>

#include <llvm/Support/raw_os_ostream.h>

#include <mlir/Analysis/LoopAnalysis.h>
#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/GPU/GPUDialect.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/IR/AffineMap.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/BlockAndValueMapping.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Transforms/DialectConversion.h>
#include <mlir/Transforms/InliningUtils.h>

#include <iostream>
#include <optional>
#include <stack>

using namespace accera::ir;
using namespace accera::ir::loopnest;
using namespace accera::ir::value;
using namespace accera::transforms;
using namespace mlir;

namespace lnir = accera::ir::loopnest;

//
// Rewrite patterns defined in C++
//

namespace
{

// Simple helper function that returns a string as printed from a op.
template <typename T>
static std::string debugString(T& op)
{
    std::string instr_str;
    llvm::raw_string_ostream os(instr_str);
    op.print(os);
    return os.str();
}

// `predicate` is a function of the form: `bool(Operation*)`
template <typename FunctionType>
Operation* GetParent(Operation* op, FunctionType&& predicate)
{
    while ((op = op->getParentOp()))
        if (predicate(op))
            return op;
    return nullptr;
}

// Taken from implementation of replaceAllUsesIf
bool ReplaceAllUsesWith(mlir::Value valueToReplace, function_ref<mlir::Value(mlir::OpOperand&)> getReplacement)
{
    bool result = true;
    for (mlir::OpOperand& use : llvm::make_early_inc_range(valueToReplace.getUses()))
    {
        if (auto newValue = getReplacement(use))
        {
            use.set(newValue);
        }
        else
        {
            result = false;
        }
    }
    return result;
}

mlir::Value FindIndexVariable(Index index, Operation* where)
{
    mlir::Operation* op = where;

    // TODO: need to get the domain somehow and check if the index is a computed index
    // (or somehow add compound indices to the outer loop where they're fully-defined)
    // from here, search block operands for a match
    while (op)
    {
        if (auto attr = op->getAttrOfType<IndexAttr>("index"))
        {
            if (attr.getValue() == index)
            {
                if (op->getNumRegions() > 0)
                {
                    // TODO: deal with >1 block argument
                    auto& opBlock = op->getRegion(0).front();
                    if (opBlock.getNumArguments() == 1)
                    {
                        return opBlock.getArgument(0);
                    }
                }
            }
        }

        // Go up a level
        auto block = op->getBlock();
        op = block ? block->getParentOp() : nullptr;
    }

    return {};
}

struct ScheduleOpDomainResolution : public OpRewritePattern<ScheduleOp>
{
    using OpRewritePattern::OpRewritePattern;
    LogicalResult matchAndRewrite(ScheduleOp op, PatternRewriter& rewriter) const final;
};

struct ScheduleOpConversion : public OpRewritePattern<ScheduleOp>
{
    using OpRewritePattern::OpRewritePattern;
    ScheduleOpConversion(MLIRContext* context, bool printLoops = false) :
        OpRewritePattern(context, /* benefit */ 1),
        printLoops(printLoops)
    {}
    LogicalResult matchAndRewrite(ScheduleOp op, PatternRewriter& rewriter) const final;

    bool printLoops = false;
};

struct SaturatedAccumulateLoopRewrite : public OpRewritePattern<AffineForOp>
{
    using OpRewritePattern<AffineForOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(AffineForOp op, PatternRewriter& rewriter) const final;
};

struct ScheduledLoopOpRewrite : public OpRewritePattern<ScheduledLoopOp>
{
    using OpRewritePattern<ScheduledLoopOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(ScheduledLoopOp op, PatternRewriter& rewriter) const final;
};

struct GPUMappedAffineForOpRewrite : public OpRewritePattern<mlir::AffineForOp>
{
    using OpRewritePattern<mlir::AffineForOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(mlir::AffineForOp affineForOp, PatternRewriter& rewriter) const final;
};

struct ScheduledLoopOpIndexConversion : public OpRewritePattern<ScheduledLoopOp>
{
    using OpRewritePattern::OpRewritePattern;

    LogicalResult matchAndRewrite(ScheduledLoopOp op, PatternRewriter& rewriter) const final;
};

struct DimSizeOpConversion : public OpRewritePattern<DimSizeOp>
{
    DimSizeOpConversion(MLIRContext* context) :
        OpRewritePattern(context, /* benefit */ 1) {}

    LogicalResult matchAndRewrite(DimSizeOp op, PatternRewriter& rewriter) const final;
};

struct ReplaceSymbolicIndexOpPattern : public OpRewritePattern<SymbolicIndexOp>
{
    ReplaceSymbolicIndexOpPattern(MLIRContext* context) :
        OpRewritePattern(context, /* benefit */ 1) {}

    LogicalResult matchAndRewrite(SymbolicIndexOp op, PatternRewriter& rewriter) const final;
};

struct UnlinkAndRemoveSymbolicIndexOpPattern : public OpRewritePattern<SymbolicIndexOp>
{
    UnlinkAndRemoveSymbolicIndexOpPattern(MLIRContext* context) :
        OpRewritePattern(context, /* benefit */ 1) {}

    LogicalResult matchAndRewrite(SymbolicIndexOp op, PatternRewriter& rewriter) const final;
};

template <typename OpType>
struct RemoveKernelLikeOpPattern : public OpRewritePattern<OpType>
{
    using OpRewritePattern<OpType>::OpRewritePattern;

    LogicalResult matchAndRewrite(OpType op, PatternRewriter& rewriter) const final
    {
        auto symTableOp = SymbolTable::getNearestSymbolTable(op);
        auto kernelName = op.getId();
        auto isUnused = SymbolTable::symbolKnownUseEmpty(kernelName, symTableOp);

        [[maybe_unused]] auto symUses = SymbolTable::getSymbolUses(kernelName, symTableOp);

        return success(isUnused);
    }
};

struct RemoveKernelOpPattern : public RemoveKernelLikeOpPattern<KernelOp>
{
    using RemoveKernelLikeOpPattern<KernelOp>::RemoveKernelLikeOpPattern;
};

struct RemoveScheduledKernelOpPattern : public RemoveKernelLikeOpPattern<ScheduledKernelOp>
{
    using RemoveKernelLikeOpPattern<ScheduledKernelOp>::RemoveKernelLikeOpPattern;
};

struct RemoveSymIndexOpPattern : public OpRewritePattern<SymbolicIndexOp>
{
    using OpRewritePattern::OpRewritePattern;

    LogicalResult matchAndRewrite(SymbolicIndexOp op, PatternRewriter& rewriter) const final
    {
        if (op.use_empty())
        {
            rewriter.eraseOp(op);
        }
        return success();
    }
};

struct MergeNearlyIdenticalSiblingLoops : public OpRewritePattern<AffineForOp>
{
    MergeNearlyIdenticalSiblingLoops(MLIRContext* context) :
        OpRewritePattern(context, /* benefit */ 10) {}

    void initialize()
    {
        setHasBoundedRewriteRecursion();
        setDebugName("MergeNearlyIdenticalSiblingLoops");
    }

    LogicalResult matchAffineForOps(AffineForOp op1, AffineForOp op2, bool topLevel = false) const;

    LogicalResult matchAndRewrite(AffineForOp op, PatternRewriter& rewriter) const final;
};

struct RemoveZeroIterationLoopPattern : public OpRewritePattern<AffineForOp>
{
    RemoveZeroIterationLoopPattern(MLIRContext* context) :
        OpRewritePattern(context, /* benefit */ 1) {}

    LogicalResult matchAndRewrite(AffineForOp op, PatternRewriter& rewriter) const final;
};

struct PromoteSingleIterationLoopPattern : public OpRewritePattern<AffineForOp>
{
    PromoteSingleIterationLoopPattern(MLIRContext* context) :
        OpRewritePattern(context, /* benefit */ 1) {}

    LogicalResult matchAndRewrite(AffineForOp op, PatternRewriter& rewriter) const final;
};

} // namespace

//
// ScheduleOpDomainResolution
//

void ResolveAffineRangeEnd(PatternRewriter& rewriter, mlir::AffineApplyOp affineApplyOp)
{
    // Recursively resolve DimSizeOps in the AffineApplyOp operands to constant index ops then canonicalize the AffineApplyOp
    for (auto operand : llvm::make_early_inc_range(affineApplyOp.mapOperands()))
    {
        auto op = operand.getDefiningOp();
        if (auto dimSizeOp = mlir::dyn_cast_or_null<DimSizeOp>(op))
        {
            auto dimensionIndex = dimSizeOp.dimensionIndex();
            auto dimSize = util::GetDimSizeAt(dimensionIndex.getValue(), op);
            assert(dimSize.has_value());
            rewriter.replaceOpWithNewOp<mlir::ConstantIndexOp>(op, *dimSize);
        }
        else if (auto innerAffineApply = mlir::dyn_cast_or_null<mlir::AffineApplyOp>(op))
        {
            // Nested affine apply, so recursively resolve
            ResolveAffineRangeEnd(rewriter, innerAffineApply);
        }
    }

    // TODO : once we move to AffineMaps for LoopNest lower/upper bounds this should be removed
    //        as the Affine simplifications can happen outside of this pass

    // We need to do AffineApplyOp folding directly here since we need to replace the AffineApplyOp
    // with the PatternRewriter from the current pass
    // This is currently necessary because nested LoopNests get resolved as part of a single pass
    // but if a nested LoopNest relies on DimSizeOps and affine maps of those ops, then we need
    // to resolve the sizes during the ScheduleOp lowering
    std::vector<mlir::Attribute> constantAttrs;
    for (auto operand : llvm::make_early_inc_range(affineApplyOp.mapOperands()))
    {
        auto op = operand.getDefiningOp();
        if (auto constantOp = mlir::dyn_cast_or_null<mlir::ConstantOp>(op))
        {
            constantAttrs.push_back(constantOp.getValue());
        }
        else
        {
            assert(false && "Operands must be lowered to constants by now");
        }
    }
    auto foldResult = affineApplyOp.fold(constantAttrs);
    auto foldAttr = foldResult.get<mlir::Attribute>();
    rewriter.replaceOpWithNewOp<mlir::ConstantOp>(affineApplyOp, foldAttr);
}

void ResolveRange(ScheduleOp& op, PatternRewriter& rewriter, lnir::Range& range)
{
    if (range.HasConstantEnd())
    {
        return;
    }
    if (range.HasIndexEnd())
    {
        auto dimensionIndex = range.EndIndex();
        auto dimSize = util::GetDimSizeAt(dimensionIndex, op.getOperation());
        assert(dimSize.has_value());
        range = lnir::Range{ range.Begin(), *dimSize, range.Increment() };
    }
    else if (range.HasOperandIndexEnd())
    {
        int64_t operandIndex = range.EndOperandIndex().GetIndex();
        mlir::Value endVal = op.getOperand(operandIndex);
        Operation* endValOp = endVal.getDefiningOp();

        if (auto affineApplyOp = mlir::dyn_cast_or_null<mlir::AffineApplyOp>(endValOp))
        {
            ResolveAffineRangeEnd(rewriter, affineApplyOp);

            // the AffineApplyOp has been resolved but we don't have a handle to the ssa
            // value that replaced it, so we need to grab it from the schedule operands again
            endVal = op.getOperand(operandIndex);
            endValOp = endVal.getDefiningOp();
        }

        if (auto constantEnd = mlir::dyn_cast_or_null<mlir::ConstantOp>(endValOp))
        {
            // If this affine end has already been canonicalized to a constant, replace the range with the constant
            auto constantAttr = constantEnd.getValue();
            assert(constantAttr.isa<mlir::IntegerAttr>() && "Range Ends must be an integer constant");
            auto constantVal = constantAttr.cast<mlir::IntegerAttr>().getInt();
            range = lnir::Range{ range.Begin(), constantVal, range.Increment() };
        }
        else if (auto dimSizeOp = mlir::dyn_cast_or_null<DimSizeOp>(endValOp))
        {
            // If this affine end been canonicalized to a DimSizeOp, replace the range with the resolved dim size op value
            auto dimensionIndex = dimSizeOp.dimensionIndex();
            auto dimSize = util::GetDimSizeAt(dimensionIndex.getValue(), endValOp);
            assert(dimSize.has_value());
            range = lnir::Range{ range.Begin(), *dimSize, range.Increment() };
        }
        else
        {
            assert(false && "Unhandled Range end type");
        }
    }
}

LogicalResult ScheduleOpDomainResolution::matchAndRewrite(ScheduleOp op, PatternRewriter& rewriter) const
{
    if (op->getParentOfType<KernelOp>())
    {
        // ScheduleOps still inside of kernels should be left alone as other lowering
        // passes will still move them around into their final position
        return failure();
    }

    if (op.hasConstantRanges())
    {
        return failure();
    }

    rewriter.startRootUpdate(op);

    auto domain = op.getDomain().getValue();
    domain.ResolveRangeValues([&](lnir::Range& range) {
        ResolveRange(op, rewriter, range);
    });
    op.setDomain(domain);
    rewriter.finalizeRootUpdate(op);
    return success();
}

//
// ScheduleOpConversion
//

LogicalResult ScheduleOpConversion::matchAndRewrite(ScheduleOp op, PatternRewriter& rewriter) const
{
    if (op->getParentOfType<KernelOp>())
    {
        // ScheduleOps still inside of kernels should be left alone as other lowering
        // passes will still move them around into their final position
        return failure();
    }

    auto loc = util::GetLocation(rewriter, "ScheduleOpConversion", op.getLoc());

    auto domain = op.getDomain().getValue();
    domain.ResolveRangeValues([&](lnir::Range& range) {
        ResolveRange(op, rewriter, range);
    });
    op.setDomain(domain);

    LoopNestBuilder builder(op, rewriter, printLoops);
    builder.BuildLoopNest();
    rewriter.eraseOp(op);
    return success();
}

// The SaturatedAccumulateLoopRewrite pattern rewrites loops that look like 8-bit dot-product-like computations so
// LLVM can (hopefully) generate more efficient vectorized assembly code. There are 2 main things the optimization does:
//
// - Adds "saturated" arithmetic
// - Unrolls and reorders dot-product-like loops
//
// The saturated arithmetic transformation operates on the output of a 32-bit addition operator where both of the inputs
// are the products of 8-bit integers. The transformation adds instructions to clamp the output of the addition to lie in
// the range of a 16-bit (signed) integer. It transforms
// `x = a + b`
//   into:
// `x = clamp(a + b, -32768, 32767)`
//
// The unroll-and-reorder transformation takes a fixed-length loop that accumulates a sequence of products into a memory
// location, unrolls the loop, and reorders the operations to be more friendly to LLVM's code generation. To illustrate,
// consider the loop:
//  ```
//  for i in [0, 4)
//    c += f(i)
//  ```
//
// After unrolling and removing intermediate loads/stores this becomes:
// ```
// c = f(3) + (f(2) + (f(1) + (f(0) + c)))
// ```
//
// The reorder transformation turns this into:
// ```
// c = ((f(3) + f(2)) + (f(1) + f(0))) + c
// ```

LogicalResult SaturatedAccumulateLoopRewrite::matchAndRewrite(AffineForOp loopOp, PatternRewriter& rewriter) const
{
    if (!loopOp->getAttr("rcv_saturated"))
        return success();

    if (!loopOp.hasConstantBounds())
        return success();

    // Only operate on the innermost loop
    bool isInnermost = true;
    loopOp.getBody()->walk([&](AffineForOp innerLoop) { isInnermost = false; });
    if (!isInnermost)
        return success();

    // Currently only check loops of length 4
    int64_t begin = loopOp.getConstantLowerBound();
    int64_t end = loopOp.getConstantUpperBound();
    int64_t step = loopOp.getStep();
    int64_t numIter = (end - begin) / step;
    if (numIter != 4)
        return success();

    // Look for [Affine]LoadOps and [Affine]StoreOps
    mlir::Operation* loadOp = nullptr;
    mlir::Operation* storeOp = nullptr;
    Value valueToStore;
    for (auto& op : loopOp.getBody()->without_terminator())
    {
        mlir::Operation* thisLoadOp = nullptr;
        mlir::Value memRef;
        mlir::ValueRange indices;
        mlir::AffineMap map;
        if (auto load = mlir::dyn_cast<mlir::memref::LoadOp>(op))
        {
            thisLoadOp = &op;
            memRef = load.getMemRef();
            indices = load.getIndices();
        }
        else if (auto load = mlir::dyn_cast<mlir::AffineLoadOp>(op))
        {
            thisLoadOp = &op;
            memRef = load.getMemRef();
            indices = load.indices();
            map = load.getAffineMap();
        }

        if (thisLoadOp)
        {
            // Check that the load op is eventually used by a StoreOp to the same memory location
            auto stores = util::getRecursiveUsesOfType<mlir::memref::StoreOp>(op.getResult(0));
            for (auto store : stores)
            {
                if (store.getMemRef() == memRef && store.getIndices() == indices && map.isIdentity())
                {
                    loadOp = thisLoadOp;
                    storeOp = store.getOperation();
                    valueToStore = store.getValueToStore();
                }
            }

            // Didn't find a StoreOp, look for an AffineStoreOp
            if (!storeOp)
            {
                auto stores = util::getRecursiveUsesOfType<mlir::AffineStoreOp>(op.getResult(0));
                for (auto store : stores)
                {
                    if (store.getMemRef() == memRef && store.indices() == indices && store.getAffineMap() == map)
                    {
                        loadOp = thisLoadOp;
                        storeOp = store.getOperation();
                        valueToStore = store.value();
                    }
                }
            }
        }

        if (storeOp) break;
    }

    if (!storeOp)
        return success();

    auto storeType = valueToStore.getType();

    // Currently only check for result type == int32
    if (!storeType.isInteger(32))
        return success();

    // Is the stored value from an add?
    value::BinOp binOp;
    if (binOp = mlir::dyn_cast_or_null<value::BinOp>(valueToStore.getDefiningOp()); !binOp || binOp.getPredicate() != value::BinaryOpPredicate::ADD)
        return success();

    auto isExtFromInt8 = [](mlir::Value v) {
        return (mlir::isa<SignExtendIOp>(v.getDefiningOp()) ||
                mlir::isa<ZeroExtendIOp>(v.getDefiningOp())) &&
               v.getDefiningOp()->getOperand(0).getType().isInteger(8);
    };
    auto isMulOfExtFromInt8 = [&](mlir::Value v) {
        if (auto binOp = mlir::dyn_cast_or_null<value::BinOp>(v.getDefiningOp()); binOp && binOp.getPredicate() == value::BinaryOpPredicate::MUL)
        {
            return isExtFromInt8(binOp.getOperand(0)) && isExtFromInt8(binOp.getOperand(1));
        }
        return false;
    };

    // Verify one operand is the product of two upcasted 8-bit values
    if (!isMulOfExtFromInt8(binOp.getOperand(0)) && !isMulOfExtFromInt8(binOp.getOperand(1)))
    {
        return success();
    }

    // Is one operand a load and the other a mul of values extended from int8?
    value::BinOp mulOp;
    if (isMulOfExtFromInt8(binOp.getOperand(0)) && binOp.getOperand(1).getDefiningOp() && binOp.getOperand(1).getDefiningOp() == loadOp)
    {
        mulOp = mlir::cast<value::BinOp>(binOp.getOperand(0).getDefiningOp());
    }
    else if (isMulOfExtFromInt8(binOp.getOperand(1)) && binOp.getOperand(0).getDefiningOp() && binOp.getOperand(0).getDefiningOp() == loadOp)
    {
        mulOp = mlir::cast<value::BinOp>(binOp.getOperand(1).getDefiningOp());
    }

    if (!mulOp)
        return success();

    // TODO: Also check that the operands of the mul operation are loads, and that they load from adjacent locations for adjacent values of the loop induction variable

    // Pattern matches! Now unroll the loop and convert it to something of the form:
    //   sat(p0 + p1) + sat(p2 + p3)

    auto loc = loopOp.getLoc();
    mlir::Value initialLoadVal;
    mlir::Value prevStoreVal;
    BlockAndValueMapping indexMap;

    auto log2 = [](int64_t n) {
        // reasonably efficient for small n
        int64_t result = 0;
        while (n != 0)
        {
            n >>= 1;
            result += 1;
        }
        return result - 1;
    };

    auto inductionVar = loopOp.getInductionVar();

    // Push ceil(log2(n)) + 1 empty vectors onto list of stacks
    std::vector<std::vector<mlir::Value>> accumStacks(log2(numIter - 1) + 2);
    auto inductionVarMap = AffineMap::get(1, 1, rewriter.getAffineDimExpr(0) + step * rewriter.getAffineSymbolExpr(0));
    for (int64_t i = 0; i < numIter; ++i)
    {
        auto offset = rewriter.create<mlir::ConstantIndexOp>(loc, step * i + begin);
        indexMap.map(inductionVar, offset);

        for (auto& op : loopOp.getBody()->without_terminator())
        {
            // Check if op is a bin op whose value is stored by the storeOp
            auto isAccumulate = mlir::isa<value::BinOp>(&op) && op.getResult(0) == valueToStore;

            // If we get an "accumulate" operation, push the non-loadOp arg onto stack 0
            //   if stack 0 has 2 arguments, pop them off, emit an add, and push the result onto stack 1
            //   if stack 1 has 2 arguments, pop them off, emit an add, and push the result onto stack 2
            //   ...and so on
            // If we've processed a power-of-2 number of addends, there is hopefully just one non-empty stack,
            // containing the value to accumulate
            if (isAccumulate)
            {
                if (op.getOperand(0).getDefiningOp() == loadOp)
                {
                    // push operand 1
                    accumStacks[0].push_back(indexMap.lookupOrNull(op.getOperand(1)));
                }
                else if (op.getOperand(1).getDefiningOp() == loadOp)
                {
                    // push operand 0
                    accumStacks[0].push_back(indexMap.lookupOrNull(op.getOperand(0)));
                }
                else
                {
                    op.emitError("Accumulate op didn't have a load as one of the arguments");
                    return failure();
                }

                // Now fix up stacks and emit add ops
                for (unsigned s = 0; s < accumStacks.size() - 1; ++s)
                {
                    if (accumStacks[s].size() > 2)
                        op.emitError("Error! stack size > 2");

                    if (accumStacks[s].size() == 2)
                    {
                        auto sum = rewriter.create<value::BinOp>(op.getLoc(), value::BinaryOpPredicate::ADD, accumStacks[s][0], accumStacks[s][1]);
                        auto sumVal = sum.getResult();
                        // At level 0 (where the values are up-casted 8-bit values), saturate the output to an intermediate 16-bit value
                        if (s == 0)
                        {
                            sumVal = SaturateValue(rewriter, sumVal, 16, true);
                        }

                        accumStacks[s + 1].push_back(sumVal);
                        accumStacks[s].clear();
                    }
                }
            }
            else if (&op == loadOp && i != 0)
            {
                indexMap.map(op.getResult(0), prevStoreVal);
            }
            else if (&op == storeOp && i != numIter - 1)
            {
                prevStoreVal = indexMap.lookupOrNull(valueToStore);
            }
            else if (&op == storeOp && i == numIter - 1)
            {
                // emit final add(s) & map to the value to store
                mlir::Value newSum;
                for (auto& stack : accumStacks)
                {
                    for (auto accumVal : stack)
                    {
                        if (!newSum)
                        {
                            newSum = accumVal;
                        }
                        else
                        {
                            auto sumOp = rewriter.create<value::BinOp>(op.getLoc(), value::BinaryOpPredicate::ADD, newSum, accumVal);
                            newSum = sumOp.getResult();
                        }
                    }
                }

                // now add the original load value
                auto sumOp = rewriter.create<value::BinOp>(op.getLoc(), value::BinaryOpPredicate::ADD, newSum, initialLoadVal);
                newSum = sumOp.getResult();

                indexMap.map(op.getOperand(0), newSum); // value to store
                [[maybe_unused]] auto mappedClonedOp = rewriter.clone(op, indexMap);
            }
            else
            {
                // common case: just clone the operation
                auto mappedClonedOp = rewriter.clone(op, indexMap);

                // special case for the first load operation: save the initial load value
                if (&op == loadOp && i == 0)
                {
                    initialLoadVal = mappedClonedOp->getResult(0);
                }
            }
        }
    }

    rewriter.eraseOp(loopOp);
    return success();
}

LogicalResult ScheduledLoopOpRewrite::matchAndRewrite(ScheduledLoopOp op, PatternRewriter& rewriter) const
{
    auto loc = op.getLoc();

    int64_t begin = op.beginAttr().getInt();
    int64_t end = op.endAttr().getInt();
    int64_t step = op.stepAttr().getInt();

    // First create the body loop and move the nested region into it
    auto bodyLoop = rewriter.create<AffineForOp>(op.getLoc(), begin, end, step);

    // Transfer attributes
    auto scheduledLoopOpAttrs = op->getAttrs();
    for (auto& [identifier, attrValue] : scheduledLoopOpAttrs)
    {
        bodyLoop->setAttr(identifier, attrValue);
    }

    // Insert the prologue in the nested loop before the rest of the body
    rewriter.inlineRegionBefore(op.prologue(), bodyLoop.region(), bodyLoop.region().end());

    // Insert the body in the nested loop after the prologue
    rewriter.inlineRegionBefore(op.body(), bodyLoop.region(), bodyLoop.region().end());

    // Insert the epilogue in the nested loop after the body
    rewriter.inlineRegionBefore(op.epilogue(), bodyLoop.region(), bodyLoop.region().end());

    // Now iterate through the newly created blocks and merge them together and erase terminator ops along the way
    auto blockIter = bodyLoop.region().begin();
    auto& initialBlock = *blockIter;
    ++blockIter;
    auto& prologueBlock = *blockIter;
    ++blockIter;
    auto& bodyBlock = *blockIter;
    ++blockIter;
    auto& epilogueBlock = *blockIter;

    // Erase terminators
    Operation* clonedTerminator = initialBlock.getTerminator()->clone();
    rewriter.eraseOp(initialBlock.getTerminator());
    rewriter.eraseOp(prologueBlock.getTerminator());
    rewriter.eraseOp(bodyBlock.getTerminator());
    rewriter.eraseOp(epilogueBlock.getTerminator());

    mlir::Value bodyBlockArg = initialBlock.getArgument(0);

    std::vector<mlir::Value> replacementArgs = { bodyBlockArg };

    // Merge the prologue block into the initial block
    rewriter.mergeBlocks(&prologueBlock, &initialBlock, replacementArgs);

    // Merge the body block into the initial block
    rewriter.mergeBlocks(&bodyBlock, &initialBlock, replacementArgs);

    // Merge the epilogue block into the initial block
    rewriter.mergeBlocks(&epilogueBlock, &initialBlock, replacementArgs);

    // Create an affine terminator
    rewriter.setInsertionPointToEnd(bodyLoop.getBody());
    rewriter.insert(clonedTerminator);

    // Replace usage of the symbolic index op and the body block arg with the body loop's induction variable
    auto symbolicIndexOp = op.getSymbolicIndex();

    bodyLoop.walk([&](Operation* walkOp) {
        walkOp->replaceUsesOfWith(symbolicIndexOp, bodyLoop.getInductionVar());
        walkOp->replaceUsesOfWith(bodyBlockArg, bodyLoop.getInductionVar());
    });

    rewriter.eraseOp(op);
    return success();
}

LogicalResult GPUMappedAffineForOpRewrite::matchAndRewrite(mlir::AffineForOp affineForOp, PatternRewriter& rewriter) const
{
    auto loc = affineForOp.getLoc();

    if (auto gpuMapAttr = affineForOp->getAttrOfType<StringAttr>("rcv_gpu_map"))
    {
        auto iv = affineForOp.getInductionVar();
        int64_t begin = affineForOp.getConstantLowerBound();
        int64_t end = affineForOp.getConstantUpperBound();
        int64_t step = affineForOp.getStep();

        auto processor = *symbolizeProcessor(gpuMapAttr.getValue());
        auto affineScopeParent = affineForOp->getParentWithTrait<mlir::OpTrait::AffineScope>();
        auto& firstRegion = affineScopeParent->getRegion(0);
        auto& firstBlock = firstRegion.front();
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPoint(&firstBlock, firstBlock.begin());
        loc = affineForOp.getLoc();

        namespace vir = accera::ir::value;
        using Processor = vir::Processor;
        auto gpuValue = [&]() -> mlir::Value {
            switch (processor)
            {
            case Processor::ThreadX:
                return rewriter.create<gpu::ThreadIdOp>(loc, rewriter.getIndexType(), "x");
            case Processor::ThreadY:
                return rewriter.create<gpu::ThreadIdOp>(loc, rewriter.getIndexType(), "y");
            case Processor::ThreadZ:
                return rewriter.create<gpu::ThreadIdOp>(loc, rewriter.getIndexType(), "z");
            case Processor::BlockX:
                return rewriter.create<gpu::BlockIdOp>(loc, rewriter.getIndexType(), "x");
            case Processor::BlockY:
                return rewriter.create<gpu::BlockIdOp>(loc, rewriter.getIndexType(), "y");
            case Processor::BlockZ:
                return rewriter.create<gpu::BlockIdOp>(loc, rewriter.getIndexType(), "z");
            case Processor::Sequential:
                [[fallthrough]];
            default:
                llvm_unreachable("Unexpected");
            }
        }();
        // We're going to replace all uses of the affine loop's induction variable with GPU hardware mapping instead, so
        // make the AffineForOp effectively a no-op

        // TODO : incorporate the grid/block dims to make this more like an unroll across the hardware dims
        //        rather than always setting it to [0,1)
        affineForOp.setStep(1);
        affineForOp.setConstantLowerBound(0);
        affineForOp.setConstantUpperBound(1);
        auto idxMap = AffineMap::get(1, 0, rewriter.getAffineDimExpr(0) * rewriter.getAffineConstantExpr(step));
        mlir::Value affineApplyResult = rewriter.create<mlir::AffineApplyOp>(loc, idxMap, ValueRange{ gpuValue });

        affineForOp.walk([&](Operation* walkOp) {
            walkOp->replaceUsesOfWith(iv, affineApplyResult);
        });
    }

    return success();
}

//
// ScheduledLoopOpIndexConversion
//

void IndexReplacedOpUpdates(Operation* op, PatternRewriter& rewriter)
{
    // Make any adjustments to ops now that their symbolic index operands have been replaces with AffineForOp IVs
    PatternRewriter::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(op);
    if (auto affineApplyOp = mlir::dyn_cast_or_null<mlir::AffineApplyOp>(op))
    {
        // TODO : remove SymbolicIndexOps and we won't need to do this anymore
        // Convert symbol arguments to dimension arguments
        auto map = affineApplyOp.getAffineMap();
        std::vector<AffineExpr> dimReplacements;
        std::vector<AffineExpr> symReplacements;
        symReplacements.reserve(map.getNumSymbols());
        for (unsigned idx = map.getNumDims(); idx <= map.getNumDims() + map.getNumSymbols(); ++idx)
        {
            symReplacements.push_back(rewriter.getAffineDimExpr(idx));
        }
        auto updatedMap = map.replaceDimsAndSymbols(dimReplacements, symReplacements, map.getNumDims() + map.getNumSymbols(), 0);
        rewriter.replaceOpWithNewOp<mlir::AffineApplyOp>(op, updatedMap, affineApplyOp.getOperands());
    }
}

LogicalResult ScheduledLoopOpIndexConversion::matchAndRewrite(ScheduledLoopOp op, PatternRewriter& rewriter) const
{
    auto loc = op.getLoc();

    rewriter.startRootUpdate(op);
    auto indexOp = op.getSymbolicIndex();

    op.prologue().walk([&](Operation* innerOp) {
        innerOp->replaceUsesOfWith(indexOp, op.getPrologueArgPlaceholder());
        IndexReplacedOpUpdates(innerOp, rewriter);
    });
    op.body().walk([&](Operation* innerOp) {
        innerOp->replaceUsesOfWith(indexOp, op.getBodyArgPlaceholder());
        IndexReplacedOpUpdates(innerOp, rewriter);
    });
    op.epilogue().walk([&](Operation* innerOp) {
        innerOp->replaceUsesOfWith(indexOp, op.getEpilogueArgPlaceholder());
        IndexReplacedOpUpdates(innerOp, rewriter);
    });
    rewriter.finalizeRootUpdate(op);

    return success();
}

//
// DimSizeOpConversion
//

LogicalResult DimSizeOpConversion::matchAndRewrite(DimSizeOp op, PatternRewriter& rewriter) const
{
    // DimSizeOps still inside of kernels should be left alone
    if (op->getParentOfType<KernelOp>())
    {
        return failure();
    }

    auto loc = util::GetLocation(rewriter, "DimSizeOpConversion", op.getLoc());

    auto dimensionIndexAttr = op.dimensionIndex().dyn_cast_or_null<IndexAttr>();
    assert(dimensionIndexAttr != nullptr && "DimSizeOp must have a dimensionIndex as an IndexAttr");
    auto dimensionIndex = dimensionIndexAttr.getValue();

    auto dimensionSize = util::GetDimSizeAt(dimensionIndex, op.getOperation());
    if (dimensionSize.has_value())
    {
        auto constantOp = rewriter.create<mlir::ConstantOp>(loc, rewriter.getIntegerAttr(rewriter.getIndexType(), *dimensionSize));
        constantOp->setAttr(DimSizeOp::getIndexAttrName(), dimensionIndexAttr);
        rewriter.replaceOp(op, constantOp.getResult());
    }
    else
    {
        // If DimSizeOp doesn't have a parent or doesn't have one of the subdomain size or index order attributes, then treat it
        // as being outside the iteration domain and return 0
        auto constantOp = rewriter.create<mlir::ConstantOp>(loc, rewriter.getIntegerAttr(rewriter.getIndexType(), 0));
        constantOp->setAttr(DimSizeOp::getIndexAttrName(), dimensionIndexAttr);
        rewriter.replaceOp(op, constantOp.getResult());
    }

    return success();
}

LogicalResult ReplaceSymbolicIndexOpPattern::matchAndRewrite(SymbolicIndexOp op, PatternRewriter& rewriter) const
{
    auto loc = util::GetLocation(rewriter, "ReplaceSymbolicIndexOpPattern", op.getLoc());

    // TODO: need to get the domain somehow and check if the index is a computed index
    // (or somehow add compound indices to the outer loop where they're fully-defined)
    auto index = op.getValue();
    auto didReplaceAll = ReplaceAllUsesWith(op.getResult(), [&](mlir::OpOperand& use) -> mlir::Value {
        return FindIndexVariable(index, use.getOwner());
    });

    if (didReplaceAll)
    {
        rewriter.eraseOp(op);
        return success();
    }
    return failure();
}

LogicalResult UnlinkAndRemoveSymbolicIndexOpPattern::matchAndRewrite(SymbolicIndexOp op, PatternRewriter& rewriter) const
{
    auto loc = util::GetLocation(rewriter, "UnlinkAndRemoveSymbolicIndexOpPattern", op.getLoc());

    auto zeroOp = rewriter.create<mlir::ConstantOp>(loc, rewriter.getIntegerAttr(rewriter.getIndexType(), 0));
    op.getOperation()->replaceAllUsesWith(zeroOp);

    rewriter.eraseOp(op);
    return success();
}

LogicalResult MergeNearlyIdenticalSiblingLoops::matchAffineForOps(AffineForOp op1, AffineForOp op2, bool topLevel /* = false */) const
{
    if (op1->hasAttr("merge_exempt") || op2->hasAttr("merge_exempt"))
    {
        return failure();
    }

    // if top level, they need to be siblings
    if (topLevel)
    {
        if (op1->getBlock() != op2->getBlock())
        {
            return failure();
        }
    }

    // Gather any begin/end cache regions around the second op, if any
    auto nextOp1 = op1->getNextNode(), nextOp2 = op2->getNextNode();
    llvm::SmallVector<executionPlan::BeginCacheRegionOp> op2BeginRegionOps;
    llvm::SmallVector<executionPlan::EndCacheRegionOp> op2EndRegionOps;
    while (nextOp1 || nextOp2)
    {

        if (auto endOp1 = dyn_cast_or_null<executionPlan::EndCacheRegionOp>(nextOp1),
            endOp2 = dyn_cast_or_null<executionPlan::EndCacheRegionOp>(nextOp2);
            // if there's a mismatch, bail
            !!endOp1 ^ !!endOp2)
        {
            return failure();
        }
        else if (endOp1 && endOp2)
        {
            // if we have a mismatch in begin region ids, bail
            if (auto op2BeginRegionOp = endOp2.getBeginOp(); endOp1.getBeginOp().id() != op2BeginRegionOp.id())
            {
                return failure();
            }
            else
            {
                op2BeginRegionOps.push_back(op2BeginRegionOp);
            }

            op2EndRegionOps.push_back(endOp2);

            // move to the next opp
            nextOp1 = nextOp1->getNextNode();
            nextOp2 = nextOp2->getNextNode();
        }
        else if (!endOp1 && !endOp2)
        {
            break;
        }
    }

    // we expect there to be matching pairs
    assert(op2BeginRegionOps.size() == op2EndRegionOps.size());

    if (topLevel)
    {
        // if we're at the top, then we expect the number of ops between our "adjacent" affine for ops to be related to the number of begin/end region ops found
        Block::iterator op1It{ op1 }, op2It{ op2 };
        if (static_cast<size_t>(std::distance(op1It, op2It)) != (op2BeginRegionOps.size() + op2EndRegionOps.size() + 1))
        {
            return failure();
        }

        // Ensure the op1's upper bound matches op2's lower bound, the steps match,
        // and the amount of left over in the iteration space remains the same
        if (auto ub = op2.getConstantUpperBound(), lb = op2.getConstantLowerBound(), step = op2.getStep();
            lb != op1.getConstantUpperBound() ||
            step != op1.getStep() ||
            ((ub - op1.getConstantLowerBound()) % step != 0))
        {
            return failure();
        }
    }
    else
    {
        // if we're not at the top, then we expect our descendant affine for bodies to contain at most an affine for op, a terminator, and any relevant begin/end cache region ops
        auto block1 = op1->getBlock(), block2 = op2->getBlock();
        if (block1->getOperations().size() != block2->getOperations().size() ||
            block1->getOperations().size() != (op2BeginRegionOps.size() + // begin regions
                                               1 + // affine for op
                                               op2EndRegionOps.size() + // end regions
                                               1 // terminator
                                               ))
        {
            return failure();
        }

        // we also ensure that the iteration space matches
        if (op1.getConstantLowerBound() != op2.getConstantLowerBound() ||
            op1.getConstantUpperBound() != op2.getConstantUpperBound() ||
            op1.getStep() != op2.getStep())
        {
            return failure();
        }
    }

    auto op1Kernels = op1->getAttrOfType<ArrayAttr>("kernels");
    auto op2Kernels = op2->getAttrOfType<ArrayAttr>("kernels");

    // if there's a mismatch, bail
    if (!!op1Kernels ^ !!op2Kernels)
    {
        return failure();
    }
    // both ops have the kernels attribute and they match. no need to descend any further (one hopes)
    else if (op1Kernels && op2Kernels)
    {
        if (op1Kernels != op2Kernels)
        {
            return failure();
        }
        else
        {
            return success();
        }
    }
    // if there are no kernels, descend further into the affine for ops, in case there are more nested loops
    else if (!op1Kernels && !op2Kernels)
    {
        for (auto&& [op1Child, op2Child] : llvm::zip(op1.getLoopBody().getOps(), op2.getLoopBody().getOps()))
        {
            // it's only valid for the first ops to be either begin cache region ops or affine for ops. anything else, we can't handle

            if (
                auto beginOp1 = dyn_cast<executionPlan::BeginCacheRegionOp>(&op1Child),
                beginOp2 = dyn_cast<executionPlan::BeginCacheRegionOp>(&op2Child);
                // if there's a mismatch, bail
                !!beginOp1 ^ !!beginOp2)
            {
                return failure();
            }
            // if they both have a begin cache region op, make sure the id matches
            else if (beginOp1 && beginOp2)
            {
                if (beginOp1.id() != beginOp2.id())
                {
                    return failure();
                }
                continue;
            }

            // if we're here, then we're out of begin cache region ops
            if (auto affineChild1 = dyn_cast<mlir::AffineForOp>(&op1Child),
                affineChild2 = dyn_cast<mlir::AffineForOp>(&op2Child);
                !!affineChild1 ^ !!affineChild2)
            {
                return failure();
            }
            else if (affineChild1 && affineChild2)
            {
                return matchAffineForOps(affineChild1, affineChild2);
            }

            return failure();
        }
    }

    return failure();
}

LogicalResult MergeNearlyIdenticalSiblingLoops::matchAndRewrite(AffineForOp op, PatternRewriter& rewriter) const
{
    auto firstLoop = op;

    Operation* rawOp = op->getNextNode();
    while (rawOp)
    {
        if (auto secondLoop = dyn_cast<AffineForOp>(rawOp))
        {
            auto res = matchAffineForOps(firstLoop, secondLoop, /* topLevel */ true);
            if (succeeded(res))
            {
                rewriter.updateRootInPlace(firstLoop, [&] {
                    firstLoop.setConstantUpperBound(secondLoop.getConstantUpperBound());

                    // the "end" attribute isn't technically needed at this point. it's a benign artefact from an earlier
                    // conversion in the pipeline, but just to prevent confusion, we update it too
                    if (auto endAttr = secondLoop->getAttr("end"); firstLoop->hasAttr("end") && endAttr)
                    {
                        firstLoop->setAttr("end", endAttr);
                    }
                });
                rewriter.eraseOp(secondLoop);
            }
            return res;
        }
        rawOp = rawOp->getNextNode();
    }

    return failure();
}

LogicalResult RemoveZeroIterationLoopPattern::matchAndRewrite(AffineForOp op, PatternRewriter& rewriter) const
{
    Optional<uint64_t> mayBeConstantTripCount = mlir::getConstantTripCount(op);
    if (mayBeConstantTripCount.hasValue())
    {
        uint64_t constantTripCount = mayBeConstantTripCount.getValue();
        if (constantTripCount == 0)
        {
            rewriter.eraseOp(op);
            return success();
        }
    }
    return failure();
}

LogicalResult PromoteSingleIterationLoopPattern::matchAndRewrite(AffineForOp op, PatternRewriter& rewriter) const
{
    return util::PromoteIfSingleIteration(rewriter, op);
}

namespace
{
#include "nest/LoopNestToValue.inc"
} // namespace

namespace accera::transforms
{

void populateRangeResolutionPatterns(mlir::OwningRewritePatternList& patterns)
{
    mlir::MLIRContext* context = patterns.getContext();
    patterns.insert<ScheduleOpDomainResolution>(context);
}

void populateScheduleScaffoldingPatterns(bool printLoops, mlir::OwningRewritePatternList& patterns)
{
    mlir::MLIRContext* context = patterns.getContext();
    patterns.insert<ScheduleOpConversion>(context, printLoops);
}

void populateScheduledOperationsPatterns(mlir::OwningRewritePatternList& patterns)
{
    mlir::MLIRContext* context = patterns.getContext();
    patterns.insert<DimSizeOpConversion,
                    RemoveScheduledKernelOpPattern,
                    ScheduledLoopOpIndexConversion>(context);
}

void populateScheduleToValueRewritePatterns(mlir::OwningRewritePatternList& patterns)
{
    mlir::MLIRContext* context = patterns.getContext();
    patterns.insert<ScheduledLoopOpRewrite>(context);
}

void populateGPUIndexMappingRewritePatterns(mlir::OwningRewritePatternList& patterns)
{
    mlir::MLIRContext* context = patterns.getContext();
    patterns.insert<GPUMappedAffineForOpRewrite>(context);
}

void populateScheduleToValuePatterns(mlir::OwningRewritePatternList& patterns)
{
    mlir::MLIRContext* context = patterns.getContext();
    populateWithGenerated(patterns);
    patterns.insert<RemoveKernelOpPattern,
                    RemoveScheduledKernelOpPattern,
                    RemoveSymIndexOpPattern>(context);
}

void populateSymIndexCleanupPatterns(mlir::OwningRewritePatternList& patterns)
{
    mlir::MLIRContext* context = patterns.getContext();
    patterns.insert<RemoveSymIndexOpPattern>(context);
}

void populateLoopOptimizationPatterns(mlir::OwningRewritePatternList& patterns)
{
    mlir::MLIRContext* context = patterns.getContext();
    patterns.insert<SaturatedAccumulateLoopRewrite>(context);
}

void populateLoopMergingPatterns(mlir::OwningRewritePatternList& patterns)
{
    mlir::MLIRContext* context = patterns.getContext();
    patterns.insert<MergeNearlyIdenticalSiblingLoops>(context);
}

void populateLoopSimplificationPatterns(mlir::OwningRewritePatternList& patterns)
{
    mlir::MLIRContext* context = patterns.getContext();
    patterns.insert<RemoveZeroIterationLoopPattern,
                    PromoteSingleIterationLoopPattern>(context);
}

} // namespace accera::transforms
