////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "nest/LoopNestPasses.h"
#include "util/MathUtilities.h"

#include "ir/include/nest/LoopNestAttributes.h"
#include "ir/include/nest/TransformedDomain.h"
#include <ir/include/IRUtil.h>
#include <ir/include/exec/ExecutionPlanOps.h>
#include <ir/include/nest/AffineExpression.h>
#include <ir/include/nest/LoopIndexInfo.h>
#include <ir/include/nest/LoopNestBuilder.h>
#include <ir/include/nest/LoopNestOps.h>
#include <ir/include/nest/Util.h>
#include <ir/include/value/ValueDialect.h>

#include <utilities/include/Exception.h>

#include <llvm/Support/Debug.h>
#include <llvm/Support/raw_os_ostream.h>

#include <mlir/Dialect/Affine/Analysis/LoopAnalysis.h>
#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/Arithmetic/IR/Arithmetic.h>
#include <mlir/Dialect/GPU/GPUDialect.h>
#include <mlir/Dialect/LLVMIR/ROCDLDialect.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/IR/AffineMap.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/BlockAndValueMapping.h>
#include <mlir/IR/IntegerSet.h>
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
int64_t log2(int64_t n)
{
    // reasonably efficient for small n
    int64_t result = 0;
    while (n != 0)
    {
        n >>= 1;
        result += 1;
    }
    return result - 1;
}

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

[[maybe_unused]] SymbolicIndexOp GetOrCreateSymbolicIndex(OpBuilder& builder, Index index, Operation* where)
{
    auto parentFuncOp = where->getParentOfType<ValueFuncOp>();

    for (auto& region : parentFuncOp->getRegions())
    {
        SymbolicIndexOp result;
        region.walk([&](Operation* op) {
            if (auto indexOp = dyn_cast_or_null<SymbolicIndexOp>(op))
            {
                if (indexOp.getValue() == index)
                {
                    result = indexOp;
                    return WalkResult::interrupt();
                }
            }

            return WalkResult::advance();
        });

        if (result)
            return result;
    }

    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(&parentFuncOp.body().front());
    return builder.create<SymbolicIndexOp>(parentFuncOp.getLoc(), index);
}

mlir::Value EmitIndexExpression(OpBuilder& builder, Location loc, const AffineExpression& expr, const TransformedDomain& domain)
{
    std::vector<mlir::Value> symbols;
    AffineExpr affineExpr = expr.GetAffineExpr();
    auto exprIndices = expr.GetIndices();
    std::vector<mlir::Value> indices;
    auto currBlock = builder.getInsertionBlock();
    std::transform(exprIndices.begin(), exprIndices.end(), std::back_inserter(indices), [&](auto i) -> mlir::Value {
        if (auto var = FindIndexVariable(i, currBlock->getParentOp()))
            return var;

        return GetOrCreateSymbolicIndex(builder, i, currBlock->getParentOp());
    });
    auto map = AffineMap::get(indices.size(), symbols.size(), affineExpr);
    indices.insert(indices.end(), symbols.begin(), symbols.end());
    auto exprOp = builder.create<AffineApplyOp>(loc, map, indices);

    return exprOp;
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

struct LowPrecisionIntAccumulateLoopRewrite : public OpRewritePattern<AffineForOp>
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
        auto kernelName = StringAttr::get(op.getContext(), op.getId());
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
            rewriter.replaceOpWithNewOp<arith::ConstantIndexOp>(op, *dimSize);
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
        if (auto constantOp = mlir::dyn_cast_or_null<arith::ConstantOp>(op))
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
    rewriter.replaceOpWithNewOp<arith::ConstantOp>(affineApplyOp, foldAttr);
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

        if (auto constantEnd = mlir::dyn_cast_or_null<arith::ConstantOp>(endValOp))
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
        else if (endVal.isa<mlir::BlockArgument>())
        {
            // A runtime size was provided as an argument to the nestOp
            range = lnir::Range{ range.Begin(), endVal, range.Increment() };
        }
        else
        {
            assert(false && "Unhandled Range end type");
        }
    }
    else if (range.HasSymbolNameEnd())
    {
        // "{arg, idx}""
        std::string symName = range.SymbolNameEnd();
        int start = symName.find_first_of(',');
        int end = symName.find_first_of('}');
        int idx = std::stoi(symName.substr(start + 1, end - start - 1));
        auto parentFuncOp = op.getOperation()->getParentOfType<ValueFuncOp>();
        auto endValue = parentFuncOp.getArgument(idx);
        range = lnir::Range{ range.Begin(), endValue, range.Increment() };
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
    if (op->getParentOfType<NestOp>())
    {
        // ScheduleOps still inside of NestOps should also be left alone as other lowering
        // passes will still update nest state before creating the ValueLambdaOp that the nest should be expanded in
        return failure();
    }

    auto domain = op.getDomain().getValue();
    domain.ResolveRangeValues([&](lnir::Range& range) {
        ResolveRange(op, rewriter, range);
    });
    op.setDomain(domain);

    LoopNestBuilder builder(op, rewriter, printLoops);
    auto loops = builder.BuildLoopNest();

    // Copy the domain to the generated loops
    auto domainAttr = TransformedDomainAttr::get(domain, op->getContext());
    for (auto& loop : loops)
    {
        loop->setAttr("domain", domainAttr);
        loop.walk([&](ScheduledLoopOp innerLoop) {
            innerLoop->setAttr("domain", domainAttr);
        });
    }
    rewriter.eraseOp(op);
    return success();
}

// The LowPrecisionIntAccumulateLoopRewrite pattern rewrites loops that look like 8-bit dot-product-like computations so
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

LogicalResult LowPrecisionIntAccumulateLoopRewrite::matchAndRewrite(AffineForOp loopOp, PatternRewriter& rewriter) const
{
    if (!loopOp.hasConstantBounds())
        return failure();

    // Only operate on the innermost loop
    bool isInnermost = true;
    loopOp.getBody()->walk([&](AffineForOp innerLoop) { isInnermost = false; });
    if (!isInnermost)
        return failure();

    // Currently only check loops of length 4
    int64_t begin = loopOp.getConstantLowerBound();
    int64_t end = loopOp.getConstantUpperBound();
    int64_t step = loopOp.getStep();
    int64_t numIter = (end - begin) / step;
    if (numIter != 4)
        return failure();

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
        return failure();

    auto storeType = valueToStore.getType();

    // Currently only check for result type == int32
    if (!storeType.isInteger(32))
        return failure();

    // Is the stored value from an add?
    value::BinOp binOp;
    if (binOp = mlir::dyn_cast_or_null<value::BinOp>(valueToStore.getDefiningOp()); !binOp || binOp.getPredicate() != value::BinaryOpPredicate::ADD)
        return failure();

    // true if v is of the form: sext(<i8 value>) or zext(<i8 value>)
    auto isExtFromInt8 = [](mlir::Value v) {
        return (mlir::isa<arith::ExtSIOp>(v.getDefiningOp()) ||
                mlir::isa<arith::ExtUIOp>(v.getDefiningOp())) &&
               v.getDefiningOp()->getOperand(0).getType().isInteger(8);
    };

    // true if v is of the form: mul( ?ext(<i8 value>), ?ext(<i8 value>) )
    auto isMulOfExtFromInt8 = [&](mlir::Value v) {
        if (auto binOp = mlir::dyn_cast_or_null<value::BinOp>(v.getDefiningOp()); binOp && binOp.getPredicate() == value::BinaryOpPredicate::MUL)
        {
            return isExtFromInt8(binOp.getOperand(0)) && isExtFromInt8(binOp.getOperand(1));
        }
        return false;
    };

    // Verify one operand of binOp is the product of two upcasted 8-bit values (that is, is of the form mul( ?ext(<i8 value>), ?ext(<i8 value>) ) )
    if (!isMulOfExtFromInt8(binOp.getOperand(0)) && !isMulOfExtFromInt8(binOp.getOperand(1)))
    {
        return failure();
    }

    // Is one operand of binOp a load and the other a mul of values extended from int8?
    // This is, does it match the pattern:
    //   (mul (ext (v1)) (ext (v2))) + (load ...)    or
    //   (load ...) + (mul (ext (v1)) (ext (v2)))
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
        return failure();

    // TODO: Also check that the operands of the mul operation are loads, and that they load from adjacent locations for adjacent values of the loop induction variable

    // Pattern matches! Now unroll the loop and convert it to something of the form:
    //   sat(p0 + p1) + sat(p2 + p3)

    auto loc = loopOp.getLoc();
    mlir::Value initialLoadVal;
    mlir::Value prevStoreVal;
    BlockAndValueMapping indexMap;

    auto inductionVar = loopOp.getInductionVar();

    // Push ceil(log2(n)) + 1 empty vectors onto list of stacks
    std::vector<std::vector<mlir::Value>> accumStacks(log2(numIter - 1) + 2);
    [[maybe_unused]] auto inductionVarMap = AffineMap::get(1, 1, rewriter.getAffineDimExpr(0) + step * rewriter.getAffineSymbolExpr(0));
    for (int64_t i = 0; i < numIter; ++i)
    {
        auto offset = rewriter.create<arith::ConstantIndexOp>(loc, step * i + begin);
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
                    {
                        op.emitError("Error! stack size > 2");
                        return failure();
                    }

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
            } // not an accumulate op:
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
    ScheduledLoopOp::Adaptor adaptor{ op };

    // First create the body loop and move the nested region into it
    auto bodyLoop = rewriter.create<AffineForOp>(loc, adaptor.beginOperands(), adaptor.beginMap(), adaptor.endOperands(), adaptor.endMap(), adaptor.step());

    // Transfer attributes
    auto scheduledLoopOpAttrs = op->getAttrs();
    for (auto& attr : scheduledLoopOpAttrs)
    {
        // HACK: Don't copy the domain attribute in case we later inline a dynamically-sized domain into a statically-sized region and the domain doesn't adjust correctly for serialization
        //       (we also no longer need the domain after building out the loopnest)
        if (attr.getName() != "domain")
        {
            bodyLoop->setAttr(attr.getName(), attr.getValue());
        }
    }
    // Hack for erasing loops
    if (bodyLoop->hasAttr("_erase"))
    {
        bodyLoop.setConstantLowerBound(0);
        bodyLoop.setConstantUpperBound(1);
        bodyLoop.setStep(1);
    }

    auto bodyLoopRegion = &bodyLoop.region();
    auto bodyRegion = bodyLoopRegion;

    // If necessary, add a conditional
    mlir::AffineIfOp conditional = nullptr;
    bool useConditional = op->hasAttr("accv_upper_limit");
    if (useConditional)
    {
        auto upperLimit = op->getAttrOfType<IntegerAttr>("accv_upper_limit").getInt();
        auto upperLimitIndex = op->getAttrOfType<IndexAttr>("accv_upper_limit_index").getValue();
        TransformedDomain domain = op->getAttrOfType<TransformedDomainAttr>("domain").getValue();
        // TODO: if the index hasn't been seen yet, replace it with its range begin
        // if (...)
        // {
        //     AffineExpr affineExpr = rewriter.getAffineConstantExpr(domain.GetIndexRange(upperLimitIndex).Begin());
        //     expr = AffineExpression(affineExpr, {});
        // }
        AffineExpression expr;
        if (!domain.IsComputedIndex(upperLimitIndex))
        {
            // TODO: Eventually fix `GetReducedIndexExpr` to work in this case as well
            expr = domain.GetIndexExpr(upperLimitIndex);
            if (expr.IsIdentity())
            {
                expr = AffineExpression(rewriter.getAffineDimExpr(0), { upperLimitIndex });
            }
        }
        else
        {
            expr = domain.GetReducedIndexExpr(upperLimitIndex, rewriter.getContext());
        }

        rewriter.setInsertionPointToStart(bodyLoop.getBody());
        auto indexValue = EmitIndexExpression(rewriter, loc, expr, domain);

        mlir::AffineExpr endExpr = rewriter.getAffineConstantExpr(upperLimit - 1) - rewriter.getAffineDimExpr(0);
        auto inRange = mlir::IntegerSet::get(1, 0, endExpr, false);
        conditional = rewriter.create<mlir::AffineIfOp>(loc, inRange, ValueRange{ indexValue }, false); // false indicating we do not want an "else" region
        bodyRegion = &conditional.thenRegion();
    }

    // Insert the prologue in the nested loop before the rest of the body
    rewriter.inlineRegionBefore(op.prologue(), *bodyRegion, bodyRegion->end());

    // Insert the body in the nested loop after the prologue
    rewriter.inlineRegionBefore(op.body(), *bodyRegion, bodyRegion->end());

    // Insert the epilogue in the nested loop after the body
    rewriter.inlineRegionBefore(op.epilogue(), *bodyRegion, bodyRegion->end());

    // Now iterate through the newly created blocks and merge them together and erase terminator ops along the way
    auto& initialLoopBlock = *bodyLoopRegion->begin();
    auto& initialBlock = *bodyRegion->begin();

    auto blockIter = bodyRegion->begin();
    ++blockIter;
    auto& prologueBlock = *blockIter;
    ++blockIter;
    auto& bodyBlock = *blockIter;
    ++blockIter;
    auto& epilogueBlock = *blockIter;

    // Erase terminators
    Operation* clonedTerminator = initialLoopBlock.getTerminator()->clone();
    rewriter.eraseOp(initialLoopBlock.getTerminator());
    rewriter.eraseOp(prologueBlock.getTerminator());
    rewriter.eraseOp(bodyBlock.getTerminator());
    rewriter.eraseOp(epilogueBlock.getTerminator());

    mlir::Operation* clonedConditionalTerminator = nullptr;
    if (useConditional)
    {
        clonedConditionalTerminator = initialBlock.getTerminator()->clone();
        rewriter.eraseOp(initialBlock.getTerminator());
    }

    mlir::Value bodyBlockArg = bodyLoopRegion->begin()->getArgument(0);
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

    if (useConditional)
    {
        rewriter.setInsertionPointToEnd(&initialBlock);
        rewriter.insert(clonedConditionalTerminator);
    }

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

    if (auto gpuMapAttr = affineForOp->getAttrOfType<mlir::DictionaryAttr>("accv_gpu_map"))
    {
        auto iv = affineForOp.getInductionVar();
        [[maybe_unused]] int64_t begin = affineForOp.getConstantLowerBound();
        [[maybe_unused]] int64_t end = affineForOp.getConstantUpperBound();
        int64_t step = affineForOp.getStep();

        auto procStr = gpuMapAttr.get("proc").cast<mlir::StringAttr>().getValue();
        auto bindingMap = gpuMapAttr.get("map").cast<mlir::AffineMapAttr>().getValue();

        auto processor = *symbolizeProcessor(procStr);
        auto affineScopeParent = affineForOp->getParentWithTrait<mlir::OpTrait::AffineScope>();
        auto& firstRegion = affineScopeParent->getRegion(0);
        auto& firstBlock = firstRegion.front();
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPoint(&firstBlock, firstBlock.begin());

        auto gpuValue = util::GetGPUIndex(processor, rewriter, loc, util::ResolveExecutionRuntime(affineForOp));
        auto replacementVal = rewriter.create<mlir::AffineApplyOp>(loc, bindingMap, mlir::ValueRange{ gpuValue });

        // We're going to replace all uses of the affine loop's induction variable with GPU hardware mapping instead, so
        // make the AffineForOp effectively a no-op

        // TODO : incorporate the grid/block dims to make this more like an unroll across the hardware dims
        //        rather than always setting it to [0,1)
        affineForOp.setStep(1);
        affineForOp.setConstantLowerBound(0);
        affineForOp.setConstantUpperBound(1);
        auto idxMap = AffineMap::get(1, 0, rewriter.getAffineDimExpr(0) * rewriter.getAffineConstantExpr(step));
        mlir::Value affineApplyResult = rewriter.create<mlir::AffineApplyOp>(loc, idxMap, ValueRange{ replacementVal });

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
    [[maybe_unused]] auto loc = op.getLoc();

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
        auto constantOp = rewriter.create<arith::ConstantOp>(loc, rewriter.getIntegerAttr(rewriter.getIndexType(), *dimensionSize));
        constantOp->setAttr(DimSizeOp::getIndexAttrName(), dimensionIndexAttr);
        rewriter.replaceOp(op, constantOp.getResult());
    }
    else
    {
        // If DimSizeOp doesn't have a parent or doesn't have one of the subdomain size or index order attributes, then treat it
        // as being outside the iteration domain and return 0
        auto constantOp = rewriter.create<arith::ConstantOp>(loc, rewriter.getIntegerAttr(rewriter.getIndexType(), 0));
        constantOp->setAttr(DimSizeOp::getIndexAttrName(), dimensionIndexAttr);
        rewriter.replaceOp(op, constantOp.getResult());
    }

    return success();
}

LogicalResult ReplaceSymbolicIndexOpPattern::matchAndRewrite(SymbolicIndexOp op, PatternRewriter& rewriter) const
{
    [[maybe_unused]] auto loc = util::GetLocation(rewriter, "ReplaceSymbolicIndexOpPattern", op.getLoc());

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

    auto zeroOp = rewriter.create<arith::ConstantOp>(loc, rewriter.getIntegerAttr(rewriter.getIndexType(), 0));
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

    if (!op1.hasConstantBounds() || !op2.hasConstantBounds())
        return failure();

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
    llvm::SmallVector<executionPlan::BeginCacheRegion> op2BeginRegionOps;
    llvm::SmallVector<executionPlan::EndCacheRegion> op2EndRegionOps;
    while (nextOp1 || nextOp2)
    {

        if (auto endOp1 = dyn_cast_or_null<executionPlan::EndCacheRegion>(nextOp1),
            endOp2 = dyn_cast_or_null<executionPlan::EndCacheRegion>(nextOp2);
            // if there's a mismatch, bail
            !!endOp1 ^ !!endOp2)
        {
            return failure();
        }
        else if (endOp1 && endOp2)
        {
            // if we have a mismatch in begin region ids, bail
            if (auto op2BeginRegionOp = mlir::dyn_cast<executionPlan::BeginCacheRegion>(endOp2.getBeginOp()); mlir::dyn_cast<executionPlan::BeginCacheRegion>(endOp1.getBeginOp()).getId() != op2BeginRegionOp.getId())
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
                auto beginOp1 = dyn_cast<executionPlan::BeginCreateCacheOp>(&op1Child),
                beginOp2 = dyn_cast<executionPlan::BeginCreateCacheOp>(&op2Child);
                // if there's a mismatch, bail
                !!beginOp1 ^ !!beginOp2)
            {
                return failure();
            }
            // if they both have a begin cache region op, make sure the id matches
            else if (beginOp1 && beginOp2)
            {
                if (beginOp1.getId() != beginOp2.getId())
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
    if (!op.hasConstantBounds())
        return failure();

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

void populateRangeResolutionPatterns(mlir::RewritePatternSet& patterns)
{
    mlir::MLIRContext* context = patterns.getContext();
    patterns.insert<ScheduleOpDomainResolution>(context);
}

void populateScheduleScaffoldingPatterns(bool printLoops, mlir::RewritePatternSet& patterns)
{
    mlir::MLIRContext* context = patterns.getContext();
    patterns.insert<ScheduleOpConversion>(context, printLoops);
}

void populateScheduledOperationsPatterns(mlir::RewritePatternSet& patterns)
{
    mlir::MLIRContext* context = patterns.getContext();
    patterns.insert<DimSizeOpConversion,
                    RemoveScheduledKernelOpPattern,
                    ScheduledLoopOpIndexConversion>(context);
}

void populateScheduleToValueRewritePatterns(mlir::RewritePatternSet& patterns)
{
    mlir::MLIRContext* context = patterns.getContext();
    patterns.insert<ScheduledLoopOpRewrite>(context);
}

void populateGPUIndexMappingRewritePatterns(mlir::RewritePatternSet& patterns)
{
    mlir::MLIRContext* context = patterns.getContext();
    patterns.insert<GPUMappedAffineForOpRewrite>(context);
}

void populateScheduleToValuePatterns(mlir::RewritePatternSet& patterns)
{
    mlir::MLIRContext* context = patterns.getContext();
    populateWithGenerated(patterns);
    patterns.insert<RemoveKernelOpPattern,
                    RemoveScheduledKernelOpPattern,
                    RemoveSymIndexOpPattern>(context);
}

void populateSymIndexCleanupPatterns(mlir::RewritePatternSet& patterns)
{
    mlir::MLIRContext* context = patterns.getContext();
    patterns.insert<RemoveSymIndexOpPattern>(context);
}

void populateLoopOptimizationPatterns(mlir::RewritePatternSet& patterns)
{
    mlir::MLIRContext* context = patterns.getContext();
    patterns.insert<LowPrecisionIntAccumulateLoopRewrite>(context);
}

void populateLoopMergingPatterns(mlir::RewritePatternSet& patterns)
{
    mlir::MLIRContext* context = patterns.getContext();
    patterns.insert<MergeNearlyIdenticalSiblingLoops>(context);
}

void populateLoopSimplificationPatterns(mlir::RewritePatternSet& patterns)
{
    mlir::MLIRContext* context = patterns.getContext();
    patterns.insert<RemoveZeroIterationLoopPattern,
                    PromoteSingleIterationLoopPattern>(context);
}

} // namespace accera::transforms
