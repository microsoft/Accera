////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "vectorization/VectorizationPass.h"

#include "AcceraPasses.h"
#include "vectorization/VectorizationUtil.h"

#include <ir/include/IRUtil.h>
#include <ir/include/value/ValueDialect.h>
#include <ir/include/exec/ExecutionPlanAttributes.h>

#include <utilities/include/MathUtil.h>

#include <mlir/Dialect/Affine/Analysis/LoopAnalysis.h>
#include <mlir/Dialect/Affine/LoopUtils.h>
#include <mlir/Dialect/Affine/Utils.h>
#include <mlir/IR/Operation.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#include <llvm/ADT/TypeSwitch.h>

#include <algorithm>
#include <memory>
#include <numeric>

using namespace accera::ir;
using namespace accera::ir::executionPlan;
using namespace accera::ir::value;
using namespace accera::ir::loopnest;
namespace v = accera::ir::value;
using namespace accera::transforms;
using namespace accera::ir::util;
using namespace accera::utilities;

using namespace mlir;

namespace
{

// Vectorization info functions
bool HasVectorizationInfo(Operation* op)
{
    auto vectorizationInfoAttr = op->getAttrOfType<VectorizationInfoAttr>(VectorizationInfoAttr::getKeyName());

    return vectorizationInfoAttr != nullptr;
}

VectorizationInfo GetVectorizationInfo(Operation* op)
{
    auto vectorizationInfoAttr = op->getAttrOfType<VectorizationInfoAttr>(VectorizationInfoAttr::getKeyName());
    assert(vectorizationInfoAttr != nullptr);

    return vectorizationInfoAttr.getValue();
}

void RemoveVectorizationInfo(Operation* op)
{
    OpBuilder builder(op);
    auto vectorizationInfoIdentifier = builder.getStringAttr(VectorizationInfoAttr::getKeyName());
    op->removeAttr(vectorizationInfoIdentifier);
}

// In-place-unroll-related functions

bool HasInPlaceUnrollInfo(Operation* op)
{
    auto inPlaceUnrollInfoAttr = op->getAttrOfType<InPlaceUnrollInfoAttr>(InPlaceUnrollInfoAttr::getKeyName());

    return inPlaceUnrollInfoAttr != nullptr;
}

InPlaceUnrollInfo GetInPlaceUnrollInfo(Operation* op)
{
    auto inPlaceUnrollInfoAttr = op->getAttrOfType<InPlaceUnrollInfoAttr>(InPlaceUnrollInfoAttr::getKeyName());
    assert(inPlaceUnrollInfoAttr != nullptr);

    return inPlaceUnrollInfoAttr.getValue();
}

[[maybe_unused]] void SetInPlaceUnrollInfo(Operation* op, const InPlaceUnrollInfo& inPlaceUnrollInfo)
{
    op->setAttr(InPlaceUnrollInfoAttr::getKeyName(), InPlaceUnrollInfoAttr::get(inPlaceUnrollInfo, op->getContext()));
}

[[maybe_unused]] void SetInPlaceUnrollInfo(ScheduleOp op, Index index, const InPlaceUnrollInfo& inPlaceUnrollInfo)
{
    OpBuilder builder(op);
    auto inPlaceUnrollInfoIdentifier = builder.getStringAttr(InPlaceUnrollInfoAttr::getKeyName());
    op.addLoopAttribute(index, inPlaceUnrollInfoIdentifier, InPlaceUnrollInfoAttr::get(inPlaceUnrollInfo, builder.getContext()));
}

[[maybe_unused]] void SetInPlaceUnrollInfo(ScheduleOp op, SymbolicIndexOp index, const InPlaceUnrollInfo& inPlaceUnrollInfo)
{
    SetInPlaceUnrollInfo(op, index.getValue(), inPlaceUnrollInfo);
}

void RemoveInPlaceUnrollInfo(Operation* op)
{
    OpBuilder builder(op);
    auto inPlaceUnrollInfoIdentifier = builder.getStringAttr(InPlaceUnrollInfoAttr::getKeyName());
    op->removeAttr(inPlaceUnrollInfoIdentifier);
}

struct VectorizeAffineForOpConversion : public OpRewritePattern<AffineForOp>
{
    using OpRewritePattern<AffineForOp>::OpRewritePattern;
    VectorizeAffineForOpConversion(MLIRContext* context, bool printVectorizationDetails = false) :
        OpRewritePattern(context, /* benefit */ 1),
        printVectorizationDetails(printVectorizationDetails)
    {}

    LogicalResult matchAndRewrite(AffineForOp affineForOp, PatternRewriter& rewriter) const final;

    // status-reporting helper methods
    void emitVectorizationRemark(mlir::Operation* sourceOp, const std::string& remark) const;
    void didVectorizeOp(mlir::Operation* sourceOp, VectorizedOp& vectorizedOp) const;
    void vectorizeOpsInBlock(PatternRewriter& rewriter,
                             mlir::Block::iterator begin,
                             mlir::Block::iterator end,
                             mlir::Value unrollingIV,
                             const VectorizationInfo& vectorInfo,
                             VectorizedOpMap& vectorizedOps,
                             std::vector<BlockAndValueMapping>& laneMappings,
                             int64_t step,
                             int64_t unrollMax) const;

    bool printVectorizationDetails = false;
};

struct InPlaceUnrollAffineForOpConversion : public OpRewritePattern<AffineForOp>
{
    using OpRewritePattern<AffineForOp>::OpRewritePattern;
    InPlaceUnrollAffineForOpConversion(MLIRContext* context, bool printVectorizationDetails = false) :
        OpRewritePattern(context, /* benefit */ 1),
        printVectorizationDetails(printVectorizationDetails)
    {}

    LogicalResult matchAndRewrite(AffineForOp affineForOp, PatternRewriter& rewriter) const final;

    bool printVectorizationDetails = false;
};




void VectorizeAffineForOpConversion::vectorizeOpsInBlock(PatternRewriter& rewriter,
                                                         mlir::Block::iterator begin,
                                                         mlir::Block::iterator endPrevSentinel,
                                                         mlir::Value unrollingIV,
                                                         const VectorizationInfo& vectorInfo,
                                                         VectorizedOpMap& vectorizedOps,
                                                         std::vector<BlockAndValueMapping>& laneMappings,
                                                         int64_t step,
                                                         int64_t unrollMax) const
{
    std::stack<Operation*> opsToErase;
    // Note: this loop needs to check std::next(endPrevSentinel) on every iteration since the vectorized ops are being inserted
    //       in the same block that this iterator is traversing, so std::next(endPrevSentinel) is initially the terminator op,
    //       but the new ops get inserted before the terminator op so std::next(endPrevSentinel) will change
    for (auto it = begin; it != std::next(endPrevSentinel); it++)
    {
        Operation* sourceOp = &(*it);

        // If this op can be vectorized, do it
        // Clone the op and then delete it if we were successful in vectorizing it.
        // When cloning, use a BlockAndValueMapping to remap the induction variable
        if (!vectorInfo.unrollOnly && CanVectorizeOp(sourceOp, vectorizedOps, laneMappings, unrollingIV, step, unrollMax))
        {
            auto result = VectorizeOp(rewriter, sourceOp, vectorizedOps, laneMappings, unrollingIV, step, unrollMax);
            if (result.has_value())
            {
                vectorizedOps.Map(sourceOp, *result);
                didVectorizeOp(sourceOp, *result);
            }
        }

        emitVectorizationRemark(sourceOp, "Unrolling op if needed");

        // Unroll the contents of 'forOpToUnroll' by replacing its contents with vectorSize mapped copies of it.
        for (int64_t unrollIdx = 0; unrollIdx < unrollMax; unrollIdx++)
        {
            auto& operandMap = laneMappings[unrollIdx];
            if (unrollIdx == 0)
            {
                opsToErase.push(sourceOp);
            }

            if (util::IsTerminalOp(sourceOp) && vectorizedOps.Lookup(sourceOp))
            {
                // this op has already been vectorized, and nothing else depends on it, so don't do anything
                emitVectorizationRemark(sourceOp, "Terminal op, vectorized");
            }
            else
            {
                if (util::IsTerminalOp(sourceOp))
                {
                    emitVectorizationRemark(sourceOp, "Terminal op, not vectorized");
                }

                [[maybe_unused]] auto mappedClonedOp = rewriter.clone(*it, operandMap);
            }
        }
    }

    while (!opsToErase.empty())
    {
        auto eraseOp = opsToErase.top();
        if (eraseOp->use_empty())
        {
            rewriter.eraseOp(eraseOp);
        }
        opsToErase.pop();
    }
}

LogicalResult VectorizeAffineForOpConversion::matchAndRewrite(AffineForOp affineForOp, PatternRewriter& rewriter) const
{
    if (!HasVectorizationInfo(affineForOp))
    {
        // This isn't an AffineForOp marked for vectorization so just return without modifying it
        return failure();
    }

    if (!affineForOp.hasConstantBounds())
    {
        // Dynamically-sized loops can't be vectorized
        RemoveVectorizationInfo(affineForOp);
        return failure();
    }

    // First, check if we have a custom match and rewrite pattern for this exact subgraph
    auto knownSubgraphResult = TryVectorizeKnownSubgraph(affineForOp, rewriter);
    if (succeeded(knownSubgraphResult))
    {
        RemoveVectorizationInfo(affineForOp);
        return knownSubgraphResult;
    }

    auto vectorInfo = GetVectorizationInfo(affineForOp);

    // Enforce some simplifying assumptions about the affine for loop:
    //  - the loop must have a constant trip count
    //  - the loop must have a constant lower bound
    //  - the loop must have a constant upper bound
    // TODO : eventually we'll want to relax these requirements
    auto mayBeConstantTripCount = mlir::getConstantTripCount(affineForOp);
    assert(mayBeConstantTripCount.hasValue() && "Vectorized loops must have a constant trip count");
    uint64_t constantTripCount = mayBeConstantTripCount.getValue();
    if (constantTripCount == 0)
    {
        // Discard loops that never run
        rewriter.eraseOp(affineForOp);
        return success();
    }

    // If this isn't the innermost loop in the nest and we don't have custom handling for this pattern,
    // then in-place unroll the loops between this loop and the innermost loop and vectorize the innermost loop
    SmallVector<AffineForOp, 4> nestedLoops;
    mlir::getPerfectlyNestedLoops(nestedLoops, affineForOp);
    if (nestedLoops.size() > 1)
    {
        RemoveVectorizationInfo(affineForOp);
        for (unsigned loopIdx = 0; loopIdx < nestedLoops.size() - 1; loopIdx++)
        {
            InPlaceUnrollInfo inPlaceUnrollInfo{ 0 }; // 0 for full unroll
            SetInPlaceUnrollInfo(nestedLoops[loopIdx], inPlaceUnrollInfo);
        }
        auto vecInfoAttr = VectorizationInfoAttr::get(vectorInfo, rewriter.getContext());
        nestedLoops[nestedLoops.size() - 1]->setAttr(VectorizationInfoAttr::getKeyName(), vecInfoAttr);
        return failure();
    }

    auto affineForOpIV = affineForOp.getInductionVar();

    if (affineForOpIV.use_empty())
    {
        // Don't vectorize loops that never uses the induction variable
        return success();
    }

    rewriter.startRootUpdate(affineForOp);

    assert(affineForOp.hasConstantLowerBound() && "Vectorized loops must have a constant lower bound");
    assert(affineForOp.hasConstantUpperBound() && "Vectorized loops must have a constant upper bound");

    // Unroll this AffineForOp and replace the appropriate CacheLoads and CacheStores with VectorizedCacheLoad and VectorizedCacheStore

    // this is a partial port of the meaty bits of mlir::loopUnrollByFactor() from mlir/lib/Dialect/Affine/Utils/LoopUtils.cpp
    // but with access to the unroll indices in order to make VectorizedCacheLoad and VectorizedCacheStore
    // and with some more simplifying assumptions and op replacements

    // remove the vectorization attribute from the AffineForOp
    RemoveVectorizationInfo(affineForOp);

    bool erasedBaseLoop = false;
    int64_t step = affineForOp.getStep();

    // Scale the step of loop being unrolled by unroll factor.
    auto numIters = CeilDiv(affineForOp.getConstantUpperBound() - affineForOp.getConstantLowerBound(), affineForOp.getStep());
    int64_t unrollMax = std::min(affineForOp.getConstantUpperBound() - affineForOp.getConstantLowerBound(), numIters);
    affineForOp.setStep(step * numIters);

    // Insert unrolled bodies just before the terminator of the body of 'affineForOp'.
    rewriter.setInsertionPoint(affineForOp.getBody(), affineForOp.getBody()->getTerminator()->getIterator());

    // Keep a pointer to the last non-terminator operation in the original block
    // so that we know what to clone (since we are doing this in-place).
    Block::iterator srcBlockEnd = std::prev(affineForOp.getBody()->end(), 2);

    VectorizedOpMap vectorizedOps;
    std::vector<BlockAndValueMapping> laneMappings(unrollMax);

    if (!affineForOpIV.use_empty())
    {
        // Initialize the mappings with an offset version of the induction variable
        auto loc = affineForOp.getLoc();
        auto inductionVarMap = AffineMap::get(1, 1, rewriter.getAffineDimExpr(0) + step * rewriter.getAffineSymbolExpr(0));
        for (int64_t i = 0; i < unrollMax; ++i)
        {
            auto offset = rewriter.create<arith::ConstantIndexOp>(loc, i);
            auto offsetInductionVar = rewriter.create<AffineApplyOp>(loc, inductionVarMap, ValueRange{ affineForOpIV, offset });

            BlockAndValueMapping& operandMap = laneMappings[i];
            operandMap.map(affineForOpIV, offsetInductionVar);
        }
    }

    vectorizeOpsInBlock(rewriter, affineForOp.getBody()->begin(), srcBlockEnd, affineForOpIV, vectorInfo, vectorizedOps, laneMappings, step, unrollMax);

    if (!erasedBaseLoop)
    {
        (void)util::PromoteIfSingleIteration(rewriter, affineForOp);
    }

    rewriter.finalizeRootUpdate(affineForOp);

    return success();
}

void VectorizeAffineForOpConversion::didVectorizeOp(mlir::Operation* sourceOp, VectorizedOp& vectorizedOp) const
{
    if (printVectorizationDetails)
    {
        auto diagnostic = sourceOp->emitRemark("Vectorized");
        if (vectorizedOp.HasVectorType())
        {
            auto vecResult = vectorizedOp.GetVectorResult();
            if (vecResult && vecResult.getDefiningOp())
            {
                diagnostic << " -- " << vecResult.getDefiningOp();
            }
            else if (auto resultOp = vectorizedOp.GetOp())
            {
                diagnostic << " -- terminal op: " << resultOp;
            }
            else
            {
                diagnostic << " -- terminal op";
            }
        }
    }

    if (!vectorizedOp.HasVectorType())
    {
        // also add to vectorized ops?
        if (printVectorizationDetails)
        {
            sourceOp->emitRemark("Vectorized to a non-vector type");
        }
    }
}

void VectorizeAffineForOpConversion::emitVectorizationRemark(mlir::Operation* sourceOp, const std::string& remark) const
{
    if (printVectorizationDetails)
    {
        sourceOp->emitRemark(remark);
    }
}

// TODO : de-dupe with vectorization
void InPlaceUnrollOpsInBlock(PatternRewriter& rewriter,
                             mlir::Block::iterator begin,
                             mlir::Block::iterator endPrevSentinel,
                             mlir::Value unrollingIV,
                             const InPlaceUnrollInfo& inPlaceUnrollInfo,
                             VectorizedOpMap& vectorizedOps,
                             std::vector<BlockAndValueMapping>& laneMappings,
                             int64_t step,
                             int64_t unrollMax)
{
    std::stack<Operation*> opsToErase;
    // Note: this loop needs to check std::next(endPrevSentinel) on every iteration since the unrolled ops are being inserted
    //       in the same block that this iterator is traversing, so std::next(endPrevSentinel) is initially the terminator op,
    //       but the new ops get inserted before the terminator op so std::next(endPrevSentinel) will change
    for (auto it = begin; it != std::next(endPrevSentinel); it++)
    {
        Operation* sourceOp = &(*it);

        // If this op is an AffineForOp that is also being in-place-unrolled or vectorized, then in-place unroll the ops inside it from the point of view of this unrollingIV without
        // unrolling/vectorizing the AffineForOp itself
        if (auto innerForOp = mlir::dyn_cast<mlir::AffineForOp>(sourceOp); innerForOp && (HasVectorizationInfo(innerForOp) || HasInPlaceUnrollInfo(innerForOp)))
        {
            // In-place unroll the ops inside this loop, but don't unroll the loop terminator
            // Get a sentinel op for where we should stop vectorizing by stepping back from the end of the for loop by stepping back 2 ops from the end of the body:
            // - stepping back 1 op from the end would get us the terminator op, which will move as we insert the new vectorized ops before the terminator
            // - stepping back 2 ops from the end will get us the last original op that should be vectorized, so we can check if the iterator == std::next(innerLoopBlockEndSentinel) to determine when we've gone too far
            Block::iterator innerLoopBlockEndSentinel = std::prev(innerForOp.getBody()->end(), 2);
            mlir::OpBuilder::InsertionGuard insertGuard(rewriter);
            rewriter.setInsertionPoint(innerForOp.getBody(), innerForOp.getBody()->getTerminator()->getIterator());
            InPlaceUnrollOpsInBlock(rewriter, innerForOp.getBody()->begin(), innerLoopBlockEndSentinel, unrollingIV, inPlaceUnrollInfo, vectorizedOps, laneMappings, step, unrollMax);
            continue;
        }

        // Unroll the contents of 'forOpToUnroll' by replacing its contents with vectorSize mapped copies of it.
        for (int64_t unrollIdx = 0; unrollIdx < unrollMax; unrollIdx++)
        {
            auto& operandMap = laneMappings[unrollIdx];
            if (unrollIdx == 0)
            {
                opsToErase.push(sourceOp);
            }

            if (!(util::IsTerminalOp(sourceOp) && vectorizedOps.Lookup(sourceOp)))
            {
                [[maybe_unused]] auto mappedClonedOp = rewriter.clone(*it, operandMap);
            }
        }
    }

    while (!opsToErase.empty())
    {
        auto eraseOp = opsToErase.top();
        if (eraseOp->use_empty())
        {
            rewriter.eraseOp(eraseOp);
        }
        opsToErase.pop();
    }
}

LogicalResult InPlaceUnrollAffineForOpConversion::matchAndRewrite(AffineForOp affineForOp, PatternRewriter& rewriter) const
{
    if (!HasInPlaceUnrollInfo(affineForOp))
    {
        // This isn't an AffineForOp marked for in-place unroll so just return without modifying it
        return failure();
    }

    auto inPlaceUnrollInfo = GetInPlaceUnrollInfo(affineForOp);
    bool fullyUnroll = (inPlaceUnrollInfo.loopUnrollFactor == 0);

    // Enforce some simplifying assumptions about the affine for loop:
    //  - the loop must have a constant trip count
    //  - the loop must have a constant lower bound
    //  - the loop must have a constant upper bound
    // TODO : eventually we'll want to relax these requirements
    auto mayBeConstantTripCount = mlir::getConstantTripCount(affineForOp);
    assert(mayBeConstantTripCount.hasValue() && "Vectorized loops must have a constant trip count");
    uint64_t constantTripCount = mayBeConstantTripCount.getValue();
    if (constantTripCount == 0)
    {
        // Discard loops that never run
        rewriter.eraseOp(affineForOp);
        return success();
    }
    int64_t loopUnrollFactor = fullyUnroll ? constantTripCount : inPlaceUnrollInfo.loopUnrollFactor;

    auto originalInductionVar = affineForOp.getInductionVar();

    if (originalInductionVar.use_empty())
    {
        // Don't unroll loops that never uses the induction variable
        return success();
    }

    rewriter.startRootUpdate(affineForOp);

    assert(affineForOp.hasConstantLowerBound() && "In-place unrolled loops must have a constant lower bound");
    assert(affineForOp.hasConstantUpperBound() && "In-place unrolled loops must have a constant upper bound");

    // remove the in-place unroll attribute from the AffineForOp
    RemoveInPlaceUnrollInfo(affineForOp);

    std::vector<AffineForOp> forOpsToUnroll;

    bool erasedBaseLoop = false;
    int64_t step = affineForOp.getStep();

    // Generate the cleanup loop if trip count isn't a multiple of loopUnrollFactor
    auto cleanupIterations = constantTripCount % loopUnrollFactor;
    if (cleanupIterations != 0)
    {
        rewriter.setInsertionPoint(affineForOp.getOperation()->getBlock(),
                                   std::next(Block::iterator(affineForOp)));
        auto cleanupForOp = cast<AffineForOp>(rewriter.clone(*affineForOp));

        // Compute lower bound of the cleanup loop. The (new) base loop has (constantTripCount-cleanupIterations) iterations,
        // for a total extent of (constantTripCount-cleanupIterations) * step.
        int64_t originalLowerBound = affineForOp.hasConstantLowerBound() ? affineForOp.getConstantLowerBound() : 0; // handle the case where the base loop doesn't start at 0
        int64_t cleanupLowerBound = originalLowerBound + ((constantTripCount - cleanupIterations) * step);
        cleanupForOp.setConstantLowerBound(cleanupLowerBound);

        // Adjust upper bound of the original loop; this is the same as the lower
        // bound of the cleanup loop.
        affineForOp.setConstantUpperBound(cleanupLowerBound);

        // If the non-cleanup loop now has 0 iterations, erase it, otherwise enqueue it to be unrolled
        auto adjustedMayBeConstantTripCount = mlir::getConstantTripCount(affineForOp);
        assert(adjustedMayBeConstantTripCount.hasValue() && "In-place unrolled loops must have a constant trip count");
        uint64_t adjustedConstantTripCount = adjustedMayBeConstantTripCount.getValue();
        if (adjustedConstantTripCount == 0)
        {
            rewriter.eraseOp(affineForOp);
            erasedBaseLoop = true;
        }
        else
        {
            forOpsToUnroll.push_back(affineForOp);
        }

        forOpsToUnroll.push_back(cleanupForOp);
    }
    else
    {
        forOpsToUnroll.push_back(affineForOp);
    }

    for (auto& forOpToUnroll : forOpsToUnroll)
    {
        // Scale the step of loop being unrolled by unroll factor.
        auto numIters = CeilDiv(forOpToUnroll.getConstantUpperBound() - forOpToUnroll.getConstantLowerBound(), forOpToUnroll.getStep());
        auto thisLoopUnrollFactor = std::min(numIters, loopUnrollFactor);
        forOpToUnroll.setStep(step * thisLoopUnrollFactor);

        // Insert unrolled bodies just before the terminator of the body of 'forOpToUnroll'.
        rewriter.setInsertionPoint(forOpToUnroll.getBody(), forOpToUnroll.getBody()->getTerminator()->getIterator());

        // Keep a pointer to the last non-terminator operation in the original block
        // so that we know what to clone (since we are doing this in-place).
        Block::iterator srcBlockEnd = std::prev(forOpToUnroll.getBody()->end(), 2);

        auto forOpToUnrollIV = forOpToUnroll.getInductionVar();

        // Clean up some ops after we've unrolled and mapped everything
        std::stack<Operation*> opsToErase;
        std::map<Operation*, Operation*> cacheLoadReplacementMapping; // TODO : figure out why BlockAndValueMapping isn't handling these cases

        int64_t unrollMax = std::min(forOpToUnroll.getConstantUpperBound() - forOpToUnroll.getConstantLowerBound(), thisLoopUnrollFactor);

        VectorizedOpMap vectorizedOps;
        std::vector<BlockAndValueMapping> laneMappings(unrollMax);

        if (!forOpToUnrollIV.use_empty())
        {
            // Initialize the mappings with an offset version of the induction variable
            auto loc = forOpToUnroll.getLoc();
            auto inductionVarMap = AffineMap::get(1, 1, rewriter.getAffineDimExpr(0) + step * rewriter.getAffineSymbolExpr(0));
            for (int64_t i = 0; i < unrollMax; ++i)
            {
                auto offset = rewriter.create<arith::ConstantIndexOp>(loc, i);
                auto offsetInductionVar = rewriter.create<AffineApplyOp>(loc, inductionVarMap, ValueRange{ forOpToUnrollIV, offset });

                BlockAndValueMapping& operandMap = laneMappings[i];
                operandMap.map(forOpToUnrollIV, offsetInductionVar);
            }
        }

        InPlaceUnrollOpsInBlock(rewriter, forOpToUnroll.getBody()->begin(), srcBlockEnd, forOpToUnrollIV, inPlaceUnrollInfo, vectorizedOps, laneMappings, step, unrollMax);
    }

    if (!erasedBaseLoop)
    {
        (void)util::PromoteIfSingleIteration(rewriter, affineForOp);
    }

    rewriter.finalizeRootUpdate(affineForOp);

    return success();
}


// TODO : implement
// struct VectorizationPass : public accera::transforms::AcceraVectorizationPassBase<VectorizationPass>
// {
//     void runOnOperation() final
//     {
//         auto* context = &getContext();
//         auto op = getOperation();
//         // TODO : implement with LoopNestToValueFunc vectorization sequence
//     }
// };

} // namespace

namespace accera::transforms::vectorization
{

void populateVectorizePatterns(bool printVectorizationDetails, mlir::RewritePatternSet& patterns)
{
    patterns.insert<VectorizeAffineForOpConversion>(patterns.getContext(), printVectorizationDetails);
}
void populateVectorizeUnrollPatterns(bool printVectorizationDetails, mlir::RewritePatternSet& patterns)
{
    patterns.insert<InPlaceUnrollAffineForOpConversion>(patterns.getContext(), printVectorizationDetails);
}

// TODO : implement
// std::unique_ptr<mlir::Pass> createVectorizationPass()
// {
//     return std::make_unique<VectorizationPass>();
// }
} // namespace accera::transforms::vectorization
