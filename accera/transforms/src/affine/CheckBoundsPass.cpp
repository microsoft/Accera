////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "affine/CheckBoundsPass.h"

#include "AcceraPasses.h"

#include <ir/include/IRUtil.h>
#include <ir/include/value/ValueDialect.h>
#include <ir/include/exec/VectorizationInfo.h>

#include <mlir/Dialect/Affine/Analysis/AffineStructures.h>
#include <mlir/Dialect/Affine/Analysis/LoopAnalysis.h>
#include <mlir/Dialect/Affine/Analysis/Utils.h>
#include <mlir/IR/IntegerSet.h>
#include <mlir/IR/Operation.h>

using namespace accera;
namespace v = accera::ir::value;

namespace
{

bool IsBoundsChecked(mlir::Operation* op)
{
    return op->getAttr(transforms::affine::BoundsCheckedAttrName) != nullptr;
}

void SetBoundsChecked(mlir::OpBuilder& builder, mlir::Operation* op)
{
    op->setAttr(transforms::affine::BoundsCheckedAttrName, builder.getUnitAttr());
}

template <typename LoadOrStoreOp>
bool HasOutOfBoundsAccess(LoadOrStoreOp op, mlir::Location loc)
{
    // This is a pared down version of mlir::boundCheckLoadOrStoreOp, which has a bug currently where it only returns failure (out of bounds)
    // if the last thing it checks has a failure, rather than anything it checks.

    mlir::MemRefRegion accessRegion(loc);
    auto memRefType = op.getMemRefType();
    unsigned rank = memRefType.getRank();
    (void)accessRegion.compute(op, 0, nullptr /*sliceState*/, false /*addMemRefDimBounds */);
    bool outOfBounds = false;

    // TODO : handle dynamic dimension out of bounds checks generically
    if (!memRefType.hasStaticShape())
    {
        return false;
    }

    // For each dimension, check for out of bounds.
    for (unsigned dim = 0; dim < rank; ++dim)
    {
        // Intersect memory region with constraint capturing out of bounds (both out
        // of upper and out of lower), and check if the constraint system is
        // feasible. If it is, there is at least one point out of bounds.

        // Check for overflow: d_i >= memref dim size.
        mlir::FlatAffineValueConstraints upperConstraints(*accessRegion.getConstraints());
        int64_t dimSize = memRefType.getDimSize(dim);
        upperConstraints.addBound(mlir::FlatAffineConstraints::LB, dim, dimSize);

        // Check for a negative index: d_i <= -1.
        mlir::FlatAffineValueConstraints lowerConstraints(*accessRegion.getConstraints());
        lowerConstraints.addBound(mlir::FlatAffineConstraints::UB, dim, -1);

        if (!upperConstraints.isEmpty() || !lowerConstraints.isEmpty())
        {
            outOfBounds = true;
            break;
        }
    }
    return outOfBounds;
}

struct OutOfBoundsLoadRewrite : public mlir::OpRewritePattern<mlir::memref::LoadOp>
{
    using mlir::OpRewritePattern<mlir::memref::LoadOp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(mlir::memref::LoadOp loadOp, mlir::PatternRewriter& rewriter) const final;
};

struct OutOfBoundsAffineLoadRewrite : public mlir::OpRewritePattern<mlir::AffineLoadOp>
{
    using mlir::OpRewritePattern<mlir::AffineLoadOp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(mlir::AffineLoadOp affineLoadOp, mlir::PatternRewriter& rewriter) const final;
};

struct OutOfBoundsStoreRewrite : public mlir::OpRewritePattern<mlir::memref::StoreOp>
{
    using mlir::OpRewritePattern<mlir::memref::StoreOp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(mlir::memref::StoreOp toreOp, mlir::PatternRewriter& rewriter) const final;
};

struct OutOfBoundsAffineStoreRewrite : public mlir::OpRewritePattern<mlir::AffineStoreOp>
{
    using mlir::OpRewritePattern<mlir::AffineStoreOp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(mlir::AffineStoreOp affineStoreOp, mlir::PatternRewriter& rewriter) const final;
};

mlir::LogicalResult OutOfBoundsLoadRewriteCommon(mlir::AffineLoadOp affineLoadOp, mlir::PatternRewriter& rewriter)
{
    if (IsBoundsChecked(affineLoadOp))
    {
        return mlir::success();
    }
    auto loc = affineLoadOp.getLoc();
    mlir::AffineLoadOp::Adaptor adaptor{ affineLoadOp };

    if (HasOutOfBoundsAccess(affineLoadOp, loc))
    {
        // This load has a potential out-of-bounds access, so replace it with a conditional load

        auto accessMapAttr = affineLoadOp.getAffineMapAttr();
        auto accessMap = accessMapAttr.getValue();
        auto loadSrc = affineLoadOp.memref();
        auto loadSrcType = loadSrc.getType();
        assert(loadSrcType.isa<mlir::MemRefType>());
        auto memRefType = loadSrcType.cast<mlir::MemRefType>();

        auto loadResultType = affineLoadOp.result().getType();

        std::vector<mlir::AffineExpr> constraintExprs;
        constraintExprs.reserve(accessMap.getNumResults() * 2); // One lower bound and one upper bound check per src dimension
        std::vector<mlir::Value> accessIndices(adaptor.indices().begin(), adaptor.indices().end());
        auto resolvedAccessIndices = ir::util::MultiDimAffineApply(rewriter, loc, accessMap, accessIndices);
        mlir::SmallVector<bool, 4> constraintEqFlags(accessMap.getNumResults() * 2, false);
        for (size_t srcDim = 0; srcDim < accessMap.getNumResults(); srcDim++)
        {
            // Lower bound check
            constraintExprs.push_back(rewriter.getAffineDimExpr(srcDim)); // Will check whether this index is >= 0

            // Upper bound check
            constraintExprs.push_back(memRefType.getDimSize(srcDim) - rewriter.getAffineDimExpr(srcDim) - rewriter.getAffineConstantExpr(1)); // Will check whether (this dimSize - this index - 1) >= 0 (note: -1 since we're doing a >= check with 0-based indices)
        }

        std::vector<int64_t> tmpBufferShape{ 1 }; // only one element of type loadResultType
        mlir::MemRefType tmpElementType;
        std::optional<v::ExecutionTarget> execTargetOpt = ir::util::ResolveExecutionTarget(affineLoadOp);
        assert(execTargetOpt.has_value());
        auto execTarget = *execTargetOpt;
        mlir::Value tmpBuffer;
        if (execTarget == v::ExecutionTarget::GPU)
        {
            tmpElementType = mlir::MemRefType::get(tmpBufferShape, loadResultType, {}, static_cast<unsigned>(v::MemorySpace::Private));
            tmpBuffer = rewriter.create<v::AllocOp>(loc, tmpElementType, llvm::None);
        }
        else
        {
            tmpElementType = mlir::MemRefType::get(tmpBufferShape, loadResultType);
            tmpBuffer = rewriter.create<mlir::memref::AllocaOp>(loc, tmpElementType, mlir::ValueRange{}, rewriter.getI64IntegerAttr(ir::executionPlan::AVX2Alignment));
        }

        auto zeroIndex = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 0);

        auto srcBoundsCheckSet = mlir::IntegerSet::get(resolvedAccessIndices.size(), 0, constraintExprs, constraintEqFlags);
        auto ifOp = rewriter.create<mlir::AffineIfOp>(loc, srcBoundsCheckSet, mlir::ValueRange{ resolvedAccessIndices }, true); // true indicating we want an "else" region

        auto thenBuilder = ifOp.getThenBodyBuilder();
        auto newLoadOp = thenBuilder.create<mlir::AffineLoadOp>(loc, loadSrc, accessMap, accessIndices);
        SetBoundsChecked(thenBuilder, newLoadOp);

        auto thenStoreOp = thenBuilder.create<mlir::memref::StoreOp>(loc, newLoadOp.getResult(), tmpBuffer, mlir::ValueRange{ zeroIndex });
        SetBoundsChecked(thenBuilder, thenStoreOp);

        auto elseBuilder = ifOp.getElseBodyBuilder();
        // TODO : support user-specified padding value rather than always using 0
        auto constantZero = elseBuilder.create<mlir::arith::ConstantOp>(loc, elseBuilder.getZeroAttr(loadResultType));
        auto elseStoreOp = elseBuilder.create<mlir::memref::StoreOp>(loc, constantZero.getResult(), tmpBuffer, mlir::ValueRange{ zeroIndex });
        SetBoundsChecked(elseBuilder, elseStoreOp);

        auto tmpSlotLoad = rewriter.create<mlir::memref::LoadOp>(loc, tmpBuffer, mlir::ValueRange{ zeroIndex });
        SetBoundsChecked(rewriter, tmpSlotLoad);

        affineLoadOp.replaceAllUsesWith(tmpSlotLoad.getResult());
        rewriter.eraseOp(affineLoadOp);
    }

    return mlir::success();
}

mlir::LogicalResult OutOfBoundsLoadRewrite::matchAndRewrite(mlir::memref::LoadOp loadOp, mlir::PatternRewriter& rewriter) const
{
    // Only check for out-of-bounds-accesses inside of ops that are marked for bounds checking
    if (!ir::util::AncestorOpContainsAttrOfName(loadOp, transforms::affine::AccessBoundsCheckAttrName))
    {
        return mlir::success();
    }

    if (IsBoundsChecked(loadOp))
    {
        return mlir::success();
    }
    // Convert std.load to affine.load with an identity map
    auto loc = loadOp.getLoc();
    mlir::memref::LoadOp::Adaptor adaptor{ loadOp };
    auto memRefType = adaptor.memref().getType().cast<mlir::MemRefType>();
    auto affineLoadOp = rewriter.create<mlir::AffineLoadOp>(loc, adaptor.memref(), rewriter.getMultiDimIdentityMap(memRefType.getRank()), adaptor.indices());
    loadOp.replaceAllUsesWith(affineLoadOp.getResult());
    auto result = OutOfBoundsLoadRewriteCommon(affineLoadOp, rewriter);
    rewriter.eraseOp(loadOp);
    return result;
}

mlir::LogicalResult OutOfBoundsAffineLoadRewrite::matchAndRewrite(mlir::AffineLoadOp affineLoadOp, mlir::PatternRewriter& rewriter) const
{
    // Only check for out-of-bounds-accesses inside of ops that are marked for bounds checking
    if (!ir::util::AncestorOpContainsAttrOfName(affineLoadOp, transforms::affine::AccessBoundsCheckAttrName))
    {
        return mlir::success();
    }

    return OutOfBoundsLoadRewriteCommon(affineLoadOp, rewriter);
}

mlir::LogicalResult OutOfBoundsStoreRewriteCommon(mlir::AffineStoreOp affineStoreOp, mlir::PatternRewriter& rewriter)
{
    if (IsBoundsChecked(affineStoreOp))
    {
        return mlir::success();
    }

    auto loc = affineStoreOp.getLoc();
    mlir::AffineStoreOp::Adaptor adaptor{ affineStoreOp };

    if (HasOutOfBoundsAccess(affineStoreOp, loc))
    {
        // This store has a potential out-of-bounds access, so replace it with a conditional store

        auto accessMapAttr = affineStoreOp.getAffineMapAttr();
        auto accessMap = accessMapAttr.getValue();
        auto storeDst = affineStoreOp.memref();
        auto storeDstType = storeDst.getType();
        assert(storeDstType.isa<mlir::MemRefType>());
        auto memRefType = storeDstType.cast<mlir::MemRefType>();

        // TODO : de-dupe affine.if constraint code with load case
        std::vector<mlir::AffineExpr> constraintExprs;
        constraintExprs.reserve(accessMap.getNumResults() * 2); // One lower bound and one upper bound check per src dimension
        std::vector<mlir::Value> accessIndices(adaptor.indices().begin(), adaptor.indices().end());
        auto resolvedAccessIndices = ir::util::MultiDimAffineApply(rewriter, loc, accessMap, accessIndices);
        mlir::SmallVector<bool, 4> constraintEqFlags(accessMap.getNumResults() * 2, false);
        for (size_t srcDim = 0; srcDim < accessMap.getNumResults(); srcDim++)
        {
            // Lower bound check
            constraintExprs.push_back(rewriter.getAffineDimExpr(srcDim)); // Will check whether this index is >= 0

            // Upper bound check
            constraintExprs.push_back(memRefType.getDimSize(srcDim) - rewriter.getAffineDimExpr(srcDim) - rewriter.getAffineConstantExpr(1)); // Will check whether (this dimSize - this index - 1) >= 0 (note: -1 since we're doing a >= check with 0-based indices)
        }

        auto srcBoundsCheckSet = mlir::IntegerSet::get(resolvedAccessIndices.size(), 0, constraintExprs, constraintEqFlags);
        auto ifOp = rewriter.create<mlir::AffineIfOp>(loc, srcBoundsCheckSet, mlir::ValueRange{ resolvedAccessIndices }, true); // true indicating we want an "else" region

        auto thenBuilder = ifOp.getThenBodyBuilder();
        auto newStoreOp = thenBuilder.create<mlir::AffineStoreOp>(loc, affineStoreOp.value(), affineStoreOp.memref(), accessMap, accessIndices);
        SetBoundsChecked(thenBuilder, newStoreOp);

        rewriter.eraseOp(affineStoreOp);
    }

    return mlir::success();
}

mlir::LogicalResult OutOfBoundsStoreRewrite::matchAndRewrite(mlir::memref::StoreOp storeOp, mlir::PatternRewriter& rewriter) const
{
    // Only check for out-of-bounds-accesses inside of ops that are marked for bounds checking
    if (!ir::util::AncestorOpContainsAttrOfName(storeOp, transforms::affine::AccessBoundsCheckAttrName))
    {
        return mlir::success();
    }

    if (IsBoundsChecked(storeOp))
    {
        return mlir::success();
    }
    // Convert std.store to affine.store with an identity map
    auto loc = storeOp.getLoc();
    mlir::memref::StoreOp::Adaptor adaptor{ storeOp };
    auto memRefType = adaptor.memref().getType().cast<mlir::MemRefType>();
    auto affineStoreOp = rewriter.create<mlir::AffineStoreOp>(loc, adaptor.value(), adaptor.memref(), rewriter.getMultiDimIdentityMap(memRefType.getRank()), adaptor.indices());
    auto result = OutOfBoundsStoreRewriteCommon(affineStoreOp, rewriter);
    rewriter.eraseOp(storeOp);
    return result;
}

mlir::LogicalResult OutOfBoundsAffineStoreRewrite::matchAndRewrite(mlir::AffineStoreOp affineStoreOp, mlir::PatternRewriter& rewriter) const
{
    // Only check for out-of-bounds-accesses inside of ops that are marked for bounds checking
    if (!ir::util::AncestorOpContainsAttrOfName(affineStoreOp, transforms::affine::AccessBoundsCheckAttrName))
    {
        return mlir::success();
    }

    return OutOfBoundsStoreRewriteCommon(affineStoreOp, rewriter);
}

} // namespace

namespace accera::transforms::affine
{


void populateBoundsCheckingPatterns(mlir::RewritePatternSet& patterns)
{
    patterns.insert<OutOfBoundsLoadRewrite,
                    OutOfBoundsStoreRewrite,
                    OutOfBoundsAffineLoadRewrite,
                    OutOfBoundsAffineStoreRewrite>(patterns.getContext());
}

// TODO : implement
// std::unique_ptr<mlir::Pass> createBoundsCheckingPass(){}

}