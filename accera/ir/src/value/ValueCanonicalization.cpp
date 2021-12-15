////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "IRUtil.h"
#include "value/ValueDialect.h"

#include <llvm/ADT/APFloat.h>

#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>

#include <mlir/IR/Builders.h>
#include <mlir/IR/Matchers.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/IR/PatternMatch.h>

namespace v = accera::ir::value;

namespace
{
struct SimplifyRank0SliceFollowedByCopy : public mlir::OpRewritePattern<v::CopyOp>
{
    using OpRewritePattern::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(v::CopyOp op,
                                        mlir::PatternRewriter& rewriter) const override
    {
        auto outputDefOp = op.output().getDefiningOp();
        if (auto sliceParent = llvm::dyn_cast_or_null<v::SliceOp>(outputDefOp))
        {
            mlir::MemRefType sliceResTy = sliceParent.getType();
            if (sliceResTy.getRank() != 0)
            {
                return mlir::failure();
            }

            rewriter.replaceOpWithNewOp<mlir::memref::StoreOp>(op, op.input(), sliceParent.source(), sliceParent.offsets());
            return mlir::success();
        }

        return mlir::failure();
    }
};

struct MergeSliceOps : public mlir::OpRewritePattern<v::SliceOp>
{
    MergeSliceOps(mlir::MLIRContext* context) :
        OpRewritePattern(context, 10)
    {}

    mlir::LogicalResult match(v::SliceOp op) const override
    {
        v::SliceOpAdaptor adaptor{ op };

        if (op.getResult().hasOneUse() && llvm::isa_and_nonnull<v::SliceOp>(adaptor.source().getDefiningOp()))
        {
            return mlir::success();
        }

        return mlir::failure();
    }

    void rewrite(v::SliceOp op, mlir::PatternRewriter& rewriter) const override
    {
        v::SliceOpAdaptor adaptor{ op };

        auto incomingSliceOp = adaptor.source().getDefiningOp<v::SliceOp>();
        v::SliceOpAdaptor incoming{ incomingSliceOp };
        auto newSource = incoming.source();

        auto incomingDims = accera::ir::util::ConvertArrayAttrToIntVector(incoming.sliceDimensions());
        auto incomingDimsAndOffsets = llvm::zip(incomingDims, incoming.offsets());
        auto thisDims = accera::ir::util::ConvertArrayAttrToIntVector(adaptor.sliceDimensions());
        auto thisDimsAndOffsets = llvm::zip(thisDims, adaptor.offsets());
        auto totalResolvedDims = incomingDims.size() + thisDims.size();
        size_t resolvedIdx = 0;
        llvm::SmallVector<int64_t> resolvedDims(totalResolvedDims);
        llvm::SmallVector<mlir::Value> resolvedOffsets(totalResolvedDims);
        auto incomingIt = incomingDimsAndOffsets.begin(),
             incomingE = incomingDimsAndOffsets.end(),
             thisIt = thisDimsAndOffsets.begin(),
             thisE = thisDimsAndOffsets.end();
        for (; resolvedIdx < totalResolvedDims; ++resolvedIdx)
        {

            // There are two pairs of dims/offsets to keep track of. The one that comes from "this" op,
            // and the one that comes from the predecessor ("incoming"). The predecessor's result is "this" op's first
            // operand. We have to reconstruct a replacement for this op that uses the predecessor's source and merged
            // dimension indices and offsets.
            // At this point, we've allocated a large enough array to hold the merged/resolved dims/offsets
            // For each resolved element, we start by iterating over the incoming op's dim/offset pairs.
            // If we're not at the end of the incoming dim/offset pair
            if (incomingIt != incomingE)
            {
                // And we're not at the end of the this op's dim/offset pair
                if (thisIt != thisE)
                {
                    // Get the two dimension values
                    auto& thisDim = std::get<0>(*thisIt);
                    auto& incomingDim = std::get<0>(*incomingIt);

                    // If the incoming dimension is less than or equal to this dimension,
                    // set the resolved dimension/offset to it
                    // We also increment the values of this op's dimensions, because the merged view
                    // takes the dimension values wrt to incoming source, whereas the dimension
                    // values in this op are wrt to result of the predecessor
                    // Finally, we move to the next incoming pair's element, but still continue to point at the element
                    // from this op's pair
                    if (incomingDim <= thisDim)
                    {
                        resolvedDims[resolvedIdx] = incomingDim;
                        resolvedOffsets[resolvedIdx] = std::get<1>(*incomingIt);
                        std::transform(
                            thisIt,
                            thisE,
                            thisIt,
                            [](std::tuple<int64_t, mlir::Value> t) -> std::tuple<int64_t, mlir::Value> {
                                return { std::get<0>(t) + 1, std::get<1>(t) };
                            });
                        // ++thisDim;
                        ++incomingIt;
                    }
                    // Otherwise, this dim/offset pair gets set to as the resolved one and we move to the next one
                    // This represents the case where the preceding slice operates on dimensions that are closer to the
                    // innermost
                    else
                    {
                        resolvedDims[resolvedIdx] = thisDim;
                        resolvedOffsets[resolvedIdx] = std::get<1>(*thisIt);
                        ++thisIt;
                    }
                }
                // If we're at the end of this op's pairs, then just append the rest of the predecessor's pairs
                else
                {
                    auto& incomingDim = std::get<0>(*incomingIt);
                    resolvedDims[resolvedIdx] = incomingDim;
                    resolvedOffsets[resolvedIdx] = std::get<1>(*incomingIt);
                    ++incomingIt;
                }
            }
            // We've run out of dim/offsets from the preceding op, add the rest from this op
            else
            {
                assert(thisIt != thisE);
                auto& thisDim = std::get<0>(*thisIt);
                resolvedDims[resolvedIdx] = thisDim;
                resolvedOffsets[resolvedIdx] = std::get<1>(*thisIt);
                ++thisIt;
            }
        }
        assert(incomingIt == incomingE);
        assert(thisIt == thisE);

        rewriter.replaceOpWithNewOp<v::SliceOp>(
            op,
            newSource,
            resolvedDims,
            resolvedOffsets,
            op.getType());
    }
};

/// The matcher that matches a constant scalar / vector splat / tensor splat
/// float operation and binds the constant float value.
struct constant_float_op_binder
{
    mlir::FloatAttr::ValueType* bind_value;

    /// Creates a matcher instance that binds the value to bv if match succeeds.
    constant_float_op_binder(mlir::FloatAttr::ValueType* bv) :
        bind_value(bv) {}

    bool match(mlir::Operation* op)
    {
        mlir::Attribute attr;
        if (!mlir::detail::constant_op_binder<mlir::Attribute>(&attr).match(op))
            return false;

        auto type = op->getResult(0).getType();

        if (type.isa<mlir::FloatType>())
            return mlir::detail::attr_value_binder<mlir::FloatAttr>(bind_value).match(attr);

        if (type.isa<mlir::VectorType, mlir::RankedTensorType>())
        {
            if (auto splatAttr = attr.dyn_cast<mlir::SplatElementsAttr>())
            {
                return mlir::detail::attr_value_binder<mlir::FloatAttr>(bind_value)
                    .match(splatAttr.getSplatValue());
            }
        }
        return false;
    }
};

/// The matcher that matches a given target constant scalar / vector splat /
/// tensor splat float value.
struct constant_float_value_matcher
{
    constant_float_value_matcher(double target, double eps = 1e-5) :
        _target(target), _eps(eps) {}

    bool match(mlir::Operation* op)
    {
        llvm::APFloat value{ 0.0 };
        if (constant_float_op_binder(&value).match(op))
        {
            bool lossless = false;
            value.convert(llvm::APFloat::IEEEdouble(), llvm::APFloat::rmNearestTiesToAway, &lossless);
            return llvm::abs(value - _target) <= _eps;
        }
        return false;
    }

    llvm::APFloat _target;
    llvm::APFloat _eps;
};

struct ValueBinOpSimplification : public mlir::OpRewritePattern<v::BinOp>
{
    using OpRewritePattern::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(v::BinOp op, mlir::PatternRewriter& rewriter) const override
    {
        auto zeroish = constant_float_value_matcher(0.0);
        auto oneish = constant_float_value_matcher(1.0);

        switch (op.predicate())
        {
        case v::BinaryOpPredicate::ADD:
            // y = 0 + x => y = x
            if (mlir::matchPattern(op.lhs(), zeroish))
            {
                rewriter.replaceOp(op, { op.rhs() });
                return mlir::success();
            }
            // y = x + 0 => y = x
            if (mlir::matchPattern(op.rhs(), zeroish))
            {
                rewriter.replaceOp(op, { op.lhs() });
                return mlir::success();
            }
            break;
        case v::BinaryOpPredicate::SUB:
            // TODO: ?
            // y = 0 - x => y = -x
            // if (mlir::matchPattern(op.lhs(), zeroish))
            // {
            // }

            // y = x - 0 => y = x
            if (mlir::matchPattern(op.rhs(), zeroish))
            {
                rewriter.replaceOp(op, { op.lhs() });
                return mlir::success();
            }
            break;
        case v::BinaryOpPredicate::MUL:
            // y = 0 * x => y = 0
            if (mlir::matchPattern(op.lhs(), zeroish))
            {
                rewriter.replaceOp(op, { op.lhs() });
                return mlir::success();
            }
            // y = x * 0 => y = 0
            if (mlir::matchPattern(op.rhs(), zeroish))
            {
                rewriter.replaceOp(op, { op.rhs() });
                return mlir::success();
            }

            // y = 1 * x => y = x
            if (mlir::matchPattern(op.lhs(), oneish))
            {
                rewriter.replaceOp(op, { op.rhs() });
                return mlir::success();
            }
            // y = x * 1 => y = x
            if (mlir::matchPattern(op.rhs(), oneish))
            {
                rewriter.replaceOp(op, { op.lhs() });
                return mlir::success();
            }
            break;
        case v::BinaryOpPredicate::DIV:
            // y = x / 1 => y = x
            if (mlir::matchPattern(op.rhs(), oneish))
            {
                rewriter.replaceOp(op, { op.lhs() });
                return mlir::success();
            }
            break;
        default:
            break;
        }
        return mlir::failure();
    }
};

struct BarrierCanonicalizationPattern : public mlir::OpRewritePattern<v::BarrierOp>
{
    using OpRewritePattern::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(v::BarrierOp barrierOp,
                                        mlir::PatternRewriter& rewriter) const override
    {
        mlir::Operation* op = barrierOp;
        auto block = op->getBlock();
        auto begin = block->begin();
        mlir::Block::iterator it{ barrierOp };

        if (begin != it)
        {
            if (auto prevBarrier = llvm::dyn_cast<v::BarrierOp>(--it))
            {
                auto prevBarrierAttrs = prevBarrier->getAttrs();
                auto myAttrs = op->getAttrs();
                if (myAttrs == prevBarrierAttrs)
                {
                    rewriter.eraseOp(barrierOp);
                    return mlir::success();
                }
            }
        }

        return mlir::failure();
    }
};

struct ValueModuleOpConversion : public mlir::OpRewritePattern<v::ValueModuleOp>
{
    using OpRewritePattern<v::ValueModuleOp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(
        v::ValueModuleOp moduleOp,
        mlir::PatternRewriter& rewriter) const override
    {
        if (moduleOp.getOperation()->getNumRegions() == 0)
        {
            rewriter.eraseOp(moduleOp);
            return mlir::success();
        }

        if (moduleOp.getOperation()->getNumRegions() == 1 &&
            llvm::hasSingleElement(moduleOp.body()))
        {
            mlir::Block& block = moduleOp.body().front();
            if (!block.empty() && mlir::isa<v::ModuleTerminatorOp>(block.front()))
            {
                rewriter.eraseBlock(&block);
                rewriter.eraseOp(moduleOp);
                return mlir::success();
            }
        }

        return mlir::failure();
    }
};

} // namespace

void v::CopyOp::getCanonicalizationPatterns(mlir::OwningRewritePatternList& patterns,
                                            mlir::MLIRContext* context)
{
    patterns.insert<SimplifyRank0SliceFollowedByCopy>(context);
}

void v::BarrierOp::getCanonicalizationPatterns(mlir::OwningRewritePatternList& patterns, mlir::MLIRContext* context)
{
    patterns.insert<BarrierCanonicalizationPattern>(context);
}

void v::ValueModuleOp::getCanonicalizationPatterns(mlir::OwningRewritePatternList& patterns, mlir::MLIRContext* context)
{
    patterns.insert<ValueModuleOpConversion>(context);
}

void v::BinOp::getCanonicalizationPatterns(mlir::OwningRewritePatternList& patterns, mlir::MLIRContext* context)
{
    patterns.insert<ValueBinOpSimplification>(context);
}

void v::SliceOp::getCanonicalizationPatterns(mlir::OwningRewritePatternList& patterns, mlir::MLIRContext* context)
{
    patterns.insert<MergeSliceOps>(context);
}

namespace
{
struct SimplifyRank0SliceFollowedByGetElement : public mlir::OpRewritePattern<v::GetElementOp>
{
    using OpRewritePattern<v::GetElementOp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(v::GetElementOp op,
                                        mlir::PatternRewriter& rewriter) const override
    {
        auto operandDefOp = op.value().getDefiningOp();
        if (auto sliceParent = llvm::dyn_cast_or_null<v::SliceOp>(operandDefOp))
        {
            mlir::MemRefType sliceResTy = sliceParent.getType();
            if (sliceResTy.getRank() != 0)
            {
                return mlir::failure();
            }

            llvm::SmallVector<mlir::Value, 1> results;
            (void)rewriter.tryFold(sliceParent, results);

            rewriter.replaceOpWithNewOp<mlir::memref::LoadOp>(op, sliceParent.source(), sliceParent.offsets());
            return mlir::success();
        }

        return mlir::failure();
    }
};
} // namespace

void v::GetElementOp::getCanonicalizationPatterns(mlir::OwningRewritePatternList& patterns,
                                                  mlir::MLIRContext* context)
{
    patterns.insert<SimplifyRank0SliceFollowedByGetElement>(context);
}

// Slice op folding is disabled for now
#if 0

// Fold a slice of a slice into a new slice
mlir::OpFoldResult v::SliceOp::fold(llvm::ArrayRef<mlir::Attribute> operands)
{
    mlir::OpBuilder builder(getContext());
    auto sliceParent = source().getDefiningOp<v::SliceOp>();
    if (!sliceParent)
    {
        return {};
    }

    [[maybe_unused]] llvm::SmallVector<mlir::Value, 1> results;
    if (mlir::succeeded(builder.tryFold(sliceParent, results)))
    {
        return {};
    }

    sourceMutable().assign(sliceParent.source());

    auto combinedOffsets = llvm::to_vector<6>(sliceParent.offsets());
    combinedOffsets.append(offsets().begin(), offsets().end());
    offsetsMutable().assign(combinedOffsets);

    auto combinedSliceDimensions = llvm::to_vector<6>(sliceParent.sliceDimensions().getValue());
    auto thisSliceDimensions = sliceDimensions().getValue();

    // Increment the dimension from this op by the number of dimensions in the source that were removed before it:
    for (auto& d : thisSliceDimensions)
    {
        auto dVal = d.cast<IntegerAttr>().getInt();
        dVal += std::count_if(combinedSliceDimensions.begin(), combinedSliceDimensions.end(), [&](mlir::Attribute x) { return x.cast<IntegerAttr>().getInt() <= dVal; });
        combinedSliceDimensions.push_back(IntegerAttr::get(builder.getI64Type(), dVal));
    }

    sliceDimensionsAttr(builder.getArrayAttr(combinedSliceDimensions));

    return result();
}
#endif
