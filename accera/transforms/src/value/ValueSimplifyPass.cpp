////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
//  Authors: Kern Handa, Mason Remy
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "AcceraPasses.h"

#include <ir/include/value/ValueDialect.h>

#include <mlir/Conversion/AffineToStandard/AffineToStandard.h>
#include <mlir/Conversion/LinalgToLLVM/LinalgToLLVM.h>
#include <mlir/Conversion/SCFToStandard/SCFToStandard.h>
#include <mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h>
#include <mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h>

#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/Arithmetic/IR/Arithmetic.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/SCF.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>

#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassManager.h>

#include <mlir/Transforms/Passes.h>

#include <llvm/Support/raw_os_ostream.h>

#include <iostream>

using namespace mlir;

namespace
{
using namespace scf;
#include "value/ValueConversion.inc"
} // namespace

using namespace accera::ir;
using namespace accera::transforms;
using namespace accera::ir::value;

using ValueCopyOp = accera::ir::value::CopyOp;
struct CopyOpLowering : public OpRewritePattern<ValueCopyOp>
{
    using OpRewritePattern<ValueCopyOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(
        ValueCopyOp op,
        PatternRewriter& rewriter) const override;
};

using ValueSliceOp = accera::ir::value::SliceOp;
struct ValueSliceSimplifyPattern : public OpRewritePattern<ValueSliceOp>
{
    using OpRewritePattern::OpRewritePattern;

    LogicalResult matchAndRewrite(
        ValueSliceOp op,
        PatternRewriter& rewriter) const final
    {
        using accera::ir::value::GetElementOp;

        auto loc = op.getLoc();
        auto indexType = rewriter.getIndexType();
        auto source = op.source();

        auto sourceType = source.getType().cast<mlir::MemRefType>();
        auto shape = sourceType.getShape();
        auto zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
        auto one = rewriter.create<arith::ConstantIndexOp>(loc, 1);

        // Initialize to a full view (no sliced dimensions)
        llvm::SmallVector<mlir::Value, 4> resolvedOffsets(shape.size(), zero);
        llvm::SmallVector<mlir::Value, 4> sizes;
        for (auto extent : shape)
        {
            sizes.push_back(rewriter.create<arith::ConstantIndexOp>(loc, extent));
        }

        llvm::SmallVector<mlir::Value, 4> strides(shape.size(), one);

        auto sliceDimensions = op.sliceDimensions().getValue();
        auto offsets = op.offsets();

        for (int64_t i = 0; i < static_cast<int64_t>(offsets.size()); ++i)
        {
            auto dim = sliceDimensions[i].cast<IntegerAttr>().getInt();
            auto index = offsets[i];
            if (index.getType().isIndex())
            {
                resolvedOffsets[dim] = index;
                sizes[dim] = one;
            }
            else
            {
                auto indexShape = index.getType().cast<mlir::ShapedType>().getShape();
                if (indexShape.size() == 0 || indexShape.size() == 1)
                {
                    resolvedOffsets[dim] = rewriter.create<mlir::arith::IndexCastOp>(loc, rewriter.create<GetElementOp>(loc, index), indexType);
                }
                else
                {
                    assert(false && "Unknown offset shape for slice op");
                    return failure();
                }
            }
        }
        rewriter.replaceOpWithNewOp<memref::SubViewOp>(op, mlir::MemRefType{}, source, resolvedOffsets, sizes, strides);

        return success();
    }
};

struct ValueSimplifyPass : public ConvertValueSimplifyBase<ValueSimplifyPass>
{
    void runOnOperation() final;
};

LogicalResult CopyOpLowering::matchAndRewrite(
    ValueCopyOp op,
    PatternRewriter& rewriter) const
{
    auto loc = op.getLoc();
    auto zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);

    auto input = op.input();
    auto output = op.output();
    auto inputType = input.getType();

    auto outputMemRef = output.getType().cast<MemRefType>();
    if (auto inputShape = inputType.dyn_cast<ShapedType>())
    {
        if (inputShape.hasStaticShape() && inputShape.getNumElements() == 1)
        {
            llvm::SmallVector<Value, 4> indices((size_t)outputMemRef.getRank(), zero);
            auto loadedElt = rewriter.create<GetElementOp>(loc, input);
            rewriter.create<StoreOp>(loc, loadedElt, output, indices);
        }
        else
        {
            (void)rewriter.create<memref::CopyOp>(loc, input, output);
        }
    }
    else if (inputType.isIndex())
    {
        if (outputMemRef.getElementType().isInteger(64)) // this should really be target dependent...
        {
            (void)rewriter.create<memref::StoreOp>(loc,
                                                   rewriter.create<mlir::arith::IndexCastOp>(loc, input, rewriter.getIntegerType(64)),
                                                   output,
                                                   std::vector<mlir::Value>(outputMemRef.getRank(), zero));
        }
        else
        {
            mlir::emitError(loc, "Index types can only be stored within MemRefs of I64");
        }
    }
    else if (inputType == outputMemRef.getElementType())
    {
        if (outputMemRef.getElementType() == inputType)
        {
            (void)rewriter.create<memref::StoreOp>(loc, input, output, std::vector<mlir::Value>(outputMemRef.getRank(), zero));
        }
        else
        {
            mlir::emitError(loc, "Output MemRef element type must match input type for scalar copying");
        }
    }
    else
    {
        mlir::emitError(loc, "Unknown input type to accv.CopyOp");
    }

    rewriter.eraseOp(op);
    return success();
}

using ValueBinOp = accera::ir::value::BinOp;
struct IndexCombinationBinOpLowering : public OpRewritePattern<ValueBinOp>
{
    // Convert value BinOps that are just combinations of index types into affine apply ops
    using OpRewritePattern<ValueBinOp>::OpRewritePattern;

    // returns true on success, false on failure
    bool ReplaceConstantIntWithIndex(OpBuilder& builder, mlir::Value& val) const
    {
        auto op = val.getDefiningOp();
        if (auto constantOp = mlir::dyn_cast_or_null<arith::ConstantIntOp>(op))
        {
            OpBuilder::InsertionGuard insertGuard(builder);
            builder.setInsertionPointAfter(constantOp);
            auto constantIndex = builder.create<arith::ConstantIndexOp>(constantOp.getLoc(), constantOp.value());
            val.replaceAllUsesWith(constantIndex.getResult());
            val = constantIndex.getResult();
            return true;
        }
        else
        {
            return false;
        }
    }

    std::optional<int64_t> GetConstantIntValue(mlir::Value& val) const
    {
        if (auto op = val.getDefiningOp())
        {
            if (auto constantIntOp = mlir::dyn_cast<arith::ConstantIntOp>(op))
            {
                return constantIntOp.value();
            }
            else if (auto constantIndexOp = mlir::dyn_cast<arith::ConstantIndexOp>(op))
            {
                return constantIndexOp.value();
            }
        }
        return {};
    }

    LogicalResult matchAndRewrite(
        ValueBinOp op,
        PatternRewriter& rewriter) const override
    {
        [[maybe_unused]] auto loc = op.getLoc();

        auto lhs = op.lhs();
        auto rhs = op.rhs();
        auto result = op.result();

        auto lhsType = lhs.getType();
        auto rhsType = rhs.getType();
        auto resultType = result.getType();

        if (resultType.isIndex())
        {
            // If the inputs aren't index types but are constants, replace them with constant index types
            // if they aren't constants, then we can't convert them currently so just return
            mlir::AffineExpr lhsExpr;
            mlir::AffineExpr rhsExpr;
            std::vector<mlir::Value> exprInputs;
            int nextDimIdx = 0;
            auto constLhsIntOpt = GetConstantIntValue(lhs);
            if (constLhsIntOpt.has_value())
            {
                lhsExpr = rewriter.getAffineConstantExpr(*constLhsIntOpt);
            }
            else
            {
                lhsExpr = rewriter.getAffineDimExpr(nextDimIdx++);
                if (!lhsType.isIndex())
                {
                    if (!ReplaceConstantIntWithIndex(rewriter, lhs))
                    {
                        return failure();
                    }
                }
                exprInputs.push_back(lhs);
            }

            auto constRhsIntOpt = GetConstantIntValue(rhs);
            if (constRhsIntOpt.has_value())
            {
                rhsExpr = rewriter.getAffineConstantExpr(*constRhsIntOpt);
            }
            else
            {
                rhsExpr = rewriter.getAffineDimExpr(nextDimIdx++);
                if (!rhsType.isIndex())
                {
                    if (!ReplaceConstantIntWithIndex(rewriter, rhs))
                    {
                        return failure();
                    }
                }
                exprInputs.push_back(rhs);
            }

            // Replace this BinOp with an AffineApplyOp since it's just a combination of index types
            mlir::AffineExpr combinationExpr;
            switch (op.getPredicate())
            {
            case BinaryOpPredicate::ADD:
                combinationExpr = lhsExpr + rhsExpr;
                break;
            case BinaryOpPredicate::SUB:
                combinationExpr = lhsExpr - rhsExpr;
                break;
            case BinaryOpPredicate::MUL:
                combinationExpr = lhsExpr * rhsExpr;
                break;
            case BinaryOpPredicate::DIV:
                combinationExpr = lhsExpr.floorDiv(rhsExpr);
                break;
            case BinaryOpPredicate::MOD:
                combinationExpr = lhsExpr % rhsExpr;
                break;
            default:
                assert(false);
            }
            auto map = mlir::AffineMap::get(nextDimIdx, 0, combinationExpr);
            rewriter.replaceOpWithNewOp<mlir::AffineApplyOp>(op, map, exprInputs);
        }
        return success();
    }
};

void ValueSimplifyPass::runOnOperation()
{
    RewritePatternSet patterns(&getContext());
    accera::transforms::value::populateValueSimplifyPatterns(patterns);

    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
}

namespace accera::transforms::value
{

void populateValueSimplifyPatterns(mlir::RewritePatternSet& patterns)
{
    mlir::MLIRContext* context = patterns.getContext();
    populateWithGenerated(patterns);
    patterns.insert<CopyOpLowering, ValueSliceSimplifyPattern, IndexCombinationBinOpLowering>(context);
}

std::unique_ptr<mlir::Pass> createValueSimplifyPass()
{
    return std::make_unique<ValueSimplifyPass>();
}

} // namespace accera::transforms::value
