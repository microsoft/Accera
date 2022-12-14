////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "AcceraPasses.h"

#include "affine/AffineSimplifications.h"
#include "util/RangeValueUtilities.h"
#include "nest/LoopNestToValue.h"
#include "value/RangeValueOptimizePass.h"

#include <ir/include/IRUtil.h>
#include <ir/include/value/ValueDialect.h>

#include <mlir/Conversion/AffineToStandard/AffineToStandard.h>
#include <mlir/Dialect/Affine/Analysis/LoopAnalysis.h>
#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/GPU/GPUDialect.h>
#include <mlir/Dialect/LLVMIR/ROCDLDialect.h>
#include <mlir/IR/Operation.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#include <llvm/ADT/TypeSwitch.h>

#include <algorithm>
#include <numeric>

using namespace accera::ir::value;
using namespace accera::ir::util;
using namespace mlir;

namespace
{

template <typename AffineOpTy>
struct AffineSimplifyHelper
{
    AffineSimplifyHelper(AffineOpTy op) :
        affineOp(op)
    {
        rangeAnalysis.addOperation(op);

        // Need to get the affine map for this access and the ranges for all of the operands to that map
        map = affineOp.getAffineMap();
        dimCount = map.getNumDims();
        symbolCount = map.getNumSymbols();
        auto operands = affineOp.getMapOperands();
        dimOperands.assign(operands.begin(), operands.begin() + dimCount);
        symOperands.assign(operands.begin() + dimCount, operands.end());
        assert(symbolCount == symOperands.size());
        std::transform(dimOperands.begin(), dimOperands.end(), std::back_inserter(dimOperandRanges), [&](mlir::Value dimOperand) {
            assert(rangeAnalysis.hasRange(dimOperand));
            return rangeAnalysis.getRange(dimOperand);
        });
        std::transform(symOperands.begin(), symOperands.end(), std::back_inserter(symOperandRanges), [&](mlir::Value symOperand) {
            assert(rangeAnalysis.hasRange(symOperand));
            return rangeAnalysis.getRange(symOperand);
        });
    }

    AffineOpTy affineOp;
    RangeValueAnalysis rangeAnalysis;
    mlir::AffineMap map;
    size_t dimCount;
    size_t symbolCount;
    std::vector<mlir::Value> dimOperands;
    std::vector<mlir::Value> symOperands;
    std::vector<RangeValue> dimOperandRanges;
    std::vector<RangeValue> symOperandRanges;
};

bool IsConstantOrSymbolicMul(mlir::AffineExpr expr)
{
    if (auto binOp = expr.dyn_cast<mlir::AffineBinaryOpExpr>(); expr.getKind() == mlir::AffineExprKind::Mul)
    {
        if (!(binOp.getLHS().isa<mlir::AffineBinaryOpExpr>()) &&
            !(binOp.getRHS().isa<mlir::AffineBinaryOpExpr>()) &&
            (binOp.getLHS().isSymbolicOrConstant() || binOp.getRHS().isSymbolicOrConstant()))
        {
            return true;
        }
    }
    return false;
}

// Gets the dim or symbol expr in the given expr
// Requires that the given expr is either a Dim expr, symbol expr, or a binary op expr where either the LHS or RHS is a constant and the other is a Dim/Symbol expr
[[maybe_unused]] mlir::AffineExpr GetContainedDimOrSymbolExpr(mlir::AffineExpr expr)
{
    if (expr.isa<mlir::AffineDimExpr>() || expr.isa<mlir::AffineSymbolExpr>())
    {
        return expr;
    }
    else if (auto binOp = expr.dyn_cast<mlir::AffineBinaryOpExpr>())
    {
        auto lhs = binOp.getLHS();
        auto rhs = binOp.getRHS();
        if (lhs.isa<mlir::AffineDimExpr>() || lhs.isa<mlir::AffineSymbolExpr>())
        {
            return lhs;
        }
        else if (rhs.isa<mlir::AffineDimExpr>() || rhs.isa<mlir::AffineSymbolExpr>())
        {
            return rhs;
        }
        else
        {
            assert(false);
            return nullptr;
        }
    }
    assert(false);
    return nullptr;
}

// Returns true if the expression is of the form (a_0*x_0 + a_1*x_1 + ... + a_n*x_n)
// for constants a_0,..., a_n and dimensions/symbols x_0, ..., x_n
// Returns false otherwise
bool IsLinearExpression(mlir::AffineExpr expr)
{
    if (auto binOp = expr.dyn_cast<mlir::AffineBinaryOpExpr>())
    {
        if (IsConstantOrSymbolicMul(binOp))
        {
            return true;
        }
        else if (binOp.getKind() == mlir::AffineExprKind::Add)
        {
            auto lhsDotProduct = IsLinearExpression(binOp.getLHS());
            auto rhsDotProduct = IsLinearExpression(binOp.getRHS());
            return lhsDotProduct && rhsDotProduct;
        }
        else
        {
            return false;
        }
    }
    else
    {
        // The expr is either a Constant, DimId, or SymbolId, which is the same as (1 * operand) + 0 and is therefore a simple dot product for our purposes
        return true;
    }
}

bool GetDotProductTerms(mlir::AffineExpr expr, std::vector<mlir::AffineExpr>& resultVec)
{
    if (IsLinearExpression(expr))
    {
        if (IsConstantOrSymbolicMul(expr) || !expr.isa<mlir::AffineBinaryOpExpr>())
        {
            resultVec.push_back(expr);
            return true;
        }
        // If it is a dot product but this binary op expr is not a constant multiplication, then this expr must be a sum
        auto binOp = expr.cast<mlir::AffineBinaryOpExpr>();
        assert(binOp.getKind() == mlir::AffineExprKind::Add);
        return GetDotProductTerms(binOp.getLHS(), resultVec) && GetDotProductTerms(binOp.getRHS(), resultVec);
    }
    return false;
}

// Creates an equivalent AffineExpr where the outermost expr is a sum of the term with the smallest coefficient and the remaining terms, repeating down the tree
// e.g.:                 +
//                     /   \
//                   +      (smallest coefficient) * operand_0
//                 /   \
//               +      (second smallest coefficient) * operand_1
//          ...
mlir::AffineExpr ReorderDotProduct(mlir::AffineExpr expr, std::vector<std::pair<int64_t, mlir::AffineExpr>>& coefficientAndExprs)
{
    if (!IsLinearExpression(expr))
    {
        return nullptr;
    }
    // Get all the individual terms
    std::vector<mlir::AffineExpr> individualMultiplies;
    if (!GetDotProductTerms(expr, individualMultiplies))
    {
        return nullptr;
    }
    if (individualMultiplies.empty())
    {
        return nullptr;
    }

    // Separate the coefficients from the Dim / Symbol id expr
    std::transform(individualMultiplies.begin(), individualMultiplies.end(), std::back_inserter(coefficientAndExprs), [](const mlir::AffineExpr& innerExpr) {
        return std::make_pair(innerExpr.getLargestKnownDivisor(), innerExpr);
    });

    // Now sort from largest coefficient to smallest
    std::sort(coefficientAndExprs.begin(), coefficientAndExprs.end(), [](const std::pair<int64_t, mlir::AffineExpr>& left, const std::pair<int64_t, mlir::AffineExpr>& right) {
        // Return true if left should be ordered first, which in this case means left has a larger coefficient
        return left.first > right.first;
    });

    // Now produce the final AffineExpr by summing each term in order to a running accumulation
    // Since we've ordered the coefficientAndExprs vec from largest coefficient to smallest, this means that the innermost
    // affine expr will be the product of the largest coefficient and its operand

    mlir::AffineExpr accumulatingExpr = coefficientAndExprs.front().second;
    for (size_t idx = 1; idx < coefficientAndExprs.size(); ++idx)
    {
        accumulatingExpr = accumulatingExpr + coefficientAndExprs[idx].second;
    }

    return accumulatingExpr;
}

[[maybe_unused]] std::vector<std::pair<int64_t, mlir::AffineExpr>> GetOrderedGCDsWithDenominatorHelper(const mlir::AffineExpr& numeratorExpr, int64_t denominator, mlir::AffineExpr& reorderedDotProduct)
{
    std::vector<std::pair<int64_t, mlir::AffineExpr>> orderedCoefficientAndExprs;
    reorderedDotProduct = ReorderDotProduct(numeratorExpr, orderedCoefficientAndExprs);
    std::vector<std::pair<int64_t, mlir::AffineExpr>> successiveCoefficientGCDsAndExprs; // vector of [ gcd(orderedCoefficients[0]), gcd(orderedCoefficients[0:1]), gcd(orderedCoefficients[0:2], ...)]
    int64_t currentGCD = denominator;
    std::transform(orderedCoefficientAndExprs.begin(), orderedCoefficientAndExprs.end(), std::back_inserter(successiveCoefficientGCDsAndExprs), [&](const std::pair<int64_t, mlir::AffineExpr>& coefficientAndExpr) {
        currentGCD = std::gcd(currentGCD, coefficientAndExpr.first);
        return std::make_pair(currentGCD, coefficientAndExpr.second);
    });
    return successiveCoefficientGCDsAndExprs;
}

template <typename FnType>
mlir::AffineExpr RunOnBinaryOpSubExpr(mlir::AffineExprKind exprKind, mlir::AffineExpr expr, size_t dimCount, size_t symbolCount, FnType fn)
{
    if (auto binOp = expr.dyn_cast<mlir::AffineBinaryOpExpr>())
    {
        auto newLhsExpr = RunOnBinaryOpSubExpr(exprKind, binOp.getLHS(), dimCount, symbolCount, fn);
        auto newRhsExpr = RunOnBinaryOpSubExpr(exprKind, binOp.getRHS(), dimCount, symbolCount, fn);
        newLhsExpr = mlir::simplifyAffineExpr(newLhsExpr, dimCount, symbolCount);
        newRhsExpr = mlir::simplifyAffineExpr(newRhsExpr, dimCount, symbolCount);
        auto newExpr = mlir::getAffineBinaryOpExpr(binOp.getKind(), newLhsExpr, newRhsExpr);
        if (newExpr.getKind() == exprKind)
        {
            return fn(newExpr);
        }
        else
        {
            return newExpr;
        }
    }
    return expr;
}

mlir::AffineValueMap GetAffineValueMap(mlir::AffineStoreOp& storeOp)
{
    return mlir::AffineValueMap(storeOp.getAffineMap(), storeOp.getMapOperands());
}
mlir::AffineValueMap GetAffineValueMap(mlir::AffineLoadOp& loadOp)
{
    return mlir::AffineValueMap(loadOp.getAffineMap(), loadOp.getMapOperands());
}
mlir::AffineValueMap GetAffineValueMap(mlir::AffineApplyOp& applyOp)
{
    return applyOp.getAffineValueMap();
}

template<typename AffineOpTy>
bool AllOperandDefsAreInScope(AffineOpTy op)
{
    auto operands = op.getMapOperands();
    for (auto operand : operands)
    {
        mlir::Operation* defOp = GetDefiningOpOrForLoop(operand);
        if (defOp == nullptr)
        {
            return false;
        }
    }
    return true;
}

void ReplaceOpUsingNewValueMap(PatternRewriter& rewriter, mlir::AffineLoadOp loadOp, mlir::AffineValueMap newAffineValueMap)
{
    rewriter.replaceOpWithNewOp<mlir::AffineLoadOp>(loadOp, loadOp.memref(), newAffineValueMap.getAffineMap(), newAffineValueMap.getOperands());
}
void ReplaceOpUsingNewValueMap(PatternRewriter& rewriter, mlir::AffineStoreOp storeOp, mlir::AffineValueMap newAffineValueMap)
{
    rewriter.replaceOpWithNewOp<mlir::AffineStoreOp>(storeOp, storeOp.value(), storeOp.memref(), newAffineValueMap.getAffineMap(), newAffineValueMap.getOperands());
}
void ReplaceOpUsingNewValueMap(PatternRewriter& rewriter, mlir::AffineApplyOp applyOp, mlir::AffineValueMap newAffineValueMap)
{
    rewriter.replaceOpWithNewOp<mlir::AffineApplyOp>(applyOp, newAffineValueMap.getAffineMap(), newAffineValueMap.getOperands());
}

template <typename AffineOpTy>
struct SmallNumeratorTermFloorDivSimplification : public OpRewritePattern<AffineOpTy>
{
    using OpRewritePattern<AffineOpTy>::OpRewritePattern;

    LogicalResult matchAndRewrite(AffineOpTy affineOp, PatternRewriter& rewriter) const final
    {
        // See docs/Reference/gpu_caching_floor_divisions.md for a proof of the equivalence this simplification leverages

        if (!AllOperandDefsAreInScope(affineOp))
        {
            return failure();
        }

        AffineSimplifyHelper helper(affineOp);
        auto loc = affineOp.getLoc();

        // Handle each expression in the map separately
        // Need to walk each expression and find sub-expressions that are floor divides where the numerator is a sum of positive numbers
        auto exprs = helper.map.getResults();
        MutableAffineMap mutableMap(helper.map);
        bool modifiedMap = false;
        auto constantZeroOp = rewriter.create<mlir::arith::ConstantIntOp>(loc, 0, rewriter.getI64Type());
        auto constantZeroRV = helper.rangeAnalysis.addOperation(constantZeroOp);
        for (size_t exprIdx = 0; exprIdx < exprs.size(); ++exprIdx)
        {
            auto newExpr = RunOnBinaryOpSubExpr(mlir::AffineExprKind::FloorDiv, exprs[exprIdx], helper.dimCount, helper.symbolCount, [&](mlir::AffineExpr floorDivExpr) {
                assert(floorDivExpr.isa<mlir::AffineBinaryOpExpr>());
                auto floorDivBinaryOpExpr = floorDivExpr.cast<mlir::AffineBinaryOpExpr>();
                auto numerator = floorDivBinaryOpExpr.getLHS();
                auto denominator = floorDivBinaryOpExpr.getRHS();
                if (!IsLinearExpression(numerator) || !denominator.isa<mlir::AffineConstantExpr>())
                {
                    return floorDivExpr;
                }
                int64_t denominatorValue = denominator.cast<mlir::AffineConstantExpr>().getValue();
                mlir::AffineExpr reorderedDotProduct;
                auto successiveCoefficientGCDsAndExprs = GetOrderedGCDsWithDenominatorHelper(numerator, denominatorValue, reorderedDotProduct);

                bool modifiedCurrentExpr = true;
                mlir::AffineExpr currentFloorDiv = floorDivExpr;
                mlir::AffineExpr currentReorderedDotProduct = reorderedDotProduct;
                while (modifiedCurrentExpr)
                {
                    // Now get the range of the operands and check if the the term with the smallest coefficient can be removed from the map
                    auto lastExpr = successiveCoefficientGCDsAndExprs.back().second;

                    // Expand the expression of the (coefficient * operand) term with the smallest coefficient
                    // and compue the RangeValue of it
                    mlir::Value smallestTermExpanded = mlir::expandAffineExpr(rewriter, loc, lastExpr, helper.dimOperands, helper.symOperands);
                    mlir::Operation* smallestTermExpandedOp = GetDefiningOpOrForLoop(smallestTermExpanded);

                    RangeValue lastExprRange = helper.rangeAnalysis.addOperation(smallestTermExpandedOp);

                    if (successiveCoefficientGCDsAndExprs.size() == 1)
                    {
                        // If there's only one term in the numerator, check if its range is greater than 0 and less than the denominator
                        // In which case this floordiv is always 0
                        auto constantDenominatorOp = rewriter.create<mlir::arith::ConstantIntOp>(loc, denominatorValue, rewriter.getI64Type());
                        RangeValue constantDenominatorRV = helper.rangeAnalysis.addOperation(constantDenominatorOp);
                        if (lastExprRange.icmp(llvm::CmpInst::Predicate::ICMP_SGE, constantZeroRV) &&
                            lastExprRange.icmp(llvm::CmpInst::Predicate::ICMP_SLT, constantDenominatorRV))
                        {
                            modifiedMap = true;
                            return rewriter.getAffineConstantExpr(0);
                        }
                    }
                    if (successiveCoefficientGCDsAndExprs.size() >= 2)
                    {
                        auto secondSmallestGCD = successiveCoefficientGCDsAndExprs[successiveCoefficientGCDsAndExprs.size() - 2].first;
                        auto constantGCD = rewriter.create<mlir::arith::ConstantIntOp>(loc, secondSmallestGCD, rewriter.getI64Type());
                        RangeValue constantGCDRV = helper.rangeAnalysis.addOperation(constantGCD);
                        if (lastExprRange.icmp(llvm::CmpInst::Predicate::ICMP_SGE, constantZeroRV) &&
                            lastExprRange.icmp(llvm::CmpInst::Predicate::ICMP_SLT, constantGCDRV))
                        {
                            // The last term is always smaller than the smallest GCD, so the last term can be removed
                            auto reorderedDotProductBinOp = currentReorderedDotProduct.cast<mlir::AffineBinaryOpExpr>();
                            currentReorderedDotProduct = reorderedDotProductBinOp.getLHS();
                            currentFloorDiv = currentReorderedDotProduct.floorDiv(denominatorValue);
                            modifiedMap = true;

                            successiveCoefficientGCDsAndExprs.pop_back();
                        }
                        else
                        {
                            modifiedCurrentExpr = false;
                        }
                    }
                    else
                    {
                        modifiedCurrentExpr = false;
                    }
                }
                return currentFloorDiv;
            });

            mutableMap.setResult(exprIdx, newExpr);
        }

        if (modifiedMap)
        {
            auto newMap = mutableMap.getAffineMap();
            mlir::AffineValueMap newAffineValueMap(newMap, affineOp.getMapOperands());
            (void)newAffineValueMap.canonicalize();
            ReplaceOpUsingNewValueMap(rewriter, affineOp, newAffineValueMap);
            return success();
        }
        return failure();
    }
};

template <typename AffineOpTy>
struct SmallNumeratorTermModSimplification : public OpRewritePattern<AffineOpTy>
{
    using OpRewritePattern<AffineOpTy>::OpRewritePattern;

    LogicalResult matchAndRewrite(AffineOpTy affineOp, PatternRewriter& rewriter) const final
    {
        // See docs/Reference/gpu_caching_mod.md for a proof of the equivalence this simplification leverages

        if (!AllOperandDefsAreInScope(affineOp))
        {
            return failure();
        }

        AffineSimplifyHelper helper(affineOp);
        auto loc = affineOp.getLoc();

        // Handle each expression in the map separately
        // Need to walk each expression and find sub-expressions that are floor divides where the numerator is a sum of positive numbers
        auto exprs = helper.map.getResults();
        MutableAffineMap mutableMap(helper.map);
        bool modifiedMap = false;
        auto constantZeroOp = rewriter.create<mlir::arith::ConstantIntOp>(loc, 0, rewriter.getI64Type());
        auto constantZeroRV = helper.rangeAnalysis.addOperation(constantZeroOp);
        for (size_t exprIdx = 0; exprIdx < exprs.size(); ++exprIdx)
        {
            auto newExpr = RunOnBinaryOpSubExpr(mlir::AffineExprKind::Mod, exprs[exprIdx], helper.dimCount, helper.symbolCount, [&](mlir::AffineExpr modExpr) {
                assert(modExpr.isa<mlir::AffineBinaryOpExpr>());
                auto modBinaryOpExpr = modExpr.cast<mlir::AffineBinaryOpExpr>();
                auto numerator = modBinaryOpExpr.getLHS();
                auto denominator = modBinaryOpExpr.getRHS();
                if (!IsLinearExpression(numerator) || !denominator.isa<mlir::AffineConstantExpr>())
                {
                    return modExpr;
                }
                int64_t denominatorValue = denominator.cast<mlir::AffineConstantExpr>().getValue();
                mlir::AffineExpr reorderedDotProduct;
                auto successiveCoefficientGCDsAndExprs = GetOrderedGCDsWithDenominatorHelper(numerator, denominatorValue, reorderedDotProduct);

                bool modifiedCurrentExpr = true;
                mlir::AffineExpr currentModExpr = modExpr;
                mlir::AffineExpr currentReorderedDotProduct = reorderedDotProduct;
                mlir::AffineExpr extractedSumExpr = rewriter.getAffineConstantExpr(0);
                while (modifiedCurrentExpr)
                {
                    // Now get the range of the operands and check if the the term with the smallest coefficient can be removed from the map
                    auto lastExpr = successiveCoefficientGCDsAndExprs.back().second;

                    // Expand the expression of the (coefficient * operand) term with the smallest coefficient
                    // and compue the RangeValue of it
                    mlir::Value smallestTermExpanded = mlir::expandAffineExpr(rewriter, loc, lastExpr, helper.dimOperands, helper.symOperands);
                    mlir::Operation* smallestTermExpandedOp = GetDefiningOpOrForLoop(smallestTermExpanded);

                    RangeValue lastExprRange = helper.rangeAnalysis.addOperation(smallestTermExpandedOp);

                    if (successiveCoefficientGCDsAndExprs.size() == 1)
                    {
                        // If there's only one term in the numerator, check if its range is greater than 0 and less than the denominator
                        // In which case this mod is unnecessary
                        auto constantDenominatorOp = rewriter.create<mlir::arith::ConstantIntOp>(loc, denominatorValue, rewriter.getI64Type());
                        RangeValue constantDenominatorRV = helper.rangeAnalysis.addOperation(constantDenominatorOp);
                        if (lastExprRange.icmp(llvm::CmpInst::Predicate::ICMP_SGE, constantZeroRV) &&
                            lastExprRange.icmp(llvm::CmpInst::Predicate::ICMP_SLT, constantDenominatorRV))
                        {
                            modifiedMap = true;
                            return numerator;
                        }
                    }
                    if (successiveCoefficientGCDsAndExprs.size() >= 2)
                    {
                        auto secondSmallestGCD = successiveCoefficientGCDsAndExprs[successiveCoefficientGCDsAndExprs.size() - 2].first;
                        auto constantGCD = rewriter.create<mlir::arith::ConstantIntOp>(loc, secondSmallestGCD, rewriter.getI64Type());
                        RangeValue constantGCDRV = helper.rangeAnalysis.addOperation(constantGCD);
                        if (lastExprRange.icmp(llvm::CmpInst::Predicate::ICMP_SGE, constantZeroRV) &&
                            lastExprRange.icmp(llvm::CmpInst::Predicate::ICMP_SLT, constantGCDRV))
                        {
                            // The last term is always smaller than the smallest GCD, so the last term can be removed
                            auto reorderedDotProductBinOp = currentReorderedDotProduct.cast<mlir::AffineBinaryOpExpr>();
                            currentReorderedDotProduct = reorderedDotProductBinOp.getLHS();
                            extractedSumExpr = extractedSumExpr + reorderedDotProductBinOp.getRHS();
                            currentModExpr = currentReorderedDotProduct % denominatorValue;
                            modifiedMap = true;

                            successiveCoefficientGCDsAndExprs.pop_back();
                        }
                        else
                        {
                            modifiedCurrentExpr = false;
                        }
                    }
                    else
                    {
                        modifiedCurrentExpr = false;
                    }
                }
                return extractedSumExpr + currentModExpr;
            });

            mutableMap.setResult(exprIdx, newExpr);
        }

        if (modifiedMap)
        {
            auto newMap = mutableMap.getAffineMap();
            mlir::AffineValueMap newAffineValueMap(newMap, affineOp.getMapOperands());
            (void)newAffineValueMap.canonicalize();
            ReplaceOpUsingNewValueMap(rewriter, affineOp, newAffineValueMap);
            return success();
        }
        return failure();
    }
};

template <typename AffineOpTy>
struct PropagateGPUConstants : public OpRewritePattern<AffineOpTy>
{
    using OpRewritePattern<AffineOpTy>::OpRewritePattern;

    LogicalResult matchAndRewrite(AffineOpTy affineOp, PatternRewriter& rewriter) const final
    {
        if (!AllOperandDefsAreInScope(affineOp))
        {
            return failure();
        }

        auto loc = affineOp.getLoc();

        std::vector<mlir::Operation*> opsToErase;

        // Examine the operands, if any are block dim ops or grid dim ops, then replace them with their equivalent constant values and simplify the map
        bool replaced = false;
        for (auto operand : affineOp.getMapOperands())
        {
            if (auto definingOp = GetDefiningOpOrForLoop(operand))
            {
                auto handleBlockDimOp = [&](gpu::BlockDimOp blockDimOp) {
                    auto dimSize = GetBlockDimSize(blockDimOp);
                    if (dimSize.has_value())
                    {
                        mlir::Value dimSizeConstantOp = rewriter.create<mlir::arith::ConstantIntOp>(loc, *dimSize, rewriter.getI64Type());
                        affineOp->replaceUsesOfWith(operand, dimSizeConstantOp);
                        replaced = true;
                    }
                };
                auto handleGridDimOp = [&](gpu::GridDimOp gridDimOp) {
                    auto dimSize = GetGridDimSize(gridDimOp);
                    if (dimSize.has_value())
                    {
                        mlir::Value dimSizeConstantOp = rewriter.create<mlir::arith::ConstantIntOp>(loc, *dimSize, rewriter.getI64Type());
                        affineOp->replaceUsesOfWith(operand, dimSizeConstantOp);
                        replaced = true;
                    }
                };
                mlir::TypeSwitch<mlir::Operation*>(definingOp)
                    .Case([&](gpu::BlockDimOp blockDimOp) {
                        handleBlockDimOp(blockDimOp);
                    })
                    .Case([&](gpu::GridDimOp gridDimOp) {
                        handleGridDimOp(gridDimOp);
                    })
                    .Case([&](mlir::arith::ConstantOp constantOp) {
                        // Canonicalize the constant int into the operation
                        replaced = true;
                    });
            }
        }
        for (auto op : opsToErase)
        {
            rewriter.eraseOp(op);
        }

        if (replaced)
        {
            // Simplify the affine maps and recreate the op
            mlir::AffineValueMap affineValueMap = GetAffineValueMap(affineOp);
            (void)affineValueMap.canonicalize();
            ReplaceOpUsingNewValueMap(rewriter, affineOp, affineValueMap);
            return success();
        }
        else
        {
            return failure();
        }
    }
};

struct AffineForOpSimplifyBounds : public OpRewritePattern<AffineForOp>
{
    using OpRewritePattern<AffineForOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(AffineForOp affineForOp, PatternRewriter& rewriter) const final
    {
        RangeValueAnalysis rangeValue;

        auto lowerBound = affineForOp.getLowerBound();
        auto initLowerBoundMap = lowerBound.getMap();
        std::vector<mlir::Value> lowerBoundOperands(lowerBound.operandBegin(), lowerBound.operandEnd());
        mlir::AffineValueMap lowerBoundValueMap(initLowerBoundMap, lowerBoundOperands);
        lowerBoundValueMap = SimplifyAffineValueMap(lowerBoundValueMap);
        auto simplifiedLowerBoundMap = lowerBoundValueMap.getAffineMap();

        auto upperBound = affineForOp.getUpperBound();
        auto initUpperBoundMap = upperBound.getMap();
        std::vector<mlir::Value> upperBoundOperands(upperBound.operandBegin(), upperBound.operandEnd());
        mlir::AffineValueMap upperBoundValueMap(initUpperBoundMap, upperBoundOperands);
        upperBoundValueMap = SimplifyAffineValueMap(upperBoundValueMap);
        auto simplifiedUpperBoundMap = upperBoundValueMap.getAffineMap();

        rewriter.updateRootInPlace(affineForOp, [&]()
        {
            if (simplifiedLowerBoundMap.isSingleConstant())
            {
                auto lowerBoundConst = simplifiedLowerBoundMap.getSingleConstantResult();
                affineForOp.setConstantLowerBound(lowerBoundConst);
            }
            else
            {
                affineForOp.setUpperBound(upperBoundValueMap.getOperands(), upperBoundValueMap.getAffineMap());
            }

            if (simplifiedUpperBoundMap.isSingleConstant())
            {
                auto upperBoundConst = simplifiedUpperBoundMap.getSingleConstantResult();
                affineForOp.setConstantUpperBound(upperBoundConst);
            }
            else
            {
                affineForOp.setLowerBound(lowerBoundValueMap.getOperands(), lowerBoundValueMap.getAffineMap());
            }
        });

        if (affineForOp.hasConstantBounds())
        {
            auto constantTripCountOpt = mlir::getConstantTripCount(affineForOp);
            if (constantTripCountOpt.getValue() == 0)
            {
                rewriter.eraseOp(affineForOp);
                return success();
            }
            return PromoteIfSingleIteration(rewriter, affineForOp);
        }

        // Didn't remove the loop, but possibly modified it. Let another rewrite try to simplify it
        return failure();
    }
};

struct AffineSimplificationPass : public accera::transforms::AcceraAffineSimplificationBase<AffineSimplificationPass>
{
    void runOnOperation() final
    {
        auto* context = &getContext();
        auto op = getOperation();

        {
            mlir::GreedyRewriteConfig singleIterationConfig;
            singleIterationConfig.maxIterations = 1;

            OwningRewritePatternList patterns(context);
            accera::transforms::affine::populateAcceraAffineExprSimplificationPatterns(patterns);
            (void)applyPatternsAndFoldGreedily(op, std::move(patterns), singleIterationConfig);
        }

        // Apply RangeValueOptimize and affine value map simplification to try to simplify possibly-dynamic loop bounds
        {
            mlir::GreedyRewriteConfig topDownConfig;
            topDownConfig.useTopDownTraversal = true;

            OwningRewritePatternList patterns(context);
            accera::transforms::affine::populateAcceraAffineLoopSimplificationPatterns(patterns);
            (void)applyPatternsAndFoldGreedily(op, std::move(patterns), topDownConfig);
        }
    }
};

} // namespace

namespace accera::transforms::affine
{

void populateAcceraAffineExprSimplificationPatterns(mlir::OwningRewritePatternList& patterns)
{
    patterns.insert<SmallNumeratorTermFloorDivSimplification<mlir::AffineLoadOp>>(patterns.getContext());
    patterns.insert<SmallNumeratorTermFloorDivSimplification<mlir::AffineStoreOp>>(patterns.getContext());
    patterns.insert<SmallNumeratorTermFloorDivSimplification<mlir::AffineApplyOp>>(patterns.getContext());
    patterns.insert<SmallNumeratorTermModSimplification<mlir::AffineLoadOp>>(patterns.getContext());
    patterns.insert<SmallNumeratorTermModSimplification<mlir::AffineStoreOp>>(patterns.getContext());
    patterns.insert<SmallNumeratorTermModSimplification<mlir::AffineApplyOp>>(patterns.getContext());
    patterns.insert<PropagateGPUConstants<mlir::AffineLoadOp>>(patterns.getContext());
    patterns.insert<PropagateGPUConstants<mlir::AffineStoreOp>>(patterns.getContext());
    patterns.insert<PropagateGPUConstants<mlir::AffineApplyOp>>(patterns.getContext());
}

void populateAcceraAffineLoopSimplificationPatterns(mlir::OwningRewritePatternList& patterns)
{
    patterns.insert<AffineForOpSimplifyBounds>(patterns.getContext());
    accera::transforms::value::populateRangeValueOptimizePatterns(patterns);
}

std::unique_ptr<mlir::Pass> createAffineSimplificationPass()
{
    return std::make_unique<AffineSimplificationPass>();
}
} // namespace accera::transforms::affine
