////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
////////////////////////////////////////////////////////////////////////////////////////////////////

#include <catch2/catch_all.hpp>

#include "IRTestVerification.h"
#include "TestContext.h"

#include <ir/include/AffineConstraintsHelper.h>
#include <ir/include/DialectRegistry.h>
#include <ir/include/IRUtil.h>
#include <ir/include/InitializeAccera.h>
#include <ir/include/nest/Index.h>
#include <ir/include/nest/IndexRange.h>
#include <ir/include/nest/IterationDomain.h>
#include <ir/include/nest/LoopNestAttributes.h>
#include <ir/include/nest/LoopNestOps.h>
#include <ir/include/nest/LoopNestTypes.h>
#include <ir/include/nest/Range.h>
#include <ir/include/nest/TransformedDomain.h>
#include <ir/include/value/ValueDialect.h>

#include <transforms/include/nest/LoopNestPasses.h>

#include <utilities/include/Logger.h>
#include <utilities/include/MemoryLayout.h>
#include <utilities/include/TypeTraits.h>

#include <llvm/Support/InitLLVM.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/Support/raw_os_ostream.h>

#include <mlir/Dialect/Affine/Analysis/AffineStructures.h>
#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/Affine/LoopUtils.h>
#include <mlir/Dialect/Arithmetic/IR/Arithmetic.h>
#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/Dialect/Vector/IR/VectorOps.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Verifier.h>
#include <mlir/Parser.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/Passes.h>

#include <array>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

using namespace accera::ir;
using namespace accera::ir::util;
using namespace loopnest;
using namespace value;
using namespace mlir;

#define DEBUG_PRINT true

using IdWrapper = AffineConstraintsHelper::IdWrapper;

namespace
{
mlir::Value Alloca(mlir::OpBuilder& builder, mlir::MemRefType bufferType)
{
    // TODO : return to using std_alloc when aligned_alloc/_aligned_malloc issue in MLIR is fixed on Windows
    return builder.create<mlir::memref::AllocaOp>(builder.getUnknownLoc(), bufferType);
}

mlir::Value SumValues(mlir::OpBuilder& builder, const std::vector<mlir::Value>& values)
{
    mlir::AffineExpr sumExpr = builder.getAffineConstantExpr(0);
    for (size_t idx = 0; idx < values.size(); idx++)
    {
        sumExpr = sumExpr + builder.getAffineDimExpr(idx);
    }
    auto sumMap = mlir::AffineMap::get(values.size(), 0, sumExpr);
    return builder.create<mlir::AffineApplyOp>(builder.getUnknownLoc(), sumMap, values);
}

mlir::AffineExpr MakeIdSumExpr(const std::vector<IdWrapper>& ids, mlir::MLIRContext* context)
{
    mlir::AffineExpr expr = mlir::getAffineConstantExpr(0, context);
    for (const auto& id : ids)
    {
        expr = expr + id.GetExpr(context);
    }
    return expr;
}

struct PartitionCreationInfo
{
    IdWrapper dimId;
    int64_t stepSize;
};


struct PartitionInfo
{
    mlir::Value partitionValue;
    mlir::Value largestMainLoopIVValue;

    mlir::AffineValueMap partitionValueMap;
    mlir::AffineValueMap localExprAffineValueMap;
    mlir::AffineValueMap largestMainLoopIVValueMap;
};

PartitionInfo MakeSplitPartition(mlir::OpBuilder& builder, mlir::Value begin, mlir::Value end, int64_t splitSize)
{
    PartitionInfo info;
    auto d0 = builder.getAffineDimExpr(0);
    auto d1 = builder.getAffineDimExpr(1);
    auto loc = builder.getUnknownLoc();

    // The split partition separates the range that gets full splits from the range that can't get a full split in
    // cleanup_amount = (end - begin) % splitSize
    // partition_value = end - cleanup_amount

    // Note: x mod y = x - (x floordiv y) * y (see proof elsewhere)
    //       So x - (x mod y) = (x floordiv y) * y
    // So rewrite:
    // partition_value = end - cleanup_amount
    //                 = end - (end - begin) % splitSize
    //                 = begin + (end - begin) - (end - begin) % splitSize
    //                 = begin + ((end - begin) floordiv splitSize) * splitSize

    auto range = d1 - d0;
    auto numFullIterations = (range).floorDiv(splitSize);

    // Floor divides of dynamic values introduce local expressions in a constraint system because floor division is a non-affine operation
    // So compute what the local expression and operands would be and hold onto them for use in later bounds resolution
    auto localExpr = numFullIterations;
    auto localExprMap = mlir::AffineMap::get(2, 0, localExpr);
    llvm::SmallVector<mlir::Value, 2> localExprOperands{ begin, end };
    mlir::fullyComposeAffineMapAndOperands(&localExprMap, &localExprOperands);
    mlir::canonicalizeMapAndOperands(&localExprMap, &localExprOperands);
    info.localExprAffineValueMap.reset(localExprMap, localExprOperands);

    auto partitionValueExpr = numFullIterations * splitSize;
    auto partitionValueMap = mlir::AffineMap::get(2, 0, partitionValueExpr);

    mlir::AffineMap simplifiedPartitionValueMap(partitionValueMap);
    llvm::SmallVector<mlir::Value, 2> simplifiedPartitionValueOperands{ begin, end };
    mlir::fullyComposeAffineMapAndOperands(&simplifiedPartitionValueMap, &simplifiedPartitionValueOperands);
    mlir::canonicalizeMapAndOperands(&simplifiedPartitionValueMap, &simplifiedPartitionValueOperands);
    if (simplifiedPartitionValueMap.isSingleConstant())
    {
        // Constants simplify more easily in the constraint system, so prefer returning constant ops over an AffineApplyOp with a constant map
        info.partitionValue = builder.create<mlir::arith::ConstantIndexOp>(loc, simplifiedPartitionValueMap.getSingleConstantResult());
    }
    else
    {
        info.partitionValue = builder.create<mlir::AffineApplyOp>(loc, simplifiedPartitionValueMap, simplifiedPartitionValueOperands);
    }
    info.partitionValueMap.reset(simplifiedPartitionValueMap, simplifiedPartitionValueOperands);

    // Now compute the largest value that the IV will take in the main loop
    auto largestMainLoopIVMap = mlir::AffineMap::get(1, 0, d0 - splitSize);
    llvm::SmallVector<mlir::Value, 2> simplifiedLargestMainIVOperands{ info.partitionValue };

    info.largestMainLoopIVValueMap.reset(largestMainLoopIVMap, simplifiedLargestMainIVOperands);

    return info;
}

struct PartitionedLoopInfo
{
    IdWrapper loopId;
    int64_t stepSize;
    IdWrapper partitionValueId;
    IdWrapper largestMainLoopIVId;
};

PartitionedLoopInfo AddSplitPartitionHelper(AffineConstraintsHelper& cst,
                                            IdWrapper loopId,
                                            const std::vector<IdWrapper>& innerLoopIds,
                                            mlir::OpBuilder& builder,
                                            mlir::Location loc,
                                            int64_t stepSize)
{
    // Get the [begin, end) range for this loop id
    AffineConstraintsHelper resolveRangeCst = cst.Clone();

    auto [beginValueMap, endValueMap] = resolveRangeCst.GetLowerAndUpperBound(loopId, builder, loc, innerLoopIds);

    // Produce a begin and end value using affine apply ops
    mlir::Value beginVal = mlir::makeComposedAffineApply(builder, loc, beginValueMap.getAffineMap(), beginValueMap.getOperands());
    mlir::Value endVal = mlir::makeComposedAffineApply(builder, loc, endValueMap.getAffineMap(), endValueMap.getOperands());

    auto partitionInfo = MakeSplitPartition(builder, beginVal, endVal, stepSize);

    // Add the partitionInfo.partitionValue and partitionInfo.largestMainLoopIVValue as symbols
    // The partition value is the first value not touched by the main loop
    // The largest main loop IV value is the largest value the main loop IV will have
    // Some Examples:
    //      1) Given an outer loop 0 ... 16 step 4,
    //              the partitionValue would be 16 (because 16 - ((16 - 0) % 4) == 16)
    //              and the largest main loop IV value would be 12, namely (partitionValue - stepSize)
    //      2) Given an outer loop 0 ... 19 step 4,
    //              note there will now be a boundary cleanup loop.
    //              The partitionValue will again be 16 (because 19 - ((19 - 0) % 4) == 19 - (19 % 4) == 19 - 3 == 16),
    //              and the largest main loop IV value will still be (partitionValue - stepSize) == 12
    //              note: this is not simply (end - stepSize)

    IdWrapper partitionValueId = cst.AddSymbol(partitionInfo.partitionValue);
    IdWrapper largestMainLoopIVId = cst.AddSymbol(partitionInfo.largestMainLoopIVValue);
    PartitionedLoopInfo info{ loopId,
                              stepSize,
                              partitionValueId,
                              largestMainLoopIVId };

    // Set partitionValue >= beginValue
    cst.AddLowerBoundMap(info.partitionValueId, beginValueMap);

    // Set partitionValue <= endValue
    cst.AddUpperBoundMap(info.partitionValueId, endValueMap, false /* exclusive */);

    // Bound the partition value relative to the step size and end value
    // 0 <= end - PV < step
    // 0 <= end - PV <= step - 1
    // PV <= end <= PV + step - 1
    // PV >= end - step + 1
    auto alignedEndMap = cst.AlignAffineValueMap(endValueMap);
    auto endExprs = alignedEndMap.getResults();
    std::vector<mlir::AffineExpr> endMinusStepPlusOneExprs(endExprs.begin(), endExprs.end());
    for (auto& expr : endMinusStepPlusOneExprs)
    {
        expr = expr - stepSize + 1;
    }
    auto endMinusStepPlusOneMap = cst.GetMap(endMinusStepPlusOneExprs);
    cst.AddLowerBoundMap(info.partitionValueId, endMinusStepPlusOneMap);

    // Set the partition value and largest main loop IV symbols equal to a function of the other dims and symbols in the constraint system
    auto alignedPartitionValueLocalExprMap = cst.AlignAffineValueMap(partitionInfo.localExprAffineValueMap);
    auto partitionValueLocalExpr = alignedPartitionValueLocalExprMap.getResult(0);
    cst.SetEqualMap(info.partitionValueId, partitionInfo.partitionValueMap, partitionValueLocalExpr);

    auto alignedLargestMainLoopIVValue_PV_map = cst.AlignAffineValueMap(partitionInfo.largestMainLoopIVValueMap);
    cst.SetEqualMap(info.largestMainLoopIVId, alignedLargestMainLoopIVValue_PV_map);

    return info;
}


template <typename FnTy>
void CreateLoopPartitionsHelperRecursive(const AffineConstraintsHelper& cst,
                                         mlir::OpBuilder& builder,
                                         mlir::Location loc,
                                         const std::vector<PartitionCreationInfo>& orderedLoopInfos,
                                         unsigned currentIdx,
                                         const std::vector<mlir::Value>& outerLoopIVs,
                                         FnTy&& innerFunction)
{
    if (cst.IsEmpty())
    {
        return;
    }
    if (currentIdx >= orderedLoopInfos.size())
    {
        innerFunction(builder, outerLoopIVs);
        return;
    }
    const auto& currentLoopInfo = orderedLoopInfos[currentIdx];

    // Create the paritition for this loop level
    AffineConstraintsHelper levelScopedConstraints = cst.Clone();

    auto getDimId = [](const PartitionCreationInfo& info) {
        return info.dimId;
    };
    std::vector<IdWrapper> outerLoopIds;
    std::vector<IdWrapper> innerLoopIds;
    std::transform(orderedLoopInfos.begin(), orderedLoopInfos.begin() + currentIdx, std::back_inserter(outerLoopIds), getDimId);
    std::transform(orderedLoopInfos.begin() + currentIdx + 1, orderedLoopInfos.end(), std::back_inserter(innerLoopIds), getDimId);

    auto partitionInfo = AddSplitPartitionHelper(levelScopedConstraints,
                                                 currentLoopInfo.dimId,
                                                 innerLoopIds,
                                                 builder,
                                                 loc,
                                                 currentLoopInfo.stepSize);

    // Create the main loop regardless of whether there is a partition
    {
        std::vector<mlir::Value> currentLoopIVs = outerLoopIVs;

        // Fork the constraints for inside the main loop
        AffineConstraintsHelper mainScopedConstraints = levelScopedConstraints.Clone();

        // Bound loopId <= largest main loop IV
        mainScopedConstraints.AddUpperBound(partitionInfo.loopId, partitionInfo.largestMainLoopIVId, false /*exclusive*/);

        // Fork the constraints for resolving the current main loop bounds
        AffineConstraintsHelper tempResolveConstraints = levelScopedConstraints.Clone();

        // Bound loopId <= partition value. This is a looser constraint than we put on the mainScopedConstraints, but it is helpful
        // for getting a simpler loop bound
        tempResolveConstraints.AddUpperBound(currentLoopInfo.dimId, partitionInfo.partitionValueId);

        // Project out and simplify resolution constraints
        tempResolveConstraints.ProjectOut(innerLoopIds);
        tempResolveConstraints.Simplify();

        if (!tempResolveConstraints.IsEmpty())
        {
            auto [lbValueMap, ubValueMap] = tempResolveConstraints.GetLowerAndUpperBound(partitionInfo.loopId, builder, loc);

            auto loop = builder.create<mlir::AffineForOp>(loc, lbValueMap.getOperands(), lbValueMap.getAffineMap(), ubValueMap.getOperands(), ubValueMap.getAffineMap(), partitionInfo.stepSize);
            auto loopBuilder = util::MakeBodyBuilder(loop);
            auto currentIV = loop.getInductionVar();
            currentLoopIVs.push_back(currentIV);
            mainScopedConstraints.SetValue(partitionInfo.loopId, currentIV);
            CreateLoopPartitionsHelperRecursive(mainScopedConstraints, loopBuilder, loc, orderedLoopInfos, currentIdx + 1, currentLoopIVs, innerFunction);
        }
    }

    // Cleanup loop
    {
        std::vector<mlir::Value> currentLoopIVs = outerLoopIVs;

        // Fork the constraints for inside the cleanup loop
        AffineConstraintsHelper cleanupScopedConstraints = levelScopedConstraints.Clone();

        // Set loop id equal to partition value inside the cleanup loop
        cleanupScopedConstraints.SetEqual(partitionInfo.loopId, partitionInfo.partitionValueId);

        // Fork the constraints for resolving the current cleanup loop bounds
        AffineConstraintsHelper tempResolveConstraints = levelScopedConstraints.Clone();

        // Bound loopId >= partition value. This is a looser constraint than we put on the mainScopedConstraints, but it is helpful
        // for getting a simpler loop bound
        tempResolveConstraints.AddLowerBound(partitionInfo.loopId, partitionInfo.partitionValueId);

        // Project out and simplify resolution constraints
        tempResolveConstraints.ProjectOut(innerLoopIds);
        tempResolveConstraints.Simplify();

        if (!tempResolveConstraints.IsEmpty())
        {
            auto [lbValueMap, ubValueMap] = tempResolveConstraints.GetLowerAndUpperBound(partitionInfo.loopId, builder, loc);
            auto loop = builder.create<mlir::AffineForOp>(loc, lbValueMap.getOperands(), lbValueMap.getAffineMap(), ubValueMap.getOperands(), ubValueMap.getAffineMap(), partitionInfo.stepSize);
            auto loopBuilder = util::MakeBodyBuilder(loop);
            auto currentIV = loop.getInductionVar();
            currentLoopIVs.push_back(currentIV);
            cleanupScopedConstraints.SetValue(partitionInfo.loopId, currentIV);
            CreateLoopPartitionsHelperRecursive(cleanupScopedConstraints, loopBuilder, loc, orderedLoopInfos, currentIdx + 1, currentLoopIVs, innerFunction);
        }
    }
}

template <typename FnTy>
void CreateLoopPartitionsHelper(const AffineConstraintsHelper& cst,
                                mlir::OpBuilder& builder,
                                mlir::Location loc,
                                const std::vector<PartitionCreationInfo>& orderedLoopInfos,
                                FnTy&& innerFunction)
{
    std::vector<mlir::Value> outerLoopIVs;
    CreateLoopPartitionsHelperRecursive(cst, builder, loc, orderedLoopInfos, 0, outerLoopIVs, innerFunction);
}

} // namespace

TEST_CASE("TestDynamicLoopRange")
{
    int64_t M = 256;
    int64_t dynSize = mlir::ShapedType::kDynamicSize;
    TestContext context(
        "TestDynamicLoopRange",
        [&]() -> std::vector<mlir::Type> {
                            auto& builder = GetTestBuilder();

                            auto f32Type = builder.getF32Type();
                            auto sizeType = builder.getIndexType();

                            auto inType = mlir::MemRefType::get({ M, dynSize }, f32Type);
                            auto outType = mlir::MemRefType::get({ M }, f32Type);

                           return { inType, sizeType, outType }; },
        [&](std::vector<mlir::Value> args) {
            auto& builder = GetTestBuilder();
            auto loc = builder.getUnknownLoc();

            auto f32Type = builder.getF32Type();

            auto inputMatrix = args[0];
            auto columnCount = args[1];
            auto outputVector = args[2];

            auto floatZero = builder.create<mlir::arith::ConstantOp>(loc, f32Type, builder.getFloatAttr(f32Type, 0.0));
            auto indexZero = builder.create<mlir::arith::ConstantIndexOp>(loc, 0);

            auto mBegin = builder.create<mlir::arith::ConstantIndexOp>(loc, 0);
            auto mEnd = builder.create<mlir::arith::ConstantIndexOp>(loc, M);
            auto nBegin = builder.create<mlir::arith::ConstantIndexOp>(loc, 0);
            auto nEnd = columnCount;

            auto symIdentityMap = builder.getSymbolIdentityMap();
            auto mLoopLBMap = symIdentityMap;
            auto mLoopUBMap = symIdentityMap;
            auto nLoopLBMap = symIdentityMap;
            auto nLoopUBMap = symIdentityMap;
            auto accum = Alloca(builder, mlir::MemRefType::get({ 1 }, f32Type));

            auto mLoop = builder.create<mlir::AffineForOp>(loc, mlir::ValueRange{ mBegin }, mLoopLBMap, mlir::ValueRange{ mEnd }, mLoopUBMap, 1);
            auto mIV = mLoop.getInductionVar();
            auto& mLoopRegion = mLoop.region();
            mlir::Block* mLoopFront = &mLoopRegion.front();
            mlir::OpBuilder mLoopBodyBuilder({ mLoopFront, std::prev(mLoopFront->end()) });

            mLoopBodyBuilder.create<mlir::memref::StoreOp>(loc, floatZero, accum, mlir::ValueRange{ indexZero });

            auto nLoop = mLoopBodyBuilder.create<mlir::AffineForOp>(loc, mlir::ValueRange{ nBegin }, nLoopLBMap, mlir::ValueRange{ nEnd }, nLoopUBMap, 1);
            auto nIV = nLoop.getInductionVar();
            auto& nLoopRegion = nLoop.region();
            mlir::Block* nLoopFront = &nLoopRegion.front();
            mlir::OpBuilder nLoopBodyBuilder({ nLoopFront, std::prev(nLoopFront->end()) });

            auto currentAccum = nLoopBodyBuilder.create<mlir::memref::LoadOp>(loc, accum, mlir::ValueRange{ indexZero });
            auto currentVal = nLoopBodyBuilder.create<mlir::memref::LoadOp>(loc, inputMatrix, mlir::ValueRange{ mIV, nIV });
            auto addedVal = nLoopBodyBuilder.create<mlir::arith::AddFOp>(loc, currentAccum, currentVal);

            nLoopBodyBuilder.create<mlir::memref::StoreOp>(loc, addedVal, accum, mlir::ValueRange{ indexZero });

            auto finalAccum = mLoopBodyBuilder.create<mlir::memref::LoadOp>(loc, accum, mlir::ValueRange{ indexZero });
            mLoopBodyBuilder.create<mlir::memref::StoreOp>(loc, finalAccum, outputVector, mlir::ValueRange{ mIV });
        });

    SECTION("Parsing")
    {
        REQUIRE(VerifyParse(context, true, "TestDynamicLoopRange.mlir"));
    }

    SECTION("Std")
    {
        REQUIRE(VerifyLowerToStd(context, true, "TestDynamicLoopRange_std.mlir"));
    }

    SECTION("LLVM")
    {
        REQUIRE(VerifyTranslateToLLVMIR(context, false, true, "TestDynamicLoopRange_noopt.ll"));
        REQUIRE(VerifyTranslateToLLVMIR(context, true, true, "TestDynamicLoopRange.ll"));
    }
}

TEST_CASE("TestDynamicLoopRangeStaticSplit")
{
    int64_t M = 256;
    int64_t NSplitSize = 8;
    int64_t dynSize = mlir::ShapedType::kDynamicSize;
    TestContext context(
        "TestDynamicLoopRangeStaticSplit",
        [&]() -> std::vector<mlir::Type> {
                            auto& builder = GetTestBuilder();

                            auto f32Type = builder.getF32Type();
                            auto sizeType = builder.getIndexType();

                            auto inType = mlir::MemRefType::get({ M, dynSize }, f32Type);
                            auto outType = mlir::MemRefType::get({ M }, f32Type);

                           return { inType, sizeType, outType }; },
        [&](std::vector<mlir::Value> args) {
            auto& builder = GetTestBuilder();
            auto loc = builder.getUnknownLoc();

            auto f32Type = builder.getF32Type();

            auto inputMatrix = args[0];
            auto columnCount = args[1];
            auto outputVector = args[2];

            auto floatZero = builder.create<mlir::arith::ConstantOp>(loc, f32Type, builder.getFloatAttr(f32Type, 0.0));
            auto indexZero = builder.create<mlir::arith::ConstantIndexOp>(loc, 0);

            auto mBegin = builder.create<mlir::arith::ConstantIndexOp>(loc, 0);
            auto mEnd = builder.create<mlir::arith::ConstantIndexOp>(loc, M);
            auto nBegin = builder.create<mlir::arith::ConstantIndexOp>(loc, 0);
            auto nInnerBegin = builder.create<mlir::arith::ConstantIndexOp>(loc, 0);
            auto nEnd = columnCount;
            auto nSplitConstant = builder.create<mlir::arith::ConstantIndexOp>(loc, NSplitSize);

            auto d0 = builder.getAffineDimExpr(0);
            auto d1 = builder.getAffineDimExpr(1);
            auto s0 = builder.getAffineSymbolExpr(0);
            auto splitLeftoversMap = mlir::AffineMap::get(2, 1, (d1 - d0) % s0); // args: (start, end)[split_constant]
            mlir::Value nSplitLeftoversValue = builder.create<mlir::AffineApplyOp>(loc, splitLeftoversMap, mlir::ValueRange{ nBegin, nEnd, nSplitConstant });

            auto symIdentityMap = builder.getSymbolIdentityMap();
            auto mLoopLBMap = symIdentityMap;
            auto mLoopUBMap = symIdentityMap;

            // The outer N loop has a main loop and a cleanup loop
            auto nMainLoopLBMap = symIdentityMap;
            auto nMainLoopUBMap = mlir::AffineMap::get(2, 0, d1 - d0); // end - leftovers
            auto nCleanupLoopLBMap = nMainLoopUBMap;
            auto nCleanupLoopUBMap = symIdentityMap; // end - leftovers

            // The inner N loop inside the outer N main loop will have a fixed range
            // And the inner N loop inside the outer N cleanup loop will have a dynamic range based on how much is leftover
            auto nInnerLoopLBMap = symIdentityMap;
            auto nInnerLoopUBMap = symIdentityMap;

            auto accum = Alloca(builder, mlir::MemRefType::get({ 1 }, f32Type));

            auto mLoop = builder.create<mlir::AffineForOp>(loc, mlir::ValueRange{ mBegin }, mLoopLBMap, mlir::ValueRange{ mEnd }, mLoopUBMap, 1);
            auto mIV = mLoop.getInductionVar();
            auto& mLoopRegion = mLoop.region();
            mlir::Block* mLoopFront = &mLoopRegion.front();
            mlir::OpBuilder mLoopBodyBuilder({ mLoopFront, std::prev(mLoopFront->end()) });

            mLoopBodyBuilder.create<mlir::memref::StoreOp>(loc, floatZero, accum, mlir::ValueRange{ indexZero });

            auto fillNLoopContents = [&](mlir::AffineForOp nOuterLoop, bool main) {
                auto nOuterIV = nOuterLoop.getInductionVar();
                auto& nOuterLoopRegion = nOuterLoop.region();
                mlir::Block* nOuterLoopFront = &nOuterLoopRegion.front();
                mlir::OpBuilder nOuterLoopBodyBuilder({ nOuterLoopFront, std::prev(nOuterLoopFront->end()) });

                mlir::Value nInnerEnd = main ? nSplitConstant : nSplitLeftoversValue;

                auto nInnerLoop = nOuterLoopBodyBuilder.create<mlir::AffineForOp>(loc, mlir::ValueRange{ nInnerBegin }, nInnerLoopLBMap, mlir::ValueRange{ nInnerEnd }, nInnerLoopUBMap, 1);
                auto nInnerIV = nInnerLoop.getInductionVar();
                auto& nInnerLoopRegion = nInnerLoop.region();
                mlir::Block* nInnerLoopFront = &nInnerLoopRegion.front();
                mlir::OpBuilder nInnerLoopBodyBuilder({ nInnerLoopFront, std::prev(nInnerLoopFront->end()) });

                auto compositeNIV = nInnerLoopBodyBuilder.create<mlir::arith::AddIOp>(loc, nOuterIV, nInnerIV);

                auto currentAccum = nInnerLoopBodyBuilder.create<mlir::memref::LoadOp>(loc, accum, mlir::ValueRange{ indexZero });
                auto currentVal = nInnerLoopBodyBuilder.create<mlir::memref::LoadOp>(loc, inputMatrix, mlir::ValueRange{ mIV, compositeNIV });
                auto addedVal = nInnerLoopBodyBuilder.create<mlir::arith::AddFOp>(loc, currentAccum, currentVal);

                nInnerLoopBodyBuilder.create<mlir::memref::StoreOp>(loc, addedVal, accum, mlir::ValueRange{ indexZero });
            };

            auto nMainLoop = mLoopBodyBuilder.create<mlir::AffineForOp>(loc, mlir::ValueRange{ nBegin }, nMainLoopLBMap, mlir::ValueRange{ nSplitLeftoversValue, nEnd }, nMainLoopUBMap, NSplitSize);
            auto nCleanupLoop = mLoopBodyBuilder.create<mlir::AffineForOp>(loc, mlir::ValueRange{ nSplitLeftoversValue, nEnd }, nCleanupLoopLBMap, mlir::ValueRange{ nEnd }, nCleanupLoopUBMap, NSplitSize);

            fillNLoopContents(nMainLoop, true);
            fillNLoopContents(nCleanupLoop, false);

            auto finalAccum = mLoopBodyBuilder.create<mlir::memref::LoadOp>(loc, accum, mlir::ValueRange{ indexZero });
            mLoopBodyBuilder.create<mlir::memref::StoreOp>(loc, finalAccum, outputVector, mlir::ValueRange{ mIV });
        });

    SECTION("Parsing")
    {
        REQUIRE(VerifyParse(context, true, "TestDynamicLoopRangeStaticSplit.mlir"));
    }

    SECTION("Std")
    {
        REQUIRE(VerifyLowerToStd(context, true, "TestDynamicLoopRangeStaticSplit_std.mlir"));
    }

    SECTION("LLVM")
    {
        REQUIRE(VerifyTranslateToLLVMIR(context, false, true, "TestDynamicLoopRangeStaticSplit_noopt.ll"));
        REQUIRE(VerifyTranslateToLLVMIR(context, true, true, "TestDynamicLoopRangeStaticSplit.ll"));
    }
}

// Loops:
// M = 256
// N dynamic
// for i = 0 ... M step 1 {
//     dynamic_leftovers = (N - 0) % 64
//     for j_outer = 0 ... (N - dynamic_leftovers) step 64 {
//         for j_middle = 0 ... 64 step 8 {
//             for j_inner = 0 ... 8 step 1 {
//                 j = j_outer + j_middle + j_inner;
//                 kernel(i, j);
//             }
//         }
//     }
//     for j_outer = (N - dynamic_leftovers) ... N step 64 {
//         middle_range = (N - (N - dynamic_leftovers))
//         middle_range = N - N + dynamic_leftovers
//         middle_range = dynamic_leftovers
//         middle_range = N % 64
//         middle_dynamic_leftovers = (middle_range - 0) % 8
//                                  = (N % 64) % 8
//                                  = N % 8 (needs proof?)

//         for j_middle = 0 ... (middle_range - middle_dynamic_leftovers) step 8 {
//             for j_inner = 0 ... 8 step 1 {
//                 j = j_outer + j_middle + j_inner;
//                 kernel(i, j);
//             }
//         }
//         for j_middle = (middle_range - middle_dynamic_leftovers) ... middle_range step 8 {
//             inner_range = middle_range - (middle_range - middle_dynamic_leftovers)
//                         = middle_dynamic_leftovers
//                         = N % 8
//             inner_dynamic_leftovers = (inner_range - 0) % 1
//                                     = (N % 8) % 1
//                                     = 0
//             for j_inner = 0 ... (inner_range - inner_dynamic_leftovers) step 1 {
//                 j = j_outer + j_middle + j_inner;
//                 kernel(i, j);
//             }
//             for j_inner = (inner_range - inner_dynamic_leftovers) ... inner_range step 1 {
//                 j = j_outer + j_middle + j_inner;
//                 kernel(i, j);
//             }
//         }
//     }
// }
TEST_CASE("TestDynamicLoopRangeTwoStaticSplit")
{
    int64_t M = 256;
    int64_t NOuterSplitSize = 64;
    int64_t NInnerSplitSize = 8;
    int64_t dynSize = mlir::ShapedType::kDynamicSize;
    TestContext context(
        "TestDynamicLoopRangeTwoStaticSplit",
        [&]() -> std::vector<mlir::Type> {
                            auto& builder = GetTestBuilder();

                            auto f32Type = builder.getF32Type();
                            auto sizeType = builder.getIndexType();

                            auto inType = mlir::MemRefType::get({ M, dynSize }, f32Type);
                            auto outType = mlir::MemRefType::get({ M }, f32Type);

                           return { inType, sizeType, outType }; },
        [&](std::vector<mlir::Value> args) {
            auto& builder = GetTestBuilder();
            auto loc = builder.getUnknownLoc();

            auto f32Type = builder.getF32Type();

            auto d0 = builder.getAffineDimExpr(0);
            auto d1 = builder.getAffineDimExpr(1);
            auto s0 = builder.getAffineSymbolExpr(0);

            auto inputMatrix = args[0];
            auto columnCount = args[1];
            auto outputVector = args[2];

            auto floatZero = builder.create<mlir::arith::ConstantOp>(loc, f32Type, builder.getFloatAttr(f32Type, 0.0));
            auto indexZero = builder.create<mlir::arith::ConstantIndexOp>(loc, 0);
            auto indexOne = builder.create<mlir::arith::ConstantIndexOp>(loc, 1);

            auto mEnd = builder.create<mlir::arith::ConstantIndexOp>(loc, M);
            auto nEnd = columnCount;
            auto nOuterSplitConstant = builder.create<mlir::arith::ConstantIndexOp>(loc, NOuterSplitSize);
            auto nInnerSplitConstant = builder.create<mlir::arith::ConstantIndexOp>(loc, NInnerSplitSize);

            auto symIdentityMap = builder.getSymbolIdentityMap();
            auto mLoopLBMap = symIdentityMap;
            auto mLoopUBMap = symIdentityMap;

            // The N loops all have main loops and cleanup loops
            auto mainLoopLBMap = symIdentityMap; // Starts at 0
            auto mainLoopUBMap = mlir::AffineMap::get(2, 0, d1 - d0); // Goes up to (end - leftovers)
            auto cleanupLoopLBMap = mainLoopUBMap; // Starts at (end - leftovers)
            auto cleanupLoopUBMap = symIdentityMap; // Goes up to the inner range end

            auto accum = Alloca(builder, mlir::MemRefType::get({ 1 }, f32Type));

            auto mLoop = builder.create<mlir::AffineForOp>(loc, mlir::ValueRange{ indexZero }, mLoopLBMap, mlir::ValueRange{ mEnd }, mLoopUBMap, 1);
            auto mIV = mLoop.getInductionVar();
            auto mLoopBodyBuilder = util::MakeBodyBuilder(mLoop);

            mLoopBodyBuilder.create<mlir::memref::StoreOp>(loc, floatZero, accum, mlir::ValueRange{ indexZero });

            auto buildInnerLoopContents = [&](mlir::OpBuilder& builder, const std::vector<mlir::Value>& nIVs) {
                mlir::AffineExpr nSumExpr = builder.getAffineConstantExpr(0);
                for (size_t idx = 0; idx < nIVs.size(); idx++)
                {
                    nSumExpr = nSumExpr + builder.getAffineDimExpr(idx);
                }
                auto combineNMap = mlir::AffineMap::get(nIVs.size(), 0, nSumExpr);
                mlir::Value combinedN = builder.create<mlir::AffineApplyOp>(loc, combineNMap, nIVs);

                auto currentAccum = builder.create<mlir::memref::LoadOp>(loc, accum, mlir::ValueRange{ indexZero });
                auto currentVal = builder.create<mlir::memref::LoadOp>(loc, inputMatrix, mlir::ValueRange{ mIV, combinedN });
                auto addedVal = builder.create<mlir::arith::AddFOp>(loc, currentAccum, currentVal);

                builder.create<mlir::memref::StoreOp>(loc, addedVal, accum, mlir::ValueRange{ indexZero });
            };

            auto splitLeftoversMap = mlir::AffineMap::get(2, 1, (d1 - d0) % s0); // args: (start, end)[split_constant]
            mlir::Value nOuterLeftoversValue = mLoopBodyBuilder.create<mlir::AffineApplyOp>(loc, splitLeftoversMap, mlir::ValueRange{ indexZero, nEnd, nOuterSplitConstant });
            auto nOuterMainLoop = mLoopBodyBuilder.create<mlir::AffineForOp>(loc, mlir::ValueRange{ indexZero }, mainLoopLBMap, mlir::ValueRange{ nOuterLeftoversValue, nEnd }, mainLoopUBMap, NOuterSplitSize);
            {
                auto nOuterLoopBodyBuilder = util::MakeBodyBuilder(nOuterMainLoop);
                auto nMiddleMainLoop = nOuterLoopBodyBuilder.create<mlir::AffineForOp>(loc, mlir::ValueRange{ indexZero }, mainLoopLBMap, mlir::ValueRange{ indexZero, nOuterSplitConstant }, mainLoopUBMap, NInnerSplitSize);
                {
                    auto nMiddleLoopBodyBuilder = util::MakeBodyBuilder(nMiddleMainLoop);
                    auto nInnerMainLoop = nMiddleLoopBodyBuilder.create<mlir::AffineForOp>(loc, mlir::ValueRange{ indexZero }, mainLoopLBMap, mlir::ValueRange{ indexZero, nInnerSplitConstant }, mainLoopUBMap, NOuterSplitSize);
                    auto nInnerLoopBodyBuilder = util::MakeBodyBuilder(nInnerMainLoop);

                    buildInnerLoopContents(nInnerLoopBodyBuilder, { nOuterMainLoop.getInductionVar(), nMiddleMainLoop.getInductionVar(), nInnerMainLoop.getInductionVar() });
                    // No cleanup loop in this special hard-coded case
                }
                // No cleanup loop in this special hard-coded case
            }

            auto nOuterCleanupLoop = mLoopBodyBuilder.create<mlir::AffineForOp>(loc, mlir::ValueRange{ nOuterLeftoversValue, nEnd }, cleanupLoopLBMap, mlir::ValueRange{ nEnd }, cleanupLoopUBMap, NOuterSplitSize);
            {
                auto nOuterLoopBodyBuilder = util::MakeBodyBuilder(nOuterCleanupLoop);
                auto nMiddleRange = nOuterLeftoversValue;
                mlir::Value nMiddleLeftoversValue = nOuterLoopBodyBuilder.create<mlir::AffineApplyOp>(loc, splitLeftoversMap, mlir::ValueRange{ indexZero, nMiddleRange, nInnerSplitConstant });
                auto nMiddleMainLoop = nOuterLoopBodyBuilder.create<mlir::AffineForOp>(loc, mlir::ValueRange{ indexZero }, mainLoopLBMap, mlir::ValueRange{ nMiddleLeftoversValue, nMiddleRange }, mainLoopUBMap, NInnerSplitSize);
                {
                    auto nMiddleLoopBodyBuilder = util::MakeBodyBuilder(nMiddleMainLoop);
                    auto nInnerMainLoop = nMiddleLoopBodyBuilder.create<mlir::AffineForOp>(loc, mlir::ValueRange{ indexZero }, mainLoopLBMap, mlir::ValueRange{ indexZero, nInnerSplitConstant }, mainLoopUBMap, 1);
                    {
                        auto nInnerLoopBodyBuilder = util::MakeBodyBuilder(nInnerMainLoop);
                        buildInnerLoopContents(nInnerLoopBodyBuilder, { nOuterCleanupLoop.getInductionVar(), nMiddleMainLoop.getInductionVar(), nInnerMainLoop.getInductionVar() });
                    }
                    // No cleanup loop for this hard-coded special case
                }

                auto nMiddleCleanupLoop = nOuterLoopBodyBuilder.create<mlir::AffineForOp>(loc, mlir::ValueRange{ nMiddleLeftoversValue, nMiddleRange }, cleanupLoopLBMap, mlir::ValueRange{ nMiddleRange }, cleanupLoopUBMap, NInnerSplitSize);
                {
                    auto nMiddleLoopBodyBuilder = util::MakeBodyBuilder(nMiddleCleanupLoop);
                    auto nInnerRange = nMiddleLeftoversValue;
                    mlir::Value nInnerLeftoversValue = nMiddleLoopBodyBuilder.create<mlir::AffineApplyOp>(loc, splitLeftoversMap, mlir::ValueRange{ indexZero, nInnerRange, indexOne });
                    auto nInnerMainLoop = nMiddleLoopBodyBuilder.create<mlir::AffineForOp>(loc, mlir::ValueRange{ indexZero }, mainLoopLBMap, mlir::ValueRange{ nInnerLeftoversValue, nInnerRange }, mainLoopUBMap, 1);
                    {
                        auto nInnerLoopBodyBuilder = util::MakeBodyBuilder(nInnerMainLoop);
                        buildInnerLoopContents(nInnerLoopBodyBuilder, { nOuterCleanupLoop.getInductionVar(), nMiddleCleanupLoop.getInductionVar(), nInnerMainLoop.getInductionVar() });
                    }

                    auto nInnerCleanupLoop = nMiddleLoopBodyBuilder.create<mlir::AffineForOp>(loc, mlir::ValueRange{ nInnerLeftoversValue, nInnerRange }, cleanupLoopLBMap, mlir::ValueRange{ nInnerRange }, cleanupLoopUBMap, 1);
                    {
                        auto nInnerLoopBodyBuilder = util::MakeBodyBuilder(nInnerCleanupLoop);
                        buildInnerLoopContents(nInnerLoopBodyBuilder, { nOuterCleanupLoop.getInductionVar(), nMiddleCleanupLoop.getInductionVar(), nInnerCleanupLoop.getInductionVar() });
                    }
                }
            }

            auto finalAccum = mLoopBodyBuilder.create<mlir::memref::LoadOp>(loc, accum, mlir::ValueRange{ indexZero });
            mLoopBodyBuilder.create<mlir::memref::StoreOp>(loc, finalAccum, outputVector, mlir::ValueRange{ mIV });
        });

    SECTION("Parsing")
    {
        REQUIRE(VerifyParse(context, true, "TestDynamicLoopRangeTwoStaticSplit.mlir"));
    }

    SECTION("Std")
    {
        REQUIRE(VerifyLowerToStd(context, true, "TestDynamicLoopRangeTwoStaticSplit_std.mlir"));
    }

    SECTION("LLVM")
    {
        REQUIRE(VerifyTranslateToLLVMIR(context, false, true, "TestDynamicLoopRangeTwoStaticSplit_noopt.ll"));
        REQUIRE(VerifyTranslateToLLVMIR(context, true, true, "TestDynamicLoopRangeTwoStaticSplit.ll"));
    }
}

TEST_CASE("TestDynamicLoopRangeOneStaticSplitAffineConstraints")
{
    int64_t M = 256;
    int64_t NSplitSize = 64;
    int64_t dynSize = mlir::ShapedType::kDynamicSize;
    TestContext context(
        "TestDynamicLoopRangeOneStaticSplitAffineConstraints",
        [&]() -> std::vector<mlir::Type> {
                            auto& builder = GetTestBuilder();

                            auto f32Type = builder.getF32Type();
                            auto sizeType = builder.getIndexType();

                            auto inType = mlir::MemRefType::get({ M, dynSize }, f32Type);
                            auto outType = mlir::MemRefType::get({ M }, f32Type);

                           return { inType, sizeType, outType }; },
        [&](std::vector<mlir::Value> args) {
            auto& builder = GetTestBuilder();
            auto loc = builder.getUnknownLoc();
            auto context = builder.getContext();

            auto inputMatrix = args[0];
            auto columnCount = args[1];
            auto outputVector = args[2];

            AffineConstraintsHelper cstHelper(context);
            cstHelper.SetDebugPrinting(DEBUG_PRINT);

            auto iId = cstHelper.AddDim();
            auto jOuterId = cstHelper.AddDim();
            auto jInnerId = cstHelper.AddDim();

            [[maybe_unused]] auto zeroId = cstHelper.AddConstant(0);
            [[maybe_unused]] auto MId = cstHelper.AddConstant(M);
            [[maybe_unused]] auto NId = cstHelper.AddSymbol(columnCount);

            [[maybe_unused]] auto NSplitId = cstHelper.AddConstant(NSplitSize);

            auto MStepSize = 1;
            auto NOuterStepSize = NSplitSize;
            auto NInnerStepSize = 1;

            std::vector<IdWrapper> loopIds{ iId, jOuterId, jInnerId };
            auto indexZeroVal = builder.create<mlir::arith::ConstantIndexOp>(loc, 0);
            for (auto id : loopIds)
            {
                cstHelper.SetValue(id, indexZeroVal);
            }

            std::vector<PartitionCreationInfo> orderedLoopPartitionInfo;
            PartitionCreationInfo iIdLoopPartitionInfo{ iId, MStepSize };
            PartitionCreationInfo jOuterIdLoopPartitionInfo{ jOuterId, NOuterStepSize };
            PartitionCreationInfo jInnerIdLoopPartitionInfo{ jInnerId, NInnerStepSize };

            orderedLoopPartitionInfo.push_back(iIdLoopPartitionInfo);
            orderedLoopPartitionInfo.push_back(jOuterIdLoopPartitionInfo);
            orderedLoopPartitionInfo.push_back(jInnerIdLoopPartitionInfo);

            // 0 <= N
            cstHelper.AddLowerBound(NId, 0);

            // 0 <= i <= M - 1 < M
            cstHelper.AddLowerBound(iId, 0);
            cstHelper.AddUpperBound(iId, MId);

            // 0 <= jOuter <= N - 1
            cstHelper.AddLowerBound(jOuterId, 0);
            cstHelper.AddUpperBound(jOuterId, NId);

            // 0 <= j_inner <= NSplitSize - 1 < NSplitSize
            cstHelper.AddLowerBound(jInnerId, 0);
            cstHelper.AddUpperBound(jInnerId, NSplitSize);

            // j_outer + j_inner <= N - 1 < N
            // rewritten: N >= j_outer + j_inner + 1
            auto jSumExpr = MakeIdSumExpr({ jOuterId, jInnerId }, context);
            cstHelper.AddLowerBoundExpr(NId, jSumExpr + 1);

            auto buildInnerLoopContents = [&](mlir::OpBuilder& builder, mlir::Value iIV, mlir::Value jIV) {
                auto currentAccum = builder.create<mlir::memref::LoadOp>(loc, outputVector, mlir::ValueRange{ iIV });
                auto currentVal = builder.create<mlir::memref::LoadOp>(loc, inputMatrix, mlir::ValueRange{ iIV, jIV });
                auto addedVal = builder.create<mlir::arith::AddFOp>(loc, currentAccum, currentVal);
                builder.create<mlir::memref::StoreOp>(loc, addedVal, outputVector, mlir::ValueRange{ iIV });
            };

            CreateLoopPartitionsHelper(cstHelper, builder, loc, orderedLoopPartitionInfo, [&](mlir::OpBuilder& bodyBuilder, const std::vector<mlir::Value>& loopIVs) {
                auto iIV = loopIVs[0];
                auto jIV = SumValues(bodyBuilder, std::vector{ loopIVs[1], loopIVs[2] });
                buildInnerLoopContents(bodyBuilder, iIV, jIV);
            });
        });

    SECTION("Parsing")
    {
        REQUIRE(VerifyParse(context, true, "TestDynamicLoopRangeOneStaticSplitAffineConstraints.mlir"));
    }

    SECTION("Std")
    {
        REQUIRE(VerifyLowerToStd(context, true, "TestDynamicLoopRangeOneStaticSplitAffineConstraints_std.mlir"));
    }

    SECTION("LLVM")
    {
        REQUIRE(VerifyTranslateToLLVMIR(context, false, true, "TestDynamicLoopRangeOneStaticSplitAffineConstraints_noopt.ll"));
        REQUIRE(VerifyTranslateToLLVMIR(context, true, true, "TestDynamicLoopRangeOneStaticSplitAffineConstraints.ll"));
    }
}

TEST_CASE("TestDynamicLoopRangeTwoStaticSplitsAffineConstraints")
{
    int64_t M = 256;
    int64_t NOuterSplitSize = 64;
    int64_t NInnerSplitSize = 8;
    int64_t dynSize = mlir::ShapedType::kDynamicSize;
    TestContext context(
        "TestDynamicLoopRangeTwoStaticSplitsAffineConstraints",
        [&]() -> std::vector<mlir::Type> {
                            auto& builder = GetTestBuilder();

                            auto f32Type = builder.getF32Type();
                            auto sizeType = builder.getIndexType();

                            auto inType = mlir::MemRefType::get({ M, dynSize }, f32Type);
                            auto outType = mlir::MemRefType::get({ M }, f32Type);

                           return { inType, sizeType, outType }; },
        [&](std::vector<mlir::Value> args) {
            auto& builder = GetTestBuilder();
            auto loc = builder.getUnknownLoc();
            auto context = builder.getContext();

            auto inputMatrix = args[0];
            auto columnCount = args[1];
            auto outputVector = args[2];

            AffineConstraintsHelper cstHelper(context);
            cstHelper.SetDebugPrinting(DEBUG_PRINT);

            auto iId = cstHelper.AddDim();
            auto jOuterId = cstHelper.AddDim();
            auto jMiddleId = cstHelper.AddDim();
            auto jInnerId = cstHelper.AddDim();

            [[maybe_unused]] auto zeroId = cstHelper.AddConstant(0);
            [[maybe_unused]] auto MId = cstHelper.AddConstant(M);
            [[maybe_unused]] auto NId = cstHelper.AddSymbol(columnCount);

            [[maybe_unused]] auto NOuterSplitId = cstHelper.AddConstant(NOuterSplitSize);
            [[maybe_unused]] auto NInnerSplitId = cstHelper.AddConstant(NInnerSplitSize);

            auto MStepSize = 1;
            auto NOuterStepSize = NOuterSplitSize;
            auto NMiddleStepSize = NInnerSplitSize;
            auto NInnerStepSize = 1;

            std::vector<IdWrapper> loopIds{ iId, jOuterId, jMiddleId, jInnerId };
            auto indexZeroVal = builder.create<mlir::arith::ConstantIndexOp>(loc, 0);
            for (auto id : loopIds)
            {
                cstHelper.SetValue(id, indexZeroVal);
            }

            std::vector<PartitionCreationInfo> orderedLoopPartitionInfo;
            PartitionCreationInfo iIdLoopPartitionInfo{ iId, MStepSize };
            PartitionCreationInfo jOuterIdLoopPartitionInfo{ jOuterId, NOuterStepSize };
            PartitionCreationInfo jMiddleIdLoopPartitionInfo{ jMiddleId, NMiddleStepSize };
            PartitionCreationInfo jInnerIdLoopPartitionInfo{ jInnerId, NInnerStepSize };

            orderedLoopPartitionInfo.push_back(iIdLoopPartitionInfo);
            orderedLoopPartitionInfo.push_back(jOuterIdLoopPartitionInfo);
            orderedLoopPartitionInfo.push_back(jMiddleIdLoopPartitionInfo);
            orderedLoopPartitionInfo.push_back(jInnerIdLoopPartitionInfo);

            // 0 <= N
            cstHelper.AddLowerBound(NId, 0);

            // 0 <= i <= M - 1 < M
            cstHelper.AddLowerBound(iId, 0);
            cstHelper.AddUpperBound(iId, MId);

            // 0 <= jOuter <= N - 1
            cstHelper.AddLowerBound(jOuterId, 0);
            cstHelper.AddUpperBound(jOuterId, NId);

            // 0 <= jMiddle <= NOuterSplitSize - 1 < NOuterSplitSize = 64
            cstHelper.AddLowerBound(jMiddleId, 0);
            cstHelper.AddUpperBound(jMiddleId, NOuterSplitSize);

            // 0 <= j_inner <= NInnerSplitSize - 1 < NInnerSplitSize
            cstHelper.AddLowerBound(jInnerId, 0);
            cstHelper.AddUpperBound(jInnerId, NInnerSplitSize);

            // j_outer + j_middle + j_inner <= N - 1 < N
            // rewritten: N >= j_outer + j_middle + j_inner + 1
            auto jSumExpr = MakeIdSumExpr({ jOuterId, jMiddleId, jInnerId }, context);
            cstHelper.AddLowerBoundExpr(NId, jSumExpr + 1);

            // j_middle + j_inner <= NOuterSplitSize - 1 < NOuterSplitSize
            // rewritten: NOuterSplitSize >= j_middle + j_inner + 1
            auto jMiddleInnerSumExpr = MakeIdSumExpr({ jMiddleId, jInnerId }, context);
            cstHelper.AddLowerBoundExpr(NOuterSplitId, jMiddleInnerSumExpr + 1);

            auto buildInnerLoopContents = [&](mlir::OpBuilder& builder, mlir::Value iIV, mlir::Value jIV) {
                auto currentAccum = builder.create<mlir::memref::LoadOp>(loc, outputVector, mlir::ValueRange{ iIV });
                auto currentVal = builder.create<mlir::memref::LoadOp>(loc, inputMatrix, mlir::ValueRange{ iIV, jIV });
                auto addedVal = builder.create<mlir::arith::AddFOp>(loc, currentAccum, currentVal);
                builder.create<mlir::memref::StoreOp>(loc, addedVal, outputVector, mlir::ValueRange{ iIV });
            };

            CreateLoopPartitionsHelper(cstHelper, builder, loc, orderedLoopPartitionInfo, [&](mlir::OpBuilder& bodyBuilder, const std::vector<mlir::Value>& loopIVs) {
                auto iIV = loopIVs[0];
                auto jIV = SumValues(bodyBuilder, std::vector{ loopIVs[1], loopIVs[2], loopIVs[3] });
                buildInnerLoopContents(bodyBuilder, iIV, jIV);
            });
        });

    SECTION("Parsing")
    {
        REQUIRE(VerifyParse(context, true, "TestDynamicLoopRangeTwoStaticSplitsAffineConstraints.mlir"));
    }

    SECTION("Std")
    {
        REQUIRE(VerifyLowerToStd(context, true, "TestDynamicLoopRangeTwoStaticSplitsAffineConstraints_std.mlir"));
    }

    SECTION("LLVM")
    {
        REQUIRE(VerifyTranslateToLLVMIR(context, false, true, "TestDynamicLoopRangeTwoStaticSplitsAffineConstraints_noopt.ll"));
        REQUIRE(VerifyTranslateToLLVMIR(context, true, true, "TestDynamicLoopRangeTwoStaticSplitsAffineConstraints.ll"));
    }
}

TEST_CASE("TestDynamicLoopRangeAffineConstraintsOneStaticDimSplitTwoDynamicDimSplit")
{
    int64_t M = 256;
    int64_t MSplitSize = 50;
    int64_t NOuterSplitSize = 64;
    int64_t NInnerSplitSize = 8;
    int64_t dynSize = mlir::ShapedType::kDynamicSize;
    TestContext context(
        "TestDynamicLoopRangeAffineConstraintsOneStaticDimSplitTwoDynamicDimSplit",
        [&]() -> std::vector<mlir::Type> {
                            auto& builder = GetTestBuilder();

                            auto f32Type = builder.getF32Type();
                            auto sizeType = builder.getIndexType();

                            auto inType = mlir::MemRefType::get({ M, dynSize }, f32Type);
                            auto outType = mlir::MemRefType::get({ M }, f32Type);

                           return { inType, sizeType, outType }; },
        [&](std::vector<mlir::Value> args) {
            auto& builder = GetTestBuilder();
            auto loc = builder.getUnknownLoc();
            auto context = builder.getContext();

            auto inputMatrix = args[0];
            auto columnCount = args[1];
            auto outputVector = args[2];

            AffineConstraintsHelper cstHelper(context);
            cstHelper.SetDebugPrinting(DEBUG_PRINT);

            auto iOuterId = cstHelper.AddDim();
            auto jOuterId = cstHelper.AddDim();
            auto iInnerId = cstHelper.AddDim();
            auto jMiddleId = cstHelper.AddDim();
            auto jInnerId = cstHelper.AddDim();

            [[maybe_unused]] auto zeroId = cstHelper.AddConstant(0);
            [[maybe_unused]] auto MId = cstHelper.AddConstant(M);
            [[maybe_unused]] auto NId = cstHelper.AddSymbol(columnCount);

            [[maybe_unused]] auto MSplitId = cstHelper.AddConstant(MSplitSize);
            [[maybe_unused]] auto NOuterSplitId = cstHelper.AddConstant(NOuterSplitSize);
            [[maybe_unused]] auto NInnerSplitId = cstHelper.AddConstant(NInnerSplitSize);

            auto MOuterStepSize = MSplitSize;
            auto MInnerStepSize = 1;

            auto NOuterStepSize = NOuterSplitSize;
            auto NMiddleStepSize = NInnerSplitSize;
            auto NInnerStepSize = 1;

            std::vector<IdWrapper> loopIds{ iOuterId, iInnerId, jOuterId, jMiddleId, jInnerId };
            auto indexZeroVal = builder.create<mlir::arith::ConstantIndexOp>(loc, 0);
            for (auto id : loopIds)
            {
                cstHelper.SetValue(id, indexZeroVal);
            }

            std::vector<PartitionCreationInfo> orderedLoopPartitionInfo;
            PartitionCreationInfo iOuterIdLoopPartitionInfo{ iOuterId, MOuterStepSize };
            PartitionCreationInfo iInnerIdLoopPartitionInfo{ iInnerId, MInnerStepSize };
            PartitionCreationInfo jOuterIdLoopPartitionInfo{ jOuterId, NOuterStepSize };
            PartitionCreationInfo jMiddleIdLoopPartitionInfo{ jMiddleId, NMiddleStepSize };
            PartitionCreationInfo jInnerIdLoopPartitionInfo{ jInnerId, NInnerStepSize };

            orderedLoopPartitionInfo.push_back(iOuterIdLoopPartitionInfo);
            orderedLoopPartitionInfo.push_back(jOuterIdLoopPartitionInfo);
            orderedLoopPartitionInfo.push_back(iInnerIdLoopPartitionInfo);
            orderedLoopPartitionInfo.push_back(jMiddleIdLoopPartitionInfo);
            orderedLoopPartitionInfo.push_back(jInnerIdLoopPartitionInfo);

            // 0 <= N
            cstHelper.AddLowerBound(NId, 0);

            // 0 <= i_outer <= M - 1 < M
            cstHelper.AddLowerBound(iOuterId, 0);
            cstHelper.AddUpperBound(iOuterId, MId);

            // 0 <= i_inner <= MSplitSize - 1 < MSplitSize
            cstHelper.AddLowerBound(iInnerId, 0);
            cstHelper.AddUpperBound(iInnerId, MSplitSize);

            // 0 <= jOuter <= N - 1
            cstHelper.AddLowerBound(jOuterId, 0);
            cstHelper.AddUpperBound(jOuterId, NId);

            // 0 <= jMiddle <= NOuterSplitSize - 1 < NOuterSplitSize = 64
            cstHelper.AddLowerBound(jMiddleId, 0);
            cstHelper.AddUpperBound(jMiddleId, NOuterSplitSize);

            // 0 <= j_inner <= NInnerSplitSize - 1 < NInnerSplitSize
            cstHelper.AddLowerBound(jInnerId, 0);
            cstHelper.AddUpperBound(jInnerId, NInnerSplitSize);

            // i_outer + i_inner <= M - 1 < M
            // rewritten: M >= i_outer + i_inner + 1
            auto iSumExpr = MakeIdSumExpr({ iOuterId, iInnerId }, context);
            cstHelper.AddLowerBoundExpr(MId, iSumExpr + 1);

            // j_outer + j_middle + j_inner <= N - 1 < N
            // rewritten: N >= j_outer + j_middle + j_inner + 1
            auto jSumExpr = MakeIdSumExpr({ jOuterId, jMiddleId, jInnerId }, context);
            cstHelper.AddLowerBoundExpr(NId, jSumExpr + 1);

            // j_middle + j_inner <= NOuterSplitSize - 1 < NOuterSplitSize
            // rewritten: NOuterSplitSize >= j_middle + j_inner + 1
            auto jMiddleInnerSumExpr = MakeIdSumExpr({ jMiddleId, jInnerId }, context);
            cstHelper.AddLowerBoundExpr(NOuterSplitId, jMiddleInnerSumExpr + 1);

            auto buildInnerLoopContents = [&](mlir::OpBuilder& builder, mlir::Value iIV, mlir::Value jIV) {
                auto currentAccum = builder.create<mlir::memref::LoadOp>(loc, outputVector, mlir::ValueRange{ iIV });
                auto currentVal = builder.create<mlir::memref::LoadOp>(loc, inputMatrix, mlir::ValueRange{ iIV, jIV });
                auto addedVal = builder.create<mlir::arith::AddFOp>(loc, currentAccum, currentVal);
                builder.create<mlir::memref::StoreOp>(loc, addedVal, outputVector, mlir::ValueRange{ iIV });
            };

            CreateLoopPartitionsHelper(cstHelper, builder, loc, orderedLoopPartitionInfo, [&](mlir::OpBuilder& bodyBuilder, const std::vector<mlir::Value>& loopIVs) {
                auto iIV = SumValues(bodyBuilder, std::vector{ loopIVs[0], loopIVs[2] });
                auto jIV = SumValues(bodyBuilder, std::vector{ loopIVs[1], loopIVs[3], loopIVs[4] });
                buildInnerLoopContents(bodyBuilder, iIV, jIV);
            });
        });

    SECTION("Parsing")
    {
        REQUIRE(VerifyParse(context, true, "TestDynamicLoopRangeAffineConstraintsOneStaticDimSplitTwoDynamicDimSplit.mlir"));
    }

    SECTION("Std")
    {
        REQUIRE(VerifyLowerToStd(context, true, "TestDynamicLoopRangeAffineConstraintsOneStaticDimSplitTwoDynamicDimSplit_std.mlir"));
    }

    SECTION("LLVM")
    {
        REQUIRE(VerifyTranslateToLLVMIR(context, false, true, "TestDynamicLoopRangeAffineConstraintsOneStaticDimSplitTwoDynamicDimSplit_noopt.ll"));
        REQUIRE(VerifyTranslateToLLVMIR(context, true, true, "TestDynamicLoopRangeAffineConstraintsOneStaticDimSplitTwoDynamicDimSplit.ll"));
    }
}

TEST_CASE("TestDynamicLoopRangeAffineConstraintsOneStaticDimSplitTwoDynamicDimSplitCleanupMiddle")
{
    int64_t M = 256;
    int64_t MSplitSize = 50;
    int64_t NOuterSplitSize = 64;
    int64_t NInnerSplitSize = 6;
    int64_t dynSize = mlir::ShapedType::kDynamicSize;
    TestContext context(
        "TestDynamicLoopRangeAffineConstraintsOneStaticDimSplitTwoDynamicDimSplitCleanupMiddle",
        [&]() -> std::vector<mlir::Type> {
                            auto& builder = GetTestBuilder();

                            auto f32Type = builder.getF32Type();
                            auto sizeType = builder.getIndexType();

                            auto inType = mlir::MemRefType::get({ M, dynSize }, f32Type);
                            auto outType = mlir::MemRefType::get({ M }, f32Type);

                           return { inType, sizeType, outType }; },
        [&](std::vector<mlir::Value> args) {
            auto& builder = GetTestBuilder();
            auto loc = builder.getUnknownLoc();
            auto context = builder.getContext();

            auto inputMatrix = args[0];
            auto columnCount = args[1];
            auto outputVector = args[2];

            AffineConstraintsHelper cstHelper(context);
            cstHelper.SetDebugPrinting(DEBUG_PRINT);

            auto iOuterId = cstHelper.AddDim();
            auto jOuterId = cstHelper.AddDim();
            auto iInnerId = cstHelper.AddDim();
            auto jMiddleId = cstHelper.AddDim();
            auto jInnerId = cstHelper.AddDim();

            [[maybe_unused]] auto zeroId = cstHelper.AddConstant(0);
            [[maybe_unused]] auto MId = cstHelper.AddConstant(M);
            [[maybe_unused]] auto NId = cstHelper.AddSymbol(columnCount);

            [[maybe_unused]] auto MSplitId = cstHelper.AddConstant(MSplitSize);
            [[maybe_unused]] auto NOuterSplitId = cstHelper.AddConstant(NOuterSplitSize);
            [[maybe_unused]] auto NInnerSplitId = cstHelper.AddConstant(NInnerSplitSize);

            auto MOuterStepSize = MSplitSize;
            auto MInnerStepSize = 1;

            auto NOuterStepSize = NOuterSplitSize;
            auto NMiddleStepSize = NInnerSplitSize;
            auto NInnerStepSize = 1;

            std::vector<IdWrapper> loopIds{ iOuterId, iInnerId, jOuterId, jMiddleId, jInnerId };
            auto indexZeroVal = builder.create<mlir::arith::ConstantIndexOp>(loc, 0);
            for (auto id : loopIds)
            {
                cstHelper.SetValue(id, indexZeroVal);
            }

            std::vector<PartitionCreationInfo> orderedLoopPartitionInfo;
            PartitionCreationInfo iOuterIdLoopPartitionInfo{ iOuterId, MOuterStepSize };
            PartitionCreationInfo iInnerIdLoopPartitionInfo{ iInnerId, MInnerStepSize };
            PartitionCreationInfo jOuterIdLoopPartitionInfo{ jOuterId, NOuterStepSize };
            PartitionCreationInfo jMiddleIdLoopPartitionInfo{ jMiddleId, NMiddleStepSize };
            PartitionCreationInfo jInnerIdLoopPartitionInfo{ jInnerId, NInnerStepSize };

            orderedLoopPartitionInfo.push_back(iOuterIdLoopPartitionInfo);
            orderedLoopPartitionInfo.push_back(jOuterIdLoopPartitionInfo);
            orderedLoopPartitionInfo.push_back(iInnerIdLoopPartitionInfo);
            orderedLoopPartitionInfo.push_back(jMiddleIdLoopPartitionInfo);
            orderedLoopPartitionInfo.push_back(jInnerIdLoopPartitionInfo);

            // 0 <= N
            cstHelper.AddLowerBound(NId, 0);

            // 0 <= i_outer <= M - 1 < M
            cstHelper.AddLowerBound(iOuterId, 0);
            cstHelper.AddUpperBound(iOuterId, MId);

            // 0 <= i_inner <= MSplitSize - 1 < MSplitSize
            cstHelper.AddLowerBound(iInnerId, 0);
            cstHelper.AddUpperBound(iInnerId, MSplitSize);

            // 0 <= jOuter <= N - 1
            cstHelper.AddLowerBound(jOuterId, 0);
            cstHelper.AddUpperBound(jOuterId, NId);

            // 0 <= jMiddle <= NOuterSplitSize - 1 < NOuterSplitSize = 64
            cstHelper.AddLowerBound(jMiddleId, 0);
            cstHelper.AddUpperBound(jMiddleId, NOuterSplitSize);

            // 0 <= j_inner <= NInnerSplitSize - 1 < NInnerSplitSize
            cstHelper.AddLowerBound(jInnerId, 0);
            cstHelper.AddUpperBound(jInnerId, NInnerSplitSize);

            // i_outer + i_inner <= M - 1 < M
            // rewritten: M >= i_outer + i_inner + 1
            auto iSumExpr = MakeIdSumExpr({ iOuterId, iInnerId }, context);
            cstHelper.AddLowerBoundExpr(MId, iSumExpr + 1);

            // j_outer + j_middle + j_inner <= N - 1 < N
            // rewritten: N >= j_outer + j_middle + j_inner + 1
            auto jSumExpr = MakeIdSumExpr({ jOuterId, jMiddleId, jInnerId }, context);
            cstHelper.AddLowerBoundExpr(NId, jSumExpr + 1);

            // j_middle + j_inner <= NOuterSplitSize - 1 < NOuterSplitSize
            // rewritten: NOuterSplitSize >= j_middle + j_inner + 1
            auto jMiddleInnerSumExpr = MakeIdSumExpr({ jMiddleId, jInnerId }, context);
            cstHelper.AddLowerBoundExpr(NOuterSplitId, jMiddleInnerSumExpr + 1);

            auto buildInnerLoopContents = [&](mlir::OpBuilder& builder, mlir::Value iIV, mlir::Value jIV) {
                auto currentAccum = builder.create<mlir::memref::LoadOp>(loc, outputVector, mlir::ValueRange{ iIV });
                auto currentVal = builder.create<mlir::memref::LoadOp>(loc, inputMatrix, mlir::ValueRange{ iIV, jIV });
                auto addedVal = builder.create<mlir::arith::AddFOp>(loc, currentAccum, currentVal);
                builder.create<mlir::memref::StoreOp>(loc, addedVal, outputVector, mlir::ValueRange{ iIV });
            };

            CreateLoopPartitionsHelper(cstHelper, builder, loc, orderedLoopPartitionInfo, [&](mlir::OpBuilder& bodyBuilder, const std::vector<mlir::Value>& loopIVs) {
                auto iIV = SumValues(bodyBuilder, std::vector{ loopIVs[0], loopIVs[2] });
                auto jIV = SumValues(bodyBuilder, std::vector{ loopIVs[1], loopIVs[3], loopIVs[4] });
                buildInnerLoopContents(bodyBuilder, iIV, jIV);
            });
        });

    SECTION("Parsing")
    {
        REQUIRE(VerifyParse(context, true, "TestDynamicLoopRangeAffineConstraintsOneStaticDimSplitTwoDynamicDimSplitCleanupMiddle.mlir"));
    }

    SECTION("Std")
    {
        REQUIRE(VerifyLowerToStd(context, true, "TestDynamicLoopRangeAffineConstraintsOneStaticDimSplitTwoDynamicDimSplitCleanupMiddle_std.mlir"));
    }

    SECTION("LLVM")
    {
        REQUIRE(VerifyTranslateToLLVMIR(context, false, true, "TestDynamicLoopRangeAffineConstraintsOneStaticDimSplitTwoDynamicDimSplitCleanupMiddle_noopt.ll"));
        REQUIRE(VerifyTranslateToLLVMIR(context, true, true, "TestDynamicLoopRangeAffineConstraintsOneStaticDimSplitTwoDynamicDimSplitCleanupMiddle.ll"));
    }
}