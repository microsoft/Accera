////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
////////////////////////////////////////////////////////////////////////////////////////////////////

// TODO : merge with nest/AffineConstraints.cpp
#include "nest/LoopNestAffineConstraints.h"

#include <utilities/include/Exception.h>

#include <algorithm>

using namespace accera::ir;
using namespace accera::ir::loopnest;

using IdWrapper = util::AffineConstraintsHelper::IdWrapper;

namespace
{

struct SplitPartitionInfo
{
    mlir::Value partitionValue;
    mlir::Value largestMainLoopIVValue;

    mlir::AffineValueMap partitionValueMap;
    mlir::AffineValueMap localExprAffineValueMap;
    mlir::AffineValueMap largestMainLoopIVValueMap;
};

SplitPartitionInfo MakeSplitPartition(mlir::OpBuilder& builder, mlir::Value begin, mlir::Value end, int64_t splitSize)
{
    SplitPartitionInfo info;
    auto d0 = builder.getAffineDimExpr(0);
    auto d1 = builder.getAffineDimExpr(1);
    auto loc = builder.getUnknownLoc(); // TODO : plumb location

    // The split partition separates the range that runs full splitSize iteration chunks from the range that runs the final incomplete cleanup iteration
    // cleanup_amount = (end - begin) % splitSize
    // partition_value = end - cleanup_amount

    // Note: x mod y = x - (x floordiv y) * y
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

struct SplitLoopInfo
{
    Index loopIndex;
    IdWrapper loopId;
    int64_t stepSize;
    IdWrapper partitionValueId;
    IdWrapper largestMainLoopIVId;
};

std::optional<SplitLoopInfo> AddSplitPartitionHelper(LoopNestAffineConstraints& cst,
                                                     const Index& loopIndex,
                                                     mlir::OpBuilder& builder,
                                                     mlir::Location loc,
                                                     int64_t stepSize)
{
    // Get the [begin, end) range for this loop id
    LoopNestAffineConstraints resolveRangeCst = cst.Clone();

    auto [beginValueMap, endValueMap] = resolveRangeCst.GetLowerAndUpperBound(loopIndex, builder, loc);

    // Produce a begin and end value using affine apply ops
    auto beginApplyOp = mlir::makeComposedAffineApply(builder, loc, beginValueMap.getAffineMap(), beginValueMap.getOperands());
    auto endApplyOp = mlir::makeComposedAffineApply(builder, loc, endValueMap.getAffineMap(), endValueMap.getOperands());

    // If either the begin or end values are empty, then we've recursed into an empty part of the space and we should bail out without creating a loop
    auto beginMap = beginApplyOp.getAffineMap();
    auto endMap = endApplyOp.getAffineMap();

    if (beginMap.isEmpty() || endMap.isEmpty())
    {
        return std::nullopt;
    }

    mlir::Value beginVal = beginApplyOp.getResult();
    mlir::Value endVal = endApplyOp.getResult();

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

    IdWrapper loopId = cst.GetId(loopIndex);
    IdWrapper partitionValueId = cst.AddSymbol(partitionInfo.partitionValue);
    IdWrapper largestMainLoopIVId = cst.AddSymbol(partitionInfo.largestMainLoopIVValue);
    SplitLoopInfo info{ loopIndex,
                        loopId,
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

} // namespace

namespace accera::ir
{
namespace loopnest
{
    LoopNestAffineConstraints::LoopNestAffineConstraints(const std::vector<Index>& orderedIndices, mlir::MLIRContext* context) :
        util::AffineConstraintsHelper(context),
        _orderedIndices(orderedIndices)
    {
        for (auto& index : orderedIndices)
        {
            _indexIds.insert({ index, AddDim() });
        }
    }

    void LoopNestAffineConstraints::AddConstraint(const Index& index, Range range)
    {
        auto id = GetId(index);

        auto [beginId, endId] = AddRange(range);
        AddLowerBound(id, beginId);
        AddUpperBound(id, endId);
    }

    void LoopNestAffineConstraints::SetValue(const Index& index, mlir::Value val)
    {
        auto id = GetId(index);
        AffineConstraintsHelper::SetValue(id, val);
    }

    std::pair<mlir::AffineValueMap, mlir::AffineValueMap> LoopNestAffineConstraints::GetLowerAndUpperBound(const Index& index, mlir::OpBuilder& builder, mlir::Location loc) const
    {
        // Assumes that _orderedIndices is in the loopnest order, so everything after `index` in the vector is an index deeper in the nest
        auto idsToProjectOut = GetInnerIds(index);

        auto id = GetId(index);
        return GetLowerAndUpperBound(id, builder, loc, idsToProjectOut);
    }

    IdWrapper LoopNestAffineConstraints::GetId(const Index& index) const
    {
        auto findIter = _indexIds.find(index);
        assert(findIter != _indexIds.end() && "Index isn't part of constraint system");
        return findIter->second;
    }

    std::vector<IdWrapper> LoopNestAffineConstraints::GetIds(const std::vector<Index>& indices) const
    {
        std::vector<IdWrapper> ids;
        std::transform(indices.begin(), indices.end(), std::back_inserter(ids), [&](const Index& index) {
            return GetId(index);
        });
        return ids;
    }

    mlir::AffineExpr LoopNestAffineConstraints::GetAlignedIndexSumExpr(const std::vector<Index>& indices) const
    {
        auto ids = GetIds(indices);
        mlir::AffineExpr expr = mlir::getAffineConstantExpr(0, _context);
        auto context = GetContext();
        for (auto& id : ids)
        {
            expr = expr + id.GetExpr(context);
        }
        return expr;
    }

    IdWrapper LoopNestAffineConstraints::AddRangeBegin(const Range& range, std::optional<mlir::AffineExpr> localExpr)
    {
        if (range.HasConstantBegin())
        {
            return AddConstant(range.Begin());
        }
        else
        {
            assert(range.HasValueMapBegin());
            return AddSymbol(range.ValueMapBegin(), localExpr);
        }
    }

    IdWrapper LoopNestAffineConstraints::AddRangeEnd(const Range& range, std::optional<mlir::AffineExpr> localExpr)
    {
        if (range.HasConstantEnd())
        {
            return AddConstant(range.End());
        }
        if (range.HasValueMapEnd())
        {
            return AddSymbol(range.ValueMapEnd(), localExpr);
        }
        else if (range.HasVariableEnd())
        {
            return AddSymbol(range.VariableEnd());
        }
        else if (range.HasIndexEnd())
        {
            auto endIndex = range.EndIndex();
            return GetId(endIndex);
        }
        else
        {
            // Operand or symbol name end values are unsupported
            throw accera::utilities::InputException(accera::utilities::InputExceptionErrors::invalidArgument, "Can only constrain an index by a constant, symbol, or other index");
        }
    }

    std::pair<IdWrapper, IdWrapper> LoopNestAffineConstraints::AddRange(const Range& range, std::optional<mlir::AffineExpr> localExpr)
    {
        auto beginId = AddRangeBegin(range, localExpr);
        auto endId = AddRangeEnd(range, localExpr);
        return std::make_pair(beginId, endId);
    }

    void LoopNestAffineConstraints::AddIndexSumConstraint(const std::vector<Index>& indices, const Range& range)
    {
        auto [lowerBoundId, upperBoundId] = AddRange(range);
        auto sumExpr = GetAlignedIndexSumExpr(indices);

        // lowerBoundId <= sum(indices) < upperBoundId
        // So add constraints:
        //      lowerBoundId <= sum(indices) as an inclusive upper bound on lowerBoundId
        //      sum(indices) < upperBoundId as a lower bound on upperBoundId
        //          Note: since the inequalities are all <= inequalities, modify this to be sum(indices) + 1 <= upperBoundId
        AddUpperBoundExpr(lowerBoundId, sumExpr, false /* exclusive */);
        AddLowerBoundExpr(upperBoundId, sumExpr + 1);
    }

    std::vector<LoopPartitionConstraints> LoopNestAffineConstraints::SplitIndex(mlir::OpBuilder& builder, mlir::Location loc, const Index& index, int64_t splitSize) const
    {
        // Create the partition for this loop level
        auto levelScopedConstraints = Clone();
        auto loopId = levelScopedConstraints.GetId(index);

        auto partitionInfoOpt = AddSplitPartitionHelper(levelScopedConstraints,
                                                        index,
                                                        builder,
                                                        loc,
                                                        splitSize);

        std::vector<LoopPartitionConstraints> partitionedLoopConstraints;
        if (!partitionInfoOpt.has_value())
        {
            return partitionedLoopConstraints;
        }
        auto partitionInfo = *partitionInfoOpt;

        // Main loop partition
        {
            // Fork the constraints for inside the main loop
            auto mainScopedConstraints = levelScopedConstraints.Clone();

            // Fork the constraints for resolving the current main loop bounds
            auto mainResolveConstraints = levelScopedConstraints.Clone();

            // Bound loopId <= largest main loop IV
            mainScopedConstraints.AddUpperBound(loopId, partitionInfo.largestMainLoopIVId, false /*exclusive*/);

            // Bound loopId <= partition value. This is a looser constraint than we put on the mainScopedConstraints, but it is helpful
            // for getting a simpler loop bound
            mainResolveConstraints.AddUpperBound(loopId, partitionInfo.partitionValueId);

            LoopPartitionConstraints mainPartitionConstraints(mainResolveConstraints, mainScopedConstraints);
            partitionedLoopConstraints.push_back(mainPartitionConstraints);
        }

        // Cleanup loop partition
        {
            // Fork the constraints for inside the cleanup loop
            auto cleanupScopedConstraints = levelScopedConstraints.Clone();

            // Fork the constraints for resolving the current cleanup loop bounds
            auto cleanupResolveConstraints = levelScopedConstraints.Clone();

            // Set loop id equal to partition value inside the cleanup loop
            cleanupScopedConstraints.SetEqual(loopId, partitionInfo.partitionValueId);

            // Bound loopId >= partition value.
            cleanupResolveConstraints.AddLowerBound(loopId, partitionInfo.partitionValueId);

            LoopPartitionConstraints cleanupPartitionConstraints(cleanupResolveConstraints, cleanupScopedConstraints);
            partitionedLoopConstraints.push_back(cleanupPartitionConstraints);
        }

        return partitionedLoopConstraints;
    }

    std::vector<LoopPartitionConstraints> LoopNestAffineConstraints::PartitionIndex(mlir::OpBuilder& builder, mlir::Location loc, const Index& index, int64_t partitionValue) const
    {
        return PartitionIndex(builder, loc, index, std::set<int64_t>{ partitionValue });
    }

    std::vector<LoopPartitionConstraints> LoopNestAffineConstraints::PartitionIndex(mlir::OpBuilder& builder, mlir::Location loc, const Index& index, const std::set<int64_t> partitionValues) const
    {
        std::vector<LoopPartitionConstraints> partitionedLoopConstraints;
        int64_t previousPartitionValue = -1;
        for (auto partitionValue : partitionValues)
        {
            // Fork the constraints for inside the loop
            LoopNestAffineConstraints scopedConstraints = Clone();
            auto scopedLoopId = scopedConstraints.GetId(index);

            // Bound loopId <= next partition value
            scopedConstraints.AddUpperBound(scopedLoopId, partitionValue);

            // Bound loopId >= previous partition value if it has been set
            if (previousPartitionValue != -1)
            {
                scopedConstraints.AddLowerBound(scopedLoopId, previousPartitionValue);
            }
            previousPartitionValue = partitionValue;

            // Fork the constraints for resolving the current loop bounds
            LoopNestAffineConstraints resolveConstraints = scopedConstraints.Clone();

            LoopPartitionConstraints resolvePartitionConstraints(resolveConstraints, scopedConstraints);
            partitionedLoopConstraints.push_back(resolvePartitionConstraints);
        }

        // Remaining range up to the end of the initial range
        {
            // Fork the constraints for inside the loop
            LoopNestAffineConstraints scopedConstraints = Clone();
            auto scopedLoopId = scopedConstraints.GetId(index);

            // Bound loopId >= previous partition value if it has been set
            if (previousPartitionValue != -1)
            {
                scopedConstraints.AddLowerBound(scopedLoopId, previousPartitionValue);
            }

            // Fork the constraints for resolving the current loop bounds
            LoopNestAffineConstraints resolveConstraints = scopedConstraints.Clone();

            LoopPartitionConstraints resolvePartitionConstraints(resolveConstraints, scopedConstraints);
            partitionedLoopConstraints.push_back(resolvePartitionConstraints);
        }

        return partitionedLoopConstraints;
    }

    std::vector<IdWrapper> LoopNestAffineConstraints::GetInnerIds(const Index& index) const
    {
        std::vector<IdWrapper> innerIds;
        auto findIter = std::find(_orderedIndices.begin(), _orderedIndices.end(), index);
        std::transform(findIter + 1, _orderedIndices.end(), std::back_inserter(innerIds), [&](const Index& innerIndex) {
            return GetId(innerIndex);
        });
        return innerIds;
    }

    LoopNestAffineConstraints LoopNestAffineConstraints::Clone() const
    {
        return LoopNestAffineConstraints(*this);
    }

} // namespace loopnest
} // namespace accera::ir