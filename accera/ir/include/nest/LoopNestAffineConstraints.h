////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
////////////////////////////////////////////////////////////////////////////////////////////////////

// TODO : merge with AffineConstraints.h
#pragma once

#include <ir/include/AffineConstraintsHelper.h>

#include "Index.h"
#include "Range.h"


#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/Affine/IR/AffineValueMap.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/MLIRContext.h>

#include <optional>
#include <set>
#include <unordered_map>
#include <vector>

namespace accera::ir
{
namespace loopnest
{
    struct LoopPartitionConstraints;
    class LoopNestAffineConstraints : public util::AffineConstraintsHelper
    {
    public:
        using IdWrapper = util::AffineConstraintsHelper::IdWrapper;

        LoopNestAffineConstraints(const std::vector<Index>& orderedIndices, mlir::MLIRContext* context);
        LoopNestAffineConstraints(const LoopNestAffineConstraints& other) = default;

        void AddConstraint(const Index& index, Range range);
        void SetValue(const Index& index, mlir::Value val);

        mlir::AffineExpr GetAlignedIndexSumExpr(const std::vector<Index>& indices) const;

        std::pair<IdWrapper, IdWrapper> AddRange(const Range& range, std::optional<mlir::AffineExpr> localExpr = std::nullopt);
        IdWrapper AddRangeBegin(const Range& range, std::optional<mlir::AffineExpr> localExpr = std::nullopt);
        IdWrapper AddRangeEnd(const Range& range, std::optional<mlir::AffineExpr> localExpr = std::nullopt);
        void AddIndexSumConstraint(const std::vector<Index>& indices, const Range& range);

        std::pair<mlir::AffineValueMap, mlir::AffineValueMap> GetLowerAndUpperBound(const Index& index, mlir::OpBuilder& builder, mlir::Location loc) const;
        using AffineConstraintsHelper::GetLowerAndUpperBound;

        std::vector<LoopPartitionConstraints> SplitIndex(mlir::OpBuilder& builder, mlir::Location loc, const Index& index, int64_t splitSize) const;

        std::vector<LoopPartitionConstraints> PartitionIndex(mlir::OpBuilder& builder, mlir::Location loc, const Index& index, int64_t partitionValue) const;
        std::vector<LoopPartitionConstraints> PartitionIndex(mlir::OpBuilder& builder, mlir::Location loc, const Index& index, const std::set<int64_t> partitionValues) const;

        IdWrapper GetId(const Index& index) const;
        std::vector<IdWrapper> GetIds(const std::vector<Index>& indices) const;
        std::vector<IdWrapper> GetInnerIds(const Index& index) const;

        LoopNestAffineConstraints Clone() const;

    private:
        std::unordered_map<Index, IdWrapper> _indexIds;
        std::vector<Index> _orderedIndices;
    };

    struct LoopPartitionConstraints
    {
        LoopPartitionConstraints(const LoopNestAffineConstraints& resolveConstraints_, const LoopNestAffineConstraints& innerConstraints_) :
            resolveConstraints(resolveConstraints_),
            innerConstraints(innerConstraints_)
        {}

        // Constraints used to resolve the loop range for this partition
        LoopNestAffineConstraints resolveConstraints;

        // Constraints set for the iteration sub-space internal to this partition's loop
        LoopNestAffineConstraints innerConstraints;
    };
} // namespace loopnest
} // namespace accera::ir