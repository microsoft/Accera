////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
//  Authors: Chuck Jacobs
////////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

#include "AffineConstraints.h"
#include "AffineExpression.h"
#include "IterationDomain.h"

#include <mlir/IR/MLIRContext.h>
#include <mlir/Analysis/AffineStructures.h>

#include <optional>
#include <ostream>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace accera::ir
{
namespace loopnest
{
    /// <summary>
    /// An `IterationDomain` where some of the dimensions may have been split or otherwise transformed
    /// </summary>
    class TransformedDomain
    {
    public:
        TransformedDomain(const IterationDomain& domain);

        /// Return the number of dimensions in this domain --- the number of original indices before any splitting or other
        /// transformations have been done
        int64_t NumDimensions() const;
        /// Return the dimensions in this domain --- the original indices before any splitting or other transformations have
        /// been done
        std::vector<Index> GetDimensions() const;

        /// Return the number of "loop indices" --- indices that correspond to loops in the schedule using this domain
        int64_t NumLoopIndices() const;
        /// Return the "loop indices" --- indices that correspond to loops in the schedule using this domain
        std::vector<Index> GetAllLoopIndices() const;

        /// Return the total number of indices in the domain --- both "loop indices" and "computed" indices (indices that
        /// have been split and no longer correspond to an acutal loop)
        int64_t NumIndices() const;
        /// Return all of the indices in the domain --- both "loop indices" and "computed" indices (indices that
        /// have been split and no longer correspond to an actual loop)
        std::vector<Index> GetIndices() const;

        bool Exists(const Index& index) const; // any index in this domain
        bool IsDimension(const Index& index) const; // the index corresponding to the original range
        bool IsLoopIndex(const Index& index) const; // a leaf node in the index tree
        bool IsComputedIndex(const Index& index) const;

        Index Pad(const Index& index, int64_t size, bool padFront, mlir::MLIRContext* context);
        SplitIndex Split(const Index& index, int64_t splitSize, mlir::MLIRContext* context);
        Index Skew(const Index& index, const Index& referenceIndex, mlir::MLIRContext* context);

        bool HasConstantDimensionSize(const Index& dimensionIndex) const;
        int64_t GetDimensionSize(const Index& dimensionIndex) const;
        int64_t GetDimensionBegin(const Index& dimensionIndex) const;
        Range GetIndexRange(const Index& index) const;
        std::vector<int64_t> GetIndexPadding(const Index& index) const;
        AffineExpression GetIndexExpr(const Index& index) const;
        AffineExpression GetReducedIndexExpr(const Index& index, mlir::MLIRContext* context) const; // gets an expression that depends only on loop indices

        std::vector<Index> GetLoopIndicesForDimension(const Index& dimensionIndex) const;
        std::vector<Index> GetComputedIndicesForDimension(const Index& dimensionIndex) const;
        std::vector<Index> GetDependentIndices(const Index& index) const; // "dependent" == child. When an index is split, the 2 new indices are dependent indices
        std::vector<Index> GetDependentLoopIndices(const Index& index, bool includeSelf = false) const;

        // TODO: document what "depends on" means (it means that index1 has an expression that depends on the value of index2. E.g., index1 was split and index2 is one of the child indices)
        bool DependsOn(const Index& index1, const Index& index2) const;
        bool HasParentIndex(const Index& index) const;
        std::vector<Index> GetParentIndices(const Index& index) const;
        bool IsSplitIndex(const Index& index, bool inner) const;
        Index GetOtherSplitIndex(const Index& index) const;
        bool IsPaddedIndex(const Index& index) const;
        bool IsFusedPaddedIndex(const Index& index) const;
        bool IsPrePaddedIndexOf(const Index& index, const Index& paddedIndex) const;

        std::vector<Index> GetBaseIndices(const Index& index) const;
        Index GetBaseIndex(const Index& index) const;
        std::optional<std::pair<bool, Index>> IsSkewedOrReferenceIndex(const Index& index) const;

        void ResolveRangeValues(const std::function<void(Range&)>& resolveFn);

        static TransformedDomain Fuse(const std::vector<TransformedDomain>& domains, const std::vector<std::vector<Index>>& indexCorrespondences);

        void AddDimension(const Index& dimension, const Range& range);

        void Print(std::ostream& os) const;

        AffineConstraints GetConstraints() const;

        // Vector of (index, isDimension, expr, range, padding) tuples
        struct AttributeKey
        {
            std::vector<Index> dimensions;
            std::vector<std::tuple<Index, AffineExpression, Range, std::vector<int64_t>>> indices;
            friend bool operator==(const AttributeKey& a, const AttributeKey& b) { return (a.dimensions == b.dimensions) && (a.indices == b.indices); }
        };
        TransformedDomain(const AttributeKey& info);

    private:
        TransformedDomain() = default;
        void GetBaseIndices(const Index& index, std::unordered_set<Index>& baseIndices) const;

        struct IndexInfo
        {
            IndexInfo() :
                expr({}), range(0, 0), padding({}), parents({}) {}

            IndexInfo(const AffineExpression& expr, const Range& range, std::vector<int64_t> padding, const std::vector<Index>& parents) :
                expr(expr), range(range), padding(padding)
            {
                this->parents.insert(parents.begin(), parents.end());
            }

            AffineExpression expr;
            Range range;
            std::unordered_set<Index> parents;

            // Padding (the sign indicates location in the iteration space):
            //   negative: front-padding (at beginning of index),
            //   positive: back-padding (at end of index)
            // When multiple values are present, this index has been fused with one or more padded indices
            // the padding will follow the correspondence index (i.e. its fusing-index) ordering
            std::vector<int64_t> padding;

            friend bool operator==(const TransformedDomain::IndexInfo& a, const TransformedDomain::IndexInfo& b);
            friend bool operator!=(const TransformedDomain::IndexInfo& a, const TransformedDomain::IndexInfo& b);
        };

        void CollectLoopIndicesForIndex(const Index& index, std::unordered_set<Index>& loopIndices) const;
        void CollectComputedIndicesForIndex(const Index& index, std::unordered_set<Index>& computedIndices) const;

        std::vector<Index> _dimensions;
        std::unordered_set<Index> _loopIndices; // == leaf nodes
        std::unordered_map<Index, IndexInfo> _indices;
    };

    bool operator==(const TransformedDomain& a, const TransformedDomain& b);
    bool operator!=(const TransformedDomain& a, const TransformedDomain& b);
} // namespace loopnest
} // namespace accera::ir
