////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
//  Authors: Chuck Jacobs
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "nest/TransformedDomain.h"

#include <utilities/include/Exception.h>
#include <utilities/include/ZipIterator.h>

#include <mlir/IR/AffineExpr.h>
#include <mlir/IR/AffineMap.h>

#include <algorithm>
#include <cassert>
#include <numeric>
#include <queue>

namespace accera::ir
{
namespace loopnest
{
    namespace
    {
        bool IsSplitExpression(const AffineExpression& expr)
        {
            //  split: d2 = d0 + d1
            auto affineExpr = expr.GetAffineExpr();
            auto binOpExpr = affineExpr.dyn_cast_or_null<mlir::AffineBinaryOpExpr>();
            return binOpExpr && binOpExpr.getKind() == mlir::AffineExprKind::Add &&
                binOpExpr.getRHS().isa<mlir::AffineDimExpr>() && binOpExpr.getLHS().isa<mlir::AffineDimExpr>();
        }

        bool IsSkewExpression(const AffineExpression& expr)
        {
            //  skew: d2 = d0 - d1 = d0 + (d1 * -1)
            auto affineExpr = expr.GetAffineExpr();
            auto binOpExpr = affineExpr.dyn_cast_or_null<mlir::AffineBinaryOpExpr>();
            if (binOpExpr && binOpExpr.getKind() == mlir::AffineExprKind::Add)
            {
                auto rhs = binOpExpr.getRHS().dyn_cast_or_null<mlir::AffineBinaryOpExpr>();
                if (rhs && rhs.getRHS().isa<mlir::AffineConstantExpr>())
                {
                    int64_t coeff = rhs.getRHS().cast<mlir::AffineConstantExpr>().getValue();
                    return coeff == -1;
                }
            }
            return false;
        }
    }

    TransformedDomain::TransformedDomain(const IterationDomain& domain)
    {
        auto ranges = domain.GetRanges();
        for (auto r : ranges)
        {
            auto index = r.GetIndex();
            _dimensions.emplace_back(index);
            _loopIndices.insert(index);
            auto range = r.GetRange();
            _indices[index] = { /*expr=*/{}, range, /*padding=*/{}, /*parents=*/{} };
        }
    }

    TransformedDomain::TransformedDomain(const TransformedDomain::AttributeKey& info)
    {
        _dimensions = info.dimensions;

        for (const auto& [index, expr, range, padding] : info.indices)
        {
            _indices[index] = { expr, range, padding, /*parents=*/ {} };
            if (expr.IsIdentity())
            {
                _loopIndices.insert(index);
            }
        }

        // fix up parents
        for (const auto& indexInfo : _indices)
        {
            for (const auto& expIndex : indexInfo.second.expr.GetIndices())
            {
                _indices[expIndex].parents.insert(indexInfo.first);
            }
        }
    }

    int64_t TransformedDomain::NumDimensions() const
    {
        return static_cast<int64_t>(_dimensions.size());
    }

    std::vector<Index> TransformedDomain::GetDimensions() const
    {
        return _dimensions;
    }

    int64_t TransformedDomain::NumLoopIndices() const
    {
        return static_cast<int64_t>(_loopIndices.size());
    }

    std::vector<Index> TransformedDomain::GetAllLoopIndices() const
    {
        return { _loopIndices.begin(), _loopIndices.end() };
    }

    int64_t TransformedDomain::NumIndices() const
    {
        return static_cast<int64_t>(_indices.size());
    }

    std::vector<Index> TransformedDomain::GetIndices() const
    {
        std::vector<Index> result;
        result.reserve(NumIndices());
        for (const auto& indexInfo : _indices)
        {
            result.push_back(indexInfo.first);
        }
        return result;
    }

    bool TransformedDomain::Exists(const Index& index) const
    {
        return _indices.count(index) != 0;
    }

    bool TransformedDomain::IsLoopIndex(const Index& index) const
    {
        return _loopIndices.count(index) != 0;
    }

    bool TransformedDomain::IsComputedIndex(const Index& index) const
    {
        if (!Exists(index))
        {
            return false;
        }
        return !_indices.at(index).expr.IsIdentity();
    }

    bool TransformedDomain::IsDimension(const Index& index) const
    {
        return std::find(_dimensions.begin(), _dimensions.end(), index) != _dimensions.end();
    }

    Index TransformedDomain::Pad(const Index& index, int64_t size, bool padFront, mlir::MLIRContext* context)
    {
        auto prefix = index.GetName() + "_";
        Index paddedIndex = { prefix + "p"};
        if (_indices.count(index) == 0)
            throw accera::utilities::InputException(accera::utilities::InputExceptionErrors::invalidArgument, "Padding an unknown index");
        if (!IsLoopIndex(index))
            throw accera::utilities::InputException(accera::utilities::InputExceptionErrors::invalidArgument, "Can't pad already-transformed index");

        auto parentRange = _indices[index].range;
        if (!parentRange.HasConstantEnd())
            throw accera::utilities::InputException(accera::utilities::InputExceptionErrors::invalidArgument, "Can't pad non-constant index ranges");
        if (parentRange.Increment() != 1)
            throw accera::utilities::InputException(accera::utilities::InputExceptionErrors::invalidArgument, "Can't pad index ranges with increments != 1");

        // create the padded index
        // Note: this expands the iteration space beyond the original dimensions, so AffineConstraints should be used to enforce
        // the dimension bounds.
        auto N = parentRange.Size();
        if (padFront)
        {
            // index(shape=[0, N)), paddedIndex(shape=[0, N+size))
            _indices[paddedIndex] = { {}, { 0, N + size }, { -size }, { index } };

            // create affine expression for the parent index
            auto padExpr = mlir::getAffineDimExpr(0, context);

            // pad inserts size elements in the beginning of the dimension
            // therefore, the parent index i = i_p - size
            auto parentExpr = AffineExpression(padExpr - size, { paddedIndex });
            _indices[index].expr = parentExpr;
        }
        else
        {
            // index(shape=[0, N)), paddedIndex(shape=[0, N+size))
            _indices[paddedIndex] = { {}, { 0, N + size }, { size }, { index } };

            // create affine expression for the parent index (for consistency - not really necessary)
            auto padExpr = mlir::getAffineDimExpr(0, context);

            // pad inserts size elements at the end of the dimension
            // i = i_p, constrained by i's range
            auto parentExpr = AffineExpression(padExpr, { paddedIndex });
            _indices[index].expr = parentExpr;
        }

        _loopIndices.erase(index);
        _loopIndices.insert(paddedIndex);
        return paddedIndex;
    }

    SplitIndex TransformedDomain::Split(const Index& index, int64_t splitSize, mlir::MLIRContext* context)
    {
        auto prefix = index.GetName() + "_";
        Index outer = { prefix + "o" };
        Index inner = { prefix + "i" };
        if (_indices.count(index) == 0)
            throw accera::utilities::InputException(accera::utilities::InputExceptionErrors::invalidArgument, "Splitting an unknown index");
        if (!IsLoopIndex(index))
            throw accera::utilities::InputException(accera::utilities::InputExceptionErrors::invalidArgument, "Cannot split an already-transformed index");

        auto parentRange = _indices[index].range;
        auto parentIncrement = parentRange.Increment();
        auto parentPadding = _indices[index].padding;

        // Add outer index
        if (parentRange.HasConstantEnd())
        {
            auto parentEndValue = parentRange.End();
            _indices[outer] = { {}, { parentRange.Begin(), parentEndValue, splitSize }, parentPadding, { index } };
        }
        else if (parentRange.HasIndexEnd())
        {
            auto parentEndIndex = parentRange.EndIndex();
            _indices[outer] = { {}, { parentRange.Begin(), parentEndIndex, splitSize }, parentPadding, { index } };
        }
        else
        {
            auto parentEndOperandIndex = parentRange.EndOperandIndex();
            _indices[outer] = { {}, { parentRange.Begin(), parentEndOperandIndex, splitSize }, parentPadding, { index } };
        }

        // Add inner index
        // Note: this will need to be clamped to parentRange's Begin and End during loop nest building
        _indices[inner] = { {}, { 0, splitSize, parentIncrement }, /*padding=*/{}, { index } };

        // create affine expression for original index
        auto outerExpr = mlir::getAffineDimExpr(0, context);
        auto innerExpr = mlir::getAffineDimExpr(1, context);

        // Update parent index
        // Note: the outer loop isn't normalized, so we just add the outer index. If we decide
        //       to use normalized loops, multiply outerExpr by splitSize
        auto parentExpr = AffineExpression(outerExpr + innerExpr, { outer, inner });
        _indices[index].expr = parentExpr;
        _indices[index].range = parentRange;

        _loopIndices.erase(index);
        _loopIndices.insert(outer);
        _loopIndices.insert(inner);
        return { outer, inner };
    }

    Index TransformedDomain::Skew(const Index& index, const Index& referenceIndex, mlir::MLIRContext* context)
    {
        auto prefix = index.GetName() + "_";
        Index skewedIndex = { prefix + "s"};
        if (_indices.count(index) == 0 || _indices.count(referenceIndex) == 0)
            throw accera::utilities::InputException(accera::utilities::InputExceptionErrors::invalidArgument, "Skewing or referencing an unknown index");
        if (!IsLoopIndex(index) || !IsLoopIndex(referenceIndex))
            throw accera::utilities::InputException(accera::utilities::InputExceptionErrors::invalidArgument, "Can't skew or reference an already-transformed index");

        auto parentRange = _indices[index].range;
        auto referenceRange = _indices[referenceIndex].range;
        if (!parentRange.HasConstantEnd() || !referenceRange.HasConstantEnd())
            throw accera::utilities::InputException(accera::utilities::InputExceptionErrors::invalidArgument, "Can't skew non-constant index ranges");
        if (parentRange.Increment() != 1 || referenceRange.Increment() != 1)
            throw accera::utilities::InputException(accera::utilities::InputExceptionErrors::invalidArgument, "Can't skew index ranges with increments != 1");

        if (!_indices[index].padding.empty())
            throw accera::utilities::InputException(accera::utilities::InputExceptionErrors::invalidArgument, "Can't skew padded indices");

        // create the skewed index. index(dim=N), referenceIndex(dim=M), skewedIndex(dim=N+M-1)
        // Note: this expands the iteration space beyond the original dimensions, so FlatAffineConstraints should be used to enforce
        // the dimension bounds.
        auto skewFactor = referenceRange.Size();
        _indices[skewedIndex] = { {}, { parentRange.Begin(), parentRange.Size() + skewFactor - 1 }, /*padding=*/{}, { index } };
        _indices[referenceIndex].parents.insert(index); // add a linkage between the parent and the reference index

        // create affine expression for the parent index
        auto skewExpr = mlir::getAffineDimExpr(0, context);
        auto referenceExpr = mlir::getAffineDimExpr(1, context);

        // skew results in: (i, j) => (i_s, j) = (i+j, j)
        // therefore, the parent index i = i_s-j
        auto parentExpr = AffineExpression(skewExpr - referenceExpr, { skewedIndex, referenceIndex });
        _indices[index].expr = parentExpr;

        _loopIndices.erase(index);
        _loopIndices.insert(skewedIndex);
        return skewedIndex;
    }

    bool TransformedDomain::HasConstantDimensionSize(const Index& dimensionIndex) const
    {
        if (!IsDimension(dimensionIndex))
            throw accera::utilities::InputException(accera::utilities::InputExceptionErrors::invalidArgument, "unknown dimension");

        auto range = GetIndexRange(dimensionIndex);
        return range.HasConstantEnd();
    }

    int64_t TransformedDomain::GetDimensionSize(const Index& dimensionIndex) const
    {
        if (!IsDimension(dimensionIndex))
            throw accera::utilities::InputException(accera::utilities::InputExceptionErrors::invalidArgument, "unknown dimension");

        auto range = GetIndexRange(dimensionIndex);
        return range.Size();
    }

    int64_t TransformedDomain::GetDimensionBegin(const Index& dimensionIndex) const
    {
        if (!IsDimension(dimensionIndex))
            throw accera::utilities::InputException(accera::utilities::InputExceptionErrors::invalidArgument, "unknown dimension");

        auto range = GetIndexRange(dimensionIndex);
        return range.Begin();
    }

    Range TransformedDomain::GetIndexRange(const Index& index) const
    {
        if (_indices.count(index) == 0)
            throw accera::utilities::InputException(accera::utilities::InputExceptionErrors::invalidArgument, "unknown index");

        return _indices.at(index).range;
    }

    std::vector<int64_t> TransformedDomain::GetIndexPadding(const Index& index) const
    {
        if (_indices.count(index) == 0)
            throw accera::utilities::InputException(accera::utilities::InputExceptionErrors::invalidArgument, "unknown index");

        return _indices.at(index).padding;
    }

    // TODO: fix this to work with loop indices, by taking an MLIR context and filling in an actual affine expression
    AffineExpression TransformedDomain::GetIndexExpr(const Index& index) const
    {
        if (_indices.count(index) == 0)
            throw accera::utilities::InputException(accera::utilities::InputExceptionErrors::invalidArgument, "unknown index");

        return _indices.at(index).expr;
    }

    // TODO: fix this to work with loop indices
    AffineExpression TransformedDomain::GetReducedIndexExpr(const Index& index, mlir::MLIRContext* context) const
    {
        auto expr = GetIndexExpr(index);
        if (std::all_of(expr.GetIndices().begin(), expr.GetIndices().end(), [this](auto i) { return IsLoopIndex(i); }))
            return expr;

        std::vector<Index> newIndices;
        std::map<Index, unsigned> indexPosMap;
        auto addIndex = [&indexPosMap, &newIndices](auto i) { if (indexPosMap.count(i) == 0) {
            indexPosMap[i] = newIndices.size();
            newIndices.push_back(i); } };

        for (auto exprIdx : expr.GetIndices())
        {
            if (IsLoopIndex(exprIdx))
            {
                addIndex(exprIdx);
            }
        }

        std::vector<mlir::AffineExpr> subExprs;
        for (auto exprIndex : expr.GetIndices())
        {
            auto subExpr = GetReducedIndexExpr(exprIndex, context);
            if (subExpr.IsIdentity())
            {
                subExprs.emplace_back(mlir::getAffineDimExpr(indexPosMap[exprIndex], context));
            }
            else
            {
                auto subIndices = subExpr.GetIndices();
                for (auto subIndex : subIndices)
                {
                    assert(IsLoopIndex(subIndex));
                    addIndex(subIndex);
                }

                mlir::DenseMap<mlir::AffineExpr, mlir::AffineExpr> subSubExprMap;
                subExpr.GetAffineExpr().walk([&subSubExprMap, &subIndices, &indexPosMap, context](mlir::AffineExpr subSubExpr) {
                    if (subSubExpr.getKind() == mlir::AffineExprKind::DimId)
                    {
                        auto dimExpr = subSubExpr.cast<mlir::AffineDimExpr>();
                        auto pos = dimExpr.getPosition();
                        auto newPos = indexPosMap[subIndices[pos]];
                        subSubExprMap[subSubExpr] = mlir::getAffineDimExpr(newPos, context);
                    }
                    else if (subSubExpr.getKind() == mlir::AffineExprKind::SymbolId)
                    {
                        // TODO: in the future, if we use symbols for constants, do the symbol renumbering here
                    }
                });
                subExprs.emplace_back(subExpr.GetAffineExpr().replace(subSubExprMap));
            }
        }
        unsigned numDims = GetAllLoopIndices().size();
        unsigned numSymbols = 0;
        // auto subExprMap = simplifyAffineMap(mlir::AffineMap::get(numDims, numSymbols, subExprs, context));
        auto subExprMap = mlir::AffineMap::get(numDims, numSymbols, subExprs, context);
        auto newExpr = expr.GetAffineExpr().compose(subExprMap);

        return { newExpr, newIndices };
    }

    std::vector<Index> TransformedDomain::GetLoopIndicesForDimension(const Index& dimensionIndex) const
    {
        if (!IsDimension(dimensionIndex))
            throw accera::utilities::InputException(accera::utilities::InputExceptionErrors::invalidArgument, "index isn't a dimension");

        std::unordered_set<Index> loopIndices;
        CollectLoopIndicesForIndex(dimensionIndex, loopIndices);
        return { loopIndices.begin(), loopIndices.end() };
    }

    void TransformedDomain::CollectLoopIndicesForIndex(const Index& index, std::unordered_set<Index>& loopIndices) const
    {
        auto expr = GetIndexExpr(index);
        for (auto i : expr.GetIndices())
        {
            if (IsLoopIndex(i))
            {
                loopIndices.insert(i);
            }
            else
            {
                CollectLoopIndicesForIndex(i, loopIndices);
            }
        }
    }

    std::vector<Index> TransformedDomain::GetComputedIndicesForDimension(const Index& dimensionIndex) const
    {
        if (!IsDimension(dimensionIndex))
            throw accera::utilities::InputException(accera::utilities::InputExceptionErrors::invalidArgument, "index isn't a dimension");

        std::unordered_set<Index> computedIndices;
        CollectComputedIndicesForIndex(dimensionIndex, computedIndices);
        return { computedIndices.begin(), computedIndices.end() };
    }

    void TransformedDomain::CollectComputedIndicesForIndex(const Index& index, std::unordered_set<Index>& computedIndices) const
    {
        auto expr = GetIndexExpr(index);
        for (auto i : expr.GetIndices())
        {
            if (IsComputedIndex(i))
            {
                computedIndices.insert(i);
            }
            else
            {
                CollectComputedIndicesForIndex(i, computedIndices);
            }
        }
    }

    std::vector<Index> TransformedDomain::GetDependentIndices(const Index& index) const
    {
        std::unordered_set<Index> result;
        std::queue<Index> indicesToVisit;
        indicesToVisit.push(index);

        // get all children, children of children, etc.
        while (!indicesToVisit.empty())
        {
            auto i = indicesToVisit.front();
            indicesToVisit.pop();
            if (!IsLoopIndex(i))
            {
                auto expr = _indices.at(i).expr;
                auto dependentIndices = expr.GetIndices();
                for (auto d : dependentIndices)
                {
                    if (result.count(d) == 0)
                    {
                        if (!IsLoopIndex(d))
                        {
                            indicesToVisit.push(d);
                        }
                        result.insert(d);
                    }
                }
            }
        }
        return { result.begin(), result.end() };
    }

    std::vector<Index> TransformedDomain::GetDependentLoopIndices(const Index& index, bool includeSelf) const
    {
        if (IsLoopIndex(index))
        {
            return { index };
        }
        else
        {
            std::unordered_set<Index> result;
            auto dependentIndices = GetDependentIndices(index);
            for (auto i : dependentIndices)
            {
                if (IsLoopIndex(i))
                {
                    result.insert(i);
                }
            }
            return { result.begin(), result.end() };
        }
    }

    bool TransformedDomain::DependsOn(const Index& index1, const Index& index2) const
    {
        // does 'index1' depend on 'index2'? (after a split, the parent depends on the new loop indices)
        // look at expr for index1 and see if it contains index2. Do so recursively
        auto index1Expr = GetIndexExpr(index1);
        auto index1Vars = index1Expr.GetIndices();
        if (std::find(index1Vars.begin(), index1Vars.end(), index2) != index1Vars.end())
        {
            return true;
        }

        for (auto i : index1Vars)
        {
            if (DependsOn(i, index2))
            {
                return true;
            }
        }
        return false;
    }

    bool TransformedDomain::HasParentIndex(const Index& index) const
    {
        // TODO: rethink this
        return !_indices.at(index).parents.empty();
    }

    std::vector<Index> TransformedDomain::GetParentIndices(const Index& index) const
    {
        const auto& parents = _indices.at(index).parents;
        return { parents.begin(), parents.end() };
    }

    bool TransformedDomain::IsSplitIndex(const Index& index, bool inner) const
    {
        if (!HasParentIndex(index))
            return false;

        auto parents = GetParentIndices(index);
        for (auto parent : parents)
        {
            auto parentExpr = GetIndexExpr(parent);
            if (IsSplitExpression(parentExpr))
            {
                auto parentExprIndices = parentExpr.GetIndices();
                unsigned pos = inner ? 1 : 0; // expr: parent = outer + inner
                if (parentExprIndices[pos] == index)
                {
                    return true;
                }
            }
        }
        return false;
    }

    Index TransformedDomain::GetOtherSplitIndex(const Index& index) const
    {
        auto parents = GetParentIndices(index);
        for (auto parent : parents)
        {
            auto parentExpr = GetIndexExpr(parent);
            if (IsSplitExpression(parentExpr))
            {
                auto parentExprIndices = parentExpr.GetIndices();
                if (parentExprIndices[0] == index)
                {
                    return parentExprIndices[1];
                }
                else if (parentExprIndices[1] == index)
                {
                    return parentExprIndices[0];
                }
            }
        }
        throw accera::utilities::InputException(accera::utilities::InputExceptionErrors::invalidArgument, "GetOtherSplitIndex() --- called on a non-split index");
    }

    bool TransformedDomain::IsPaddedIndex(const Index& index) const
    {
        auto padding = _indices.at(index).padding;
        return std::any_of(padding.cbegin(), padding.cend(), [](auto p) { return p != 0; });
    }

    bool TransformedDomain::IsFusedPaddedIndex(const Index& index) const
    {
        return IsPaddedIndex(index) && GetIndexPadding(index).size() > 1;
    }

    bool TransformedDomain::IsPrePaddedIndexOf(const Index& index, const Index& paddedIndex) const
    {
        if (!IsPaddedIndex(paddedIndex))
            return false;

        const auto& parents = _indices.at(paddedIndex).parents;
        return std::find(parents.begin(), parents.end(), index) != parents.end();
    }

    std::optional<std::pair<bool, Index>> TransformedDomain::IsSkewedOrReferenceIndex(const Index& index) const
    {
        if (!HasParentIndex(index))
            return std::nullopt;

        auto parents = GetParentIndices(index);
        for (auto parent : parents)
        {
            auto parentExpr = GetIndexExpr(parent);
            if (IsSkewExpression(parentExpr))
            {
                auto parentExprIndices = parentExpr.GetIndices();
                if (parentExprIndices[0] == index)
                {
                    return std::pair{ true, parentExprIndices[1] };
                }
                else if (parentExprIndices[1] == index)
                {
                    return std::pair{ false, parentExprIndices[0] };
                }
            }
        }
        return std::nullopt;
    }

    std::vector<Index> TransformedDomain::GetBaseIndices(const Index& index) const
    {
        std::unordered_set<Index> baseIndices;
        GetBaseIndices(index, baseIndices);
        return { std::begin(baseIndices), std::end(baseIndices) };
    }

    void TransformedDomain::GetBaseIndices(const Index& index, std::unordered_set<Index>& baseIndices) const
    {
        if (!HasParentIndex(index))
        {
            baseIndices.insert(index);
        }

        for (auto parent : GetParentIndices(index))
        {
            GetBaseIndices(parent, baseIndices);
        }
    }

    Index TransformedDomain::GetBaseIndex(const Index& index) const
    {
        if (!HasParentIndex(index))
            return index;

        auto baseIndices = GetBaseIndices(index);
        if (baseIndices.size() != 1)
            throw accera::utilities::InputException(accera::utilities::InputExceptionErrors::invalidArgument, "Getting base index of an index with multiple parents");

        for (auto i : baseIndices)
            return i;
        return Index::none;
    }

    void TransformedDomain::ResolveRangeValues(const std::function<void(Range&)>& resolveFn)
    {
        for (auto& indexInfo : _indices)
        {
            resolveFn(indexInfo.second.range);
        }
    }

    TransformedDomain TransformedDomain::Fuse(const std::vector<TransformedDomain>& domains, const std::vector<std::vector<Index>>& indexCorrespondences)
    {
        TransformedDomain result;

        if (const auto numDomains = domains.size();
            std::any_of(indexCorrespondences.begin(),
                        indexCorrespondences.end(),
                        [&numDomains](const std::vector<Index>& indices) { return indices.size() != numDomains; }))
        {
            throw accera::utilities::InputException(accera::utilities::InputExceptionErrors::invalidArgument, "Index correspondences don't match number of domains");
        }

        // First concatenate all the info from all domains
        for (auto& domain : domains)
        {
            result._dimensions.insert(result._dimensions.end(), domain._dimensions.begin(), domain._dimensions.end());
            result._loopIndices.insert(domain._loopIndices.begin(), domain._loopIndices.end());
            result._indices.insert(domain._indices.begin(), domain._indices.end());
        }

        // Update padding information for the fused index
        auto updateFusedPadding = [](const TransformedDomain& domain, const Index& index, std::vector<int64_t>& fusedPadding) -> void {
            auto padding = domain.GetIndexPadding(index);
            if (padding.empty())
            {
                fusedPadding.push_back(0); // fill in a zero entry
            }
            else if (padding.size() > 1)
            {
                // unsupported until there is a use case for fusing a fused index multiple times that is not addressed by one-shot fusing
                // instead of a flat vector of padding sizes, we will likely need to track index correspondences with per fusing dimension
                // i.e. different padding sizes along each fusing dimension
                throw accera::utilities::InputException(accera::utilities::InputExceptionErrors::invalidArgument, "Fusing an already-fused padded index is not supported.");
            }
            else
            {
                fusedPadding.push_back(padding[0]); // fill in the actual padding value (in index correspondence order)
            }
        };

        // Remove the entries from domain2 that are fused to something in domain1
        for (const auto& correspondingIndices : indexCorrespondences)
        {
            auto i1 = correspondingIndices[0];

            std::vector<int64_t> fusedPadding;
            updateFusedPadding(domains[0], i1, fusedPadding);

            for (unsigned idx = 1; idx < correspondingIndices.size(); ++idx)
            {
                auto i2 = correspondingIndices[idx];

                if (result._indices[i2].range != result._indices[i1].range)
                {
                    throw accera::utilities::InputException(accera::utilities::InputExceptionErrors::invalidArgument, "Fused index ranges do not match. The smaller index needs to be padded.");
                }

                updateFusedPadding(domains[idx], i2, fusedPadding);

                // Resolve any padded indices to its dimension index
                Index i2dimIndex = i2;
                if (domains[idx].IsPaddedIndex(i2))
                {
                    auto dimIndices = domains[idx].GetBaseIndices(i2);
                    if (dimIndices.size() > 1)
                    {
                        throw accera::utilities::InputException(accera::utilities::InputExceptionErrors::invalidArgument, "Fusing an index based on multiple dimensions is not supported.");
                    }
                    i2dimIndex = dimIndices[0];
                }

                // Remove dimensions from domain2
                result._dimensions.erase(
                    std::remove_if(
                        result._dimensions.begin(), result._dimensions.end(), [i2dimIndex](const Index& i) {
                            return i == i2dimIndex;
                        }),
                    result._dimensions.end());

                // Need to remove all child indices of a fused index
                auto dependents = result.GetDependentIndices(i2);
                for (auto dependentIndex : dependents)
                {
                    result._loopIndices.erase(dependentIndex); // Note: don't need to check it's a loop index -- unordered_sets don't complain when removing something not there
                    result._indices.erase(dependentIndex);
                }

                result._loopIndices.erase(i2);
                result._indices.erase(i2);
            }

            // Skip padding if there are none at all
            if (std::any_of(fusedPadding.cbegin(), fusedPadding.cend(), [](auto p) { return p != 0; }))
            {
                result._indices[i1].padding = fusedPadding;
            }
        }

        // Now fix up any expressions still referencing the old indices
        std::vector<Index> unusedIndices;
        for (auto& resultIndex : result._indices)
        {
            if (!result.IsLoopIndex(resultIndex.first) && !result.IsDimension(resultIndex.first))
            {
                unusedIndices.push_back(resultIndex.first);
                continue;
            }

            auto exprIndices = resultIndex.second.expr.GetIndices();
            for (const auto& correspondingIndices : indexCorrespondences)
            {
                auto i1 = correspondingIndices[0];
                for (unsigned idx = 1; idx < correspondingIndices.size(); ++idx)
                {
                    auto i2 = correspondingIndices[idx];

                    // replace expr indices
                    if (auto indexIt = std::find(exprIndices.begin(), exprIndices.end(), i2); indexIt != exprIndices.end())
                    {
                        *indexIt = i1;
                    }

                    // replace parents
                    if (resultIndex.second.parents.count(i2) > 0)
                    {
                        resultIndex.second.parents.erase(i2);
                        resultIndex.second.parents.insert(i1);
                    }
                }
            }

            resultIndex.second.expr = AffineExpression(resultIndex.second.expr.GetAffineExpr(), exprIndices);
        }

        // remove unused indices (e.g. pre-padded indices)
        for (auto& unusedIndex : unusedIndices)
        {
            result._indices.erase(unusedIndex);
        }

        return result;
    }

    void TransformedDomain::AddDimension(const Index& dimension, const Range& range)
    {
        _dimensions.emplace_back(dimension);
        _loopIndices.insert(dimension);

        assert(range.Begin() == 0 && "Dimension should begin at 0");
        _indices[dimension] = { {}, range, /*padding=*/{}, /*parents=*/{} };
    }

    void TransformedDomain::Print(std::ostream& os) const
    {
        for (const auto& d : _dimensions)
        {
            os << d << "\n";
        }
    }

    AffineConstraints TransformedDomain::GetConstraints() const
    {
        std::vector<Index> indices(_indices.size());
        std::transform(_indices.cbegin(), _indices.cend(), indices.begin(), [](auto entry) { return entry.first; });

        AffineConstraints constraints(indices);

        // For each index:
        //   add its range (lower bound, upper bound), subtracting any padding
        //   add an equality if there is an affine expression
        for (const auto& [index, indexInfo] : _indices)
        {
            auto padding = indexInfo.padding;
            if (!padding.empty())
            {
                // Multiple padding values may exist (e.g. for a fused index),
                // conservatively pick the largest absolute padding value so that
                // it becomes the split point for loop unswitching
                // Note: fusion range predicates still need to account for the
                // intermediate ranges, so that kernels run in their respective sub-ranges
                auto paddingIt = std::max_element(padding.cbegin(), padding.cend(), [](auto p1, auto p2) {
                    return std::abs(p1) < std::abs(p2);
                });
                auto paddingValue = *paddingIt;
                assert(paddingValue != 0 && "Unexpected padding size"); // coding error

                auto paddedRange = indexInfo.range;
                assert(paddedRange.End() > std::abs(paddingValue) && "Invalid unpadded range"); // coding error

                auto unpaddedRange = paddingValue > 0 ?
                    // back-padding: (0, N-padding)
                    Range(paddedRange.Begin(), paddedRange.End() - paddingValue, paddedRange.Increment()) :
                    // front-padding: (abs(padding), N)
                    Range(std::abs(paddingValue), paddedRange.End(), paddedRange.Increment());

                constraints.AddConstraint(index, unpaddedRange);
            }
            else
            {
                constraints.AddConstraint(index, indexInfo.range);
            }

            if (!indexInfo.expr.IsIdentity())
            {
                constraints.AddConstraint(index, indexInfo.expr, [](AffineExpression expr) -> std::vector<int64_t>
                {
                    // determine the coefficients by parsing the expression type
                    // coefficients are in the same order as expr's arguments
                    if (IsSplitExpression(expr))
                    {
                        // split: d2 = d0 + d1
                        return {1, 1};
                    }
                    else if (IsSkewExpression(expr))
                    {
                        // skew: d2 = d0 - d1
                        return {1, -1};
                    }
                    return {}; // unrecognized expr, do nothing
                });
            }
        }
        return constraints;
    }

    bool operator==(const TransformedDomain& a, const TransformedDomain& b)
    {
        if ((a.NumDimensions() != b.NumDimensions()) ||
            (a.NumIndices() != b.NumIndices()) ||
            (a.NumLoopIndices() != b.NumLoopIndices()))
        {
            return false;
        }

        auto aDimensions = a.GetDimensions();
        auto bDimensions = b.GetDimensions();
        auto dimRange = accera::utilities::MakeZipRange(aDimensions, bDimensions);
        auto dimsMatch = std::all_of(begin(dimRange), end(dimRange), [&](auto it) {
            return (std::get<0>(it) == std::get<1>(it)) && (a.GetIndexRange(std::get<0>(it)) == b.GetIndexRange(std::get<1>(it)));
        });
        if (!dimsMatch)
        {
            return false;
        }

        auto aIndices = a.GetIndices();
        auto bIndices = b.GetIndices();
        auto indicesRange = accera::utilities::MakeZipRange(aIndices, bIndices);
        auto indicesMatch = std::all_of(begin(indicesRange), end(indicesRange), [&](auto it) {
            return (std::get<0>(it) == std::get<1>(it)) && (a.GetIndexRange(std::get<0>(it)) == b.GetIndexRange(std::get<1>(it))) && (a.GetIndexExpr(std::get<0>(it)) == b.GetIndexExpr(std::get<1>(it)));
        });
        if (!indicesMatch)
        {
            return false;
        }

        return true;
    }

    bool operator!=(const TransformedDomain& a, const TransformedDomain& b)
    {
        return !(a == b);
    }

} // namespace loopnest
} // namespace accera::ir
