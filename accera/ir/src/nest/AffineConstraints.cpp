////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "nest/AffineConstraints.h"

#include <llvm/ADT/SmallVector.h>

#include <algorithm>

namespace accera::ir
{
namespace loopnest
{
    AffineConstraints::AffineConstraints(std::vector<Index> indices) : _indices(indices)
    {
        // Initial guesstimate for # rows, more inequalities/equalities can be added later
        auto numInequalities = indices.size() * 2; // lower bound, upper bound
        auto numEqualities = indices.size(); // assume at most 1 affine expression per index

        // Unlike rows, the maximum # columns and dimensions are fixed at construction time
        auto numConstraintDimensions = indices.size();
        auto numConstraintColumns = numConstraintDimensions + 1; // +1 column for the constant dimension

        _constraints = mlir::FlatAffineConstraints(numInequalities, numEqualities,
                                                   numConstraintColumns, numConstraintDimensions,
                                                   /*numSymbols=*/0, /*numLocals=*/0);
    }

    void AffineConstraints::AddConstraint(Index index, Range range)
    {
        auto pos = GetIndexPosition(index);
        _constraints.addBound(mlir::FlatAffineConstraints::LB, pos, range.Begin());

        if (range.End() > 0)
        {
            _constraints.addBound(mlir::FlatAffineConstraints::UB, pos, range.End()-1); // half open bound -> closed bound
        }
        else
        {
            assert(range.End() == range.Begin() && "Bad range (end < begin)");
            _constraints.addBound(mlir::FlatAffineConstraints::UB, pos, range.Begin()); // single-value closed bound
        }
    }

    void AffineConstraints::AddConstraint(Index index, AffineExpression expr, std::function<std::vector<int64_t>(AffineExpression)> getCoeffsFn)
    {
        if (expr.IsIdentity()) return;

        auto pos = GetIndexPosition(index);

        // Given an affine expression of this form:
        //   index = coeff0 * d0 + coeff1 * d1 + ... [+ coeffN * 1]
        //
        // The corresponding equality expression will be of the form:
        //   index - coeff0 * d0 - coeff1 * d1 - .... [- coeffN * 1] = 0
        //
        //   where d0, d1, ... are the affine expression args
        //   and the last coefficient (if present) is for the constant dimension
        auto args = expr.GetIndices();
        std::vector<unsigned> argPositions(args.size());
        std::transform(args.cbegin(), args.cend(), argPositions.begin(), [=](auto idx) { return GetIndexPosition(idx); });

        auto coeffs = getCoeffsFn(expr);
        if (coeffs.size())
        {
            assert((coeffs.size() == argPositions.size() ||
                    coeffs.size() == argPositions.size() + 1 ) && "Invalid number of coefficients");

            auto numConstraintColumns = _indices.size() + 1; // +1 column for the constant dimension
            llvm::SmallVector<int64_t, 4> equality(numConstraintColumns, 0);

            for (size_t i = 0; i < argPositions.size() && i < coeffs.size(); ++i)
            {
                equality[argPositions[i]] = -1 * coeffs[i];
            }

            if (coeffs.size() == argPositions.size() + 1)
            {
                // there exists a constant dimension in the expression (e.g. pad)
                equality[numConstraintColumns-1] = -1 * coeffs[coeffs.size()-1];
            }

            equality[pos] = 1; // the coefficient of the current index is always 1 in its equality expression
            _constraints.addEquality(equality);
        }
    }

    std::pair<int64_t, int64_t> AffineConstraints::GetEffectiveRangeBounds(Index index) const
    {
        auto pos = GetIndexPosition(index);
        auto lowerBound = _constraints.getConstantBound(mlir::FlatAffineConstraints::LB, pos);
        auto upperBound = _constraints.getConstantBound(mlir::FlatAffineConstraints::UB, pos);

        assert(lowerBound && upperBound && "Index has no upper or lower bound"); // coding error

        // Note: not a full-fledged Range because increment is not represented by constraints
        return { *lowerBound, *upperBound + 1 }; // closed bound -> half-open bound
    }

    uint64_t AffineConstraints::GetIndexPosition(Index index) const
    {
        auto it = std::find(_indices.cbegin(), _indices.cend(), index);
        if (it == _indices.cend()) throw std::out_of_range("Index not found");

        return static_cast<uint64_t>(std::distance(_indices.cbegin(), it));
    }

} // namespace loopnest
} // namespace accera::ir
