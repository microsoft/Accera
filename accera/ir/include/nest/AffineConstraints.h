////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
////////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

#include "Range.h"
#include "AffineExpression.h"

#include <mlir/Dialect/Affine/Analysis/AffineStructures.h>

#include <vector>
#include <utility>

namespace accera::ir
{
namespace loopnest
{
    class AffineConstraints
    {
    public:
        AffineConstraints(std::vector<Index> indices);

        void AddConstraint(Index index, Range range);
        void AddConstraint(Index index, AffineExpression expr, std::function<std::vector<int64_t>(AffineExpression)> getCoeffsFn);

        std::pair<int64_t, int64_t> GetEffectiveRangeBounds(Index index) const;

        void dump() const { _constraints.dump(); };

    private:
        uint64_t GetIndexPosition(Index index) const;

        mlir::FlatAffineConstraints _constraints;
        std::vector<Index> _indices;
    };

} // namespace loopnest
} // namespace accera::ir
