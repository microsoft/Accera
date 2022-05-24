////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
//  Authors: Chuck Jacobs, Kern Handa
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "nest/LoopVisitSchedule.h"

#include <numeric>
#include <stdexcept>

namespace accera::ir
{
namespace loopnest
{
    //
    // LoopVisitSchedule
    //

    LoopVisitSchedule::LoopVisitSchedule(std::vector<IndexRange> loopRanges, int level) :
        _level(level),
        _loopRanges(std::move(loopRanges))
    {}

    Range LoopVisitSchedule::GetActiveLoopRange(const TransformedDomain& domain, const Index& loopIndex, const LoopIndexSymbolTable& activeRanges) const
    {
        // Compute the active loop range bounds. This will fix the end boundaries of a loop index.
        if (!domain.IsLoopIndex(loopIndex))
        {
            throw std::runtime_error("Error: emitting a loop for a non-loop index");
        }

        auto loopRange = domain.GetIndexRange(loopIndex);
        int begin = loopRange.Begin();
        int end = loopRange.End();

        int rangeSize = end - begin;
        int increment = loopRange.Increment();

        auto fixBoundaryRange = [&](Index index) {
            // check activeRanges for parent
            auto outerIndex = domain.GetOtherSplitIndex(index);
            if (domain.IsLoopIndex(outerIndex) && activeRanges.count(outerIndex) != 0)
            {
                // check if it's a boundary --- if so, set size to its size
                auto parentRange = activeRanges.at(outerIndex).loopRange;
                if (parentRange.Size() < rangeSize)
                {
                    auto constraints = domain.GetConstraints();
                    constraints.AddConstraint(loopIndex, Range(begin, begin + parentRange.Size()));

                    auto [begin1, end1] = constraints.GetEffectiveRangeBounds(loopIndex);
                    loopRange = { begin1, end1, increment };
                }
            }
            else
            {
                // Use full loop range if outer index isn't defined yet.
            }
        };

        if (domain.IsPaddedIndex(loopIndex))
        {
            const auto constraints = domain.GetConstraints();
            auto [unpaddedBegin, unused] = constraints.GetEffectiveRangeBounds(loopIndex);

            // clamp the front-padded ranges for non-fused indices only
            // (fused indices will require the full range to apply predicates)
            if (unpaddedBegin > 0 && !domain.IsFusedPaddedIndex(loopIndex))
            {
                loopRange = { unpaddedBegin, end, increment };
            }
        }
        if (domain.IsSplitIndex(loopIndex, /*inner=*/ true))
        {
            fixBoundaryRange(loopIndex);
        }
        else if (domain.HasParentIndex(loopIndex))
        {
            auto parents = domain.GetParentIndices(loopIndex);
            for (auto parentIndex : parents)
            {
                if (domain.IsSplitIndex(parentIndex, /*inner=*/ true))
                {
                    fixBoundaryRange(parentIndex);
                }
            }
        }

        if (auto skewedOrReference = domain.IsSkewedOrReferenceIndex(loopIndex))
        {
            auto [isSkewedIndex, dependentIndex] = *skewedOrReference;
            if (WasIterationVariableDefined(dependentIndex) && activeRanges.count(dependentIndex) != 0)
            {
                // dependent index is in the outer part of the loop, find its active range and
                // apply as a constraint on the effective current index bounds
                auto dependentIndexRange = activeRanges.at(dependentIndex).loopRange;
                auto constraints = domain.GetConstraints();
                constraints.AddConstraint(dependentIndex, dependentIndexRange);

                auto [begin_, end_] = constraints.GetEffectiveRangeBounds(loopIndex);
                loopRange = { begin_, end_, loopRange.Increment() };
            }
        }

        return loopRange;
    }

    bool LoopVisitSchedule::IsDone() const
    {
        return _level == static_cast<int>(_loopRanges.size());
    }

    bool LoopVisitSchedule::IsInnermostLoop() const
    {
        return _level == static_cast<int>(_loopRanges.size()) - 1;
    }

    bool LoopVisitSchedule::IsOutermostLoop() const
    {
        return _level == 0;
    }

    Index LoopVisitSchedule::CurrentLoopIndex() const
    {
        return _loopRanges[_level].GetIndex();
    }

    int64_t LoopVisitSchedule::CurrentLoopLevel() const
    {
        return _level;
    }

    LoopVisitSchedule LoopVisitSchedule::Next() const
    {
        if (IsDone())
        {
            throw std::runtime_error("Error: calling Next() at end of schedule");
        }
        return { _loopRanges, _level + 1 };
    }

    LoopVisitSchedule LoopVisitSchedule::Prev() const
    {
        if (_level == 0)
        {
            throw std::runtime_error("Error: calling Prev() on first loop level");
        }
        return { _loopRanges, _level - 1 };
    }

    bool LoopVisitSchedule::WasIterationVariableDefined(const Index& index) const
    {
        for (auto it = _loopRanges.begin(); it != _loopRanges.begin() + _level + 1; ++it)
        {
            auto iterVar = it->GetIndex();
            if (iterVar == index)
            {
                return true;
            }
        }
        return false;
    }
} // namespace loopnest
} // namespace accera::ir
