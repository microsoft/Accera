////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
//  Authors:  Chuck Jacobs, Kern Handa
////////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

#include "Index.h"
#include "IndexRange.h"
#include "LoopIndexInfo.h"
#include "TransformedDomain.h"

#include <ostream>
#include <unordered_map>
#include <vector>

namespace accera::ir
{
namespace loopnest
{
    /// <summary>
    /// Represents the concrete sequence of loops to be generated, in detail. Derived from the loop nest and the
    /// order of the loops.
    /// </summary>
    class LoopVisitSchedule
    {
    public:
        /// <summary> Returns `true` if all the loops have been visited </summary>
        bool IsDone() const;

        /// <summary> Returns `true` if the current loop is the innermost level </summary>
        bool IsInnermostLoop() const;

        /// <summary> Returns `true` if the current loop is the outermost level </summary>
        bool IsOutermostLoop() const;

        /// <summary> The index of the current loop (e.g., `i_1`) </summary>
        Index CurrentLoopIndex() const;

        int64_t CurrentLoopLevel() const;

        bool WasIterationVariableDefined(const Index& index) const;

        LoopVisitSchedule Next() const;
        LoopVisitSchedule Prev() const;

        Range GetActiveLoopRange(const TransformedDomain& domain, const Index& loopIndex, const LoopIndexSymbolTable& activeRanges) const;

        LoopVisitSchedule(std::vector<IndexRange> ranges, int level = 0);

    private:
        int _level; // == current position in loop range list
        std::vector<IndexRange> _loopRanges;
    };

} // namespace loopnest
} // namespace accera::ir
