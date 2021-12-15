////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
//  Authors:  Chuck Jacobs, Kern Handa
////////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

#include "Index.h"
#include "Range.h"

#include <string>

namespace accera::ir
{
namespace loopnest
{
    /// <summary>
    /// A range of integer values, used to express the interval that an index variable may take on.
    /// </summary>
    class IndexRange
    {
    public:
        IndexRange(const Index& index, const Range& range);
        IndexRange(const std::string& name, const Range& range);

        const Index& GetIndex() const;
        const std::string& GetName() const;
        int Begin() const;
        int End() const;
        int Size() const;
        int Increment() const;
        Range GetRange() const;

        void ResolveRangeValues(const std::function<void(Range&)>& resolveFn);

    private:
        Index _index;
        Range _range;

        friend inline bool operator==(const IndexRange& i1, const IndexRange& i2) { return (i1.GetIndex() == i2.GetIndex()) && (i1.GetRange() == i2.GetRange()); }
        friend inline bool operator!=(const IndexRange& i1, const IndexRange& i2) { return (i1.GetIndex() != i2.GetIndex()) || (i1.GetRange() != i2.GetRange()); }
        friend inline bool operator<(const IndexRange& i1, const IndexRange& i2) { return (i1.GetIndex() != i2.GetIndex()) ? (i1.GetIndex() < i2.GetIndex()) : (i1.GetRange() < i2.GetRange()); }
    };

} // namespace loopnest
} // namespace accera::ir
