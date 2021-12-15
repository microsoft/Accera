////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
//  Authors: Chuck Jacobs, Kern Handa
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "nest/IndexRange.h"

namespace accera::ir
{
namespace loopnest
{
    IndexRange::IndexRange(const Index& index, const Range& range) :
        _index(index),
        _range(range)
    {
    }

    IndexRange::IndexRange(const std::string& name, const Range& range) :
        _index({ name }),
        _range(range)
    {
    }

    const Index& IndexRange::GetIndex() const
    {
        return _index;
    }

    const std::string& IndexRange::GetName() const
    {
        return _index.GetName();
    }

    int IndexRange::Begin() const
    {
        return _range.Begin();
    }

    int IndexRange::End() const
    {
        return _range.End();
    }

    int IndexRange::Size() const
    {
        return _range.Size();
    }

    int IndexRange::Increment() const
    {
        return _range.Increment();
    }

    Range IndexRange::GetRange() const
    {
        return _range;
    }

    void IndexRange::ResolveRangeValues(const std::function<void(Range&)>& resolveFn)
    {
        resolveFn(_range);
    }
} // namespace loopnest
} // namespace accera::ir
