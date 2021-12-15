////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
//  Authors: Kern Handa
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "ValueOperations.h"
#include "EmitterContext.h"
#include "Scalar.h"

#include <cassert>

namespace accera
{
namespace value
{
    using namespace utilities;

    namespace detail
    {
        Scalar CalculateOffset(const MemoryLayout& layout, std::vector<Scalar> coordinates)
        {
            if (layout == ScalarLayout)
            {
                assert(coordinates.empty());
                return { 0 };
            }
            else
            {
                const auto& offset = layout.GetOffset();
                const auto& increment = layout.GetIncrement();
                const auto numDimensions = layout.NumDimensions();

                Scalar result;
                for (int index = 0; index < numDimensions; ++index)
                {
                    result += increment[index] * (coordinates[index] + offset[index]);
                }

                return result;
            }
        }

    } // namespace detail

    void ForSequence(Scalar end, std::function<void(Scalar)> fn)
    {
        throw LogicException(LogicExceptionErrors::notImplemented);
    }

    void For(MemoryLayout layout, std::function<void(Scalar)> fn)
    {
        GetContext().For(layout, [&layout, fn = std::move(fn)](std::vector<Scalar> coords) {
            fn(detail::CalculateOffset(layout, coords));
        });
    }

    void For(Scalar start, Scalar stop, Scalar step, std::function<void(Scalar)> fn)
    {
        GetContext().For(start, stop, step, fn);
    }

    Scalar Cast(Scalar value, ValueType type)
    {
        if (value.GetType() == type)
        {
            return value;
        }

        return GetContext().Cast(value, type);
    }

    Scalar UnsignedCast(Scalar value, ValueType type)
    {
        if (value.GetType() == type)
        {
            return value;
        }

        return GetContext().UnsignedCast(value, type);
    }

} // namespace value
} // namespace accera
