////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
//  Authors: Kern Handa
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "Boolean.h"

namespace accera
{
namespace utilities
{

    Boolean::Boolean() = default;

    Boolean::Boolean(bool value_) :
        value(value_) {}

    bool operator==(Boolean b1, Boolean b2)
    {
        return static_cast<bool>(b1) == static_cast<bool>(b2);
    }

    bool operator==(bool b1, Boolean b2)
    {
        return b1 == static_cast<bool>(b2);
    }

    bool operator==(Boolean b1, bool b2)
    {
        return static_cast<bool>(b1) == b2;
    }

    bool operator!=(Boolean b1, Boolean b2)
    {
        return static_cast<bool>(b1) != static_cast<bool>(b2);
    }

    bool operator!=(bool b1, Boolean b2)
    {
        return b1 != static_cast<bool>(b2);
    }

    bool operator!=(Boolean b1, bool b2)
    {
        return static_cast<bool>(b1) != b2;
    }

} // namespace utilities
} // namespace accera
