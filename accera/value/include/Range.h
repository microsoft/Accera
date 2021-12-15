////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
//  Authors: Chuck Jacobs, Kern Handa
////////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <ir/include/nest/Range.h>

namespace accera
{
namespace value
{
    namespace loopnests
    {
        /// <summary>
        /// A class representing the half-open interval `[begin, end)`, with an increment between points of _increment.
        /// </summary>
        using Range = accera::ir::loopnest::Range;
    } // namespace loopnests
} // namespace value
} // namespace accera
