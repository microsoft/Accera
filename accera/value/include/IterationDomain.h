////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
//  Authors: Chuck Jacobs, Kern Handa
////////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <ir/include/nest/IterationDomain.h>

namespace accera
{
namespace value
{
    namespace loopnests
    {
        /// <summary>
        /// The set of all points (IterationVectors) to be visited by a loop or loop nest.
        /// </summary>
        using IterationDomain = accera::ir::loopnest::IterationDomain;
    } // namespace loopnests
} // namespace value
} // namespace accera
