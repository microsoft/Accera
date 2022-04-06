////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
////////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <cstdint>

namespace accera::ir
{
namespace executionPlan
{
    struct InPlaceUnrollInfo
    {
        int64_t loopUnrollFactor = 0;

    private:
        friend inline bool operator==(const InPlaceUnrollInfo& ipu1, const InPlaceUnrollInfo& ipu2)
        {
            return (ipu1.loopUnrollFactor == ipu2.loopUnrollFactor);
        }
        friend inline bool operator!=(const InPlaceUnrollInfo& ipu1, const InPlaceUnrollInfo& ipu2)
        {
            return !(ipu1 == ipu2);
        }
    };
} // namespace executionPlan
} // namespace accera::ir
