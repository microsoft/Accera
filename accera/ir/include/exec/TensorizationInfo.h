////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
////////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <array>

namespace accera::ir
{
namespace executionPlan
{
    struct TensorizationInfo
    {
        std::array<int64_t, 3> dim{0,0,0};
        bool useStaticOffsets{};
    private:
        friend inline bool operator==(const TensorizationInfo& p1, const TensorizationInfo& p2)
        {
            return p1.dim == p2.dim && p1.useStaticOffsets == p2.useStaticOffsets;
        }
        friend inline bool operator!=(const TensorizationInfo& p1, const TensorizationInfo& p2)
        {
            return !(p1 == p2);
        }
    };
} // namespace executionPlan
} // namespace accera::ir
