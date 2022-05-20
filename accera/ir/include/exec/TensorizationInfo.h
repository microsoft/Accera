////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
////////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

#include "../value/ValueMMAOp.h"
#include <array>

namespace accera::ir
{
namespace executionPlan
{
    struct TensorizationInfo
    {
        accera::ir::value::MMAShape dim;
        int numTotalPasses{ 1 };
        bool useStaticOffsets{};
        int numFusedPasses{ -1 };
        accera::ir::value::MMASchedulingPolicy schedulingPolicy{};

    private:
        friend inline bool operator==(const TensorizationInfo& p1, const TensorizationInfo& p2)
        {
            return p1.dim == p2.dim && p1.useStaticOffsets == p2.useStaticOffsets && p1.numTotalPasses == p2.numTotalPasses && p1.numFusedPasses == p2.numFusedPasses && p1.schedulingPolicy == p2.schedulingPolicy;
        }
        friend inline bool operator!=(const TensorizationInfo& p1, const TensorizationInfo& p2)
        {
            return !(p1 == p2);
        }
    };
} // namespace executionPlan
} // namespace accera::ir
