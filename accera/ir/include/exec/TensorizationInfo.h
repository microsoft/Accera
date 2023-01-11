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
        accera::ir::value::MMAFragmentOp prologueOp{};
        double prologueArg{};
        accera::ir::value::MMAFragmentOp epilogueOp{};
        double epilogueArg{};
        bool _useRocWMMA{};

    private:
        friend inline bool operator==(const TensorizationInfo& p1, const TensorizationInfo& p2)
        {
            return p1.dim == p2.dim && p1.useStaticOffsets == p2.useStaticOffsets && p1.numTotalPasses == p2.numTotalPasses && p1.numFusedPasses == p2.numFusedPasses && p1.schedulingPolicy == p2.schedulingPolicy && p1.prologueOp == p2.prologueOp && p1.prologueArg == p2.prologueArg && p1.epilogueOp == p2.epilogueOp && p1.epilogueArg == p2.epilogueArg && p1._useRocWMMA == p2._useRocWMMA;
        }
        friend inline bool operator!=(const TensorizationInfo& p1, const TensorizationInfo& p2)
        {
            return !(p1 == p2);
        }
    };
} // namespace executionPlan
} // namespace accera::ir
