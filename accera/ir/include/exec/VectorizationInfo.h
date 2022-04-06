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
    struct VectorizationInfo
    {
        int64_t vectorBytes = 0;
        int64_t vectorUnitCount = 0;
        bool unrollOnly = false;

    private:
        friend inline bool operator==(const VectorizationInfo& v1, const VectorizationInfo& v2)
        {
            return (v1.vectorBytes == v2.vectorBytes) && (v1.vectorUnitCount == v2.vectorUnitCount) && (v1.unrollOnly == v2.unrollOnly);
        }
        friend inline bool operator!=(const VectorizationInfo& v1, const VectorizationInfo& v2)
        {
            return !(v1 == v2);
        }
    };

    const int64_t AVX2Alignment = 32;
} // namespace executionPlan
} // namespace accera::ir
