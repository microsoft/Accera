////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
//  Authors: Chuck Jacobs
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "nest/AffineExpression.h"

namespace accera::ir
{
namespace loopnest
{
    bool AffineExpression::IsIdentity() const
    {
        return (_expr == nullptr);
    }
} // namespace loopnest
} // namespace accera::ir
