////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
//  Authors: Abdul Dakkak
////////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <ir/include/exec/ExecutionOptions.h>

namespace accera
{
namespace value
{
    namespace targets
    {
        using namespace ir::targets;
    } // namespace targets

    using ExecutionTarget = targets::Target;
    using ExecutionRuntime = targets::Runtime;
} // namespace value
} // namespace accera