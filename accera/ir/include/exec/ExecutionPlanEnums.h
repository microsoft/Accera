////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
////////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once


#include <mlir/IR/Attributes.h>
#include <mlir/IR/BuiltinAttributes.h>

#include "exec/ExecutionPlanEnums.h.inc" 

enum class GPUIndexDimension
{
    X,
    Y,
    Z,
    Invalid = -1
};