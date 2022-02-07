////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
//  Authors:  Kern Handa
////////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <mlir/IR/AffineMap.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/BuiltinAttributes.h>

namespace accera::ir::value
{
using mlir::AffineMapAttr;
class ExecutionTargetAttr;
class ExecutionRuntimeAttr;
}

#include "value/ValueAttrs.h.inc"
