////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
////////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Value.h>

#include <cstddef>

namespace accera::transforms
{
mlir::Value SaturateValue(mlir::PatternRewriter& rewriter, mlir::Value value, int64_t bitWidth, bool isSigned);
} // namespace accera::transforms
