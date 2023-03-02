////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
////////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <memory>
#include <string>

namespace mlir
{
class Pass;
class RewritePatternSet;
} // namespace mlir

namespace accera::transforms::affine
{

// Unit attr name for controlling whether bounds checking has already been performed on an op
const std::string BoundsCheckedAttrName = "accaffine.bounds_checked";

// Unit attr name for controlling whether bounds checking is done for ops within a marked op
const std::string AccessBoundsCheckAttrName = "accaffine.access_bounds_check";

void populateBoundsCheckingPatterns(mlir::RewritePatternSet& patterns);

// TODO : implement
// std::unique_ptr<mlir::Pass> createBoundsCheckingPass();

} // namespace accera::transforms::affine
