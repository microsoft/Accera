////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
//  Authors: Abdul Dakkak
////////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <memory>

// fwd decls
namespace mlir
{
class Pass;
class RewritePatternSet;
} // namespace mlir

namespace accera::transforms::value
{
void populateRangeValueOptimizePatterns(mlir::RewritePatternSet& patterns);

std::unique_ptr<mlir::Pass> createRangeValueOptimizePass();
} // namespace accera::transforms::value
