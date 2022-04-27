////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
////////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <memory>

namespace mlir
{
class Pass;
class RewritePatternSet;
using OwningRewritePatternList = RewritePatternSet;
} // namespace mlir

namespace accera::transforms::affine
{
void populateAcceraAffineSimplificationPatterns(mlir::OwningRewritePatternList& patterns);
std::unique_ptr<mlir::Pass> createAffineSimplificationPass();
} // namespace accera::transforms::affine
