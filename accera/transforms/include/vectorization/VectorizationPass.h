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
} // namespace mlir

namespace accera::transforms::vectorization
{
void populateVectorizePatterns(bool printVectorizationDetails, mlir::RewritePatternSet& patterns);
void populateVectorizeUnrollPatterns(bool printVectorizationDetails, mlir::RewritePatternSet& patterns);

std::unique_ptr<mlir::Pass> createVectorizationPass();

} // namespace accera::transforms::vectorization
