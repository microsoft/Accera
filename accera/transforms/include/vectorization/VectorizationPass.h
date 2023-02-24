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
struct VectorizationPassOptions
{
    bool printVecOpDetails = false;
};

void populateVectorizePatterns(bool printVectorizationDetails, mlir::RewritePatternSet& patterns);
void populateVectorizeUnrollPatterns(bool printVectorizationDetails, mlir::RewritePatternSet& patterns);

std::unique_ptr<mlir::Pass> createVectorizationPass(const VectorizationPassOptions& options);
std::unique_ptr<mlir::Pass> createVectorizationPass();
std::unique_ptr<mlir::Pass> createVectorizationUnrollPass(const VectorizationPassOptions& options);
std::unique_ptr<mlir::Pass> createVectorizationUnrollPass();

} // namespace accera::transforms::vectorization
