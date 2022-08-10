////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
//  Authors: Kern Handa, Mason Remy
////////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <memory>

// fwd decls
namespace mlir
{
class Pass;

class RewritePatternSet;
using RewritePatternSet = RewritePatternSet;
} // namespace mlir

namespace accera::transforms::value
{
void populateValueSimplifyPatterns(mlir::RewritePatternSet& patterns);
std::unique_ptr<mlir::Pass> createValueSimplifyPass();
} // namespace accera::transforms::value
