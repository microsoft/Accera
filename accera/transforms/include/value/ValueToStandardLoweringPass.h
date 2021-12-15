////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
//  Authors: Kern Handa
////////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <memory>

// fwd decls
namespace mlir
{
class FuncOp;

template <typename OpT>
class OperationPass;

class RewritePatternSet;
using OwningRewritePatternList = RewritePatternSet;
} // namespace mlir

namespace accera::transforms::value
{
void populateVectorizeValueOpPatterns(mlir::RewritePatternSet& patterns);
void populateValueToStandardPatterns(bool enableProfiling, mlir::RewritePatternSet& patterns);
void populateValueLaunchFuncPatterns(mlir::RewritePatternSet& patterns);
void populateValueModuleRewritePatterns(mlir::RewritePatternSet& patterns);

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createValueToStdPass(bool enableProfiling = false);
} // namespace accera::transforms::value
