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
class MLIRContext;
class ModuleOp;
class Pass;
template <typename OpT> class OperationPass;

class RewritePatternSet;
using OwningRewritePatternList = RewritePatternSet;
} // namespace mlir

namespace accera::transforms::value
{

void populateValueLambdaToFuncPatterns(mlir::MLIRContext* context, mlir::OwningRewritePatternList& patterns);
void populateValueFuncToTargetPatterns(mlir::MLIRContext* context, mlir::OwningRewritePatternList& patterns);
void populateValueLaunchFuncInlinerPatterns(mlir::MLIRContext*, mlir::OwningRewritePatternList&);

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createValueFuncToTargetPass();
} // namespace accera::transforms::value
