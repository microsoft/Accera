////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
//  Authors: Kern Handa
////////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <memory>

namespace mlir
{
class MLIRContext;
class ModuleOp;
class RewritePatternSet;
class Pass;
class PassManager;
class SPIRVTypeConverter;
using OwningRewritePatternList = RewritePatternSet;

template <typename OpT>
class OperationPass;

} // namespace mlir

namespace accera::transforms
{
void populateAcceraToSPIRVPatterns(mlir::SPIRVTypeConverter& typeConverter, mlir::MLIRContext* context, mlir::OwningRewritePatternList& patterns);
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createAcceraToSPIRVPass();

} // namespace accera::transforms
