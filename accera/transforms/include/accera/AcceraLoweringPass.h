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
class MLIRContext;
class ModuleOp;
class Pass;
class PassManager;
class LLVMTypeConverter;

template <typename OpT>
class OperationPass;

class RewritePatternSet;
using OwningRewritePatternList = RewritePatternSet;
} // namespace mlir

namespace accera::transforms::rc
{
void populateGlobalAcceraToLLVMPatterns(mlir::LLVMTypeConverter& typeConverter, mlir::OwningRewritePatternList& patterns);
void populateAcceraToLLVMPatterns(mlir::LLVMTypeConverter& typeConverter, mlir::OwningRewritePatternList& patterns);
void populateAcceraLoweringPatterns(mlir::OwningRewritePatternList& patterns);
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createAcceraLoweringPass();
std::unique_ptr<mlir::OperationPass<mlir::FuncOp>> createAcceraToLLVMPass();
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createGlobalAcceraToLLVMPass();

void addAcceraToStandardPasses(mlir::PassManager&);

} // namespace accera::transforms::rc
