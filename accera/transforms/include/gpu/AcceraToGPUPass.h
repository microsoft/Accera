////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
//  Authors:  Abdul Dakkak, Kern Handa
////////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

#include "ir/include/value/ValueEnums.h"
#include "value/include/FunctionDeclaration.h"
#include <memory>
#include <mlir/Dialect/GPU/GPUDialect.h>

namespace mlir
{
class MLIRContext;
class ModuleOp;
class RewritePatternSet;
class Pass;
class PassManager;
class SPIRVTypeConverter;
class LLVMTypeConverter;
class TypeConverter;
using OwningRewritePatternList = RewritePatternSet;

template <typename OpT>
class OperationPass;

namespace gpu
{
    class GPUModuleOp;
};

} // namespace mlir

namespace accera::transforms
{
void populateAcceraToSPIRVPatterns(
    mlir::SPIRVTypeConverter& typeConverter,
    mlir::MLIRContext* context,
    mlir::OwningRewritePatternList& patterns);
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createAcceraToSPIRVPass();

void populateAcceraToNVVMPatterns(mlir::OwningRewritePatternList& patterns);
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createAcceraToNVVMPass();

void populateAcceraToROCDLPatterns(mlir::OwningRewritePatternList& patterns);
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createAcceraToROCDLPass();

std::unique_ptr<mlir::OperationPass<mlir::gpu::GPUModuleOp>> createSerializeToHSACOPass();

// Abstract method which dispatches to SPIRV, NVVM, or ROCDL depending on the execution environment's runtime
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createAcceraToGPUPass(accera::value::ExecutionRuntime runtime);

} // namespace accera::transforms
