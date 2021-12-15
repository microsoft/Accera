////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
//  Authors: Kern Handa, Chuck Jacobs
////////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <llvm/IR/Module.h>

#include <memory>

namespace mlir
{
class MLIRContext;
class OwningModuleRef;
} // namespace mlir

namespace accera
{
namespace ir
{
    std::unique_ptr<llvm::Module> TranslateToLLVMIR(mlir::OwningModuleRef& module, llvm::LLVMContext& context);
} // namespace mlirHelpers
} // namespace accera
