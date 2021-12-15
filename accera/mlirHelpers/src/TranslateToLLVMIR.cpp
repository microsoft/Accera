////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
//  Authors: Chuck Jacobs
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef _MSC_VER
#pragma warning(disable : 4146)
#endif

#include "TranslateToLLVMIR.h"

#include <llvm/IR/Module.h>

#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/Target/LLVMIR/Export.h>

#ifdef _MSC_VER
#pragma warning(enable : 4146)
#endif

namespace accera::ir
{
std::unique_ptr<llvm::Module> TranslateToLLVMIR(mlir::OwningModuleRef& module, llvm::LLVMContext& context)
{
    return mlir::translateModuleToLLVMIR(*module, context);
}

} // namespace accera::ir
