////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
//  Authors: Kern Handa, Chuck Jacobs
////////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/Pass/PassManager.h>

#include <functional>
#include <optional>
#include <string>

namespace accera::value
{
class Value;
}

namespace accera
{
namespace ir
{
    mlir::ModuleOp ConvertToLLVM(
        mlir::ModuleOp module,
        std::function<void(mlir::PassManager& pm)> addStdPassesFn,
        std::function<void(mlir::PassManager& pm)> addLLVMPassesFn);

} // namespace mlirHelpers
} // namespace accera
