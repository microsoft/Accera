////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
//  Authors: Chuck Jacobs
////////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <utilities/include/Exception.h>
#include <utilities/include/TypeAliases.h>

#include <mlir/ExecutionEngine/ExecutionEngine.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/BuiltinOps.h>

#include <functional>
#include <memory>
#include <string>
#include <type_traits>

namespace accera
{
namespace ir
{
    /// Function signature for a basic function that takes no input and returns no output
    typedef void (*DynamicFunction)(void);

    class MLIRExecutionEngine
    {
    public:
        MLIRExecutionEngine(mlir::ModuleOp module);

        ~MLIRExecutionEngine();

        void RunFunction(std::string functionName);

    private:
        std::unique_ptr<mlir::ExecutionEngine> _engine;
        mlir::OwningModuleRef _module;
    };

} // namespace mlirHelpers
} // namespace accera
