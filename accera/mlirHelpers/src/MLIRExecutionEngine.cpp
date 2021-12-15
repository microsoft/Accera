////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
//  Authors: Chuck Jacobs
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef _MSC_VER
#pragma warning(disable : 4146)
#endif

#include "MLIRExecutionEngine.h"

#include <mlir/ExecutionEngine/ExecutionEngine.h>
#include <mlir/ExecutionEngine/OptUtils.h>
#include <mlir/IR/BuiltinOps.h>

#include <llvm/ADT/StringRef.h>
#include <llvm/ADT/StringSwitch.h>
#include <llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h>
#include <llvm/Support/TargetSelect.h>

#ifdef _MSC_VER
#pragma warning(enable : 4146)
#endif

using namespace mlir;
using llvm::StringRef;

namespace accera
{
namespace ir
{
    MLIRExecutionEngine::MLIRExecutionEngine(mlir::ModuleOp module) :
        _module(module)
    {
        llvm::InitializeNativeTarget();
        llvm::InitializeNativeTargetAsmPrinter();

        int optLevel = 0;
        auto optPipeline = mlir::makeOptimizingTransformer(
            optLevel, /*sizeLevel=*/0,
            /*targetMachine=*/nullptr);

        auto maybeEngine = mlir::ExecutionEngine::create(module, /*llvmModuleBuilder=*/nullptr, optPipeline);
        assert(maybeEngine && "failed to construct an execution engine");

        _engine = std::move(*maybeEngine);
    }

    MLIRExecutionEngine::~MLIRExecutionEngine() = default;

    void MLIRExecutionEngine::RunFunction(std::string functionName)
    {
        // Invoke the JIT-compiled function.
        auto invocationResult = _engine->invoke(functionName);
        if (invocationResult)
        {
            llvm::errs() << "JIT invocation failed\n";
        }
    }
} // namespace mlirHelpers
} // namespace accera
