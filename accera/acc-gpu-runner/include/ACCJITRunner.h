////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
//  Authors: Mason Remy
////////////////////////////////////////////////////////////////////////////////////////////////////

// TODO : move to a JIT library and merge with accera\mlirHelpers\include\MLIRExecutionEngine.h

#include <llvm/Support/CodeGen.h>
#include <llvm/Support/Error.h>
#include <mlir/ExecutionEngine/ExecutionEngine.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>

#include <memory>
#include <optional>

namespace accera
{
namespace jit
{
    enum class OptLevel
    {
        O0 = llvm::CodeGenOpt::Level::None,
        O1 = llvm::CodeGenOpt::Level::Less,
        O2 = llvm::CodeGenOpt::Level::Default,
        O3 = llvm::CodeGenOpt::Level::Aggressive
    };

    mlir::OwningOpRef<mlir::ModuleOp> parseMLIRInput(const std::string& inputFilename, mlir::MLIRContext* context);

    class ACCJITRunner
    {
    public:
        static std::optional<ACCJITRunner> MakeACCJITRunner(mlir::ModuleOp module, mlir::MLIRContext* context, const std::vector<std::string>& dynamicLibPaths, OptLevel opt = OptLevel::O0);

        ACCJITRunner() = delete;
        ACCJITRunner(ACCJITRunner&&) = default;

        // TODO : support returning values from the JIT-run function
        llvm::Error Run(const std::string& functionName);
        llvm::Error Run(const std::string& functionName, void** args);

        void DumpToObjectFile(const std::string& filename);

    private:
        ACCJITRunner(std::unique_ptr<mlir::ExecutionEngine>&& engine);
        std::unique_ptr<mlir::ExecutionEngine> _engine;
    };

} // namespace jit
} // namespace accera
