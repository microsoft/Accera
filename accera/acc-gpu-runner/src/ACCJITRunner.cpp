////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
//  Authors: Mason Remy
//
//  Port of mlir\lib\ExecutionEngine\JitRunner.cpp for use with custom dialects
////////////////////////////////////////////////////////////////////////////////////////////////////

// TODO : move to a JIT library and merge with accera\mlirHelpers\src\MLIRExecutionEngine.cpp

// This is a port of some parts of mlir\lib\ExecutionEngine\JitRunner.cpp but is separated from
// the MLIR input reading and lowering process so that custom dialects are usable in input ir and
// specific command line args aren't depended on to drive the JIT run

#ifdef _MSC_VER
#pragma warning(disable : 4146)
#endif

#include "ACCJITRunner.h"

#include <mlir/ExecutionEngine/ExecutionEngine.h>
#include <mlir/ExecutionEngine/OptUtils.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/Parser.h>
#include <mlir/Support/FileUtilities.h>

#include <llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h>
#include <llvm/Support/FileUtilities.h>
#include <llvm/Support/SourceMgr.h>

#ifdef _MSC_VER
#pragma warning(enable : 4146)
#endif

#include <cstdint>
#include <numeric>

namespace accera
{
namespace jit
{
    // TODO : move to either utilities or ir library
    mlir::OwningOpRef<mlir::ModuleOp> parseMLIRInput(const std::string& inputFilename,
                                         mlir::MLIRContext* context)
    {
        // Set up the input file.
        std::string errorMessage;
        auto file = mlir::openInputFile(inputFilename, &errorMessage);
        if (!file)
        {
            llvm::errs() << errorMessage << "\n";
            return nullptr;
        }

        llvm::SourceMgr sourceMgr;
        sourceMgr.AddNewSourceBuffer(std::move(file), llvm::SMLoc());
        return mlir::OwningOpRef<mlir::ModuleOp>(mlir::parseSourceFile(sourceMgr, context));
    }

    static inline llvm::Error make_string_error(const llvm::Twine& message)
    {
        return llvm::make_error<llvm::StringError>(message.str(),
                                                   llvm::inconvertibleErrorCode());
    }

    std::optional<ACCJITRunner> ACCJITRunner::MakeACCJITRunner(mlir::ModuleOp module, mlir::MLIRContext* context, const std::vector<std::string>& dynamicLibPaths, OptLevel opt)
    {
        auto tmBuilderOrError = llvm::orc::JITTargetMachineBuilder::detectHost();
        if (!tmBuilderOrError)
        {
            llvm::errs() << "Failed to create a JITTargetMachineBuilder for the host\n";
            return std::nullopt;
        }
        auto tmOrError = tmBuilderOrError->createTargetMachine();
        if (!tmOrError)
        {
            llvm::errs() << "Failed to create a TargetMachine for the host\n";
            return std::nullopt;
        }

        llvm::SmallVector<const llvm::PassInfo*, 4> passes;
        auto transformer = mlir::makeLLVMPassesTransformer(passes, static_cast<llvm::CodeGenOpt::Level>(opt), /*targetMachine=*/tmOrError->get(), 0);

        llvm::SmallVector<llvm::StringRef, 4> dynamicLibs(dynamicLibPaths.begin(), dynamicLibPaths.end());
        auto expectedEngine = mlir::ExecutionEngine::create(module, /*llvmModuleBuilder=*/{}, transformer, static_cast<llvm::CodeGenOpt::Level>(opt), dynamicLibs);
        [[maybe_unused]] bool isOK = static_cast<bool>(expectedEngine);

        if (isOK)
        {
            return std::move(ACCJITRunner(std::move(*expectedEngine)));
        }
        return std::nullopt;
    }

    ACCJITRunner::ACCJITRunner(std::unique_ptr<mlir::ExecutionEngine>&& engine) :
        _engine(std::move(engine))
    {}

    llvm::Error ACCJITRunner::Run(const std::string& functionName)
    {
        void* empty = nullptr;
        return Run(functionName, &empty);
    }

    llvm::Error ACCJITRunner::Run(const std::string& functionName, void** args)
    {
        auto expectedFPtr = _engine->lookup(functionName);
        if (!expectedFPtr)
            return expectedFPtr.takeError();

        auto fptr = reinterpret_cast<void(*)(void **)>(*expectedFPtr);
        (*fptr)(args);

        return llvm::Error::success();
    }

    void ACCJITRunner::DumpToObjectFile(const std::string& filename)
    {
        _engine->dumpToObjectFile(filename);
    }

} // namespace jit
} // namespace accera
