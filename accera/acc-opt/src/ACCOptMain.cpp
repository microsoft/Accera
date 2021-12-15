////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
//  Authors: Kern Handa
////////////////////////////////////////////////////////////////////////////////////////////////////

#include <llvm/Support/CommandLine.h>
#include <llvm/Support/FileUtilities.h>
#include <llvm/Support/InitLLVM.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/ToolOutputFile.h>
#include <mlir/IR/AsmState.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/InitAllDialects.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Support/FileUtilities.h>
#include <mlir/Support/MlirOptMain.h>

#include <ir/include/DialectRegistry.h>
#include <transforms/include/AcceraPasses.h>

static llvm::cl::opt<std::string> input_filename(llvm::cl::Positional,
                                                 llvm::cl::desc("<input file>"),
                                                 llvm::cl::init("-"));

static llvm::cl::opt<std::string>
    output_filename("o", llvm::cl::desc("Output filename"), llvm::cl::value_desc("filename"), llvm::cl::init("-"));

static llvm::cl::opt<bool> split_input_file(
    "split-input-file",
    llvm::cl::desc("Split the input file into pieces and process each "
                   "chunk independently"),
    llvm::cl::init(false));

static llvm::cl::opt<bool> verify_diagnostics(
    "verify-diagnostics",
    llvm::cl::desc("Check that emitted diagnostics match "
                   "expected-* lines on the corresponding line"),
    llvm::cl::init(false));

static llvm::cl::opt<bool> verify_passes(
    "verify-each",
    llvm::cl::desc("Run the verifier after each transformation pass"),
    llvm::cl::init(true));

static llvm::cl::opt<bool> allowUnregisteredDialects(
    "allow-unregistered-dialect",
    llvm::cl::desc("Allow operation with no registered dialects"),
    llvm::cl::init(false));

static llvm::cl::opt<bool>
    showDialects("show-dialects",
                 llvm::cl::desc("Print the list of registered dialects"),
                 llvm::cl::init(false));

int main(int argc, char** argv)
{
    llvm::InitLLVM y(argc, argv);
    accera::transforms::RegisterAllPasses();

    // Register any pass manager command line options.
    mlir::registerAsmPrinterCLOptions();
    mlir::registerMLIRContextCLOptions();
    mlir::registerPassManagerCLOptions();
    mlir::PassPipelineCLParser passPipeline("", "Compiler passes to run");
    llvm::cl::ParseCommandLineOptions(argc, argv, "Accera MLIR compiler\n");

    if (showDialects)
    {
        llvm::outs() << "Registered Dialects:\n";
        mlir::MLIRContext context;
        context.appendDialectRegistry(accera::ir::GetDialectRegistry());
        context.loadAllAvailableDialects();
        std::vector<std::string> dialectNamespaces;
        auto loadedDialects = context.getLoadedDialects();
        std::transform(loadedDialects.begin(), loadedDialects.end(), std::back_inserter(dialectNamespaces), [](mlir::Dialect* dialect) { return dialect->getNamespace().str(); });
        // Re-order dialects alphabetically for stable test assertions
        std::sort(dialectNamespaces.begin(), dialectNamespaces.end());
        for (const auto& dialectNamespace : dialectNamespaces)
        {
            llvm::outs() << dialectNamespace << "\n";
        }
        return 0;
    }

    // Set up the input file.
    std::string error_message;
    auto file = mlir::openInputFile(input_filename, &error_message);
    assert(file);
    if (!file)
    {
        llvm::errs() << error_message << "\n";
        return 1;
    }

    auto output = mlir::openOutputFile(output_filename, &error_message);
    assert(output);
    if (!output)
    {
        llvm::errs() << error_message << "\n";
        return 1;
    }

    return failed(mlir::MlirOptMain(
        output->os(),
        std::move(file),
        passPipeline,
        accera::ir::GetDialectRegistry(),
        split_input_file,
        verify_diagnostics,
        verify_passes,
        allowUnregisteredDialects));
}
