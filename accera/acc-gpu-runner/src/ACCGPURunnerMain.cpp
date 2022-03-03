////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
//  Authors: Mason Remy
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef _MSC_VER
#pragma warning(disable : 4146)
#endif

#include "ACCJITRunner.h"

#include <ir/include/DialectRegistry.h>
#include <transforms/include/AcceraPasses.h>
#include <transforms/include/gpu/AcceraVulkanPasses.h>
#include <transforms/include/value/RangeValueOptimizePass.h>
#include <transforms/include/value/ValueSimplifyPass.h>
#include <transforms/include/value/ValueToStandardLoweringPass.h>

#include <mlir/Conversion/GPUToSPIRV/GPUToSPIRVPass.h>
#include <mlir/Conversion/GPUToVulkan/ConvertGPUToVulkanPass.h>
#include <mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h>
#include <mlir/Conversion/StandardToSPIRV/StandardToSPIRVPass.h>
#include <mlir/Dialect/GPU/Passes.h>
#include <mlir/Dialect/SPIRV/IR/SPIRVOps.h>
#include <mlir/Dialect/SPIRV/Transforms/Passes.h>
#include <mlir/ExecutionEngine/OptUtils.h>
#include <mlir/InitAllDialects.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/LocationSnapshot.h>
#include <mlir/Transforms/Passes.h>

#include <llvm/Support/CommandLine.h>
#include <llvm/Support/InitLLVM.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Support/raw_os_ostream.h>

#ifdef _MSC_VER
#pragma warning(enable : 4146)
#endif

#include <iostream>
#include <string>
#include <vector>

using namespace mlir;
using namespace accera::jit;

#ifdef MLIR_RUNNER_UTILS_PATH
const char* DefaultMLIRRunnerUtilsPath = MLIR_RUNNER_UTILS_PATH;
#else
const char* DefaultMLIRRunnerUtilsPath = "";
#endif

#ifdef VULKAN_RUNTIME_WRAPPERS_PATH
const char* DefaultVulkanRuntimeWrappersPath = VULKAN_RUNTIME_WRAPPERS_PATH;
#else
const char* DefaultVulkanRuntimeWrappersPath = "";
#endif

namespace
{
static llvm::cl::OptionCategory RCGPURunnerOptions("Accera GPU Runner Options");
static std::vector<const llvm::cl::OptionCategory*> AcceraGPURunnerCategories{
    &RCGPURunnerOptions
};

llvm::cl::opt<std::string> inputFilename{ llvm::cl::Positional,
                                          llvm::cl::desc("<input file>"),
                                          llvm::cl::init("-"),
                                          llvm::cl::cat(RCGPURunnerOptions) };
llvm::cl::opt<std::string> mlirRunnerUtilsPath{ "mlir-runner-utils",
                                                llvm::cl::desc("Path to mlir_runner_utils shared library"),
                                                llvm::cl::init(DefaultMLIRRunnerUtilsPath),
                                                llvm::cl::cat(RCGPURunnerOptions) };
llvm::cl::opt<std::string> vulkanRuntimeWrapperPath{ "vulkan-runtime-wrapper",
                                                     llvm::cl::desc("Path to Vulkan runtime wrapper shared library"),
                                                     llvm::cl::init(DefaultVulkanRuntimeWrappersPath),
                                                     llvm::cl::cat(RCGPURunnerOptions) };
llvm::cl::opt<bool> verbose{ "verbose",
                             llvm::cl::desc("Print out current lowering stage"),
                             llvm::cl::init(false),
                             llvm::cl::cat(RCGPURunnerOptions) };
llvm::cl::opt<bool> printRCIR{ "printRCIR",
                               llvm::cl::desc("Print out MLIR after Accera passes"),
                               llvm::cl::init(false),
                               llvm::cl::cat(RCGPURunnerOptions) };
llvm::cl::opt<bool> printVulkanIR{ "printVulkanIR",
                                   llvm::cl::desc("Print out MLIR after GPU/Vulkan passes"),
                                   llvm::cl::init(false),
                                   llvm::cl::cat(RCGPURunnerOptions) };

llvm::cl::opt<bool> printTiming{ "printTiming",
                                 llvm::cl::desc("Print timestamp timing info"),
                                 llvm::cl::init(false),
                                 llvm::cl::cat(RCGPURunnerOptions) };
llvm::cl::opt<int> warmupCount{ "warmupCount",
                                llvm::cl::desc("Number of warmup runs to perform"),
                                llvm::cl::init(0),
                                llvm::cl::cat(RCGPURunnerOptions) };
llvm::cl::opt<int> runCount{ "runCount",
                             llvm::cl::desc("Number of timed runs to perform"),
                             llvm::cl::init(1),
                             llvm::cl::cat(RCGPURunnerOptions) };

// This function needs to be kept updated with runMLIRPasses(ModuleOp module) in
// mlir\tools\mlir-vulkan-runner\mlir-vulkan-runner.cpp in llvm-project
void AddMLIRVulkanRunnerPasses(PassManager& passManager)
{
    passManager.addPass(createGpuKernelOutliningPass());
    passManager.addPass(createConvertGPUToSPIRVPass());
    passManager.addPass(accera::transforms::createAcceraToSPIRVPass());
    passManager.addPass(createCanonicalizerPass());

    OpPassManager& modulePM = passManager.nest<spirv::ModuleOp>();
    modulePM.addPass(spirv::createLowerABIAttributesPass());
    modulePM.addPass(spirv::createUpdateVersionCapabilityExtensionPass());

    passManager.addPass(createConvertGpuLaunchFuncToVulkanLaunchFuncPass());
    passManager.addPass(accera::transforms::vulkan::createEmitVulkanWrapperPass());
    passManager.addPass(createLowerToCFGPass());
    passManager.addPass(LLVM::createLegalizeForExportPass());
    LowerToLLVMOptions llvmOptions(passManager.getContext());
    llvmOptions.useBarePtrCallConv = false;
    llvmOptions.emitCWrappers = true;
    passManager.addPass(accera::transforms::value::createValueToLLVMPass(llvmOptions));
    passManager.addPass(createLowerToLLVMPass(llvmOptions));
    passManager.addPass(accera::transforms::value::createFunctionPointerResolutionPass());
    accera::transforms::vulkan::VulkanTimingOptions vulkanTimingOptions = {
        printTiming,
        warmupCount,
        runCount
    };
    passManager.addPass(accera::transforms::vulkan::createConvertVulkanLaunchFuncToVulkanCallsWithTimingPass(vulkanTimingOptions));
}

void AddAcceraLoweringPasses(PassManager& passManager)
{
    namespace v = accera::ir::value;

    // TODO : converge locations containing the Accera pass pipeline

    auto& valueFuncOpPM = passManager.nest<v::ValueModuleOp>().nest<v::ValueFuncOp>();
    valueFuncOpPM.addPass(createCanonicalizerPass());
    valueFuncOpPM.addPass(accera::transforms::loopnest::createLoopNestToValueFuncPass());
    passManager.addPass(accera::transforms::value::createValueFuncToTargetPass());

    auto& funcOpPM = passManager.nest<v::ValueModuleOp>().nest<FuncOp>();
    funcOpPM.addPass(createConvertLinalgToAffineLoopsPass());
    funcOpPM.addPass(createSimplifyAffineStructuresPass());
    funcOpPM.addPass(createLowerAffinePass());

    passManager.addPass(accera::transforms::value::createValueToStdPass());

    passManager.addPass(createCanonicalizerPass());
    passManager.addPass(createCSEPass());
}
} // namespace

int main(int argc, char** argv)
{
    llvm::llvm_shutdown_obj x;
    llvm::raw_os_ostream llvmOut(std::cout);

    // Hide general cl options inherited from MLIR
    llvm::cl::HideUnrelatedOptions(AcceraGPURunnerCategories);

    // explicitly include the pass manager cl options for things like pass statistics or printing IR after passes, etc.
    registerPassManagerCLOptions();

    llvm::cl::ParseCommandLineOptions(argc, argv, "Accera GPU JIT Runner\n");

    accera::transforms::RegisterAllPasses();
    llvm::InitLLVM y(argc, argv);
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();
    mlir::initializeLLVMPasses();

    MLIRContext context;
    context.appendDialectRegistry(accera::ir::GetDialectRegistry());
    context.loadAllAvailableDialects();

    mlir::OwningModuleRef moduleRef = parseMLIRInput(inputFilename, &context);
    // TODO: this will segfault if the module has no name
    auto moduleName = moduleRef->getName()->str();
    PassManager passManager(&context);
    applyPassManagerCLOptions(passManager);

    AddAcceraLoweringPasses(passManager);
    if (printRCIR)
    {
        passManager.addPass(
            createLocationSnapshotPass(
                OpPrintingFlags{}.enableDebugInfo(),
                moduleName + "-rc.mlir"));
    }

    AddMLIRVulkanRunnerPasses(passManager);

    if (printVulkanIR)
    {
        passManager.addPass(
            createLocationSnapshotPass(
                OpPrintingFlags{}.enableDebugInfo(),
                moduleName + "-vk.mlir"));
    }

    if (failed(passManager.run(*moduleRef)))
    {
        llvm::errs() << "Failed to compile file\n";
        return 1;
    }

    std::vector<std::string> dynamicLibPaths = {
        mlirRunnerUtilsPath,
        vulkanRuntimeWrapperPath
    };
    if (auto runner = ACCJITRunner::MakeACCJITRunner(*moduleRef, &context, dynamicLibPaths, OptLevel::O3))
    {
        // TODO: we may not want to hard code this but rather have the init and deinit functions
        // be automatically called before and after the main function, if there is one
        if (runner->Run(moduleName + "_initialize"))
        {
            llvm::errs() << "Module initialization failed\n";
            return 1;
        }
        if (runner->Run("main"))
        {
            llvm::errs() << "Running main failed\n";
            return 1;
        }
        if (runner->Run(moduleName + "_deinitialize"))
        {
            llvm::errs() << "Module cleanup failed\n";
            return 1;
        }
    }
    else
    {
        std::cout << "Failed to create JIT runner" << std::endl;
        return 1;
    }
    return 0;
}
