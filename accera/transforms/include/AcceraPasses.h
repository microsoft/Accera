////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
//  Authors: Kern Handa
////////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

#include "accera/AcceraLoweringPass.h"
#include "exec/ExecutionPlanToAffineLoweringPass.h"
#include "gpu/AcceraToGPUPass.h"
#include "gpu/AcceraToSPIRVPass.h"
#include "gpu/AcceraVulkanPasses.h"
#include "ir/include/value/ValueEnums.h"
#include "nest/LoopNestPasses.h"
#include "nest/LoopNestToValueFunc.h"
#include "value/FunctionPointerResolutionPass.h"
#include "value/ValueFuncToTargetPass.h"
#include "value/ValueSimplifyPass.h"
#include "value/ValueToLLVMLoweringPass.h"
#include "value/ValueToStandardLoweringPass.h"

#include <ir/include/exec/ExecutionPlanOps.h>
#include <ir/include/nest/LoopNestOps.h>
#include <ir/include/value/ValueDialect.h>
#include <ir/include/value/ValueEnums.h>

#include <value/include/ExecutionOptions.h>

#include <mlir/InitAllDialects.h>
#include <mlir/InitAllPasses.h>
#include <mlir/Pass/Pass.h>

namespace accera::transforms
{

using mlir::Pass;

/// A model for providing module pass specific utilities.
///
/// Derived module passes are expected to provide the following:
///   - A 'void runOnModule()' method.
class ModulePass : public ::mlir::OperationPass<::mlir::ModuleOp>
{
public:
    using ::mlir::OperationPass<::mlir::ModuleOp>::OperationPass;

    /// The polymorphic API that runs the pass over the currently held function.
    virtual void runOnModule() = 0;

    /// The polymorphic API that runs the pass over the currently held operation.
    void runOnOperation() final
    {
        runOnModule();
    }

    /// Return the current module being transformed.
    ::mlir::ModuleOp getModule() { return this->getOperation(); }
};

#define GEN_PASS_CLASSES
#define GEN_PASS_REGISTRATION
#include "AcceraPasses.h.inc"

struct AcceraPassPipelineOptions : mlir::PassPipelineOptions<AcceraPassPipelineOptions>
{
    Option<bool> dumpPasses{ *this, "dump-passes", llvm::cl::init(false) };
    Option<bool> gpuOnly{ *this, "gpu-only", llvm::cl::init(false) };
    Option<bool> dumpIntraPassIR{ *this, "dump-intra-pass-ir", llvm::cl::init(false) };
    Option<std::string> basename{ *this, "basename", llvm::cl::init(std::string{}) };
    Option<std::string> target{ *this, "target", llvm::cl::init("host") };
    Option<accera::value::ExecutionRuntime> runtime{
        *this,
        "runtime",
        llvm::cl::desc("Execution runtime"),
        llvm::cl::values(
            clEnumValN(accera::value::ExecutionRuntime::Default, "default", "default runtime"),
            clEnumValN(accera::value::ExecutionRuntime::Vulkan, "vulkan", "Vulkan runtime"),
            clEnumValN(accera::value::ExecutionRuntime::Rocm, "rocm", "Rocm runtime"),
            clEnumValN(accera::value::ExecutionRuntime::CUDA, "cuda", "CUDA runtime")),
        llvm::cl::init(accera::value::ExecutionRuntime::Default)
    };
    Option<bool> enableAsync{ *this, "enable-async", llvm::cl::init(false) };
    Option<bool> enableProfile{ *this, "enable-profiling", llvm::cl::init(false) };
    Option<bool> printLoops{ *this, "print-loops", llvm::cl::init(false) };
    Option<bool> printVecOpDetails{ *this, "print-vec-details", llvm::cl::init(false) };
};

void addAcceraToLLVMPassPipeline(mlir::OpPassManager& pm, const AcceraPassPipelineOptions& options);

void registerAcceraToLLVMPipeline();

inline void RegisterAllPasses()
{
    mlir::registerAllPasses();
    registerPasses();
    registerAcceraToLLVMPipeline();
}

} // namespace accera::transforms
