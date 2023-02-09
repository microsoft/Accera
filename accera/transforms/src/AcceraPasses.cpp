////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
//  Authors: Kern Handa
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "AcceraPasses.h"

#include <ir/include/InitializeAccera.h>
#include <value/include/TargetDevice.h>

#include <mlir/Dialect/SPIRV/IR/SPIRVOps.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Pass/PassRegistry.h>

#include <llvm/Support/CommandLine.h>
#include <llvm/Support/FormatVariadic.h>

using namespace llvm;
using namespace mlir;

namespace v = accera::ir::value;

namespace
{

// Note: The MLIR pass manager infra doesn't appear to support finding
//       ops of a given type at arbitrary nesting depths if there are
//       no registered pass managers for the intermediate ops
//       E.g. if we register a pass that runs on ValueFuncOps but don't register
//            it on a pass manager that runs on ValueModuleOps, then the base
//            pass manager won't introspect into ValueModuleOps in order to find
//            ValueFuncOps
//       To enable breaking up nested pass managers for the purposes of printing
//       op snapshots, wrap the pass manager usage and supply a lambda for creating
//       nested pass managers so they can conditionally be created once for each
//       nested pass if we're dumping pass snapshots, or can be created only once
//       and re-used for each pass for better performance and pipelining but
//       without the snapshotting utility

// Forward declare
template <typename PassManagerGeneratorFn>
struct NestedPassAdaptor;

// Utility wrapper around an OpPassManager to optionally add snapshotting after each pass
// and add nested pass managers
struct PassManagerAdaptor
{
    PassManagerAdaptor(OpPassManager& basePM, bool dumpPasses, const std::string& basename) :
        _basePM(basePM),
        _idx(0),
        _dumpPasses(dumpPasses),
        _basename(basename)
    {
        if (!_basename.empty() && _basename[_basename.size() - 1] != '/' && _basename[_basename.size() - 1] != '\\')
        {
            _basename += "_";
        }
    }

    void addPass(std::unique_ptr<mlir::Pass> pass)
    {
        auto passName = pass->getName();
        _basePM.addPass(std::move(pass));
        if (_dumpPasses)
        {
            addLocationSnapshot(passName);
        }
    }

    template <typename PassManagerGeneratorFn>
    NestedPassAdaptor<PassManagerGeneratorFn> nestPassManager(PassManagerGeneratorFn&& pmGeneratorFn)
    {
        return NestedPassAdaptor<PassManagerGeneratorFn>(*this, std::forward<PassManagerGeneratorFn>(pmGeneratorFn), _dumpPasses);
    }

    void addLocationSnapshot(llvm::StringRef passName)
    {
        _basePM.addPass(
            createLocationSnapshotPass(
                OpPrintingFlags{}.enableDebugInfo(),
                llvm::formatv("{0}{1}_{2}.mlir", _basename, ++_idx, passName).str()));
    }

    OpPassManager& _basePM;
    size_t _idx;
    bool _dumpPasses;
    std::string _basename;
};

// Utility adaptor for nested passes that can conditionally:
//  - Create one nested pass manager and add several passes to it, which
//      will enable better parallelization of lowering passes on different
//      instances of the nested ops
//  or
//  - Create one nested pass manager for each pass being added to it, which
//      is required if snapshots after each pass stage are requested.
template <typename PassManagerGeneratorFn>
struct NestedPassAdaptor
{
    NestedPassAdaptor(PassManagerAdaptor& parent,
                      PassManagerGeneratorFn&& pmGeneratorFn,
                      bool dumpPasses) :
        _parent(parent),
        _pmGeneratorFn(std::forward<PassManagerGeneratorFn>(pmGeneratorFn)),
        _dumpPasses(dumpPasses)
    {}

    void addPass(std::unique_ptr<mlir::Pass> pass)
    {
        auto passName = pass->getName();
        _pmGeneratorFn().addPass(std::move(pass));

        if (_dumpPasses)
        {
            _parent.addLocationSnapshot(passName);
        }
    }

    PassManagerAdaptor& _parent;
    PassManagerGeneratorFn _pmGeneratorFn;
    bool _dumpPasses;
};
}; // namespace

namespace accera::transforms
{

void simplifyAndLowerAffine(PassManagerAdaptor& pmAdaptor)
{
    pmAdaptor.addPass(affine::createAffineSimplificationPass());
    pmAdaptor.addPass(value::createRangeValueOptimizePass());
    pmAdaptor.addPass(createLowerAffinePass());
}

void addAcceraToLLVMPassPipeline(OpPassManager& pm, const AcceraPassPipelineOptions& options)
{
    ir::InitializeAccera();

    accera::value::ExecutionRuntime execRuntime = options.runtime;

    PassManagerAdaptor pmAdaptor(pm, options.dumpPasses.getValue(), options.basename);
    pmAdaptor.addPass(createEmitDebugFunctionPass());

    auto valueFuncOpPM = pmAdaptor.nestPassManager([&]() -> OpPassManager& { return pm.nest<v::ValueModuleOp>().nest<v::ValueFuncOp>(); });

    // Can't use ValueSimplify here because ExecToAffine doesn't know how to handle "simplified" ops (memref::SubView, etc.)
    // valueFuncOpPM.addPass(value::createValueSimplifyPass());
    valueFuncOpPM.addPass(createCanonicalizerPass());
    valueFuncOpPM.addPass(loopnest::createLoopNestToValueFuncPass({ { options.dumpIntraPassIR.getValue(), options.basename + "LoopNestToValueFuncPass_Subpasses" }, options.printLoops.getValue(), options.printVecOpDetails.getValue() }));

    pmAdaptor.addPass(value::createValueFuncToTargetPass({ options.dumpIntraPassIR.getValue(), options.basename + "ValueFuncToTargetPass_Subpasses" }));
    pmAdaptor.addPass(createSymbolDCEPass());
    pmAdaptor.addPass(affine::createAffineSimplificationPass());
    pmAdaptor.addPass(value::createValueUnrollingPass());

    auto funcOpPM = pmAdaptor.nestPassManager([&]() -> OpPassManager& { return pm.nest<v::ValueModuleOp>().nest<FuncOp>(); });
    funcOpPM.addPass(createConvertLinalgToAffineLoopsPass());
    funcOpPM.addPass(createSimplifyAffineStructuresPass());
    funcOpPM.addPass(createCanonicalizerPass());
    funcOpPM.addPass(createLoopInvariantCodeMotionPass());
    funcOpPM.addPass(createCSEPass());

    pmAdaptor.addPass(createConvertSCFToOpenMPPass());
    pmAdaptor.addPass(value::createValueToStdPass(options.enableProfile));
    pmAdaptor.addPass(value::createRangeValueOptimizePass());
    pmAdaptor.addPass(createCanonicalizerPass());
    pmAdaptor.addPass(createCSEPass());

    if (execRuntime == accera::value::ExecutionRuntime::VULKAN)
    {
        // The spirv lowering doesn't generate affine dialect ops, and the SPIRV dialect doesn't play nicely with them, so lower the affine ops before running the GPU lowering
        simplifyAndLowerAffine(pmAdaptor);
    }

    pmAdaptor.addPass(createGpuKernelOutliningPass());
    auto gpuPass = createAcceraToGPUPass(execRuntime);
    if (gpuPass)
    {
        pmAdaptor.addPass(createGPUSimplificationPass());
        pmAdaptor.addPass(value::createBarrierOptPass(options.writeBarrierGraph.getValue(), options.barrierGraphFilename.getValue()));
        pmAdaptor.addPass(std::move(gpuPass));
    }

    if (execRuntime != accera::value::ExecutionRuntime::VULKAN)
    {
        // lowering to runtimes other than SPIRV generates affine dialect ops so optimize and lower those now
        simplifyAndLowerAffine(pmAdaptor);
        if (execRuntime == accera::value::ExecutionRuntime::ROCM)
        {
            pmAdaptor.addPass(createGPUToROCDLPass());
        }
    }

    pmAdaptor.addPass(createLoopInvariantCodeMotionPass());
    pmAdaptor.addPass(createCSEPass());
    pmAdaptor.addPass(value::createRangeValueOptimizePass());
    pmAdaptor.addPass(createCanonicalizerPass());
    pmAdaptor.addPass(createCSEPass());

    if (execRuntime == accera::value::ExecutionRuntime::VULKAN)
    {
        OpPassManager& spirvModulePM = pm.nest<spirv::ModuleOp>();
        spirvModulePM.addPass(spirv::createLowerABIAttributesPass());
        spirvModulePM.addPass(spirv::createUpdateVersionCapabilityExtensionPass());

        pmAdaptor.addPass(createConvertGpuLaunchFuncToVulkanLaunchFuncPass());
        pmAdaptor.addPass(vulkan::createEmitVulkanWrapperPass());
    }
    else
    {
        PassManagerAdaptor gpuModulePM(pm.nest<gpu::GPUModuleOp>(), options.dumpPasses.getValue(), options.basename + "_gpu_module");
        if (execRuntime == accera::value::ExecutionRuntime::CUDA)
        {
            // TODO: enable this codepath when we add nvvm lowering, also enable NVVM translation in DialectRegistry.cpp
            //gpuModulePM.addPass(createLowerGpuOpsToNVVMOpsPass(32));
        }
        gpuModulePM.addPass(createStripDebugInfoPass());
        if (options.gpuOnly) return;
    }

    funcOpPM.addPass(createConvertVectorToSCFPass(
        VectorTransferToSCFOptions{} /*.setLowerPermutationMaps(true) .setLowerTensors(true).setUnroll(true) */));
    pmAdaptor.addPass(createLowerToCFGPass());

    if (execRuntime != accera::value::ExecutionRuntime::VULKAN)
    {
        PassManagerAdaptor gpuModulePM(pm.nest<gpu::GPUModuleOp>(), options.dumpPasses.getValue(), options.basename + "_rocm_module");
        if (execRuntime == accera::value::ExecutionRuntime::ROCM)
        {
            gpuModulePM.addPass(createLowerGpuOpsToROCDLOpsPass(kDeriveIndexBitwidthFromDataLayout));
            // TODO: enable this codepath when we add HSACO lowering (for ROCM)
            // gpuModulePM.addPass(createSerializeToHSACOPass());
        }

        PassManagerAdaptor funcPm(pm.nest<FuncOp>(), options.dumpPasses.getValue(), options.basename + "_fun_op");
        if (options.enableAsync) funcPm.addPass(createGpuAsyncRegionPass());
    }

    pmAdaptor.addPass(value::createValueToLLVMPass(
        /* useBasePtrCallConv = */ false,
        /* emitCWrappers = */ false,
        /* indexBitwidth = */ kDeriveIndexBitwidthFromDataLayout,
        /* useAlignedAlloc = */ true,
        /* dataLayout = */ llvm::DataLayout(accera::value::GetTargetDevice(options.target).dataLayout),
        { options.dumpIntraPassIR.getValue(), options.basename + "ValueToLLVM_Subpasses" }));
    pmAdaptor.addPass(createCanonicalizerPass());
    pmAdaptor.addPass(LLVM::createLegalizeForExportPass());
    pmAdaptor.addPass(value::createFunctionPointerResolutionPass());

    if (execRuntime == accera::value::ExecutionRuntime::VULKAN)
    {
        pmAdaptor.addPass(vulkan::createConvertVulkanLaunchFuncToVulkanCallsWithTimingPass({ false }));
        pmAdaptor.addPass(createGpuToLLVMConversionPass());
    }
    else
    {
        pmAdaptor.addPass(createGpuToLLVMConversionPass());
        if (options.enableAsync)
        {
            pmAdaptor.addPass(createAsyncToAsyncRuntimePass());
            pmAdaptor.addPass(createConvertAsyncToLLVMPass());
        }
    }
}

void registerAcceraToLLVMPipeline()
{
    PassPipelineRegistration<AcceraPassPipelineOptions>{
        "acc-to-llvm",
        "Accera to LLVM",
        addAcceraToLLVMPassPipeline
    };
}

} // namespace accera::transforms
