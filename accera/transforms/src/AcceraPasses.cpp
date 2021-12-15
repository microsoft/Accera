////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
//  Authors: Kern Handa
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "AcceraPasses.h"

#include <ir/include/InitializeAccera.h>
#include <value/include/TargetDevice.h>

#include <mlir/Pass/PassManager.h>
#include <mlir/Pass/PassRegistry.h>
#include <mlir/Dialect/SPIRV/IR/SPIRVOps.h>

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
        return NestedPassAdaptor<PassManagerGeneratorFn>(*this, std::move(pmGeneratorFn), _dumpPasses);
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
        _pmGeneratorFn(std::move(pmGeneratorFn)),
        _dumpPasses(dumpPasses)
    {
        if (!_dumpPasses)
        {
            _singlePM = &(_pmGeneratorFn());
        }
    }

    void addPass(std::unique_ptr<mlir::Pass> pass)
    {
        if (_dumpPasses)
        {
            auto passName = pass->getName();
            auto& pm = _pmGeneratorFn();
            pm.addPass(std::move(pass));

            _parent.addLocationSnapshot(passName);
        }
        else
        {
            (*_singlePM)->addPass(std::move(pass));
        }
    }

    std::optional<OpPassManager*> _singlePM;
    PassManagerAdaptor& _parent;
    PassManagerGeneratorFn _pmGeneratorFn;
    bool _dumpPasses;
};
}; // namespace

namespace accera::transforms
{

void addAcceraToLLVMPassPipeline(OpPassManager& pm, const AcceraPassPipelineOptions& options)
{
    ir::InitializeAccera();

    PassManagerAdaptor pmAdaptor(pm, options.dumpPasses.getValue(), options.basename);

    auto valueFuncOpPM = pmAdaptor.nestPassManager([&]() -> OpPassManager& { return pm.nest<v::ValueModuleOp>().nest<v::ValueFuncOp>(); });

    // Can't use ValueSimplify here because ExecToAffine doesn't know how to handle "simplified" ops (memref::SubView, etc.)
    // valueFuncOpPM.addPass(value::createValueSimplifyPass());

    valueFuncOpPM.addPass(createCanonicalizerPass());
    valueFuncOpPM.addPass(loopnest::createLoopNestToValueFuncPass({ { options.dumpIntraPassIR.getValue(), options.basename + "LoopNestToValueFuncPass_Subpasses" }, options.printLoops.getValue(), options.printVecOpDetails.getValue() }));

    pmAdaptor.addPass(value::createValueFuncToTargetPass());
    pmAdaptor.addPass(createSymbolDCEPass());

    auto funcOpPM = pmAdaptor.nestPassManager([&]() -> OpPassManager& { return pm.nest<v::ValueModuleOp>().nest<FuncOp>(); });
    funcOpPM.addPass(createConvertLinalgToAffineLoopsPass());
    funcOpPM.addPass(createSimplifyAffineStructuresPass());
    funcOpPM.addPass(createCanonicalizerPass());
    funcOpPM.addPass(createLowerAffinePass());
    funcOpPM.addPass(createConvertSCFToOpenMPPass());

    pmAdaptor.addPass(value::createValueToStdPass(options.enableProfile));
    pmAdaptor.addPass(createCSEPass());
    pmAdaptor.addPass(createCanonicalizerPass());

    pmAdaptor.addPass(createGpuKernelOutliningPass());
    pmAdaptor.addPass(createAcceraToSPIRVPass());
    pmAdaptor.addPass(createCanonicalizerPass());

    OpPassManager &spirvModulePM = pm.nest<spirv::ModuleOp>();
    spirvModulePM.addPass(spirv::createLowerABIAttributesPass());
    spirvModulePM.addPass(spirv::createUpdateVersionCapabilityExtensionPass());

    pmAdaptor.addPass(createConvertGpuLaunchFuncToVulkanLaunchFuncPass());
    pmAdaptor.addPass(vulkan::createEmitVulkanWrapperPass());

    funcOpPM.addPass(createConvertVectorToSCFPass(VectorTransferToSCFOptions{}/*.setLowerPermutationMaps(true) .setLowerTensors(true).setUnroll(true) */));
    pmAdaptor.addPass(createLowerToCFGPass());

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
    pmAdaptor.addPass(vulkan::createConvertVulkanLaunchFuncToVulkanCallsWithTimingPass({ false }));
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
