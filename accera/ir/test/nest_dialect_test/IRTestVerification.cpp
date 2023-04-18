////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef _MSC_VER
#pragma warning(disable : 4146)
#endif

#include <ir/include/nest/Index.h>
#include <ir/include/nest/IndexRange.h>
#include <ir/include/nest/IterationDomain.h>
#include <ir/include/nest/LoopNestAttributes.h>
#include <ir/include/nest/LoopNestOps.h>
#include <ir/include/nest/LoopNestTypes.h>
#include <ir/include/nest/Range.h>
#include <ir/include/nest/TransformedDomain.h>

#include <value/include/TargetDevice.h>
#include <transforms/include/nest/LoopNestPasses.h>
#include <transforms/include/value/ValueToLLVMLoweringPass.h>
#include <transforms/include/value/ValueToStandardLoweringPass.h>

#include <mlirHelpers/include/ConvertToLLVM.h>
#include <mlirHelpers/include/MLIRExecutionEngine.h>
#include <mlirHelpers/include/TranslateToLLVMIR.h>

#include <utilities/include/Logger.h>
#include <utilities/include/MemoryLayout.h>
#include <utilities/include/TypeTraits.h>

#include <mlir/Conversion/Passes.h>
#include <mlir/Conversion/VectorToLLVM/ConvertVectorToLLVM.h>
#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/Arithmetic/Transforms/Passes.h>
#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/Dialect/SCF/SCF.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Verifier.h>
#include <mlir/Parser.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/Passes.h>

#include <llvm/ADT/Triple.h>
#include <llvm/Analysis/TargetLibraryInfo.h>
#include <llvm/Analysis/TargetTransformInfo.h>
#include <llvm/CodeGen/TargetPassConfig.h>
#include <llvm/IR/IRPrintingPasses.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/IR/Verifier.h>
#include <llvm/MC/TargetRegistry.h>
#include <llvm/Support/Host.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Support/raw_os_ostream.h>
#include <llvm/Target/TargetMachine.h>
#include <llvm/Transforms/IPO.h>
#include <llvm/Transforms/IPO/PassManagerBuilder.h>
#include <llvm/Transforms/Scalar.h>
#include <llvm/Transforms/Scalar/GVN.h>

#ifdef _MSC_VER
#pragma warning(enable : 4146)
#endif

#include <array>
#include <cassert>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

using namespace accera::utilities::logging;
using namespace accera::ir;
using namespace accera::transforms::value;
using namespace accera::transforms::loopnest;
// using namespace accera::mlirHelpers;
using namespace loopnest;
using namespace mlir;

namespace
{
mlir::OpBuilder* s_builder;
}

void SetTestBuilder(mlir::OpBuilder* builder)
{
    s_builder = builder;
}

mlir::OpBuilder& GetTestBuilder()
{
    return *s_builder;
}

namespace
{

std::pair<mlir::ModuleOp, bool> LowerToStd(mlir::ModuleOp module)
{
    auto moduleCopy = module.clone();

    mlir::PassManager pm(moduleCopy.getContext());

    // Apply any generic pass manager command line options and run the pipeline.
    mlir::applyPassManagerCLOptions(pm);

    // Add a run of the canonicalizer to optimize the mlir module.
    pm.addPass(mlir::createCanonicalizerPass());

    // auto& fPm = pm.nest<FuncOp>();
    // Add the lowering to std/affine/whatever pass
    // fpM addLoopNestLoweringPasses(fPm);
    // pm.setNesting
    addLoopNestLoweringPasses(pm);

    // Add value->std pass
    pm.addPass(createValueToStdPass());

    // Turn off multithreading
    // pm.disableMultithreading();

    // Run the passes
    bool ok = true;
    if (mlir::failed(pm.run(moduleCopy)))
    {
        std::cerr << "Error running lowering and optimization passes" << std::endl;
        moduleCopy = nullptr;
        ok = false;
        // throw std::runtime_error("Error running lowering and optimization passes");
    }

    return { moduleCopy, ok };
}

llvm::TargetMachine* GetTargetMachine(llvm::Module& module)
{
    bool useFastMath = true;
    bool optimize = true;

    std::string error;

    auto hostTripleString = llvm::sys::getProcessTriple();
    llvm::Triple hostTriple(hostTripleString);
    auto triple = hostTriple.normalize();
    [[maybe_unused]] auto architecture = llvm::Triple::getArchTypeName(hostTriple.getArch());
    auto cpu = llvm::sys::getHostCPUName();

    // llvm::StringMap<bool> features;
    std::string features;
    llvm::StringMap<bool> cpuFeatures;
    llvm::sys::getHostCPUFeatures(cpuFeatures);
    for (const auto& feature : cpuFeatures)
    {
        if (feature.second)
        {
            features += '+' + feature.first().str() + ",";
        }
    }
    if (!features.empty())
    {
        features.pop_back();
    }

    // Get triple
    const llvm::Target* target = llvm::TargetRegistry::lookupTarget(triple, error);
    if (target == nullptr)
    {
        return nullptr;
    }

    // auto relocModel = parameters.targetDevice.IsWindows() ? llvm::Reloc::Model::Static : llvm::Reloc::Model::PIC_;
    auto relocModel = llvm::Reloc::Model::PIC_;
    llvm::TargetOptions options;
    options.FloatABIType = llvm::FloatABI::Default;
    options.AllowFPOpFusion = useFastMath ? llvm::FPOpFusion::Standard : llvm::FPOpFusion::Fast;
    options.UnsafeFPMath = useFastMath ? 1 : 0;
    options.NoInfsFPMath = useFastMath ? 1 : 0;
    options.NoNaNsFPMath = useFastMath ? 1 : 0;
    options.NoSignedZerosFPMath = useFastMath ? 1 : 0;

    const llvm::CodeModel::Model codeModel = llvm::CodeModel::Small;
    auto tm = target->createTargetMachine(llvm::Triple::normalize(triple),
                                          cpu,
                                          features,
                                          options,
                                          relocModel,
                                          codeModel,
                                          optimize ? llvm::CodeGenOpt::Level::Aggressive : llvm::CodeGenOpt::Level::Default);
    return tm;
}

void OptimizeLLVM(llvm::Module& module)
{
    llvm::legacy::PassManager modulePasses;
    llvm::legacy::FunctionPassManager functionPasses(&module);

    auto targetMachine = GetTargetMachine(module);
    if (!targetMachine)
    {
        throw std::runtime_error("Unable to allocate target machine");
    }

    auto& llvmTargetMachine = static_cast<llvm::LLVMTargetMachine&>(*targetMachine);
    auto config = static_cast<llvm::Pass*>(llvmTargetMachine.createPassConfig(modulePasses));
    // auto config = llvmTargetMachine.createPassConfig(modulePasses);
    modulePasses.add(config);

    llvm::TargetLibraryInfoImpl targetLibraryInfo(llvm::Triple(module.getTargetTriple()));
    modulePasses.add(new llvm::TargetLibraryInfoWrapperPass(targetLibraryInfo));

    // Add internal analysis passes from the target machine.
    modulePasses.add(llvm::createTargetTransformInfoWrapperPass(targetMachine ? targetMachine->getTargetIRAnalysis()
                                                                              : llvm::TargetIRAnalysis()));

    functionPasses.add(llvm::createTargetTransformInfoWrapperPass(
        targetMachine ? targetMachine->getTargetIRAnalysis() : llvm::TargetIRAnalysis()));

    functionPasses.add(llvm::createVerifierPass());

    llvm::PassManagerBuilder builder;
    builder.OptLevel = 3;
    builder.SizeLevel = 0;
    builder.Inliner = llvm::createFunctionInliningPass(builder.OptLevel, builder.SizeLevel, false);
    builder.LoopVectorize = true;
    builder.SLPVectorize = true;
    builder.DisableUnrollLoops = false;

    if (targetMachine)
    {
        targetMachine->adjustPassManager(builder);
    }

    builder.populateFunctionPassManager(functionPasses);
    builder.populateModulePassManager(modulePasses);

    functionPasses.doInitialization();
    modulePasses.run(module);
    functionPasses.doFinalization();
}
} // namespace

//
// Test verification functions exposed in the header
//

bool VerifyGenerate(mlir::OwningOpRef<mlir::ModuleOp>& module, mlir::FuncOp& fnOp, std::string outputFile)
{
    llvm::raw_os_ostream out(Log());
    if (ShouldLog())
    {
        OpPrintingFlags flags;
        module->print(out, flags);
    }

    auto moduleOp = module.get();
    if (failed(mlir::verify(moduleOp)))
    {
        moduleOp.emitError("module verification error");
        return false;
    }

    if (!outputFile.empty())
    {
        std::error_code err;
        llvm::raw_fd_ostream fp(outputFile, err);
        module->print(fp);
    }

    return true;
}

bool VerifyParse(mlir::OwningOpRef<mlir::ModuleOp>& module, mlir::FuncOp& fnOp, std::string outputFile)
{
    auto context = module->getContext();
    llvm::raw_os_ostream out(Log());
    llvm::raw_os_ostream cout(std::cout);

    // Verify the original module
    auto moduleOp = module.get();
    if (failed(mlir::verify(moduleOp)))
    {
        moduleOp.emitError("module verification error");
        return false;
    }

    // First, get the module's MLIR output
    std::stringstream str;
    llvm::raw_os_ostream s(str);
    module->print(s);
    s.flush();

    // Destroy the old module
    module.release().erase();

    auto newModule = mlir::parseSourceString(str.str(), context);
    if (!newModule)
    {
        std::cerr << "Error parsing emitted MLIR" << std::endl;
        return false;
    }

    auto newModuleOp = newModule.get();
    if (failed(mlir::verify(newModuleOp)))
    {
        newModuleOp.emitError("re-parsed module verification error");
        return false;
    }

    if (!outputFile.empty())
    {
        std::error_code err;
        llvm::raw_fd_ostream fp(outputFile, err);
        newModule->print(fp);
    }

    return true;
}

bool VerifyLowerToStd(mlir::OwningOpRef<mlir::ModuleOp>& module, mlir::FuncOp& fnOp, std::string outputFile)
{
    llvm::raw_os_ostream out(Log());

    if (failed(mlir::verify(*module)))
    {
        module->emitError("module verification error");
        return false;
    }

    auto [newModule, ok] = LowerToStd(*module);

    if (ShouldLog())
    {
        OpPrintingFlags flags;
        if (!ok)
            flags.printGenericOpForm();
        newModule.print(out, flags);
        out.flush();
    }

    if (!outputFile.empty())
    {
        std::error_code err;
        llvm::raw_fd_ostream fp(outputFile, err);
        newModule.print(fp);
    }
    return ok;
}

bool VerifyLowerToLLVM(mlir::OwningOpRef<mlir::ModuleOp>& module, mlir::FuncOp& fnOp, std::string outputFile)
{
    llvm::raw_os_ostream out(Log());

    if (failed(mlir::verify(*module)))
    {
        module->emitError("module verification error");
        return false;
    }

    auto [stdModule, ok] = LowerToStd(*module);

    auto newModule = ConvertToLLVM(
        stdModule,
        [](mlir::PassManager& pm) {
        },
        [](mlir::PassManager& pm) {

            auto funcPm = pm.nest<mlir::FuncOp>();

            funcPm.addPass(mlir::arith::createArithmeticExpandOpsPass()); //  --arith-expand 
            pm.addPass(mlir::createLowerAffinePass());  //  --lower-affine 
            pm.addPass(mlir::createConvertSCFToCFPass());  //  --convert-scf-to-cf 
            pm.addPass(mlir::createMemRefToLLVMPass());  //  --convert-memref-to-llvm 
            pm.addPass(mlir::createLowerToLLVMPass());  //  --convert-std-to-llvm="use-bare-ptr-memref-call-conv" 
            pm.addPass(mlir::createConvertVectorToLLVMPass());  //  --convert-vector-to-llvm
            pm.addPass(mlir::createReconcileUnrealizedCastsPass());  //   --reconcile-unrealized-casts

            // Add another canonicalizer pass (because that's what the Toy example does)
            pm.addPass(mlir::createCanonicalizerPass());
        });

    if (failed(mlir::verify(newModule)))
    {
        newModule.emitError("Failed to lower to LLVM dialect");
        return false;
    }

    if (ShouldLog())
    {
        newModule.print(out);
        out.flush();
    }

    if (!outputFile.empty())
    {
        std::error_code err;
        llvm::raw_fd_ostream fp(outputFile, err);
        newModule.print(fp);
    }
    return true;
}

bool VerifyTranslateToLLVMIR(mlir::OwningOpRef<mlir::ModuleOp>& module, mlir::FuncOp& fnOp, bool optimize, std::string outputFile)
{
    llvm::raw_os_ostream out(Log());

    if (failed(mlir::verify(*module)))
    {
        module->emitError("module verification error");
        return false;
    }

    if (ShouldLog() && false)
    {
        out << "Before lowering:\n";
        module->print(out);
        out << "\n\n";
        out.flush();
    }

    mlir::OwningOpRef<mlir::ModuleOp> newModule = ConvertToLLVM(
        *module,
        [](mlir::PassManager& pm) {
            // Add a run of the canonicalizer to optimize the mlir module.
            pm.addPass(mlir::createCanonicalizerPass());

        },
        [](mlir::PassManager& pm) {
            auto funcPm = pm.nest<mlir::FuncOp>();

            funcPm.addPass(mlir::arith::createArithmeticExpandOpsPass()); //  --arith-expand 
            pm.addPass(mlir::createLowerAffinePass());  // --lower-affine 
            pm.addPass(mlir::createConvertSCFToCFPass());  //   --convert-scf-to-cf 
            pm.addPass(mlir::createMemRefToLLVMPass());  //   --convert-memref-to-llvm 
            pm.addPass(mlir::createLowerToLLVMPass());  //   --convert-std-to-llvm="use-bare-ptr-memref-call-conv" 
            pm.addPass(mlir::createConvertVectorToLLVMPass());  //  --convert-vector-to-llvm
            pm.addPass(mlir::createReconcileUnrealizedCastsPass());  //   --reconcile-unrealized-casts

            // Add another canonicalizer pass (because that's what the Toy example does)
            pm.addPass(mlir::createCanonicalizerPass());
        });

    if (failed(mlir::verify(*newModule)))
    {
        newModule->emitError("Failed to lower to LLVM dialect");
        return false;
    }

    if (ShouldLog() && false)
    {
        out.flush();
        newModule->print(out);
        out.flush();
    }
    llvm::LLVMContext context;
    auto llvmIR = TranslateToLLVMIR(newModule, context);

    if (!llvmIR)
    {
        std::cerr << "Failed to translate to LLVM IR";
        return false;
    }

    if (ShouldLog() && !optimize)
    {
        out.flush();
        llvmIR->dump();
        out.flush();
    }

    if (optimize)
    {
        OptimizeLLVM(*llvmIR);

        if (ShouldLog())
        {
            out.flush();
            llvmIR->dump();
            out.flush();
        }
    }

    if (!outputFile.empty())
    {
        std::error_code err;
        llvm::raw_fd_ostream fp(outputFile, err);
        llvmIR->print(fp, nullptr);
    }

    return true;
}
