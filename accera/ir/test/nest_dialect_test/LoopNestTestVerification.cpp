////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
////////////////////////////////////////////////////////////////////////////////////////////////////

#include <ir/include/nest/Index.h>
#include <ir/include/nest/IndexRange.h>
#include <ir/include/nest/IterationDomain.h>
#include <ir/include/nest/LoopNestAttributes.h>
#include <ir/include/nest/LoopNestOps.h>
#include <ir/include/nest/LoopNestTypes.h>
#include <ir/include/nest/Range.h>
#include <ir/include/nest/TransformedDomain.h>

#include <transforms/include/nest/LoopNestPasses.h>
#include <transforms/include/value/ValueToLLVMLoweringPass.h>
#include <transforms/include/value/ValueToStandardLoweringPass.h>

#include <mlirHelpers/include/ConvertToLLVM.h>
#include <mlirHelpers/include/MLIRExecutionEngine.h>
#include <mlirHelpers/include/TranslateToLLVMIR.h>

#include <utilities/include/Logger.h>
#include <utilities/include/MemoryLayout.h>
#include <utilities/include/TypeTraits.h>

#include <llvm/Support/SourceMgr.h>
#include <llvm/Support/raw_os_ostream.h>

#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/Linalg/IR/LinalgOps.h>
#include <mlir/Dialect/SCF/SCF.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Verifier.h>
#include <mlir/Parser.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/Passes.h>

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
std::pair<mlir::ModuleOp, bool> LowerToValue(mlir::ModuleOp module)
{
    auto moduleCopy = module.clone();

    mlir::PassManager pm(moduleCopy.getContext());

    // Apply any generic pass manager command line options and run the pipeline.
    mlir::applyPassManagerCLOptions(pm);

    // Add a run of the canonicalizer to optimize the mlir module.
    pm.addPass(mlir::createCanonicalizerPass());

    // Add the lowering to std/affine/whatever pass
    addLoopNestLoweringPasses(pm);

    // Add another canonicalizer pass (because that's what the Toy example does)
    pm.addPass(mlir::createCanonicalizerPass());

    // Turn off multithreading
    // pm.disableMultithreading();

    // Run the passes
    bool ok = true;
    if (mlir::failed(pm.run(moduleCopy)))
    {
        std::cerr << "Error running lowering and optimization passes" << std::endl;
        ok = false;
        // throw std::runtime_error("Error running lowering and optimization passes");
    }

    return { moduleCopy, ok };
}

std::pair<mlir::ModuleOp, bool> LowerToStd(mlir::ModuleOp module)
{
    auto moduleCopy = module.clone();

    mlir::PassManager pm(moduleCopy.getContext());

    // Apply any generic pass manager command line options and run the pipeline.
    mlir::applyPassManagerCLOptions(pm);

    // Add a run of the canonicalizer to optimize the mlir module.
    pm.addPass(mlir::createCanonicalizerPass());

    // Add the lowering to std/affine/whatever pass
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
} // namespace

//
// Test verification functions exposed in the header
//

bool VerifyGenerate(mlir::OwningModuleRef& module, mlir::FuncOp& fnOp)
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
    return true;
}

bool VerifyParse(mlir::OwningModuleRef& module, mlir::FuncOp& fnOp)
{
    auto context = module->getContext();
    llvm::raw_os_ostream out(Log());
    llvm::raw_os_ostream cout(std::cout);

    std::stringstream str;
    llvm::raw_os_ostream s(str);

    if (ShouldLog())
    {
        module->print(out);
        out << "\n\n";
        out.flush();
    }

    // Verify the original module
    auto moduleOp = module.get();
    if (failed(mlir::verify(moduleOp)))
    {
        moduleOp.emitError("module verification error");
        return false;
    }

    // First, get the module's MLIR output
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

    if (ShouldLog())
    {
        out << "Re-read module:";
        newModuleOp.print(out);
        out.flush();
    }
    return true;
}

bool VerifyLowerToValue(mlir::OwningModuleRef& module, mlir::FuncOp& fnOp)
{
    llvm::raw_os_ostream out(Log());

    if (ShouldLog())
    {
        out << "Before lowering:\n";
        module->print(out);
        out << "\n\n";
        out.flush();
    }

    if (failed(mlir::verify(*module)))
    {
        module->emitError("module verification error");
        return false;
    }

    auto [newModule, ok] = LowerToValue(*module);

    if (ShouldLog())
    {
        out << "After lowering:\n";
        OpPrintingFlags flags;
        if (!ok)
            flags.printGenericOpForm();
        newModule.print(out, flags);
        out.flush();
    }
    return ok;
}

bool VerifyLowerToStd(mlir::OwningModuleRef& module, mlir::FuncOp& fnOp)
{
    llvm::raw_os_ostream out(Log());

    if (ShouldLog())
    {
        out << "Before lowering:\n";
        module->print(out);
        out << "\n\n";
        out.flush();
    }

    if (failed(mlir::verify(*module)))
    {
        module->emitError("module verification error");
        return false;
    }

    auto [newModule, ok] = LowerToStd(*module);

    if (ShouldLog())
    {
        out << "After lowering:\n";
        OpPrintingFlags flags;
        if (!ok)
            flags.printGenericOpForm();
        newModule.print(out, flags);
        out.flush();
    }
    return ok;
}

#if 0
bool VerifyLowerToLLVM(mlir::OwningModuleRef& module, mlir::FuncOp& fnOp)
{
    llvm::raw_os_ostream out(Log());

    if (ShouldLog())
    {
        out << "Before lowering:\n";
        module->print(out);
        out << "\n\n";
        out.flush();
    }

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
            pm.addPass(createValueToLLVMPass(pm.getContext()));

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
        out << "After lowering:\n";
        newModule.print(out);
        out.flush();
    }

    return true;
}
#endif

#if 0
bool VerifyTranslateToLLVMIR(mlir::OwningModuleRef& module, mlir::FuncOp& fnOp)
{
    llvm::raw_os_ostream out(Log());

    if (failed(mlir::verify(*module)))
    {
        module->emitError("module verification error");
        return false;
    }

    if (ShouldLog())
    {
        out << "Before lowering:\n";
        out.flush();
        module->print(out);
        out.flush();
        out << "\n\n";
        out.flush();
    }
    mlir::OwningModuleRef newModule = ConvertToLLVM(
        *module,
        [](mlir::PassManager& pm) {
            // Add a run of the canonicalizer to optimize the mlir module.
            pm.addPass(mlir::createCanonicalizerPass());

            // Add the lowering from loopnest dialect pass
            addLoopNestLoweringPasses(pm);

            // Add the lowering from value to llvm pass
            pm.addPass(createValueToStdPass());
        },
        [](mlir::PassManager& pm) {
            pm.addPass(createValueToLLVMPass(pm.getContext()));

            // Add another canonicalizer pass (because that's what the Toy example does)
            pm.addPass(mlir::createCanonicalizerPass());
        });

    if (failed(mlir::verify(*newModule)))
    {
        newModule->emitError("Failed to lower to LLVM dialect");
        return false;
    }

    if (ShouldLog())
    {
        out << "After lowering to LLVM dialect:\n";
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

    if (ShouldLog())
    {
        out << "After translating to LLVMIR:\n";
        out.flush();
        llvmIR->dump();
        out.flush();
    }
    return true;
}
#endif

#if 0
bool VerifyJIT(mlir::OwningModuleRef& module, mlir::FuncOp& fnOp)
{
    llvm::raw_os_ostream out(Log());

    auto newModule = ConvertToLLVM(
        *module,
        [](mlir::PassManager& pm) {
            // Add a run of the canonicalizer to optimize the mlir module.
            pm.addPass(mlir::createCanonicalizerPass());

            // Add the lowering from loopnest dialect pass
            addLoopNestLoweringPasses(pm);

            // Add the lowering from value to llvm pass
            pm.addPass(createValueToStdPass());
            //    pm.addPass(mlir::createCanonicalizerPass());
        },
        [](mlir::PassManager& pm) {
            pm.addPass(createValueToLLVMPass(pm.getContext()));
            pm.addPass(mlir::createCanonicalizerPass());
        });

    if (failed(mlir::verify(newModule)))
    {
        newModule.emitError("Failed to lower to LLVM dialect");
        return false;
    }

    auto functionName = std::string(mlir::SymbolTable::getSymbolName(fnOp.getOperation()));

    if (ShouldLog())
    {
        Log() << "Jitting function " << functionName << std::endl;
    }

    MLIRExecutionEngine engine(newModule);
    engine.RunFunction(functionName);

    if (ShouldLog())
    {
        Log() << "..Done jitting function " << functionName << std::endl;
    }

    return true;
}
#endif
