////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
//  Authors: Abdul Dakkak
////////////////////////////////////////////////////////////////////////////////////////////////////

#include <mlir/InitAllDialects.h>
#include <mlir/InitAllTranslations.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Translation.h>

#include <mlir/Target/LLVMIR/Dialect/All.h>
#include <mlir/Target/LLVMIR/Export.h>

#include <llvm/Support/CommandLine.h>
#include <llvm/Support/InitLLVM.h>
#include <llvm/Support/Process.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Support/ToolOutputFile.h>

#include <ir/include/DialectRegistry.h>
#include <ir/include/argo/ArgoOps.h>

#include "Target/Cpp/TranslateToCpp.h"

#include "Target/LLVMIR/IntrinsicToLLVMIRTranslation.h"



using namespace mlir;


namespace
{

llvm::cl::opt<int> indexBitwidth{ "indexBitwidth",
                                  llvm::cl::desc("The bitwidth of the unsigned integer type used to represent indices"),
                                  llvm::cl::init(32) };

// This function should be called before creating any MLIRContext if one
// expects all the possible translations to be made available to the context
// automatically.
inline void registerArgoTranslations()
{
    [[maybe_unused]] static bool initOnce = []() {
        TranslateFromMLIRRegistration printCppRegistration(
            "print-cpp",
            [&](Operation* m, llvm::raw_ostream& os) -> LogicalResult { return translateModuleToCpp(m, os, indexBitwidth); },
            [](DialectRegistry& registry) {
                registerAllDialects(registry);
                accera::ir::GetDialectRegistry().appendTo(registry);
                registry.insert<argo::ArgoDialect>();
            });

        return true;
    }();
}

void registerAcceraToLLVMIRTranslation() {
  TranslateFromMLIRRegistration registration(
      "acc-to-llvmir",
      [](ModuleOp module, llvm::raw_ostream &output) {
        llvm::LLVMContext llvmContext;
        auto llvmModule = translateModuleToLLVMIR(module, llvmContext);
        if (!llvmModule)
          return failure();

        llvmModule->print(output, nullptr);
        return success();
      },
      [](DialectRegistry &registry) {
        registerAllDialects(registry);
        accera::ir::GetDialectRegistry().appendTo(registry);
        accera::transforms::intrinsics::registerIntrinsicsDialectTranslation(registry);
        registerAllToLLVMIRTranslations(registry);
      });
}
} // namespace

int main(int argc, char** argv)
{
    registerArgoTranslations();
    registerAcceraToLLVMIRTranslation();
    mlir::registerAllTranslations();

    return failed(mlirTranslateMain(argc, argv, "acc-translate"));
}
