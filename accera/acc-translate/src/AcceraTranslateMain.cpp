////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
////////////////////////////////////////////////////////////////////////////////////////////////////

#include <mlir/InitAllDialects.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Translation.h>

#include <llvm/Support/CommandLine.h>
#include <llvm/Support/InitLLVM.h>
#include <llvm/Support/Process.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Support/ToolOutputFile.h>

#include <ir/include/DialectRegistry.h>
#include <ir/include/argo/ArgoOps.h>

#include "Target/Cpp/TranslateToCpp.h"

using namespace mlir;

namespace
{
// This function should be called before creating any MLIRContext if one
// expects all the possible translations to be made available to the context
// automatically.
inline void registerArgoTranslations()
{
    static bool initOnce = []() {
        TranslateFromMLIRRegistration printCudaRegistration(
            "print-cuda",
            [](ModuleOp module, raw_ostream& output) {
                (void)translateModuleToCpp(module, output, /*isCuda*/ true);

                return success();
            },
            [](DialectRegistry& registry) {
                registerAllDialects(registry);
                accera::ir::GetDialectRegistry().appendTo(registry);
                registry.insert<argo::ArgoDialect>();
            });


        TranslateFromMLIRRegistration printCppRegistration(
            "print-cpp",
            [](ModuleOp module, raw_ostream& output) {
                (void)translateModuleToCpp(module, output, /*isCuda*/ false);

                return success();
            },
            [](DialectRegistry& registry) {
                registerAllDialects(registry);
                accera::ir::GetDialectRegistry().appendTo(registry);
                registry.insert<argo::ArgoDialect>();
            });


        return true;
    }();
    (void)initOnce;
}
} // namespace

int main(int argc, char** argv)
{
    registerArgoTranslations();

    return failed(mlirTranslateMain(argc, argv, "acc-translate"));
}
