////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
//  Authors:  Abdul Dakkak, Kern Handa
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "util/DebugUtilities.h"

#include <mlir/IR/BuiltinOps.h>
#include <mlir/Pass/Pass.h>

using namespace mlir;

namespace accera::transforms
{
struct DumpModulePass : public PassWrapper<DumpModulePass, OperationPass<mlir::ModuleOp>>
{
    void runOnOperation() final
    {
        auto mod = getOperation();
        mod.dump();
    }
};

std::unique_ptr<mlir::Pass> createDumpModulePass()
{
    return std::make_unique<DumpModulePass>();
}
} // namespace accera::transforms