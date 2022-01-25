////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
//  Authors: Abdul Dakkak
////////////////////////////////////////////////////////////////////////////////////////////////////

#include <ir/include/DialectRegistry.h> 
#include <transforms/include/AcceraPasses.h>

#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Tools/mlir-lsp-server/MlirLspServerMain.h"

using namespace mlir;

int main(int argc, char** argv)
{
    mlir::DialectRegistry registry;
    registerAllDialects(registry);
    accera::ir::GetDialectRegistry().appendTo(registry);
    accera::transforms::RegisterAllPasses();
    return failed(MlirLspServerMain(argc, argv, registry));
}