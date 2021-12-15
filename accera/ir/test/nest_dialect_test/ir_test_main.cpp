////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
//  Authors: Chuck Jacobs
//
//  Unit tests for nest-related code
////////////////////////////////////////////////////////////////////////////////////////////////////

#define CATCH_CONFIG_RUNNER
#include <catch2/catch.hpp>

#include <ir/include/DialectRegistry.h>
#include <ir/include/InitializeAccera.h>

#include <llvm/Support/InitLLVM.h>

#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/Location.h>

namespace
{
mlir::OpBuilder* s_builder;
}

mlir::OpBuilder& GetTestBuilder()
{
    return *s_builder;
}

int main(int argc, char** argv)
{
    mlir::MLIRContext context;

    llvm::InitLLVM initLLVM(argc, argv);
    accera::ir::InitializeAccera();

    context.appendDialectRegistry(accera::ir::GetDialectRegistry());
    context.loadAllAvailableDialects();
    mlir::ModuleOp module(mlir::ModuleOp::create(mlir::UnknownLoc::get(&context), llvm::StringRef("testModule")));
    mlir::OpBuilder builder(&context);
    s_builder = &builder;
    builder.setInsertionPoint(module.getBody(), module.getBody()->begin());

    int result = Catch::Session().run(argc, argv);
    return result;
}
