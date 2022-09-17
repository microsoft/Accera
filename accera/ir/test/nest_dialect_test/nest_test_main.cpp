////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
//  Authors: Chuck Jacobs
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

int main(int argc, char** argv)
{
    int result = Catch::Session().run(argc, argv);
    return result;
}
