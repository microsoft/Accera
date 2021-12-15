////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
////////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <mlir/IR/BuiltinOps.h>

#include <functional>
#include <string>

using SetupFunc = std::function<void()>;
using VerifyFunc = std::function<bool(mlir::OwningModuleRef&, mlir::FuncOp&)>;

void RunTest(std::string testName, SetupFunc&& setupFunc, std::string verifyName, VerifyFunc&& verifyFunc);

//
// RUN_TEST macro
//
#define RUN_TEST(Test, Verify)                     \
    do                                             \
    {                                              \
        RunTest(#Test, (Test), #Verify, (Verify)); \
    } while (0)
