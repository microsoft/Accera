////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "Test.h"

#include <ir/include/DialectRegistry.h>
#include <ir/include/nest/LoopNestOps.h>
#include <testing/include/testing.h>

#include <llvm/Support/SourceMgr.h>

#include <iostream>

mlir::edsc::ScopedContext CreateNewScope(mlir::OpBuilder& builder, mlir::OpBuilder::InsertPoint insertionPoint = {})
{
    if (insertionPoint.isSet())
    {
        return mlir::edsc::ScopedContext(builder, insertionPoint, builder.getUnknownLoc());
    }
    else
    {
        return mlir::edsc::ScopedContext(builder, builder.getUnknownLoc());
    }
}

mlir::FuncOp CreateFunction(mlir::OpBuilder& builder, std::string functionName, llvm::ArrayRef<mlir::Type> argTypes, llvm::ArrayRef<mlir::Type> returnTypes)
{
    auto fnType = builder.getFunctionType(argTypes, returnTypes);
    auto fnOp = builder.create<mlir::FuncOp>(mlir::edsc::ScopedContext::getLocation(), functionName, fnType, llvm::None);

    return fnOp;
}

template <typename FnType>
mlir::FuncOp CreateFunction(std::string functionName, mlir::ArrayRef<mlir::Type> argTypes, mlir::ArrayRef<mlir::Type> returnTypes, FnType&& body)
{
    auto& builder = mlir::edsc::ScopedContext::getBuilderRef();
    auto fnOp = CreateFunction(builder, functionName, argTypes, returnTypes);
    auto& entryBlock = *fnOp.addEntryBlock();
    auto bodyBuilder = mlir::OpBuilder(entryBlock.getParent());
    auto scope = CreateNewScope(bodyBuilder, { &entryBlock, std::prev(entryBlock.end()) });

    body();

    return fnOp;
}

void RunTest(std::string setupName, SetupFunc&& setupFunc, std::string verifyName, VerifyFunc&& verifyFunc)
{
    auto testName = setupName + " -- " + verifyName;
    try
    {
        mlir::MLIRContext context;
        context.appendDialectRegistry(accera::ir::GetDialectRegistry());
        context.loadAllAvailableDialects();

        llvm::SourceMgr sourceMgr;
        mlir::SourceMgrDiagnosticHandler sourceMgrHandler(sourceMgr, &context);

        mlir::OpBuilder builder(&context);

        auto moduleOp = mlir::ModuleOp::create(mlir::FileLineColLoc::get("", /*line=*/0, /*column=*/0, &context), llvm::StringRef(testName));
        mlir::OwningOpRef<mlir::ModuleOp> module(moduleOp);

        mlir::edsc::ScopedContext scope(builder, { moduleOp.getBody(), moduleOp.getBody()->begin() }, moduleOp.getLoc());
        auto fnOp = CreateFunction("test", {}, {}, [&]() {
            setupFunc();

            mlir::edsc::intrinsics::std_ret{};
        });

        bool ok = verifyFunc(module, fnOp);

        // accera::testing::ProcessTest(testName, ok);
    }
    catch (const std::exception& e)
    {
        std::cerr << e.what() << '\n';
        // accera::testing::ProcessTest(testName, false);
    }
}
